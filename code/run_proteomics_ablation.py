#!/usr/bin/env python3
"""
Proteomics Ablation Experiment for MetaGNN
===========================================
Compares: Baseline (RNA+Proteomics) vs RNA-only (proteomics zeroed)
Runs on Apple MPS (M4 Max) for GPU acceleration.

Usage:
    conda activate metagnn  (or /opt/anaconda3/envs/metagnn)
    python run_proteomics_ablation.py --data_root ../data --output_dir ../results/ablation_proteomics
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')
logger = logging.getLogger(__name__)

# Import model and data utilities
sys.path.insert(0, os.path.dirname(__file__))
from metagnn_model import MetaGNN
from data_loader import MetaGNNDataset, stratified_split


# ─── Modified Dataset that zeros proteomics ─────────────────────────────────
class RNAOnlyDataset(MetaGNNDataset):
    """Wraps MetaGNNDataset but zeros out the proteomics column (col 1)."""
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # Zero out proteomics column (column 1), keep RNA (column 0)
        data['reaction'].x = data['reaction'].x.clone()
        data['reaction'].x[:, 1] = 0.0
        return data


# ─── Loss ────────────────────────────────────────────────────────────────────
class MetaGNNLoss(nn.Module):
    def __init__(self, stoich_matrix, lambda_mb=0.2):
        super().__init__()
        self.register_buffer('S', stoich_matrix)
        self.lambda_mb = lambda_mb
        self.bce = nn.BCELoss()

    def forward(self, s_r, y_r):
        l_bce = self.bce(s_r, y_r.float())
        net_flux = (self.S * s_r.unsqueeze(0)).sum(dim=1)
        l_mb = net_flux.pow(2).mean()
        return l_bce + self.lambda_mb * l_mb


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(y_pred, y_true, threshold=0.15):
    y_bin = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    prec = precision_score(y_true, y_bin, zero_division=0)
    rec = recall_score(y_true, y_bin, zero_division=0)
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_pred)
        pr, re, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(re, pr)
    else:
        auroc = auprc = float('nan')
    return {'F1': f1, 'AUROC': auroc, 'AUPRC': auprc, 'Precision': prec, 'Recall': rec}


# ─── Training ────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        s_r = model(
            x_dict={'reaction': batch['reaction'].x, 'metabolite': batch['metabolite'].x},
            edge_index_dict={rel: batch[rel].edge_index for rel in batch.edge_types},
        )
        loss = criterion(s_r, batch['reaction'].y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.15):
    model.eval()
    all_pred, all_true, total_loss = [], [], 0.0
    for batch in loader:
        batch = batch.to(device)
        s_r = model(
            x_dict={'reaction': batch['reaction'].x, 'metabolite': batch['metabolite'].x},
            edge_index_dict={rel: batch[rel].edge_index for rel in batch.edge_types},
        )
        loss = criterion(s_r, batch['reaction'].y)
        total_loss += loss.item()
        all_pred.append(s_r.cpu().numpy())
        all_true.append(batch['reaction'].y.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    metrics = compute_metrics(y_pred, y_true, threshold)
    metrics['loss'] = total_loss / len(loader)
    return metrics


# ─── Per-patient evaluation ──────────────────────────────────────────────────
@torch.no_grad()
def evaluate_per_patient(model, dataset, device, threshold=0.15):
    """Evaluate per-patient to get mean±std metrics."""
    model.eval()
    patient_metrics = []
    for i in range(len(dataset)):
        data = dataset[i].to(device)
        s_r = model(
            x_dict={'reaction': data['reaction'].x, 'metabolite': data['metabolite'].x},
            edge_index_dict={rel: data[rel].edge_index for rel in data.edge_types},
        )
        y_pred = s_r.cpu().numpy()
        y_true = data['reaction'].y.cpu().numpy()
        m = compute_metrics(y_pred, y_true, threshold)
        patient_metrics.append(m)

    result = {}
    for key in ['F1', 'AUROC', 'AUPRC', 'Precision', 'Recall']:
        vals = [m[key] for m in patient_metrics if not np.isnan(m[key])]
        result[f'{key}_mean'] = float(np.mean(vals))
        result[f'{key}_std'] = float(np.std(vals))
    return result


# ─── Run single condition ────────────────────────────────────────────────────
def run_condition(condition_name, dataset_cls, data_root, train_ids, val_ids, test_ids,
                  device, stoich_matrix, cfg, seed):
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {condition_name} (seed={seed})")
    logger.info(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = dataset_cls(data_root, train_ids)
    val_ds = dataset_cls(data_root, val_ids)
    test_ds = dataset_cls(data_root, test_ids)

    train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = PyGLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)

    model = MetaGNN(
        rxn_in_dim=2, met_in_dim=519,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg['dropout'],
    ).to(device)

    criterion = MetaGNNLoss(stoich_matrix=stoich_matrix.to(device), lambda_mb=cfg['lambda_mb'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'])

    # Load pretrained weights if available
    pretrain_path = os.path.join(data_root, 'model_weights_pretrained.pt')
    if os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded pre-trained weights")

    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None
    start_time = time.time()

    for epoch in range(1, cfg['n_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg['threshold'])
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"val_F1={val_metrics['F1']:.4f}  val_AUROC={val_metrics['AUROC']:.4f}"
            )

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time
    logger.info(f"  Training time: {train_time:.1f}s, best val F1: {best_val_f1:.4f}")

    # Load best model and evaluate on test set (per-patient)
    model.load_state_dict(best_state)
    test_results = evaluate_per_patient(model, test_ds, device, cfg['threshold'])

    logger.info(f"  Test AUROC: {test_results['AUROC_mean']:.4f} ± {test_results['AUROC_std']:.4f}")
    logger.info(f"  Test F1:    {test_results['F1_mean']:.4f} ± {test_results['F1_std']:.4f}")
    logger.info(f"  Test AUPRC: {test_results['AUPRC_mean']:.4f} ± {test_results['AUPRC_std']:.4f}")

    test_results['best_val_f1'] = best_val_f1
    test_results['train_time_s'] = train_time
    test_results['seed'] = seed
    return test_results


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MetaGNN Proteomics Ablation')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='../results/ablation_proteomics')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--n_seeds', type=int, default=3)
    args = parser.parse_args()

    cfg = dict(
        hidden_dim=256, n_layers=3, heads=8, dropout=0.20,
        lr=1e-3, weight_decay=1e-5, batch_size=1,
        n_epochs=args.n_epochs, patience=20, threshold=0.15,
        lambda_mb=0.2, seed=2024,
    )

    # Device selection: CPU for stability (MPS OOM with 9.6M param model)
    # CPU training takes ~8 min per 200 epochs on M4 Max
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Load data splits
    meta_df = pd.read_csv(os.path.join(args.data_root, 'clinical_metadata.tsv'), sep='\t')
    train_ids, val_ids, test_ids = stratified_split(meta_df, seed=cfg['seed'])
    logger.info(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Load stoichiometric matrix
    with h5py.File(os.path.join(args.data_root, 'recon3d_stoich.h5'), 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32)

    os.makedirs(args.output_dir, exist_ok=True)

    seeds = [2024, 42, 123][:args.n_seeds]
    all_results = {'baseline': [], 'rna_only': []}

    for seed in seeds:
        # Baseline: RNA + Proteomics (normal)
        res_baseline = run_condition(
            'Baseline (RNA+Proteomics)', MetaGNNDataset,
            args.data_root, train_ids, val_ids, test_ids,
            device, S, cfg, seed
        )
        all_results['baseline'].append(res_baseline)

        # RNA-only: proteomics zeroed
        res_rna = run_condition(
            'RNA-only (proteomics zeroed)', RNAOnlyDataset,
            args.data_root, train_ids, val_ids, test_ids,
            device, S, cfg, seed
        )
        all_results['rna_only'].append(res_rna)

    # Aggregate results
    summary = {}
    for condition in ['baseline', 'rna_only']:
        runs = all_results[condition]
        agg = {}
        for key in ['AUROC_mean', 'F1_mean', 'AUPRC_mean', 'Precision_mean', 'Recall_mean']:
            vals = [r[key] for r in runs]
            agg[key] = float(np.mean(vals))
            agg[key.replace('_mean', '_std_across_seeds')] = float(np.std(vals))
        agg['per_seed'] = runs
        summary[condition] = agg

    # Compute delta
    summary['delta'] = {
        'AUROC': summary['baseline']['AUROC_mean'] - summary['rna_only']['AUROC_mean'],
        'F1': summary['baseline']['F1_mean'] - summary['rna_only']['F1_mean'],
        'AUPRC': summary['baseline']['AUPRC_mean'] - summary['rna_only']['AUPRC_mean'],
    }

    out_path = os.path.join(args.output_dir, 'proteomics_ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"PROTEOMICS ABLATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Baseline (RNA+Prot): AUROC={summary['baseline']['AUROC_mean']:.4f}, F1={summary['baseline']['F1_mean']:.4f}")
    logger.info(f"RNA-only:            AUROC={summary['rna_only']['AUROC_mean']:.4f}, F1={summary['rna_only']['F1_mean']:.4f}")
    logger.info(f"Delta:               AUROC={summary['delta']['AUROC']:+.4f}, F1={summary['delta']['F1']:+.4f}")
    logger.info(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
