#!/usr/bin/env python3
"""
MetaGNN Architecture Scaling Experiments — GPU (RTX 5090)
==========================================================
Self-contained script that trains MetaGNN at multiple architecture
configurations to produce a scaling comparison table for the paper.

Configurations tested:
  Config A (Current):  hidden=128, layers=2, heads=4   — matches existing CPU results
  Config B (Paper):    hidden=256, layers=3, heads=8   — paper-described architecture
  Config C (Large):    hidden=512, layers=3, heads=8   — extended for GPU capacity
  Config D (Deep):     hidden=256, layers=5, heads=8   — depth scaling study

Each config runs 3 seeds × {full data, RNA-only ablation} = 6 runs per config.
Total: 24 training runs.

Output: results_gpu/scaling_results.json — copy this back to Cowork.

Author: Thiptanawat Phongwattana
Hardware target: Windows 11 + NVIDIA RTX 5090 (32 GB VRAM)
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score,
                             precision_recall_curve, auc,
                             precision_score, recall_score)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════
CONFIGS = {
    'A_current': dict(
        hidden_dim=128, n_layers=2, heads=4,
        label='Current (128/2/4)',
        batch_size=4,           # Small model, can handle larger batches
    ),
    'B_paper': dict(
        hidden_dim=256, n_layers=3, heads=8,
        label='Paper (256/3/8)',
        batch_size=2,           # Fits in 32GB VRAM with batch=2
    ),
    'C_large': dict(
        hidden_dim=512, n_layers=3, heads=8,
        label='Large (512/3/8)',
        batch_size=1,           # 512 hidden + 8 heads → ~4x memory of B_paper
    ),
    'D_deep': dict(
        hidden_dim=256, n_layers=5, heads=8,
        label='Deep (256/5/8)',
        batch_size=1,           # 5 layers accumulate more activation memory
    ),
}

# Shared training hyperparameters (same as existing experiments)
TRAIN_CFG = dict(
    dropout=0.20,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=2,          # Reduced for RTX 5090 32GB (GATv2 attention is memory-intensive)
    n_epochs=80,
    patience=15,
    threshold=0.15,
    lambda_mb=0.2,
    rxn_in_dim=2,
    met_in_dim=519,        # 7 physico-chem + 512 Morgan FP
)

SEEDS = [2024, 42, 123]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (self-contained — no imports from other files)
# ═══════════════════════════════════════════════════════════════════════════════
class MCDropout(nn.Module):
    """Monte Carlo Dropout — always on (even during eval)."""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


class HGATLayer(nn.Module):
    """Heterogeneous GATv2 layer with 3 relation types."""
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2, residual=True):
        super().__init__()
        self.residual = residual
        self.conv = HeteroConv({
            ('metabolite', 'substrate_of', 'reaction'): GATv2Conv(
                in_channels['metabolite'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=False, concat=False),
            ('reaction', 'produces', 'metabolite'): GATv2Conv(
                in_channels['reaction'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=False, concat=False),
            ('reaction', 'shared_metabolite', 'reaction'): GATv2Conv(
                in_channels['reaction'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=True, concat=False),
        }, aggr='mean')
        self.norm_rxn = nn.LayerNorm(out_channels)
        self.norm_met = nn.LayerNorm(out_channels)
        self.proj_rxn = (nn.Linear(in_channels['reaction'], out_channels)
                         if in_channels['reaction'] != out_channels else nn.Identity())
        self.proj_met = (nn.Linear(in_channels['metabolite'], out_channels)
                         if in_channels['metabolite'] != out_channels else nn.Identity())
        self.mc_drop = MCDropout(p=dropout)

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        if self.residual:
            out['reaction'] = self.norm_rxn(F.elu(out['reaction']) + self.proj_rxn(x_dict['reaction']))
            out['metabolite'] = self.norm_met(F.elu(out['metabolite']) + self.proj_met(x_dict['metabolite']))
        else:
            out['reaction'] = self.norm_rxn(F.elu(out['reaction']))
            out['metabolite'] = self.norm_met(F.elu(out['metabolite']))
        out['reaction'] = self.mc_drop(out['reaction'])
        out['metabolite'] = self.mc_drop(out['metabolite'])
        return out


class MetaGNN(nn.Module):
    """Heterogeneous GATv2 for metabolic reaction activity prediction."""
    def __init__(self, rxn_in_dim=2, met_in_dim=519, hidden_dim=128,
                 n_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.proj_rxn = nn.Sequential(Linear(rxn_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.proj_met = nn.Sequential(Linear(met_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.layers = nn.ModuleList([
            HGATLayer({'reaction': hidden_dim, 'metabolite': hidden_dim},
                      hidden_dim, heads=heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
            MCDropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )

    def forward(self, x_dict, edge_index_dict):
        h = {'reaction': self.proj_rxn(x_dict['reaction']),
             'metabolite': self.proj_met(x_dict['metabolite'])}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return self.output_head(h['reaction']).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS (BCE + mass-balance regularisation)
# ═══════════════════════════════════════════════════════════════════════════════
class MetaGNNLoss(nn.Module):
    def __init__(self, stoich_matrix, lambda_mb=0.2):
        super().__init__()
        self.register_buffer('S', stoich_matrix)
        self.lambda_mb = lambda_mb
        self.bce = nn.BCELoss()

    def forward(self, s_r, y_r):
        l_bce = self.bce(s_r, y_r.float())
        n_rxn = self.S.shape[1]
        if s_r.shape[0] > n_rxn and s_r.shape[0] % n_rxn == 0:
            s_r_2d = s_r.view(-1, n_rxn)
            l_mb = torch.tensor(0.0, device=s_r.device)
            for i in range(s_r_2d.shape[0]):
                net_flux = (self.S * s_r_2d[i].unsqueeze(0)).sum(dim=1)
                l_mb = l_mb + net_flux.pow(2).mean()
            l_mb = l_mb / s_r_2d.shape[0]
        elif s_r.shape[0] == n_rxn:
            net_flux = (self.S * s_r.unsqueeze(0)).sum(dim=1)
            l_mb = net_flux.pow(2).mean()
        else:
            l_mb = torch.tensor(0.0, device=s_r.device)
        return l_bce + self.lambda_mb * l_mb


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET (self-contained — no import from data_loader.py)
# ═══════════════════════════════════════════════════════════════════════════════
class MetaGNNDataset(torch.utils.data.Dataset):
    """One HeteroData graph per patient."""
    def __init__(self, data_root, patient_ids=None):
        self.data_root = data_root
        self.meta_df = pd.read_csv(
            os.path.join(data_root, 'clinical_metadata.tsv'), sep='\t'
        )
        if patient_ids is not None:
            self.meta_df = self.meta_df[
                self.meta_df['tcga_barcode'].isin(patient_ids)
            ].reset_index(drop=True)
        self.patient_ids = self.meta_df['tcga_barcode'].tolist()

        # Load shared tensors once
        with h5py.File(os.path.join(data_root, 'metabolite_features.h5'), 'r') as f:
            self.X_M = torch.tensor(f['X_M'][:], dtype=torch.float32)

        self.edge_index_dict = {
            ('metabolite', 'substrate_of',      'reaction'):
                torch.load(os.path.join(data_root, 'edge_indices', 'substrate_of.pt'),
                           weights_only=True),
            ('reaction',   'produces',          'metabolite'):
                torch.load(os.path.join(data_root, 'edge_indices', 'produces.pt'),
                           weights_only=True),
            ('reaction',   'shared_metabolite', 'reaction'):
                torch.load(os.path.join(data_root, 'edge_indices', 'shared_metabolite.pt'),
                           weights_only=True),
        }

        self.y_r = torch.load(os.path.join(data_root, 'activity_pseudolabels.pt'),
                              weights_only=True)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        rxn_feat_path = os.path.join(self.data_root, 'reaction_features', f'{pid}.h5')
        with h5py.File(rxn_feat_path, 'r') as f:
            X_R = torch.tensor(f['X_R'][:], dtype=torch.float32)

        data = HeteroData()
        data['reaction'].x   = X_R
        data['metabolite'].x = self.X_M
        data['reaction'].y   = self.y_r
        data['reaction'].pid = pid

        for rel, ei in self.edge_index_dict.items():
            src_type, rel_type, dst_type = rel
            data[src_type, rel_type, dst_type].edge_index = ei

        return data


class RNAOnlyDataset(MetaGNNDataset):
    """Wraps MetaGNNDataset but zeros out proteomics column (col 1)."""
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['reaction'].x = data['reaction'].x.clone()
        data['reaction'].x[:, 1] = 0.0
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# STRATIFIED SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
def stratified_split(meta_df, train_frac=0.70, val_frac=0.15, seed=2024):
    strat_col = meta_df['msi_status'].fillna('Unknown')
    ids = meta_df['tcga_barcode'].tolist()
    strat = strat_col.tolist()

    train_ids, tmp_ids, _, tmp_strat = train_test_split(
        ids, strat, train_size=train_frac, stratify=strat, random_state=seed,
    )
    val_frac_of_tmp = val_frac / (1 - train_frac)
    val_ids, test_ids = train_test_split(
        tmp_ids, train_size=val_frac_of_tmp, stratify=tmp_strat, random_state=seed,
    )
    return train_ids, val_ids, test_ids


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_pred, y_true, threshold=0.15):
    y_bin = (y_pred >= threshold).astype(int)
    f1   = f1_score(y_true, y_bin, zero_division=0)
    prec = precision_score(y_true, y_bin, zero_division=0)
    rec  = recall_score(y_true, y_bin, zero_division=0)
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_pred)
        pr, re, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(re, pr)
    else:
        auroc = auprc = float('nan')
    return {'F1': f1, 'AUROC': auroc, 'AUPRC': auprc, 'Precision': prec, 'Recall': rec}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device):
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
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, threshold=0.15):
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
    metrics['loss'] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def evaluate_per_patient(model, dataset, device, threshold=0.15):
    """Per-patient evaluation for meaningful std."""
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
        result[f'{key}_std']  = float(np.std(vals))
    return result


def run_single_config(config_name, config, dataset_cls, data_root,
                      train_ids, val_ids, test_ids, device, stoich_matrix, seed):
    """Train one config+seed and return test metrics."""
    cfg = {**TRAIN_CFG, **config}
    label = cfg.pop('label', config_name)

    logger.info(f"\n{'='*70}")
    logger.info(f"  Config: {label}  |  Seed: {seed}  |  Dataset: {dataset_cls.__name__}")
    logger.info(f"  hidden_dim={cfg['hidden_dim']}, n_layers={cfg['n_layers']}, heads={cfg['heads']}")
    logger.info(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = dataset_cls(data_root, train_ids)
    val_ds   = dataset_cls(data_root, val_ids)
    test_ds  = dataset_cls(data_root, test_ids)

    train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = PyGLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)

    model = MetaGNN(
        rxn_in_dim=cfg['rxn_in_dim'], met_in_dim=cfg['met_in_dim'],
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,}")

    # Log GPU memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    criterion = MetaGNNLoss(stoich_matrix=stoich_matrix.to(device), lambda_mb=cfg['lambda_mb'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'])

    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None
    start_time = time.time()
    best_epoch = 0

    for epoch in range(1, cfg['n_epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device, cfg['threshold'])
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"val_F1={val_metrics['F1']:.4f}  val_AUROC={val_metrics['AUROC']:.4f}"
            )

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time

    # Peak GPU memory
    peak_gpu_mb = 0
    if device.type == 'cuda':
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6

    logger.info(f"  Training: {train_time:.1f}s, best val F1={best_val_f1:.4f} @ epoch {best_epoch}")
    if peak_gpu_mb > 0:
        logger.info(f"  Peak GPU memory: {peak_gpu_mb:.0f} MB")

    # Evaluate on test set per-patient
    model.load_state_dict(best_state)
    test_results = evaluate_per_patient(model, test_ds, device, cfg['threshold'])

    logger.info(f"  Test F1:    {test_results['F1_mean']:.4f} ± {test_results['F1_std']:.4f}")
    logger.info(f"  Test AUROC: {test_results['AUROC_mean']:.4f} ± {test_results['AUROC_std']:.4f}")
    logger.info(f"  Test AUPRC: {test_results['AUPRC_mean']:.4f} ± {test_results['AUPRC_std']:.4f}")

    test_results['config'] = config_name
    test_results['label'] = label
    test_results['seed'] = seed
    test_results['n_params'] = n_params
    test_results['train_time_s'] = round(train_time, 1)
    test_results['best_epoch'] = best_epoch
    test_results['best_val_f1'] = best_val_f1
    test_results['peak_gpu_mb'] = round(peak_gpu_mb, 0)
    test_results['hidden_dim'] = cfg['hidden_dim']
    test_results['n_layers'] = cfg['n_layers']
    test_results['heads'] = cfg['heads']
    test_results['batch_size'] = cfg['batch_size']
    test_results['dataset'] = dataset_cls.__name__

    return test_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='MetaGNN GPU Scaling Experiments')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='./results_gpu',
                        help='Directory for output JSON results')
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS.keys()),
                        help='Which configs to run (default: all)')
    parser.add_argument('--n_seeds', type=int, default=3,
                        help='Number of random seeds (1-3)')
    parser.add_argument('--skip_ablation', action='store_true',
                        help='Skip RNA-only ablation (run full data only)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.warning("No CUDA GPU detected — running on CPU (will be slow)")

    # ── Load stoichiometric matrix ────────────────────────────────────────────
    stoich_path = os.path.join(args.data_root, 'recon3d_stoich.h5')
    logger.info(f"Loading stoichiometric matrix from {stoich_path}")
    with h5py.File(stoich_path, 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32)
    logger.info(f"  S matrix shape: {S.shape}")

    # ── Load metadata and split ───────────────────────────────────────────────
    meta_df = pd.read_csv(os.path.join(args.data_root, 'clinical_metadata.tsv'), sep='\t')
    train_ids, val_ids, test_ids = stratified_split(meta_df, seed=2024)
    logger.info(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # ── Run experiments ───────────────────────────────────────────────────────
    seeds = SEEDS[:args.n_seeds]
    all_results = []

    for config_name in args.configs:
        if config_name not in CONFIGS:
            logger.warning(f"Unknown config '{config_name}', skipping")
            continue

        config = CONFIGS[config_name]

        for seed in seeds:
            # Full data (RNA + Proteomics)
            result = run_single_config(
                config_name, dict(config), MetaGNNDataset,
                args.data_root, train_ids, val_ids, test_ids,
                device, S, seed
            )
            all_results.append(result)

            # RNA-only ablation
            if not args.skip_ablation:
                result_rna = run_single_config(
                    config_name + '_rna_only', dict(config), RNAOnlyDataset,
                    args.data_root, train_ids, val_ids, test_ids,
                    device, S, seed
                )
                all_results.append(result_rna)

            # Save incrementally (in case of crash)
            incremental_path = os.path.join(args.output_dir, 'scaling_results_incremental.json')
            with open(incremental_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    logger.info("\n" + "#" * 70)
    logger.info("# FINAL SUMMARY — Architecture Scaling")
    logger.info("#" * 70)

    summary = {}
    for config_name in args.configs:
        if config_name not in CONFIGS:
            continue

        # Full data runs
        runs = [r for r in all_results
                if r['config'] == config_name and r['dataset'] == 'MetaGNNDataset']
        if runs:
            agg = {'label': CONFIGS[config_name]['label']}
            agg['n_params'] = runs[0]['n_params']
            agg['hidden_dim'] = runs[0]['hidden_dim']
            agg['n_layers'] = runs[0]['n_layers']
            agg['heads'] = runs[0]['heads']
            for metric in ['F1_mean', 'AUROC_mean', 'AUPRC_mean', 'Precision_mean', 'Recall_mean']:
                vals = [r[metric] for r in runs]
                agg[metric] = round(float(np.mean(vals)), 4)
                agg[metric.replace('_mean', '_std_seeds')] = round(float(np.std(vals)), 4)
            agg['avg_train_time_s'] = round(float(np.mean([r['train_time_s'] for r in runs])), 1)
            agg['avg_peak_gpu_mb'] = round(float(np.mean([r['peak_gpu_mb'] for r in runs])), 0)
            agg['per_seed'] = runs
            summary[config_name] = agg

            logger.info(f"\n  {agg['label']} ({agg['n_params']:,} params)")
            logger.info(f"    F1:    {agg['F1_mean']:.4f} ± {agg.get('F1_std_seeds', 0):.4f}")
            logger.info(f"    AUROC: {agg['AUROC_mean']:.4f} ± {agg.get('AUROC_std_seeds', 0):.4f}")
            logger.info(f"    AUPRC: {agg['AUPRC_mean']:.4f} ± {agg.get('AUPRC_std_seeds', 0):.4f}")
            logger.info(f"    Time:  {agg['avg_train_time_s']:.1f}s, GPU: {agg['avg_peak_gpu_mb']:.0f} MB")

        # RNA-only runs
        rna_runs = [r for r in all_results
                    if r['config'] == config_name + '_rna_only']
        if rna_runs:
            rna_agg = {'label': CONFIGS[config_name]['label'] + ' (RNA-only)'}
            rna_agg['n_params'] = rna_runs[0]['n_params']
            for metric in ['F1_mean', 'AUROC_mean', 'AUPRC_mean']:
                vals = [r[metric] for r in rna_runs]
                rna_agg[metric] = round(float(np.mean(vals)), 4)
                rna_agg[metric.replace('_mean', '_std_seeds')] = round(float(np.std(vals)), 4)
            rna_agg['per_seed'] = rna_runs
            summary[config_name + '_rna_only'] = rna_agg

    # ── Save final results ────────────────────────────────────────────────────
    output = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'seeds': seeds,
            'n_train': len(train_ids),
            'n_val': len(val_ids),
            'n_test': len(test_ids),
            'configs_run': args.configs,
        },
        'summary': summary,
        'all_runs': all_results,
    }

    final_path = os.path.join(args.output_dir, 'scaling_results.json')
    with open(final_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Results saved to: {final_path}")
    logger.info(f"  Copy this file back to Cowork for analysis.")
    logger.info("  Done!")


if __name__ == '__main__':
    main()
