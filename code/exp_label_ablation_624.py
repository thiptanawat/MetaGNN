#!/usr/bin/env python3
"""
Label Ablation: Expression-Thresholded vs HMA Labels on 624 Patients
=====================================================================
Isolates the effect of pseudo-label quality on MetaGNN performance.

This script runs the same 5-fold CV as the full-cohort evaluation but
substitutes expression-thresholded consensus labels for HMA tissue-model
labels.  All other variables (features, graph, architecture) are held
constant, enabling a clean comparison:

  Condition A (already done): 2-D features + HMA labels  -> AUROC 0.663
  Condition B (this script):  2-D features + expr-thresh  -> AUROC ???

The difference isolates the label-quality effect.

Usage (on RTX 5090):
  conda activate metagnn
  python exp_label_ablation_624.py --data_root ..\..\MetaGNN-CRC\code\processed_690

Author: Thiptanawat Phongwattana
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score,
                             precision_recall_curve, auc,
                             precision_score, recall_score)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('label_ablation_624.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)


# ======================================================================
# MODEL (identical to run_full_cohort_experiments.py)
# ======================================================================
class MCDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


class HGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2,
                 residual=True, edge_types=None):
        super().__init__()
        self.residual = residual
        conv_dict = {
            ('metabolite', 'substrate_of', 'reaction'): GATv2Conv(
                in_channels['metabolite'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=False, concat=False),
            ('reaction', 'produces', 'metabolite'): GATv2Conv(
                in_channels['reaction'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=False, concat=False),
        }
        if edge_types is None or ('reaction', 'shared_metabolite', 'reaction') in edge_types:
            conv_dict[('reaction', 'shared_metabolite', 'reaction')] = GATv2Conv(
                in_channels['reaction'], out_channels, heads=heads,
                dropout=dropout, add_self_loops=True, concat=False)
        self.conv = HeteroConv(conv_dict, aggr='mean')
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
            out['reaction'] = self.norm_rxn(
                F.elu(out['reaction']) + self.proj_rxn(x_dict['reaction']))
            out['metabolite'] = self.norm_met(
                F.elu(out['metabolite']) + self.proj_met(x_dict['metabolite']))
        else:
            out['reaction'] = self.norm_rxn(F.elu(out['reaction']))
            out['metabolite'] = self.norm_met(F.elu(out['metabolite']))
        out['reaction'] = self.mc_drop(out['reaction'])
        out['metabolite'] = self.mc_drop(out['metabolite'])
        return out


class MetaGNN(nn.Module):
    def __init__(self, rxn_in_dim=2, met_in_dim=519, hidden_dim=128,
                 n_layers=2, heads=4, dropout=0.2, edge_types=None):
        super().__init__()
        self.proj_rxn = nn.Sequential(
            Linear(rxn_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.proj_met = nn.Sequential(
            Linear(met_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.layers = nn.ModuleList([
            HGATLayer({'reaction': hidden_dim, 'metabolite': hidden_dim},
                      hidden_dim, heads=heads, dropout=dropout,
                      edge_types=edge_types)
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


class MetaGNNLoss(nn.Module):
    def __init__(self, stoich_matrix=None, lambda_mb=0.2):
        super().__init__()
        if stoich_matrix is not None:
            self.register_buffer('S', stoich_matrix)
        else:
            self.S = None
        self.lambda_mb = lambda_mb
        self.bce = nn.BCELoss()

    def forward(self, s_r, y_r):
        l_bce = self.bce(s_r, y_r.float())
        if self.S is None or self.lambda_mb == 0:
            return l_bce
        n_rxn = self.S.shape[1]
        if s_r.shape[0] == n_rxn:
            net_flux = (self.S * s_r.unsqueeze(0)).sum(dim=1)
            l_mb = net_flux.pow(2).mean()
        else:
            l_mb = torch.tensor(0.0, device=s_r.device)
        return l_bce + self.lambda_mb * l_mb


# ======================================================================
# EXPRESSION-THRESHOLDED LABEL GENERATION
# ======================================================================
def generate_expression_thresholded_labels(data_root, patient_ids):
    """
    Generate expression-thresholded consensus labels from GPR-mapped
    reaction features, analogous to the 220-patient labelling strategy.

    For each reaction:
      1. Collect X_R[rxn, 0] (GPR-aggregated expression) across all patients
      2. If all patients have expression == 0 -> no GPR -> default ACTIVE
      3. Otherwise, compute cohort median; patient labels rxn as active if
         expression > median; consensus = majority vote (> 50% patients)

    Returns: (y_r, stats_dict)
    """
    feat_dir = os.path.join(data_root, 'reaction_features')
    n_patients = len(patient_ids)

    # Load first patient to get n_reactions
    with h5py.File(os.path.join(feat_dir, f'{patient_ids[0]}.h5'), 'r') as f:
        n_rxn = f['X_R'].shape[0]

    logger.info(f"  Generating expression-thresholded labels for {n_patients} "
                f"patients, {n_rxn} reactions...")

    # Build expression matrix: (n_patients, n_rxn)
    expr_matrix = np.zeros((n_patients, n_rxn), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        with h5py.File(os.path.join(feat_dir, f'{pid}.h5'), 'r') as f:
            expr_matrix[i] = f['X_R'][:, 0]  # GPR-aggregated RNA-seq
        if (i + 1) % 100 == 0:
            logger.info(f"    Loaded {i+1}/{n_patients} patients")

    # Identify GPR vs non-GPR reactions
    has_expression = (expr_matrix > 0).any(axis=0)  # True if any patient > 0
    n_gpr = int(has_expression.sum())
    n_nogpr = n_rxn - n_gpr
    logger.info(f"  GPR-mapped reactions: {n_gpr}, non-GPR: {n_nogpr}")

    # Compute cohort median per reaction (across patients)
    # Only for GPR reactions; non-GPR have median = 0
    cohort_median = np.median(expr_matrix, axis=0)

    # Per-patient binary activity: active if expression > cohort median
    per_patient_active = (expr_matrix > cohort_median).astype(np.float32)

    # Consensus: majority vote (> 50% of patients)
    consensus_frac = per_patient_active.mean(axis=0)
    consensus_labels = (consensus_frac > 0.5).astype(np.float32)

    # Non-GPR reactions: default active (same convention as 220-patient pipeline)
    consensus_labels[~has_expression] = 1.0

    n_active = int(consensus_labels.sum())
    n_inactive = n_rxn - n_active
    pct_active = 100.0 * n_active / n_rxn

    stats = {
        'n_reactions': n_rxn,
        'n_gpr': n_gpr,
        'n_nogpr': n_nogpr,
        'n_active': n_active,
        'n_inactive': n_inactive,
        'pct_active': round(pct_active, 1),
    }
    logger.info(f"  Expression-thresholded labels: {n_active} active / "
                f"{n_inactive} inactive ({pct_active:.1f}% active)")

    y_r = torch.tensor(consensus_labels, dtype=torch.float32)
    return y_r, stats


# ======================================================================
# DATASET
# ======================================================================
def sparsify_shared_edges(shared_ei, max_k=10):
    """Top-k sparsification of shared_metabolite edges."""
    n_nodes = shared_ei.max().item() + 1
    if shared_ei.shape[1] <= max_k * n_nodes:
        return shared_ei
    n_orig = shared_ei.shape[1]
    src = shared_ei[0]
    perm = torch.randperm(n_orig)
    sort_idx = src[perm].argsort(stable=True)
    src_sorted = src[perm][sort_idx]
    _, counts = torch.unique_consecutive(src_sorted, return_counts=True)
    offsets = counts.cumsum(0)
    starts = torch.zeros_like(offsets)
    starts[1:] = offsets[:-1]
    group_id = torch.repeat_interleave(torch.arange(len(counts)), counts)
    rank = torch.arange(n_orig) - starts[group_id]
    kept_indices = perm[sort_idx[rank < max_k]]
    shared_ei = shared_ei[:, kept_indices]
    logger.info(f"  Sparsified shared_metabolite: {n_orig:,} -> "
                f"{shared_ei.shape[1]:,} edges (top-{max_k}/node)")
    return shared_ei


class LabelAblationDataset(torch.utils.data.Dataset):
    """Dataset with pluggable labels (either HMA or expression-thresholded)."""

    def __init__(self, data_root, patient_ids, y_r):
        self.data_root = data_root
        self.patient_ids = list(patient_ids)
        self.y_r = y_r

        # Load metabolite features
        with h5py.File(os.path.join(data_root, 'metabolite_features.h5'), 'r') as f:
            self.X_M = torch.tensor(f['X_M'][:], dtype=torch.float32)

        # Load edges
        edge_dir = os.path.join(data_root, 'edge_indices')
        self.edge_index_dict = {
            ('metabolite', 'substrate_of', 'reaction'):
                torch.load(os.path.join(edge_dir, 'substrate_of.pt'), weights_only=True),
            ('reaction', 'produces', 'metabolite'):
                torch.load(os.path.join(edge_dir, 'produces.pt'), weights_only=True),
        }
        shared_path = os.path.join(edge_dir, 'shared_metabolite.pt')
        if os.path.exists(shared_path):
            shared_ei = torch.load(shared_path, weights_only=True)
            shared_ei = sparsify_shared_edges(shared_ei, max_k=10)
            self.edge_index_dict[('reaction', 'shared_metabolite', 'reaction')] = shared_ei

        self.feat_dir = os.path.join(data_root, 'reaction_features')

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        with h5py.File(os.path.join(self.feat_dir, f'{pid}.h5'), 'r') as f:
            X_R = torch.tensor(f['X_R'][:], dtype=torch.float32)

        data = HeteroData()
        data['reaction'].x = X_R
        data['metabolite'].x = self.X_M
        data['reaction'].y = self.y_r
        data['reaction'].pid = pid
        for rel, ei in self.edge_index_dict.items():
            src_type, rel_type, dst_type = rel
            data[src_type, rel_type, dst_type].edge_index = ei
        return data


# ======================================================================
# TRAINING & EVALUATION
# ======================================================================
TRAIN_CFG = dict(
    dropout=0.20, lr=1e-3, weight_decay=1e-5,
    n_epochs=80, patience=15, threshold=0.15, lambda_mb=0.2,
)

CONFIG_B = dict(hidden_dim=256, n_layers=3, heads=8, batch_size=1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(out, data['reaction'].y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * data['reaction'].x.shape[0]
        n += data['reaction'].x.shape[0]
    return total_loss / max(n, 1)


def evaluate_batch(model, loader, criterion, device, threshold=0.15):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0
    n = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x_dict, data.edge_index_dict)
            loss = criterion(out, data['reaction'].y)
            total_loss += loss.item() * data['reaction'].x.shape[0]
            n += data['reaction'].x.shape[0]
            all_scores.append(out.cpu())
            all_labels.append(data['reaction'].y.cpu())
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (scores > threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    auroc = roc_auc_score(labels, scores)
    prec_arr, rec_arr, _ = precision_recall_curve(labels, scores)
    auprc = auc(rec_arr, prec_arr)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    return {
        'loss': total_loss / max(n, 1),
        'F1': f1, 'AUROC': auroc, 'AUPRC': auprc,
        'Precision': prec, 'Recall': rec,
    }


def run_fold(fold_idx, train_ids, val_ids, test_ids, data_root, y_r, device):
    """Train one fold with Config B and return metrics dict."""
    cfg = CONFIG_B
    tcfg = TRAIN_CFG

    train_ds = LabelAblationDataset(data_root, train_ids, y_r)
    val_ds = LabelAblationDataset(data_root, val_ids, y_r)
    test_ds = LabelAblationDataset(data_root, test_ids, y_r)

    train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = PyGLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
    test_loader = PyGLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

    # Detect edge types from first sample
    sample = train_ds[0]
    edge_types = list(sample.edge_types)

    rxn_in = sample['reaction'].x.shape[1]
    met_in = sample['metabolite'].x.shape[1]

    model = MetaGNN(
        rxn_in_dim=rxn_in, met_in_dim=met_in,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        heads=cfg['heads'], dropout=tcfg['dropout'],
        edge_types=edge_types,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Fold {fold_idx}: {n_params:,} params, "
                f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    optimizer = optim.AdamW(model.parameters(), lr=tcfg['lr'],
                            weight_decay=tcfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg['n_epochs'])
    criterion = MetaGNNLoss(lambda_mb=tcfg['lambda_mb']).to(device)

    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, tcfg['n_epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_metrics = evaluate_batch(model, val_loader, criterion, device,
                                     threshold=tcfg['threshold'])

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"    Epoch {epoch:3d}  loss={train_loss:.4f}  "
                        f"val_F1={val_metrics['F1']:.4f}  "
                        f"val_AUROC={val_metrics['AUROC']:.4f}")

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tcfg['patience']:
                logger.info(f"    Early stop at epoch {epoch}")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    test_metrics = evaluate_batch(model, test_loader, criterion, device,
                                  threshold=tcfg['threshold'])

    peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    result = {
        'fold': fold_idx,
        'F1_mean': test_metrics['F1'],
        'AUROC_mean': test_metrics['AUROC'],
        'AUPRC_mean': test_metrics['AUPRC'],
        'Precision_mean': test_metrics['Precision'],
        'Recall_mean': test_metrics['Recall'],
        'best_val_f1': best_val_f1,
        'train_time_s': round(train_time, 1),
        'best_epoch': epoch - patience_counter if patience_counter >= tcfg['patience'] else epoch,
        'peak_gpu_mb': round(peak_gpu_mb),
        'n_params': n_params,
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'n_test': len(test_ids),
    }
    logger.info(f"  Fold {fold_idx} done: F1={test_metrics['F1']:.4f}  "
                f"AUROC={test_metrics['AUROC']:.4f}  time={train_time:.0f}s")
    return result


# ======================================================================
# MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Label ablation: expression-thresholded vs HMA on 624 patients")
    parser.add_argument('--data_root', required=True,
                        help='Path to processed_690 directory')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output_dir', default='results_label_ablation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                    f"VRAM: {total_mem / 1e9:.1f} GB")

    # Load patient IDs
    feat_dir = os.path.join(args.data_root, 'reaction_features')
    patient_ids = sorted([
        f.replace('.h5', '') for f in os.listdir(feat_dir) if f.endswith('.h5')
    ])
    n_patients = len(patient_ids)
    logger.info(f"Found {n_patients} patients in {feat_dir}")

    # ---- Generate expression-thresholded labels ----
    logger.info("=" * 60)
    logger.info("GENERATING EXPRESSION-THRESHOLDED CONSENSUS LABELS")
    logger.info("=" * 60)
    y_expr, label_stats = generate_expression_thresholded_labels(
        args.data_root, patient_ids)

    # Also load HMA labels for comparison
    hma_path = os.path.join(args.data_root, 'activity_pseudolabels.pt')
    y_hma = torch.load(hma_path, weights_only=True)
    hma_active = int(y_hma.sum())
    logger.info(f"  HMA labels for reference: {hma_active} active / "
                f"{len(y_hma) - hma_active} inactive "
                f"({100*hma_active/len(y_hma):.1f}%)")

    # Label agreement
    agreement = (y_expr == y_hma).float().mean().item()
    logger.info(f"  Label agreement (expr-thresh vs HMA): {agreement:.1%}")

    # Save expression-thresholded labels
    label_save_path = os.path.join(args.data_root,
                                    'activity_pseudolabels_expr_thresh.pt')
    torch.save(y_expr, label_save_path)
    logger.info(f"  Saved expression-thresholded labels to {label_save_path}")

    # ---- 5-fold CV with expression-thresholded labels ----
    logger.info("=" * 60)
    logger.info("5-FOLD CV WITH EXPRESSION-THRESHOLDED LABELS (Config B)")
    logger.info("=" * 60)

    # Stratification: use simple random for now (no MSI labels in processed_690)
    np.random.seed(args.seed)
    strat_labels = np.zeros(n_patients, dtype=int)  # dummy stratification

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)

    fold_results = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(
            skf.split(np.arange(n_patients), strat_labels), 1):

        # Split train_val into train and val (85/15 of train_val)
        n_val = max(1, len(train_val_idx) // (args.n_folds + 1))
        val_idx = train_val_idx[:n_val]
        train_idx = train_val_idx[n_val:]

        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]
        test_ids = [patient_ids[i] for i in test_idx]

        logger.info(f"\n-- Fold {fold_idx}/{args.n_folds} --")

        try:
            result = run_fold(fold_idx, train_ids, val_ids, test_ids,
                              args.data_root, y_expr, device)
            fold_results.append(result)
        except torch.cuda.OutOfMemoryError:
            logger.error(f"  OOM at fold {fold_idx}! Skipping.")
            torch.cuda.empty_cache()
            continue

        # Save incrementally
        _save_results(fold_results, label_stats, args, y_hma, y_expr)

    # ---- Final summary ----
    if fold_results:
        _save_results(fold_results, label_stats, args, y_hma, y_expr)
        _print_summary(fold_results)
    else:
        logger.error("No folds completed successfully!")


def _save_results(fold_results, label_stats, args, y_hma, y_expr):
    """Save current results to JSON."""
    metrics = ['F1_mean', 'AUROC_mean', 'AUPRC_mean', 'Precision_mean', 'Recall_mean']
    summary = {
        'experiment': 'label_ablation_expression_thresholded',
        'n_folds_completed': len(fold_results),
        'n_patients': label_stats['n_reactions'],  # will fix below
        'config': 'B_paper (256/3/8)',
        'seed': 2024,
        'label_stats': label_stats,
        'hma_label_stats': {
            'n_active': int(y_hma.sum()),
            'n_inactive': int(len(y_hma) - y_hma.sum()),
            'pct_active': round(100 * y_hma.sum().item() / len(y_hma), 1),
        },
        'label_agreement': round((y_expr == y_hma).float().mean().item(), 4),
    }
    for m in metrics:
        vals = [r[m] for r in fold_results]
        summary[m.replace('_mean', '_mean')] = round(np.mean(vals), 4)
        summary[m.replace('_mean', '_std')] = round(np.std(vals), 4)

    summary['per_fold'] = fold_results

    out_path = os.path.join(args.output_dir, 'label_ablation_expr_thresh.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Results saved to {out_path}")


def _print_summary(fold_results):
    """Print final summary."""
    logger.info("\n" + "=" * 60)
    logger.info("LABEL ABLATION SUMMARY (Expression-Thresholded Labels)")
    logger.info("=" * 60)
    metrics = ['F1_mean', 'AUROC_mean', 'AUPRC_mean', 'Precision_mean', 'Recall_mean']
    for m in metrics:
        vals = [r[m] for r in fold_results]
        name = m.replace('_mean', '')
        logger.info(f"  {name:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    logger.info("\nCompare with HMA-label results:")
    logger.info("  F1          : 0.4460 +/- 0.0041  (HMA labels)")
    logger.info("  AUROC       : 0.6631 +/- 0.0010  (HMA labels)")
    logger.info("\nIf expression-thresholded AUROC >> 0.663, the label quality")
    logger.info("hypothesis is confirmed: supervision alignment drives performance.")


if __name__ == '__main__':
    main()
