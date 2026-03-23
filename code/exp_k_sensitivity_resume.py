#!/usr/bin/env python3
"""
Resume k-sensitivity experiment from crash
============================================
Parses the existing log to recover completed results, then runs only
the remaining k values / folds.

Usage:
  python exp_k_sensitivity_resume.py \
      --data_root /path/to/processed_690 \
      --prev_log k_sensitivity.log \
      --k_values 50 \
      --skip_all

  # Or to also attempt k=100 (may OOM on 32GB):
  python exp_k_sensitivity_resume.py \
      --data_root /path/to/processed_690 \
      --prev_log k_sensitivity.log \
      --k_values 50,100

Author: Thiptanawat Phongwattana
"""

import os
import re
import sys
import time
import json
import logging
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
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
        logging.FileHandler('k_sensitivity_resume.log', mode='a', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log parser: extract completed results from previous run
# ---------------------------------------------------------------------------
def parse_previous_log(log_path):
    """Parse completed fold results from a previous k_sensitivity log.

    Returns:
        completed: dict mapping (k_str, fold_idx) -> result_dict
        partial_results: list of result dicts for completed folds
    """
    completed = {}
    partial_results = []

    if not os.path.exists(log_path):
        logger.warning(f"Previous log not found: {log_path}")
        return completed, partial_results

    current_k = None
    current_fold = None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Match k value header
        k_match = re.search(r'k\s*=\s*(\d+|all)\s+\((\d+)-fold CV\)', line)
        if k_match:
            current_k = k_match.group(1)
            continue

        # Match fold header
        fold_match = re.search(r'Fold\s+(\d+)/(\d+):\s+Train=(\d+),\s+Val=(\d+),\s+Test=(\d+)', line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            continue

        # Match completed result line
        result_match = re.search(
            r'Result:\s+F1=([\d.]+)\+/-([\d.]+)\s+AUROC=([\d.]+)\s+'
            r'Edges=([\d,]+)\s+GPU=(\d+)MB\s+Time=(\d+)s',
            line
        )
        if result_match and current_k is not None and current_fold is not None:
            result = {
                'k_shared': current_k,
                'fold': current_fold,
                'F1_mean': float(result_match.group(1)),
                'F1_std': float(result_match.group(2)),
                'AUROC_mean': float(result_match.group(3)),
                'n_shared_edges': int(result_match.group(4).replace(',', '')),
                'peak_gpu_mb': float(result_match.group(5)),
                'train_time_s': float(result_match.group(6)),
            }
            key = (current_k, current_fold)
            completed[key] = result
            partial_results.append(result)
            logger.info(f"  Recovered: k={current_k}, fold={current_fold}, "
                        f"F1={result['F1_mean']:.4f}, AUROC={result['AUROC_mean']:.4f}")

    logger.info(f"Recovered {len(completed)} completed fold(s) from previous log")
    return completed, partial_results


# ---------------------------------------------------------------------------
# Model (identical to original)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Sparsification utility
# ---------------------------------------------------------------------------
def sparsify_shared_edges(shared_ei: torch.Tensor, k: int) -> torch.Tensor:
    n_orig = shared_ei.shape[1]
    n_nodes = shared_ei.max().item() + 1
    if k >= n_orig // max(n_nodes, 1):
        return shared_ei
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
    kept_indices = perm[sort_idx[rank < k]]
    return shared_ei[:, kept_indices]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class KSensitivityDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, patient_ids, k_shared='all'):
        self.data_root = data_root
        self.patient_ids = list(patient_ids)
        self.k_shared = k_shared

        with h5py.File(os.path.join(data_root, 'metabolite_features.h5'), 'r') as f:
            self.X_M = torch.tensor(f['X_M'][:], dtype=torch.float32)

        edge_dir = os.path.join(data_root, 'edge_indices')
        self.edge_index_dict = {
            ('metabolite', 'substrate_of', 'reaction'):
                torch.load(os.path.join(edge_dir, 'substrate_of.pt'), weights_only=True),
            ('reaction', 'produces', 'metabolite'):
                torch.load(os.path.join(edge_dir, 'produces.pt'), weights_only=True),
        }

        shared_path = os.path.join(edge_dir, 'shared_metabolite.pt')
        if os.path.exists(shared_path) and k_shared != 0:
            shared_ei = torch.load(shared_path, weights_only=True)
            if isinstance(k_shared, int) and k_shared > 0:
                shared_ei = sparsify_shared_edges(shared_ei, k_shared)
            self.edge_index_dict[('reaction', 'shared_metabolite', 'reaction')] = shared_ei
            self.n_shared_edges = shared_ei.shape[1]
        else:
            self.n_shared_edges = 0

        self.y_r = torch.load(
            os.path.join(data_root, 'activity_pseudolabels.pt'), weights_only=True)
        self.feat_dir = os.path.join(data_root, 'reaction_features')

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        with h5py.File(os.path.join(self.feat_dir, f'{pid}.h5'), 'r') as f:
            X_R = torch.tensor(f['X_R'][:], dtype=torch.float32)

        data = HeteroData()
        data['reaction'].x = X_R
        data['reaction'].y = self.y_r
        data['metabolite'].x = self.X_M.clone()
        for rel, ei in self.edge_index_dict.items():
            data[rel].edge_index = ei
        return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
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
    return {'F1': f1, 'AUROC': auroc, 'AUPRC': auprc,
            'Precision': prec, 'Recall': rec}


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------
TRAIN_CFG = dict(
    dropout=0.20, lr=1e-3, weight_decay=1e-4,
    n_epochs=100, patience=20, lambda_mb=0.2, threshold=0.15,
)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        s_r = model(
            x_dict={'reaction': batch['reaction'].x,
                    'metabolite': batch['metabolite'].x},
            edge_index_dict={rel: batch[rel].edge_index
                            for rel in batch.edge_types},
        )
        loss = criterion(s_r, batch['reaction'].y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_per_patient(model, dataset, device, threshold=0.15):
    model.eval()
    patient_metrics = []
    for i in range(len(dataset)):
        data = dataset[i].to(device)
        s_r = model(
            x_dict={'reaction': data['reaction'].x,
                    'metabolite': data['metabolite'].x},
            edge_index_dict={rel: data[rel].edge_index
                            for rel in data.edge_types},
        )
        y_pred = s_r.cpu().numpy()
        y_true = data['reaction'].y.cpu().numpy()
        m = compute_metrics(y_pred, y_true, threshold)
        patient_metrics.append(m)

    result = {}
    for key in ['F1', 'AUROC', 'AUPRC', 'Precision', 'Recall']:
        vals = [m[key] for m in patient_metrics if not np.isnan(m[key])]
        result[f'{key}_mean'] = float(np.mean(vals)) if vals else float('nan')
        result[f'{key}_std']  = float(np.std(vals)) if vals else float('nan')
    return result


@torch.no_grad()
def evaluate_batch(model, loader, criterion, device, threshold=0.15):
    model.eval()
    all_pred, all_true, total_loss = [], [], 0.0
    for batch in loader:
        batch = batch.to(device)
        s_r = model(
            x_dict={'reaction': batch['reaction'].x,
                    'metabolite': batch['metabolite'].x},
            edge_index_dict={rel: batch[rel].edge_index
                            for rel in batch.edge_types},
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


def run_single_fold(data_root, k_shared, train_ids, val_ids, test_ids,
                    device, stoich_matrix, seed, n_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    train_ds = KSensitivityDataset(data_root, train_ids, k_shared=k_shared)
    val_ds   = KSensitivityDataset(data_root, val_ids, k_shared=k_shared)
    test_ds  = KSensitivityDataset(data_root, test_ids, k_shared=k_shared)

    n_shared = train_ds.n_shared_edges
    edge_types = list(train_ds.edge_index_dict.keys())

    sample = train_ds[0]
    rxn_in_dim = sample['reaction'].x.shape[1]
    met_in_dim = sample['metabolite'].x.shape[1]

    train_loader = PyGLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = PyGLoader(val_ds, batch_size=1, shuffle=False)

    cfg = dict(TRAIN_CFG)
    cfg['n_epochs'] = n_epochs

    model = MetaGNN(
        rxn_in_dim=rxn_in_dim, met_in_dim=met_in_dim,
        hidden_dim=256, n_layers=3, heads=8, dropout=cfg['dropout'],
        edge_types=edge_types,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = MetaGNNLoss(
        stoich_matrix=stoich_matrix.to(device) if stoich_matrix is not None else None,
        lambda_mb=cfg['lambda_mb'],
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'])

    best_val_f1, patience_counter, best_state, best_epoch = 0.0, 0, None, 0
    t0 = time.time()

    for epoch in range(1, cfg['n_epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_batch(model, val_loader, criterion, device, cfg['threshold'])
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"      Epoch {epoch:3d}  loss={train_loss:.4f}  "
                        f"val_F1={val_metrics['F1']:.4f}  val_AUROC={val_metrics['AUROC']:.4f}")

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                logger.info(f"      Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t0
    peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if device.type == 'cuda' else 0)

    model.load_state_dict(best_state)
    test_results = evaluate_per_patient(model, test_ds, device, cfg['threshold'])

    test_results.update({
        'k_shared': str(k_shared),
        'n_shared_edges': n_shared,
        'seed': seed,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'train_time_s': round(train_time, 1),
        'peak_gpu_mb': round(peak_gpu_mb, 0),
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'n_test': len(test_ids),
    })
    return test_results


# ---------------------------------------------------------------------------
# Main: resume from previous log
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Resume k-sensitivity experiment from crash')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--prev_log', type=str, default='k_sensitivity.log',
                        help='Path to previous run log to recover results from')
    parser.add_argument('--output_dir', type=str, default='./results_k_sensitivity')
    parser.add_argument('--k_values', type=str, default='50',
                        help='Comma-separated k values to run (only incomplete ones)')
    parser.add_argument('--skip_all', action='store_true', default=True)
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  RESUMING k-sensitivity experiment")
    logger.info("=" * 60)

    # Step 1: Parse previous log to recover completed results
    logger.info(f"\nParsing previous log: {args.prev_log}")
    completed, recovered_results = parse_previous_log(args.prev_log)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load patient IDs (same as original)
    meta_path = os.path.join(args.data_root, 'clinical_metadata.tsv')
    meta_df = pd.read_csv(meta_path, sep='\t')
    patient_ids = meta_df['tcga_barcode'].tolist()
    strat = meta_df['msi_status'].fillna('Unknown').tolist()
    logger.info(f"Loaded {len(patient_ids)} patients")

    # Stoichiometric matrix
    stoich_path = os.path.join(args.data_root, 'recon3d_stoich.h5')
    stoich_matrix = None
    if os.path.exists(stoich_path):
        with h5py.File(stoich_path, 'r') as f:
            stoich_matrix = torch.tensor(f['S'][:], dtype=torch.float32)

    # Parse k values to run
    k_values = [int(x) for x in args.k_values.split(',')]
    logger.info(f"k values to run: {k_values}")

    # CV splits (MUST match original for reproducibility)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(patient_ids, strat))

    all_results = list(recovered_results)  # Start with recovered results

    for k in k_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"  k = {k}  ({args.n_folds}-fold CV)")
        logger.info(f"{'='*60}")

        for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
            fold_num = fold_idx + 1
            key = (str(k), fold_num)

            # Skip if already completed
            if key in completed:
                logger.info(f"\n    Fold {fold_num}/{args.n_folds}: SKIPPED (recovered from log)")
                continue

            train_val_ids = [patient_ids[i] for i in train_val_idx]
            test_ids = [patient_ids[i] for i in test_idx]

            val_size = max(1, len(train_val_ids) // 5)
            np.random.seed(args.seed + fold_idx)
            val_idx = np.random.choice(len(train_val_ids), val_size, replace=False)
            val_mask = np.zeros(len(train_val_ids), dtype=bool)
            val_mask[val_idx] = True
            train_ids = [train_val_ids[i] for i in range(len(train_val_ids)) if not val_mask[i]]
            val_ids_list = [train_val_ids[i] for i in range(len(train_val_ids)) if val_mask[i]]

            logger.info(f"\n    Fold {fold_num}/{args.n_folds}: "
                        f"Train={len(train_ids)}, Val={len(val_ids_list)}, Test={len(test_ids)}")

            try:
                result = run_single_fold(
                    args.data_root, k_shared=k,
                    train_ids=train_ids, val_ids=val_ids_list, test_ids=test_ids,
                    device=device, stoich_matrix=stoich_matrix,
                    seed=args.seed + fold_idx,
                    n_epochs=args.n_epochs,
                )
                result['fold'] = fold_num
                all_results.append(result)

                logger.info(f"    Result: F1={result['F1_mean']:.4f}+/-{result['F1_std']:.4f}  "
                            f"AUROC={result['AUROC_mean']:.4f}  "
                            f"Edges={result['n_shared_edges']:,}  "
                            f"GPU={result['peak_gpu_mb']:.0f}MB  "
                            f"Time={result['train_time_s']:.0f}s")

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"    OOM at k={k}, fold {fold_num} -- skipping")
                all_results.append({
                    'k_shared': str(k), 'fold': fold_num,
                    'F1_mean': float('nan'), 'AUROC_mean': float('nan'),
                    'AUPRC_mean': float('nan'), 'n_shared_edges': -1,
                    'peak_gpu_mb': -1, 'train_time_s': -1,
                    'note': 'OOM',
                })
                torch.cuda.empty_cache()
                break

            # Save after each fold
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(args.output_dir, 'k_sensitivity_results.csv'), index=False)

    # -----------------------------------------------------------------------
    # Final summary (all k values including recovered)
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'k_sensitivity_results.csv'), index=False)

    # Determine all k values present
    all_k_values = df['k_shared'].unique().tolist()
    # Sort numerically
    def sort_key(x):
        try:
            return int(x)
        except ValueError:
            return 999999
    all_k_values.sort(key=sort_key)

    summary_rows = []
    for k_str in all_k_values:
        subset = df[df['k_shared'] == k_str]
        valid = subset.dropna(subset=['F1_mean'])
        if len(valid) == 0:
            summary_rows.append({'k': k_str, 'n_folds': 0, 'note': 'OOM'})
            continue
        row = {
            'k': k_str,
            'n_folds': len(valid),
            'n_shared_edges': int(valid['n_shared_edges'].iloc[0]) if 'n_shared_edges' in valid else 0,
            'F1_mean': valid['F1_mean'].mean(),
            'F1_std': valid['F1_mean'].std(),
            'AUROC_mean': valid['AUROC_mean'].mean(),
            'AUROC_std': valid['AUROC_mean'].std(),
            'avg_train_time_s': valid['train_time_s'].mean() if 'train_time_s' in valid else 0,
            'avg_peak_gpu_mb': valid['peak_gpu_mb'].mean() if 'peak_gpu_mb' in valid else 0,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, 'k_sensitivity_summary.csv'), index=False)

    logger.info("\n" + "=" * 70)
    logger.info("  SENSITIVITY ANALYSIS SUMMARY (including recovered results)")
    logger.info("=" * 70)
    for _, row in summary_df.iterrows():
        if 'note' in row and row.get('note') == 'OOM':
            logger.info(f"  k={row['k']:>5s}  OOM (out of memory)")
        else:
            logger.info(
                f"  k={row['k']:>5s}  |  edges={int(row.get('n_shared_edges', 0)):>8,}  |  "
                f"F1={row['F1_mean']:.4f}+/-{row['F1_std']:.4f}  |  "
                f"AUROC={row['AUROC_mean']:.4f}+/-{row['AUROC_std']:.4f}  |  "
                f"GPU={row['avg_peak_gpu_mb']:.0f}MB  |  "
                f"Time={row['avg_train_time_s']:.0f}s")

    # -----------------------------------------------------------------------
    # Generate plot
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        valid_summary = summary_df.dropna(subset=['F1_mean'])
        if len(valid_summary) >= 2:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            x_labels = valid_summary['k'].astype(str).tolist()
            x_pos = np.arange(len(x_labels))

            ax = axes[0]
            ax.bar(x_pos, valid_summary['F1_mean'],
                   yerr=valid_summary['F1_std'], capsize=4,
                   color='#2196F3', alpha=0.8, edgecolor='#1565C0')
            ax.set_xlabel('k (neighbours per reaction)')
            ax.set_ylabel('F1 Score')
            ax.set_title('F1 vs. Edge Sparsification')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.grid(axis='y', alpha=0.3)

            ax = axes[1]
            ax.bar(x_pos, valid_summary['AUROC_mean'],
                   yerr=valid_summary['AUROC_std'], capsize=4,
                   color='#4CAF50', alpha=0.8, edgecolor='#2E7D32')
            ax.set_xlabel('k (neighbours per reaction)')
            ax.set_ylabel('AUROC')
            ax.set_title('AUROC vs. Edge Sparsification')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.grid(axis='y', alpha=0.3)

            ax = axes[2]
            mem_gb = valid_summary['avg_peak_gpu_mb'] / 1000
            ax.bar(x_pos, mem_gb,
                   color='#FF9800', alpha=0.8, edgecolor='#E65100')
            ax.set_xlabel('k (neighbours per reaction)')
            ax.set_ylabel('Peak GPU Memory (GB)')
            ax.set_title('VRAM Usage vs. Edge Sparsification')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.axhline(y=32, color='red', linestyle='--', alpha=0.5, label='RTX 5090 VRAM')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.suptitle('Shared-Metabolite Edge Sparsification Sensitivity (624 patients, 3-fold CV)',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            plot_path = os.path.join(args.output_dir, 'k_sensitivity_plot.png')
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            logger.info(f"\nPlot saved: {plot_path}")

    except ImportError:
        logger.warning("matplotlib not installed -- skipping plot generation")

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
