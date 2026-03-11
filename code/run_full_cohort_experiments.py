#!/usr/bin/env python3
"""
MetaGNN Full-Cohort Experiments — Combined Overnight Run
=========================================================
Runs three experiment phases to replace "future work" statements:

  Phase 1: 5-fold Stratified CV on real patient data (GPU)
           → Replaces: "cross-validation on the full TCGA cohort is planned"
  Phase 2: Architecture Scaling on real patient data (GPU)
           → Replaces: "architecture comparison on the full cohort is planned"
  Phase 3: FBA Viability on ALL patients (CPU)
           → Replaces: "full FBA-based reconstruction is deferred"

Supports two data formats:
  A) MetaGNN-CRC (DIB) format: .pt files in patient_features/
  B) MethodsX evaluation format: .h5 files in reaction_features/

Usage:
  conda activate metagnn
  python run_full_cohort_experiments.py --data_root <path_to_data>

  # If you have the MetaGNN-CRC dataset (690 patients):
  python run_full_cohort_experiments.py --data_root /path/to/MetaGNN-CRC/data/processed

  # If using MethodsX evaluation data (220 patients):
  python run_full_cohort_experiments.py --data_root ../data

  # For FBA (Phase 3), add:
  python run_full_cohort_experiments.py --data_root ../data --recon3d_xml /path/to/Recon3D.xml

  # To skip specific phases:
  python run_full_cohort_experiments.py --data_root ../data --skip_fba --skip_scaling

Author: Thiptanawat Phongwattana
Hardware target: Windows 11 + Core i9-285K + RTX 5090 (32 GB VRAM)
Estimated runtime: ~8-12 hours overnight (all phases)
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score,
                             precision_recall_curve, auc,
                             precision_score, recall_score)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('full_cohort_experiments.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════
CONFIGS = {
    'A_current': dict(
        hidden_dim=128, n_layers=2, heads=4,
        label='Current (128/2/4)',
        batch_size=1,
    ),
    'B_paper': dict(
        hidden_dim=256, n_layers=3, heads=8,
        label='Paper (256/3/8)',
        batch_size=1,
    ),
    'C_large': dict(
        hidden_dim=512, n_layers=3, heads=8,
        label='Large (512/3/8)',
        batch_size=1,
    ),
    'D_deep': dict(
        hidden_dim=256, n_layers=5, heads=8,
        label='Deep (256/5/8)',
        batch_size=1,
    ),
}

TRAIN_CFG = dict(
    dropout=0.20,
    lr=1e-3,
    weight_decay=1e-5,
    n_epochs=80,
    patience=15,
    threshold=0.15,
    lambda_mb=0.2,
)

SEEDS = [2024, 42, 123]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (self-contained)
# ═══════════════════════════════════════════════════════════════════════════════
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
        # Add shared_metabolite if present in data
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


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (auto-detect format)
# ═══════════════════════════════════════════════════════════════════════════════
def detect_data_format(data_root):
    """Detect whether this is MetaGNN-CRC (.pt) or MethodsX (.h5) format."""
    pt_dir = os.path.join(data_root, 'patient_features')
    h5_dir = os.path.join(data_root, 'reaction_features')
    if os.path.isdir(pt_dir):
        return 'metagnn_crc'
    elif os.path.isdir(h5_dir):
        return 'methodsx'
    else:
        raise FileNotFoundError(
            f"Cannot detect data format in {data_root}. "
            f"Expected 'patient_features/' (MetaGNN-CRC) or 'reaction_features/' (MethodsX)."
        )


class UnifiedDataset(torch.utils.data.Dataset):
    """Loads patient graphs from either MetaGNN-CRC or MethodsX format."""

    def __init__(self, data_root, patient_ids, data_format='auto'):
        self.data_root = data_root
        if data_format == 'auto':
            data_format = detect_data_format(data_root)
        self.data_format = data_format
        self.patient_ids = list(patient_ids)

        if data_format == 'metagnn_crc':
            self._load_metagnn_crc()
        else:
            self._load_methodsx()

    def _load_metagnn_crc(self):
        """Load from MetaGNN-CRC format (DIB dataset)."""
        # Graph structure
        graph_path = os.path.join(self.data_root, 'graph_structure.pt')
        graph = torch.load(graph_path, weights_only=False)
        self.edge_index_dict = {}
        for rel in graph.edge_types:
            self.edge_index_dict[rel] = graph[rel].edge_index

        # Metabolite features
        self.X_M = graph['metabolite'].x

        # Labels
        label_path = os.path.join(self.data_root, 'hma_labels_thresholded.pt')
        if not os.path.exists(label_path):
            label_path = os.path.join(self.data_root, 'activity_pseudolabels.pt')
        self.y_r = torch.load(label_path, weights_only=True)

        self.feat_dir = os.path.join(self.data_root, 'patient_features')
        self.feat_ext = '.pt'

    def _load_methodsx(self):
        """Load from MethodsX format (evaluation data)."""
        with h5py.File(os.path.join(self.data_root, 'metabolite_features.h5'), 'r') as f:
            self.X_M = torch.tensor(f['X_M'][:], dtype=torch.float32)

        edge_dir = os.path.join(self.data_root, 'edge_indices')
        self.edge_index_dict = {
            ('metabolite', 'substrate_of', 'reaction'):
                torch.load(os.path.join(edge_dir, 'substrate_of.pt'), weights_only=True),
            ('reaction', 'produces', 'metabolite'):
                torch.load(os.path.join(edge_dir, 'produces.pt'), weights_only=True),
        }
        shared_path = os.path.join(edge_dir, 'shared_metabolite.pt')
        if os.path.exists(shared_path):
            shared_ei = torch.load(shared_path, weights_only=True)
            # Sparsify dense R-R edges to fit in 32 GB VRAM.
            # GATv2Conv on 7.5M edges x 3 layers x hidden_dim causes OOM.
            # Keep only top-K random neighbours per source reaction.
            MAX_K = 10
            n_nodes = shared_ei.max().item() + 1
            if shared_ei.shape[1] > MAX_K * n_nodes:
                n_orig = shared_ei.shape[1]
                src = shared_ei[0]
                # Vectorised top-k: shuffle, sort by src (stable), compute
                # within-group rank, keep rank < MAX_K
                perm = torch.randperm(n_orig)
                sort_idx = src[perm].argsort(stable=True)
                src_sorted = src[perm][sort_idx]
                _, counts = torch.unique_consecutive(src_sorted, return_counts=True)
                offsets = counts.cumsum(0)
                starts = torch.zeros_like(offsets)
                starts[1:] = offsets[:-1]
                group_id = torch.repeat_interleave(torch.arange(len(counts)), counts)
                rank = torch.arange(n_orig) - starts[group_id]
                kept_indices = perm[sort_idx[rank < MAX_K]]
                shared_ei = shared_ei[:, kept_indices]
                logger.info(f"  Sparsified shared_metabolite: {n_orig:,} -> "
                            f"{shared_ei.shape[1]:,} edges (top-{MAX_K}/node)")
            self.edge_index_dict[('reaction', 'shared_metabolite', 'reaction')] = shared_ei

        self.y_r = torch.load(
            os.path.join(self.data_root, 'activity_pseudolabels.pt'), weights_only=True)

        self.feat_dir = os.path.join(self.data_root, 'reaction_features')
        self.feat_ext = '.h5'

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        if self.feat_ext == '.pt':
            X_R = torch.load(
                os.path.join(self.feat_dir, f'{pid}.pt'), weights_only=True)
            if not isinstance(X_R, torch.Tensor):
                X_R = torch.tensor(X_R, dtype=torch.float32)
            X_R = X_R.float()
        else:
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


class RNAOnlyWrapper(torch.utils.data.Dataset):
    """Wraps UnifiedDataset but zeros out proteomics column."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.patient_ids = base_dataset.patient_ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        data['reaction'].x = data['reaction'].x.clone()
        data['reaction'].x[:, 1] = 0.0
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT ID LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_patient_ids(data_root, data_format):
    """Load all available patient IDs and stratification info."""
    if data_format == 'metagnn_crc':
        # Try metadata.csv first, then clinical_metadata.tsv
        meta_path = os.path.join(data_root, 'metadata.csv')
        if os.path.exists(meta_path):
            meta_df = pd.read_csv(meta_path)
            # Identify barcode column
            for col in ['tcga_barcode', 'patient_id', 'sample_id']:
                if col in meta_df.columns:
                    id_col = col
                    break
            else:
                id_col = meta_df.columns[0]
            patient_ids = meta_df[id_col].tolist()
        else:
            # Fall back to listing .pt files
            feat_dir = os.path.join(data_root, 'patient_features')
            patient_ids = [f.replace('.pt', '') for f in os.listdir(feat_dir)
                          if f.endswith('.pt')]
        # Try to get stratification
        strat = None
        if os.path.exists(meta_path):
            for col in ['msi_status', 'tumor_stage', 'stage']:
                if col in meta_df.columns:
                    strat = meta_df[col].fillna('Unknown').tolist()
                    break
    else:
        meta_path = os.path.join(data_root, 'clinical_metadata.tsv')
        meta_df = pd.read_csv(meta_path, sep='\t')
        patient_ids = meta_df['tcga_barcode'].tolist()
        strat = meta_df['msi_status'].fillna('Unknown').tolist()

    if strat is None:
        strat = ['A'] * len(patient_ids)  # dummy stratification

    logger.info(f"Loaded {len(patient_ids)} patient IDs ({data_format} format)")
    return patient_ids, strat


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS & TRAINING
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
    return {'F1': f1, 'AUROC': auroc, 'AUPRC': auprc,
            'Precision': prec, 'Recall': rec}


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
        result[f'{key}_mean'] = float(np.mean(vals))
        result[f'{key}_std']  = float(np.std(vals))
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


def train_and_evaluate(config, data_root, data_format, train_ids, val_ids,
                       test_ids, device, stoich_matrix, seed, edge_types=None):
    """Train one configuration and return test metrics."""
    cfg = {**TRAIN_CFG, **config}
    label = cfg.pop('label', 'config')
    batch_size = cfg.pop('batch_size', 2)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    # Build datasets
    train_ds = UnifiedDataset(data_root, train_ids, data_format)
    val_ds   = UnifiedDataset(data_root, val_ids, data_format)
    test_ds  = UnifiedDataset(data_root, test_ids, data_format)

    # Infer feature dims from first sample
    sample = train_ds[0]
    rxn_in_dim = sample['reaction'].x.shape[1]
    met_in_dim = sample['metabolite'].x.shape[1]
    if edge_types is None:
        edge_types = list(train_ds.edge_index_dict.keys())

    train_loader = PyGLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = PyGLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MetaGNN(
        rxn_in_dim=rxn_in_dim, met_in_dim=met_in_dim,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg['dropout'],
        edge_types=edge_types,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = MetaGNNLoss(
        stoich_matrix=stoich_matrix.to(device) if stoich_matrix is not None else None,
        lambda_mb=cfg['lambda_mb'],
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=cfg['n_epochs'])

    best_val_f1, patience_counter, best_state, best_epoch = 0.0, 0, None, 0
    start_time = time.time()

    for epoch in range(1, cfg['n_epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)
        val_metrics = evaluate_batch(model, val_loader, criterion,
                                     device, cfg['threshold'])
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"    Epoch {epoch:3d}  loss={train_loss:.4f}  "
                f"val_F1={val_metrics['F1']:.4f}  val_AUROC={val_metrics['AUROC']:.4f}")

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                logger.info(f"    Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time
    peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if device.type == 'cuda' else 0)

    model.load_state_dict(best_state)
    test_results = evaluate_per_patient(model, test_ds, device, cfg['threshold'])

    logger.info(f"    Test F1={test_results['F1_mean']:.4f}±{test_results['F1_std']:.4f}  "
                f"AUROC={test_results['AUROC_mean']:.4f}  Time={train_time:.0f}s")

    test_results.update({
        'label': label, 'seed': seed, 'n_params': n_params,
        'train_time_s': round(train_time, 1), 'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'peak_gpu_mb': round(peak_gpu_mb, 0),
        'hidden_dim': cfg['hidden_dim'], 'n_layers': cfg['n_layers'],
        'heads': cfg['heads'], 'batch_size': batch_size,
        'n_train': len(train_ids), 'n_val': len(val_ids),
        'n_test': len(test_ids),
    })

    # Also return model + test_ds for FBA phase
    return test_results, model, test_ds


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: 5-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def run_kfold_cv(data_root, data_format, patient_ids, strat, device,
                 stoich_matrix, output_dir, n_folds=5, seed=2024):
    """5-fold stratified CV using Config B (paper architecture)."""
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1: 5-Fold Stratified Cross-Validation")
    logger.info(f"  Config B (256/3/8) | {len(patient_ids)} patients | {n_folds} folds")
    logger.info("=" * 70)

    config = dict(CONFIGS['B_paper'])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    ids_arr = np.array(patient_ids)
    strat_arr = np.array(strat)

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(ids_arr, strat_arr)):
        logger.info(f"\n  -- Fold {fold_idx + 1}/{n_folds} --")
        test_ids = ids_arr[test_idx].tolist()

        # Split train_val into train (85%) and val (15%)
        tv_ids = ids_arr[train_val_idx]
        tv_strat = strat_arr[train_val_idx]
        train_ids, val_ids = train_test_split(
            tv_ids.tolist(), train_size=0.85,
            stratify=tv_strat.tolist(), random_state=seed,
        )

        logger.info(f"    Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

        results, _, _ = train_and_evaluate(
            dict(config), data_root, data_format,
            train_ids, val_ids, test_ids,
            device, stoich_matrix, seed=seed,
        )
        results['fold'] = fold_idx + 1
        fold_results.append(results)

    # Aggregate
    summary = {'n_folds': n_folds, 'n_patients': len(patient_ids),
               'config': 'B_paper (256/3/8)', 'seed': seed}
    for metric in ['F1_mean', 'AUROC_mean', 'AUPRC_mean', 'Precision_mean', 'Recall_mean']:
        vals = [r[metric] for r in fold_results]
        summary[metric] = round(float(np.mean(vals)), 4)
        summary[metric.replace('_mean', '_std')] = round(float(np.std(vals)), 4)
        summary[metric + '_95ci'] = [
            round(float(np.mean(vals) - 1.96 * np.std(vals) / np.sqrt(n_folds)), 4),
            round(float(np.mean(vals) + 1.96 * np.std(vals) / np.sqrt(n_folds)), 4),
        ]

    summary['per_fold'] = fold_results

    logger.info(f"\n  -- K-Fold CV Summary ({len(patient_ids)} patients) --")
    logger.info(f"    F1:    {summary['F1_mean']:.4f} ± {summary['F1_std']:.4f}")
    logger.info(f"    AUROC: {summary['AUROC_mean']:.4f} ± {summary['AUROC_std']:.4f}")
    logger.info(f"    AUPRC: {summary['AUPRC_mean']:.4f} ± {summary['AUPRC_std']:.4f}")

    out_path = os.path.join(output_dir, 'kfold_cv_full_cohort.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"    Saved to: {out_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: ARCHITECTURE SCALING
# ═══════════════════════════════════════════════════════════════════════════════
def run_scaling(data_root, data_format, patient_ids, strat, device,
                stoich_matrix, output_dir, seed=2024):
    """Architecture scaling across 4 configs with 3 seeds."""
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 2: Architecture Scaling on Real Data")
    logger.info(f"  {len(patient_ids)} patients | 4 configs × 3 seeds × 2 conditions = 24 runs")
    logger.info("=" * 70)

    # 70/15/15 split
    train_ids, tmp_ids, _, tmp_strat = train_test_split(
        patient_ids, strat, train_size=0.70, stratify=strat, random_state=2024)
    val_frac_of_tmp = 0.15 / 0.30
    val_ids, test_ids = train_test_split(
        tmp_ids, train_size=val_frac_of_tmp, stratify=tmp_strat, random_state=2024)

    logger.info(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    all_results = []
    best_model_for_fba = None
    best_test_ds_for_fba = None

    for config_name, config in CONFIGS.items():
        for s in SEEDS:
            logger.info(f"\n  -- {config['label']} | seed={s} | Full --")
            result, model, test_ds = train_and_evaluate(
                dict(config), data_root, data_format,
                train_ids, val_ids, test_ids,
                device, stoich_matrix, seed=s,
            )
            result['config'] = config_name
            result['dataset'] = 'MetaGNNDataset'
            all_results.append(result)

            # Keep best model (B_paper, seed=2024) for FBA
            if config_name == 'B_paper' and s == 2024:
                best_model_for_fba = model
                best_test_ds_for_fba = test_ds

            # RNA-only ablation
            logger.info(f"  -- {config['label']} | seed={s} | RNA-only --")
            rna_train_ds = RNAOnlyWrapper(
                UnifiedDataset(data_root, train_ids, data_format))
            rna_val_ds = RNAOnlyWrapper(
                UnifiedDataset(data_root, val_ids, data_format))
            rna_test_ds = RNAOnlyWrapper(
                UnifiedDataset(data_root, test_ids, data_format))

            # Re-use train_and_evaluate but we need to handle RNA-only differently
            # Simplest: create a temp wrapper that uses RNAOnlyWrapper datasets
            rna_result = _train_rna_only(
                dict(config), rna_train_ds, rna_val_ds, rna_test_ds,
                device, stoich_matrix, seed=s,
            )
            rna_result['config'] = config_name + '_rna_only'
            rna_result['dataset'] = 'RNAOnlyDataset'
            rna_result['label'] = config['label'] + ' (RNA-only)'
            all_results.append(rna_result)

            # Save incrementally
            inc_path = os.path.join(output_dir, 'scaling_full_cohort_incremental.json')
            with open(inc_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Aggregate summary
    summary = _aggregate_scaling(all_results)
    output = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else 'CPU'),
            'n_patients': len(patient_ids),
            'n_train': len(train_ids),
            'n_val': len(val_ids),
            'n_test': len(test_ids),
            'seeds': SEEDS,
        },
        'summary': summary,
        'all_runs': all_results,
    }

    out_path = os.path.join(output_dir, 'scaling_full_cohort.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Scaling results saved to: {out_path}")

    return output, best_model_for_fba, test_ids


def _train_rna_only(config, train_ds, val_ds, test_ds, device,
                    stoich_matrix, seed):
    """Train on RNA-only wrapped datasets."""
    cfg = {**TRAIN_CFG, **config}
    label = cfg.pop('label', 'config')
    batch_size = cfg.pop('batch_size', 2)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    sample = train_ds[0]
    rxn_in_dim = sample['reaction'].x.shape[1]
    met_in_dim = sample['metabolite'].x.shape[1]
    edge_types = list(train_ds.base.edge_index_dict.keys())

    train_loader = PyGLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = PyGLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MetaGNN(
        rxn_in_dim=rxn_in_dim, met_in_dim=met_in_dim,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg['dropout'],
        edge_types=edge_types,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = MetaGNNLoss(
        stoich_matrix=stoich_matrix.to(device) if stoich_matrix is not None else None,
        lambda_mb=cfg['lambda_mb'],
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=cfg['n_epochs'])

    best_val_f1, patience_counter, best_state, best_epoch = 0.0, 0, None, 0
    start_time = time.time()

    for epoch in range(1, cfg['n_epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)
        val_metrics = evaluate_batch(model, val_loader, criterion,
                                     device, cfg['threshold'])
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"    Epoch {epoch:3d}  loss={train_loss:.4f}  "
                f"val_F1={val_metrics['F1']:.4f}")

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                logger.info(f"    Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time
    peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if device.type == 'cuda' else 0)

    model.load_state_dict(best_state)
    test_results = evaluate_per_patient(model, test_ds, device, cfg['threshold'])

    logger.info(f"    Test F1={test_results['F1_mean']:.4f}±{test_results['F1_std']:.4f}  "
                f"AUROC={test_results['AUROC_mean']:.4f}")

    test_results.update({
        'seed': seed, 'n_params': n_params,
        'train_time_s': round(train_time, 1), 'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'peak_gpu_mb': round(peak_gpu_mb, 0),
        'hidden_dim': cfg['hidden_dim'], 'n_layers': cfg['n_layers'],
        'heads': cfg['heads'], 'batch_size': batch_size,
    })
    return test_results


def _aggregate_scaling(all_results):
    """Aggregate per-seed results into summary."""
    summary = {}
    config_names = sorted(set(r['config'] for r in all_results))
    for cname in config_names:
        runs = [r for r in all_results if r['config'] == cname]
        if not runs:
            continue
        agg = {'label': runs[0].get('label', cname), 'n_params': runs[0]['n_params']}
        for metric in ['F1_mean', 'AUROC_mean', 'AUPRC_mean',
                       'Precision_mean', 'Recall_mean']:
            vals = [r[metric] for r in runs if metric in r]
            if vals:
                agg[metric] = round(float(np.mean(vals)), 4)
                agg[metric.replace('_mean', '_std_seeds')] = round(float(np.std(vals)), 4)
        agg['avg_train_time_s'] = round(
            float(np.mean([r['train_time_s'] for r in runs])), 1)
        agg['avg_peak_gpu_mb'] = round(
            float(np.mean([r.get('peak_gpu_mb', 0) for r in runs])), 0)
        agg['per_seed'] = runs
        summary[cname] = agg

        logger.info(f"  {agg['label']}: F1={agg.get('F1_mean', 0):.4f}±"
                    f"{agg.get('F1_std_seeds', 0):.4f}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: FBA VIABILITY (CPU)
# ═══════════════════════════════════════════════════════════════════════════════
def run_fba_viability(data_root, data_format, patient_ids, recon3d_xml,
                      device, stoich_matrix, output_dir,
                      model=None, seed=2024):
    """FBA viability test on all patients using trained model."""
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 3: FBA Viability Analysis")
    logger.info(f"  {len(patient_ids)} patients | Recon3D: {recon3d_xml}")
    logger.info("=" * 70)

    try:
        import cobra
        from cobra.io import read_sbml_model
    except ImportError:
        logger.error("COBRApy not installed. Run: pip install cobra")
        logger.error("Skipping FBA phase.")
        return None

    # Load Recon3D COBRA model
    logger.info("  Loading Recon3D SBML model...")
    recon3d = read_sbml_model(recon3d_xml)
    logger.info(f"  Loaded: {len(recon3d.reactions)} reactions, "
                f"{len(recon3d.metabolites)} metabolites")

    # If no pre-trained model, train one (Config B)
    if model is None:
        logger.info("  Training Config B for FBA predictions...")
        strat = ['A'] * len(patient_ids)  # dummy
        train_ids, tmp_ids, _, tmp_strat = train_test_split(
            patient_ids, strat, train_size=0.70, stratify=strat,
            random_state=seed)
        val_ids, test_ids = train_test_split(
            tmp_ids, train_size=0.5, stratify=tmp_strat, random_state=seed)
        config = dict(CONFIGS['B_paper'])
        _, model, _ = train_and_evaluate(
            config, data_root, data_format,
            train_ids, val_ids, test_ids,
            device, stoich_matrix, seed=seed,
        )

    # Generate predictions for ALL patients
    logger.info(f"  Generating predictions for all {len(patient_ids)} patients...")
    all_ds = UnifiedDataset(data_root, patient_ids, data_format)
    model.eval()
    cpu_device = torch.device('cpu')
    model_cpu = model.to(cpu_device)

    patient_predictions = {}
    with torch.no_grad():
        for i in range(len(all_ds)):
            data = all_ds[i].to(cpu_device)
            s_r = model_cpu(
                x_dict={'reaction': data['reaction'].x,
                        'metabolite': data['metabolite'].x},
                edge_index_dict={rel: data[rel].edge_index
                                for rel in data.edge_types},
            )
            pid = all_ds.patient_ids[i]
            patient_predictions[pid] = s_r.numpy()
            if (i + 1) % 100 == 0:
                logger.info(f"    Predicted {i+1}/{len(all_ds)} patients")

    # FBA viability test
    threshold = 0.15

    def test_fba_viability(cobra_model, active_scores, threshold=0.15):
        with cobra_model as m:
            inactive_count = 0
            for i, rxn in enumerate(m.reactions):
                if i < len(active_scores) and active_scores[i] < threshold:
                    rxn.bounds = (0, 0)
                    inactive_count += 1
            try:
                sol = m.optimize()
                if sol.status == 'optimal':
                    return sol.objective_value, inactive_count
                return 0.0, inactive_count
            except Exception:
                return 0.0, inactive_count

    logger.info(f"\n  Running FBA for {len(patient_predictions)} patients (τ={threshold})...")
    patient_results = []
    for idx, (pid, scores) in enumerate(patient_predictions.items()):
        t0 = time.time()
        biomass, n_inactive = test_fba_viability(recon3d, scores, threshold)
        viable = biomass > 1e-6
        dt = time.time() - t0

        result = {
            'patient_id': pid, 'biomass_flux': float(biomass),
            'viable': viable, 'n_inactive': int(n_inactive),
            'n_active': int(len(scores) - n_inactive),
            'fba_time_s': round(dt, 2),
        }
        patient_results.append(result)
        status = '✓' if viable else '✗'
        if (idx + 1) % 50 == 0 or not viable:
            logger.info(f"    {status} [{idx+1}/{len(patient_predictions)}] "
                       f"{pid}: biomass={biomass:.4f}, time={dt:.1f}s")

    # Random baseline (50 MC trials)
    logger.info("\n  Running random baseline (50 trials)...")
    label_vec = all_ds.y_r.numpy() if hasattr(all_ds, 'y_r') else np.ones(len(scores))
    active_ratio = float((label_vec > 0.5).mean())
    random_viable = 0
    n_trials = 50
    for t in range(n_trials):
        rand_scores = np.random.random(len(label_vec))
        rand_mask = (rand_scores >= (1 - active_ratio))
        biomass, _ = test_fba_viability(
            recon3d, 1.0 - rand_mask.astype(float), threshold=0.5)
        if biomass > 1e-6:
            random_viable += 1
        if (t + 1) % 25 == 0:
            logger.info(f"    Trial {t+1}/{n_trials}: {random_viable}/{t+1} viable")

    # Summary
    n_viable = sum(1 for r in patient_results if r['viable'])
    biomass_vals = [r['biomass_flux'] for r in patient_results if r['viable']]

    summary = {
        'n_patients': len(patient_results),
        'n_viable': n_viable,
        'viability_rate': round(n_viable / len(patient_results), 4),
        'biomass_mean': round(float(np.mean(biomass_vals)), 6) if biomass_vals else 0,
        'biomass_std': round(float(np.std(biomass_vals)), 6) if biomass_vals else 0,
        'threshold': threshold,
        'random_baseline_viability': round(random_viable / n_trials, 4),
        'per_patient': patient_results,
    }

    out_path = os.path.join(output_dir, 'fba_full_cohort.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  FBA Summary: {n_viable}/{len(patient_results)} viable "
                f"({summary['viability_rate']*100:.1f}%) vs "
                f"random {summary['random_baseline_viability']*100:.1f}%")
    logger.info(f"  Saved to: {out_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='MetaGNN Full-Cohort Experiments (Overnight Run)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to data directory (MetaGNN-CRC or MethodsX)')
    parser.add_argument('--output_dir', type=str, default='./results_full_cohort',
                        help='Directory for output JSON results')
    parser.add_argument('--recon3d_xml', type=str, default=None,
                        help='Path to Recon3D.xml for FBA (Phase 3)')
    parser.add_argument('--skip_kfold', action='store_true',
                        help='Skip Phase 1 (5-fold CV)')
    parser.add_argument('--skip_scaling', action='store_true',
                        help='Skip Phase 2 (Architecture scaling)')
    parser.add_argument('--skip_fba', action='store_true',
                        help='Skip Phase 3 (FBA viability)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -- Device ----------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.warning("No CUDA GPU — running on CPU")

    # -- Detect data format and load patient IDs ------------------------------
    data_format = detect_data_format(args.data_root)
    patient_ids, strat = load_patient_ids(args.data_root, data_format)

    # -- Load stoichiometric matrix (if available) ----------------------------
    stoich_path = os.path.join(args.data_root, 'recon3d_stoich.h5')
    if os.path.exists(stoich_path):
        with h5py.File(stoich_path, 'r') as f:
            stoich_matrix = torch.tensor(f['S'][:], dtype=torch.float32)
        logger.info(f"Stoichiometric matrix: {stoich_matrix.shape}")
    else:
        logger.warning("No recon3d_stoich.h5 found — mass-balance loss disabled")
        stoich_matrix = None

    # -- Run phases ------------------------------------------------------------
    start_all = time.time()
    model_for_fba = None

    if not args.skip_kfold:
        run_kfold_cv(args.data_root, data_format, patient_ids, strat,
                     device, stoich_matrix, args.output_dir)

    if not args.skip_scaling:
        scaling_out, model_for_fba, _ = run_scaling(
            args.data_root, data_format, patient_ids, strat,
            device, stoich_matrix, args.output_dir)

    if not args.skip_fba:
        if args.recon3d_xml and os.path.exists(args.recon3d_xml):
            run_fba_viability(
                args.data_root, data_format, patient_ids, args.recon3d_xml,
                device, stoich_matrix, args.output_dir,
                model=model_for_fba)
        else:
            logger.warning("Skipping FBA: --recon3d_xml not provided or file not found")
            logger.info("  To run FBA, download Recon3D.xml from https://www.vmh.life")
            logger.info("  Then re-run with: --recon3d_xml /path/to/Recon3D.xml")

    total_time = time.time() - start_all
    logger.info(f"\n{'#' * 70}")
    logger.info(f"  ALL PHASES COMPLETE — Total time: {total_time/3600:.1f} hours")
    logger.info(f"  Results in: {args.output_dir}/")
    logger.info(f"{'#' * 70}")


if __name__ == '__main__':
    main()
