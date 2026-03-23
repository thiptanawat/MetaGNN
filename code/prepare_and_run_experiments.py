#!/usr/bin/env python3
"""
MetaGNN: Complete Data Preparation & Experiment Runner
======================================================
Generates a realistic synthetic dataset matching Recon3D v3 dimensions,
then runs all three reviewer-requested experiments:
  Exp 1: Graph-structure ablation (real vs rewired vs disconnected)
  Exp 2: Baseline model comparisons (MLP, HomoGAT, HomoGCN)
  Exp 3: 5-fold stratified cross-validation

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

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import HeteroConv, GATv2Conv, GCNConv, Linear
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration matching the paper
# ═══════════════════════════════════════════════════════════════════════════════
RECON3D = dict(
    n_reactions=10600,         # Recon3D v3: 10,600 reactions
    n_metabolites=5835,        # Recon3D v3: 5,835 metabolites
    n_genes=2248,
    n_stoich_edges=40425,
    active_ratio=0.701,        # 70.1% active reactions
)

DEFAULTS = dict(
    hidden_dim=128,            # 128 (paper uses 256; 128 for faster CPU training)
    n_layers=2,                # 2 layers (paper uses 3; 2 for CPU speed)
    heads=4,                   # 4 heads (paper uses 8; 4 for CPU speed)
    dropout=0.20,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=4,              # small batch for memory
    n_epochs=80,
    patience=15,
    threshold=0.15,
    seed=2024,
    lambda_mb=0.2,
    rxn_in_dim=2,
    met_in_dim=519,            # 7 physico-chem + 512 Morgan FP
)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
class MCDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


class HGATLayer(nn.Module):
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


class ReactionMLP(nn.Module):
    """MLP baseline — no graph structure."""
    def __init__(self, in_dim=2, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )
    def forward(self, x_dict, edge_index_dict):
        return self.net(x_dict['reaction']).squeeze(-1)


class HomoGAT(nn.Module):
    """Homogeneous GAT — collapses node types."""
    def __init__(self, in_dim=519, hidden_dim=128, heads=4, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())
        self.n_rxn = 0

    def forward(self, x_dict, edge_index_dict):
        n_rxn = x_dict['reaction'].shape[0]
        n_met = x_dict['metabolite'].shape[0]
        self.n_rxn = n_rxn
        # Pad reaction features to match metabolite dim
        rxn_padded = F.pad(x_dict['reaction'], (0, 519 - x_dict['reaction'].shape[1]))
        x_all = torch.cat([rxn_padded, x_dict['metabolite']], dim=0)
        x_all = self.proj(x_all)
        # Merge all edges into homogeneous
        edges = []
        for (src_type, _, dst_type), ei in edge_index_dict.items():
            src_offset = 0 if src_type == 'reaction' else n_rxn
            dst_offset = 0 if dst_type == 'reaction' else n_rxn
            edges.append(torch.stack([ei[0] + src_offset, ei[1] + dst_offset]))
        edge_index = torch.cat(edges, dim=1)
        h = self.norm1(F.elu(self.gat1(x_all, edge_index)) + x_all)
        h = self.norm2(F.elu(self.gat2(h, edge_index)) + h)
        return self.head(h[:n_rxn]).squeeze(-1)


class HomoGCN(nn.Module):
    """Homogeneous GCN — no attention."""
    def __init__(self, in_dim=519, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())

    def forward(self, x_dict, edge_index_dict):
        n_rxn = x_dict['reaction'].shape[0]
        n_met = x_dict['metabolite'].shape[0]
        rxn_padded = F.pad(x_dict['reaction'], (0, 519 - x_dict['reaction'].shape[1]))
        x_all = torch.cat([rxn_padded, x_dict['metabolite']], dim=0)
        x_all = self.proj(x_all)
        edges = []
        for (src_type, _, dst_type), ei in edge_index_dict.items():
            src_offset = 0 if src_type == 'reaction' else n_rxn
            dst_offset = 0 if dst_type == 'reaction' else n_rxn
            edges.append(torch.stack([ei[0] + src_offset, ei[1] + dst_offset]))
        edge_index = torch.cat(edges, dim=1)
        h = self.norm1(F.elu(self.gcn1(x_all, edge_index)) + x_all)
        h = self.drop(h)
        h = self.norm2(F.elu(self.gcn2(h, edge_index)) + h)
        return self.head(h[:n_rxn]).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS & METRICS
# ═══════════════════════════════════════════════════════════════════════════════
class MetaGNNLoss(nn.Module):
    def __init__(self, stoich_matrix, lambda_mb=0.2):
        super().__init__()
        self.register_buffer('S', stoich_matrix)
        self.lambda_mb = lambda_mb
        self.bce = nn.BCELoss()

    def forward(self, s_r, y_r):
        l_bce = self.bce(s_r, y_r.float())
        # Handle batched predictions: reshape to (batch, n_rxn), compute per-patient
        n_rxn = self.S.shape[1]
        if s_r.shape[0] > n_rxn and s_r.shape[0] % n_rxn == 0:
            s_r_2d = s_r.view(-1, n_rxn)  # (batch, n_rxn)
            # Memory-efficient: compute per patient then average
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


def compute_metrics(y_pred, y_true, threshold=0.15):
    y_bin = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_pred)
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(rec, prec)
    else:
        auroc = auprc = float('nan')
    return {'F1': f1, 'AUROC': auroc, 'AUPRC': auprc}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION — Realistic synthetic Recon3D-like metabolic network
# ═══════════════════════════════════════════════════════════════════════════════
def generate_recon3d_like_data(data_root, n_patients=220, seed=2024):
    """
    Generate a realistic synthetic dataset matching Recon3D v3 dimensions.
    - Stoichiometric matrix with biologically plausible sparsity
    - 519-dim metabolite features (mimicking physico-chem + Morgan FP)
    - 2-dim reaction features per patient (GPR-mapped expression)
    - Pseudo-labels with 70.1% active rate
    - Clinical metadata with MSI status stratification
    """
    logger.info("Generating synthetic Recon3D-like dataset...")
    rng = np.random.RandomState(seed)
    os.makedirs(data_root, exist_ok=True)

    n_rxn = RECON3D['n_reactions']
    n_met = RECON3D['n_metabolites']

    # ── Stoichiometric matrix S (n_met × n_rxn) ──────────────────────────────
    # Each reaction has 2-6 substrates and 2-6 products (biologically realistic)
    logger.info(f"  Building S matrix ({n_met} × {n_rxn})...")
    S = np.zeros((n_met, n_rxn), dtype=np.float32)
    for r in range(n_rxn):
        n_sub = rng.randint(2, 7)
        n_prod = rng.randint(2, 7)
        subs = rng.choice(n_met, n_sub, replace=False)
        prods = rng.choice(n_met, n_prod, replace=False)
        S[subs, r] = -rng.uniform(0.5, 2.0, size=n_sub)
        S[prods, r] = rng.uniform(0.5, 2.0, size=n_prod)

    n_edges_actual = int(np.count_nonzero(S))
    logger.info(f"  Stoichiometric edges: {n_edges_actual} (target ~{RECON3D['n_stoich_edges']})")

    # Save S matrix
    with h5py.File(os.path.join(data_root, 'recon3d_stoich.h5'), 'w') as f:
        f.create_dataset('S', data=S, compression='gzip')

    # ── Edge indices from S ───────────────────────────────────────────────────
    logger.info("  Computing edge indices...")
    os.makedirs(os.path.join(data_root, 'edge_indices'), exist_ok=True)

    sub_rows, sub_cols = np.where(S < 0)
    ei_substrate = torch.tensor(np.stack([sub_rows, sub_cols]), dtype=torch.long)
    torch.save(ei_substrate, os.path.join(data_root, 'edge_indices', 'substrate_of.pt'))

    prod_met, prod_rxn = np.where(S > 0)
    ei_produces = torch.tensor(np.stack([prod_rxn, prod_met]), dtype=torch.long)
    torch.save(ei_produces, os.path.join(data_root, 'edge_indices', 'produces.pt'))

    # Shared-metabolite edges (reaction-reaction)
    P = (S != 0).astype(np.float32)
    shared = P.T @ P
    np.fill_diagonal(shared, 0)
    # Sparsify: keep only top connections to manage memory
    shared[shared < 2] = 0  # require ≥2 shared metabolites
    r1s, r2s = np.where(shared > 0)
    ei_shared = torch.tensor(np.stack([r1s, r2s]), dtype=torch.long)
    torch.save(ei_shared, os.path.join(data_root, 'edge_indices', 'shared_metabolite.pt'))
    logger.info(f"  Edge counts: substrate={ei_substrate.shape[1]}, "
                f"produces={ei_produces.shape[1]}, shared={ei_shared.shape[1]}")

    # ── Metabolite features (519-dim) ─────────────────────────────────────────
    logger.info("  Generating metabolite features (519-dim)...")
    # 7 physico-chemical: MW, logP, HBA, HBD, TPSA, rings, charge
    physico = np.column_stack([
        rng.lognormal(5.5, 0.8, n_met),      # MW (~250-1000)
        rng.normal(1.5, 2.0, n_met),          # logP
        rng.poisson(3, n_met).astype(float),   # HBA
        rng.poisson(2, n_met).astype(float),   # HBD
        rng.lognormal(3.5, 0.8, n_met),       # TPSA
        rng.poisson(2, n_met).astype(float),   # rings
        rng.choice([-2,-1,0,0,0,0,1,2], n_met).astype(float),  # charge
    ])
    # 512 Morgan fingerprints (sparse binary, ~5-15% density)
    morgan = rng.binomial(1, 0.10, (n_met, 512)).astype(np.float32)
    X_M = np.hstack([physico.astype(np.float32), morgan])
    with h5py.File(os.path.join(data_root, 'metabolite_features.h5'), 'w') as f:
        f.create_dataset('X_M', data=X_M, compression='gzip')

    # ── Pseudo-labels (shared across cohort, 70.1% active) ───────────────────
    active_mask = rng.random(n_rxn) < RECON3D['active_ratio']
    y_r = torch.tensor(active_mask.astype(np.float32))
    torch.save(y_r, os.path.join(data_root, 'activity_pseudolabels.pt'))
    n_active = int(active_mask.sum())
    logger.info(f"  Labels: {n_active} active / {n_rxn - n_active} inactive "
                f"({100*n_active/n_rxn:.1f}%)")

    # ── Clinical metadata ─────────────────────────────────────────────────────
    patient_ids = [f"TCGA-{rng.choice(list('ABCDEF'))}{rng.choice(list('ABCDEF'))}-"
                   f"{rng.randint(1000,9999)}-01A" for _ in range(n_patients)]
    # MSI status distribution: ~15% MSI-H, ~85% MSS (matches CRC)
    msi_status = rng.choice(['MSI-H', 'MSS'], n_patients, p=[0.15, 0.85])
    meta_df = pd.DataFrame({
        'tcga_barcode': patient_ids,
        'msi_status': msi_status,
        'tumor_stage': rng.choice(['I', 'II', 'III', 'IV'], n_patients, p=[0.15, 0.35, 0.35, 0.15]),
    })
    meta_df.to_csv(os.path.join(data_root, 'clinical_metadata.tsv'), sep='\t', index=False)

    # ── Per-patient reaction features (2-dim: RNA GPR + protein GPR) ──────────
    logger.info(f"  Generating reaction features for {n_patients} patients...")
    os.makedirs(os.path.join(data_root, 'reaction_features'), exist_ok=True)

    # Create correlated features: active reactions have higher expression
    base_rna = rng.lognormal(2.0, 1.0, n_rxn).astype(np.float32)
    base_prot = rng.lognormal(1.5, 1.2, n_rxn).astype(np.float32)
    # Active reactions get boosted expression (creates learnable signal)
    base_rna[active_mask] *= 2.5
    base_prot[active_mask] *= 2.0

    for pid in patient_ids:
        # Per-patient noise
        noise_rna = rng.lognormal(0, 0.3, n_rxn).astype(np.float32)
        noise_prot = rng.lognormal(0, 0.4, n_rxn).astype(np.float32)
        X_R = np.column_stack([
            np.log2(base_rna * noise_rna + 1),
            np.log2(base_prot * noise_prot + 1),
        ])
        with h5py.File(os.path.join(data_root, 'reaction_features', f'{pid}.h5'), 'w') as f:
            f.create_dataset('X_R', data=X_R)

    logger.info("  Data generation complete!")
    return meta_df, patient_ids


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class MetaGNNDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, patient_ids=None):
        self.data_root = data_root
        self.meta_df = pd.read_csv(os.path.join(data_root, 'clinical_metadata.tsv'), sep='\t')
        if patient_ids is not None:
            self.meta_df = self.meta_df[self.meta_df['tcga_barcode'].isin(patient_ids)].reset_index(drop=True)
        self.patient_ids = self.meta_df['tcga_barcode'].tolist()

        with h5py.File(os.path.join(data_root, 'metabolite_features.h5'), 'r') as f:
            self.X_M = torch.tensor(f['X_M'][:], dtype=torch.float32)

        self.edge_index_dict = {
            ('metabolite', 'substrate_of', 'reaction'):
                torch.load(os.path.join(data_root, 'edge_indices', 'substrate_of.pt'), weights_only=True),
            ('reaction', 'produces', 'metabolite'):
                torch.load(os.path.join(data_root, 'edge_indices', 'produces.pt'), weights_only=True),
            ('reaction', 'shared_metabolite', 'reaction'):
                torch.load(os.path.join(data_root, 'edge_indices', 'shared_metabolite.pt'), weights_only=True),
        }
        self.y_r = torch.load(os.path.join(data_root, 'activity_pseudolabels.pt'), weights_only=True)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        with h5py.File(os.path.join(self.data_root, 'reaction_features', f'{pid}.h5'), 'r') as f:
            X_R = torch.tensor(f['X_R'][:], dtype=torch.float32)
        data = HeteroData()
        data['reaction'].x = X_R
        data['metabolite'].x = self.X_M
        data['reaction'].y = self.y_r
        for rel, ei in self.edge_index_dict.items():
            src_type, rel_type, dst_type = rel
            data[src_type, rel_type, dst_type].edge_index = ei
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
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
    return metrics, y_pred, y_true


def train_model(model, train_loader, val_loader, cfg, criterion, device, label=""):
    """Full training loop with early stopping. Returns best test metrics."""
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'])
    best_val_f1, patience_ctr = 0.0, 0
    best_state = None

    for epoch in range(1, cfg['n_epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m, _, _ = evaluate_model(model, val_loader, criterion, device, cfg['threshold'])
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"  [{label}] Ep {epoch:3d}  loss={train_loss:.4f}  "
                        f"val_F1={val_m['F1']:.4f}  val_AUROC={val_m['AUROC']:.4f}")

        if val_m['F1'] > best_val_f1:
            best_val_f1 = val_m['F1']
            patience_ctr = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= cfg['patience']:
                logger.info(f"  [{label}] Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH REWIRING (for Experiment 1)
# ═══════════════════════════════════════════════════════════════════════════════
def degree_preserving_rewire(edge_index, n_swaps=None, seed=42):
    """Maslov-Sneppen degree-preserving rewiring."""
    rng = np.random.RandomState(seed)
    edges = edge_index.t().numpy().copy()
    n_edges = len(edges)
    if n_swaps is None:
        n_swaps = n_edges * 10

    for _ in range(n_swaps):
        i, j = rng.randint(0, n_edges, 2)
        if i == j:
            continue
        # Swap targets
        new_e1 = [edges[i][0], edges[j][1]]
        new_e2 = [edges[j][0], edges[i][1]]
        if new_e1[0] != new_e1[1] and new_e2[0] != new_e2[1]:
            edges[i] = new_e1
            edges[j] = new_e2

    return torch.tensor(edges.T, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Graph-Structure Ablation
# ═══════════════════════════════════════════════════════════════════════════════
def run_experiment1(data_root, output_dir, cfg, device):
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT 1: Graph-Structure Ablation")
    logger.info("="*70)

    meta_df = pd.read_csv(os.path.join(data_root, 'clinical_metadata.tsv'), sep='\t')
    ids = meta_df['tcga_barcode'].tolist()
    strat = meta_df['msi_status'].fillna('Unknown').tolist()
    train_ids, tmp_ids, _, tmp_strat = train_test_split(ids, strat, train_size=0.70, stratify=strat, random_state=cfg['seed'])
    val_ids, test_ids = train_test_split(tmp_ids, train_size=0.5, stratify=tmp_strat, random_state=cfg['seed'])

    with h5py.File(os.path.join(data_root, 'recon3d_stoich.h5'), 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32).to(device)

    conditions = {}
    for cond_name in ['real', 'rewired', 'disconnected']:
        logger.info(f"\n--- Condition: {cond_name} ---")
        seed_results = []
        for s in range(3):  # 3 seeds
            run_seed = cfg['seed'] + s
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)

            train_ds = MetaGNNDataset(data_root, train_ids)
            val_ds = MetaGNNDataset(data_root, val_ids)
            test_ds = MetaGNNDataset(data_root, test_ids)

            if cond_name == 'rewired':
                for ds in [train_ds, val_ds, test_ds]:
                    new_ei = {}
                    for rel, ei in ds.edge_index_dict.items():
                        new_ei[rel] = degree_preserving_rewire(ei, seed=run_seed)
                    ds.edge_index_dict = new_ei
            elif cond_name == 'disconnected':
                for ds in [train_ds, val_ds, test_ds]:
                    new_ei = {}
                    for rel, ei in ds.edge_index_dict.items():
                        new_ei[rel] = torch.zeros((2, 0), dtype=torch.long)
                    ds.edge_index_dict = new_ei

            train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
            val_loader = PyGLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
            test_loader = PyGLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

            model = MetaGNN(rxn_in_dim=cfg['rxn_in_dim'], met_in_dim=cfg['met_in_dim'],
                           hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
                           heads=cfg['heads'], dropout=cfg['dropout']).to(device)
            criterion = MetaGNNLoss(S, cfg['lambda_mb'])

            model = train_model(model, train_loader, val_loader, cfg, criterion, device,
                              label=f"{cond_name}-s{s}")
            test_m, _, _ = evaluate_model(model, test_loader, criterion, device, cfg['threshold'])
            logger.info(f"  [{cond_name} seed={s}] F1={test_m['F1']:.4f} AUROC={test_m['AUROC']:.4f}")
            seed_results.append(test_m)

        conditions[cond_name] = {
            'F1_mean': float(np.mean([r['F1'] for r in seed_results])),
            'F1_std': float(np.std([r['F1'] for r in seed_results])),
            'AUROC_mean': float(np.mean([r['AUROC'] for r in seed_results])),
            'AUROC_std': float(np.std([r['AUROC'] for r in seed_results])),
            'AUPRC_mean': float(np.mean([r['AUPRC'] for r in seed_results])),
            'AUPRC_std': float(np.std([r['AUPRC'] for r in seed_results])),
            'per_seed': seed_results,
        }

    # Compute deltas
    real_f1 = conditions['real']['F1_mean']
    for cond in ['rewired', 'disconnected']:
        conditions[cond]['F1_delta'] = real_f1 - conditions[cond]['F1_mean']

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'graph_ablation_results.json'), 'w') as f:
        json.dump(conditions, f, indent=2, default=str)

    logger.info("\n" + "-"*50)
    logger.info("EXPERIMENT 1 SUMMARY:")
    for cond, res in conditions.items():
        logger.info(f"  {cond:>15s}: F1={res['F1_mean']:.4f}±{res['F1_std']:.4f}  "
                    f"AUROC={res['AUROC_mean']:.4f}±{res['AUROC_std']:.4f}")
    return conditions


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Baseline Model Comparisons
# ═══════════════════════════════════════════════════════════════════════════════
def run_experiment2(data_root, output_dir, cfg, device):
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT 2: Baseline Model Comparisons")
    logger.info("="*70)

    meta_df = pd.read_csv(os.path.join(data_root, 'clinical_metadata.tsv'), sep='\t')
    ids = meta_df['tcga_barcode'].tolist()
    strat = meta_df['msi_status'].fillna('Unknown').tolist()
    train_ids, tmp_ids, _, tmp_strat = train_test_split(ids, strat, train_size=0.70, stratify=strat, random_state=cfg['seed'])
    val_ids, test_ids = train_test_split(tmp_ids, train_size=0.5, stratify=tmp_strat, random_state=cfg['seed'])

    with h5py.File(os.path.join(data_root, 'recon3d_stoich.h5'), 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32).to(device)

    model_configs = {
        'MetaGNN': lambda: MetaGNN(rxn_in_dim=cfg['rxn_in_dim'], met_in_dim=cfg['met_in_dim'],
                                    hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
                                    heads=cfg['heads'], dropout=cfg['dropout']),
        'ReactionMLP': lambda: ReactionMLP(in_dim=cfg['rxn_in_dim'], hidden_dim=cfg['hidden_dim'],
                                            dropout=cfg['dropout']),
        'HomoGAT': lambda: HomoGAT(in_dim=cfg['met_in_dim'], hidden_dim=cfg['hidden_dim'],
                                    heads=cfg['heads'], dropout=cfg['dropout']),
        'HomoGCN': lambda: HomoGCN(in_dim=cfg['met_in_dim'], hidden_dim=cfg['hidden_dim'],
                                    dropout=cfg['dropout']),
    }

    all_results = {}
    for model_name, model_fn in model_configs.items():
        logger.info(f"\n--- Model: {model_name} ---")
        seed_results = []
        for s in range(3):
            run_seed = cfg['seed'] + s
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)

            train_ds = MetaGNNDataset(data_root, train_ids)
            val_ds = MetaGNNDataset(data_root, val_ids)
            test_ds = MetaGNNDataset(data_root, test_ids)

            train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
            val_loader = PyGLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
            test_loader = PyGLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

            model = model_fn().to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            criterion = MetaGNNLoss(S, cfg['lambda_mb'])

            t0 = time.time()
            model = train_model(model, train_loader, val_loader, cfg, criterion, device,
                              label=f"{model_name}-s{s}")
            elapsed = time.time() - t0
            test_m, _, _ = evaluate_model(model, test_loader, criterion, device, cfg['threshold'])
            test_m['n_params'] = n_params
            test_m['training_time_s'] = elapsed
            logger.info(f"  [{model_name} s={s}] F1={test_m['F1']:.4f} AUROC={test_m['AUROC']:.4f} "
                        f"params={n_params:,} time={elapsed:.0f}s")
            seed_results.append(test_m)

        all_results[model_name] = {
            'F1_mean': float(np.mean([r['F1'] for r in seed_results])),
            'F1_std': float(np.std([r['F1'] for r in seed_results])),
            'AUROC_mean': float(np.mean([r['AUROC'] for r in seed_results])),
            'AUROC_std': float(np.std([r['AUROC'] for r in seed_results])),
            'AUPRC_mean': float(np.mean([r['AUPRC'] for r in seed_results])),
            'AUPRC_std': float(np.std([r['AUPRC'] for r in seed_results])),
            'n_params': seed_results[0]['n_params'],
            'per_seed': seed_results,
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'baseline_comparison_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "-"*50)
    logger.info("EXPERIMENT 2 SUMMARY:")
    logger.info(f"  {'Model':>15s}  {'F1':>12s}  {'AUROC':>12s}  {'Params':>10s}")
    for name, res in all_results.items():
        logger.info(f"  {name:>15s}  {res['F1_mean']:.4f}±{res['F1_std']:.4f}  "
                    f"{res['AUROC_mean']:.4f}±{res['AUROC_std']:.4f}  {res['n_params']:>10,}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: 5-Fold Stratified Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════════
def run_experiment3(data_root, output_dir, cfg, device):
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT 3: 5-Fold Stratified Cross-Validation")
    logger.info("="*70)

    meta_df = pd.read_csv(os.path.join(data_root, 'clinical_metadata.tsv'), sep='\t')
    patient_ids = meta_df['tcga_barcode'].values
    strat_labels = meta_df['msi_status'].fillna('Unknown').values

    with h5py.File(os.path.join(data_root, 'recon3d_stoich.h5'), 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32).to(device)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg['seed'])
    fold_results = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(patient_ids, strat_labels)):
        logger.info(f"\n--- Fold {fold_idx+1}/5 ---")
        test_ids = patient_ids[test_idx].tolist()
        trainval_ids = patient_ids[trainval_idx]
        trainval_strat = strat_labels[trainval_idx]

        train_ids, val_ids = train_test_split(
            trainval_ids, test_size=0.15, stratify=trainval_strat,
            random_state=cfg['seed'] + fold_idx)
        train_ids = train_ids.tolist()
        val_ids = val_ids.tolist()

        torch.manual_seed(cfg['seed'] + fold_idx)
        np.random.seed(cfg['seed'] + fold_idx)

        train_ds = MetaGNNDataset(data_root, train_ids)
        val_ds = MetaGNNDataset(data_root, val_ids)
        test_ds = MetaGNNDataset(data_root, test_ids)

        train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
        val_loader = PyGLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
        test_loader = PyGLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

        model = MetaGNN(rxn_in_dim=cfg['rxn_in_dim'], met_in_dim=cfg['met_in_dim'],
                        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
                        heads=cfg['heads'], dropout=cfg['dropout']).to(device)
        criterion = MetaGNNLoss(S, cfg['lambda_mb'])

        t0 = time.time()
        model = train_model(model, train_loader, val_loader, cfg, criterion, device,
                          label=f"Fold{fold_idx+1}")
        elapsed = time.time() - t0

        test_m, y_pred, y_true = evaluate_model(model, test_loader, criterion, device, cfg['threshold'])
        test_m['fold'] = fold_idx
        test_m['n_train'] = len(train_ids)
        test_m['n_val'] = len(val_ids)
        test_m['n_test'] = len(test_ids)
        test_m['training_time_s'] = elapsed

        # Baselines for this fold
        majority = float(y_true.mean() >= 0.5)
        maj_pred = np.full_like(y_true, majority)
        test_m['majority_F1'] = float(f1_score(y_true, maj_pred.astype(int), zero_division=0))
        aa_pred = np.ones_like(y_true)
        test_m['all_active_F1'] = float(f1_score(y_true, aa_pred.astype(int), zero_division=0))

        logger.info(f"  [Fold {fold_idx+1}] F1={test_m['F1']:.4f} AUROC={test_m['AUROC']:.4f} "
                    f"AUPRC={test_m['AUPRC']:.4f} ({elapsed:.0f}s)")
        fold_results.append(test_m)

    # Aggregate
    f1s = [r['F1'] for r in fold_results]
    aurocs = [r['AUROC'] for r in fold_results]
    auprcs = [r['AUPRC'] for r in fold_results]

    import scipy.stats as st
    summary = {
        'n_folds': 5,
        'F1_mean': float(np.mean(f1s)), 'F1_std': float(np.std(f1s)),
        'AUROC_mean': float(np.mean(aurocs)), 'AUROC_std': float(np.std(aurocs)),
        'AUPRC_mean': float(np.mean(auprcs)), 'AUPRC_std': float(np.std(auprcs)),
    }
    if len(f1s) > 2:
        ci = st.t.interval(0.95, df=len(f1s)-1, loc=np.mean(f1s), scale=st.sem(f1s))
        summary['F1_95CI'] = [float(ci[0]), float(ci[1])]
        ci_auc = st.t.interval(0.95, df=len(aurocs)-1, loc=np.mean(aurocs), scale=st.sem(aurocs))
        summary['AUROC_95CI'] = [float(ci_auc[0]), float(ci_auc[1])]

    results = {'summary': summary, 'folds': fold_results}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'kfold_cv_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "-"*50)
    logger.info("EXPERIMENT 3 SUMMARY:")
    logger.info(f"  {'Fold':<6} {'F1':>8} {'AUROC':>8} {'AUPRC':>8}")
    for r in fold_results:
        logger.info(f"  {r['fold']+1:<6} {r['F1']:.4f}  {r['AUROC']:.4f}  {r['AUPRC']:.4f}")
    logger.info(f"  {'Mean':<6} {summary['F1_mean']:.4f}  {summary['AUROC_mean']:.4f}  {summary['AUPRC_mean']:.4f}")
    logger.info(f"  {'±Std':<6} {summary['F1_std']:.4f}  {summary['AUROC_std']:.4f}  {summary['AUPRC_std']:.4f}")
    if 'F1_95CI' in summary:
        logger.info(f"  95% CI F1:    [{summary['F1_95CI'][0]:.4f}, {summary['F1_95CI'][1]:.4f}]")
        logger.info(f"  95% CI AUROC: [{summary['AUROC_95CI'][0]:.4f}, {summary['AUROC_95CI'][1]:.4f}]")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetaGNN Full Experiment Suite')
    parser.add_argument('--data_root', type=str, default='./experiment_data')
    parser.add_argument('--output_root', type=str, default='./experiment_results')
    parser.add_argument('--n_patients', type=int, default=220)
    parser.add_argument('--experiments', type=str, default='1,2,3',
                        help='Comma-separated experiment numbers to run')
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg['data_root'] = args.data_root
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Step 1: Generate data
    if not os.path.exists(os.path.join(args.data_root, 'clinical_metadata.tsv')):
        generate_recon3d_like_data(args.data_root, n_patients=args.n_patients, seed=cfg['seed'])
    else:
        logger.info("Data already exists, skipping generation.")

    exps = [int(x.strip()) for x in args.experiments.split(',')]
    all_exp_results = {}

    if 3 in exps:
        res3 = run_experiment3(args.data_root, os.path.join(args.output_root, 'kfold'), cfg, device)
        all_exp_results['experiment3_kfold'] = res3

    if 1 in exps:
        res1 = run_experiment1(args.data_root, os.path.join(args.output_root, 'ablation'), cfg, device)
        all_exp_results['experiment1_ablation'] = res1

    if 2 in exps:
        res2 = run_experiment2(args.data_root, os.path.join(args.output_root, 'baselines'), cfg, device)
        all_exp_results['experiment2_baselines'] = res2

    # Save combined summary
    with open(os.path.join(args.output_root, 'all_experiments_summary.json'), 'w') as f:
        json.dump(all_exp_results, f, indent=2, default=str)

    logger.info("\n" + "="*70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Results saved to: {args.output_root}/")
    logger.info("="*70)
