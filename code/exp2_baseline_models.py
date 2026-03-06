#!/usr/bin/env python3
"""
Experiment 2: Baseline Model Comparisons
==========================================
Trains three additional baselines to isolate the contributions of:
  (a) Graph topology (MLP has no graph)
  (b) Heterogeneous encoding (Homogeneous GAT collapses node types)
  (c) Attention mechanism (GCN uses uniform aggregation)

Baselines:
  1. MLP          — Same features, same hidden dim, NO graph structure
  2. Homo-GAT     — GATv2 on a homogeneous version of Recon3D (single node type)
  3. Homo-GCN     — GCN on homogeneous Recon3D (simpler aggregation)

All models use the same train/val/test split, same seed, and same
number of parameters (as close as possible) for fair comparison.

Usage:
    cd /path/to/MetaGNN
    python code/exp2_baseline_models.py --data_root ./data --output_dir ./results/baselines

Author: Thiptanawat Phongwattana
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metagnn_model import MetaGNN, MCDropout
from data_loader import MetaGNNDataset, stratified_split
from train_metagnn import MetaGNNLoss, compute_metrics, DEFAULTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 1: MLP (no graph structure at all)
# ═══════════════════════════════════════════════════════════════════════════════
class ReactionMLP(nn.Module):
    """
    MLP baseline operating on per-reaction features only (no graph).
    Same hidden dim and depth as MetaGNN for fair comparison.

    If MetaGNN ≈ MLP, then the graph topology adds no value.
    """

    def __init__(self, in_dim=2, hidden_dim=256, n_layers=3, dropout=0.2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ELU())
        layers.append(MCDropout(dropout))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            layers.append(MCDropout(dropout))

        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        layers.append(nn.ELU())
        layers.append(MCDropout(dropout))
        layers.append(nn.Linear(hidden_dim // 2, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x_dict, edge_index_dict=None):
        """Accept same interface as MetaGNN for compatibility.
        Ignores metabolite features and edge indices entirely."""
        x_r = x_dict['reaction']  # (n_rxn, 2)
        return self.net(x_r).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 2: Homogeneous GAT (single node type)
# ═══════════════════════════════════════════════════════════════════════════════
class HomoGAT(nn.Module):
    """
    Homogeneous GATv2 on Recon3D — treats all nodes (reactions + metabolites)
    as a single type with a unified feature space.

    Tests whether heterogeneous encoding (separate projections per node type)
    matters. If MetaGNN ≈ HomoGAT, heterogeneous encoding adds no value.
    """

    def __init__(self, in_dim=519, hidden_dim=256, n_layers=3,
                 heads=8, dropout=0.2, n_reactions=13543):
        super().__init__()
        self.n_reactions = n_reactions

        # Input projection to shared space
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GATv2Conv(
                hidden_dim, hidden_dim, heads=heads,
                dropout=dropout, add_self_loops=True, concat=False
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.drops.append(MCDropout(dropout))

        # Output head (only for reaction nodes)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            MCDropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_dict, edge_index_dict):
        """
        Merges heterogeneous data into a single homogeneous graph.
        Reactions are nodes [0, n_rxn), metabolites are [n_rxn, n_rxn+n_met).
        """
        x_r = x_dict['reaction']    # (n_rxn, 2)
        x_m = x_dict['metabolite']  # (n_met, 519)
        n_rxn = x_r.shape[0]
        n_met = x_m.shape[0]

        # Pad reaction features to match metabolite dim (519)
        x_r_padded = torch.zeros(n_rxn, x_m.shape[1], device=x_r.device)
        x_r_padded[:, :x_r.shape[1]] = x_r

        # Concatenate all nodes: [reactions, metabolites]
        x_all = torch.cat([x_r_padded, x_m], dim=0)  # (n_rxn+n_met, 519)

        # Merge edge indices into homogeneous graph
        # Offset metabolite indices by n_rxn
        all_edges = []
        for (src_type, rel, dst_type), ei in edge_index_dict.items():
            src_offset = 0 if src_type == 'reaction' else n_rxn
            dst_offset = 0 if dst_type == 'reaction' else n_rxn
            offset_ei = ei.clone()
            offset_ei[0] += src_offset
            offset_ei[1] += dst_offset
            all_edges.append(offset_ei)

        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x_r.device)

        # Forward pass
        h = self.proj(x_all)
        for conv, norm, drop in zip(self.convs, self.norms, self.drops):
            h_new = conv(h, edge_index)
            h = norm(nn.functional.elu(h_new) + h)  # residual
            h = drop(h)

        # Output: only reaction nodes
        h_rxn = h[:n_rxn]
        s_r = self.output_head(h_rxn).squeeze(-1)
        return s_r


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 3: Homogeneous GCN (simpler aggregation)
# ═══════════════════════════════════════════════════════════════════════════════
class HomoGCN(nn.Module):
    """
    Homogeneous GCN — uniform aggregation (no attention mechanism).
    Tests whether the attention mechanism in GATv2 matters.
    """

    def __init__(self, in_dim=519, hidden_dim=256, n_layers=3,
                 dropout=0.2, n_reactions=13543):
        super().__init__()
        self.n_reactions = n_reactions

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.drops.append(MCDropout(dropout))

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            MCDropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_dict, edge_index_dict):
        x_r = x_dict['reaction']
        x_m = x_dict['metabolite']
        n_rxn = x_r.shape[0]
        n_met = x_m.shape[0]

        x_r_padded = torch.zeros(n_rxn, x_m.shape[1], device=x_r.device)
        x_r_padded[:, :x_r.shape[1]] = x_r
        x_all = torch.cat([x_r_padded, x_m], dim=0)

        all_edges = []
        for (src_type, rel, dst_type), ei in edge_index_dict.items():
            src_offset = 0 if src_type == 'reaction' else n_rxn
            dst_offset = 0 if dst_type == 'reaction' else n_rxn
            offset_ei = ei.clone()
            offset_ei[0] += src_offset
            offset_ei[1] += dst_offset
            all_edges.append(offset_ei)

        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x_r.device)

        h = self.proj(x_all)
        for conv, norm, drop in zip(self.convs, self.norms, self.drops):
            h_new = conv(h, edge_index)
            h = norm(nn.functional.elu(h_new) + h)
            h = drop(h)

        h_rxn = h[:n_rxn]
        s_r = self.output_head(h_rxn).squeeze(-1)
        return s_r


# ═══════════════════════════════════════════════════════════════════════════════
# Unified training function
# ═══════════════════════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader, test_loader,
                criterion, cfg, device, model_name, seed):
    """Train any model that accepts (x_dict, edge_index_dict) interface."""

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['n_epochs_finetune']
    )

    best_val_f1 = 0.0
    patience_counter = 0
    save_path = os.path.join(
        cfg['output_dir'], f'baseline_{model_name}_seed{seed}.pt'
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  [{model_name}] Parameters: {n_params:,}")

    t0 = time.time()

    # Custom train/eval that works with heterogeneous dict interface
    def train_epoch_generic(model, loader, optimizer, criterion, device):
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
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_generic(model, loader, criterion, device, threshold=0.15):
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
        metrics['loss'] = total_loss / len(loader)
        return metrics

    for epoch in range(1, cfg['n_epochs_finetune'] + 1):
        train_loss = train_epoch_generic(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = eval_generic(
            model, val_loader, criterion, device, cfg['threshold']
        )
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  [{model_name}] Epoch {epoch:3d}  "
                f"train_loss={train_loss:.4f}  "
                f"val_F1={val_metrics['F1']:.4f}  "
                f"val_AUROC={val_metrics['AUROC']:.4f}"
            )

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                logger.info(f"  [{model_name}] Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0

    # Test evaluation
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = eval_generic(
        model, test_loader, criterion, device, cfg['threshold']
    )
    test_metrics['training_time_s'] = elapsed
    test_metrics['n_params'] = n_params
    test_metrics['converged_epoch'] = epoch - patience_counter

    logger.info(
        f"  [{model_name}] TEST — F1={test_metrics['F1']:.4f}  "
        f"AUROC={test_metrics['AUROC']:.4f}  "
        f"AUPRC={test_metrics['AUPRC']:.4f}  "
        f"({n_params:,} params, {elapsed:.0f}s)"
    )
    return test_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='MetaGNN Baseline Comparisons (Experiment 2)'
    )
    parser.add_argument('--data_root',  type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/baselines')
    parser.add_argument('--n_seeds',    type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = dict(DEFAULTS)
    cfg['data_root']  = args.data_root
    cfg['output_dir'] = args.output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    seeds = [2024, 2025, 2026][:args.n_seeds]

    # Load data (shared across all models)
    import pandas as pd
    meta_df = pd.read_csv(
        os.path.join(cfg['data_root'], 'clinical_metadata.tsv'), sep='\t'
    )
    train_ids, val_ids, test_ids = stratified_split(
        meta_df, train_frac=0.70, val_frac=0.15, seed=cfg['seed']
    )

    train_ds = MetaGNNDataset(cfg['data_root'], train_ids)
    val_ds   = MetaGNNDataset(cfg['data_root'], val_ids)
    test_ds  = MetaGNNDataset(cfg['data_root'], test_ids)

    train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = PyGLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)
    test_loader  = PyGLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False)

    # Load stoichiometric matrix for loss
    import h5py
    with h5py.File(os.path.join(cfg['data_root'], 'recon3d_stoich.h5'), 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32).to(device)

    # Model configurations
    model_configs = {
        'MetaGNN': lambda seed: MetaGNN(
            rxn_in_dim=2, met_in_dim=519,
            hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
            heads=cfg['heads'], dropout=cfg['dropout'],
        ),
        'MLP': lambda seed: ReactionMLP(
            in_dim=2, hidden_dim=cfg['hidden_dim'],
            n_layers=cfg['n_layers'], dropout=cfg['dropout'],
        ),
        'HomoGAT': lambda seed: HomoGAT(
            in_dim=519, hidden_dim=cfg['hidden_dim'],
            n_layers=cfg['n_layers'], heads=cfg['heads'],
            dropout=cfg['dropout'],
        ),
        'HomoGCN': lambda seed: HomoGCN(
            in_dim=519, hidden_dim=cfg['hidden_dim'],
            n_layers=cfg['n_layers'], dropout=cfg['dropout'],
        ),
    }

    all_results = {}

    for model_name, model_factory in model_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Training: {model_name}")
        logger.info(f"{'='*60}")

        model_runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = model_factory(seed).to(device)
            criterion = MetaGNNLoss(
                stoich_matrix=S, lambda_mb=cfg.get('lambda_mb', 0.2)
            )

            metrics = train_model(
                model, train_loader, val_loader, test_loader,
                criterion, cfg, device, model_name, seed
            )
            metrics['seed'] = seed
            model_runs.append(metrics)

        # Aggregate
        f1s    = [r['F1'] for r in model_runs]
        aurocs = [r['AUROC'] for r in model_runs]
        auprcs = [r['AUPRC'] for r in model_runs]

        all_results[model_name] = {
            'runs': model_runs,
            'summary': {
                'F1_mean':    float(np.mean(f1s)),
                'F1_std':     float(np.std(f1s)),
                'AUROC_mean': float(np.mean(aurocs)),
                'AUROC_std':  float(np.std(aurocs)),
                'AUPRC_mean': float(np.mean(auprcs)),
                'AUPRC_std':  float(np.std(auprcs)),
                'n_params':   model_runs[0]['n_params'],
            }
        }

    # Save
    out_path = os.path.join(args.output_dir, 'baseline_comparison_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 86)
    print("BASELINE MODEL COMPARISON")
    print("=" * 86)
    print(f"{'Model':<14} {'Params':>10} {'F1':>14} {'AUROC':>14} {'AUPRC':>14}")
    print("-" * 86)
    for name in ['MetaGNN', 'MLP', 'HomoGAT', 'HomoGCN']:
        s = all_results[name]['summary']
        print(f"{name:<14} {s['n_params']:>10,} "
              f"{s['F1_mean']:.3f}±{s['F1_std']:.3f}   "
              f"{s['AUROC_mean']:.3f}±{s['AUROC_std']:.3f}   "
              f"{s['AUPRC_mean']:.3f}±{s['AUPRC_std']:.3f}")
    print("=" * 86)

    # Key comparisons
    meta_f1 = all_results['MetaGNN']['summary']['F1_mean']
    mlp_f1  = all_results['MLP']['summary']['F1_mean']
    hgat_f1 = all_results['HomoGAT']['summary']['F1_mean']
    hgcn_f1 = all_results['HomoGCN']['summary']['F1_mean']

    print(f"\nKey comparisons (ΔF1 vs MetaGNN):")
    print(f"  MetaGNN vs MLP:     {meta_f1 - mlp_f1:+.3f}  "
          f"({'Graph helps' if meta_f1 > mlp_f1 + 0.02 else 'Graph adds little'})")
    print(f"  MetaGNN vs HomoGAT: {meta_f1 - hgat_f1:+.3f}  "
          f"({'Hetero helps' if meta_f1 > hgat_f1 + 0.01 else 'Hetero adds little'})")
    print(f"  MetaGNN vs HomoGCN: {meta_f1 - hgcn_f1:+.3f}  "
          f"({'Attention helps' if meta_f1 > hgcn_f1 + 0.01 else 'Attention adds little'})")


if __name__ == '__main__':
    main()
