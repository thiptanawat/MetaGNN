#!/usr/bin/env python3
"""
Experiment 1: Graph-Structure Ablation
=======================================
The definitive test of whether Recon3D topology contributes to prediction quality.

Three conditions:
  (a) REAL     — Original Recon3D graph topology (control)
  (b) REWIRED  — Degree-preserving randomly rewired graph (same degree distribution,
                 destroyed biological topology)
  (c) DISCONNECTED — Fully disconnected graph (no message passing; equivalent to
                     a per-node MLP with the same architecture)

If MetaGNN's performance is preserved on the rewired or disconnected graph,
the topology claim collapses.

Usage:
    cd /path/to/MetaGNN
    python code/exp1_graph_structure_ablation.py --data_root ./data --output_dir ./results/ablation

Author: Thiptanawat Phongwattana
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGLoader

# Add code/ to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metagnn_model import MetaGNN
from data_loader import MetaGNNDataset, stratified_split
from train_metagnn import MetaGNNLoss, compute_metrics, train_epoch, evaluate, DEFAULTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Graph rewiring utilities
# ─────────────────────────────────────────────────────────────────────────────
def degree_preserving_rewire(edge_index: torch.Tensor, n_swaps: int = None,
                              seed: int = 42) -> torch.Tensor:
    """
    Degree-preserving random rewiring (Maslov & Sneppen, Science 2002).

    Randomly picks two edges (u→v, x→y) and swaps targets (u→y, x→v),
    provided the new edges don't already exist. This preserves the degree
    sequence exactly while destroying biological topology.

    Args:
        edge_index: Tensor(2, n_edges) — directed edge list
        n_swaps:    number of swap attempts (default: 10× n_edges)
        seed:       random seed for reproducibility

    Returns:
        rewired edge_index: Tensor(2, n_edges)
    """
    rng = np.random.RandomState(seed)
    edges = edge_index.numpy().T.copy()  # (n_edges, 2)
    n_edges = len(edges)

    if n_swaps is None:
        n_swaps = 10 * n_edges

    # Build a set of existing edges for O(1) lookup
    edge_set = set(map(tuple, edges))

    successful = 0
    for _ in range(n_swaps):
        # Pick two random edges
        i, j = rng.randint(0, n_edges, size=2)
        if i == j:
            continue

        u, v = edges[i]
        x, y = edges[j]

        # Proposed swap: (u→y, x→v)
        if u == y or x == v:  # no self-loops
            continue
        if (u, y) in edge_set or (x, v) in edge_set:  # no multi-edges
            continue

        # Perform swap
        edge_set.discard((u, v))
        edge_set.discard((x, y))
        edges[i] = [u, y]
        edges[j] = [x, v]
        edge_set.add((u, y))
        edge_set.add((x, v))
        successful += 1

    logger.info(f"  Rewiring: {successful}/{n_swaps} swaps successful "
                f"({successful/n_swaps:.1%})")
    return torch.tensor(edges.T, dtype=torch.long)


def create_empty_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Create an empty edge index (no edges) with same dtype."""
    return torch.zeros((2, 0), dtype=torch.long)


def rewire_dataset(dataset: MetaGNNDataset, mode: str, seed: int = 42):
    """
    Modify a dataset's edge indices in-place.

    Args:
        dataset: MetaGNNDataset instance
        mode:    'real' (no change), 'rewired', or 'disconnected'
        seed:    random seed
    """
    if mode == 'real':
        return  # no modification

    for rel_key, ei in dataset.edge_index_dict.items():
        if mode == 'disconnected':
            dataset.edge_index_dict[rel_key] = create_empty_edges(ei)
        elif mode == 'rewired':
            logger.info(f"  Rewiring edge type: {rel_key[1]} ({ei.shape[1]} edges)")
            dataset.edge_index_dict[rel_key] = degree_preserving_rewire(
                ei, seed=seed
            )


# ─────────────────────────────────────────────────────────────────────────────
# Single ablation run
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(cfg: dict, mode: str, seed: int) -> dict:
    """
    Train MetaGNN from scratch under a specific graph condition.

    Args:
        cfg:  config dict (same as train_metagnn.py)
        mode: 'real', 'rewired', or 'disconnected'
        seed: random seed for this run

    Returns:
        dict with test metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"\n{'='*60}")
    logger.info(f"  Ablation: mode={mode}, seed={seed}")
    logger.info(f"{'='*60}")

    # Data splits (same split as original paper)
    import pandas as pd
    meta_df = pd.read_csv(
        os.path.join(cfg['data_root'], 'clinical_metadata.tsv'), sep='\t'
    )
    train_ids, val_ids, test_ids = stratified_split(
        meta_df, train_frac=0.70, val_frac=0.15, seed=cfg['seed']  # FIXED split
    )

    # Create datasets and apply graph modification
    train_ds = MetaGNNDataset(cfg['data_root'], train_ids)
    val_ds   = MetaGNNDataset(cfg['data_root'], val_ids)
    test_ds  = MetaGNNDataset(cfg['data_root'], test_ids)

    logger.info(f"  Applying graph modification: {mode}")
    rewire_dataset(train_ds, mode, seed=seed)
    rewire_dataset(val_ds,   mode, seed=seed)
    rewire_dataset(test_ds,  mode, seed=seed)

    train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = PyGLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)
    test_loader  = PyGLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False)

    # Model (fresh initialisation — no pre-trained weights for fair comparison)
    model = MetaGNN(
        rxn_in_dim=2, met_in_dim=519,
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        heads=cfg['heads'],
        dropout=cfg['dropout'],
    ).to(device)

    import h5py
    with h5py.File(os.path.join(cfg['data_root'], 'recon3d_stoich.h5'), 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32).to(device)

    criterion = MetaGNNLoss(stoich_matrix=S, lambda_mb=cfg.get('lambda_mb', 0.2))
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['n_epochs_finetune']
    )

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    save_path = os.path.join(
        cfg['output_dir'], f'ablation_{mode}_seed{seed}.pt'
    )

    t0 = time.time()
    for epoch in range(1, cfg['n_epochs_finetune'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg['threshold'])
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  [{mode}] Epoch {epoch:3d}  "
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
                logger.info(f"  [{mode}] Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    logger.info(f"  [{mode}] Training completed in {elapsed:.0f}s")

    # Test evaluation
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, cfg['threshold'])
    test_metrics['training_time_s'] = elapsed
    test_metrics['converged_epoch'] = epoch - patience_counter

    logger.info(
        f"  [{mode}] TEST — F1={test_metrics['F1']:.4f}  "
        f"AUROC={test_metrics['AUROC']:.4f}  "
        f"AUPRC={test_metrics['AUPRC']:.4f}"
    )
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main: run all conditions × 3 seeds
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='MetaGNN Graph-Structure Ablation (Experiment 1)'
    )
    parser.add_argument('--data_root',  type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/ablation')
    parser.add_argument('--n_seeds',    type=int, default=3,
                        help='Number of random seeds per condition')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = dict(DEFAULTS)
    cfg['data_root']  = args.data_root
    cfg['output_dir'] = args.output_dir

    modes = ['real', 'rewired', 'disconnected']
    seeds = [2024, 2025, 2026][:args.n_seeds]

    all_results = {}

    for mode in modes:
        mode_results = []
        for seed in seeds:
            metrics = run_ablation(cfg, mode, seed)
            metrics['seed'] = seed
            mode_results.append(metrics)

        # Aggregate across seeds
        f1s    = [r['F1'] for r in mode_results]
        aurocs = [r['AUROC'] for r in mode_results]
        auprcs = [r['AUPRC'] for r in mode_results]

        all_results[mode] = {
            'runs': mode_results,
            'summary': {
                'F1_mean':    float(np.mean(f1s)),
                'F1_std':     float(np.std(f1s)),
                'AUROC_mean': float(np.mean(aurocs)),
                'AUROC_std':  float(np.std(aurocs)),
                'AUPRC_mean': float(np.mean(auprcs)),
                'AUPRC_std':  float(np.std(auprcs)),
            }
        }

    # Save results
    out_path = os.path.join(args.output_dir, 'graph_ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 72)
    print("GRAPH-STRUCTURE ABLATION RESULTS")
    print("=" * 72)
    print(f"{'Condition':<18} {'F1':>14} {'AUROC':>14} {'AUPRC':>14}")
    print("-" * 72)
    for mode in modes:
        s = all_results[mode]['summary']
        print(f"{mode:<18} "
              f"{s['F1_mean']:.3f}±{s['F1_std']:.3f}   "
              f"{s['AUROC_mean']:.3f}±{s['AUROC_std']:.3f}   "
              f"{s['AUPRC_mean']:.3f}±{s['AUPRC_std']:.3f}")
    print("=" * 72)

    # Interpretation
    real_f1 = all_results['real']['summary']['F1_mean']
    rewired_f1 = all_results['rewired']['summary']['F1_mean']
    disc_f1 = all_results['disconnected']['summary']['F1_mean']

    delta_rewired = real_f1 - rewired_f1
    delta_disc    = real_f1 - disc_f1

    print(f"\nΔF1 (real - rewired):      {delta_rewired:+.3f}")
    print(f"ΔF1 (real - disconnected): {delta_disc:+.3f}")

    if delta_rewired > 0.02:
        print("\n✓ Biological topology contributes meaningfully to prediction.")
        print("  The Recon3D graph structure is NOT interchangeable with a random graph.")
    else:
        print("\n⚠ Topology contribution is marginal (<2% F1 difference).")
        print("  Consider whether node features alone drive the predictions.")

    if delta_disc > 0.05:
        print("✓ Message passing (graph structure) provides >5% F1 improvement over no-graph.")
    else:
        print("⚠ Message passing provides <5% F1 improvement — feature engineering may dominate.")


if __name__ == '__main__':
    main()
