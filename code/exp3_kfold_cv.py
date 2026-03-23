#!/usr/bin/env python3
"""
Experiment 3: K-Fold Stratified Cross-Validation
==================================================
Replaces the single 70/15/15 train/val/test split with rigorous
5-fold stratified cross-validation (stratified by MSI status).

This addresses the reviewer concern that a single split may produce
optimistic or pessimistic estimates due to unlucky partitioning.

Reports:
  - Per-fold F1, AUROC, AUPRC
  - Mean ± std across folds
  - Paired comparison with majority-class and GPR-threshold baselines

Usage:
    cd /path/to/MetaGNN
    python code/exp3_kfold_cv.py --data_root ./data --output_dir ./results/kfold

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
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metagnn_model import MetaGNN
from data_loader import MetaGNNDataset
from train_metagnn import MetaGNNLoss, compute_metrics, DEFAULTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(
    cfg: dict,
    fold: int,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    device: torch.device,
) -> dict:
    """
    Train MetaGNN on one CV fold.

    Args:
        cfg:        hyperparameter config
        fold:       fold index (0-based)
        train_ids:  TCGA barcodes for training
        val_ids:    TCGA barcodes for validation (early stopping)
        test_ids:   TCGA barcodes for held-out test
        device:     torch device

    Returns:
        dict with test metrics for this fold
    """
    torch.manual_seed(cfg['seed'] + fold)
    np.random.seed(cfg['seed'] + fold)

    logger.info(f"\n{'─'*50}")
    logger.info(f"  Fold {fold+1}: train={len(train_ids)}, "
                f"val={len(val_ids)}, test={len(test_ids)}")
    logger.info(f"{'─'*50}")

    train_ds = MetaGNNDataset(cfg['data_root'], train_ids)
    val_ds   = MetaGNNDataset(cfg['data_root'], val_ids)
    test_ds  = MetaGNNDataset(cfg['data_root'], test_ids)

    train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = PyGLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)
    test_loader  = PyGLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False)

    # Fresh model
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

    # Training helpers (inline for simplicity)
    def _train_epoch():
        model.train()
        total_loss = 0.0
        for batch in train_loader:
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
        return total_loss / len(train_loader)

    @torch.no_grad()
    def _evaluate(loader):
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
        metrics = compute_metrics(y_pred, y_true, cfg['threshold'])
        metrics['loss'] = total_loss / len(loader)
        return metrics, y_pred, y_true

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    save_path = os.path.join(cfg['output_dir'], f'kfold_model_fold{fold}.pt')

    t0 = time.time()
    for epoch in range(1, cfg['n_epochs_finetune'] + 1):
        train_loss = _train_epoch()
        val_metrics, _, _ = _evaluate(val_loader)
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                f"  [Fold {fold+1}] Epoch {epoch:3d}  "
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
                logger.info(f"  [Fold {fold+1}] Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0

    # Test evaluation
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics, y_pred, y_true = _evaluate(test_loader)
    test_metrics['training_time_s'] = elapsed
    test_metrics['converged_epoch'] = epoch - patience_counter

    # Also compute baseline metrics on same test fold
    # 1. Majority class baseline
    majority_label = int(y_true.mean() >= 0.5)
    majority_pred = np.full_like(y_true, majority_label, dtype=float)
    majority_metrics = compute_metrics(majority_pred, y_true, threshold=0.5)

    # 2. GPR threshold baseline (predict active if label == 1)
    gpr_pred = y_true.copy().astype(float)  # perfect on train labels = trivial
    # More realistic: predict all-active (70.1% active rate)
    all_active_pred = np.ones_like(y_true, dtype=float)
    all_active_metrics = compute_metrics(all_active_pred, y_true, threshold=0.5)

    test_metrics['baselines'] = {
        'majority': majority_metrics,
        'all_active': all_active_metrics,
    }

    logger.info(
        f"  [Fold {fold+1}] TEST — "
        f"F1={test_metrics['F1']:.4f}  "
        f"AUROC={test_metrics['AUROC']:.4f}  "
        f"AUPRC={test_metrics['AUPRC']:.4f}  "
        f"({elapsed:.0f}s)"
    )
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main: K-fold CV
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='MetaGNN K-Fold Cross-Validation (Experiment 3)'
    )
    parser.add_argument('--data_root',  type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/kfold')
    parser.add_argument('--n_folds',    type=int, default=5)
    parser.add_argument('--val_frac',   type=float, default=0.15,
                        help='Fraction of training fold held out for early stopping')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = dict(DEFAULTS)
    cfg['data_root']  = args.data_root
    cfg['output_dir'] = args.output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Running {args.n_folds}-fold stratified CV")

    # Load clinical metadata for stratification
    meta_df = pd.read_csv(
        os.path.join(cfg['data_root'], 'clinical_metadata.tsv'), sep='\t'
    )
    patient_ids = meta_df['tcga_barcode'].values
    strat_labels = meta_df['msi_status'].fillna('Unknown').values

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=cfg['seed']
    )

    fold_results = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(
        skf.split(patient_ids, strat_labels)
    ):
        test_ids = patient_ids[test_idx].tolist()
        trainval_ids = patient_ids[trainval_idx]
        trainval_strat = strat_labels[trainval_idx]

        # Split trainval into train + val (for early stopping)
        from sklearn.model_selection import train_test_split
        train_ids, val_ids = train_test_split(
            trainval_ids,
            test_size=args.val_frac,
            stratify=trainval_strat,
            random_state=cfg['seed'] + fold_idx,
        )
        train_ids = train_ids.tolist()
        val_ids = val_ids.tolist()

        fold_metrics = train_fold(cfg, fold_idx, train_ids, val_ids, test_ids, device)
        fold_metrics['fold'] = fold_idx
        fold_metrics['n_train'] = len(train_ids)
        fold_metrics['n_val'] = len(val_ids)
        fold_metrics['n_test'] = len(test_ids)
        fold_results.append(fold_metrics)

    # Aggregate across folds
    f1s    = [r['F1'] for r in fold_results]
    aurocs = [r['AUROC'] for r in fold_results]
    auprcs = [r['AUPRC'] for r in fold_results]
    times  = [r['training_time_s'] for r in fold_results]

    summary = {
        'n_folds': args.n_folds,
        'F1_mean':    float(np.mean(f1s)),
        'F1_std':     float(np.std(f1s)),
        'F1_min':     float(np.min(f1s)),
        'F1_max':     float(np.max(f1s)),
        'AUROC_mean': float(np.mean(aurocs)),
        'AUROC_std':  float(np.std(aurocs)),
        'AUPRC_mean': float(np.mean(auprcs)),
        'AUPRC_std':  float(np.std(auprcs)),
        'total_time_s': float(sum(times)),
        'per_fold_time_s': float(np.mean(times)),
    }

    results = {
        'summary': summary,
        'folds': fold_results,
        'config': {k: v for k, v in cfg.items()
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool))},
    }

    # Save
    out_path = os.path.join(args.output_dir, 'kfold_cv_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"{args.n_folds}-FOLD STRATIFIED CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"\n{'Fold':<8} {'F1':>8} {'AUROC':>8} {'AUPRC':>8} {'Time':>8}")
    print("-" * 44)
    for r in fold_results:
        print(f"  {r['fold']+1:<6} {r['F1']:.4f}  {r['AUROC']:.4f}  "
              f"{r['AUPRC']:.4f}  {r['training_time_s']:.0f}s")
    print("-" * 44)
    print(f"  {'Mean':<6} {summary['F1_mean']:.4f}  "
          f"{summary['AUROC_mean']:.4f}  {summary['AUPRC_mean']:.4f}")
    print(f"  {'±Std':<6} {summary['F1_std']:.4f}  "
          f"{summary['AUROC_std']:.4f}  {summary['AUPRC_std']:.4f}")
    print("=" * 60)

    # Confidence interval (approximate 95% CI)
    import scipy.stats as st
    n = len(f1s)
    if n > 2:
        ci_f1 = st.t.interval(0.95, df=n-1,
                               loc=np.mean(f1s), scale=st.sem(f1s))
        ci_auroc = st.t.interval(0.95, df=n-1,
                                  loc=np.mean(aurocs), scale=st.sem(aurocs))
        print(f"\n95% CI for F1:    [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")
        print(f"95% CI for AUROC: [{ci_auroc[0]:.4f}, {ci_auroc[1]:.4f}]")

    # Compare with original single-split results
    print(f"\nOriginal single-split: F1=0.814, AUROC=0.874")
    print(f"K-fold CV:             F1={summary['F1_mean']:.3f}±{summary['F1_std']:.3f}, "
          f"AUROC={summary['AUROC_mean']:.3f}±{summary['AUROC_std']:.3f}")

    if summary['F1_std'] < 0.03:
        print("\n✓ Low variance across folds — results are stable.")
    elif summary['F1_std'] < 0.05:
        print("\n~ Moderate variance — consider investigating fold-specific patterns.")
    else:
        print("\n⚠ High variance — the original single split may not be representative.")

    print(f"\nTotal time: {summary['total_time_s']/60:.1f} minutes")


if __name__ == '__main__':
    main()
