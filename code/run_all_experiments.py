#!/usr/bin/env python3
"""
Combined Experiment Runner for MetaGNN MethodsX — Missing Experiments
=====================================================================
Runs the TWO pending experiments sequentially on CPU:
  1. Proteomics Ablation: Baseline (RNA+Prot) vs RNA-only (prot zeroed), 3 seeds
  2. FBA Expansion: Validate predictions on all 34 test patients via COBRApy

Uses the EXACT same model configuration as prepare_and_run_experiments.py:
  hidden_dim=128, n_layers=2, heads=4, batch_size=4, n_epochs=80, patience=15

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
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
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
# Configuration — MUST match prepare_and_run_experiments.py exactly
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    hidden_dim=128,       # 128 (matches existing results; paper describes 256)
    n_layers=2,           # 2 layers (matches existing results)
    heads=4,              # 4 heads (matches existing results)
    dropout=0.20,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=4,         # batch=4 for memory
    n_epochs=80,
    patience=15,
    threshold=0.15,
    seed=2024,
    lambda_mb=0.2,
    rxn_in_dim=2,
    met_in_dim=519,       # 7 physico-chem + 512 Morgan FP
)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION — copied from prepare_and_run_experiments.py
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


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS — uses reshape approach from prepare_and_run_experiments.py
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
# DATASET — uses data_loader.py from existing codebase
# ═══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import MetaGNNDataset, stratified_split


class RNAOnlyDataset(MetaGNNDataset):
    """Wraps MetaGNNDataset but zeros out the proteomics column (col 1)."""
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['reaction'].x = data['reaction'].x.clone()
        data['reaction'].x[:, 1] = 0.0
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════
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
    return metrics


@torch.no_grad()
def evaluate_per_patient(model, dataset, device, threshold=0.15):
    """Evaluate per-patient for more meaningful std measurements."""
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


def run_condition(condition_name, dataset_cls, data_root, train_ids, val_ids, test_ids,
                  device, stoich_matrix, cfg, seed):
    """Train one condition and return test metrics + best model state."""
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
        rxn_in_dim=cfg['rxn_in_dim'], met_in_dim=cfg['met_in_dim'],
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model parameters: {n_params:,}")

    criterion = MetaGNNLoss(stoich_matrix=stoich_matrix.to(device), lambda_mb=cfg['lambda_mb'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'])

    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None
    start_time = time.time()

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

    logger.info(f"  Test F1:    {test_results['F1_mean']:.4f} ± {test_results['F1_std']:.4f}")
    logger.info(f"  Test AUROC: {test_results['AUROC_mean']:.4f} ± {test_results['AUROC_std']:.4f}")
    logger.info(f"  Test AUPRC: {test_results['AUPRC_mean']:.4f} ± {test_results['AUPRC_std']:.4f}")

    test_results['best_val_f1'] = best_val_f1
    test_results['train_time_s'] = train_time
    test_results['seed'] = seed
    test_results['n_params'] = n_params
    return test_results, best_state


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: PROTEOMICS ABLATION
# ═══════════════════════════════════════════════════════════════════════════════
def run_ablation(args, cfg, device, train_ids, val_ids, test_ids, S):
    logger.info("\n" + "#"*60)
    logger.info("# PART 1: PROTEOMICS ABLATION EXPERIMENT")
    logger.info("#"*60)

    output_dir = os.path.join(args.output_dir, 'ablation_proteomics')
    os.makedirs(output_dir, exist_ok=True)

    seeds = [2024, 42, 123][:args.n_seeds]
    all_results = {'baseline': [], 'rna_only': []}
    best_baseline_state = None
    best_baseline_f1 = -1.0

    for seed in seeds:
        # Baseline: RNA + Proteomics (normal)
        res_baseline, state = run_condition(
            'Baseline (RNA+Proteomics)', MetaGNNDataset,
            args.data_root, train_ids, val_ids, test_ids,
            device, S, cfg, seed
        )
        all_results['baseline'].append(res_baseline)
        if res_baseline['F1_mean'] > best_baseline_f1:
            best_baseline_f1 = res_baseline['F1_mean']
            best_baseline_state = state

        # RNA-only: proteomics zeroed
        res_rna, _ = run_condition(
            'RNA-only (proteomics zeroed)', RNAOnlyDataset,
            args.data_root, train_ids, val_ids, test_ids,
            device, S, cfg, seed
        )
        all_results['rna_only'].append(res_rna)

    # Save best baseline checkpoint for FBA
    ckpt_path = os.path.join(output_dir, 'best_baseline_model.pt')
    torch.save(best_baseline_state, ckpt_path)
    logger.info(f"Saved best baseline checkpoint to {ckpt_path}")

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

    out_path = os.path.join(output_dir, 'proteomics_ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("PROTEOMICS ABLATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Baseline (RNA+Prot): F1={summary['baseline']['F1_mean']:.4f}, "
                f"AUROC={summary['baseline']['AUROC_mean']:.4f}, "
                f"AUPRC={summary['baseline']['AUPRC_mean']:.4f}")
    logger.info(f"RNA-only:            F1={summary['rna_only']['F1_mean']:.4f}, "
                f"AUROC={summary['rna_only']['AUROC_mean']:.4f}, "
                f"AUPRC={summary['rna_only']['AUPRC_mean']:.4f}")
    logger.info(f"Delta:               F1={summary['delta']['F1']:+.4f}, "
                f"AUROC={summary['delta']['AUROC']:+.4f}, "
                f"AUPRC={summary['delta']['AUPRC']:+.4f}")
    logger.info(f"Results saved to {out_path}")

    return summary, ckpt_path


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: FBA EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════
def run_fba_expansion(args, cfg, device, train_ids, val_ids, test_ids, S, ckpt_path):
    logger.info("\n" + "#"*60)
    logger.info("# PART 2: FBA EXPANSION (ALL TEST PATIENTS)")
    logger.info("#"*60)

    try:
        import cobra
        from cobra.io import read_sbml_model
    except ImportError:
        logger.error("COBRApy not installed. Run: pip install cobra")
        return None

    output_dir = os.path.join(args.output_dir, 'fba_expansion')
    os.makedirs(output_dir, exist_ok=True)

    # Load Recon3D
    recon3d_path = args.recon3d_xml
    if not os.path.exists(recon3d_path):
        for p in ['/Users/thiptanawat/Documents/GitHub/MetaGNN/data/raw/recon3d_model.xml',
                   os.path.join(args.data_root, 'recon3d_model.xml')]:
            if os.path.exists(p):
                recon3d_path = p
                break
    logger.info(f"Loading Recon3D from {recon3d_path}...")
    recon3d = read_sbml_model(recon3d_path)
    logger.info(f"  {len(recon3d.reactions)} reactions, {len(recon3d.metabolites)} metabolites")

    # Load trained model
    model = MetaGNN(
        rxn_in_dim=cfg['rxn_in_dim'], met_in_dim=cfg['met_in_dim'],
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg['dropout'],
    ).to(device)

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded trained checkpoint from {ckpt_path}")
    else:
        logger.error(f"No checkpoint at {ckpt_path}! Cannot run FBA.")
        return None

    # Load test dataset and generate predictions
    test_ds = MetaGNNDataset(args.data_root, test_ids)
    model.eval()
    patient_predictions = {}
    with torch.no_grad():
        for i in range(len(test_ds)):
            data = test_ds[i].to(device)
            s_r = model(
                x_dict={'reaction': data['reaction'].x, 'metabolite': data['metabolite'].x},
                edge_index_dict={rel: data[rel].edge_index for rel in data.edge_types},
            )
            pid = test_ds.patient_ids[i]
            patient_predictions[pid] = s_r.cpu().numpy()

    logger.info(f"Generated predictions for {len(patient_predictions)} test patients")

    # Prediction distribution stats
    all_preds = np.concatenate(list(patient_predictions.values()))
    logger.info(f"Prediction stats: mean={all_preds.mean():.4f}, std={all_preds.std():.4f}, "
                f"min={all_preds.min():.4f}, max={all_preds.max():.4f}")
    logger.info(f"Active (>={args.fba_threshold}): {(all_preds >= args.fba_threshold).mean():.1%}")

    # FBA viability test
    def test_fba_viability(cobra_model, active_scores, threshold):
        with cobra_model as m:
            inactive_count = 0
            for i, rxn in enumerate(m.reactions):
                if i < len(active_scores):
                    if active_scores[i] < threshold:
                        rxn.bounds = (0, 0)
                        inactive_count += 1
            try:
                sol = m.optimize()
                if sol.status == 'optimal':
                    return sol.objective_value, inactive_count
                else:
                    return 0.0, inactive_count
            except Exception:
                return 0.0, inactive_count

    # Run FBA for all test patients
    threshold = args.fba_threshold
    logger.info(f"\nRunning FBA viability for {len(patient_predictions)} patients at τ={threshold}...")
    patient_results = []

    for pid, scores in patient_predictions.items():
        t0 = time.time()
        biomass, n_inactive = test_fba_viability(recon3d, scores, threshold)
        viable = biomass > 1e-6
        dt = time.time() - t0
        result = {
            'patient_id': pid,
            'biomass_flux': float(biomass),
            'viable': viable,
            'n_inactive': int(n_inactive),
            'n_active': int(len(scores) - n_inactive),
            'fba_time_s': round(dt, 2),
        }
        patient_results.append(result)
        status = '✓' if viable else '✗'
        logger.info(f"  {status} {pid}: biomass={biomass:.4f}, inactive={n_inactive}/{len(scores)}, time={dt:.1f}s")

    # Random baseline (100 MC trials)
    logger.info("\nRunning random baseline (100 Monte Carlo trials)...")
    n_rxns_model = len(list(patient_predictions.values())[0])
    active_ratio = float(np.mean([
        (scores >= threshold).mean() for scores in patient_predictions.values()
    ]))
    random_viable = 0
    random_biomasses = []
    n_trials = 100
    for t in range(n_trials):
        rand_scores = np.random.random(n_rxns_model)
        rand_binary = (rand_scores < active_ratio).astype(float)
        biomass, _ = test_fba_viability(recon3d, rand_binary, 0.5)
        random_biomasses.append(biomass)
        if biomass > 1e-6:
            random_viable += 1
        if (t + 1) % 25 == 0:
            logger.info(f"  Trial {t+1}/{n_trials}: {random_viable}/{t+1} viable")

    # Summary
    n_viable = sum(1 for r in patient_results if r['viable'])
    biomass_vals = [r['biomass_flux'] for r in patient_results if r['viable']]

    summary = {
        'n_patients': len(patient_results),
        'n_viable': n_viable,
        'viability_rate': n_viable / len(patient_results),
        'biomass_range': [float(min(biomass_vals)), float(max(biomass_vals))] if biomass_vals else [0, 0],
        'biomass_mean': float(np.mean(biomass_vals)) if biomass_vals else 0,
        'biomass_std': float(np.std(biomass_vals)) if biomass_vals else 0,
        'threshold': threshold,
        'prediction_stats': {
            'mean': float(all_preds.mean()),
            'std': float(all_preds.std()),
            'active_ratio': float((all_preds >= threshold).mean()),
        },
        'random_baseline': {
            'viability_rate': random_viable / n_trials,
            'n_trials': n_trials,
            'mean_biomass': float(np.mean(random_biomasses)),
        },
        'per_patient': patient_results,
    }

    out_path = os.path.join(output_dir, 'fba_expansion_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"FBA EXPANSION SUMMARY ({len(patient_results)} test patients)")
    logger.info(f"{'='*60}")
    logger.info(f"MetaGNN viable: {n_viable}/{len(patient_results)} ({100*n_viable/len(patient_results):.0f}%)")
    if biomass_vals:
        logger.info(f"Biomass range: {min(biomass_vals):.4f} – {max(biomass_vals):.4f} mmol/gDW/h")
        logger.info(f"Biomass mean: {np.mean(biomass_vals):.4f} ± {np.std(biomass_vals):.4f}")
    logger.info(f"Random baseline: {random_viable}/{n_trials} viable ({100*random_viable/n_trials:.0f}%)")
    logger.info(f"Results saved to {out_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='MetaGNN Missing Experiments')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--recon3d_xml', type=str,
                        default='/Users/thiptanawat/Documents/GitHub/MetaGNN/data/raw/recon3d_model.xml')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--fba_threshold', type=float, default=0.15)
    parser.add_argument('--skip_ablation', action='store_true',
                        help='Skip ablation, use existing checkpoint for FBA')
    args = parser.parse_args()

    cfg = dict(DEFAULTS)

    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Config: hidden_dim={cfg['hidden_dim']}, n_layers={cfg['n_layers']}, "
                f"heads={cfg['heads']}, batch_size={cfg['batch_size']}, "
                f"n_epochs={cfg['n_epochs']}, patience={cfg['patience']}")

    # Load data splits (same as prepare_and_run_experiments.py)
    meta_df = pd.read_csv(os.path.join(args.data_root, 'clinical_metadata.tsv'), sep='\t')
    ids = meta_df['tcga_barcode'].tolist()
    strat = meta_df['msi_status'].fillna('Unknown').tolist()
    train_ids, tmp_ids, _, tmp_strat = train_test_split(
        ids, strat, train_size=0.70, stratify=strat, random_state=cfg['seed'])
    val_ids, test_ids = train_test_split(
        tmp_ids, train_size=0.5, stratify=tmp_strat, random_state=cfg['seed'])
    logger.info(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Load stoichiometric matrix
    with h5py.File(os.path.join(args.data_root, 'recon3d_stoich.h5'), 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32)

    total_start = time.time()
    ckpt_path = os.path.join(args.output_dir, 'ablation_proteomics', 'best_baseline_model.pt')

    # Part 1: Proteomics Ablation
    if not args.skip_ablation:
        ablation_summary, ckpt_path = run_ablation(
            args, cfg, device, train_ids, val_ids, test_ids, S)
    else:
        logger.info("Skipping ablation (--skip_ablation flag set)")

    # Part 2: FBA Expansion
    fba_summary = run_fba_expansion(
        args, cfg, device, train_ids, val_ids, test_ids, S, ckpt_path)

    total_time = time.time() - total_start
    logger.info(f"\n{'#'*60}")
    logger.info(f"ALL EXPERIMENTS COMPLETE — Total time: {total_time/3600:.1f}h")
    logger.info(f"{'#'*60}")


if __name__ == '__main__':
    main()
