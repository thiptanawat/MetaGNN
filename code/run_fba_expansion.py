#!/usr/bin/env python3
"""
FBA Expansion: Validate MetaGNN on all 33 test patients
========================================================
Expands the preliminary 10-patient FBA analysis to all 33 test patients.
Uses the trained model checkpoint to generate per-patient activity predictions,
then tests FBA viability for each.

Usage:
    conda activate metagnn
    python run_fba_expansion.py \
        --data_root ../data \
        --recon3d_xml /path/to/Recon3D.xml \
        --output_dir ../results/fba_expansion
"""

import os
import sys
import json
import argparse
import logging
import time

import numpy as np
import h5py
import torch
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from metagnn_model import MetaGNN
from data_loader import MetaGNNDataset, stratified_split


def run_fba_expansion(args):
    try:
        import cobra
        from cobra.io import read_sbml_model
    except ImportError:
        logger.error("COBRApy not installed. Run: pip install cobra")
        return

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cpu')  # FBA runs on CPU anyway
    logger.info(f"Using device: {device}")

    # ── Load Recon3D COBRA model ──────────────────────────────────────────────
    logger.info(f"Loading Recon3D from {args.recon3d_xml}...")
    recon3d = read_sbml_model(args.recon3d_xml)
    logger.info(f"  {len(recon3d.reactions)} reactions, {len(recon3d.metabolites)} metabolites")

    # ── Load data and splits ──────────────────────────────────────────────────
    meta_df = pd.read_csv(os.path.join(args.data_root, 'clinical_metadata.tsv'), sep='\t')
    train_ids, val_ids, test_ids = stratified_split(meta_df, seed=2024)
    logger.info(f"Test patients: {len(test_ids)}")

    # ── Load trained model ────────────────────────────────────────────────────
    model = MetaGNN(rxn_in_dim=2, met_in_dim=519, hidden_dim=256,
                    n_layers=3, heads=8, dropout=0.20).to(device)

    ckpt_path = os.path.join(args.data_root, '..', 'outputs', 'best_model.pt')
    if not os.path.exists(ckpt_path):
        # Try alternative paths
        for alt in ['../results/best_model.pt', '../best_model.pt',
                    os.path.join(args.data_root, 'best_model.pt')]:
            if os.path.exists(alt):
                ckpt_path = alt
                break
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning("No checkpoint found — using random weights for FBA test")

    # ── Load test dataset ─────────────────────────────────────────────────────
    test_ds = MetaGNNDataset(args.data_root, test_ids)

    # ── Get per-patient predictions ───────────────────────────────────────────
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

    # ── Load labels for reaction-to-Recon3D mapping ───────────────────────────
    labels = torch.load(os.path.join(args.data_root, 'activity_pseudolabels.pt'),
                        map_location='cpu')
    if isinstance(labels, torch.Tensor):
        label_vec = labels.numpy()
    else:
        label_vec = labels

    # Get reaction IDs: try to match prediction indices to COBRA model reactions
    recon3d_rxn_ids = [r.id for r in recon3d.reactions]
    n_model_rxns = len(label_vec)

    # ── FBA viability test ────────────────────────────────────────────────────
    def test_fba_viability(cobra_model, active_mask, threshold=0.15):
        """
        Constrain Recon3D based on predicted activity.
        Reactions predicted as inactive (score < threshold) have bounds set to 0.
        Returns biomass flux.
        """
        with cobra_model as m:
            inactive_count = 0
            for i, rxn in enumerate(m.reactions):
                if i < len(active_mask):
                    if active_mask[i] < threshold:
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

    # ── Run FBA for all test patients ─────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    threshold = 0.15

    logger.info(f"\nRunning FBA viability analysis for {len(patient_predictions)} patients at τ={threshold}...")
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
        logger.info(f"  {status} {pid}: biomass={biomass:.4f} mmol/gDW/h, "
                    f"inactive={n_inactive}, time={dt:.1f}s")

    # ── Random baseline (100 MC trials) ───────────────────────────────────────
    logger.info("\nRunning random baseline (100 Monte Carlo trials)...")
    active_ratio = float((label_vec > 0.5).mean())
    random_viable = 0
    random_biomasses = []
    n_trials = 100
    for t in range(n_trials):
        rand_scores = np.random.random(len(label_vec))
        rand_mask = (rand_scores >= (1 - active_ratio))  # match active ratio
        biomass, _ = test_fba_viability(recon3d, 1.0 - rand_mask.astype(float), threshold=0.5)
        random_biomasses.append(biomass)
        if biomass > 1e-6:
            random_viable += 1
        if (t + 1) % 25 == 0:
            logger.info(f"  Trial {t+1}/{n_trials}: {random_viable}/{t+1} viable")

    # ── Summary ───────────────────────────────────────────────────────────────
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
        'random_baseline': {
            'viability_rate': random_viable / n_trials,
            'n_trials': n_trials,
            'mean_biomass': float(np.mean(random_biomasses)),
        },
        'per_patient': patient_results,
    }

    out_path = os.path.join(args.output_dir, 'fba_expansion_results.json')
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
    logger.info(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--recon3d_xml', type=str,
                        default='/Users/thiptanawat/Documents/GitHub/MetaGNN/data/raw/recon3d_model.xml')
    parser.add_argument('--output_dir', type=str, default='../results/fba_expansion')
    args = parser.parse_args()
    run_fba_expansion(args)
