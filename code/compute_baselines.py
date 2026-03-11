#!/usr/bin/env python3
"""
compute_baselines.py — Compute real baseline metrics for MetaGNN paper.

Evaluates three baselines against the same expression-thresholded consensus
labels used to train/evaluate MetaGNN, on the same test split:

1. Random Classifier: analytical (no data needed)
2. Majority Class: always predict active
3. GPR Threshold (no GNN): per-patient expression threshold → per-reaction
   prediction, evaluated against consensus labels on test patients

Run from the MetaGNN repo root:
    conda activate metagnn
    python src/compute_baselines.py

Outputs: results/baseline_results.json
"""

import os, json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# ── Paths (same as improved_pipeline.py) ──────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
RAW  = os.path.join(DATA, 'raw')
PROC = os.path.join(DATA, 'processed')
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

# ── 1. Load consensus labels ──────────────────────────────────────────
import torch
labels_path = os.path.join(PROC, 'hma_labels_thresholded.pt')
consensus_labels = torch.load(labels_path, map_location='cpu').numpy()  # shape: (10600,)
n_reactions = len(consensus_labels)
n_active = int(consensus_labels.sum())
n_inactive = n_reactions - n_active
frac_active = n_active / n_reactions
print(f"Consensus labels: {n_active} active / {n_inactive} inactive ({frac_active:.3f})")

# ── 2. Load splits ────────────────────────────────────────────────────
splits_path = os.path.join(PROC, 'splits.json')
with open(splits_path) as f:
    splits = json.load(f)
test_ids = splits['test']  # list of patient UUIDs or indices
print(f"Test set: {len(test_ids)} patients")

# ── 3. Load RNA-seq data ──────────────────────────────────────────────
print("Loading RNA-seq data...")
rna_path = os.path.join(RAW, 'tcga_rna_seq.csv')
rna = pd.read_csv(rna_path, index_col=0)
print(f"RNA-seq shape: {rna.shape}")

# ── 4. Load Recon3D model for GPR rules ───────────────────────────────
print("Loading Recon3D model via COBRApy...")
import cobra
model_path = os.path.join(RAW, 'recon3d_model.xml')
model = cobra.io.read_sbml_model(model_path)
print(f"Recon3D: {len(model.reactions)} reactions, {len(model.genes)} genes")

# ── 5. Build gene-to-reaction mapping ─────────────────────────────────
# Map gene symbols from cobra gene.name to RNA-seq row indices
rna_genes = set(rna.index)
gene_symbol_map = {}  # gene.id -> gene symbol (gene.name)
for g in model.genes:
    if g.name and g.name in rna_genes:
        gene_symbol_map[g.id] = g.name

print(f"Mapped {len(gene_symbol_map)} Recon3D genes to RNA-seq symbols")

# Compute cohort median per gene (used for thresholding)
cohort_median = rna.median(axis=1)  # median TPM per gene across all patients

# ── 6. GPR Threshold baseline: per-patient, per-reaction ──────────────
# For each test patient, for each reaction with GPR:
#   - Get encoding genes
#   - A reaction is predicted active if ANY gene > cohort median (OR logic)
#   - Reactions without GPR → predict active (same default as label generation)
# Then compare per-patient predictions to consensus labels

print("Computing GPR Threshold baseline on test patients...")

# Get patient column mapping
# Patient features use UUIDs; splits.json has UUIDs
# RNA-seq columns are patient identifiers — need to map
rna_columns = list(rna.columns)

# Map test UUIDs to RNA-seq column indices
# From improved_pipeline.py: sorted UUID files mapped to RNA-seq by position
patient_feature_dir = os.path.join(PROC, 'patient_features_v2')
all_uuids = sorted([f.replace('.pt', '') for f in os.listdir(patient_feature_dir) if f.endswith('.pt')])
uuid_to_rna_idx = {uid: i for i, uid in enumerate(all_uuids) if i < len(rna_columns)}

test_rna_indices = []
for uid in test_ids:
    if uid in uuid_to_rna_idx:
        idx = uuid_to_rna_idx[uid]
        if idx < len(rna_columns):
            test_rna_indices.append(idx)

print(f"Mapped {len(test_rna_indices)} test patients to RNA-seq columns")

# For each reaction, get its GPR gene symbols
reaction_genes = []  # list of (rxn_index, [gene_symbols])
for i, rxn in enumerate(model.reactions):
    genes = rxn.genes
    if genes:
        symbols = [gene_symbol_map[g.id] for g in genes if g.id in gene_symbol_map]
        reaction_genes.append((i, symbols))
    else:
        reaction_genes.append((i, []))

# Compute per-patient predictions
all_patient_preds = []   # list of arrays, each (n_reactions,)
all_patient_scores = []  # continuous scores for AUROC

for rna_idx in test_rna_indices:
    col = rna_columns[rna_idx]
    patient_expr = rna[col]  # Series indexed by gene symbol

    preds = np.ones(n_reactions)   # default: active
    scores = np.ones(n_reactions) * 0.5  # default score

    for rxn_i, symbols in reaction_genes:
        if not symbols:
            preds[rxn_i] = 1  # no GPR → default active
            scores[rxn_i] = 0.7  # above threshold
            continue

        # OR logic: active if ANY gene > median
        gene_above = []
        gene_ratios = []
        for sym in symbols:
            if sym in patient_expr.index and sym in cohort_median.index:
                expr_val = patient_expr[sym]
                med_val = cohort_median[sym]
                if med_val > 0:
                    gene_ratios.append(expr_val / med_val)
                    gene_above.append(expr_val > med_val)
                else:
                    gene_ratios.append(1.0 if expr_val > 0 else 0.0)
                    gene_above.append(expr_val > 0)

        if gene_above:
            preds[rxn_i] = 1.0 if any(gene_above) else 0.0
            # Continuous score: max ratio across genes (OR logic)
            scores[rxn_i] = max(gene_ratios) if gene_ratios else 0.5
        else:
            preds[rxn_i] = 1  # no matched genes → default active
            scores[rxn_i] = 0.7

    all_patient_preds.append(preds)
    all_patient_scores.append(scores)

# ── 7. Evaluate all baselines ─────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE EVALUATION RESULTS")
print("="*60)

results = {}
y_true = consensus_labels

# --- Baseline 1: Random Classifier (analytical) ---
# For a random classifier on a dataset with p_active = frac_active:
# Expected precision = frac_active, recall = frac_active (for class 1)
# Actually: random predicts class 1 with prob 0.5
# TP = 0.5 * n_active, FP = 0.5 * n_inactive, FN = 0.5 * n_active
# Precision = n_active / (n_active + n_inactive) = frac_active
# Recall = 0.5
# F1 = 2 * frac * 0.5 / (frac + 0.5)
# AUROC = 0.5 (by definition)
# Better: actually generate random predictions and compute
np.random.seed(42)
n_random_trials = 1000
random_f1s, random_precs, random_recs = [], [], []
for _ in range(n_random_trials):
    y_rand = np.random.binomial(1, 0.5, n_reactions).astype(float)
    random_f1s.append(f1_score(y_true, y_rand))
    random_precs.append(precision_score(y_true, y_rand))
    random_recs.append(recall_score(y_true, y_rand))

results['random'] = {
    'f1': float(np.mean(random_f1s)),
    'f1_std': float(np.std(random_f1s)),
    'auroc': 0.500,  # by definition
    'precision': float(np.mean(random_precs)),
    'recall': float(np.mean(random_recs)),
}
print(f"\nRandom Classifier:")
print(f"  F1 = {results['random']['f1']:.4f} ± {results['random']['f1_std']:.4f}")
print(f"  AUROC = 0.5000 (by definition)")
print(f"  Precision = {results['random']['precision']:.4f}")
print(f"  Recall = {results['random']['recall']:.4f}")

# --- Baseline 2: Majority Class (always predict active) ---
y_majority = np.ones(n_reactions)
results['majority'] = {
    'f1': float(f1_score(y_true, y_majority)),
    'auroc': 0.500,  # no discriminative power
    'precision': float(precision_score(y_true, y_majority)),
    'recall': float(recall_score(y_true, y_majority)),
}
print(f"\nMajority Class (all active):")
print(f"  F1 = {results['majority']['f1']:.4f}")
print(f"  AUROC = 0.5000 (no discriminative power)")
print(f"  Precision = {results['majority']['precision']:.4f}")
print(f"  Recall = {results['majority']['recall']:.4f}")

# --- Baseline 3: GPR Threshold (no GNN) ---
# Average metrics across test patients
gpr_f1s, gpr_aurocs, gpr_precs, gpr_recs = [], [], [], []
for preds, scores in zip(all_patient_preds, all_patient_scores):
    gpr_f1s.append(f1_score(y_true, preds))
    gpr_precs.append(precision_score(y_true, preds))
    gpr_recs.append(recall_score(y_true, preds))
    try:
        gpr_aurocs.append(roc_auc_score(y_true, scores))
    except:
        pass

results['gpr_threshold'] = {
    'f1': float(np.mean(gpr_f1s)),
    'f1_std': float(np.std(gpr_f1s)),
    'auroc': float(np.mean(gpr_aurocs)) if gpr_aurocs else 0.5,
    'auroc_std': float(np.std(gpr_aurocs)) if gpr_aurocs else 0.0,
    'precision': float(np.mean(gpr_precs)),
    'recall': float(np.mean(gpr_recs)),
    'n_patients': len(test_rna_indices),
}
print(f"\nGPR Threshold (no GNN):")
print(f"  F1 = {results['gpr_threshold']['f1']:.4f} ± {results['gpr_threshold']['f1_std']:.4f}")
print(f"  AUROC = {results['gpr_threshold']['auroc']:.4f} ± {results['gpr_threshold']['auroc_std']:.4f}")
print(f"  Precision = {results['gpr_threshold']['precision']:.4f}")
print(f"  Recall = {results['gpr_threshold']['recall']:.4f}")

# --- MetaGNN (from results_summary_v2.json) ---
metagnn_path = os.path.join(RESULTS, 'results_summary_v2.json')
if os.path.exists(metagnn_path):
    with open(metagnn_path) as f:
        mg = json.load(f)
    results['metagnn'] = {
        'f1': mg['test']['f1_mean'],
        'f1_std': mg['test']['f1_std'],
        'auroc': mg['test']['auroc'],
        'precision': mg['test']['precision'],
        'recall': mg['test']['recall'],
    }
    print(f"\nMetaGNN (Ours) — from results_summary_v2.json:")
    print(f"  F1 = {results['metagnn']['f1']:.4f} ± {results['metagnn']['f1_std']:.4f}")
    print(f"  AUROC = {results['metagnn']['auroc']:.4f}")
    print(f"  Precision = {results['metagnn']['precision']:.4f}")
    print(f"  Recall = {results['metagnn']['recall']:.4f}")

# ── 8. Save results ───────────────────────────────────────────────────
output_path = os.path.join(RESULTS, 'baseline_results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# ── 9. Summary table ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Method':<30} {'F1':>8} {'AUROC':>8} {'Prec':>8} {'Recall':>8}")
print(f"{'-'*60}")
print(f"{'Random Classifier':<30} {results['random']['f1']:>8.4f} {results['random']['auroc']:>8.4f} {results['random']['precision']:>8.4f} {results['random']['recall']:>8.4f}")
print(f"{'Majority Class':<30} {results['majority']['f1']:>8.4f} {results['majority']['auroc']:>8.4f} {results['majority']['precision']:>8.4f} {results['majority']['recall']:>8.4f}")
print(f"{'GPR Threshold (no GNN)':<30} {results['gpr_threshold']['f1']:>8.4f} {results['gpr_threshold']['auroc']:>8.4f} {results['gpr_threshold']['precision']:>8.4f} {results['gpr_threshold']['recall']:>8.4f}")
if 'metagnn' in results:
    print(f"{'MetaGNN (Ours)':<30} {results['metagnn']['f1']:>8.4f} {results['metagnn']['auroc']:>8.4f} {results['metagnn']['precision']:>8.4f} {results['metagnn']['recall']:>8.4f}")
print(f"{'='*60}")
print("\nDone! Use these numbers to update the manuscript.")
