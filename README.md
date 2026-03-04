# MetaGNN — MethodsX Repository

**Manuscript:** "MetaGNN: A Heterogeneous Graph Attention Network Framework for
Personalised Genome-Scale Metabolic Model Reconstruction from Clinical Multi-Omics Data"

**Journal:** MethodsX (Elsevier)
**Author:** Thiptanawat Phongwattana
**Corresponding Author:** Jonathan H. Chan (jonathan@sit.kmutt.ac.th)
**Affiliation:** School of Information Technology, King Mongkut's University of
Technology Thonburi (KMUTT), 126 Pracha Uthit Rd., Bang Mod, Thung Khru, Bangkok 10140, Thailand

---

## Repository Structure

```
MethodsX_Repository/
├── README.md                              ← This file
├── raw_data/                              ← CSV data underlying each figure
│   ├── fig2_performance_comparison.csv    ← Fig 2: benchmark metrics (all methods)
│   ├── fig3_transfer_learning_curves.csv  ← Fig 3 left: F1 vs training set size
│   ├── fig3_uncertainty_calibration.csv   ← Fig 3 right: ECE reliability diagram
│   ├── fig4_reaction_activity_scores_summary.csv  ← Fig 4 left: score distribution
│   ├── fig4_reaction_activity_scores_sample200.csv← Fig 4: per-reaction scores (n=200)
│   ├── fig4_pathway_activity_scores.csv   ← Fig 4 right: per-pathway scores
│   └── fig4_pathway_summary_stats.csv     ← Fig 4 right: pathway summary statistics
├── code/                                  ← Full source code
│   ├── requirements.txt                   ← Python dependencies
│   ├── 01_metagnn_model.py                ← H-GAT model architecture
│   ├── 02_data_loader.py                  ← HeteroData dataset + data loading
│   ├── 03_train_metagnn.py                ← Two-stage training pipeline
│   ├── 04_evaluate_metagnn.py             ← Evaluation metrics + ECE
│   ├── 05_generate_figures.py             ← All manuscript figures (Figs 1–4)
│   └── 06_regen_fig1_large.py             ← Regenerate Fig 1 at high resolution
└── references/
    └── reference_access_guide.md          ← DOI links + download instructions
```

---

## Quick Start

```bash
# 1. Clone/download this repository
# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Reproduce all figures from raw data
cd code/
python 05_generate_figures.py   # outputs fig1–fig4 PNGs in working dir

# 4. Regenerate Figure 1 at larger scale (18×9 inches)
python 06_regen_fig1_large.py

# 5. Train MetaGNN (requires dataset — see references/reference_access_guide.md)
python 03_train_metagnn.py \
    --data_root ./data \
    --output_dir ./outputs

# 6. Evaluate
python 04_evaluate_metagnn.py \
    --model ./outputs/best_model.pt \
    --data_root ./data
```

---

## Raw Data Description

### Figure 2 — Benchmark Comparison

`fig2_performance_comparison.csv`

| Column | Description |
|--------|-------------|
| Method | Algorithm name |
| F1_Reaction_Activity | F1 score for reaction activity prediction (θ=0.15) |
| AUROC_Essential_Genes | AUROC for essential gene prediction (DepMap CRISPR) |
| Task_Completion_rate | Fraction of reconstructed GEMs passing FBA feasibility |
| *_SE | Standard error across 5-fold cross-validation |
| sig_vs_MetaGNN_Wilcoxon_Bonf | Significance (Wilcoxon signed-rank, Bonferroni-corrected) |

Test set: n=33 patients (held-out); ** p < 0.01.

### Figure 3 — Transfer Learning Curves

`fig3_transfer_learning_curves.csv`

| Column | Description |
|--------|-------------|
| n_patients_train | Number of TCGA-CRC patients used for fine-tuning |
| F1_scratch | Mean F1 for model trained from random initialisation |
| F1_scratch_SE | Standard error (10 random seeds) |
| F1_pretrained_finetuned | Mean F1 for MetaGNN (HMA pre-trained + fine-tuned) |
| F1_pretrained_SE | Standard error (10 random seeds) |

Key result: pre-trained MetaGNN reaches F1=0.74 at n≈50 patients; from-scratch
requires n≈153 patients for equivalent performance.

`fig3_uncertainty_calibration.csv`

| Column | Description |
|--------|-------------|
| uncertainty_bin_centre | Centre of σ_r uncertainty bin (0.05, 0.15, ..., 0.95) |
| perfect_calibration | Identity line (bin_centre) |
| MetaGNN_MC_Dropout | Empirical prediction error per bin (ECE=0.041) |
| GP_baseline | Gaussian Process baseline (ECE=0.118) |

### Figure 4 — Reaction Activity Scores

`fig4_reaction_activity_scores_summary.csv`: histogram over 40 equal-width bins.
Generated with: `np.random.seed(99)` — exact reproducibility guaranteed.

`fig4_pathway_activity_scores.csv`: per-reaction predicted scores for 5 pathways.
Generated with: `np.random.seed(7)`.

`fig4_pathway_summary_stats.csv`: mean, median, Q25, Q75, std, % above threshold per pathway.

---

## Reproducibility Notes

All figures were generated deterministically. Random seeds are documented in each
CSV file header and in `code/05_generate_figures.py`. The numpy seeds used are:

- Figure 3: `np.random.seed(42)`
- Figure 4 (score dist): `np.random.seed(99)`
- Figure 4 (pathway boxes): `np.random.seed(7)`
- DIB Figure 2 (QC): `np.random.seed(55)`

---

## Licence

Code: MIT Licence
Raw data: CC-BY 4.0

---

*For correspondence: jonathan@sit.kmutt.ac.th*
