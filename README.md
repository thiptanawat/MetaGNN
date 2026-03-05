# MetaGNN - MethodsX Repository

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
├── README.md                              # This file
├── raw_data/                              # CSV data underlying each figure
│   ├── fig2_performance_comparison.csv    # Fig 2: benchmark metrics (all methods)
│   ├── fig3_transfer_learning_curves.csv  # Fig 3 left: F1 vs training set size
│   ├── fig3_uncertainty_calibration.csv   # Fig 3 right: ECE reliability diagram
│   ├── fig4_reaction_activity_scores_summary.csv  # Fig 4 left: score distribution
│   ├── fig4_reaction_activity_scores_sample200.csv# Fig 4: per-reaction scores (n=200)
│   ├── fig4_pathway_activity_scores.csv   # Fig 4 right: per-pathway scores
│   └── fig4_pathway_summary_stats.csv     # Fig 4 right: pathway summary statistics
├── code/                                  # Full source code
│   ├── requirements.txt                   # Python dependencies
│   ├── 01_metagnn_model.py                # H-GAT model architecture (GATv2Conv)
│   ├── 02_data_loader.py                  # HeteroData dataset + data loading
│   ├── 03_train_metagnn.py                # Two-stage training pipeline
│   ├── 04_evaluate_metagnn.py             # Evaluation metrics + ECE
│   ├── 05_generate_figures.py             # All manuscript figures (Figs 1–4)
│   └── 06_regen_fig1_large.py             # Regenerate Fig 1 at high resolution
├── notebooks/                             # Jupyter notebooks for analysis
│   ├── 01_Training_Analysis.ipynb
│   ├── 02_Benchmark_Evaluation.ipynb
│   ├── 03_Uncertainty_Calibration.ipynb
│   └── 04_Pathway_Activity_Analysis.ipynb
├── results/                               # Pre-computed results
│   ├── INDEX.txt
│   ├── benchmark_evaluation/
│   ├── fba_feasibility/
│   ├── pathway_analysis/
│   ├── per_patient_predictions/
│   ├── statistical_tests/
│   ├── training_logs/
│   └── uncertainty_calibration/
└── references/
    └── reference_access_guide.md          # DOI links + download instructions
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

## Method Overview

MetaGNN is a heterogeneous Graph Attention Network (GATv2) framework that reconstructs
patient-specific genome-scale metabolic models (GEMs) from clinical multi-omics data.
The method operates on the Recon3D v3 human metabolic network and predicts per-reaction
binary activity labels by integrating transcriptomic and proteomic features through
topology-aware message passing.

### Key Architecture

- **Graph backbone:** Recon3D v3 (10,600 reactions, 5,835 metabolites, 2,248 genes)
- **Edge structure:** 40,425 stoichiometric edges (20,512 substrate_of + 19,913 produces)
- **Model:** 2 GATv2 layers, 4 attention heads, 128 hidden dimensions
- **Parameters:** 143,489 trainable parameters
- **Training:** AdamW optimiser (lr = 5 x 10^-4), early stopping with patience 20

### Dataset

- **Patients:** 690 TCGA-CRC patients with RNA-seq; 95 with matched CPTAC TMT proteomics
- **Labels:** Expression-thresholded consensus via GPR OR-logic + cohort majority vote
  - 7,434 active / 3,166 inactive reactions (70.1% active)
- **Split:** 482 train / 104 validation / 104 test (patient-level, no leakage)

---

## Raw Data Description

### Figure 2 — Benchmark Comparison

`fig2_performance_comparison.csv`

Compares MetaGNN against baseline approaches on test-set reaction activity prediction.

**Test-set results:**

| Method | F1 Score | AUROC | Precision | Recall |
|--------|----------|-------|-----------|--------|
| Random Classifier | 0.584 | 0.500 | 0.702 | 0.500 |
| Majority Class | 0.824 | 0.500 | 0.701 | 0.700 |
| GPR Threshold (no GNN) | 0.869 | 0.468 | 0.822 | 0.925 |
| **MetaGNN (Ours)** | **0.814 +/- 0.016** | **0.874** | **0.919** | **0.752** |

MetaGNN achieves the highest AUROC (0.874) and Precision (0.919), demonstrating
superior discriminative ability. The GPR Threshold baseline achieves high F1 through
recall but has an AUROC below random chance (0.468), indicating poor calibration.

### Figure 3 — Transfer Learning & Uncertainty Calibration

`fig3_transfer_learning_curves.csv` — F1 score as a function of fine-tuning cohort size,
comparing HMA-pre-trained MetaGNN against training from scratch.

`fig3_uncertainty_calibration.csv` — Expected calibration error (ECE) reliability diagram
comparing MetaGNN MC-Dropout uncertainty against a Gaussian Process baseline.

### Figure 4 — Reaction Activity Score Distribution

`fig4_reaction_activity_scores_summary.csv` — Histogram of predicted reaction activity
scores across all test reactions, binned into 40 equal-width intervals.

`fig4_pathway_activity_scores.csv` — Per-reaction predicted scores for 5 representative
metabolic pathways.

`fig4_pathway_summary_stats.csv` — Summary statistics (mean, median, Q25, Q75, std,
% above threshold) for each pathway.

---

## Reproducibility Notes

All figures were generated deterministically from the CSV data in `raw_data/` using
`code/05_generate_figures.py`. Random seeds are documented in each CSV file header
and in the generation code.

---

## Licence

Code: MIT Licence
Raw data: CC-BY 4.0

---
