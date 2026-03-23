# MetaGNN Results Analysis Notebooks

This directory contains four Jupyter notebooks for analysing MetaGNN results and reproducing figures from the manuscript.

## Notebooks Overview

### 1. 01_Training_Analysis.ipynb

Reproduces the training convergence analysis across two stages:

- **Stage 1**: Pre-training loss curve on 98 HMA tissue-specific GEMs
- **Stage 2**: Fine-tuning metrics (loss, F1, AUROC) on the TCGA-CRC patient cohort
- Early stopping visualisation at epoch 141 (best validation F1 = 0.806, AUROC = 0.866)
- Final best metrics summary

**Data Source**: `../results/training_logs/` (stage1_pretrain_log.csv, stage2_finetune_log.csv, hyperparameters.json)

**Generated Figures**: fig_stage1_loss.png, fig_stage2_metrics.png

---

### 2. 02_Benchmark_Evaluation.ipynb

Performance comparison across five methods with multiple metrics:

- Grouped bar chart: F1-score, AUROC, Task Completion rates
- Per-cell-line essential gene AUROC analysis (DepMap CRISPR, 61 CRC cell lines)
- Benchmark summary statistics

**Data Source**: `../results/benchmark_evaluation/` (per_patient_f1_scores.csv, per_patient_auroc_scores.csv, benchmark_summary.csv, etc.)

**Generated Figures**: fig_benchmark_comparison.png, fig_essential_gene_auroc.png

---

### 3. 03_Uncertainty_Calibration.ipynb

Analysis of uncertainty quantification using Monte Carlo Dropout (T=30 passes):

- Reliability diagrams comparing MetaGNN vs. GP baseline
- Expected Calibration Error (ECE) comparison
- MC Dropout sample distributions from example reactions
- Uncertainty (σ) value distribution
- Transfer learning efficiency curves (pre-trained vs. from scratch)

**Data Source**: `../results/uncertainty_calibration/` and `../results/statistical_tests/`

**Generated Figures**: fig_reliability_diagrams.png, fig_ece_comparison.png, fig_mc_dropout_examples.png, fig_uncertainty_distribution.png, fig_transfer_learning.png

---

### 4. 04_Pathway_Activity_Analysis.ipynb

Metabolic network analysis: reaction activity and pathway distributions:

- Bimodal histogram of reaction activity scores
- Pathway-stratified box plots showing activity variation across 88 subsystems
- FBA feasibility metrics: 32/33 test patients (97%) sustain non-zero biomass flux
- Metabolic task completion rates across patients

**Data Source**: `../results/pathway_analysis/` and `../results/fba_feasibility/`

**Generated Figures**: fig_activity_distribution.png, fig_pathway_activity.png, fig_fba_analysis.png, fig_task_completion.png

---

## Data Location

All data files are loaded from relative paths:

```
../results/
├── training_logs/
├── benchmark_evaluation/
├── uncertainty_calibration/
├── per_patient_predictions/
├── pathway_analysis/
├── fba_feasibility/
└── statistical_tests/
```

## Styling

All notebooks use publication-quality settings:

- **Font**: Times New Roman
- **Figure DPI**: 150
- **Colour scheme**: MetaGNN (#2E75B6 blue), GIMME (#A5A5A5 grey), iMAT (#9DC3E6 light blue), CORDA (#BDD7EE lighter blue), tINIT (#70AD47 green)

## Usage

Each notebook is self-contained and can be run independently:

```bash
jupyter lab 01_Training_Analysis.ipynb
jupyter nbconvert --to html 01_Training_Analysis.ipynb
```

## Requirements

```
jupyter>=7.0
nbconvert>=7.0
pandas>=1.3
numpy>=1.20
matplotlib>=3.3
seaborn>=0.11
scipy>=1.7
```

## Authors

Thiptanawat Phongwattana, Jonathan H. Chan
School of Information Technology, KMUTT
