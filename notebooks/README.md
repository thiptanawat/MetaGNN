# MethodsX Manuscript Results Analysis Notebooks

This directory contains four comprehensive Jupyter notebooks for analyzing MetaGNN results and comparing performance against baseline methods (GIMME, iMAT, CORDA, tINIT).

## Notebooks Overview

### 1. **01_Training_Analysis.ipynb**
Reproduces the training convergence analysis for MetaGNN across two stages:
- **Stage 1**: Pre-training loss curve on large-scale metabolic network corpus
- **Stage 2**: Fine-tuning metrics (loss, F1-score, AUROC) on patient-specific benchmarks
- Early stopping point visualization at peak validation AUROC
- Final best metrics summary

**Data Source**: `../results/training_logs/`
- stage1_pretrain_log.csv
- stage2_finetune_log.csv
- hyperparameters.json

**Generated Figures**:
- fig_stage1_loss.png: Pre-training convergence
- fig_stage2_metrics.png: Fine-tuning metrics (4-panel subplot)

---

### 2. **02_Benchmark_Evaluation.ipynb**
Comprehensive performance comparison across five methods with multiple metrics:
- Grouped bar chart: F1-score, AUROC, Task Completion rates
- Per-cell-line essential gene AUROC analysis
- Benchmark summary statistics

**Data Source**: `../results/benchmark_evaluation/`
- per_patient_f1_scores.csv
- per_patient_auroc_scores.csv
- per_patient_task_completion.csv
- essential_gene_auroc_per_cellline.csv
- benchmark_summary.csv

**Generated Figures**:
- fig_benchmark_comparison.png: Method performance comparison
- fig_essential_gene_auroc.png: Cell-line specific accuracy

---

### 3. **03_Uncertainty_Calibration.ipynb**
Analysis of uncertainty quantification using MC Dropout and calibration:
- Reliability diagrams comparing MetaGNN vs. GP baseline
- Expected Calibration Error (ECE) comparison
- MC Dropout sample distributions from example reactions
- Uncertainty (sigma) value distribution
- Transfer learning efficiency curves (fine-tuned vs. from scratch)

**Data Source**: `../results/uncertainty_calibration/` and `../results/statistical_tests/`
- calibration_data.csv
- gp_baseline_calibration.csv
- ece_computation.csv
- mc_dropout_samples_example.csv
- transfer_learning_curves.csv

**Generated Figures**:
- fig_reliability_diagrams.png: Calibration quality comparison
- fig_ece_comparison.png: ECE bar chart
- fig_mc_dropout_examples.png: Example MC predictions
- fig_uncertainty_distribution.png: Sigma distribution histogram
- fig_transfer_learning.png: Learning curves with error bands

---

### 4. **04_Pathway_Activity_Analysis.ipynb**
Metabolic network analysis: reaction activity and pathway distributions:
- Bimodal histogram of reaction activity scores
- Pathway-stratified box plots showing activity variation
- FBA feasibility metrics (reactions retained, biomass flux)
- Metabolic task completion rates across patients

**Data Source**: `../results/pathway_analysis/` and `../results/fba_feasibility/`
- reaction_activity_distribution.csv
- pathway_activity_per_patient.csv
- per_patient_fba_summary.csv
- metabolic_task_completion_detail.csv

**Generated Figures**:
- fig_activity_distribution.png: Reaction activity histogram
- fig_pathway_activity.png: Pathway-stratified box plot
- fig_fba_analysis.png: FBA metrics (4-panel subplot)
- fig_task_completion.png: Task completion rates

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

## Styling & Publication Quality

All notebooks use:
- **Font**: Times New Roman (publication standard)
- **Figure DPI**: 150 (high quality for manuscripts)
- **Color Scheme**:
  - MetaGNN: #2E75B6 (blue)
  - GIMME: #A5A5A5 (gray)
  - iMAT: #9DC3E6 (light blue)
  - CORDA: #BDD7EE (lighter blue)
  - tINIT: #70AD47 (green)

## Usage

Each notebook is self-contained and can be run independently:

```bash
# View in Jupyter Lab
jupyter lab 01_Training_Analysis.ipynb

# Export to HTML
jupyter nbconvert --to html 01_Training_Analysis.ipynb

# Export to PDF (requires pandoc)
jupyter nbconvert --to pdf 01_Training_Analysis.ipynb
```

## Key Findings Summary

1. **Training** (Notebook 1):
   - Effective pre-training convergence on large datasets
   - Strong fine-tuning adaptation with early stopping at optimal AUROC
   - Minimal train-val divergence indicates good generalization

2. **Benchmark Performance** (Notebook 2):
   - MetaGNN outperforms all baselines across F1, AUROC, and task completion
   - High consistency across patients (low variability)
   - Excellent cross-cell-line generalization for essential gene prediction

3. **Uncertainty Quantification** (Notebook 3):
   - Superior calibration (lower ECE) vs. GP baseline
   - MC Dropout provides reliable uncertainty estimates
   - Pre-training dramatically improves data efficiency

4. **Metabolic Feasibility** (Notebook 4):
   - Clear bimodal distribution in reaction activity scores
   - Pathway-specific activity patterns reflect biological heterogeneity
   - Patient-specific models maintain essential metabolic functions

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

Generated for MethodsX manuscript: MetaGNN - A graph neural network approach for patient-specific metabolic network inference.
