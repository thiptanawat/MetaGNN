# MetaGNN - MethodsX Repository

**Manuscript:** "MetaGNN: An Open-Source Heterogeneous Graph Attention Network Framework for
Topology-Aware Metabolic Network Reconstruction"

**Journal:** MethodsX (Elsevier)
**Author:** Thiptanawat Phongwattana
**Corresponding Author:** Jonathan H. Chan (jonathan@sit.kmutt.ac.th)
**Affiliation:** School of Information Technology, King Mongkut's University of
Technology Thonburi (KMUTT), 126 Pracha Uthit Rd., Bang Mod, Thung Khru, Bangkok 10140, Thailand

**Code Archive (Zenodo):** [https://doi.org/10.5281/zenodo.18903515](https://doi.org/10.5281/zenodo.18903515)
**Data Archive (Zenodo):** [https://doi.org/10.5281/zenodo.18903519](https://doi.org/10.5281/zenodo.18903519)

---

## Repository Structure

```
MethodsX_Repository/
├── README.md                              # This file
├── MetaGNN_MethodsX.tex                   # LaTeX manuscript source
├── MetaGNN_MethodsX.pdf                   # Compiled manuscript
├── fig1_architecture.png                  # Figure 1: Pipeline overview
├── fig2_training_curves.png               # Figure 2: Training dynamics
├── fig2_benchmarks.png                    # Figure 3: Benchmark comparison
├── raw_data/                              # CSV data underlying each figure
├── code/                                  # Full source code
│   ├── requirements.txt                   # Python dependencies
│   ├── environment.yml                    # Conda environment specification
│   ├── run.sh                             # Single-command full pipeline
│   ├── 01_metagnn_model.py                # H-GAT model architecture (GATv2Conv)
│   ├── 02_data_loader.py                  # HeteroData dataset + data loading
│   ├── 03_train_metagnn.py                # Training pipeline (BCE + mass-balance reg.)
│   ├── 04_evaluate_metagnn.py             # Evaluation metrics
│   ├── 05_generate_figures.py             # All manuscript figures
│   └── prepare_and_run_experiments.py     # Methodology validation experiments
├── notebooks/                             # Jupyter notebooks for analysis
├── results/                               # Pre-computed results
└── references/
    └── reference_access_guide.md          # DOI links + download instructions
```

---

## Quick Start

```bash
# 1. Clone/download this repository
# 2. Create environment
conda env create -f code/environment.yml
conda activate metagnn

# 3. Run complete pipeline
bash code/run.sh

# 4. Or run individual steps
python code/01_metagnn_model.py             # Sanity check
python code/02_data_loader.py               # Verify data loading
python code/03_train_metagnn.py --data_root data/
python code/04_evaluate_metagnn.py
python code/05_generate_figures.py
```

---

## Method Overview

MetaGNN is a heterogeneous Graph Attention Network (GATv2) framework that produces
patient-specific reaction activity *state* scores for genome-scale metabolic model
reconstruction from clinical multi-omics data. The method operates on the Recon3D v3
human metabolic network and predicts per-reaction binary activity labels by integrating
transcriptomic and proteomic features through topology-aware message passing.

### Architecture Configurations

| Configuration | Layers | Heads | Hidden Dim | Parameters | Purpose |
|--------------|--------|-------|------------|------------|---------|
| Default      | 3      | 8     | 256        | 143,489    | Full pipeline execution |
| Reduced-depth| 2      | 4     | 128        | 873,217*   | CPU training, methodology validation |

*The reduced-depth configuration has more parameters due to the wider input projection
from the 519-dimensional metabolite features (see manuscript Section 2.4.1).

### Graph Structure

- **Graph backbone:** Recon3D v3 (10,600 reactions, 5,835 metabolites, 2,248 genes)
- **Edge structure:** 40,425 stoichiometric edges (20,512 substrate_of + 19,913 produces)
- **Training loss:** BCE + mass-balance regularisation (λ_mb = 0.2)
- **Training:** AdamW optimiser (lr = 5×10⁻⁴), early stopping with patience 20

### Pipeline Execution

- **Patients:** 690 TCGA-CRC patients (RNA-seq); 95 with matched CPTAC TMT proteomics
- **Evaluation cohort:** 220 patients (154 train / 33 val / 33 test)
- **Labels:** Expression-thresholded consensus via GPR OR-logic + cohort majority vote
  - 7,434 active / 3,166 inactive reactions (70.1% active)

### Benchmark Results (220-patient evaluation, n=33 test patients, τ=0.15)

| Method | F1 Score | AUROC | Precision | Recall |
|--------|----------|-------|-----------|--------|
| Random Classifier | 0.584 | 0.500 | 0.701 | 0.500 |
| Majority Class | 0.824 | 0.500 | 0.701 | 1.000 |
| GPR Threshold (no GNN) | 0.869 | 0.468 | 0.822 | 0.925 |
| **MetaGNN (Ours)** | **0.796 ± 0.041** | **0.861 ± 0.030** | **0.665 ± 0.016** | **1.000 ± 0.000** |

MetaGNN achieves the highest AUROC (0.861), demonstrating that topology-aware message
passing learns generalisable reaction activity patterns beyond the label-generating heuristic.
The GPR Threshold achieves higher F1 by recapitulating the labelling rule, but its AUROC
(0.468) is below random, indicating no true discriminative power.

---

## Hardware Requirements

The complete pipeline executes in under 3 minutes on an Apple M4 Max (MPS backend).
CPU-only and CUDA GPU training are also supported.

---

## Licence

Code: MIT Licence
Raw data: CC-BY 4.0

---
