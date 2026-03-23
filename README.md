# MetaGNN

**Manuscript:** "MetaGNN: An Open-Source Heterogeneous Graph Attention Network Framework for Topology-Aware Reaction Activity Scoring Toward Metabolic Network Reconstruction"

**Journal:** MethodsX (Elsevier)
**Authors:** Thiptanawat Phongwattana, Jonathan H. Chan*
**Affiliation:** School of Information Technology, King Mongkut's University of Technology Thonburi (KMUTT), 126 Pracha Uthit Rd., Bang Mod, Thung Khru, Bangkok 10140, Thailand
*Corresponding author: jonathan@sit.kmutt.ac.th

**Code Archive (Zenodo):** [https://doi.org/10.5281/zenodo.18903515](https://doi.org/10.5281/zenodo.18903515)
**Data Archive (Zenodo):** [https://doi.org/10.5281/zenodo.18903519](https://doi.org/10.5281/zenodo.18903519)

---

## Repository Structure

```
MetaGNN/
├── README.md                              # This file
├── fig1_architecture.png                  # Figure 1: Architecture overview
├── fig2_graph_encoding.png                # Figure 2: Heterogeneous graph encoding
├── fig2_benchmarks.png                    # Figure 3: Benchmark comparison
├── raw_data/                              # CSV data underlying each figure
├── code/
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
git clone https://github.com/thiptanawat/MetaGNN.git
cd MetaGNN

# 2. Create environment
conda env create -f code/environment.yml
conda activate metagnn

# 3. Run complete pipeline (data download through inference)
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

MetaGNN is an open-source framework that reformulates reaction activity scoring as node classification on a heterogeneous bipartite graph encoding of the Recon3D v3 human metabolic network. Patient-specific transcriptomic features are propagated through GATv2 attention layers with type-specific message passing, producing per-reaction activity scores with calibrated uncertainty via Monte Carlo Dropout (T=30 passes).

The framework is transcriptomics-based with a proteomics-ready architecture: the per-patient reaction feature tensors include a dedicated proteomic channel that is zero-filled in the current release and supports future integration of matched protein abundance data.

### Graph Structure

| Property | Value |
|----------|-------|
| Reference model | Recon3D v3 |
| Reaction nodes | 10,600 |
| Metabolite nodes | 5,835 |
| Genes (GPR-mapped) | 2,248 |
| Stoichiometric edges | 40,425 (20,512 substrate_of + 19,913 produces) |
| Shared_metabolite edges | 7,517,742 full; ~83,306 at default k=10 sparsification |
| Currency metabolites | 15 (down-weighted by factor 0.1) |
| Metabolite node features | 519-dimensional (7 physico-chemical + 512-bit Morgan fingerprints) |

### Architecture Configurations

| Configuration | Layers | Heads | Hidden Dim | Edge Types | Parameters | Use Case |
|--------------|--------|-------|------------|------------|------------|----------|
| Config A (reduced-depth) | 2 | 4 | 128 | 2 (substrate_of, produces) | 873,217 | Scaling comparison baseline |
| Config B (expanded) | 3 | 8 | 256 | 3 (+shared_metabolite) | 9.67M | Full-cohort evaluation |

Config A has more parameters in bipartite-only mode because the first-layer input projection from the 519-dimensional metabolite features dominates when the hidden dimension is smaller (128 vs. 256). See manuscript Section 2.4.1.

### Training

- **Optimiser:** AdamW (lr = 5×10⁻⁴, weight decay = 1×10⁻⁵, cosine annealing)
- **Loss:** BCE (class-weighted) + mass-balance regularisation (λ_mb = 0.2)
- **Dropout:** 0.2 within GATv2 layers
- **Early stopping:** Patience 20 on validation F1
- **Convergence:** Best model at epoch 141 of 200

---

## Evaluation Results

### Full-Cohort Evaluation (624 TCGA-CRC Patients)

The primary evaluation uses 5-fold stratified cross-validation on 624 TCGA-CRC patients with HMA 11-tissue union labels as biological supervision. This setting evaluates the framework's scalability and performance under biologically derived (rather than expression-thresholded) supervision.

| Metric | Value |
|--------|-------|
| Cohort | 624 patients (5-fold stratified CV) |
| Labels | HMA 11-tissue union bounds |
| Class balance | 76.9% active / 23.1% inactive |
| Features | v1 scalar (2D: RNA + zero-filled proteomic channel) |
| Edge types | 3 (substrate_of, produces, shared_metabolite at k=10) |
| Parameters | 9.67M (Config B) |
| **AUROC** | **0.663 ± 0.001** |

The AUROC of 0.663 reflects the difficulty of biological supervision from HMA tissue models: these labels capture metabolic activity boundaries across 11 human tissues rather than CRC-specific patterns. The performance gap between expression-thresholded labels and HMA labels is itself a key finding, demonstrating that supervision quality rivals architecture as a performance driver.

### Architecture Scaling

Scaling from Config A (873K parameters) to Config B (9.67M parameters) yields ΔAUROC = +0.115 in the full-cohort setting, demonstrating capacity-dependent learning on metabolic graphs.

### FBA Viability Testing

As a stoichiometric feasibility sanity check, MetaGNN-predicted active sets were tested for biomass flux sustainability under FBA (COBRApy v0.29.0, GLPK solver). Of 33 test patients, **32 (97%) sustained non-zero biomass flux**, compared to 29% for random baselines and 88% for a GPR-threshold heuristic. The single non-viable patient (TCGA-AC-8651-01A) yielded zero biomass flux, indicating that the predicted inactive set removes an essential biosynthetic pathway for that individual. Mean biomass among viable patients was 412.87 ± 31.42 mmol/gDW/h (range: 329.57–472.52).

### Agentic Validation Module

The agentic validation module (Qwen3-32B-AWQ) was evaluated in a cumulative ablation design:

| Version | Configuration | Δ Boundary Accuracy | Significance |
|---------|---------------|---------------------|--------------|
| v1 | Base Qwen3-32B | +1.4% | — |
| v2 | + Enriched GPR context | +4.6% | McNemar p < 0.001 |
| v3 | + Advocate–Resolver | +7.6% | McNemar p < 0.001 |
| v4 | + RAG retrieval | +0.7% | p = 0.352 (n.s.) |

The v3 advocate–resolver architecture achieves the largest statistically significant improvement. Adding FAISS-based RAG retrieval (v4) does not significantly improve over the GNN baseline.

---

## Key Findings

1. **Topology helps:** Graph message passing adds 9.2 F1 points over a disconnected baseline (Table 5 in manuscript).
2. **Signal is coarse:** The topology benefit is primarily driven by degree centrality of metabolic hubs, not fine-grained stoichiometric logic. This is transparently reported.
3. **Label quality matters:** Supervision alignment is a major determinant of performance alongside architecture.
4. **Biologically coherent:** 97% FBA viability confirms metabolically meaningful reconstructions.
5. **Uncertainty quantification:** Monte Carlo Dropout (T=30) provides per-reaction epistemic uncertainty estimates.

---

## Hardware Requirements

The complete pipeline executes on consumer hardware. Tested on Apple M4 Max with MPS acceleration and CPU fallback for unsupported operations (scatter_reduce). CPU-only and CUDA GPU training are also supported. The agentic validation module requires a GPU with ≥18 GB VRAM (tested on NVIDIA RTX 5090) for running the Qwen3-32B-AWQ model.

---

## Licence

- **Code:** MIT Licence
- **Data:** CC-BY 4.0
