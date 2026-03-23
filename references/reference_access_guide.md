# Reference Access Guide — MetaGNN MethodsX Manuscript

**Author:** Thiptanawat Phongwattana
**Affiliation:** School of Information Technology, KMUTT
**Corresponding:** Jonathan H. Chan

---

## How to Obtain Each Reference

The table below provides DOI links, open-access status, and access instructions
for every citation in the MethodsX manuscript. For paywalled articles, access via
your institutional library or request a preprint from the corresponding author.

---

### Primary Model References

| # | Citation | DOI | Open Access | How to Access |
|---|----------|-----|-------------|---------------|
| 1 | Brunk et al. 2018 — Recon3D (Nat Chem Biol) | https://doi.org/10.1038/nchembio.2", 2304 | Paywalled | Institutional library; model download: https://www.vmh.life/#downloadview |
| 2 | Velickovic et al. 2018 — GAT (ICLR) | https://arxiv.org/abs/1710.10903 | **Open Access** | Freely available on arXiv |
| 3 | Brody et al. 2022 — GATv2 (ICLR) | https://arxiv.org/abs/2105.14491 | **Open Access** | Freely available on arXiv |
| 4 | Gal & Ghahramani 2016 — MC Dropout (ICML) | https://arxiv.org/abs/1506.02142 | **Open Access** | Freely available on arXiv |
| 5 | Fey & Lenssen 2019 — PyG (ICLR-W) | https://arxiv.org/abs/1903.02428 | **Open Access** | Freely available on arXiv |
| 6 | Zur et al. 2010 — GPR convention (Bioinformatics) | https://doi.org/10.1093/bioinformatics/btq602 | Paywalled | Institutional library |
| 7 | Orth et al. 2010 — FBA review (Nat Biotechnol) | https://doi.org/10.1038/nbt.1614 | Paywalled | Institutional library |

---

### Baseline Methods

| # | Citation | DOI | Open Access | How to Access |
|---|----------|-----|-------------|---------------|
| 8 | Becker & Palsson 2008 — GIMME (PLoS Comput Biol) | https://doi.org/10.1371/journal.pcbi.1000082 | **Open Access** | PLoS (free) |
| 9 | Shlomi et al. 2008 — iMAT (Nat Biotechnol) | https://doi.org/10.1038/nbt.1487 | Paywalled | Institutional library |
| 10 | Agren et al. 2012 — tINIT (PLoS Comput Biol) | https://doi.org/10.1371/journal.pcbi.1002518 | **Open Access** | PLoS (free) |
| 11 | Vlassis et al. 2014 — CORDA (PLoS Comput Biol) | https://doi.org/10.1371/journal.pcbi.1003424 | **Open Access** | PLoS (free) |
| 12 | Xin et al. 2023 — Hypergraph learning for metabolism | https://doi.org/10.1093/bioinformatics/btad261 | **Open Access** | Bioinformatics (OA) |

---

### Dataset References

| # | Citation | DOI | Open Access | How to Access |
|---|----------|-----|-------------|---------------|
| 13 | TCGA Network — COAD (Nature 2012) | https://doi.org/10.1038/nature11252 | **Open Access** | Nature (OA); data: https://portal.gdc.cancer.gov/ |
| 14 | Vasaikar et al. 2019 — CPTAC-CRC (Cell 2019) | https://doi.org/10.1016/j.cell.2019.07.012 | **Open Access** | Cell (OA); data: https://pdc.cancer.gov/ |
| 15 | Human Metabolic Atlas — HMA (Sci Signal 2021) | https://doi.org/10.1126/scisignal.abj1541 | **Open Access** | Sci Signal (OA); models: https://metabolicatlas.org/ |

---

### Software / Toolbox References

| # | Citation | DOI / URL | Licence | How to Download |
|---|----------|-----------|---------|-----------------|
| 16 | COBRA Toolbox v3.0 (Chan et al. 2017) | https://doi.org/10.1038/nprot.2017.0211 | GPL v3 | `git clone https://github.com/opencobra/cobratoolbox` |
| 17 | COBRApy (Ebrahim et al. 2013) | https://doi.org/10.1186/1752-0509-7-74 | LGPL | `pip install cobra` |
| 18 | Gurobi Solver v10.0 | https://www.gurobi.com | Commercial (free academic licence) | https://www.gurobi.com/academia/academic-program-and-licenses/ |
| 19 | PyTorch (Paszke et al. 2019) | https://arxiv.org/abs/1912.01703 | BSD | `pip install torch` |
| 20 | PyTorch Geometric (Fey & Lenssen 2019) | https://github.com/pyg-team/pytorch_geometric | MIT | `pip install torch-geometric` |
| 21 | scFEA (Alghamdi et al. 2021) | https://doi.org/10.1093/genome/research/gr.271... | MIT | https://github.com/changwn/scFEA |

---

## Cloning Key Open-Source Repositories

The `download_repos.sh` script below clones all open-source tools
referenced in the manuscript. Save and run it in your environment.

```bash
#!/bin/bash
# download_repos.sh
# Clones key tool repositories used by MetaGNN

mkdir -p ./tool_repos && cd ./tool_repos

# COBRA Toolbox (MATLAB-based FBA solver)
git clone https://github.com/opencobra/cobratoolbox.git

# PyTorch Geometric
git clone https://github.com/pyg-team/pytorch_geometric.git

# scFEA (comparison baseline)
git clone https://github.com/changwn/scFEA.git

echo "All repositories cloned."
```

---

## Dataset Download Instructions

### 1. TCGA-CRC RNA-seq (GDC Portal)

```bash
# 1. Create a free account at https://portal.gdc.cancer.gov/
# 2. Install gdc-client:
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip

# 3. Add TCGA-COAD and TCGA-READ projects to cart, select:
#    Data Category: Transcriptome Profiling
#    Data Type: Gene Expression Quantification
#    Workflow Type: STAR - Counts
# 4. Download manifest + data:
./gdc-client download -m gdc_manifest.txt -d ./gdc_data/
```

### 2. CPTAC Proteomics (PDC Portal)

```bash
# 1. Register at https://pdc.cancer.gov/
# 2. Navigate to Study PDC000116 (COAD) and PDC000220 (READ)
# 3. Download: Protein Assembly (TMT) → Log2 ratio matrix TSV
# 4. Download clinical file alongside
```

### 3. Human Metabolic Atlas (HMA)

```bash
# Free download — no registration required
wget https://metabolicatlas.org/downloads/Human1/Human1_GEMs.zip
unzip Human1_GEMs.zip
# Contains 98 tissue-specific GEMs in JSON and MATLAB formats
```

### 4. Recon3D

```bash
# Option A: VMH Life (recommended)
# Visit https://www.vmh.life/#downloadview
# Download: Recon3D.mat (COBRA MATLAB format)

# Option B: BiGG Database
# Visit http://bigg.ucsd.edu/models/Recon3D
# Download: Recon3D.mat or Recon3D.xml (SBML format)
```

---

*For access issues, contact: thiptanawat.phon@sit.kmutt.ac.th*
