# MetaGNN GPU Environment Setup — Windows 11 + RTX 5090

## Quick Setup (copy-paste into PowerShell or Cowork terminal)

### Step 1: Install Miniconda (if not installed)
```powershell
# Download and install from: https://docs.anaconda.com/miniconda/
# Or if conda already available, skip to Step 2
```

### Step 2: Create conda environment
```bash
conda create -n metagnn python=3.11 -y
conda activate metagnn
```

### Step 3: Install PyTorch with CUDA 12.4 (for RTX 5090)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install PyTorch Geometric
```bash
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### Step 5: Install remaining dependencies
```bash
pip install h5py pandas numpy scikit-learn
```

### Step 6: Verify GPU
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```
Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
VRAM: 32.0 GB
```

### Step 7: Copy data folder
Copy the entire `data/` directory from the repository to the Windows machine.
The script expects this structure:
```
data/
  reaction_features/
    TCGA-AA-XXXX-01A.h5    (one per patient)
  metabolite_features.h5
  edge_indices/
    substrate_of.pt
    produces.pt
    shared_metabolite.pt
  recon3d_stoich.h5
  activity_pseudolabels.pt
  clinical_metadata.tsv
```

### Step 8: Run the experiment
```bash
python run_gpu_scaling.py --data_root ./data --output_dir ./results_gpu
```

### Troubleshooting
- If PyG install fails, try: `pip install torch-geometric` first, then the extensions separately
- If CUDA version mismatch, check with `nvidia-smi` and adjust the cu124 to match
- RTX 5090 needs CUDA 12.4+ drivers (version ≥550.54)
- If you see OOM errors, reduce batch_size in the script (but unlikely with 32GB VRAM)
