#!/bin/bash
# MetaGNN: End-to-end pipeline for topology-aware reaction activity scoring
# Usage: conda activate metagnn && bash run.sh
#
# This script orchestrates the complete MetaGNN pipeline from data
# acquisition through training and evaluation.

set -e

echo "============================================"
echo "MetaGNN Pipeline - Starting"
echo "============================================"

# Step 1: Model sanity check
echo "[Step 1/5] Running model sanity check..."
python code/01_metagnn_model.py

# Step 2: Data loading and verification
echo "[Step 2/5] Verifying data loading..."
python code/02_data_loader.py

# Step 3: Training
echo "[Step 3/5] Training MetaGNN..."
python code/03_train_metagnn.py --data_root data/

# Step 4: Evaluation
echo "[Step 4/5] Evaluating model..."
python code/04_evaluate_metagnn.py

# Step 5: Figure generation
echo "[Step 5/5] Generating figures..."
python code/05_generate_figures.py

echo "============================================"
echo "MetaGNN Pipeline - Complete"
echo "============================================"
