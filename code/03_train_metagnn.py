"""
MetaGNN Training Pipeline
Two-stage training:
  Stage 1 — Self-supervised pre-training on 98 HMA tissue-specific GEMs
  Stage 2 — Fine-tuning on TCGA-CRC patient cohort (n=153 training patients)

Author: Thiptanawat Phongwattana
Affiliation: School of Information Technology, KMUTT
"""

import os
import argparse
import yaml
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGLoader

from metagnn_model  import MetaGNN
from data_loader    import MetaGNNDataset, stratified_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters (also configurable via YAML)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    hidden_dim   = 256,
    n_layers     = 3,
    heads        = 8,
    dropout      = 0.20,
    lr           = 1e-3,
    weight_decay = 1e-5,
    batch_size   = 8,        # patients per mini-batch
    n_epochs_pretrain = 50,
    n_epochs_finetune = 200,
    patience     = 20,       # early stopping patience (val F1)
    threshold    = 0.15,     # reaction activity binarisation threshold θ
    mc_T         = 100,      # MC Dropout inference passes
    seed         = 2024,
    lambda_mb    = 0.2,      # mass-balance regularisation weight
)


# ─────────────────────────────────────────────────────────────────────────────
# Loss function: BCE + FBA mass-balance penalty
# ─────────────────────────────────────────────────────────────────────────────
class MetaGNNLoss(nn.Module):
    """
    Combined loss:
      L = L_BCE(s_r, y_r)  +  λ · L_MB(s_r, S)

    L_MB enforces that predicted active reactions approximately satisfy
    mass balance: for each metabolite m, net production ≈ net consumption.
    This regularises the predicted GEM to remain thermodynamically plausible.

    Args:
        stoich_matrix: torch.Tensor(n_met, n_rxn) — Recon3D S matrix
        lambda_mb:     weight on mass-balance regularisation (default 0.1)
    """

    def __init__(self, stoich_matrix: torch.Tensor, lambda_mb: float = 0.2):
        super().__init__()
        self.register_buffer('S', stoich_matrix)
        self.lambda_mb  = lambda_mb
        self.bce        = nn.BCELoss()

    def forward(
        self,
        s_r:  torch.Tensor,   # (n_rxn,)
        y_r:  torch.Tensor,   # (n_rxn,) binary labels
    ) -> torch.Tensor:
        l_bce = self.bce(s_r, y_r.float())

        # Mass-balance regularisation: || S @ diag(s_r) @ 1 || ≈ 0
        # i.e., the weighted net flux through each metabolite should be small
        net_flux = (self.S * s_r.unsqueeze(0)).sum(dim=1)   # (n_met,)
        l_mb     = net_flux.pow(2).mean()

        return l_bce + self.lambda_mb * l_mb


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.15,
) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

    y_bin = (y_pred >= threshold).astype(int)
    f1    = f1_score(y_true, y_bin, zero_division=0)

    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_pred)
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(rec, prec)
    else:
        auroc = auprc = float('nan')

    return {'F1': f1, 'AUROC': auroc, 'AUPRC': auprc}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (single epoch)
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        s_r = model(
            x_dict         = {'reaction': batch['reaction'].x,
                               'metabolite': batch['metabolite'].x},
            edge_index_dict= {rel: batch[rel].edge_index
                              for rel in batch.edge_types},
        )
        loss = criterion(s_r, batch['reaction'].y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.15):
    model.eval()
    all_pred, all_true, total_loss = [], [], 0.0
    for batch in loader:
        batch = batch.to(device)
        s_r = model(
            x_dict         = {'reaction': batch['reaction'].x,
                               'metabolite': batch['metabolite'].x},
            edge_index_dict= {rel: batch[rel].edge_index
                              for rel in batch.edge_types},
        )
        loss = criterion(s_r, batch['reaction'].y)
        total_loss += loss.item()
        all_pred.append(s_r.cpu().numpy())
        all_true.append(batch['reaction'].y.cpu().numpy())

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    metrics = compute_metrics(y_pred, y_true, threshold)
    metrics['loss'] = total_loss / len(loader)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(cfg: dict):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ── Data splits ──────────────────────────────────────────────────────────
    import pandas as pd
    meta_df  = pd.read_csv(
        os.path.join(cfg['data_root'], 'clinical_metadata.tsv'), sep='\t'
    )
    train_ids, val_ids, test_ids = stratified_split(
        meta_df,
        train_frac=0.70,
        val_frac=0.15,
        seed=cfg['seed'],
    )
    logger.info(
        f"Split sizes — train: {len(train_ids)}, "
        f"val: {len(val_ids)}, test: {len(test_ids)}"
    )

    train_ds = MetaGNNDataset(cfg['data_root'], train_ids)
    val_ds   = MetaGNNDataset(cfg['data_root'], val_ids)
    test_ds  = MetaGNNDataset(cfg['data_root'], test_ids)

    train_loader = PyGLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = PyGLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False)
    test_loader  = PyGLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MetaGNN(
        rxn_in_dim=2,
        met_in_dim=519,
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        heads=cfg['heads'],
        dropout=cfg['dropout'],
    ).to(device)

    # Load Recon3D stoichiometric matrix for mass-balance penalty
    import h5py
    with h5py.File(os.path.join(cfg['data_root'], 'recon3d_stoich.h5'), 'r') as f:
        S = torch.tensor(f['S'][:], dtype=torch.float32).to(device)

    criterion = MetaGNNLoss(stoich_matrix=S, lambda_mb=cfg.get('lambda_mb', 0.2))
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['n_epochs_finetune']
    )

    # ── Stage 1: Load pre-trained weights ────────────────────────────────────
    pretrain_path = os.path.join(cfg['data_root'], 'model_weights_pretrained.pt')
    if os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded pre-trained HMA weights — Stage 2 fine-tuning starts")
    else:
        logger.warning(
            "Pre-trained weights not found — training from scratch (Stage 1 + 2)"
        )

    # ── Stage 2: Fine-tuning on TCGA-CRC ─────────────────────────────────────
    best_val_f1 = 0.0
    patience_counter = 0
    save_path = os.path.join(cfg['output_dir'], 'best_model.pt')
    os.makedirs(cfg['output_dir'], exist_ok=True)

    for epoch in range(1, cfg['n_epochs_finetune'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg['threshold'])
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_F1={val_metrics['F1']:.4f}  "
            f"val_AUROC={val_metrics['AUROC']:.4f}"
        )

        if val_metrics['F1'] > best_val_f1:
            best_val_f1 = val_metrics['F1']
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ New best val F1={best_val_f1:.4f} — model saved")
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ── Final evaluation on held-out test set ─────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, cfg['threshold'])
    logger.info(
        f"\n=== Test results ===\n"
        f"  F1    = {test_metrics['F1']:.4f}\n"
        f"  AUROC = {test_metrics['AUROC']:.4f}\n"
        f"  AUPRC = {test_metrics['AUPRC']:.4f}\n"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetaGNN training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides defaults)')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg['data_root']  = args.data_root
    cfg['output_dir'] = args.output_dir

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    main(cfg)
