"""
MetaGNN Data Loader
Constructs PyTorch Geometric HeteroData objects from TCGA-CRC
multi-omics data and Recon3D graph topology.

Author: Thiptanawat Phongwattana
Affiliation: School of Information Technology, KMUTT
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Metabolite feature builder
# ─────────────────────────────────────────────────────────────────────────────
def build_metabolite_features(
    met_ids: List[str],
    smiles_dict: dict,
    metabolomics_df: Optional[pd.DataFrame] = None,
) -> torch.Tensor:
    """
    Build metabolite node feature matrix X_M ∈ ℝ^(n_met × 519).

    Feature breakdown (519-dim):
      - 7 physico-chemical properties (MW, logP, HBA, HBD, TPSA, rings, charge)
      - 512 Morgan fingerprints (radius=2, nBits=512)

    Args:
        met_ids: list of Recon3D metabolite BiGG IDs
        smiles_dict: dict mapping BiGG ID → SMILES string
        metabolomics_df: optional LC-MS quantification (n_met × n_pat);
            if provided, appends row-mean as a 520th feature (not used in paper)

    Returns:
        X_M: Tensor(n_met, 519)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem
    except ImportError:
        raise ImportError(
            "RDKit is required for metabolite feature extraction. "
            "Install via: conda install -c conda-forge rdkit"
        )

    feats = []
    for mid in met_ids:
        smi = smiles_dict.get(mid, None)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            # Zero-fill for metabolites lacking SMILES (e.g. generic cofactors)
            feats.append(np.zeros(519))
            continue

        physico = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            float(Descriptors.RingCount(mol)),
            float(Chem.GetFormalCharge(mol)),
        ], dtype=np.float32)

        fp = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512),
            dtype=np.float32,
        )
        feats.append(np.concatenate([physico, fp]))

    return torch.tensor(np.stack(feats), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Reaction feature builder  (GPR-mapped)
# ─────────────────────────────────────────────────────────────────────────────
def build_reaction_features(
    patient_id: str,
    rnaseq_df: pd.DataFrame,
    proteomics_df: pd.DataFrame,
    gpr_table: pd.DataFrame,
    n_reactions: int = 13543,
) -> torch.Tensor:
    """
    Build reaction node feature matrix X_R ∈ ℝ^(n_rxn × 2) for one patient.

    Columns:
      0: GPR-mapped VST RNA-seq score  (log2-TPM via DESeq2 VST)
      1: GPR-mapped TMT protein abundance (log2-normalised CPTAC)

    GPR convention (Zur et al. 2010):
      AND (enzyme complex) → min over gene members
      OR  (isoenzyme)      → max over complex scores

    Args:
        patient_id:   TCGA barcode (e.g. "TCGA-A6-2672-01A")
        rnaseq_df:    VST-normalised expression DataFrame (genes × patients)
        proteomics_df: TMT protein abundance DataFrame (proteins × patients)
        gpr_table:    DataFrame with columns [reaction_id, gene_sets_str]
                      where gene_sets_str encodes AND/OR logic as nested lists
        n_reactions:  total reactions in Recon3D (13,543)

    Returns:
        X_R: Tensor(n_reactions, 2)
    """
    rna_expr  = rnaseq_df[patient_id].to_dict()    if patient_id in rnaseq_df.columns    else {}
    prot_expr = proteomics_df[patient_id].to_dict() if patient_id in proteomics_df.columns else {}

    X_R = np.zeros((n_reactions, 2), dtype=np.float32)

    for rxn_idx, row in gpr_table.iterrows():
        gene_sets = eval(row['gene_sets_str'])  # list of lists (AND-groups for OR)
        if not gene_sets:
            continue

        # RNA-seq GPR score
        rna_complexes = []
        for and_group in gene_sets:
            vals = [rna_expr.get(g, 0.0) for g in and_group]
            if vals:
                rna_complexes.append(min(vals))  # AND → min
        if rna_complexes:
            X_R[rxn_idx, 0] = max(rna_complexes)  # OR → max

        # Proteomics GPR score
        prot_complexes = []
        for and_group in gene_sets:
            vals = [prot_expr.get(g, 0.0) for g in and_group]
            if vals:
                prot_complexes.append(min(vals))
        if prot_complexes:
            X_R[rxn_idx, 1] = max(prot_complexes)

    return torch.tensor(X_R, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Graph topology (edge indices from Recon3D stoichiometry)
# ─────────────────────────────────────────────────────────────────────────────
def build_edge_indices(
    recon3d_stoich: np.ndarray,   # shape (n_met, n_rxn)  — stoichiometric matrix
    rxn_ids: List[str],
    met_ids: List[str],
) -> dict:
    """
    Derive three edge-relation types from the Recon3D stoichiometric matrix S.

    Edge types:
      substrate_of  : metabolite → reaction  (S[m,r] < 0)
      produces      : reaction → metabolite  (S[m,r] > 0)
      shared_metabolite: reaction → reaction (two reactions share ≥1 metabolite)

    Args:
        recon3d_stoich: S matrix (n_met × n_rxn); negative = substrate, positive = product
        rxn_ids: ordered list of reaction BiGG IDs
        met_ids: ordered list of metabolite BiGG IDs

    Returns:
        dict mapping edge_type tuple → edge_index Tensor(2, n_edges)
    """
    n_met, n_rxn = recon3d_stoich.shape

    # substrate_of: metabolite m → reaction r  where S[m,r] < 0
    sub_rows, sub_cols = np.where(recon3d_stoich < 0)
    ei_substrate = torch.tensor(
        np.stack([sub_rows, sub_cols]), dtype=torch.long
    )

    # produces: reaction r → metabolite m  where S[m,r] > 0
    prod_met, prod_rxn = np.where(recon3d_stoich > 0)
    ei_produces = torch.tensor(
        np.stack([prod_rxn, prod_met]), dtype=torch.long
    )

    # shared_metabolite: reaction r1 → reaction r2 if they share ≥1 metabolite
    # Build adjacency using binary participation matrix P (n_met × n_rxn)
    P = (recon3d_stoich != 0).astype(np.float32)
    shared = P.T @ P  # (n_rxn × n_rxn); shared[r1,r2] = # shared metabolites
    np.fill_diagonal(shared, 0)
    r1s, r2s = np.where(shared > 0)
    ei_shared = torch.tensor(np.stack([r1s, r2s]), dtype=torch.long)

    return {
        ('metabolite', 'substrate_of',      'reaction'): ei_substrate,
        ('reaction',   'produces',          'metabolite'): ei_produces,
        ('reaction',   'shared_metabolite', 'reaction'): ei_shared,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────
class MetaGNNDataset(torch.utils.data.Dataset):
    """
    One HeteroData graph per patient.  Metabolite features and edge indices
    are shared across patients and loaded once to save memory.

    Directory structure expected:
        data_root/
          reaction_features/
            TCGA-AA-XXXX-01A.h5     # X_R per patient
          metabolite_features.h5     # X_M (shared)
          edge_indices/
            substrate_of.pt
            produces.pt
            shared_metabolite.pt
          activity_pseudolabels.pt   # y_r (binary, from HMA)
          clinical_metadata.tsv
    """

    def __init__(self, data_root: str, patient_ids: Optional[List[str]] = None):
        self.data_root  = data_root
        self.meta_df    = pd.read_csv(
            os.path.join(data_root, 'clinical_metadata.tsv'), sep='\t'
        )
        if patient_ids is not None:
            self.meta_df = self.meta_df[
                self.meta_df['tcga_barcode'].isin(patient_ids)
            ].reset_index(drop=True)

        self.patient_ids = self.meta_df['tcga_barcode'].tolist()

        # Load shared tensors once
        with h5py.File(os.path.join(data_root, 'metabolite_features.h5'), 'r') as f:
            self.X_M = torch.tensor(f['X_M'][:], dtype=torch.float32)

        self.edge_index_dict = {
            ('metabolite', 'substrate_of',      'reaction'):
                torch.load(os.path.join(data_root, 'edge_indices', 'substrate_of.pt')),
            ('reaction',   'produces',          'metabolite'):
                torch.load(os.path.join(data_root, 'edge_indices', 'produces.pt')),
            ('reaction',   'shared_metabolite', 'reaction'):
                torch.load(os.path.join(data_root, 'edge_indices', 'shared_metabolite.pt')),
        }

        self.y_r = torch.load(os.path.join(data_root, 'activity_pseudolabels.pt'))

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> HeteroData:
        pid = self.patient_ids[idx]
        rxn_feat_path = os.path.join(
            self.data_root, 'reaction_features', f'{pid}.h5'
        )
        with h5py.File(rxn_feat_path, 'r') as f:
            X_R = torch.tensor(f['X_R'][:], dtype=torch.float32)

        data = HeteroData()
        data['reaction'].x        = X_R
        data['metabolite'].x      = self.X_M
        data['reaction'].y        = self.y_r
        data['reaction'].pid      = pid

        for rel, ei in self.edge_index_dict.items():
            src_type, rel_type, dst_type = rel
            data[src_type, rel_type, dst_type].edge_index = ei

        return data


# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test split (stratified by MSI status and tumour stage)
# ─────────────────────────────────────────────────────────────────────────────
def stratified_split(
    meta_df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed: int = 2024,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Produces train/val/test splits stratified by MSI status
    (to preserve class balance across evaluation partitions).

    Args:
        meta_df:     DataFrame with columns ['tcga_barcode', 'msi_status']
        train_frac:  fraction of patients for training (default 0.70 → 153 pt)
        val_frac:    fraction for validation (default 0.15 → 33 pt)

    Returns:
        (train_ids, val_ids, test_ids) lists of TCGA barcodes
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(seed)
    strat_col = meta_df['msi_status'].fillna('Unknown')

    ids = meta_df['tcga_barcode'].tolist()
    strat = strat_col.tolist()

    train_ids, tmp_ids, _, tmp_strat = train_test_split(
        ids, strat,
        train_size=train_frac,
        stratify=strat,
        random_state=seed,
    )
    val_frac_of_tmp = val_frac / (1 - train_frac)
    val_ids, test_ids = train_test_split(
        tmp_ids,
        train_size=val_frac_of_tmp,
        stratify=tmp_strat,
        random_state=seed,
    )
    return train_ids, val_ids, test_ids
