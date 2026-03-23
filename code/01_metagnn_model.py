"""
MetaGNN: Heterogeneous Graph Attention Network for Patient-Specific
Metabolic Network Reconstruction.

Model architecture implementation using PyTorch Geometric HeteroData.

Reference:
  Phongwattana T, Chan JH. "MetaGNN: A Heterogeneous Graph Attention
  Network Framework for Personalised Genome-Scale Metabolic Model
  Reconstruction from Clinical Multi-Omics Data." MethodsX (2024).

Affiliation:
  School of Information Technology, King Mongkut's University of
  Technology Thonburi (KMUTT), Bangkok 10140, Thailand.

Author: Thiptanawat Phongwattana
Corresponding: Jonathan H. Chan (jonathan@sit.kmutt.ac.th)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Monte Carlo Dropout wrapper
# ─────────────────────────────────────────────────────────────────────────────
class MCDropout(nn.Module):
    """Dropout that remains active at inference for uncertainty estimation
    (Gal & Ghahramani, ICML 2016)."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # training=True forces dropout even during model.eval()
        return F.dropout(x, p=self.p, training=True)


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneous GAT layer (one message-passing step)
# ─────────────────────────────────────────────────────────────────────────────
class HGATLayer(nn.Module):
    """
    Single H-GAT layer operating on the bipartite reaction–metabolite graph.
    Supports three edge-relation types:
      - ('metabolite', 'substrate_of', 'reaction')
      - ('reaction',   'produces',     'metabolite')
      - ('reaction',   'shared_metabolite', 'reaction')
    """

    def __init__(
        self,
        in_channels: Dict[str, int],
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.2,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.out_channels = out_channels
        self.heads = heads

        # GATv2Conv is strictly more expressive than the original GAT
        # (Brody et al., ICLR 2022) and handles directed heterogeneous edges.
        self.conv = HeteroConv(
            {
                ('metabolite', 'substrate_of', 'reaction'): GATv2Conv(
                    in_channels['metabolite'], out_channels,
                    heads=heads, dropout=dropout,
                    add_self_loops=False, concat=False,
                ),
                ('reaction', 'produces', 'metabolite'): GATv2Conv(
                    in_channels['reaction'], out_channels,
                    heads=heads, dropout=dropout,
                    add_self_loops=False, concat=False,
                ),
                ('reaction', 'shared_metabolite', 'reaction'): GATv2Conv(
                    in_channels['reaction'], out_channels,
                    heads=heads, dropout=dropout,
                    add_self_loops=True, concat=False,
                ),
            },
            aggr='mean',
        )

        self.norm_rxn = nn.LayerNorm(out_channels)
        self.norm_met = nn.LayerNorm(out_channels)

        # Projection for residual connections when dimensions differ
        self.proj_rxn = (
            nn.Linear(in_channels['reaction'], out_channels)
            if in_channels['reaction'] != out_channels else nn.Identity()
        )
        self.proj_met = (
            nn.Linear(in_channels['metabolite'], out_channels)
            if in_channels['metabolite'] != out_channels else nn.Identity()
        )
        self.mc_drop = MCDropout(p=dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out = self.conv(x_dict, edge_index_dict)

        # Residual + LayerNorm
        if self.residual:
            out['reaction']   = self.norm_rxn(
                F.elu(out['reaction'])   + self.proj_rxn(x_dict['reaction']))
            out['metabolite'] = self.norm_met(
                F.elu(out['metabolite']) + self.proj_met(x_dict['metabolite']))
        else:
            out['reaction']   = self.norm_rxn(F.elu(out['reaction']))
            out['metabolite'] = self.norm_met(F.elu(out['metabolite']))

        out['reaction']   = self.mc_drop(out['reaction'])
        out['metabolite'] = self.mc_drop(out['metabolite'])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MetaGNN full model
# ─────────────────────────────────────────────────────────────────────────────
class MetaGNN(nn.Module):
    """
    Full MetaGNN model:
      1. Input projection: per-node-type linear encoders
      2. 3-layer H-GAT message passing (8-head attention, d=256)
      3. Output MLP head → reaction activity score s_r ∈ (0, 1)

    Uncertainty quantification:
      Call predict_with_uncertainty() which runs T stochastic forward
      passes (MC Dropout) and returns mean score + epistemic σ_r.
    """

    def __init__(
        self,
        rxn_in_dim: int = 2,       # RNA-seq + proteomics GPR features
        met_in_dim: int = 519,     # physico-chemical (7) + Morgan FP (512)
        hidden_dim: int = 256,
        n_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.2,
        n_reactions: int = 13543,
    ):
        super().__init__()
        self.n_reactions = n_reactions
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers

        # Input projections (bring both node types to hidden_dim)
        self.proj_rxn = nn.Sequential(
            Linear(rxn_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.proj_met = nn.Sequential(
            Linear(met_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # H-GAT layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                HGATLayer(
                    in_channels={'reaction': hidden_dim, 'metabolite': hidden_dim},
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    residual=True,
                )
            )

        # Output head: MLP → scalar reaction activity score
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            MCDropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # output ∈ (0, 1)
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x_dict: {'reaction': Tensor(n_rxn, rxn_in_dim),
                     'metabolite': Tensor(n_met, met_in_dim)}
            edge_index_dict: heterogeneous edge indices

        Returns:
            s_r: Tensor(n_rxn,) — predicted activity score per reaction
        """
        # Project to shared hidden space
        h = {
            'reaction':   self.proj_rxn(x_dict['reaction']),
            'metabolite': self.proj_met(x_dict['metabolite']),
        }

        # H-GAT message passing
        for layer in self.layers:
            h = layer(h, edge_index_dict)

        # Decode reaction node embeddings
        s_r = self.output_head(h['reaction']).squeeze(-1)
        return s_r

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        T: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference.

        Args:
            T: number of stochastic forward passes

        Returns:
            mean_score: Tensor(n_rxn,) — mean predicted s_r
            sigma_r:    Tensor(n_rxn,) — epistemic std (uncertainty)
        """
        # MC Dropout remains active (training=True inside MCDropout)
        samples = torch.stack([self(x_dict, edge_index_dict) for _ in range(T)])
        mean_score = samples.mean(dim=0)
        sigma_r    = samples.std(dim=0)
        return mean_score, sigma_r


# ─────────────────────────────────────────────────────────────────────────────
# GPR feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────
def apply_gpr_rules(
    gene_expr: torch.Tensor,     # shape (n_genes,)
    gpr_and_idx: list,           # list of lists of gene indices (AND sub-complex)
    gpr_or_idx: list,            # list of lists of sub-complex results (OR isoenzyme)
    n_reactions: int = 13543,
) -> torch.Tensor:
    """
    Compute reaction-level GPR-mapped expression.

    Convention (Zur et al. 2010):
      AND gate → min (bottleneck / enzyme complex)
      OR  gate → max (isoenzyme redundancy)

    Returns:
        rxn_scores: Tensor(n_reactions,) mapped expression values
    """
    rxn_scores = torch.zeros(n_reactions)
    for rxn_idx, (and_groups, or_groups) in enumerate(
        zip(gpr_and_idx, gpr_or_idx)
    ):
        if not and_groups:
            continue  # spontaneous reaction, no gene association
        # AND within each complex
        complex_scores = [gene_expr[g].min() for g in and_groups if len(g) > 0]
        if complex_scores:
            # OR across isoenzymes
            rxn_scores[rxn_idx] = max(complex_scores)
    return rxn_scores


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("MetaGNN model — sanity check")
    model = MetaGNN(rxn_in_dim=2, met_in_dim=519, hidden_dim=256, n_layers=3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Synthetic mini-batch
    n_rxn, n_met = 100, 50
    x_dict = {
        'reaction':   torch.randn(n_rxn, 2),
        'metabolite': torch.randn(n_met, 519),
    }
    # Random bipartite edges
    edge_index_dict = {
        ('metabolite', 'substrate_of', 'reaction'):
            torch.randint(0, min(n_met, n_rxn), (2, 200)),
        ('reaction', 'produces', 'metabolite'):
            torch.randint(0, min(n_met, n_rxn), (2, 180)),
        ('reaction', 'shared_metabolite', 'reaction'):
            torch.randint(0, n_rxn, (2, 300)),
    }

    score = model(x_dict, edge_index_dict)
    mean_s, sigma = model.predict_with_uncertainty(x_dict, edge_index_dict, T=10)
    print(f"  Forward pass output shape:   {score.shape}")
    print(f"  MC-Dropout mean shape:       {mean_s.shape}")
    print(f"  MC-Dropout sigma shape:      {sigma.shape}")
    print(f"  Score range: [{score.min():.4f}, {score.max():.4f}]")
    print("  PASSED")
