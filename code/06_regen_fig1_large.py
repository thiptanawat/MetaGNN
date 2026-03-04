"""
Regenerate Figure 1 — MetaGNN Architecture Diagram at larger scale
Wider canvas (18×9 in), all font sizes enlarged, bigger nodes/boxes
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

NAVY   = '#1F3864'
BLUE   = '#2E75B6'
LBLUE  = '#5BA3D9'
LLBLUE = '#A8CEEB'
GREY   = '#888888'
LGREY  = '#F0F4F8'
GREEN  = '#2E8B57'
RED    = '#C0392B'
ORANGE = '#E67E22'

# ── scale everything to a wider, taller canvas ──────────────────────────────
# Old: figsize=(13,7), xlim=(0,13), ylim=(0,7)
# New: figsize=(18,9), xlim=(0,18), ylim=(0,9) → all coords × 18/13 ≈ 1.38
S = 18 / 13   # scale factor for x-coords
T = 9  / 7    # scale factor for y-coords

fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlim(0, 18)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# ── helpers ───────────────────────────────────────────────────────────────────
def fancy_box(x, y, w, h, color, label, sublabel=None,
              fontsize=11.5, alpha=0.93, radius=0.3):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                         boxstyle=f"round,pad=0.06,rounding_size={radius}",
                         facecolor=color, edgecolor='white',
                         linewidth=2, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x, y - 0.27, sublabel, ha='center', va='center',
                fontsize=9.5, color='white', alpha=0.9, zorder=4)

def rxn_node(x, y, r=0.32):
    c = Circle((x, y), r, facecolor=BLUE, edgecolor='white',
                linewidth=1.8, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, 'R', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=6)

def met_node(x, y, s=0.28):
    sq = FancyBboxPatch((x-s, y-s), 2*s, 2*s,
                        boxstyle="round,pad=0.04",
                        facecolor=GREEN, edgecolor='white',
                        linewidth=1.8, zorder=5)
    ax.add_patch(sq)
    ax.text(x, y, 'M', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=6)

def arr(x1, y1, x2, y2, color=GREY, lw=1.8, alpha=0.75):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, alpha=alpha,
                                mutation_scale=18))

# ── PANEL TITLE ───────────────────────────────────────────────────────────────
ax.text(9, 8.65,
        'MetaGNN Architecture: Heterogeneous Graph Attention Network (H-GAT)',
        ha='center', va='center', fontsize=14.5,
        fontweight='bold', color=NAVY)

# ── BACKGROUND BANDS (subtle zone shading) ───────────────────────────────────
for xlo, xhi, fc in [(0.15, 2.65, '#EBF4FB'),
                     (2.75, 4.6,  '#F0F8FF'),
                     (4.7,  9.2,  '#F5FAF5'),
                     (9.3,  13.8, '#F0F4FB'),
                     (13.9, 17.85,'#FEF9EE')]:
    ax.add_patch(FancyBboxPatch((xlo, 0.55), xhi-xlo, 7.4,
                 boxstyle="round,pad=0.1", facecolor=fc,
                 edgecolor='none', alpha=0.55, zorder=0))

# ── SECTION LABELS (top zone labels) ─────────────────────────────────────────
for cx, lbl in [(1.4,  'Multi-Omics\nInput'),
                (3.65, 'Input\nProjection'),
                (6.9,  'Heterogeneous Bipartite Graph'),
                (11.5, 'H-GAT Message Passing\n(3 layers, 8 attention heads each)'),
                (15.9, 'Output')]:
    ax.text(cx, 7.9, lbl, ha='center', fontsize=10,
            color=NAVY, fontweight='bold', style='italic')

# ── OMICS INPUT BLOCKS ────────────────────────────────────────────────────────
omics = [
    (1.4, 6.6, BLUE,  'RNA-seq\n(VST)',      '13,543 rxn\nfeatures'),
    (1.4, 5.3, LBLUE, 'Proteomics\n(TMT)',   '13,543 rxn\nfeatures'),
    (1.4, 4.0, GREEN, 'Metabolomics\n(LC-MS)','4,140 met\nfeatures'),
]
for x, y, c, lbl, sub in omics:
    fancy_box(x, y, 2.1, 0.95, c, lbl, sub, fontsize=10, radius=0.25)

# ── INPUT PROJECTION BOX ──────────────────────────────────────────────────────
fancy_box(3.65, 5.3, 1.7, 3.5, NAVY,
          'Feature\nEngineering', 'GPR rules\nd = 256', fontsize=11)
for y in [6.6, 5.3, 4.0]:
    arr(2.45, y, 2.8, 5.3, GREY, 1.5)

# ── GRAPH: reaction (R) and metabolite (M) nodes ─────────────────────────────
# Re-positioned for wider canvas
rxn_pos = [(5.8, 6.5), (7.6, 6.5),
           (5.8, 4.5), (7.6, 4.5),
           (6.7, 3.3)]
met_pos = [(5.1, 5.6), (6.7, 7.3),
           (8.4, 5.6), (5.1, 3.6),
           (8.4, 3.6), (6.7, 2.6)]

# edges first (below nodes)
edges = [(met_pos[0], rxn_pos[0]), (met_pos[1], rxn_pos[0]),
         (met_pos[1], rxn_pos[1]), (met_pos[2], rxn_pos[1]),
         (rxn_pos[0], met_pos[3]), (rxn_pos[1], met_pos[2]),
         (met_pos[3], rxn_pos[2]), (met_pos[4], rxn_pos[3]),
         (rxn_pos[2], met_pos[5]), (rxn_pos[3], met_pos[5]),
         (met_pos[5], rxn_pos[4])]
for (x1,y1),(x2,y2) in edges:
    ax.plot([x1,x2],[y1,y2], color='#B0C4DE', lw=1.2,
            zorder=2, alpha=0.85)

for p in rxn_pos: rxn_node(*p)
for p in met_pos: met_node(*p)

# legend
ax.legend(handles=[
    mpatches.Patch(facecolor=BLUE,  label='Reaction node  (R)'),
    mpatches.Patch(facecolor=GREEN, label='Metabolite node (M)'),
], loc='lower left', bbox_to_anchor=(4.55, 1.55),
   fontsize=10, framealpha=0.9, edgecolor='#CCCCCC', fancybox=True)

# arrow from projection to graph
arr(4.5, 5.3, 4.7, 5.3, GREY, 2.0)

# ── H-GAT LAYERS ─────────────────────────────────────────────────────────────
layer_x  = [10.1, 11.6, 13.1]
l_colors = [NAVY, BLUE, LBLUE]
l_labels = ['H-GAT\nLayer 1', 'H-GAT\nLayer 2', 'H-GAT\nLayer 3']
l_subs   = ['8-head att.\nLayerNorm', '8-head att.\nLayerNorm', '8-head att.\nLayerNorm']

arr(8.65, 5.3, 9.4, 5.3, GREY, 2.0)
for i,(x,c,lbl,sub) in enumerate(zip(layer_x, l_colors, l_labels, l_subs)):
    fancy_box(x, 5.3, 1.25, 3.6, c, lbl, sub, fontsize=11)
    if i < 2:
        arr(x+0.63, 5.3, x+1.12, 5.3, GREY, 1.8)

# ── FLOW ANNOTATION: relation types ──────────────────────────────────────────
for i, (lbl, col) in enumerate([('substrate_of', '#2980B9'),
                                  ('produces',     '#27AE60'),
                                  ('shared_met.',  '#8E44AD')]):
    ax.text(11.6, 3.2 - i*0.42, f'↺ {lbl}',
            ha='center', fontsize=9, color=col, style='italic')

# ── OUTPUT HEAD ───────────────────────────────────────────────────────────────
fancy_box(15.0, 5.3, 1.9, 3.6, ORANGE,
          'Output\nHead', 'MLP + Sigmoid\ns_r ∈ (0,1)', fontsize=11)
arr(13.73, 5.3, 14.05, 5.3, GREY, 2.0)

# MC Dropout annotation
ax.annotate('MC Dropout\nT = 100 passes\nUncertainty  σ_r',
            xy=(15.0, 3.5), xytext=(15.0, 2.4),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5,
                            mutation_scale=16),
            ha='center', fontsize=9.5, color=ORANGE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF9E7',
                      edgecolor=ORANGE, alpha=0.92))

# FBA output box
fancy_box(15.0, 1.5, 1.9, 1.1, GREEN,
          'FBA Solver', 'Patient-specific GEM', fontsize=10, radius=0.25)
arr(15.0, 3.0, 15.0, 2.05, GREEN, 1.8)

# ── BOTTOM NOTE ───────────────────────────────────────────────────────────────
ax.text(9, 0.28,
        'Reaction node (R) features: GPR-mapped VST expression + TMT protein abundance  ●  '
        'Metabolite node (M) features: physico-chemical descriptors + Morgan fingerprints',
        ha='center', fontsize=9, color=GREY, style='italic')

plt.tight_layout(pad=0.4)
fig.savefig('fig1_architecture.png', dpi=220, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 1 regenerated at 18×9 inches, 220 DPI.")
