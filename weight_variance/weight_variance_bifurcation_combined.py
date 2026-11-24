r"""
Adaptive-coupling Kuramoto: combined Δ-bifurcation figure for both adaptation rules (full width)
===============================================================================================

Merges the two figures of ``weight_variance_bifurcation_rules.py`` into one compact two-column figure
with two rows — mean-field R(Δ) with network markers, and the microscopic V_A/Ā²(Δ) — and four columns:
G = sin (antisymmetric) for μ < γ and μ > γ, then G = cos (symmetric) for μ < γ and μ > γ. The μ–Δ
planes and the Ā row of the separate figures are omitted (the fold and its condition μ > γ are given in
the text, and the bifurcation structure is revisited in the Δ–β plane of Fig. 5). Both blocks reuse the
panel functions and the hand-edited sin-figure geometry (pt) of ``weight_variance_bifurcation_rules.py``.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_bifurcation_combined.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from weight_variance_analysis import set_prl_style, _panel_label
from weight_variance_bifurcation_rules import (CONFIG, LAYOUT, Q_LABEL, C_SYNC, C_ASYNC, MARK, MSYNC,
                                               MASYNC, plot_1d, plot_micro_va, micro_for)

GAP = 10.0                                    # pt between the two blocks
TOP = 12.0                                    # pt added on top for the block titles
RULE_TITLE = {"sin": r"antisymmetric adaptation, $G = \sin$", "cos": r"symmetric adaptation, $G = \cos$"}
OUT = os.path.join(CONFIG["out_dir"], "weight_variance_bif_combined")


def draw_block(fig, W, H, x0, rule, cfg, micro, letters):
    """One rule: R and V_A/Ā² rows (μ < γ | μ > γ) at horizontal offset x0 (pt)."""
    lay = LAYOUT["sin"]
    rect = lambda x, y, w, h: [(x0 + x) / W, 1 - (TOP + y + h) / H, w / W, h / H]
    mus = [cfg["mu_mono"], cfg["mu_bi"]]
    rows = []
    for r, q in enumerate(["R", "VA"]):
        rows.append([])
        for c, mu in enumerate(mus):
            ax = fig.add_axes(rect(lay["cols"][c], lay["rows"][r], *lay["size"]))
            rows[r].append(ax)
            mrule = micro_for(micro, rule, mu)
            if q == "VA":
                plot_micro_va(ax, cfg, mrule)
            else:
                plot_1d(ax, q, mu, cfg, rule, mrule)
            if r == 0:
                ax.set_title(rf"$\mu={mu:g}$", fontsize=6.8, pad=3)
            if r == 1:
                ax.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
            if c == 0:
                ax.set_ylabel(Q_LABEL[q], labelpad=2)
            _panel_label(ax, next(letters))
    xc = (x0 + 0.5 * (lay["cols"][0] + lay["cols"][1] + lay["size"][0])) / W   # block title, centred
    fig.text(xc, 1 - 4.0 / H, RULE_TITLE[rule], ha="center", va="top", fontsize=7.4)
    return rows


def main(cfg=CONFIG):
    micro = np.load(cfg["micro_npz"], allow_pickle=True) if os.path.exists(cfg["micro_npz"]) else None
    w1, h1 = LAYOUT["sin"]["page"]
    W, H = 2 * w1 + GAP, h1 + TOP
    set_prl_style()
    fig = plt.figure(figsize=(W / 72, H / 72))
    letters = iter("abcdefgh")
    rows_sin = draw_block(fig, W, H, 0.0, "sin", cfg, micro, letters)
    rows_cos = draw_block(fig, W, H, w1 + GAP, "cos", cfg, micro, letters)
    for r in range(2):                                                         # shared y-range per row
        axes = rows_sin[r] + rows_cos[r]
        lims = [ax.get_ylim() for ax in axes]
        lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
        for k, ax in enumerate(axes):
            ax.set_ylim(lo, hi)
            if k not in (0, 2):
                ax.tick_params(labelleft=False)
    handles = [Line2D([0], [0], color=C_SYNC, lw=1.2, label="sync. (stable)"),
               Line2D([0], [0], color=C_SYNC, lw=1.0, ls=":", label="saddle (unstable)"),
               Line2D([0], [0], color=C_ASYNC, lw=1.2, label="async."),
               Line2D([0], [0], ls="none", label="fold", **MARK["fold"]),
               Line2D([0], [0], label="micro (sync)", **MSYNC),
               Line2D([0], [0], label="micro (async)", **MASYNC)]
    lx, ly = LAYOUT["sin"]["leg"]                                              # in the cos block, panel (e)
    rows_cos[0][0].legend(handles=handles, loc="upper left", fontsize=LAYOUT["sin"]["leg_fs"],
                          handlelength=1.3, labelspacing=0.18, borderaxespad=0.0,
                          bbox_to_anchor=((w1 + GAP + lx) / W, 1 - (TOP + ly) / H),
                          bbox_transform=fig.transFigure)
    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"[saved] {OUT}.svg / .png  ({W:.1f} x {H:.1f} pt)")


if __name__ == "__main__":
    main()
