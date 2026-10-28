r"""
Adaptive-coupling Kuramoto: connectivity-structure figure (cos vs sin, μ=0.04)
===============================================================================

Loads the Δ-sweep written by ``weight_variance_connectivity_sweep.py`` (K=1, γ=0.01, μ=0.04; both
adaptation rules) and builds a two-column PRL figure. Panel labels follow the manuscript convention
(one per column-group):

  (a) left column   : raw weight variance V_A vs Δ (top) + raw pair weight vs frequency difference
                      ω_j−ω_i at 4 representative Δ (bottom); cos & sin.
  (b, c, d, e)      : the four coupling-matrix columns (cos top / sin bottom) at the representative Δ,
                      on a colour scale shared per rule (linear, centred on the baseline A=1); the
                      x-ticks of the bottom (sin) row give the frequency range of the matrix axes.
  (f) bottom row    : phase-coherence dynamics R(t) over the post-transient window (cos | sin), one
                      trace per representative Δ.
Colours in (a) and (f): one sequential map per rule (cfg["rule_cmaps"]; cos blue, sin orange), one shade
per representative Δ (darker = larger Δ); the representative Δ are also marked on the V_A curves.

Layout: fixed to the hand-edited version of 2026-10-27 — exact axes positions in pt (FIG_W_PT, VA_RECT,
PW_RECT, MAT_X, MAT_Y, CB_X, RT_RECTS), no tight bounding box.

    python weight_variance_connectivity_figure.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from weight_variance_analysis import S_order, branches, set_prl_style, _panel_label  # noqa: E402

CONFIG = dict(
    npz=dp.mpmf("weight_variance_connectivity.npz"),
    rep_deltas=[0.1, 0.3, 0.5, 0.8],  # representative Δ (nearest sweep points) for the matrices
    cmap="PRGn_r",                    # diverging coupling-matrix map (centred at A=1; green = below, incl. <0);
                                      # kept clear of the blue / orange rule colours of (a) and (f)
    rule_cmaps={"cos": "GnBu", "sin": "YlOrBr"},     # one sequential map per rule, used in (a) and (f);
                                      # multi-hue maps keep the 4 Δ apart (single-hue Blues/Oranges did not)
    shade_range=(0.4, 0.95),          # map positions of the smallest / largest representative Δ
    rule_shade=0.8,                   # map position of the rule colour (V_A curves)
    out=dp.mpmf("weight_variance_connectivity"),
)

RULE_TEX = {"cos": r"\cos", "sin": r"\sin"}

# ── exact layout of the hand-edited version (2026-10-27; axes boxes in pt, origin top-left) ──
FIG_W_PT, FIG_H_PT = 530.459, 304.779
VA_RECT = (33.926, 14.580, 115.590, 80.743)                   # (a) top: V_A vs Δ
PW_RECT = (33.926, 114.471, 115.590, 80.743)                  # (a) bottom: pair weight vs ω_j−ω_i
PW_LEG_SHIFT = (3.0, 3.0)                                     # pair-weight key: right / up from the corner (pt)
MAT_X, MAT_Y, MAT_W = (166.217, 254.982, 342.247, 429.512), (17.349, 114.240), 75.204   # square matrices
CB_X, CB_W = 509.816, 3.760                                   # one colour bar per rule, right of the last matrix
RT_RECTS = ((33.926, 223.719, 226.227, 59.241), (296.349, 223.719, 226.227, 59.241))  # (f) R(t), cos | sin


def va_full(R2, A, dl, K, mu, g):
    """Raw steady-state weight variance (full closure C_A=C_S+C_F); S clipped to [0,1]."""
    R2 = np.clip(R2, 0.0, None)
    S = np.clip(S_order(R2, A, dl, K), 0.0, 1.0)
    return mu ** 2 / (2 * g ** 2 * (g + 2 * dl)) * (g * (1.0 - R2 ** 2) + 2 * dl * (S ** 2 - R2 ** 2))


def mf_va(rule, d, K, mu, g):
    """Coherent-IC-selected mean-field V_A(Δ): synchronized branch where physical, else asynchronous
    (R=0, Ā=1). cos uses the Ā₊ root; sin uses Ā≡1, R²=1−2Δ/K. Rule-dependent through S (and Ā)."""
    ones = np.ones_like(d)
    va_async = va_full(np.zeros_like(d), ones, d, K, mu, g)
    if rule == "cos":
        br = branches(d, mu, K, g)["sync"]
        va_sync = va_full(br["R"] ** 2, br["A"], d, K, mu, g)
        return np.where(np.isfinite(br["R"]), va_sync, va_async)
    R2 = 1.0 - 2.0 * d / K                                        # sin: Ā ≡ 1
    va_sync = va_full(np.clip(R2, 0.0, None), ones, d, K, mu, g)
    return np.where(R2 > 0, va_sync, va_async)




def main(cfg=CONFIG):
    d = np.load(cfg["npz"], allow_pickle=True)
    rules = [str(r) for r in d["rules"]]
    deltas = d["deltas"]
    VA, Amat, omega_axis = d["VA"], d["Amat"], d["omega_axis"]
    Rt, t_rec = d["Rt"], d["t_rec"]
    mu, g, K, N = float(d["mu"]), float(d["gamma"]), float(d["K"]), int(d["N"])
    rep_idx = [int(np.argmin(np.abs(deltas - t))) for t in cfg["rep_deltas"]]
    assert len(rep_idx) == len(MAT_X), "the fixed layout holds exactly len(MAT_X) representative Δ"
    # colours: one sequential map per rule; the rule colour for V_A, and one shade per representative Δ
    cmaps = {r: plt.get_cmap(cfg["rule_cmaps"][r]) for r in rules}
    c_rule = {r: cmaps[r](cfg["rule_shade"]) for r in rules}
    shade = {r: cmaps[r](np.linspace(*cfg["shade_range"], len(rep_idx))) for r in rules}
    tuple_map = {tuple: HandlerTuple(ndivide=None, pad=0.0)}         # legend entry = cos and sin segment

    set_prl_style()
    fig = plt.figure(figsize=(FIG_W_PT / 72, FIG_H_PT / 72))
    rect = lambda x, y, w, h: [x / FIG_W_PT, 1 - (y + h) / FIG_H_PT, w / FIG_W_PT, h / FIG_H_PT]   # pt -> fig fraction

    # ── (a) raw weight variance V_A vs Δ: micro (solid + markers) + MF (dashed) ──
    ax_va = fig.add_axes(rect(*VA_RECT))
    for jb in rep_idx:                                               # Δ shown in panels b–e
        ax_va.axvline(deltas[jb], color="0.6", ls=":", lw=0.9, zorder=1)
    dc = np.linspace(max(deltas.min(), 1e-3), deltas.max(), 400)
    for i, rule in enumerate(rules):
        ax_va.plot(deltas, VA[i], "-o", color=c_rule[rule], lw=1.2, ms=2.6, zorder=3)
        ax_va.plot(dc, mf_va(rule, dc, K, mu, g), ls="--", color=c_rule[rule], lw=1.1, zorder=2)
        ax_va.scatter(deltas[rep_idx], VA[i, rep_idx], s=14, c=shade[rule], edgecolors="k",
                      linewidths=0.4, zorder=4)                     # representative Δ in their shades
    ax_va.set_yscale("log")
    ax_va.set_xlim(0, deltas.max() * 1.02)
    ax_va.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
    ax_va.set_ylabel(r"weight variance $V_A$", labelpad=2)
    hleg = [Line2D([0], [0], color=c_rule["cos"], lw=1.2, marker="o", ms=2.6, label=r"$G=\cos$"),
            Line2D([0], [0], color=c_rule["sin"], lw=1.2, marker="o", ms=2.6, label=r"$G=\sin$"),
            Line2D([0], [0], color="0.45", lw=1.1, ls="--", label="mean field")]
    ax_va.legend(handles=hleg, loc="lower left", fontsize=5.6, handlelength=1.6, labelspacing=0.2)
    _panel_label(ax_va, "a", dx=-30)

    # ── (b) raw pair weight vs frequency difference ω_j−ω_i, at the 3 representative Δ ──
    #    colour map = rule (cfg["rule_cmaps"]), shade = Δ; binned mean over all off-diagonal
    #    entries. The two vertical dashed lines at ±2K mark where TWO ISOLATED oscillators
    #    (bidirectional coupling strength K) would lock: φ̇ = Δω − 2K sin φ ⇒ locked iff |ω_j−ω_i| ≤ 2K
    #    (here K=1 ⇒ ±2).
    ax_b = fig.add_axes(rect(*PW_RECT))
    off = ~np.eye(N, dtype=bool)
    bins = np.linspace(-10.0, 10.0, 41)
    xc = 0.5 * (bins[:-1] + bins[1:])
    for i, rule in enumerate(rules):
        for si, jb in enumerate(rep_idx):
            w = omega_axis[jb]
            X = (w[None, :] - w[:, None])[off]                       # ω_j − ω_i (col − row)
            A = Amat[i, jb][off]
            idx = np.digitize(X, bins)
            m = np.array([A[idx == k + 1].mean() if np.count_nonzero(idx == k + 1) >= 5 else np.nan
                          for k in range(xc.size)])
            ax_b.plot(xc, m, color=shade[rule][si], lw=1.1, zorder=3)
    for sgn in (-1.0, 1.0):                                          # two-oscillator locking |Δω|≤2K
        ax_b.axvline(sgn * 2.0 * K, color="0.35", ls="--", lw=0.9, zorder=1)
    ax_b.axhline(1.0, color="0.6", lw=0.4, zorder=0)                 # relaxation baseline A→1
    ax_b.set_xlim(-7.0, 7.0)
    ax_b.set_xlabel(r"frequency difference $\omega_j-\omega_i$", labelpad=1)
    ax_b.set_ylabel(r"pair weight $A_{ij}$", labelpad=2)
    hb = [tuple(Line2D([0], [0], color=shade[r][si], lw=1.1) for r in rules) for si in range(len(rep_idx))]
    lb = [rf"$\Delta={deltas[j]:.2f}$" for j in rep_idx]
    hb.append(Line2D([0], [0], color="0.35", ls="--", lw=0.9)); lb.append(r"$\pm 2K$ lock")
    w_pt, h_pt = PW_RECT[2:]
    ax_b.legend(hb, lb, handler_map=tuple_map, loc="upper right",
                bbox_to_anchor=(1 + PW_LEG_SHIFT[0] / w_pt, 1 + PW_LEG_SHIFT[1] / h_pt),
                fontsize=5.0, handlelength=2.0, labelspacing=0.2)
    # (no label: the pair-weight panel is part of the (a) left-column group)

    # ── (b–e) coupling matrices: cos (top) / sin (bottom), 4 representative Δ ──
    #    one label per Δ column, placed on the top (cos) matrix. The colour scale is SHARED by the
    #    three Δ panels of a rule (so the panels are directly comparable) and linear and symmetric
    #    about the relaxation baseline A=1, where γ(1−A) vanishes: green = weakened, purple = potentiated.
    #    Ā itself varies with Δ for cos, which is why the scale is centred on the baseline, not on Ā.
    letters = iter("bcdefgh")
    for row, rule in enumerate(rules):
        ri = rules.index(rule)
        m = max(np.percentile(np.abs(Amat[ri, j] - 1.0), 99.8) for j in rep_idx)
        for cc, j in enumerate(rep_idx):
            ax = fig.add_axes(rect(MAT_X[cc], MAT_Y[row], MAT_W, MAT_W))
            im = ax.imshow(Amat[ri, j], cmap=cfg["cmap"], vmin=1.0 - m, vmax=1.0 + m,
                           origin="upper", interpolation="nearest")      # default aspect='equal' → square
            ax.set_yticks([])
            if row == 0:
                ax.set_xticks([])
                ax.set_title(rf"$\Delta={deltas[j]:.2f}$", fontsize=7, pad=2)
                _panel_label(ax, next(letters), dx=-8)
            else:                          # frequency range of the matrix axes (same for both rules)
                w = omega_axis[j]
                ax.set_xticks([0, N - 1])
                ax.set_xticklabels([f"{w.min():.0f}", f"{w.max():.0f}"])   # style default size
                ax.tick_params(axis="x", pad=1)
                if cc == 0:
                    ax.set_xlabel(r"$\omega_j$", labelpad=1)
            if cc == 0:
                ax.set_ylabel(rf"$G={RULE_TEX[rule]}$", labelpad=2, fontsize=7.5)
        # one colorbar per rule (the scale is shared along the row)
        cb = fig.colorbar(im, cax=fig.add_axes(rect(CB_X, MAT_Y[row], CB_W, MAT_W)))
        cb.ax.tick_params(labelsize=5)
        cb.outline.set_linewidth(0.4)
        if row == 0:                       # single label for both colorbars, on top of the upper one
            cb.ax.set_title(r"$A_{ij}$", pad=3)

    # ── (f) phase-coherence dynamics R(t) over the post-transient window, 4 Δ per rule ──
    #    single label for the whole row (on the leftmost / cos panel).
    for i, rule in enumerate(rules):
        ax = fig.add_axes(rect(*RT_RECTS[i]))
        for si, jb in enumerate(rep_idx):
            ax.plot(t_rec, Rt[i, jb], color=shade[rule][si], lw=0.8)
        ax.set_xlim(t_rec[0], t_rec[-1])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"time $t$", labelpad=1)
        ax.set_title(rf"$G={RULE_TEX[rule]}(x)$", fontsize=7, pad=2)
        if i == 0:
            ax.set_ylabel(r"phase coherence $R(t)$", labelpad=2)
            ax.legend([tuple(Line2D([0], [0], color=shade[r][si], lw=0.8) for r in rules)
                       for si in range(len(rep_idx))], [rf"$\Delta={deltas[j]:.2f}$" for j in rep_idx],
                      handler_map=tuple_map, loc="center right", fontsize=5.0, handlelength=2.0,
                      labelspacing=0.2)
            _panel_label(ax, next(letters), dx=-30)

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    fig.savefig(cfg["out"] + ".svg"); fig.savefig(cfg["out"] + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {cfg['out']}.svg / .png   (representative Δ = {[float(deltas[j]) for j in rep_idx]})")


if __name__ == "__main__":
    main()
