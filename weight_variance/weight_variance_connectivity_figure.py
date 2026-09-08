r"""
Adaptive-coupling Kuramoto: connectivity-structure figure (cos vs sin, μ=0.04)
===============================================================================

Loads the Δ-sweep written by ``weight_variance_connectivity_sweep.py`` (K=1, γ=0.01, μ=0.04; both
adaptation rules) and builds a two-column PRL figure. Panel labels follow the manuscript convention
(one per column-group):

  (a) left column   : raw weight variance V_A vs Δ (top) + raw pair weight vs frequency difference
                      ω_j−ω_i at 3 representative Δ (bottom); cos & sin.
  (b, c, d)         : the three coupling-matrix columns (cos top / sin bottom) at the representative Δ.
  (e) bottom row    : phase-coherence dynamics R(t) over the post-transient window (cos | sin), one
                      trace per representative Δ, at ~2/3 the height of the rows above.

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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from weight_variance_analysis import S_order, branches, set_prl_style, _panel_label  # noqa: E402

CONFIG = dict(
    npz=dp.mpmf("weight_variance_connectivity.npz"),
    rep_deltas=[0.25, 0.5, 0.75],      # representative Δ (nearest sweep points) for the matrices
    cmap="RdBu_r",                    # diverging coupling-matrix map (centred at Ā; blue = below, incl. <0)
    out=dp.mpmf("weight_variance_connectivity"),
)

C_RULE = {"cos": "#0072B2", "sin": "#D55E00"}     # colour-blind-safe (Wong) blue / vermilion
RULE_TEX = {"cos": r"\cos", "sin": r"\sin"}


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
    VA, Amat, Abar, omega_axis = d["VA"], d["Amat"], d["Abar"], d["omega_axis"]
    Rt, t_rec = d["Rt"], d["t_rec"]
    mu, g, K, N = float(d["mu"]), float(d["gamma"]), float(d["K"]), int(d["N"])
    rep_idx = [int(np.argmin(np.abs(deltas - t))) for t in cfg["rep_deltas"]]
    delta_colors = [plt.get_cmap("Set2")(x) for x in np.arange(0, len(rep_idx))]

    set_prl_style()
    fig = plt.figure(figsize=(7.6, 4.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 0.667], hspace=0.38,
                             left=0.062, right=0.955, top=0.955, bottom=0.085)
    gs_top = outer[0].subgridspec(2, 4, width_ratios=[1.4, 1.0, 1.0, 1.0], wspace=0.28, hspace=0.20)
    gs_bot = outer[1].subgridspec(1, 2, wspace=0.16)

    # ── (a) raw weight variance V_A vs Δ: micro (solid + markers) + MF (dashed) ──
    ax_va = fig.add_subplot(gs_top[0, 0])
    for si, jb in enumerate(rep_idx):                                # Δ shown in panels b, c, d
        ax_va.axvline(deltas[jb], color=delta_colors[si], ls=":", lw=0.9, zorder=1)
    dc = np.linspace(max(deltas.min(), 1e-3), deltas.max(), 400)
    for i, rule in enumerate(rules):
        ax_va.plot(deltas, VA[i], "-o", color=C_RULE[rule], lw=1.2, ms=2.6, zorder=3)
        ax_va.plot(dc, mf_va(rule, dc, K, mu, g), ls="--", color=C_RULE[rule], lw=1.1, zorder=2)
    ax_va.set_yscale("log")
    ax_va.set_xlim(0, deltas.max() * 1.02)
    ax_va.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
    ax_va.set_ylabel(r"weight variance $V_A$", labelpad=2)
    hleg = [Line2D([0], [0], color=C_RULE["cos"], lw=1.2, marker="o", ms=2.6, label=r"$G=\cos$"),
            Line2D([0], [0], color=C_RULE["sin"], lw=1.2, marker="o", ms=2.6, label=r"$G=\sin$"),
            Line2D([0], [0], color="0.45", lw=1.1, ls="--", label="mean field")]
    ax_va.legend(handles=hleg, loc="best", fontsize=5.6, handlelength=1.6, labelspacing=0.2)
    _panel_label(ax_va, "a", dx=-30)

    # ── (b) raw pair weight vs frequency difference ω_j−ω_i, at the 3 representative Δ ──
    #    colour = rule, line style = Δ; binned mean over all off-diagonal entries. The two vertical
    #    dashed lines at ±2K mark where TWO ISOLATED oscillators (bidirectional coupling strength K)
    #    would lock: φ̇ = Δω − 2K sin φ ⇒ locked iff |ω_j−ω_i| ≤ 2K (here K=1 ⇒ ±2).
    ax_b = fig.add_subplot(gs_top[1, 0])
    off = ~np.eye(N, dtype=bool)
    ls_delta = ["-", "--", ":"]
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
            ax_b.plot(xc, m, ls=ls_delta[si], color=C_RULE[rule], lw=1.1, zorder=3)
    for sgn in (-1.0, 1.0):                                          # two-oscillator locking |Δω|≤2K
        ax_b.axvline(sgn * 2.0 * K, color="0.35", ls="--", lw=0.9, zorder=1)
    ax_b.axhline(1.0, color="0.6", lw=0.4, zorder=0)                 # relaxation baseline A→1
    ax_b.set_xlim(-7.0, 7.0)
    ax_b.set_xlabel(r"frequency difference $\omega_j-\omega_i$", labelpad=1)
    ax_b.set_ylabel(r"pair weight $A_{ij}$", labelpad=2)
    hb = [Line2D([0], [0], color="0.35", ls=ls_delta[si], label=rf"$\Delta={deltas[j]:.2f}$")
          for si, j in enumerate(rep_idx)]
    hb.append(Line2D([0], [0], color="0.35", ls="--", lw=0.9, label=r"$\pm 2K$ lock"))
    ax_b.legend(handles=hb, loc="best", fontsize=5.0, handlelength=2.0, labelspacing=0.2)
    # (no label: the pair-weight panel is part of the (a) left-column group)

    # ── (b, c, d) coupling matrices: cos (top) / sin (bottom), 3 representative Δ ──
    #    one label per Δ column, placed on the top (cos) matrix.
    letters = iter("bcd")
    for row, rule in enumerate(rules):
        ri = rules.index(rule)
        for cc, j in enumerate(rep_idx):
            ax = fig.add_subplot(gs_top[row, cc + 1])
            A = Amat[ri, j]
            m = np.percentile(np.abs(A - Abar[ri, j]), 99.0)              # symmetric range about Ā
            im = ax.imshow(A, cmap=cfg["cmap"], vmin=Abar[ri, j] - m, vmax=Abar[ri, j] + m,
                           origin="upper", interpolation="nearest")      # default aspect='equal' → square
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(rf"$\Delta={deltas[j]:.2f}$", fontsize=7, pad=2)
                _panel_label(ax, next(letters), dx=-8)
            if cc == 0:
                ax.set_ylabel(rf"$G={RULE_TEX[rule]}$", labelpad=2, fontsize=7.5)
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)  # matched colorbar
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=5)
            cb.outline.set_linewidth(0.4)

    # ── (e) phase-coherence dynamics R(t) over the post-transient window, 3 Δ per rule ──
    #    single label for the whole row (on the leftmost / cos panel).
    for i, rule in enumerate(rules):
        ax = fig.add_subplot(gs_bot[0, i])
        for si, jb in enumerate(rep_idx):
            ax.plot(t_rec, Rt[i, jb], color=delta_colors[si], lw=0.8,
                    label=rf"$\Delta={deltas[jb]:.2f}$")
        ax.set_xlim(t_rec[0], t_rec[-1])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"time $t$", labelpad=1)
        ax.set_title(rf"$G={RULE_TEX[rule]}(x)$", fontsize=7, pad=2)
        if i == 0:
            ax.set_ylabel(r"phase coherence $R(t)$", labelpad=2)
            ax.legend(loc="best", fontsize=5.0, handlelength=2.0, labelspacing=0.2)
            _panel_label(ax, "e", dx=-30)

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    fig.savefig(cfg["out"] + ".svg"); fig.savefig(cfg["out"] + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {cfg['out']}.svg / .png   (representative Δ = {[float(deltas[j]) for j in rep_idx]})")


if __name__ == "__main__":
    main()
