r"""
Adaptive-coupling Kuramoto: 1-D bifurcation diagrams vs K (Section D fixed points)
=================================================================================

Single-column PRL figure, 3 rows × 2 columns (columns = μ), of the analytic mean-field fixed points
(PRL_2026 "Weight Variance", Section D) vs the global coupling K, at fixed Δ=1, γ=0.001:
  * row 1 — V_A/Ā²        (weight-variance ratio),
  * row 2 — C_A/(Ā R²)    (fractional covariance / coupling-error driver),
  * row 3 — R             (phase coherence).
Asynchronous branch (R=0, Ā=1) in grey (solid=stable K<2Δ / dashed=unstable); synchronous branch in
blue (solid=stable node Ā₁ / dashed=unstable saddle Ā₂).  C_A/(Ā R²) is undefined on the async branch
(R=0) and shown only on the synchronous branch.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp_bifurcation.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import weight_variance_bifurcation as B                            # analytic fixed-point branches

CONFIG = dict(
    gamma=0.001, delta=1.0, mus=[0.001, 0.005], N=500,
    k_min=0.3, k_max=3.0,
    npz="/home/rgast/data/kmo_adaptive/weight_variance_ramp_micro.npz",   # for the micro ramp markers
    out="/home/rgast/data/kmo_adaptive/weight_variance_ramp_bifurcation",
)

C_ASYNC, C_SYNC, C_MICRO = "0.55", "#1f77b4", "0.2"
# (key, y-label, value func(branch_values, R_floor), async value, stable_only)
#   R uses a finite-size floor in C_A/(Ā R²) so the async stable branch evaluates to 0 (C_A=0)
QSPEC = [
    ("ratio",  r"$V_A/\bar A^2$",     lambda v, rf: v["VA"] / v["Abar"] ** 2,                  0.0, True),
    ("cratio", r"$C_A/(\bar A R^2)$", lambda v, rf: v["CA"] / (v["Abar"] * max(v["R"], rf) ** 2), 0.0, True),
    ("R",      r"$R$",                lambda v, rf: v["R"],                                    0.0, False),
]


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 5.4, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def _panel_label(ax, letter, dx=-26, dy=3):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


def plot_quantity(ax, mu, cfg, func, async_val, stable_only, R_floor):
    g, delta = cfg["gamma"], cfg["delta"]
    Ktc = 2.0 * delta
    Ks = np.linspace(cfg["k_min"], cfg["k_max"], 900)
    up, lo = np.full_like(Ks, np.nan), np.full_like(Ks, np.nan)
    for i, K in enumerate(Ks):
        A1, A2 = B.sync_abar(K, mu, g)
        for A, arr in ((A1, up), (A2, lo)):
            v = B.branch_values(K, A, mu, g, delta)
            if v:
                arr[i] = func(v, R_floor)
    if async_val is not None:                                      # async branch (R=0, Ā=1)
        m = Ks < Ktc
        ax.plot(Ks[m], np.full(m.sum(), async_val), color=C_ASYNC, lw=1.1, ls="-", zorder=2)  # stable
        if not stable_only:                                        # unstable async (K>2Δ)
            ax.plot(Ks[~m], np.full((~m).sum(), async_val), color=C_ASYNC, lw=1.0, ls="--", zorder=2)
    ax.plot(Ks, up, color=C_SYNC, lw=1.2, ls="-", zorder=3)        # synchronous stable node
    if not stable_only:                                            # synchronous saddle (unstable)
        ax.plot(Ks, lo, color=C_SYNC, lw=1.0, ls="--", zorder=3)
    ax.set_xlim(cfg["k_min"], cfg["k_max"])


def main(cfg=CONFIG):
    set_prl_style()
    mus = cfg["mus"]
    ramp = np.load(cfg["npz"]) if cfg.get("npz") and os.path.exists(cfg["npz"]) else None
    R_floor = 1.0 / np.sqrt(int(ramp["N"]) if ramp is not None else cfg["N"])

    fig, axes = plt.subplots(len(QSPEC), len(mus), figsize=(3.4, 2.95), sharex=True,
                             squeeze=False, layout="constrained")
    lab = iter("abcdef")
    for r, (key, ylab, func, aval, stable_only) in enumerate(QSPEC):
        for c, mu in enumerate(mus):
            ax = axes[r][c]
            plot_quantity(ax, mu, cfg, func, aval, stable_only, R_floor)
            if key == "R" and ramp is not None:                    # micro ramp coherence estimates
                for di, mk in ((0, "o"), (1, "s")):                # 0=fwd, 1=bwd (ascending-K order)
                    ax.plot(ramp["Ks"], ramp["Rp_m"][c, di], mk, ms=3.0, mfc="none", mec=C_MICRO,
                            mew=0.8, ls="none", zorder=5)
                ax.set_ylim(-0.03, 1.03)
            if c == 0:
                ax.set_ylabel(ylab, labelpad=2)
            if r == 0:
                ax.set_title(rf"$\mu={mu:g}$", fontsize=6.8, pad=2)
            if r == len(QSPEC) - 1:
                ax.set_xlabel(r"global coupling $K$", labelpad=1)
            _panel_label(ax, next(lab))
    axes[2][0].legend(handles=[Line2D([0], [0], color=C_SYNC, lw=1.2, label="sync. (stable)"),
                               Line2D([0], [0], color=C_SYNC, lw=1.0, ls="--", label="sync. (saddle)"),
                               Line2D([0], [0], color=C_ASYNC, lw=1.1, label="async."),
                               Line2D([0], [0], marker="o", ls="none", mfc="none", mec=C_MICRO,
                                      mew=0.8, ms=3, label="ramp fwd"),
                               Line2D([0], [0], marker="s", ls="none", mfc="none", mec=C_MICRO,
                                      mew=0.8, ms=3, label="ramp bwd")],
                      loc="upper left", fontsize=4.6, handlelength=1.4, labelspacing=0.18)
    fig.suptitle(rf"Mean-field fixed points  ($\Delta={cfg['delta']:g}$, $\gamma={cfg['gamma']:g}$)",
                 fontsize=7.5)
    fig.savefig(cfg["out"] + ".svg"); fig.savefig(cfg["out"] + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {cfg['out']}.svg / .png")


if __name__ == "__main__":
    main()
