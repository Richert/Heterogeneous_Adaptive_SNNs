r"""
Adaptive-coupling Kuramoto: combined K-ramp figure (bifurcation + ramp dynamics)
================================================================================

Single two-column PRL figure, 2 rows × 3 columns (rows = μ ∈ {0.001, 0.005}):
  * col 1 — phase-coherence bifurcation diagram R vs K (analytic branches + microscopic forward/
            backward ramp coherence markers; dashed vertical line at the critical coupling K_SN where
            the upper/synchronised branch emerges);
  * col 2 — FORWARD-ramp R(t): micro vs. mean field (Eqs. 8–10) by colour, with the mean-field
            V_A/Ā²(t) (solid) and C_A/Ā²(t) (dashed, same colour) on a secondary y-axis;
  * col 3 — the same for the BACKWARD ramp.

Loads the microscopic ramp data (weight_variance_ramp_micro.npz) and the mean-field ramp data
(weight_variance_ramp_meanfield.npz); analytic branches are recomputed from the saved parameters.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp_figure.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import weight_variance_bifurcation as B                            # analytic fixed-point branches

MICRO_NPZ = "/home/rgast/data/kmo_adaptive/weight_variance_ramp_micro.npz"
MF_NPZ = "/home/rgast/data/kmo_adaptive/weight_variance_ramp_meanfield.npz"
OUT = "/home/rgast/data/kmo_adaptive/weight_variance_ramp"

C_MICRO, C_MF, C_ASYNC, C_SYNC, C_VC = "0.2", "#c1121f", "0.55", "#1f77b4", "#2a9d8f"
DIRS = [(0, "fwd", "forward"), (1, "bwd", "backward")]


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 5.2, "xtick.labelsize": 6, "ytick.labelsize": 6,
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


def plot_R_bifurcation(ax, mu, g, delta, k_min, k_max, Ks_ramp, Rp_m):
    """Analytic R(K) branches + micro ramp markers + K_SN dashed line."""
    Ktc = 2.0 * delta
    Ks = np.linspace(k_min, k_max, 900)
    up, lo = np.full_like(Ks, np.nan), np.full_like(Ks, np.nan)
    for i, K in enumerate(Ks):
        A1, A2 = B.sync_abar(K, mu, g)
        for A, arr in ((A1, up), (A2, lo)):
            v = B.branch_values(K, A, mu, g, delta)
            if v:
                arr[i] = v["R"]
    m = Ks < Ktc
    ax.plot(Ks[m], np.zeros(m.sum()), color=C_ASYNC, lw=1.1, ls="-", zorder=2)      # async stable
    ax.plot(Ks[~m], np.zeros((~m).sum()), color=C_ASYNC, lw=1.0, ls="--", zorder=2)  # async unstable
    ax.plot(Ks, up, color=C_SYNC, lw=1.2, ls="-", zorder=3)                          # sync stable
    ax.plot(Ks, lo, color=C_SYNC, lw=1.0, ls="--", zorder=3)                         # sync saddle
    K_SN = 8.0 * g * mu * delta / (g + mu) ** 2                     # fold: upper branch emerges here
    ax.axvline(K_SN, color="0.4", ls="--", lw=0.8, zorder=1)
    for di, mk in ((0, "o"), (1, "s")):                            # micro ramp coherence (fwd/bwd)
        ax.plot(Ks_ramp, Rp_m[di], mk, ms=3.0, mfc="none", mec=C_MICRO, mew=0.8, ls="none", zorder=5)
    ax.set_xlim(k_min, k_max); ax.set_ylim(-0.03, 1.03)


def main():
    dm = np.load(MICRO_NPZ)                                        # microscopic ramp data
    dmf = np.load(MF_NPZ)                                          # mean-field ramp data
    mus = dm["mus"]
    Ks, Tm, Rm, Rp_m = dm["Ks"], dm["Tm"], dm["Rm"], dm["Rp_m"]
    Tf, Rf, VA, CA, A = dmf["Tf"], dmf["Rf"], dmf["VA"], dmf["CA"], dmf["A"]      # raw V_A, C_A (temporary)
    g, delta = float(dm["gamma"]), float(dm["delta"])
    k_min, k_max = float(dm["k_min"]), float(dm["k_max"])
    Ttot = float(dm["n_ramp"]) * float(dm["tau_d"])

    set_prl_style()
    fig, axes = plt.subplots(len(mus), 3, figsize=(7.0, 3.0), squeeze=False, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.03, wspace=0.05, hspace=0.08)
    col_title = ["coherence bifurcation", "forward ramp", "backward ramp"]
    r_handles = q_handles = None

    for i, mu in enumerate(mus):
        # ── col 1: R(K) bifurcation + ramp markers ────────────────────────────
        ax = axes[i][0]
        plot_R_bifurcation(ax, mu, g, delta, k_min, k_max, Ks, Rp_m[i])
        ax.set_ylabel(rf"$\mu={mu:g}$" + "\n" + r"$R$", labelpad=2)
        if i == len(mus) - 1:
            ax.set_xlabel(r"global coupling $K$", labelpad=1)

        # ── cols 2–3: forward / backward ramp R(t) + MF V_A/Ā², C_A/Ā² ─────────
        for di, key, name in DIRS:
            ax = axes[i][di + 1]
            lmic, = ax.plot(Tm, Rm[i, di], color=C_MICRO, lw=0.9, label="micro $R$")
            lmf, = ax.plot(Tf, Rf[i, di], color=C_MF, lw=1.1, label="mean-field $R$")
            ax.set_xlim(0, Ttot); ax.set_ylim(-0.03, 1.03)
            if i == len(mus) - 1:
                ax.set_xlabel(r"time $t$", labelpad=1)

            tw = ax.twinx()                                        # MF raw second moments (temporary)
            lvr, = tw.plot(Tf, VA[i, di] / A[i, di] ** 2, color=C_VC, lw=1.0, ls="-", label=r"$V_A$")
            lcr, = tw.plot(Tf, CA[i, di] / A[i, di] ** 2, color=C_VC, lw=1.0, ls=":", label=r"$C_A$")
            tw.tick_params(axis="y", colors=C_VC); tw.spines["right"].set_color(C_VC)
            tw.set_ylim(bottom=0.0)
            if di == 1:                                            # label only the rightmost twin
                tw.set_ylabel(r"$V_A / \bar A^2$,  $C_A / \bar A^2$", color=C_VC, labelpad=2)
            r_handles = [lmic, lmf]; q_handles = [lvr, lcr]

        if i == 0:
            for c in range(3):
                axes[0][c].set_title(col_title[c], fontsize=6.8, pad=2)

    # legends
    axes[0][0].legend(handles=[Line2D([0], [0], color=C_SYNC, lw=1.2, label="sync. (stable)"),
                               Line2D([0], [0], color=C_SYNC, lw=1.0, ls="--", label="sync. (saddle)"),
                               Line2D([0], [0], color=C_ASYNC, lw=1.1, label="async."),
                               Line2D([0], [0], marker="o", ls="none", mfc="none", mec=C_MICRO,
                                      mew=0.8, ms=3, label="ramp fwd"),
                               Line2D([0], [0], marker="s", ls="none", mfc="none", mec=C_MICRO,
                                      mew=0.8, ms=3, label="ramp bwd")],
                      loc="upper left", fontsize=4.4, handlelength=1.3, labelspacing=0.16)
    axes[0][1].legend(handles=r_handles + q_handles, loc="upper left", fontsize=4.8,
                      handlelength=1.5, labelspacing=0.16)

    for i in range(len(mus)):
        for c in range(3):
            _panel_label(axes[i][c], "abcdef"[3 * i + c])
    fig.savefig(OUT + ".svg"); fig.savefig(OUT + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {OUT}.svg / .png")


if __name__ == "__main__":
    main()
