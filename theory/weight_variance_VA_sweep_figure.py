r"""
Adaptive-coupling Kuramoto: V_A micro-vs-mean-field sweep — figure
==================================================================

Loads the (K, μ) sweep written by ``weight_variance_VA_sweep.py`` and renders a single-column PRL
figure:
  * row 1 — RMSE between the microscopic and mean-field V_A(t), vs K; colour = μ (3 values),
            line style = mean-field version (C_A=C_S+C_F vs C_A=C_S);
  * rows 2–3 — V_A(t) traces for the min and max μ (columns) at two selected K (rows); micro vs.
            mean field by colour, the two MF versions by the same line styles as row 1.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_VA_sweep_figure.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NPZ = "/home/rgast/data/kmo_adaptive/weight_variance_VA_sweep.npz"
OUT = "/home/rgast/data/kmo_adaptive/weight_variance_VA_sweep"
K_PLOT = [2.5, 5.0]                 # the two K values shown in rows 2–3 (snapped to nearest in data)

C_MICRO, C_MF = "0.2", "#c1121f"
VERSIONS = ["full", "cs"]           # mean-field versions
V_STYLE = {"full": "-", "cs": "--"}                              # line style per version (shared)
V_LABEL = {"full": r"$C_A{=}C_S{+}C_F$", "cs": r"$C_A{=}C_S$"}
MU_CMAP = "viridis"


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 5.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
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


def rmse(a, b):
    """Time-domain RMSE between two V_A(t) traces."""
    return np.sqrt(np.mean((a - b) ** 2, axis=-1))


def main():
    d = np.load(NPZ)
    t, Ks, mus = d["t"], d["Ks"], d["mus"]
    VA = {"micro": d["VA_micro"], "full": d["VA_full"], "cs": d["VA_cs"]}   # (n_mu, n_K, n_t)
    delta, gamma = float(d["delta"]), float(d["gamma"])
    cmap = plt.get_cmap(MU_CMAP)
    mu_col = [cmap(0.15 + 0.7 * i / max(1, len(mus) - 1)) for i in range(len(mus))]
    k_idx = [int(np.argmin(np.abs(Ks - kp))) for kp in K_PLOT]
    mu_plot_idx = [0, len(mus) - 1]                                          # min & max μ (columns)

    set_prl_style()
    fig = plt.figure(figsize=(3.4, 4.7), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0])

    # ── row 1: RMSE(V_A) vs K — colour = μ, line style = MF version ───────────
    ax = fig.add_subplot(gs[0, :])
    for im, mu in enumerate(mus):
        for v in VERSIONS:
            ax.plot(Ks, rmse(VA["micro"][im], VA[v][im]), color=mu_col[im], ls=V_STYLE[v],
                    lw=1.0, marker="o", ms=2.0)
    ax.set_xlabel(r"global coupling $K$", labelpad=1)
    ax.set_ylabel(r"RMSE$\,[V_A(t)]$", labelpad=2)
    ax.set_xlim(Ks[0], Ks[-1]); ax.margins(y=0.1)
    leg_mu = ax.legend(handles=[Line2D([0], [0], color=mu_col[i], lw=1.4, label=rf"$\mu={mu:g}$")
                                for i, mu in enumerate(mus)],
                       loc="upper left", fontsize=5.4, handlelength=1.4, borderaxespad=0.3)
    ax.add_artist(leg_mu)
    ax.legend(handles=[Line2D([0], [0], color="0.35", lw=1.4, ls=V_STYLE[v], label=V_LABEL[v])
                       for v in VERSIONS],
              loc="upper right", fontsize=5.4, handlelength=1.8, borderaxespad=0.3)
    _panel_label(ax, "a")

    # ── rows 2–3: V_A(t) at (μ ∈ {min,max}) × (K ∈ K_PLOT) ────────────────────
    letters = iter("bcdef")
    for r, ik in enumerate(k_idx):
        for c, im in enumerate(mu_plot_idx):
            ax = fig.add_subplot(gs[r + 1, c])
            ax.plot(t, VA["micro"][im, ik], color=C_MICRO, lw=1.0)
            for v in VERSIONS:
                ax.plot(t, VA[v][im, ik], color=C_MF, lw=0.95, ls=V_STYLE[v])
            ax.set_xlim(t[0], t[-1])
            ax.set_title(rf"$\mu={mus[im]:g}$,  $K={Ks[ik]:.2f}$", fontsize=6.4, pad=2)
            if c == 0:
                ax.set_ylabel(r"$V_A(t)$", labelpad=2)
            if r == len(k_idx) - 1:
                ax.set_xlabel(r"time $t$", labelpad=1)
            if r == 0 and c == 0:
                ax.legend(handles=[Line2D([0], [0], color=C_MICRO, lw=1.0, label="microscopic")]
                          + [Line2D([0], [0], color=C_MF, lw=0.95, ls=V_STYLE[v], label="MF " + V_LABEL[v])
                             for v in VERSIONS],
                          loc="best", fontsize=5.0, handlelength=1.6, labelspacing=0.2)
            _panel_label(ax, next(letters))

    fig.suptitle(rf"$V_A(t)$: mean field vs. microscopic  ($\Delta={delta:g}$, $\gamma={gamma:g}$)",
                 fontsize=7.0)
    fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
