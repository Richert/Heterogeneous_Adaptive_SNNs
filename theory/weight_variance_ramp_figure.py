r"""
Adaptive-coupling Kuramoto: K-ramp dynamics figure
==================================================

Two-column PRL figure of the K-ramp dynamics, 2 rows × 2 columns (columns = μ ∈ {0.001, 0.01}):
  * row 1 — forward-ramp phase coherence R(t): microscopic vs. mean field (Eqs. 8–10) by colour,
            with the mean-field V_A/Ā²(t) on a secondary y-axis;
  * row 2 — the same for the backward ramp.

The microscopic R(t) is loaded from the ramp .npz written by ``weight_variance_ramp.py``; the
(cheap, deterministic) mean-field ramp is recomputed here via ``weight_variance_ramp.mf_ramp`` from
the saved parameters, so the plot can be tuned without re-running the microscopic simulations.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp_figure.py
"""
import numpy as np
import matplotlib.pyplot as plt

MICRO_NPZ = "/home/rgast/data/kmo_adaptive/weight_variance_ramp_micro.npz"
MF_NPZ = "/home/rgast/data/kmo_adaptive/weight_variance_ramp_meanfield.npz"
OUT = "/home/rgast/data/kmo_adaptive/weight_variance_ramp"

C_MICRO, C_MF, C_VR = "0.2", "#c1121f", "#2a9d8f"                 # micro R, mean-field R, V_A/Ā²
DIRS = [(0, "fwd", "forward"), (1, "bwd", "backward")]


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


def main():
    dm = np.load(MICRO_NPZ)                                       # microscopic ramp data
    dmf = np.load(MF_NPZ)                                         # mean-field ramp data
    mus = dm["mus"]
    Tm, Rm = dm["Tm"], dm["Rm"]                                   # microscopic R(t) traces
    Tf, Rf, ratio = dmf["Tf"], dmf["Rf"], dmf["ratio"]           # mean-field R(t), V_A/Ā²(t)
    g, delta = float(dm["gamma"]), float(dm["delta"])
    Ttot = float(dm["n_ramp"]) * float(dm["tau_d"])

    set_prl_style()
    fig, axes = plt.subplots(2, len(mus), figsize=(7.0, 3.6), sharex=True, squeeze=False,
                             layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.12, hspace=0.06)
    handles = None
    for c, mu in enumerate(mus):
        for di, key, name in DIRS:                               # 0=fwd (row 0), 1=bwd (row 1)
            ax = axes[di][c]
            lmic, = ax.plot(Tm, Rm[c, di], color=C_MICRO, lw=0.9, label="micro $R$")
            lmf, = ax.plot(Tf, Rf[c, di], color=C_MF, lw=1.1, label="mean-field $R$")
            ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, Ttot)
            ax.set_title(rf"{name} ramp", fontsize=6.6, pad=2)
            if c == 0:
                ax.set_ylabel(r"$R(t)$", labelpad=2)
            if di == len(DIRS) - 1:
                ax.set_xlabel(r"time $t$", labelpad=1)

            tw = ax.twinx()                                      # mean-field V_A/Ā² on secondary axis
            lvr, = tw.plot(Tf, ratio[c, di], color=C_VR, lw=1.0, label=r"MF $V_A/\bar A^2$")
            tw.tick_params(axis="y", colors=C_VR); tw.spines["right"].set_color(C_VR)
            tw.set_ylim(bottom=0.0)
            if c == len(mus) - 1:
                tw.set_ylabel(r"$V_A/\bar A^2$", color=C_VR, labelpad=2)
            handles = [lmic, lmf, lvr]

    for c, mu in enumerate(mus):
        axes[0][c].annotate(rf"$\mu={mu:g}$", xy=(0.5, 1.22), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=7.5)
    for r in range(2):
        for c in range(len(mus)):
            _panel_label(axes[r][c], "abcd"[2 * r + c])
    axes[0][0].legend(handles=handles, loc="upper left", fontsize=5.2, handlelength=1.5,
                      labelspacing=0.2)

    fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
