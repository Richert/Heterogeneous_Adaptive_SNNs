r"""
Allen QIF summary figure (one per neuron type, L2/3 vs L5/6)
============================================================

Two-column PRL-style figure with 2 rows (top = L2/3, bottom = L5/6) and, per row:
  col 1     : the fitted excitability distribution (empirical histogram + Lorentzian
              mixture PDF + its components)          [data_fitting/allen_lorentzian_<tag>.npz]
  col 2     : the 1-D bifurcation diagram s(I)       [allen_qif_bifurcation_<tag>.npz]
  col 3-4   : mean-field vs. spiking-network firing-rate dynamics
                                                     [qif_simulations/allen_qif_meanfield_<tag>.npz]
  col 5     : (spans both rows) the codim-2 bifurcation loci in the J–I plane, overlaying
              the L2/3 and L5/6 models in the same axes.

All inputs are the self-contained .npz files written by the fit / simulation / bifurcation
scripts, so this script needs only numpy + matplotlib (no PyRates / PyCoBi / Auto).

    python allen_qif_figure.py "Pyramidal"        # or "PV+ interneuron", "SOM interneuron"
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIT_DIR = os.path.join(_HERE, "..", "data_fitting")
_SIM_DIR = os.path.join(_HERE, "..", "qif_simulations")

CELL_CLASS = sys.argv[1] if len(sys.argv) > 1 else "Pyramidal"
LAYERS = ["L2/3", "L5/6"]
LIM_2D = dict(I_min=-200.0, I_max=400.0, J_min=50.0, J_max=150.0)   # codim-2 (J–I) panel limits

C_COMP = "#2e6f95"      # mixture components
C_MIX = "#c1121f"       # mixture sum / mean field
C_EQ = "#1F77B4"        # equilibrium branch
C_MICRO = "0.25"        # spiking network
C_LAYER = {"L2/3": "#1f77b4", "L5/6": "#ef7f1a"}   # 2-D panel: one colour per layer
M_FOLD, M_HOPF = "o", "s"


def _tag(cell_class, layer):
    c = cell_class.split("+")[0].split()[0].lower()
    return f"{c}_{layer.replace('/', '').replace(' ', '')}"


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "legend.fontsize": 5.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def _load(path):
    return np.load(path, allow_pickle=False) if os.path.exists(path) else None


# ════════════════════════════════════════════════════════════════════════════
#  per-panel plotters
# ════════════════════════════════════════════════════════════════════════════
def plot_pdf(ax, fit):
    if fit is None:
        ax.text(0.5, 0.5, "no fit", ha="center", va="center", transform=ax.transAxes); return
    xs = np.asarray(fit["samples"], float)
    w, Om, De = fit["weights"], fit["omega"], fit["delta"]
    gx = np.linspace(np.percentile(xs, 0.5), np.percentile(xs, 99.5), 500)
    comps = w[None, :] * (De[None, :] / np.pi) / ((gx[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    ax.hist(xs, bins=25, range=(gx[0], gx[-1]), density=True, color="0.85", edgecolor="none", zorder=0)
    for k in range(len(w)):
        ax.plot(gx, comps[:, k], color=C_COMP, lw=0.6, alpha=0.8, zorder=2)
    ax.plot(gx, comps.sum(axis=1), color=C_MIX, lw=1.3, zorder=3)
    ax.set_xlim(gx[0], gx[-1]); ax.set_yticks([])
    ax.set_xlabel(r"$v_\theta - v_r$ (mV)", labelpad=1)
    ax.set_ylabel("density", labelpad=2)


def plot_1d(ax, bif):
    if bif is None:
        ax.text(0.5, 0.5, "no bif.", ha="center", va="center", transform=ax.transAxes); return
    I, s, stab = bif["branch_I"], bif["branch_s"], bif["branch_stab"].astype(bool)
    step = np.hypot(np.diff(I), np.diff(s))
    jump = step > 6 * np.median(step[step > 0])           # guard against the bidirectional join
    for i in range(I.size - 1):
        if jump[i]:
            continue
        ax.plot(I[i:i + 2], s[i:i + 2], color=C_EQ, lw=1.3,
                ls="-" if stab[i] else "--", zorder=2)
    ax.scatter(bif["lp_I"], bif["lp_s"], marker=M_FOLD, s=20, facecolors="none",
               edgecolors="k", linewidths=1.0, zorder=5)
    ax.scatter(bif["hb_I"], bif["hb_s"], marker=M_HOPF, s=20, facecolors="none",
               edgecolors=C_MIX, linewidths=1.0, zorder=5)
    ax.set_xlabel(r"input $I$", labelpad=1)
    ax.set_ylabel(r"$s$", labelpad=2)


def plot_rate(ax, sim):
    if sim is None:
        ax.text(0.5, 0.5, "no sim.", ha="center", va="center", transform=ax.transAxes); return
    ax.axvspan(float(sim["t_on"]), float(sim["t_off"]), color="0.92", zorder=0)  # input pulse
    ax.plot(sim["t_micro"], sim["r_micro"], color=C_MICRO, lw=0.9, label="QIF network", zorder=2)
    ax.plot(sim["t_mf"], sim["r_mf"], color=C_MIX, lw=1.2, ls="--", label="mean field", zorder=3)
    ax.set_xlim(float(sim["t_micro"][0]), float(sim["t_micro"][-1]))
    ax.set_xlabel(r"time $t$", labelpad=1)
    ax.set_ylabel(r"firing rate $r(t)$", labelpad=2)


def plot_2d(ax, bif_by_layer):
    for layer in LAYERS:
        bif = bif_by_layer.get(layer)
        if bif is None:
            continue
        col = C_LAYER[layer]
        for k in range(int(bif["n_codim2"])):
            I2, J2 = bif[f"c2_{k}_I"], bif[f"c2_{k}_J"]
            kind = str(bif["c2_kinds"][k])
            ax.plot(I2, J2, color=col, lw=1.5, ls="-" if kind == "fold" else "--", zorder=2)
        ax.axhline(float(bif["J0"]), color=col, ls=":", lw=0.7, alpha=0.6, zorder=1)
    ax.set_xlim(LIM_2D["I_min"], LIM_2D["I_max"])
    ax.set_ylim(LIM_2D["J_min"], LIM_2D["J_max"])
    ax.set_xlabel(r"input $I$", labelpad=1)
    ax.set_ylabel(r"coupling $J$", labelpad=2)
    ax.set_title("(d)  codim-2 loci ($J$–$I$)", pad=3)
    handles = [Line2D([0], [0], color=C_LAYER[ly], lw=1.5, label=ly) for ly in LAYERS]
    handles += [Line2D([0], [0], color="0.4", lw=1.5, ls="-", label="fold"),
                Line2D([0], [0], color="0.4", lw=1.5, ls="--", label="Hopf")]
    ax.legend(handles=handles, loc="best", fontsize=5.5)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    set_prl_style()
    fig = plt.figure(figsize=(7.4, 3.4), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.08)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.25])

    col_titles = ["(a)  distribution fit", "(b)  1-D bifurcation", "(c)  firing-rate dynamics"]
    bif_by_layer = {}                            # layer -> bifurcation npz (for the codim-2 panel)

    for r, layer in enumerate(LAYERS):
        tag = _tag(CELL_CLASS, layer)
        fit = _load(os.path.join(_FIT_DIR, f"allen_lorentzian_{tag}.npz"))
        sim = _load(os.path.join(_SIM_DIR, f"allen_qif_meanfield_{tag}.npz"))
        bif = _load(os.path.join(_HERE, f"allen_qif_bifurcation_{tag}.npz"))
        bif_by_layer[layer] = bif

        ax_pdf = fig.add_subplot(gs[r, 0])
        ax_1d = fig.add_subplot(gs[r, 1])
        ax_rate = fig.add_subplot(gs[r, 2:4])
        plot_pdf(ax_pdf, fit)
        plot_1d(ax_1d, bif)
        plot_rate(ax_rate, sim)

        if r == 0:
            for ax, t in zip((ax_pdf, ax_1d, ax_rate), col_titles):
                ax.set_title(t, pad=3)
            ax_rate.legend(loc="upper left", fontsize=5.5)
        # row label (layer) at the left of each row
        ax_pdf.text(-0.42, 0.5, layer, transform=ax_pdf.transAxes, rotation=90,
                    ha="center", va="center", fontweight="bold", fontsize=8)

    ax_2d = fig.add_subplot(gs[:, 4])
    plot_2d(ax_2d, bif_by_layer)

    fig.suptitle(f"{CELL_CLASS}", fontsize=9, x=0.5, y=1.04)
    out = os.path.join(_HERE, f"allen_qif_summary_{_tag(CELL_CLASS, LAYERS[0]).split('_')[0]}")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"[saved] {os.path.basename(out)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
