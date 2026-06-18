r"""
Allen QIF summary figure (one per neuron type, L2/3 vs L5/6)
============================================================

Single-column PRL-style figure, 3 rows, with the two layers distinguished by COLOUR
throughout (L2/3 vs L5/6):

  row 1 : the fitted excitability distributions, L2/3 (left) and L5/6 (right) side by side
          (empirical histogram + Lorentzian-mixture PDF + components, mixture drawn in the
          layer colour).                              [data_fitting/allen_lorentzian_<tag>.npz]
  row 2 : (left)  both 1-D bifurcation diagrams s(I) overlaid in one axis, layer colour,
                  stability via line style (solid = stable, dashed = unstable), fold markers;
          (right) the codim-2 bifurcation loci in the J–I plane, same colour scheme.
                                                       [allen_qif_bifurcation_<tag>.npz]
  row 3 : (spans both columns) the firing-rate dynamics overlaid: mean field (solid) and
          spiking network (dashed, reduced alpha) in the matching layer colour.
                                                       [qif_simulations/allen_qif_meanfield_<tag>.npz]

Reads only the self-contained .npz files (numpy + matplotlib; no PyRates / PyCoBi / Auto):

    python allen_qif_figure.py "Pyramidal"        # or "PV+ interneuron", "SOM interneuron"
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIT_DIR = os.path.join(_HERE, "..", "data_fitting")
_SIM_DIR = os.path.join(_HERE, "..", "qif_simulations")

CELL_CLASS = sys.argv[1] if len(sys.argv) > 1 else "Pyramidal"
LAYERS = ["L2/3", "L5/6"]
C_LAYER = {"L2/3": "#1f77b4", "L5/6": "#e8702a"}     # one colour per layer (used everywhere)
LIM_1D = dict(I_min=-100.0, I_max=400.0, r_min=0.0, r_max=10.0)   # codim-1 (I-r) panel limits
LIM_2D = dict(I_min=-200.0, I_max=400.0, J_min=50.0, J_max=150.0)   # codim-2 (J–I) panel limits
C_HIST = "0.85"
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


def _panel_label(ax, letter):
    """PRL-style bold panel label in the top-left corner."""
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold")


# ════════════════════════════════════════════════════════════════════════════
#  per-panel plotters
# ════════════════════════════════════════════════════════════════════════════
def plot_pdf(ax, fit, color, layer):
    if fit is None:
        ax.text(0.5, 0.5, "no fit", ha="center", va="center", transform=ax.transAxes); return
    xs = np.asarray(fit["samples"], float)
    w, Om, De = fit["weights"], fit["omega"], fit["delta"]
    gx = np.linspace(np.percentile(xs, 0.5), np.percentile(xs, 99.5), 500)
    comps = w[None, :] * (De[None, :] / np.pi) / ((gx[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    ax.hist(xs, bins=25, range=(gx[0], gx[-1]), density=True, color=C_HIST, edgecolor="none", zorder=0)
    for k in range(len(w)):
        ax.plot(gx, comps[:, k], color=color, lw=0.5, alpha=0.45, zorder=2)
    ax.plot(gx, comps.sum(axis=1), color=color, lw=1.4, zorder=3)
    ax.set_xlim(gx[0], gx[-1]); ax.set_yticks([])
    ax.set_xlabel(r"$\eta$ (mV)", labelpad=1)
    ax.set_title(layer, color=color, fontweight="bold", pad=2)


def _plot_branch(ax, I, s, stab, color):
    """Draw an equilibrium branch coloured by `color`, solid where stable / dashed where
    unstable. Each contiguous-stability RUN is one polyline (so a dashed run shows real
    dashes — drawing per-2-point-segment would render every short dash as solid), bridged
    by one point to close the folds, and broken at any large jump (bidirectional merge)."""
    stab = np.asarray(stab, bool)
    step = np.hypot(np.diff(I), np.diff(s))
    thr = 6 * np.median(step[step > 0]) if np.any(step > 0) else np.inf
    flips = list(np.where(np.diff(stab.astype(int)) != 0)[0] + 1)
    bounds = [0] + flips + [stab.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        e = min(b + 1, stab.size)                          # +1 bridges the fold to the next run
        y = s[a:e].astype(float).copy()
        for L in np.where(step[a:e - 1] > thr)[0]:         # break inside the run at a big jump
            y[L + 1] = np.nan
        ax.plot(I[a:e], y, color=color, lw=1.2, ls="-" if stab[a] else "--", zorder=2)


def plot_1d(ax, bif_by_layer, lim, input_levels=()):
    for layer in LAYERS:
        bif = bif_by_layer.get(layer)
        if bif is None:
            continue
        col = C_LAYER[layer]
        # s = a = Σ_m w_m r_m at the equilibrium, so the synaptic activation IS the mean rate r
        _plot_branch(ax, bif["branch_I"], bif["branch_s"], bif["branch_stab"], col)
        ax.scatter(bif["lp_I"], bif["lp_s"], marker=M_FOLD, s=16, facecolors="none",
                   edgecolors=col, linewidths=0.9, zorder=5)
    # the input levels used in the rate simulation (vertical, since I is the x-axis here)
    for Iv in input_levels:
        ax.axvline(float(Iv), color="0.55", ls=":", lw=1.0, zorder=1)
    ax.set_xlabel(r"input $I$", labelpad=1)
    ax.set_ylabel(r"firing rate $r$", labelpad=2)
    ax.legend(handles=[
        Line2D([0], [0], color="0.4", ls="-", lw=1.0, label="stable"),
        Line2D([0], [0], color="0.4", ls="--", lw=1.0, label="unstable"),
        Line2D([0], [0], marker=M_FOLD, color="0.3", lw=0, markerfacecolor="none", label="fold"),
    ], loc="best", fontsize=5.5, handlelength=1.4)
    ax.set_xlim(lim["I_min"], lim["I_max"])
    ax.set_ylim(lim["r_min"], lim["r_max"])


def plot_2d(ax, bif_by_layer, lim):
    for layer in LAYERS:
        bif = bif_by_layer.get(layer)
        if bif is None:
            continue
        col = C_LAYER[layer]
        for k in range(int(bif["n_codim2"])):
            kind = str(bif["c2_kinds"][k])
            ax.plot(bif[f"c2_{k}_I"], bif[f"c2_{k}_J"], color=col, lw=1.4,
                    ls="-" if kind == "fold" else "--", zorder=2)
    # horizontal dotted line at J0: the J-level of the 1-D cut shown in the 1-D bifurcation panel
    J0 = next((float(b["J0"]) for b in bif_by_layer.values() if b is not None), None)
    if J0 is not None:
        ax.axhline(J0, color="0.55", ls=":", lw=1.0, zorder=1)
        ax.text(lim["I_min"], J0, " 1-D cut", color="0.45", ha="left", va="bottom", fontsize=5.5)
    ax.set_xlim(lim["I_min"], lim["I_max"])
    ax.set_ylim(lim["J_min"], lim["J_max"])
    ax.set_xlabel(r"input $I$", labelpad=1)
    ax.set_ylabel(r"coupling $J$", labelpad=2)
    ax.legend(handles=[Line2D([0], [0], color=C_LAYER[ly], lw=1.4, label=ly) for ly in LAYERS]
              + [Line2D([0], [0], color="0.4", ls="-", lw=1.2, label="fold")],
              loc="best", fontsize=5.5, handlelength=1.4)


def plot_rate(ax, sim_by_layer):
    shaded = False; ann = None
    for layer in LAYERS:
        sim = sim_by_layer.get(layer)
        if sim is None:
            continue
        col = C_LAYER[layer]
        if not shaded:
            ax.axvspan(float(sim["t_on"]), float(sim["t_off"]), color="0.93", zorder=0)
            ann = (float(sim["I0"]), float(sim["I1"]), float(sim["t_on"]), float(sim["t_off"]),
                   float(sim["t_mf"][0]), float(sim["t_mf"][-1]))
            shaded = True
        ax.plot(sim["t_mf"], sim["r_mf"], color=col, lw=1.2, ls="-", zorder=3)            # mean field
        ax.plot(sim["t_micro"], sim["r_micro"], color=col, lw=0.9, ls="--", alpha=0.55, zorder=2)  # spiking
    # label the input level I in each of the three time windows (baseline | pulse | baseline)
    if ann is not None:
        I0, I1, t_on, t_off, t0, t1 = ann
        for xc, Iv in [(0.5 * (t0 + t_on), I0), (0.5 * (t_on + t_off), I1), (0.5 * (t_off + t1), I0)]:
            ax.text(xc, 0.5, rf"$I={Iv:.0f}$", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", fontsize=6, color="0.35")
    ax.set_xlabel(r"time $t$", labelpad=1)
    ax.set_ylabel(r"firing rate $r(t)$", labelpad=2)
    ax.legend(handles=[
        Line2D([0], [0], color="0.3", ls="-", lw=1.2, label="mean field"),
        Line2D([0], [0], color="0.3", ls="--", lw=0.9, alpha=0.55, label="QIF network"),
    ], loc="upper right", fontsize=5.5, handlelength=1.6)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    set_prl_style()
    fig = plt.figure(figsize=(3.4, 4.4), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.10)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9])

    fit_by_layer, bif_by_layer, sim_by_layer = {}, {}, {}
    for layer in LAYERS:
        tag = _tag(CELL_CLASS, layer)
        fit_by_layer[layer] = _load(os.path.join(_FIT_DIR, f"allen_lorentzian_{tag}.npz"))
        sim_by_layer[layer] = _load(os.path.join(_SIM_DIR, f"allen_qif_meanfield_{tag}.npz"))
        bif_by_layer[layer] = _load(os.path.join(_HERE, f"allen_qif_bifurcation_{tag}.npz"))

    panels = []
    # row 1: the two distribution fits side by side
    for c, layer in enumerate(LAYERS):
        ax = fig.add_subplot(gs[0, c])
        plot_pdf(ax, fit_by_layer[layer], C_LAYER[layer], layer)
        if c == 0:
            ax.set_ylabel(r"$\rho(\eta)$", labelpad=2)
        panels.append(ax)

    # row 2: 1-D bifurcation (overlaid) | codim-2 (J–I)
    levels = next(((float(s["I0"]), float(s["I1"])) for s in sim_by_layer.values() if s is not None), ())
    ax1d = fig.add_subplot(gs[1, 0]); plot_1d(ax1d, bif_by_layer, LIM_1D, levels)
    ax2d = fig.add_subplot(gs[1, 1]); plot_2d(ax2d, bif_by_layer, LIM_2D)
    panels += [ax1d, ax2d]

    # row 3: firing-rate dynamics (overlaid, spanning both columns)
    axr = fig.add_subplot(gs[2, :]); plot_rate(axr, sim_by_layer)
    panels.append(axr)

    for ax, letter in zip(panels, "abcdefg"):
        _panel_label(ax, letter)
    out = os.path.join(_HERE, f"allen_qif_summary_{_tag(CELL_CLASS, LAYERS[0]).split('_')[0]}")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"[saved] {os.path.basename(out)}.{{png,svg}}")


if __name__ == "__main__":
    main()
