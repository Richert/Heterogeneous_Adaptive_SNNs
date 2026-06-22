r"""
PV+ interneuron — figure 2 supplement: Hopf loci in (control parameter x heterogeneity h) planes
================================================================================================

Single heterogeneity knob h (build_circuit combined=True).  Three stacked panels, one per control
parameter (coupling J | input I | synaptic time constant tau_s); both PV layers overlaid by colour.
Reads the loci stored by pv_figure2_bifurcation.py:

    (a) (J, h)      (b) (I, h)      (c) (tau_s, h)

Shaded = oscillatory region (the polygon enclosed by the Hopf locus and the h=0 axis).  The dotted
line marks the common operating point (J=-100, I=430, tau_s=0.5).

Reads the pickled bif .npy, so run in the SAME env it was written in (``pycobi``):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python pv_figure2_loci_fig.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
CELL_CLASS = "PV+ Interneuron"
LAYERS = ["L2/3", "L5/6"]
C_LAYER = {"L2/3": "#1f77b4", "L5/6": "#e8702a"}

# one panel per control parameter: (locus key, x-label, xlim, operating-point value, ylim for h)
PANELS = [("J_h", r"coupling $J$", (-405.0, -40.0), -100.0, (0.0, 0.32)),
          ("Iext_h", r"input $I$", (280.0, 820.0), 430.0, (0.0, 0.42)),
          ("tau_s_h", r"synaptic time constant $\tau_s$", (0.0, 2.0), 0.5, (0.0, 1.0))]


def _tag(cell_class, layer):
    c = cell_class.split("+")[0].split()[0].lower()
    return f"{c}_{layer.replace('/', '').replace(' ', '')}"


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def _panel_label(ax, letter):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-16, 5),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


def _insert_seed(x, y, seed):
    """Insert the starting point at the junction (closest point) to bridge the bidirectional join."""
    if seed is None or x.size < 2:
        return x, y
    rx = np.ptp(x) or 1.0; ry = np.ptp(y) or 1.0
    j = int(np.argmin(((x - seed[0]) / rx) ** 2 + ((y - seed[1]) / ry) ** 2))
    return np.insert(x, j, seed[0]), np.insert(y, j, seed[1])


def _runs(mask):
    runs = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    return runs


def _shade_osc(ax, x, y, color, seed, ymax):
    """Shade the oscillatory region as the polygon enclosed by the Hopf locus and the h=0 axis."""
    x, y = _insert_seed(np.asarray(x, float), np.asarray(y, float), seed)
    phys = np.isfinite(x) & np.isfinite(y) & (y >= -2e-3)
    runs = _runs(phys)
    if not runs:
        return
    a, b = max(runs, key=lambda r: r[1] - r[0])
    xa, ya = x[a:b], np.clip(y[a:b], 0.0, ymax)
    if xa.size < 3:
        return
    px = np.concatenate([xa, [xa[-1], xa[0]]])
    py = np.concatenate([ya, [0.0, 0.0]])
    ax.fill(px, py, color=color, alpha=0.13, lw=0, zorder=1)


def _plot_curve(ax, x, y, color, seed, ymax):
    x, y = _insert_seed(np.asarray(x, float), np.asarray(y, float), seed)
    y = y.copy()
    y[(y < -2e-3) | (y > 1.04 * ymax)] = np.nan
    ax.plot(x, y, color=color, lw=1.4, zorder=3)


def main():
    set_prl_style()
    data = {}
    for layer in LAYERS:
        p = os.path.join(_HERE, f"pv_fig2_bif_{_tag(CELL_CLASS, layer)}.npy")
        data[layer] = np.load(p, allow_pickle=True).item() if os.path.exists(p) else None

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.8), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.12)

    for ax, (key, xlab, xlim, opx, ylim), letter in zip(axes, PANELS, "abc"):
        for layer in LAYERS:
            d = data[layer]
            if d is None:
                continue
            loc = d.get("loci", {}).get(key)
            if loc is not None:
                seed = loc.get("seed")
                _shade_osc(ax, loc["x"], loc["y"], C_LAYER[layer], seed, ylim[1])
                _plot_curve(ax, loc["x"], loc["y"], C_LAYER[layer], seed, ylim[1])
        ax.axvline(opx, color="0.6", ls=":", lw=0.8, zorder=1)       # common operating point
        ax.text(0.03, 0.93, "oscillatory", transform=ax.transAxes, ha="left", va="top",
                color="0.45", fontsize=5.5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(xlab, labelpad=1)
        ax.set_ylabel(r"heterogeneity $h$", labelpad=2)
        _panel_label(ax, letter)

    axes[0].legend(handles=[Line2D([0], [0], color=C_LAYER[ly], lw=1.4, label=ly) for ly in LAYERS]
                   + [Line2D([0], [0], color="0.6", ls=":", lw=0.8, label="operating point")],
                   loc="upper right", fontsize=5.5, handlelength=1.6, borderaxespad=0.3)

    out = os.path.join(_HERE, "pv_figure2_loci")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"[saved] {os.path.basename(out)}.{{png,svg}}")


if __name__ == "__main__":
    main()
