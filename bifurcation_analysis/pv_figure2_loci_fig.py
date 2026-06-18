r"""
PV+ interneuron — figure 2 supplement: Hopf loci in (drive/coupling x heterogeneity) planes
============================================================================================

2x2 PRL figure (columns = coupling J | drive I; rows = width h_Delta | centre h_eta), both
PV layers overlaid by colour.  Reads the loci stored by pv_figure2_bifurcation.py:

    (a) (J, h_Delta)   (b) (I, h_Delta)      at h_eta=0
    (c) (J, h_eta)      (d) (I, h_eta)         at h_Delta=h_lo (the mid-wedge / rate-sim value)

Each Hopf curve separates the oscillatory region (smaller heterogeneity, below the curve) from
the quiescent region.  Dotted reference lines mark the regime operating point (J=-200; I=I_fix).

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

# panel grid: (row key -> y param, label, ylim) x (col key -> x param, label, xlim)
ROWS = [("hD", r"width scaling $h_\Delta$", (0.0, 0.24)),
        ("hC", r"centre scaling $h_\eta$", (0.0, 0.12))]
COLS = [("J", r"coupling $J$", (-405.0, -40.0)),
        ("Iext", r"input $I$", (280.0, 720.0))]


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
    """Insert the starting point at the junction (closest point) to bridge the gap where the
    two bidirectional halves of a continuation meet."""
    if seed is None or x.size < 2:
        return x, y
    rx = np.ptp(x) or 1.0; ry = np.ptp(y) or 1.0
    j = int(np.argmin(((x - seed[0]) / rx) ** 2 + ((y - seed[1]) / ry) ** 2))
    return np.insert(x, j, seed[0]), np.insert(y, j, seed[1])


def _runs(mask):
    """Contiguous True runs of a boolean mask as (start, stop) index pairs."""
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
    """Shade the oscillatory region as the polygon enclosed by the Hopf locus and the h=0 axis.
    Closing the (physical, contiguous) locus along the axis yields the correct region in every
    case: below the curve where it spans the axis, and the TONGUE INTERIOR where the locus folds
    away from the axis (so the area below the lower branch is left quiescent).  Off-axis excursions
    are clipped to the top; genuine negative (unphysical) excursions are dropped."""
    x, y = _insert_seed(np.asarray(x, float), np.asarray(y, float), seed)
    phys = np.isfinite(x) & np.isfinite(y) & (y >= -2e-3)
    runs = _runs(phys)
    if not runs:
        return
    a, b = max(runs, key=lambda r: r[1] - r[0])                  # the main physical arc
    xa, ya = x[a:b], np.clip(y[a:b], 0.0, ymax)
    if xa.size < 3:
        return
    px = np.concatenate([xa, [xa[-1], xa[0]]])                   # close along the h=0 axis
    py = np.concatenate([ya, [0.0, 0.0]])
    ax.fill(px, py, color=color, alpha=0.13, lw=0, zorder=1)


def _plot_curve(ax, x, y, color, seed, ymax):
    """Plot a continuation curve: seed-insert to bridge the bidirectional join, then mask the
    unphysical (negative) and off-axis (> top) excursions so the drawn line stays in the panel."""
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

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.2), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.08)

    letters = iter("abcd")
    for ri, (yk, ylab, ylim) in enumerate(ROWS):
        for ci, (xk, xlab, xlim) in enumerate(COLS):
            ax = axes[ri, ci]
            key = f"{xk}_{yk}"
            for layer in LAYERS:
                d = data[layer]
                if d is None:
                    continue
                loc = d.get("loci", {}).get(key)
                if loc is not None:
                    seed = loc.get("seed")
                    _shade_osc(ax, loc["x"], loc["y"], C_LAYER[layer], seed, ylim[1])
                    _plot_curve(ax, loc["x"], loc["y"], C_LAYER[layer], seed, ylim[1])
            # regime reference lines: J=-200 (shared) | I=I_fix (per layer)
            if xk == "J":
                ax.axvline(-200.0, color="0.6", ls=":", lw=0.8, zorder=1)
            else:
                for layer in LAYERS:
                    if data[layer] is not None:
                        ax.axvline(float(data[layer]["I_fix"]), color=C_LAYER[layer],
                                   ls=":", lw=0.7, alpha=0.7, zorder=1)
            ax.text(0.04, 0.06, "oscillatory", transform=ax.transAxes, ha="left", va="bottom",
                    color="0.45", fontsize=5.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            if ci == 0:
                ax.set_ylabel(ylab, labelpad=2)
            if ri == len(ROWS) - 1:
                ax.set_xlabel(xlab, labelpad=1)
            _panel_label(ax, next(letters))

    axes[0, 1].legend(handles=[Line2D([0], [0], color=C_LAYER[ly], lw=1.4, label=ly) for ly in LAYERS]
                      + [Line2D([0], [0], color="0.6", ls=":", lw=0.8, label="operating point")],
                      loc="upper right", fontsize=5.5, handlelength=1.6, borderaxespad=0.3)

    out = os.path.join(_HERE, "pv_figure2_loci")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"[saved] {os.path.basename(out)}.{{png,svg}}")


if __name__ == "__main__":
    main()
