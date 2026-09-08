r"""
Pyramidal journal figure (updated to the heterogeneity / multi-stability story)
================================================================================
Same single-column 3-row layout as the original Allen-QIF summary figure, but recast for the
J=100, tau_s=0.5, width-heterogeneity (h_Delta) regime:

  row 1 : the two fitted excitability distributions, L2/3 | L5/6 (layer colour).
  row 2 : (left)  both 1-D bifurcation diagrams s(I) at h_Delta=0.1 overlaid (multi-stable),
                  stability via line style, fold markers;        [pyramidal_fig_bif_<tag>.npz]
          (right) the fold loci in the (I, h_Delta) plane, both layers in one axis; inside every
                  patch a FRACTION  (#stable equilibria L2/3) / (#stable equilibria L5/6),
                  numbers in the layer colours, separated by a grey bar.
  row 3 : (spans columns) the rate dynamics r(t) under the input protocol that walks L2/3 through
          three stable states and L5/6 through two; mean field (solid) + QIF net (dashed), log-y.
                                                                  [pyramidal_fig_rate_<tag>.npz]

Reads only self-contained .npz (numpy + matplotlib + scipy.ndimage):
    python pyramidal_fig.py            # in the ``allen`` env
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIT_DIR = os.path.join(_HERE, "..", "data_fitting")
_SIM_DIR = os.path.join(_HERE, "..", "qif_simulations")

CELL_CLASS = "Pyramidal"
LAYERS = ["L2/3", "L5/6"]
C_LAYER = {"L2/3": "#1f77b4", "L5/6": "#e8702a"}
C_HIST = "0.85"
LIM_1D = dict(I_min=0.0, I_max=400.0, s_min=8e-3, s_max=14.0)
LIM_2D = dict(I_min=40.0, I_max=300.0, h_min=0.0, h_max=1.02)
M_FOLD = "o"


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
    """Bold PRL-style panel label OUTSIDE the axis box, above its top-left corner."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-16, 5),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


# ════════════════════════════════════════════════════════════════════════════
#  panels
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
    """Equilibrium branch, solid where stable / dashed where unstable (one polyline per
    contiguous-stability run so dashes render); folds bridged, big jumps broken."""
    stab = np.asarray(stab, bool)
    step = np.hypot(np.diff(I), np.diff(s))
    thr = 6 * np.median(step[step > 0]) if np.any(step > 0) else np.inf
    flips = list(np.where(np.diff(stab.astype(int)) != 0)[0] + 1)
    bounds = [0] + flips + [stab.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        e = min(b + 1, stab.size)
        y = s[a:e].astype(float).copy()
        for L in np.where(step[a:e - 1] > thr)[0]:
            y[L + 1] = np.nan
        ax.plot(I[a:e], np.clip(y, LIM_1D["s_min"], None), color=color, lw=1.2,
                ls="-" if stab[a] else "--", zorder=2)


def plot_1d(ax, bif_by_layer, lim):
    for layer in LAYERS:
        bif = bif_by_layer.get(layer)
        if bif is None:
            continue
        col = C_LAYER[layer]
        _plot_branch(ax, bif["branch_I"], bif["branch_s"], bif["branch_stab"], col)
        ax.scatter(bif["lp_I"], np.clip(bif["lp_s"], lim["s_min"], None), marker=M_FOLD, s=14,
                   facecolors="none", edgecolors=col, linewidths=0.9, zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(lim["I_min"], lim["I_max"]); ax.set_ylim(lim["s_min"], lim["s_max"])
    ax.set_xlabel(r"input $I$", labelpad=1)
    ax.set_ylabel(r"firing rate $r$", labelpad=2)
    hd = next((float(b["hd_cut"]) for b in bif_by_layer.values() if b is not None), 0.1)
    ax.text(0.97, 0.96, rf"$h_\Delta={hd:g}$", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color="0.3")
    ax.legend(handles=[
        Line2D([0], [0], color="0.4", ls="-", lw=1.0, label="stable"),
        Line2D([0], [0], color="0.4", ls="--", lw=1.0, label="unstable"),
        Line2D([0], [0], marker=M_FOLD, color="0.3", lw=0, markerfacecolor="none", label="fold"),
    ], loc="lower right", fontsize=5.5, handlelength=1.4)


def _classify_loci(bif, h_ceiling=1.4, apex_tol=(3.0, 0.05)):
    """Split a layer's fold loci into outermost open branches (reach H_MAX, no cusp) and
    inner arches (turn around at an apex = a cusp).  Arches are seeded from both feet and so
    arrive duplicated; dedupe by apex proximity."""
    opens, arches = [], []
    for k in range(int(bif["n_loci"])):
        I = np.asarray(bif[f"loci_{k}_I"], float); h = np.asarray(bif[f"loci_{k}_h"], float)
        if I.size < 2:
            continue
        if h.max() >= h_ceiling:                       # exits the top of the window -> open
            opens.append((I, h))
        else:                                          # turns around -> arch, apex is the cusp
            a = int(np.argmax(h)); cusp = (float(I[a]), float(h[a]))
            if any(abs(cusp[0] - c[0]) < apex_tol[0] and abs(cusp[1] - c[1]) < apex_tol[1]
                   for *_, c in arches):
                continue
            arches.append((I, h, cusp))
    return opens, arches


def plot_2d(ax, bif_by_layer, lim):
    A_FILL = 0.14                       # per-level fill alpha; cumulative so inner pairs darken
    A_LINE_OUT, A_LINE_IN = 0.5, 0.9    # fold-line alpha: outermost faint -> inner pairs opaque
    LW_FOLD = 0.8                       # ~2/3 of the previous 1.2
    M_CUSP = "X"
    for layer in LAYERS:
        bif = bif_by_layer.get(layer)
        if bif is None:
            continue
        col = C_LAYER[layer]

        # nested shading from the stable-equilibrium count grid: the area bounded by the
        # outermost fold pair (#stable>=2) gets the lowest alpha; each inner pair (#stable>=3)
        # adds another translucent layer on top, so transparency drops moving inward.
        if "count_n" in bif:
            Ig = np.asarray(bif["count_I"], float); hg = np.asarray(bif["count_h"], float)
            ng = np.asarray(bif["count_n"], float)
            if hg[0] > 0.0:                             # extend down so the wedge reaches h=0
                hg = np.concatenate([[0.0], hg]); ng = np.vstack([ng[0], ng])
            ngs = gaussian_filter(ng, sigma=0.6)
            maxc = int(round(float(np.nanmax(ng))))
            for L in range(2, maxc + 1):
                ax.contourf(Ig, hg, ngs, levels=[L - 0.5, maxc + 0.5], colors=[col],
                            alpha=A_FILL, zorder=1, antialiased=True)

        # fold curves + cusp markers
        opens, arches = _classify_loci(bif)
        for I, h in opens:                             # outermost pair
            ax.plot(I, h, color=col, lw=LW_FOLD, alpha=A_LINE_OUT, zorder=3)
        for I, h, cusp in arches:                      # inner pairs (collide in a cusp)
            ax.plot(I, h, color=col, lw=LW_FOLD, alpha=A_LINE_IN, zorder=3)
            ax.scatter([cusp[0]], [cusp[1]], marker=M_CUSP, s=20, color=col,
                       edgecolors="white", linewidths=0.4, zorder=6)

    # data-fit line
    ax.axhline(1.0, color="0.55", ls=":", lw=0.8, zorder=2)
    ax.text(lim["I_max"], 1.0, " data ", color="0.45", ha="right", va="bottom", fontsize=5.0)

    ax.set_xlim(lim["I_min"], lim["I_max"]); ax.set_ylim(lim["h_min"], lim["h_max"])
    ax.set_xlabel(r"input $I$", labelpad=1)
    ax.set_ylabel(r"width heterogeneity $h_\Delta$", labelpad=2)
    handles = [Line2D([0], [0], color=C_LAYER[ly], lw=1.4, label=ly) for ly in LAYERS]
    handles.append(Line2D([0], [0], marker=M_CUSP, color="0.3", lw=0, markeredgecolor="white",
                          markeredgewidth=0.4, label="cusp"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=5.5,
              handlelength=1.4)


def plot_rate(ax, sim_by_layer):
    bounds = None
    for layer in LAYERS:
        sim = sim_by_layer.get(layer)
        if sim is None:
            continue
        col = C_LAYER[layer]
        if bounds is None and "proto" in sim:
            proto = np.asarray(sim["proto"], float); bounds = np.concatenate([[0], np.cumsum(proto[:, 1])])
            levels = proto[:, 0]
        ax.plot(sim["t_mf"], np.clip(sim["r_mf"], 1e-3, None), color=col, lw=1.1, ls="-",
                alpha=0.7, zorder=2)
        ax.plot(sim["t_micro"], np.clip(sim["r_micro"], 1e-3, None), color=col, lw=0.7, ls="--",
                zorder=3)
    ax.set_yscale("log")
    if bounds is not None:
        # grey background per input level: 4 equidistant greys, white (lowest I) -> darkest
        # (highest I); darkest kept light enough that the L2/3 blue stays clearly visible.
        uniq = sorted({round(float(lv), 6) for lv in levels})
        g_dark = 0.70
        grey = {lv: (1.0 - (1.0 - g_dark) * i / max(len(uniq) - 1, 1)) for i, lv in enumerate(uniq)}
        for lev, x0, x1 in zip(levels, bounds[:-1], bounds[1:]):
            g = grey[round(float(lev), 6)]
            if g < 1.0:
                ax.axvspan(x0, x1, facecolor=str(g), edgecolor="none", zorder=0)
        for x in bounds[1:-1]:
            ax.axvline(x, color="0.7", ls=":", lw=0.6, zorder=1)
        for lev, x0, x1 in zip(levels, bounds[:-1], bounds[1:]):
            ax.text(0.5 * (x0 + x1), 1.02, rf"$I{{=}}{lev:.0f}$", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=5.0, color="0.4", clip_on=False)
        ax.set_xlim(bounds[0], bounds[-1])
    ax.set_xlabel(r"time $t$ (ms)", labelpad=1)
    ax.set_ylabel(r"firing rate $r(t)$", labelpad=2)
    ax.legend(handles=[
        Line2D([0], [0], color="0.3", ls="-", lw=1.1, label="mean field"),
        Line2D([0], [0], color="0.3", ls="--", lw=0.7, label="QIF network"),
    ] + [Line2D([0], [0], color=C_LAYER[ly], lw=1.2, label=ly) for ly in LAYERS],
        loc="center right", fontsize=5.0, handlelength=1.4, ncol=2, columnspacing=1.0)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    set_prl_style()
    fig = plt.figure(figsize=(3.4, 4.6), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.10)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 0.95])

    fit_by_layer, bif_by_layer, sim_by_layer = {}, {}, {}
    for layer in LAYERS:
        tag = _tag(CELL_CLASS, layer)
        fit_by_layer[layer] = _load(os.path.join(_FIT_DIR, f"allen_lorentzian_{tag}.npz"))
        bif_by_layer[layer] = _load(os.path.join(_HERE, f"pyramidal_fig_bif_{tag}.npz"))
        sim_by_layer[layer] = _load(os.path.join(_SIM_DIR, f"pyramidal_fig_rate_{tag}.npz"))

    panels = []
    for c, layer in enumerate(LAYERS):
        ax = fig.add_subplot(gs[0, c]); plot_pdf(ax, fit_by_layer[layer], C_LAYER[layer], layer)
        if c == 0:
            ax.set_ylabel(r"$\rho(\eta)$", labelpad=2)
        panels.append(ax)

    ax1d = fig.add_subplot(gs[1, 0]); plot_1d(ax1d, bif_by_layer, LIM_1D)
    ax2d = fig.add_subplot(gs[1, 1]); plot_2d(ax2d, bif_by_layer, LIM_2D)
    panels += [ax1d, ax2d]

    axr = fig.add_subplot(gs[2, :]); plot_rate(axr, sim_by_layer)
    panels.append(axr)

    for ax, letter in zip(panels, "abcdefg"):
        _panel_label(ax, letter)
    out = os.path.join(_HERE, "pyramidal_fig")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"[saved] {os.path.basename(out)}.{{png,svg}}")


if __name__ == "__main__":
    main()
