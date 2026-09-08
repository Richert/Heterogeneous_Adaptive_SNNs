r"""
PV+ interneuron — figure 2 (heterogeneity-controlled oscillations)
==================================================================

Single-column PRL figure, 3 rows x 2 columns, two layers distinguished by COLOUR
(L2/3 vs L5/6), mirroring the Pyramidal summary figure:

  row 1 : fitted excitability distributions, L2/3 | L5/6
          (empirical histogram + Lorentzian-mixture PDF + components).
                                                  [data_fitting/allen_lorentzian_<tag>.npz]
  row 2 : (left)  1-D bifurcation r(h_Delta) at collapsed centres (h_eta=0), both layers:
                  equilibrium (solid stable / dashed unstable) + Hopf + limit-cycle envelope;
          (right) 2-D Hopf (+ period-doubling) loci in the (h_eta, h_Delta) heterogeneity
                  plane -> the oscillatory region is the small wedge near the origin; the
                  data fit sits at (1,1), far outside (quiescent).
                                                  [pv_fig2_bif_<tag>.npy]
  row 3 : firing-rate dynamics under a heterogeneity STEP (h down into the oscillatory wedge),
          L2/3 | L5/6, mean field (solid) vs QIF network (dashed).
                                                  [qif_simulations/pv_fig2_rate_<tag>.npz]

Reads the bifurcation .npy (pickled), so run in the SAME env it was written in (``pycobi``):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python pv_figure2.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIT_DIR = os.path.join(_HERE, "..", "data_fitting")
_SIM_DIR = os.path.join(_HERE, "..", "qif_simulations")

CELL_CLASS = "PV+ Interneuron"
LAYERS = ["L2/3", "L5/6"]
C_LAYER = {"L2/3": "#1f77b4", "L5/6": "#e8702a"}
C_HIST = "0.85"
M_HOPF, M_PD = "o", "p"          # filled circle = Hopf, filled pentagon = period-doubling
I_OP = 430.0                     # common operating-point input (marked in the (I,h) panel)
LIM_1D = dict(h_min=0.0, h_max=0.30)                              # 1-D bif in the single knob h
LIM_2D = dict(I_min=250.0, I_max=600.0, h_min=0.0, h_max=0.30)    # (I, h) Hopf-locus plane
RATE_WIN = 35.0                                                   # zoom window after the step (time units)


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


def _panel_label(ax, letter):
    """Bold PRL-style panel label OUTSIDE the axis box, above its top-left corner."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-15, 5),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


def _load_npz(path):
    return np.load(path, allow_pickle=False) if os.path.exists(path) else None


def _load_npy(path):
    return np.load(path, allow_pickle=True).item() if os.path.exists(path) else None


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


def _plot_branch(ax, x, y, stab, color, lw=1.2):
    """Equilibrium branch: solid where stable, dashed where unstable (one polyline per run)."""
    stab = np.asarray(stab, bool)
    flips = list(np.where(np.diff(stab.astype(int)) != 0)[0] + 1)
    bounds = [0] + flips + [stab.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        e = min(b + 1, stab.size)
        ax.plot(x[a:e], y[a:e], color=color, lw=lw, ls="-" if stab[a] else "--", zorder=3)


def plot_1d(ax, bif_by_layer, lim):
    vlo, vhi = np.inf, -np.inf
    for layer in LAYERS:
        d = bif_by_layer.get(layer)
        if d is None:
            continue
        col = C_LAYER[layer]
        eq = d["eq"]
        _plot_branch(ax, eq["x"], eq["ymax"], eq["stab"], col)
        m = eq["x"] <= lim["h_max"]
        vlo = min(vlo, np.nanmin(eq["ymax"][m])); vhi = max(vhi, np.nanmax(eq["ymax"][m]))
        hb = eq["bif"] == "HB"                                       # Hopf marker
        ax.scatter(eq["x"][hb], eq["ymax"][hb], marker=M_HOPF, s=18, facecolors=col,
                   edgecolors=col, linewidths=0.6, zorder=6)
        for lc in d["lc"]:                                           # limit-cycle envelope of s
            mlc = lc["x"] <= lim["h_max"]
            ax.fill_between(lc["x"], lc["ymin"], lc["ymax"], color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(lc["x"], lc["ymin"], color=col, lw=0.8, zorder=4)
            ax.plot(lc["x"], lc["ymax"], color=col, lw=0.8, zorder=4)
            vlo = min(vlo, np.nanmin(lc["ymin"][mlc])); vhi = max(vhi, np.nanmax(lc["ymax"][mlc]))
            pd = lc["bif"] == "PD"
            ax.scatter(lc["x"][pd], lc["ymax"][pd], marker=M_PD, s=20, facecolors=col,
                       edgecolors=col, linewidths=0.6, zorder=6)
    pad = 0.10 * (vhi - vlo)
    ax.set_xlim(lim["h_min"], lim["h_max"])
    ax.set_ylim(vlo - pad, vhi + 1.7 * pad)
    ax.annotate(r"$h\!=\!1$ (data) $\rightarrow$", xy=(0.98, 0.50), xycoords="axes fraction",
                ha="right", va="center", color="0.45", fontsize=5.5)
    ax.set_xlabel(r"heterogeneity $h$", labelpad=1)
    ax.set_ylabel(r"mean-field drive $s$", labelpad=2)
    ax.legend(handles=[
        Line2D([0], [0], color="0.4", ls="-", lw=1.0, label="stable"),
        Line2D([0], [0], color="0.4", ls="--", lw=1.0, label="unstable"),
        Line2D([0], [0], marker=M_HOPF, color="0.3", lw=0, markerfacecolor="0.3", markersize=4.3, label="Hopf"),
    ], loc="lower right", fontsize=5.0, handlelength=1.4, borderaxespad=0.6, ncol=1)


def plot_ih(ax, bif_by_layer, lim):
    """2-D Hopf locus in the (input I, heterogeneity h) plane; oscillatory region shaded below."""
    for layer in LAYERS:
        d = bif_by_layer.get(layer)
        if d is None:
            continue
        col = C_LAYER[layer]
        loc = d.get("loci", {}).get("Iext_h")
        if loc is None:
            continue
        x = np.asarray(loc["x"], float); y = np.asarray(loc["y"], float)
        m = np.isfinite(x) & np.isfinite(y) & (y >= -2e-3)          # physical branch (drop runaways)
        xs, ys = x[m], np.clip(y[m], 0.0, lim["h_max"])
        o = np.argsort(xs)
        ax.fill_between(xs[o], 0.0, ys[o], color=col, alpha=0.13, lw=0, zorder=1)
        yl = y.copy(); yl[(y < -2e-3) | (y > 1.04 * lim["h_max"])] = np.nan
        ax.plot(x, yl, color=col, lw=1.4, zorder=3)
    ax.axvline(I_OP, color="0.6", ls=":", lw=0.8, zorder=1)         # operating-point input
    ax.text(0.50, 0.18, "oscillatory", transform=ax.transAxes, ha="center", va="center",
            color="0.35", fontsize=6)
    ax.set_xlim(lim["I_min"], lim["I_max"]); ax.set_ylim(lim["h_min"], lim["h_max"])
    ax.set_xlabel(r"input $I$", labelpad=1)
    ax.set_ylabel(r"heterogeneity $h$", labelpad=2)
    ax.legend(handles=[Line2D([0], [0], color=C_LAYER[ly], lw=1.4, label=ly) for ly in LAYERS]
              + [Line2D([0], [0], color="0.6", ls=":", lw=0.8, label="operating point")],
              loc="upper right", fontsize=5.0, handlelength=1.4, borderaxespad=0.4)


def plot_rate(ax, sim_by_layer):
    """All layers in one axis: mean-field drive s(t) solid, microscopic-network s(t) dotted."""
    t_step = hd1 = None; lo, hi = np.inf, -np.inf
    for layer in LAYERS:
        sim = sim_by_layer.get(layer)
        if sim is None:
            continue
        col = C_LAYER[layer]; t_step = float(sim["t_step"]); hd1 = float(sim["hd1"])
        t0, t1 = t_step - 0.4 * RATE_WIN, t_step + RATE_WIN        # zoom: resolve the ~305 Hz cycles
        ax.plot(sim["t_mf"], sim["s_mf"], color=col, lw=0.9, ls="-", alpha=0.7, zorder=2)
        ax.plot(sim["t_micro"], sim["s_micro"], color=col, lw=0.7, ls=":", zorder=3)
        for t, s in [(sim["t_mf"], sim["s_mf"]), (sim["t_micro"], sim["s_micro"])]:
            w = (t >= t0) & (t <= t1)
            lo = min(lo, float(np.nanmin(s[w]))); hi = max(hi, float(np.nanmax(s[w])))
    if t_step is None:
        return
    ax.axvspan(t_step, t1, color="0.94", zorder=0)                # heterogeneity-reduced window
    ax.axvline(t_step, color="0.5", ls=":", lw=0.8, zorder=1)
    pad = 0.08 * (hi - lo)
    ax.set_xlim(t0, t1); ax.set_ylim(lo - pad, hi + 1.7 * pad)
    ax.text(0.5 * (t0 + t_step), 0.28, r"data ($h{=}1$)", transform=ax.get_xaxis_transform(),
            ha="center", va="center", color="0.4", fontsize=6)
    ax.text(0.5 * (t_step + t1), 0.045, rf"$h{{=}}{hd1:g}$",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", color="0.4", fontsize=6)
    ax.set_xlabel(r"time $t$", labelpad=1)
    ax.set_ylabel(r"mean-field drive $s$", labelpad=2)
    ax.legend(handles=[Line2D([0], [0], color=C_LAYER[ly], lw=1.2, label=ly) for ly in LAYERS]
              + [Line2D([0], [0], color="0.3", ls="-", lw=0.9, label="mean field"),
                 Line2D([0], [0], color="0.3", ls=":", lw=0.9, label="QIF network")],
              loc="upper right", fontsize=5.5, handlelength=1.6, ncol=2, columnspacing=1.0,
              borderaxespad=0.3)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    set_prl_style()
    fig = plt.figure(figsize=(3.4, 4.6), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.12)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 0.95])

    fit_by_layer, bif_by_layer, sim_by_layer = {}, {}, {}
    for layer in LAYERS:
        tag = _tag(CELL_CLASS, layer)
        fit_by_layer[layer] = _load_npz(os.path.join(_FIT_DIR, f"allen_lorentzian_{tag}.npz"))
        bif_by_layer[layer] = _load_npy(os.path.join(_HERE, f"pv_fig2_bif_{tag}.npy"))
        sim_by_layer[layer] = _load_npz(os.path.join(_SIM_DIR, f"pv_fig2_rate_{tag}.npz"))

    panels = []
    # row 1: distribution fits
    for c, layer in enumerate(LAYERS):
        ax = fig.add_subplot(gs[0, c])
        plot_pdf(ax, fit_by_layer[layer], C_LAYER[layer], layer)
        if c == 0:
            ax.set_ylabel(r"$\rho(\eta)$", labelpad=2)
        panels.append(ax)

    # row 2: 1-D bifurcation s(h) (overlaid) | 2-D Hopf locus in the (I, h) plane (overlaid)
    ax1d = fig.add_subplot(gs[1, 0]); plot_1d(ax1d, bif_by_layer, LIM_1D)
    ax2d = fig.add_subplot(gs[1, 1]); plot_ih(ax2d, bif_by_layer, LIM_2D)
    panels += [ax1d, ax2d]

    # row 3: rate dynamics under the heterogeneity step, all series in one spanning axis
    axr = fig.add_subplot(gs[2, :]); plot_rate(axr, sim_by_layer)
    panels.append(axr)

    for ax, letter in zip(panels, "abcde"):
        _panel_label(ax, letter)
    out = os.path.join(_HERE, "pv_figure2")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"[saved] {os.path.basename(out)}.{{png,svg}}")


if __name__ == "__main__":
    main()
