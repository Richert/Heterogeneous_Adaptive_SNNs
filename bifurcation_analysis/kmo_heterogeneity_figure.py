r"""
Kuramoto heterogeneity control — manuscript Fig. 3 (Gaussian-mixture example)
=============================================================================

Single-column PRL figure, 3 rows x 2 cols, demonstrating heterogeneity control of a globally
coupled Kuramoto network with an arbitrary (Gaussian-mixture) frequency distribution, via the
single lumped knob h (h=1 data, h->0 homogeneous; h scales BOTH the centre spread and the widths):

  (a) Gaussian-mixture frequency distribution + its Lorentzian-mixture fit (components + sum).
  (b) the re-parameterization: rescaling the mixture with the lumped knob h.
  (c) 1-D bifurcation R(h) of the LMMF Ott-Antonsen mean field at K=K_fix: synchronized branch +
      fold (saddle-node synchronization threshold) + Hopf, with the incoherent state R=0.
  (d) 2-D fold/Hopf loci in the (h, K) plane (the synchronization boundary).
  (e) 2-D fold loci in the (h_Delta, h_ombar) heterogeneity plane (multistability at small h_Delta /
      large h_ombar).
  (f) average phase coherence R(t) under a transient h-pulse: microscopic Kuramoto vs ensemble MF.

Reads the pickled bif .npy, so run in the SAME env they were written in (``pycobi``):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_heterogeneity_figure.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = "/home/rgast/data/mpmf_simulations"
FIT_NPZ = os.path.join(_DATA, "kmo_het_fit.npz")
RATE_NPZ = os.path.join(_DATA, "kmo_het_rate.npz")
ENV_NPZ = os.path.join(_DATA, "kmo_het_envelope.npz")
REGIONS_HK_NPZ = os.path.join(_DATA, "kmo_het_regions_hK.npz")
REGIONS_DC_NPZ = os.path.join(_DATA, "kmo_het_regions_DC.npz")
LUMPED_NPY = os.path.join(_HERE, "kmo_het_bif_lumped.npy")
TWOKNOB_NPY = os.path.join(_HERE, "kmo_het_bif_twoknob.npy")
# mono-stable region shading: -1 async (grey), 0 multistable/torus (white), +1 sync (blue), low alpha
_REG_CMAP = ListedColormap([(0.5, 0.5, 0.5, 0.16), (1, 1, 1, 0.0), (0.12, 0.47, 0.71, 0.15)])
_REG_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], _REG_CMAP.N)
C_OSC = "#2ca02c"           # oscillatory (standing-wave / torus) envelope

C_FIT = "#c1121f"           # Lorentzian-mixture fit
C_BR = "#1f77b4"            # bifurcation branch / loci
C_MICRO = "0.35"
CMAP = plt.cm.managua       # lumped-h colour scale (diverging, centred at h=1)
HMAX = 1.5
NORM = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=HMAX)
H_SWEEP = [1.4, 1.0, 0.7, 0.5]


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "legend.fontsize": 5.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    })


def _panel_label(ax, letter):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-16, 5),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


def mixture(x, w, Om, De):
    return (w[None, :] * (De[None, :] / np.pi) / ((x[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2))


def reparam(Om, De, ombar, h):
    """Single lumped knob: centres collapse toward ombar AND widths shrink, both by h."""
    return ombar + h * (Om - ombar), h * De


# ════════════════════════════════════════════════════════════════════════════
#  panels
# ════════════════════════════════════════════════════════════════════════════
def plot_fit(ax, fit):
    s, w, Om, De = fit["samples"], fit["weights"], fit["omega"], fit["delta"]
    gx = np.linspace(s.min() - 0.4, s.max() + 0.4, 600)
    comp = mixture(gx, w, Om, De)
    ax.hist(s, bins=50, range=(gx[0], gx[-1]), density=True, color="0.85", edgecolor="none", zorder=0)
    for k in range(len(w)):
        ax.plot(gx, comp[:, k], color=C_FIT, lw=0.5, alpha=0.45, zorder=2)
    ax.plot(gx, comp.sum(axis=1), color=C_FIT, lw=1.4, zorder=3)
    ax.set_xlim(gx[0], gx[-1]); ax.set_yticks([])
    ax.set_xlabel(r"frequency $\omega$", labelpad=1); ax.set_ylabel(r"$\rho(\omega)$", labelpad=2)
    ax.legend(handles=[Line2D([0], [0], color=C_FIT, lw=1.4, label="Lorentzian fit")],
              loc="upper right", fontsize=5.5)


def plot_sweep(ax, fit):
    w, Om, De, ombar = fit["weights"], fit["omega"], fit["delta"], float(fit["omega_bar"])
    gx = np.linspace(-5.0, 5.0, 700)
    peak1 = 0.0
    for h in sorted(H_SWEEP, reverse=True):
        Omh, Deh = reparam(Om, De, ombar, h)
        dens = mixture(gx, w, Omh, Deh).sum(axis=1)
        ax.plot(gx, dens, color=CMAP(NORM(h)), lw=1.1)
        if abs(h - 1.0) < 1e-6:
            peak1 = float(dens.max())
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(0, 2.25 * peak1); ax.set_yticks([])       # cap the collapse spike
    ax.set_xlabel(r"frequency $\omega$", labelpad=1); ax.set_ylabel(r"$\rho_h(\omega)$", labelpad=2)
    sm = ScalarMappable(norm=NORM, cmap=CMAP); sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, ticks=[0, 0.5, 1.0, 1.5])
    cb.ax.tick_params(labelsize=5.5); cb.set_label(r"heterogeneity $h$", fontsize=6, labelpad=1)


def _plot_branch(ax, x, y, stab, color):
    stab = np.asarray(stab, bool)
    flips = list(np.where(np.diff(stab.astype(int)) != 0)[0] + 1)
    bounds = [0] + flips + [stab.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        e = min(b + 1, stab.size)
        ax.plot(x[a:e], y[a:e], color=color, lw=1.3, ls="-" if stab[a] else "--", zorder=3)


def plot_1d(ax, lumped, env):
    """Coherence R vs lumped h, from a bidirectional stepping sweep: synchronized fixed point
    (low h) -> standing-wave (torus) envelope -> asynchronous R~0 (high h), with the torus / async
    bistable (hysteresis) window shaded."""
    h = np.asarray(env["h"], float)
    lo_u, hi_u = np.asarray(env["R_min_up"], float), np.asarray(env["R_max_up"], float)
    hi_d = np.asarray(env["R_max_dn"], float)
    W = np.where
    osc = (hi_u - lo_u) > 0.015                         # up sweep oscillatory (torus)
    coh = (~osc) & (hi_u > 0.3)                         # coherent fixed point (low h)
    asy = hi_d < 0.02                                   # asynchronous fixed point (down sweep ~0)
    ax.plot(h, W(coh, hi_u, np.nan), color=C_BR, lw=1.7, zorder=5)              # synchronized FP
    ax.plot(h, W(asy, 0.0, np.nan), color=C_BR, lw=1.7, zorder=5)              # asynchronous FP
    ax.fill_between(h, W(osc, lo_u, np.nan), W(osc, hi_u, np.nan), color=C_OSC, alpha=0.25, lw=0, zorder=2)
    ax.plot(h, W(osc, hi_u, np.nan), color=C_OSC, lw=1.2, zorder=4)
    ax.plot(h, W(osc, lo_u, np.nan), color=C_OSC, lw=1.2, zorder=4)
    # overlay the coherent equilibrium continuation (stable solid / unstable dashed) -> shows the
    # fold.  Plot in CONTINUATION ORDER (the branch folds back in h, so sorting by h would
    # interleave the folded segments and produce a spurious zig-zag).
    eq = lumped["eq"]
    _plot_branch(ax, eq["x"], eq["Rmax"], eq["stab"], C_BR)
    for lab, mk in (("LP", "o"), ("HB", "s")):           # fold / Hopf markers at the actual points
        sel = eq["bif"] == lab
        if sel.any():
            ax.scatter(eq["x"][sel], eq["Rmax"][sel], marker=mk, s=15, facecolors="none",
                       edgecolors="0.2", linewidths=0.9, zorder=6)
    ax.axvline(1.0, color="0.6", ls=":", lw=0.8, zorder=1)
    ax.text(0.995, 0.45, "data fit", transform=ax.get_xaxis_transform(), rotation=90,
            ha="right", va="center", color="0.5", fontsize=5.0)
    ax.text(0.03, 0.70, "synchronized", transform=ax.transAxes, fontsize=5.5, color="0.35")
    ax.text(0.80, 0.12, "asynchronous", transform=ax.transAxes, fontsize=5.5, color="0.35", ha="right")
    ax.set_xlim(0.4, 1.04); ax.set_ylim(-0.03, 0.90)
    ax.set_xlabel(r"heterogeneity $h$", labelpad=1)
    ax.set_ylabel(r"phase coherence $R$", labelpad=2)
    ax.annotate(rf"$K={lumped['K_fix']:g}$", xy=(0.04, 0.05), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=6, color="0.3")
    ax.legend(handles=[
        Line2D([0], [0], color=C_BR, lw=1.4, label="fixed point"),
        Line2D([0], [0], color=C_OSC, lw=1.4, label="torus"),
        Line2D([0], [0], marker="o", color="0.2", lw=0, markerfacecolor="none", label="fold"),
        Line2D([0], [0], marker="s", color="0.2", lw=0, markerfacecolor="none", label="Hopf"),
    ], loc="upper center", fontsize=5.0, handlelength=1.2, borderaxespad=0.2, ncol=2, columnspacing=1.0)


LS = ["-", "--", "-.", ":"]                         # linestyle per distinct locus of a given type
TYPE_COL = {"fold": "#1f77b4", "hopf": "#e8702a"}   # colour per bifurcation type


def _dedup_curves(curves, xlim, ylim, tol=0.08):
    """Drop near-duplicate loci (the bidirectional continuation returns each curve twice).
    Curves are normalised to the panel box and compared by symmetric Hausdorff distance."""
    from scipy.spatial.distance import directed_hausdorff
    kept, keptP = [], []
    for x, y in curves:
        x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            continue
        P = np.column_stack([(x[m] - xlim[0]) / (xlim[1] - xlim[0]),
                             (y[m] - ylim[0]) / (ylim[1] - ylim[0])])
        if any(max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0]) < tol for Q in keptP):
            continue
        kept.append((x[m], y[m])); keptP.append(P)
    return kept


def plot_loci_styled(ax, loci, xlim, ylim):
    """Plot fold/Hopf loci: colour = bifurcation TYPE, linestyle = distinct locus of that type."""
    for typ in ("fold", "hopf"):
        ded = _dedup_curves([(loc["x"], loc["y"]) for k, loc in loci.items()
                             if k.split("_")[0] == typ], xlim, ylim)
        for i, (x, y) in enumerate(ded):
            ax.plot(x, y, color=TYPE_COL[typ], ls=LS[i % len(LS)], lw=1.3, zorder=3,
                    label=typ.capitalize() if i == 0 else None)   # one legend entry per type


def _shade_regions(ax, reg):
    """Shade mono-stable synchronized (blue) / asynchronous (grey) regions; multistable/torus white."""
    ax.pcolormesh(np.asarray(reg["x"]), np.asarray(reg["y"]), np.asarray(reg["cls"]),
                  cmap=_REG_CMAP, norm=_REG_NORM, shading="nearest", zorder=0, rasterized=True)


def plot_hK(ax, d, reg=None):
    """Fold / Hopf loci in the (h, K) plane (lumped heterogeneity vs global coupling)."""
    xlim, ylim = (0.0, 1.3), (0.2, 3.0)
    if reg is not None:
        _shade_regions(ax, reg)
    plot_loci_styled(ax, d["loci"], xlim, ylim)
    ax.axhline(d["K_fix"], color="0.6", ls=":", lw=0.8, zorder=1)
    ax.text(1.29, d["K_fix"], " K=1", ha="right", va="bottom", fontsize=5.0, color="0.5")
    ax.scatter([1.0], [float(d["K_fix"])], marker="*", s=28, color="0.3", zorder=6)   # data fit (h=1, K=1)
    ax.annotate("data", xy=(1.0, float(d["K_fix"])), xytext=(-3, 2), textcoords="offset points",
                ha="right", va="bottom", fontsize=5.0, color="0.4")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"heterogeneity $h$", labelpad=1)
    ax.set_ylabel(r"coupling $K$", labelpad=2)
    ax.text(0.04, 0.62, "coherent", transform=ax.transAxes, fontsize=5.5, color="0.4")
    ax.text(0.58, 0.10, "incoherent", transform=ax.transAxes, fontsize=5.5, color="0.4")
    ax.legend(loc="upper left", fontsize=5.0, handlelength=1.6, borderaxespad=0.3)


def plot_DC(ax, d, reg=None):
    """Fold / Hopf loci in the (h_Delta, h_ombar) heterogeneity plane (synchronization boundary)."""
    xlim, ylim = (0.0, 1.3), (0.0, 1.3)
    if reg is not None:
        _shade_regions(ax, reg)
    plot_loci_styled(ax, d["loci"], xlim, ylim)
    ax.scatter([1.0], [1.0], marker="*", s=28, color="0.3", zorder=6)            # data fit (h=1)
    ax.annotate("data", xy=(1.0, 1.0), xytext=(-3, -2), textcoords="offset points",
                ha="right", va="top", fontsize=5.0, color="0.4")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"width scaling $h_\Delta$", labelpad=1)
    ax.set_ylabel(r"centre spread $h_{\bar\omega}$", labelpad=2)
    ax.text(0.04, 0.10, "synchronized", transform=ax.transAxes, fontsize=5.5, color="0.4")
    ax.text(0.40, 0.88, "asynchronous", transform=ax.transAxes, fontsize=5.5, color="0.4")
    ax.legend(loc="lower right", fontsize=5.0, handlelength=1.6, borderaxespad=0.3)


def plot_rate(ax, r):
    t1, t2 = float(r["t1"]), float(r["t2"]); h1, h2 = float(r["h1"]), float(r["h2"])
    T0, T = float(r["t_mf"][0]), float(r["t_mf"][-1])
    ax.axvspan(t1, t2, color="0.95", zorder=0)                  # h1 (standing-wave) window
    ax.axvspan(t2, T, color="0.90", zorder=0)                   # h2 (synchronized) window
    ax.axvline(t1, color="0.6", ls=":", lw=0.7, zorder=1); ax.axvline(t2, color="0.6", ls=":", lw=0.7, zorder=1)
    ax.plot(r["t_micro"], r["R_micro"], color=C_MICRO, lw=0.8, zorder=2, label="microscopic")
    ax.plot(r["t_mf"], r["R_mf"], color=C_FIT, lw=1.1, ls="--", zorder=3, label="mean field")
    ax.set_xlim(T0, T); ax.set_ylim(0.0, 1.03)
    ax.set_xlabel(r"time $t$", labelpad=1); ax.set_ylabel(r"phase coherence $R(t)$", labelpad=2)
    h0 = float(r["h0"])
    for xc, hv in [(0.5 * (T0 + t1), h0), (0.5 * (t1 + t2), h1), (0.5 * (t2 + T), h2)]:
        ax.text(xc, 1.02, rf"$h{{=}}{hv:g}$", ha="center", va="bottom",
                transform=ax.get_xaxis_transform(), color="0.4", fontsize=6, clip_on=False)
    ax.legend(loc="center right", fontsize=5.5, handlelength=1.6)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    set_prl_style()
    fit = np.load(FIT_NPZ, allow_pickle=False)
    rate = np.load(RATE_NPZ, allow_pickle=False)
    lumped = np.load(LUMPED_NPY, allow_pickle=True).item()
    twoknob = np.load(TWOKNOB_NPY, allow_pickle=True).item()

    env = np.load(ENV_NPZ, allow_pickle=False)
    fig = plt.figure(figsize=(3.4, 6.3), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.005, h_pad=0.02, wspace=0.015, hspace=0.10)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.82, 0.82, 1.0])
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, :])                                   # 1-D R(h) spans both columns
    d = fig.add_subplot(gs[2, :])                                   # transient R(t) spans both columns
    e = fig.add_subplot(gs[3, 0]); f = fig.add_subplot(gs[3, 1])

    plot_fit(a, fit)
    plot_sweep(b, fit)
    reg_hk = np.load(REGIONS_HK_NPZ, allow_pickle=False) if os.path.exists(REGIONS_HK_NPZ) else None
    reg_dc = np.load(REGIONS_DC_NPZ, allow_pickle=False) if os.path.exists(REGIONS_DC_NPZ) else None
    plot_1d(c, lumped, env)
    plot_rate(d, rate)
    plot_hK(e, lumped, reg_hk)
    plot_DC(f, twoknob, reg_dc)
    for ax, letter in zip([a, b, c, d, e, f], "abcdef"):
        _panel_label(ax, letter)

    out = os.path.join(_HERE, "kmo_heterogeneity_figure")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"[saved] {os.path.basename(out)}.{{png,svg}}")


if __name__ == "__main__":
    main()
