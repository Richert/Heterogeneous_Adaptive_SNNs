r"""
Coupling-weight structure feeds back onto synchronization — main-text figure
============================================================================

One two-column row of three panels:

  (a) G = cos, μ = 0.03: R vs Δ of the network Δ-ramps of Supplementary Note 2 (○ forward, □ backward)
      vs the mean-field branches (blue: stable, dashed saddle; grey: asynchronous) and the quasi-static
      synchronized branch (black); stars mark the two folds;
  (b) G = cos, μ ∈ {0.02, 0.04, 0.05}: R vs Δ along the synchronized branch — mean field (dashed),
      quasi-static (solid), network (●: initialized at the mean-field fixed point, Fig. 3/Figs. 1-2 data;
      ○: initialized on the quasi-static state beyond the mean-field fold, ..._validate.npz);
  (c) G = sin: collective frequency Ω/κ vs κ = μ/γ for Δ = 0.1, 0.3 — network (○), quasi-static
      (solid), first order (dashed) (weight_variance_sin_drift.npz).

and a second row on the Δ–β plane of the general rule G_β(φ) = sin(φ + β) at μ = 0.03, i.e. between the
antisymmetric (β = 0) and the symmetric (β = 90°, panel a) rule (weight_variance_beta_grid[_fine].npz,
weight_variance_beta_plane.npz):

  (d) phase coherence R of the network started from a coherent state;
  (e) bistability: R(coherent init) − R(asynchronous init);
  (f) collective frequency Ω (coherent init, symmetric-log colour scale; grey where R < 2/√N, where the
      mean phase is not defined).
  Overlays: mean-field transcritical point Δ = K/2 (dotted), mean-field fold (dashed), end of the
  quasi-static synchronized branch (solid); in (d), white circles mark the largest Δ at which the network
  started on the quasi-static state remains on it (|R − R_QS| < 0.05).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_feedback_figure.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
from prl_style import set_prl_style as _set_prl_style
from prl_style import panel_label, COL_DOUBLE
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap

from weight_variance_bifurcation_rules import ra_branches
from weight_variance_cos_quasistatic_figure import panel_fold, QS_NPZ, FIG3_NPZ, RAMP_NPZ

VAL_NPZ = dp.mpmf("weight_variance_cos_quasistatic_validate.npz")
BIF_NPZ = dp.mpmf("weight_variance_bif_micro_uniform_init.npz")
SIN_NPZ = dp.kmo_adaptive("weight_variance_sin_drift.npz")
OUT = dp.mpmf("weight_variance_feedback")
MUS_B = (0.02, 0.04, 0.05)
#: Fig. 3 colours for μ ∈ {0.02, 0.04} (viridis positions of weight_variance_rule_meanfield.py), red for 0.05
COL_B = {0.02: colormaps["viridis"](0.567), 0.04: colormaps["viridis"](0.85), 0.05: "#c1121f"}
SIN_SHADES = (0.55, 0.9)
#: RdBu_r restricted to its lighter middle range, so that the black overlay lines stay visible in (f)
OMEGA_CMAP = LinearSegmentedColormap.from_list("RdBu_r_light", colormaps["RdBu_r"](np.linspace(0.15, 0.85, 256)))
GRID_FINE_NPZ = dp.mpmf("weight_variance_beta_grid_fine.npz")
GRID_NPZ = dp.mpmf("weight_variance_beta_grid.npz")
PLANE_NPZ = dp.mpmf("weight_variance_beta_plane.npz")

set_prl_style = functools.partial(_set_prl_style, "prl", **{"legend.fontsize": 5.4})
_panel_label = functools.partial(panel_label, dx=-24, dy=3)


def panel_branches(ax, q, val, f3, bif, K, g):
    d = np.linspace(1e-3, 1.7, 1500)
    ri3, ribf = list(f3["rules"]).index("cos"), list(bif["rules"]).index("cos")
    n = f3["t"].size
    win = slice(int(round(0.8 * n)), None)
    for mu in MUS_B:
        c = COL_B[mu]
        up, _, _ = ra_branches("cos", d, mu, g, K)
        ax.plot(d, up["R"], color=c, lw=0.9, ls="--")
        k = int(np.argmin(np.abs(q["mus"] - mu)))
        ok = np.isfinite(q["R"][k])
        ax.plot(q["deltas"][ok], q["R"][k][ok], color=c, lw=1.2)
        if np.any(np.isclose(f3["mus"], mu)):                          # Fig. 3 runs (MF-fixed-point init)
            m = int(np.argmin(np.abs(f3["mus"] - mu)))
            sync = f3["mf_init"][ri3, m, :, 0] > 0
            R = np.abs(f3["Z1"][ri3, m, :, win]).mean(axis=1)
            ax.plot(f3["deltas"][sync], R[sync], "o", ms=2.8, mfc=c, mec="k", mew=0.3, zorder=5)
        if np.any(np.isclose(bif["mus"], mu)):                         # Figs. 1-2 runs (sync branch)
            m = int(np.argmin(np.abs(bif["mus"] - mu)))
            R = bif["R"][ribf, m, :, 0]
            ax.plot(bif["deltas"], R, "o", ms=2.8, mfc=c, mec="k", mew=0.3, zorder=5)
        sel = np.isclose(val["mu"], mu)                                # quasi-static-state init
        ax.plot(val["delta"][sel], val["R"][sel], "o", ms=3.0, mfc="w", mec=c, mew=0.9, zorder=6)
    ax.set_xlim(0, 1.7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$R$")
    ax.legend(handles=[Line2D([], [], color=COL_B[mu], lw=1.2, label=rf"$\mu = {mu:g}$") for mu in MUS_B]
              + [Line2D([], [], color="0.3", ls="--", lw=0.9, label="mean field"),
                 Line2D([], [], color="0.3", lw=1.2, label="quasi-static"),
                 Line2D([], [], color="0.3", ls="none", marker="o", ms=2.8, mec="k", mew=0.3,
                        label="network, MF init"),
                 Line2D([], [], color="0.3", ls="none", marker="o", ms=3.0, mfc="w", mew=0.9,
                        label="network, QS init")],
              loc="lower left", frameon=False, ncol=2, handlelength=1.6, columnspacing=0.8)


def panel_sin(ax, s):
    deltas, kappas = s["deltas"], s["kappas"]
    cols = [colormaps["YlOrBr"](x) for x in SIN_SHADES]
    for a, (D, c) in enumerate(zip(deltas, cols)):
        ax.plot(kappas, s["qs_Omega"][a] / kappas, "-", color=c, lw=1.2)
        ax.axhline(s["fo_coef"][a], color=c, ls="--", lw=0.9)
        ax.plot(kappas, s["mi_Omega"][a] / kappas, "o", ms=3.0, mfc="none", mec=c, mew=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa = \mu/\gamma$")
    ax.set_ylabel(r"$\Omega/\kappa$")
    ax.set_title(r"$G = \sin$", fontsize=6.5)
    ax.legend(handles=[Line2D([], [], color=c, lw=1.2, label=rf"$\Delta = {D:g}$") for D, c in zip(deltas, cols)]
              + [Line2D([], [], color="0.3", ls="--", lw=0.9, label="first order"),
                 Line2D([], [], color="0.3", lw=1.2, label="quasi-static"),
                 Line2D([], [], color="0.3", ls="none", marker="o", ms=3, mfc="none", label="network")],
              loc="upper left", frameon=False, ncol=2, handlelength=1.6, columnspacing=0.8)


def _beta_overlay(ax, p, K, lw=0.9):
    bd = np.degrees(p["betas"])
    ax.axvline(K / 2, color="k", ls=":", lw=0.8)
    ax.plot(p["fold_mf"], bd, color="k", ls="--", lw=lw)
    ax.plot(p["qs_last"], bd, color="w", lw=lw + 1.0)
    ax.plot(p["qs_last"], bd, color="k", lw=lw)


def _beta_axes(ax, gr, cb_label, im, fig):
    D = gr["deltas"]
    ax.set_xlim(D.min() - 0.02, D.max() + 0.02)
    ax.set_ylim(-3.75, 93.75)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_xlabel(r"$\Delta$")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(cb_label, fontsize=6)
    cb.ax.tick_params(labelsize=5)


def panels_beta(axes, gr, p, K, fig):
    D, bd = gr["deltas"], np.degrees(gr["betas"])
    inits = list(gr["inits"])
    ic, ia = inits.index("coherent"), inits.index("async")
    ax = axes[0]                                                       # (d) R, coherent init
    im = ax.pcolormesh(D, bd, gr["R"][..., ic], cmap="viridis", vmin=0, vmax=1, shading="nearest")
    _beta_overlay(ax, p, K)
    if "qs" in inits:                                                  # network stays on the QS branch up to …
        iq = inits.index("qs")
        for a, b in enumerate(bd):
            Rq = np.interp(D, p["deltas"], p["qs_R"][a], left=np.nan, right=np.nan)
            ok = np.isfinite(gr["R"][a, :, iq]) & np.isfinite(Rq) & (np.abs(gr["R"][a, :, iq] - Rq) < 0.05)
            if ok.any():
                ax.plot(D[ok].max(), b, "o", ms=2.6, mfc="w", mec="k", mew=0.5, zorder=6)
    ax.set_ylabel(r"phase lag $\beta$ (deg)")
    _beta_axes(ax, gr, r"$R$", im, fig)
    ax.legend(handles=[Line2D([], [], color="k", ls=":", lw=0.8, label="MF transcritical"),
                       Line2D([], [], color="k", ls="--", lw=0.9, label="MF fold"),
                       Line2D([], [], color="k", lw=0.9, label="quasi-static")],
              loc="lower right", frameon=True, framealpha=0.8, edgecolor="none", fontsize=4.8)
    ax = axes[1]                                                       # (e) bistability
    im = ax.pcolormesh(D, bd, gr["R"][..., ic] - gr["R"][..., ia], cmap="magma", vmin=0, vmax=0.8,
                       shading="nearest")
    _beta_overlay(ax, p, K)
    _beta_axes(ax, gr, r"$R_{\mathrm{coh}} - R_{\mathrm{async}}$", im, fig)
    ax = axes[2]                                                       # (f) collective frequency
    Om = np.where(gr["R"][..., ic] > 2.0 / np.sqrt(float(gr["N"])), gr["Omega"][..., ic], np.nan)
    ax.set_facecolor("0.85")                                          # Ω undefined where R ~ N^{-1/2}
    im = ax.pcolormesh(D, bd, Om, cmap=OMEGA_CMAP, shading="nearest",
                       norm=SymLogNorm(linthresh=0.02, vmin=-1.3, vmax=1.3, base=10))
    _beta_overlay(ax, p, K)
    _beta_axes(ax, gr, r"$\Omega$", im, fig)


def main():
    q, val, f3, rp = np.load(QS_NPZ), np.load(VAL_NPZ), np.load(FIG3_NPZ), np.load(RAMP_NPZ)
    bif, s = np.load(BIF_NPZ), np.load(SIN_NPZ)
    K, g = float(q["K"]), float(q["gamma"])
    set_prl_style()
    gr = np.load(GRID_FINE_NPZ if os.path.exists(GRID_FINE_NPZ) else GRID_NPZ)
    p = np.load(PLANE_NPZ)
    fig, axs = plt.subplots(2, 3, figsize=(COL_DOUBLE, 4.5), constrained_layout=True)
    axes = axs[0]
    panel_fold(axes[0], q, rp, K, g)
    axes[0].set_title(r"$G = \cos,\ \mu = 0.03$", fontsize=6.5)
    panel_branches(axes[1], q, val, f3, bif, K, g)
    axes[1].set_title(r"$G = \cos$", fontsize=6.5)
    panel_sin(axes[2], s)
    panels_beta(axs[1], gr, p, K, fig)
    for ax, l in zip(axs.ravel(), "abcdef"):
        _panel_label(ax, l)
    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"saved {OUT}.png/.svg")


if __name__ == "__main__":
    main()
