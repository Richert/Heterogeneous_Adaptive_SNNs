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


def main():
    q, val, f3, rp = np.load(QS_NPZ), np.load(VAL_NPZ), np.load(FIG3_NPZ), np.load(RAMP_NPZ)
    bif, s = np.load(BIF_NPZ), np.load(SIN_NPZ)
    K, g = float(q["K"]), float(q["gamma"])
    set_prl_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.2), constrained_layout=True)
    panel_fold(axes[0], q, rp, K, g)
    axes[0].set_title(r"$G = \cos,\ \mu = 0.03$", fontsize=6.5)
    panel_branches(axes[1], q, val, f3, bif, K, g)
    axes[1].set_title(r"$G = \cos$", fontsize=6.5)
    panel_sin(axes[2], s)
    for ax, l in zip(axes, "abc"):
        _panel_label(ax, l)
    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"saved {OUT}.png/.svg")


if __name__ == "__main__":
    main()
