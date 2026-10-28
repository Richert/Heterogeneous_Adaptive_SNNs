r"""
Antisymmetric adaptation: collective frequency shift — Supplementary figure
============================================================================

One two-column row of three panels, from weight_variance_sin_drift.npz:

  (a) Ω/κ vs κ = μ/γ: microscopic network (circles), quasi-static self-consistency (solid), first-order
      coefficient (dashed), one colour per Δ;
  (b) phase coherence R vs κ, same encoding, with the non-adaptive Kuramoto value √(1 − 2Δ/K) (dotted);
  (c) per-oscillator mean frequency ⟨θ̇_i⟩ − Ω vs ω_i at the (Δ, κ) stored as the profile: network (dots)
      vs quasi-static prediction (line); dotted lines mark the symmetric κ = 0 locking band ±KR.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_sin_drift_figure.py
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

NPZ = dp.kmo_adaptive("weight_variance_sin_drift.npz")
OUT = dp.kmo_adaptive("weight_variance_sin_drift")
SHADES = (0.55, 0.9)                     # YlOrBr shades per Δ (sin-rule colours of the connectivity fig.)

set_prl_style = functools.partial(_set_prl_style, "prl", **{"legend.fontsize": 5.6})
_panel_label = functools.partial(panel_label, dx=-24, dy=3)


def main():
    d = np.load(NPZ)
    deltas, kappas, K = d["deltas"], d["kappas"], float(d["K"])
    cols = [colormaps["YlOrBr"](s) for s in SHADES]
    has_micro = "mi_Omega" in d.files
    set_prl_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.0), constrained_layout=True)

    ax = axes[0]                                                   # (a) Ω/κ vs κ
    for a, (D, c) in enumerate(zip(deltas, cols)):
        ax.plot(kappas, d["qs_Omega"][a] / kappas, "-", color=c, lw=1.2)
        ax.axhline(d["fo_coef"][a], color=c, ls="--", lw=0.9)
        if has_micro:
            ax.plot(kappas, d["mi_Omega"][a] / kappas, "o", ms=3.2, mfc="none", mec=c, mew=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa = \mu/\gamma$")
    ax.set_ylabel(r"$\Omega/\kappa$")

    ax = axes[1]                                                   # (b) R vs κ
    for a, (D, c) in enumerate(zip(deltas, cols)):
        ax.plot(kappas, d["qs_R"][a], "-", color=c, lw=1.2)
        ax.axhline(np.sqrt(1.0 - 2.0 * D / K), color=c, ls=":", lw=0.9)
        if has_micro:
            ax.plot(kappas, d["mi_R"][a], "o", ms=3.2, mfc="none", mec=c, mew=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa = \mu/\gamma$")
    ax.set_ylabel(r"$R$")
    handles = [Line2D([], [], color=c, lw=1.2, label=rf"$\Delta = {D:g}$") for D, c in zip(deltas, cols)]
    handles += [Line2D([], [], color="0.3", ls="none", marker="o", ms=3.2, mfc="none", label="network"),
                Line2D([], [], color="0.3", lw=1.2, label="quasi-static"),
                Line2D([], [], color="0.3", ls="--", lw=0.9, label="first order"),
                Line2D([], [], color="0.3", ls=":", lw=0.9, label=r"$\kappa = 0$")]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=2, handlelength=1.8)

    ax = axes[2]                                                   # (c) per-oscillator frequencies
    Dp, kp = d["profile"]
    c = cols[int(np.argmin(np.abs(deltas - Dp)))]
    om = d["omega_profile"]
    b0 = K * np.sqrt(1.0 - 2.0 * Dp / K)
    for x in (-b0, b0):
        ax.axvline(x, color="0.5", ls=":", lw=0.8)
    ax.plot(om, d["qs_w_eff"], "-", color="0.15", lw=0.7, zorder=2)
    if has_micro:
        ax.plot(om, d["mi_w_eff"], "o", ms=2.4, mfc=c, mec="none", alpha=0.9, zorder=3)
    ax.legend(handles=[Line2D([], [], color=c, ls="none", marker="o", ms=2.4, label="network"),
                       Line2D([], [], color="0.15", lw=0.7, label="quasi-static")],
              loc="upper left", frameon=False)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel(r"$\omega_i$")
    ax.set_ylabel(r"$\langle\dot\theta_i\rangle_t - \Omega$")
    ax.set_title(rf"$\Delta = {Dp:g},\ \kappa = {kp:g}$", fontsize=6.5)
    axin = ax.inset_axes([0.62, 0.08, 0.35, 0.35])                 # zoom on the locking plateau
    axin.plot(om, d["qs_w_eff"], "-", color="0.15", lw=0.6, zorder=2)
    if has_micro:
        axin.plot(om, d["mi_w_eff"], "o", ms=1.8, mfc=c, mec="none", zorder=3)
    for x in (-b0, b0):
        axin.axvline(x, color="0.5", ls=":", lw=0.7)
    axin.set_xlim(-1.0, 1.0)
    axin.set_ylim(-0.25, 0.25)
    axin.tick_params(labelsize=4.5, length=1.5, pad=1)

    for ax, l in zip(axes, "abc"):
        _panel_label(ax, l)
    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"saved {OUT}.png/.svg")


if __name__ == "__main__":
    main()
