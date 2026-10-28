r"""
Symmetric adaptation: mean field vs. structure-aware quasi-static self-consistency vs. network
==============================================================================================

Supplementary figure, one two-column row of two panels (the fold panel `panel_fold` is used by the
main-text figure weight_variance_feedback_figure.py):

  (a) V_A/Ā² vs Δ for the four μ of Fig. 3 (colour): network (●, uniform-weight MF-fixed-point init;
      ○, initialized on the quasi-static state beyond the mean-field fold), mean field (dashed,
      Eq. va_ss_full), quasi-static (solid);
  (b) distance from the OA manifold vs Δ: network ⟨|Z_2 − Z²|⟩_t vs quasi-static |W − R²| (solid);
      dotted line N^{-1/2}.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_cos_quasistatic_figure.py
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

from weight_variance_bifurcation_rules import ra_branches, fold_point
from weight_variance_rule_meanfield import mf_sync_RV

QS_NPZ = dp.mpmf("weight_variance_cos_quasistatic.npz")
FIG3_NPZ = dp.mpmf("weight_variance_rule_micro_oainit_uniform_init_z2.npz")
RAMP_NPZ = dp.kmo_adaptive("weight_variance_ramp_micro.npz")
OUT = dp.mpmf("weight_variance_cos_quasistatic")
MU_RAMP = 0.03
C_SYNC, C_ASYNC, C_QS = "#1f77b4", "0.55", "0.1"

set_prl_style = functools.partial(_set_prl_style, "prl", **{"legend.fontsize": 5.4})
_panel_label = functools.partial(panel_label, dx=-24, dy=3)


def _qs_row(q, mu):
    return int(np.argmin(np.abs(q["mus"] - mu)))


def panel_fold(ax, q, rp, K, g):
    d = np.linspace(1e-3, 1.4, 1200)
    up, lo, asy = ra_branches("cos", d, MU_RAMP, g, K)
    ax.plot(d[d > K / 2], asy["R"][d > K / 2], color=C_ASYNC, lw=1.0)
    ax.plot(d[d <= K / 2], asy["R"][d <= K / 2], color=C_ASYNC, lw=0.9, ls="--")
    ax.plot(d, up["R"], color=C_SYNC, lw=1.1)
    ax.plot(d, lo["R"], color=C_SYNC, lw=0.9, ls="--")
    fp = fold_point("cos", MU_RAMP, g, K)
    ax.plot(fp[0], fp[1], "*", ms=7, mfc=C_SYNC, mec="k", mew=0.4, zorder=5)
    m = _qs_row(q, MU_RAMP)
    R = q["R"][m]
    ok = np.isfinite(R)
    ax.plot(q["deltas"][ok], R[ok], color=C_QS, lw=1.1, zorder=4)
    j = np.nonzero(ok)[0][-1]
    ax.plot(q["deltas"][j], R[j], "*", ms=7, mfc="w", mec=C_QS, mew=0.8, zorder=5)
    ri, mi = list(rp["rules"]).index("cos"), int(np.argmin(np.abs(rp["mus"] - MU_RAMP)))
    for di, mk in ((0, "o"), (1, "s")):
        ax.plot(rp["deltas"], rp["Rp_m"][ri, mi, di], mk, ms=3.0, mfc="none", mec="0.25", mew=0.8,
                zorder=6)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$R$")
    ax.set_title(rf"$\mu = {MU_RAMP:g}$", fontsize=6.5)
    ax.legend(handles=[Line2D([], [], color=C_SYNC, lw=1.1, label="mean field"),
                       Line2D([], [], color=C_QS, lw=1.1, label="quasi-static"),
                       Line2D([], [], color="0.25", ls="none", marker="o", ms=3, mfc="none",
                              label="network, forward"),
                       Line2D([], [], color="0.25", ls="none", marker="s", ms=3, mfc="none",
                              label="network, backward")],
              loc="lower left", frameon=False)


def main():
    q, f3 = np.load(QS_NPZ), np.load(FIG3_NPZ)
    val = np.load(dp.mpmf("weight_variance_cos_quasistatic_validate.npz"))
    K, g = float(q["K"]), float(q["gamma"])
    set_prl_style()
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.3), constrained_layout=True)
    axes = [None] + list(axes)                                     # keep the panel indices 1, 2

    ri = list(f3["rules"]).index("cos")
    n = f3["t"].size
    win = slice(int(round(0.8 * n)), None)                         # final 20 % of each run
    mus = f3["mus"]
    cols = [colormaps["viridis"](x) for x in np.linspace(0.0, 0.85, mus.size)]
    d = np.linspace(1e-3, 1.4, 1200)
    for m, (mu, c) in enumerate(zip(mus, cols)):
        VA = (f3["VA"][ri, m, :, win] / f3["Abar"][ri, m, :, win] ** 2).mean(axis=1)
        Z1, Z2 = f3["Z1"][ri, m, :, win], f3["Z2"][ri, m, :, win]
        dist = np.abs(Z2 - Z1 ** 2).mean(axis=1)
        k = _qs_row(q, mu)
        ok = np.isfinite(q["R"][k])
        _, rel_mf = mf_sync_RV("cos", d, mu, g, K)
        axes[1].plot(d, rel_mf, color=c, lw=0.9, ls="--")
        axes[1].plot(q["deltas"][ok], q["VArel"][k][ok], color=c, lw=1.2)
        axes[1].plot(f3["deltas"], VA, "o", ms=2.8, mfc=c, mec="k", mew=0.3)
        axes[2].plot(q["deltas"][ok], np.abs(q["W"][k] - q["R"][k] ** 2)[ok], color=c, lw=1.2)
        axes[2].plot(f3["deltas"], dist, "o", ms=2.8, mfc=c, mec="k", mew=0.3)
        sel = np.isclose(val["mu"], mu)
        axes[1].plot(val["delta"][sel], val["VArel"][sel], "o", ms=3.0, mfc="w", mec=c, mew=0.9)
        axes[2].plot(val["delta"][sel], val["oa_dist"][sel], "o", ms=3.0, mfc="w", mec=c, mew=0.9)
    axes[2].axhline(float(f3["N"]) ** -0.5, color="0.5", ls=":", lw=0.8)
    for ax in axes[1:]:
        ax.set_xlim(0, 1.35)
        ax.set_xlabel(r"$\Delta$")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$V_A/\bar A^2$")
    axes[2].set_ylabel(r"$\langle|Z_2 - Z^2|\rangle_t$,  $|W - R^2|$")
    axes[1].set_ylim(1e-4, 30)
    axes[1].legend(handles=[Line2D([], [], color="0.3", ls="--", lw=0.9, label="mean field"),
                            Line2D([], [], color="0.3", lw=1.2, label="quasi-static"),
                            Line2D([], [], color="0.3", ls="none", marker="o", ms=2.8, mec="k", mew=0.3,
                                   label="network, MF init"),
                            Line2D([], [], color="0.3", ls="none", marker="o", ms=3.0, mfc="w", mew=0.9,
                                   label="network, QS init")],
                   loc="upper center", frameon=False, ncol=2, handlelength=1.6, columnspacing=1.0)
    axes[2].legend(handles=[Line2D([], [], color=c, lw=1.2, label=rf"$\mu = {mu:g}$")
                            for mu, c in zip(mus, cols)], loc="upper left", frameon=False)
    for ax, l in zip(axes[1:], "ab"):
        _panel_label(ax, l)
    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"saved {OUT}.png/.svg")


if __name__ == "__main__":
    main()
