r"""
Adaptive-coupling Kuramoto: weight ratios vs. phase coherence — figure
======================================================================

Loads the microscopic (K, μ) sweep written by ``weight_variance_coherence_sweep.py`` and renders a
single-column PRL figure: the scale-free weight statistics V_A/Ā² and C_A/Ā² (two side-by-side panels)
plotted against the phase coherence R.  Each point is one (K, μ) combination; points sharing a μ are
joined into a curve (sorted by R) and coloured per μ.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_coherence_figure.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
from prl_style import set_prl_style as _set_prl_style
from prl_style import panel_label
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MICRO_NPZ = dp.kmo_adaptive("weight_variance_coherence_sweep.npz")
MF_NPZ = dp.kmo_adaptive("weight_variance_coherence_meanfield.npz")
OUT = dp.kmo_adaptive("weight_variance_coherence")

MU_CMAP = "viridis"


set_prl_style = functools.partial(_set_prl_style, "prl",
                                 **{"legend.fontsize": 5.6})


_panel_label = functools.partial(panel_label, dx=-24, dy=3)


def _ratios(d):
    return {"R": d["R"], "ratio_VA": d["VA"] / d["Abar"] ** 2, "ratio_CA": d["CA"] / d["Abar"] ** 2}


def main():
    dm = np.load(MICRO_NPZ)
    dmf = np.load(MF_NPZ)
    mus = dm["mus"]
    micro, mfd = _ratios(dm), _ratios(dmf)                        # micro; micro-driven MF (R = micro R)
    rmse_R = np.abs(dmf["R_full"] - dm["R"])                      # full-MF vs micro window-avg R, per (K,μ)
    cmap = plt.get_cmap(MU_CMAP)
    n_mu = len(mus)
    mu_col = [cmap(0.12 + 0.76 * i / max(1, n_mu - 1)) for i in range(n_mu)]   # one colour per μ

    set_prl_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.45), squeeze=False, layout="constrained")

    # (a) V_A/Ā² vs phase coherence R — micro (solid + markers) vs micro-driven mean field (dashed)
    ax = axes[0][0]
    for i in range(n_mu):
        om = np.argsort(micro["R"][i])                            # order each μ-curve by R
        ax.plot(micro["R"][i][om], micro["ratio_VA"][i][om], "-o", color=mu_col[i], ms=3.0,
                mec="0.3", mew=0.3, lw=0.9, label=rf"$\mu={mus[i]:g}$")
        of = np.argsort(mfd["R"][i])
        ax.plot(mfd["R"][i][of], mfd["ratio_VA"][i][of], "--", color=mu_col[i], lw=1.0)
    ax.set_ylabel(r"$V_A/\bar A^2$", labelpad=2)
    ax.set_xlabel(r"phase coherence $R$", labelpad=1)
    ax.margins(0.06)
    _panel_label(ax, "a")

    # (b) micro V_A/Ā² vs RMSE between the full mean field and micro in window-averaged R
    ax = axes[0][1]
    for i in range(n_mu):
        ax.plot(rmse_R[i], micro["ratio_VA"][i], "o", color=mu_col[i], ms=3.0, mec="0.3", mew=0.3)
    ax.set_ylabel(r"$V_A/\bar A^2$", labelpad=2)
    ax.set_xlabel(r"RMSE in $R$", labelpad=1)
    ax.margins(0.06)
    _panel_label(ax, "b")

    # micro/mean-field style legend on panel (a); μ-colour legend on panel (b), both upper left
    style_h = [Line2D([0], [0], color="0.3", marker="o", ms=3.0, lw=0.9, label="micro"),
               Line2D([0], [0], color="0.3", ls="--", lw=1.0, label="mean field")]
    mu_h = [Line2D([0], [0], color=mu_col[i], marker="o", ms=3.0, lw=0.9, label=rf"$\mu={mus[i]:g}$")
            for i in range(n_mu)]
    axes[0][0].legend(handles=style_h, loc="upper left", handletextpad=0.4, labelspacing=0.2,
                      borderaxespad=0.3)
    axes[0][1].legend(handles=mu_h, loc="upper left", handletextpad=0.4, labelspacing=0.2,
                      borderaxespad=0.3)
    fig.savefig(OUT + ".svg"); fig.savefig(OUT + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {OUT}.svg / .png")


if __name__ == "__main__":
    main()
