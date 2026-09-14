r"""
Adaptive-coupling Kuramoto: Δ-ramp figures (bifurcation + ramp dynamics), one per rule
======================================================================================

ONE two-column PRL figure PER ADAPTATION RULE — ``..._cos`` (symmetric) and ``..._sin``
(antisymmetric) — each 2 rows × 3 columns (rows = μ ∈ {0.003, 0.03}, i.e. the monostable and the
bistable cut of ``weight_variance_bifurcation_rules.py``, same K=1, γ=0.01):
  * col 1 — phase-coherence bifurcation diagram R vs Δ (analytic branches + the per-Δ-step
            microscopic ramp coherence, forward ○ / backward □; dotted line at the transcritical
            Δ = K/2, star at the saddle-node fold where it exists);
  * col 2 — FORWARD ramp (Δ increasing): R(t) micro vs mean field by colour, with the RELATIVE
            WEIGHT VARIANCE V_A/Ā²(t) on a secondary y-axis — mean field (solid) and the
            microscopic estimate (dashed), the latter measured from the off-diagonal weights at
            every recorded time point of the ramp;
  * col 3 — the same for the BACKWARD ramp (Δ decreasing).

Loads the microscopic ramp data (weight_variance_ramp_micro.npz) and the mean-field ramp data
(weight_variance_ramp_meanfield.npz); the analytic branches are recomputed from the saved parameters.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp_figure.py
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
from matplotlib.lines import Line2D

from weight_variance_bifurcation_rules import ra_branches, fold_point   # analytic Δ-branches

MICRO_NPZ = dp.kmo_adaptive("weight_variance_ramp_micro.npz")
MF_NPZ = dp.kmo_adaptive("weight_variance_ramp_meanfield.npz")
OUT = dp.kmo_adaptive("weight_variance_ramp")

C_MICRO, C_MF, C_ASYNC, C_SYNC, C_VC = "0.2", "#c1121f", "0.55", "#1f77b4", "#2a9d8f"
C_TC, C_FOLD = "#ff7f0e", "#2ca02c"
DIRS = [(0, "fwd", "forward ramp"), (1, "bwd", "backward ramp")]
RULE_TEX = {"cos": r"\cos", "sin": r"\sin"}
RULE_NAME = {"cos": "symmetric", "sin": "antisymmetric"}


set_prl_style = functools.partial(_set_prl_style, "prl",
                                 **{"legend.fontsize": 5.2})


_panel_label = functools.partial(panel_label, dx=-26, dy=3)


def plot_R_bifurcation(ax, rule, mu, g, K, d_min, d_max, deltas_ramp, Rp_m):
    """Analytic R(Δ) branches + per-step micro ramp coherence + transcritical/fold markers."""
    d = np.linspace(max(d_min, 1e-4), d_max, 900)
    up, lo, asy = ra_branches(rule, d, mu, g, K)
    Dtc = K / 2.0
    m = d > Dtc
    ax.plot(d[m], asy["R"][m], color=C_ASYNC, lw=1.1, ls="-", zorder=2)        # async stable
    ax.plot(d[~m], asy["R"][~m], color=C_ASYNC, lw=1.0, ls="--", zorder=2)     # async unstable
    ax.plot(d, up["R"], color=C_SYNC, lw=1.2, ls="-", zorder=3)                # sync stable
    ax.plot(d, lo["R"], color=C_SYNC, lw=1.0, ls="--", zorder=3)               # sync saddle
    ax.axvline(Dtc, color=C_TC, ls=":", lw=0.9, zorder=1)                      # transcritical Δ=K/2
    fp = fold_point(rule, mu, g, K)
    if fp is not None:                                                         # saddle-node fold
        ax.plot([fp[0]], [fp[1]], marker="*", ms=7, mfc=C_FOLD, mec="k", mew=0.4,
                ls="none", zorder=6, clip_on=False)
    for di, mk in ((0, "o"), (1, "s")):                                        # ramp fwd / bwd
        ax.plot(deltas_ramp, Rp_m[di], mk, ms=3.0, mfc="none", mec=C_MICRO, mew=0.8,
                ls="none", zorder=5)
    ax.set_xlim(d_min, d_max); ax.set_ylim(-0.03, 1.03)


def rule_figure(rule, ri, dm, dmf):
    """One ramp figure for adaptation rule `rule` (index `ri` in the saved arrays)."""
    mus = dm["mus"]
    deltas, Tm, Rm, Abm, VAm, Rp_m = (dm["deltas"], dm["Tm"], dm["Rm"], dm["Abm"], dm["VAm"],
                                      dm["Rp_m"])
    Tf, Rf, VA, A = dmf["Tf"], dmf["Rf"], dmf["VA"], dmf["A"]
    g, K = float(dm["gamma"]), float(dm["K"])
    d_min, d_max = float(dm["delta_min"]), float(dm["delta_max"])
    Ttot = float(dm["n_ramp"]) * float(dm["tau_d"])

    set_prl_style()
    fig, axes = plt.subplots(len(mus), 3, figsize=(COL_DOUBLE, 1.45 * len(mus) + 0.35),
                             squeeze=False, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.03, wspace=0.05, hspace=0.08)
    r_handles = q_handles = None

    for i, mu in enumerate(mus):
        # ── col 1: R(Δ) bifurcation diagram + ramp markers ────────────────────
        ax = axes[i][0]
        plot_R_bifurcation(ax, rule, float(mu), g, K, d_min, d_max, deltas, Rp_m[ri, i])
        ax.set_ylabel(rf"$\mu={mu:g}$" + "\n" + r"coherence $R$", labelpad=2)
        if i == len(mus) - 1:
            ax.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)

        # ── cols 2-3: forward / backward ramp R(t) + V_A/Ā²(t) (MF & micro) ───
        for di, key, name in DIRS:
            ax = axes[i][di + 1]
            lmic, = ax.plot(Tm, Rm[ri, i, di], color=C_MICRO, lw=0.9, label="micro $R$")
            lmf, = ax.plot(Tf, Rf[ri, i, di], color=C_MF, lw=1.1, label="mean-field $R$")
            ax.set_xlim(0, Ttot); ax.set_ylim(-0.03, 1.03)
            if i == len(mus) - 1:
                ax.set_xlabel(r"time $t$", labelpad=1)

            tw = ax.twinx()                                    # relative weight variance
            lvf, = tw.plot(Tf, VA[ri, i, di] / A[ri, i, di] ** 2, color=C_VC, lw=1.0, ls="-",
                           label=r"mean-field $V_A/\bar A^2$")
            lvm, = tw.plot(Tm, VAm[ri, i, di] / Abm[ri, i, di] ** 2, color=C_VC, lw=0.9, ls="--",
                           label=r"micro $V_A/\bar A^2$")
            tw.tick_params(axis="y", colors=C_VC); tw.spines["right"].set_color(C_VC)
            tw.set_ylim(bottom=0.0)
            if di == 1:                                        # label only the rightmost twin
                tw.set_ylabel(r"$V_A / \bar A^2$", color=C_VC, labelpad=2)
            r_handles = [lmic, lmf]; q_handles = [lvf, lvm]

        if i == 0:
            for c, title in enumerate(["coherence bifurcation"] + [n for _, _, n in DIRS]):
                axes[0][c].set_title(title, fontsize=6.8, pad=2)

    # legends
    handles = [Line2D([0], [0], color=C_SYNC, lw=1.2, label="sync. (stable)"),
               Line2D([0], [0], color=C_SYNC, lw=1.0, ls="--", label="sync. (saddle)"),
               Line2D([0], [0], color=C_ASYNC, lw=1.1, label="async."),
               Line2D([0], [0], marker="o", ls="none", mfc="none", mec=C_MICRO, mew=0.8, ms=3,
                      label="ramp fwd"),
               Line2D([0], [0], marker="s", ls="none", mfc="none", mec=C_MICRO, mew=0.8, ms=3,
                      label="ramp bwd")]
    if rule == "cos":
        handles.insert(2, Line2D([0], [0], marker="*", ls="none", mfc=C_FOLD, mec="k", mew=0.4,
                                 ms=6, label="fold"))
    axes[0][0].legend(handles=handles, loc="upper right", fontsize=4.4, handlelength=1.3,
                      labelspacing=0.16)
    axes[0][1].legend(handles=r_handles + q_handles, loc="upper right", fontsize=4.8,
                      handlelength=1.5, labelspacing=0.16)

    for i in range(len(mus)):
        for c in range(3):
            _panel_label(axes[i][c], "abcdef"[3 * i + c])

    out = f"{OUT}_{rule}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".svg"); fig.savefig(out + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {out}.svg / .png   (G = {rule}, {RULE_NAME.get(rule, '')} adaptation)")


def main():
    dm = np.load(MICRO_NPZ)                                    # microscopic ramp data
    dmf = np.load(MF_NPZ)                                      # mean-field ramp data
    rules = [str(r) for r in dm["rules"]]
    for ri, rule in enumerate(rules):
        rule_figure(rule, ri, dm, dmf)


if __name__ == "__main__":
    main()
