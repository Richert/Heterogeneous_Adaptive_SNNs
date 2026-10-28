r"""
Adaptive-coupling Kuramoto: Δ-ramp figure (steady states, protocol, coherence, weight variance)
================================================================================================

ONE two-column PRX figure, 4 rows × 4 COLUMNS — the columns are the two adaptation rates μ of the
two adaptation rules, i.e. (G, μ) ∈ {cos, sin} × {0.003, 0.03} at K=1, γ=0.01, the parameters of the
bifurcation diagrams in ``weight_variance_bifurcation_rules.py``:

  * row 1 — STEADY STATES: the analytic mean-field fixed points R(Δ) (blue = synchronized branch,
            solid stable / dashed saddle; grey = asynchronous branch), overlaid with the coherence
            measured at each Δ-step of the microscopic forward (○) and backward (□) ramp; the dotted
            vertical line marks the transcritical point Δ = K/2, the star the saddle-node fold;
  * row 2 — the Δ RAMPS themselves, i.e. the ramp protocol Δ(t);
  * row 3 — PHASE COHERENCE R(t) of the network and of the mean-field equations;
  * row 4 — RELATIVE WEIGHT VARIANCE V_A/Ā²(t) of both, the microscopic estimate being measured from
            the off-diagonal weights at every recorded time point of the ramp.

Throughout rows 2-4, the ramp DIRECTION is encoded by line style and opacity (forward: solid/opaque,
backward: dashed/translucent), and in rows 3-4 the MODEL is encoded by colour (microscopic vs.
mean field).

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

C_MICRO, C_MF, C_ASYNC, C_SYNC = "0.2", "#c1121f", "0.55", "#1f77b4"
C_TC, C_FOLD, C_RAMP = "#ff7f0e", "#2ca02c", "0.2"
#: ramp direction -> line style + opacity (+ marker for the steady-state row)
DIRS = [(0, "forward", dict(ls="-", alpha=1.0), "o"),
        (1, "backward", dict(ls="--", alpha=0.55), "s")]
RULE_TEX = {"cos": r"\cos", "sin": r"\sin"}


set_prl_style = functools.partial(_set_prl_style, "prl",
                                 **{"legend.fontsize": 5.2})


_panel_label = functools.partial(panel_label, dx=-24, dy=3)


def delta_trace(t, deltas, tau_d, direction):
    """The ramp protocol Δ(t): piecewise constant, one `tau_d`-long hold per Δ value."""
    seq = deltas if direction == 0 else deltas[::-1]
    idx = np.clip((np.asarray(t) / tau_d).astype(int), 0, seq.size - 1)
    return seq[idx]


def plot_steady_states(ax, rule, mu, g, K, d_min, d_max, deltas_ramp, Rp_m):
    """Analytic R(Δ) branches + per-Δ-step micro ramp coherence + transcritical/fold markers."""
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
        ax.plot([fp[0]], [fp[1]], marker="*", ms=6.5, mfc=C_FOLD, mec="k", mew=0.4,
                ls="none", zorder=6, clip_on=False)
    for di, _, style, mk in DIRS:                                              # ramp fwd / bwd
        ax.plot(deltas_ramp, Rp_m[di], mk, ms=3.0, mfc="none", mec=C_MICRO, mew=0.8,
                alpha=style["alpha"], ls="none", zorder=5)
    ax.set_xlim(d_min, d_max); ax.set_ylim(-0.03, 1.03)


# ════════════════════════════════════════════════════════════════════════════
#  figure
# ════════════════════════════════════════════════════════════════════════════
def main():
    dm = np.load(MICRO_NPZ)                                    # microscopic ramp data
    dmf = np.load(MF_NPZ)                                      # mean-field ramp data
    rules = [str(r) for r in dm["rules"]]
    mus = dm["mus"]
    deltas, Tm, Rm, Abm, VAm, Rp_m = (dm["deltas"], dm["Tm"], dm["Rm"], dm["Abm"], dm["VAm"],
                                      dm["Rp_m"])
    Tf, Rf, VA, A = dmf["Tf"], dmf["Rf"], dmf["VA"], dmf["A"]
    g, K, tau_d = float(dm["gamma"]), float(dm["K"]), float(dm["tau_d"])
    d_min, d_max = float(dm["delta_min"]), float(dm["delta_max"])
    Ttot = float(dm["n_ramp"]) * tau_d

    cols = [(ri, rule, mi, float(mu))                          # 4 columns: rule × μ
            for ri, rule in enumerate(rules) for mi, mu in enumerate(mus)]

    set_prl_style()
    fig, axes = plt.subplots(4, len(cols), figsize=(COL_DOUBLE, 5.3), squeeze=False,
                             layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.02, wspace=0.04, hspace=0.05)

    for c, (ri, rule, mi, mu) in enumerate(cols):
        # ── row 1: steady states (mean-field branches + per-step ramp coherence) ──
        ax = axes[0][c]
        plot_steady_states(ax, rule, mu, g, K, d_min, d_max, deltas, Rp_m[ri, mi])
        ax.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
        ax.set_title(rf"$G(x)={RULE_TEX.get(rule, rule)}(x)$, $\mu={mu:g}$", fontsize=6.8, pad=3)

        # ── row 2: the ramp protocol Δ(t) ──
        ax = axes[1][c]
        for di, _, style, _ in DIRS:
            ax.plot(Tm, delta_trace(Tm, deltas, tau_d, di), color=C_RAMP, lw=1.0, **style)
        ax.set_ylim(0, d_max)

        # ── row 3: phase coherence R(t), micro vs mean field ──
        ax = axes[2][c]
        for di, _, style, _ in DIRS:
            ax.plot(Tm, Rm[ri, mi, di], color=C_MICRO, lw=0.8, zorder=2, **style)
            ax.plot(Tf, Rf[ri, mi, di], color=C_MF, lw=1.0, zorder=3, **style)
        ax.set_ylim(-0.03, 1.03)

        # ── row 4: relative weight variance V_A/Ā²(t), micro vs mean field ──
        ax = axes[3][c]
        for di, _, style, _ in DIRS:
            ax.plot(Tm, VAm[ri, mi, di] / Abm[ri, mi, di] ** 2, color=C_MICRO, lw=0.8,
                    zorder=2, **style)
            ax.plot(Tf, VA[ri, mi, di] / A[ri, mi, di] ** 2, color=C_MF, lw=1.0, zorder=3, **style)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"time $t$", labelpad=1)

        for r in (1, 2, 3):                                    # rows 2-4 share the time axis
            axes[r][c].set_xlim(0, Ttot)
            if r < 3:
                axes[r][c].tick_params(labelbottom=False)
        for r in range(4):
            _panel_label(axes[r][c], "abcdefghijklmnop"[4 * r + c])

    for r, ylab in enumerate([r"coherence $R$", r"heterogeneity $\Delta$", r"coherence $R$",
                              r"rel. variance $V_A/\bar A^2$"]):
        axes[r][0].set_ylabel(ylab, labelpad=2)

    # legends: branches/markers (a), ramp direction (e), model (i)
    axes[0][0].legend(handles=[Line2D([0], [0], color=C_SYNC, lw=1.2, label="sync. (stable)"),
                               Line2D([0], [0], color=C_SYNC, lw=1.0, ls="--", label="sync. (saddle)"),
                               Line2D([0], [0], color=C_ASYNC, lw=1.1, label="async."),
                               Line2D([0], [0], marker="*", ls="none", mfc=C_FOLD, mec="k",
                                      mew=0.4, ms=5.5, label="fold")],
                      loc="lower left", fontsize=4.3, handlelength=1.3, labelspacing=0.15,
                      borderaxespad=0.25)
    axes[1][0].legend(handles=[Line2D([0], [0], color=C_RAMP, lw=1.0, label=name, **style)
                               for _, name, style, _ in DIRS],
                      loc="upper left", fontsize=4.6, handlelength=1.8, labelspacing=0.15,
                      borderaxespad=0.25)
    axes[2][0].legend(handles=[Line2D([0], [0], color=C_MICRO, lw=1.0, label="micro"),
                               Line2D([0], [0], color=C_MF, lw=1.0, label="mean field")],
                      loc="lower left", fontsize=4.6, handlelength=1.3, labelspacing=0.15,
                      borderaxespad=0.25)

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT + ".svg"); fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT}.svg / .png   ({len(cols)} columns: "
          + ", ".join(f"{r}/μ={m:g}" for _, r, _, m in cols) + ")")


if __name__ == "__main__":
    main()
