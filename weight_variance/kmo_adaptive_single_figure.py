r"""
Adaptive Kuramoto — micro vs. single Ott–Antonsen ensemble: comparison figures
==============================================================================

Loads the sweep written by ``kmo_adaptive_single_sweep.py`` and renders ONE two-column
PRL-style figure PER adaptation rule (cos, sin, |sin|), each a 2×4 grid:

  col 1 (stacked):
    (a) steady-state phase coherence R (R(t) time-averaged over the final TAIL_FRAC of the
        trace) vs Δ — microscopic (solid) vs single-ensemble mean field (dashed), one
        coloured line per μ; a vertical marker shows the Δ where micro R drops below R_THRESH.
    (b) relative weight variance V_A/Ā² vs Δ — microscopic (solid, VA_micro/Abar_micro² over
        the same end window) vs mean field (dashed): the analytical closed-form closure
        (Eqs. 32/37) for cos, and 0 for the variance-free sin / |sin| single-ensemble MF.
  row 1, cols 2-4 — the microscopic coupling matrix A_ij at small / intermediate / large Δ
        (EX_FRACS of the μ=EXAMPLE_MU grid); the Δ title is colour-coded.
  row 2, cols 2-4 — a single panel with the R(t) dynamics of those 3 examples: colour = Δ
        (matching the matrix titles), line style = model (micro solid / mean field dashed).

Reads the tidy CSV (discriminated by the `quantity` column); the Δ grid is per-μ.

Run with any numpy/pandas/matplotlib env, e.g.:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_single_figure.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
from prl_style import set_prl_style as _set_prl_style
from prl_style import panel_label
# ---------------------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# analytical mean-field weight-variance closure (cos rule; Eqs. 32/37)
import weight_variance_analysis as WVA

CSV = dp.mpmf("kmo_adaptive_single_sweep.csv")
OUT_SUMMARY = dp.mpmf("kmo_adaptive_single_summary")
OUT_RULE = dp.mpmf("kmo_adaptive_single_{tag}")

C_MICRO = "0.2"
C_MF = "#c1121f"
MU_CMAP = "viridis"           # colormap encoding μ in the summary lineplots
MATRIX_CMAP = "magma"         # colormap for the final coupling matrices
MU_PLOT = None                # μ values to plot per-rule (None = all in the data)
TAIL_FRAC = 0.2               # steady-state window: fraction of the trace END time-averaged
R_THRESH = 0.8                # steady-state coherence threshold marked by a vertical line
EXAMPLE_MU = None             # μ for the coupling-matrix & R(t) examples (None => largest μ)
EX_DELTAS = (0.05, 0.2, 0.8)  # target example Δ (small/intermediate/large); snapped to nearest sweep value
EX_COLORS = ["#4477AA", "#EE7733", "#CC3311"]   # per-example colour (matrix title ↔ R(t) curve)

_TAG = {"cos": "cos", "sin": "sin", "|sin|": "absin"}   # filename-safe rule tags


# ════════════════════════════════════════════════════════════════════════════
#  PRL style
# ════════════════════════════════════════════════════════════════════════════
set_prl_style = functools.partial(_set_prl_style, "prl",
                                 **{"axes.formatter.useoffset": False})


_panel_label = functools.partial(panel_label, dx=-24, dy=3)


# ════════════════════════════════════════════════════════════════════════════
#  load + metrics
# ════════════════════════════════════════════════════════════════════════════
def load(csv):
    df = pd.read_csv(csv)
    rules = list(df["G_A"].dropna().unique())
    mus = sorted(df["mu"].dropna().unique())
    if "trial" not in df.columns:                       # back-compat with single-trial CSVs
        df["trial"] = 0
    trials = sorted(int(t) for t in df["trial"].dropna().unique())
    # Δ grid is PER-μ (kmo_adaptive_single_sweep scales it with each μ's synchronized-branch
    # endpoint), so a global Δ list would mismatch; key the Δ values by μ instead.
    deltas_by_mu = {mu: sorted(df.loc[(df.mu == mu) & (df.quantity == "R_micro"), "Delta"]
                               .dropna().unique()) for mu in mus}

    traces = {}
    for q in ("R_micro", "R_mf", "Abar_micro", "Abar_mf", "VA_micro"):
        for (rule, D, mu, tr), g in df[df.quantity == q].groupby(["G_A", "Delta", "mu", "trial"]):
            g = g.sort_values("time")
            traces[(q, rule, float(D), float(mu), int(tr))] = (g["time"].to_numpy(), g["value"].to_numpy())

    mats = {}
    for (rule, D, mu, tr), g in df[df.quantity == "A_final"].groupby(["G_A", "Delta", "mu", "trial"]):
        nr, nc = int(g["row"].max()) + 1, int(g["col"].max()) + 1
        M = np.full((nr, nc), np.nan)
        M[g["row"].astype(int), g["col"].astype(int)] = g["value"].to_numpy()
        mats[(rule, float(D), float(mu), int(tr))] = M

    K = float(df["K"].dropna().iloc[0]); gamma = float(df["gamma"].dropna().iloc[0])
    return rules, deltas_by_mu, mus, trials, traces, mats, K, gamma


def rmse_R_dynamics(traces, rule, D, mu, tr):
    """RMSE over time between the micro and MF coherence dynamics R(t)."""
    _, rm = traces[("R_micro", rule, D, mu, tr)]
    _, rf = traces[("R_mf", rule, D, mu, tr)]
    n = min(len(rm), len(rf))
    return float(np.sqrt(np.mean((rm[:n] - rf[:n]) ** 2)))


def _tail_mean(v, tail=TAIL_FRAC):
    """Mean over the final `tail` fraction of a trace (steady-state estimate)."""
    return float(np.mean(v[-max(1, int(tail * len(v))):]))


def steady_R(traces, q, rule, D, mu, tr):
    """Steady-state phase coherence: R(t) time-averaged over the trace's end window (q is
    "R_micro" or "R_mf")."""
    _, r = traces[(q, rule, D, mu, tr)]
    return _tail_mean(r)


def va_ratio_micro(traces, rule, D, mu, tr):
    """Microscopic relative weight variance V_A/Ā², each moment tail-averaged over the trace's
    end window (steady state)."""
    _, va = traces[("VA_micro", rule, D, mu, tr)]
    _, ab = traces[("Abar_micro", rule, D, mu, tr)]
    return _tail_mean(va) / _tail_mean(ab) ** 2


def va_ratio_mf(rule, Dgrid, mu, K, gamma):
    """Mean-field V_A/Ā² over a Δ grid. For the cos rule this is the analytical closed-form
    synchronized-branch prediction (Eqs. 32/37); off the sync branch (async state) and for the
    variance-free sin / |sin| single-ensemble mean field it is 0."""
    Dg = np.asarray(Dgrid, float)
    if rule != "cos":
        return np.zeros_like(Dg)
    br = WVA.branches(Dg, mu, K, gamma)["sync"]
    return np.nan_to_num(br["VA"] / br["A"] ** 2, nan=0.0)


def _trial_stats(fn, trials):
    """Mean and std (across trials) of a per-trial RMSE callable fn(tr)."""
    vals = np.array([fn(tr) for tr in trials], float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.std())


def representative_trial(traces, rule, D, mu, trials):
    """The trial whose coherence-dynamics RMSE is the median across trials."""
    order = sorted(trials, key=lambda tr: rmse_R_dynamics(traces, rule, D, mu, tr))
    return order[len(order) // 2]


def r_drop_delta(traces, rule, mu, Dmu, trials, thresh=R_THRESH):
    """Δ at which the steady-state microscopic R first falls below `thresh` (linearly
    interpolated between grid points); None if it never does."""
    D = np.asarray(Dmu)
    R = np.array([_trial_stats(lambda tr: steady_R(traces, "R_micro", rule, d, mu, tr), trials)[0]
                  for d in Dmu])
    below = np.where(R < thresh)[0]
    if below.size == 0:
        return None
    i = below[0]
    if i == 0:
        return float(D[0])
    r0, r1, d0, d1 = R[i - 1], R[i], D[i - 1], D[i]
    return float(d0 + (thresh - r0) * (d1 - d0) / (r1 - r0))


# ════════════════════════════════════════════════════════════════════════════
#  per-rule figure (two-column PRL): 2 rows × 4 columns
#    col 1  — (a) steady-state R and (b) V_A/Ā² vs Δ (micro solid / MF dashed, per μ);
#             vertical marker at the Δ where micro R drops below R_THRESH.
#    row 1, cols 2-4 — micro coupling matrix A_ij at small / intermediate / large Δ.
#    row 2, cols 2-4 — one panel: R(t) for those 3 examples (colour = Δ, style = model).
# ════════════════════════════════════════════════════════════════════════════
def make_rule_figure(rule, deltas_by_mu, mus, trials, traces, mats, K, gamma):
    cmap = plt.get_cmap(MU_CMAP)
    colors = [cmap(0.12 + 0.76 * i / max(1, len(mus) - 1)) for i in range(len(mus))]
    ex_mu = EXAMPLE_MU if EXAMPLE_MU is not None else max(mus)   # μ for the example panels
    Dex = np.asarray(deltas_by_mu[ex_mu])
    ex_deltas = [float(Dex[np.argmin(np.abs(Dex - d))]) for d in EX_DELTAS]   # snap to grid

    fig = plt.figure(figsize=(7.0, 3.4), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.06)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.4, 1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_mats = [fig.add_subplot(gs[0, 1 + k]) for k in range(3)]
    ax_dyn = fig.add_subplot(gs[1, 1:4])

    # ── column 1: (a) steady-state R, (b) V_A/Ā² vs Δ — micro solid, MF dashed, per μ ──
    for mi, mu in enumerate(mus):
        Dmu = deltas_by_mu[mu]; Dx = np.asarray(Dmu)
        mRm, sRm = np.array([_trial_stats(lambda tr: steady_R(traces, "R_micro", rule, D, mu, tr), trials)
                             for D in Dmu]).T
        mRf, _ = np.array([_trial_stats(lambda tr: steady_R(traces, "R_mf", rule, D, mu, tr), trials)
                           for D in Dmu]).T
        ax_a.errorbar(Dx, mRm, yerr=sRm, color=colors[mi], ls="-", marker="o", ms=2.2, lw=0.9,
                      capsize=1.5, elinewidth=0.6, label=f"{mu:g}")
        ax_a.plot(Dx, mRf, color=colors[mi], ls="--", lw=1.0)
        mV, sV = np.array([_trial_stats(lambda tr: va_ratio_micro(traces, rule, D, mu, tr), trials)
                           for D in Dmu]).T
        ax_b.errorbar(Dx, mV, yerr=sV, color=colors[mi], ls="-", marker="o", ms=2.2, lw=0.9,
                      capsize=1.5, elinewidth=0.6)
        ax_b.plot(Dx, va_ratio_mf(rule, Dmu, mu, K, gamma), color=colors[mi], ls="--", lw=1.0)
        dc = r_drop_delta(traces, rule, mu, Dmu, trials)          # Δ where micro R < R_THRESH
        if dc is not None:
            for ax in (ax_a, ax_b):
                ax.axvline(dc, color=colors[mi], ls=":", lw=0.9, zorder=0)
    ax_a.axhline(R_THRESH, color="0.7", ls=":", lw=0.5, zorder=0)
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
    ax_a.set_ylabel(r"steady-state $R$", labelpad=2)
    ax_b.set_ylabel(r"$V_A/\bar A^2$", labelpad=2)
    ax_b.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
    ax_a.set_xticklabels([])
    leg = ax_a.legend(title=r"$\mu$", ncol=1, fontsize=5.5, title_fontsize=6, handlelength=1.3, loc="best")
    ax_a.add_artist(leg)
    ax_b.legend(handles=[Line2D([0], [0], color="0.3", ls="-", label="micro"),
                         Line2D([0], [0], color="0.3", ls="--", label="mean field")],
                fontsize=5.5, handlelength=1.6, loc="best")
    _panel_label(ax_a, "a"); _panel_label(ax_b, "b")

    # ── row 1, cols 2-4: micro coupling matrix at 3 example Δ (colour-linked to row 2) ──
    for k, (D, axm, col) in enumerate(zip(ex_deltas, ax_mats, EX_COLORS)):
        tr = representative_trial(traces, rule, D, ex_mu, trials)
        M = mats[(rule, float(D), float(ex_mu), tr)]
        im = axm.imshow(M, origin="lower", aspect="auto", cmap=MATRIX_CMAP,
                        vmin=np.nanmin(M), vmax=np.nanmax(M), interpolation="nearest")
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title(rf"$\Delta={D:.2g}$", color=col, fontsize=6.5, pad=2)
        for sp in axm.spines.values():
            sp.set_color(col); sp.set_linewidth(1.1)
        cb = fig.colorbar(im, ax=axm, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=4.5, pad=0.6)
        axm.set_xlabel(r"osc. $j$", labelpad=1)
    ax_mats[0].set_ylabel(r"osc. $i$ (by $\omega_i$)", labelpad=2)
    _panel_label(ax_mats[0], "c", dx=-4)

    # ── row 2, cols 2-4: R(t) for the 3 examples — colour = Δ, line style = model ──
    t = None
    for D, col in zip(ex_deltas, EX_COLORS):
        tr = representative_trial(traces, rule, D, ex_mu, trials)
        t, Rm = traces[("R_micro", rule, float(D), float(ex_mu), tr)]
        _, Rf = traces[("R_mf", rule, float(D), float(ex_mu), tr)]
        ax_dyn.plot(t, Rm, color=col, ls="-", lw=0.9)
        ax_dyn.plot(t, Rf, color=col, ls="--", lw=0.9)
    ax_dyn.set_xlim(t[0], t[-1]); ax_dyn.set_ylim(-0.02, 1.02)
    ax_dyn.set_yticks([0, 0.5, 1.0])
    ax_dyn.set_xlabel(r"time $t$", labelpad=1)
    ax_dyn.set_ylabel(r"$R(t)$", labelpad=2)
    d_handles = [Line2D([0], [0], color=EX_COLORS[k], lw=1.1, label=rf"${ex_deltas[k]:.2g}$")
                 for k in range(len(ex_deltas))]
    leg_d = ax_dyn.legend(handles=d_handles, title=r"$\Delta$", loc="upper right", ncol=1,
                          fontsize=5.5, title_fontsize=6, handlelength=1.4, labelspacing=0.25)
    ax_dyn.add_artist(leg_d)
    ax_dyn.legend(handles=[Line2D([0], [0], color="0.3", ls="-", label="micro"),
                           Line2D([0], [0], color="0.3", ls="--", label="mean field")],
                  loc="lower right", fontsize=5.5, handlelength=1.6, labelspacing=0.25)
    _panel_label(ax_dyn, "d", dx=-24)

    fig.suptitle(rf"$G_A = {rule}$   (matrices & dynamics at $\mu={ex_mu:g}$)",
                 fontsize=8, x=0.01, ha="left")
    out = OUT_RULE.format(tag=_TAG.get(rule, rule))
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {out}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    rules, deltas_by_mu, mus, trials, traces, mats, K, gamma = load(CSV)
    mus_plot = mus if MU_PLOT is None else [m for m in mus if m in MU_PLOT]

    set_prl_style()
    for rule in rules:
        make_rule_figure(rule, deltas_by_mu, mus_plot, trials, traces, mats, K, gamma)


if __name__ == "__main__":
    main()
