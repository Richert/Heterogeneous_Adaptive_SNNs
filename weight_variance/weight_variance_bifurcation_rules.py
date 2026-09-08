r"""
Adaptive-coupling Kuramoto: Δ-bifurcation diagrams + microscopic weight variance (cos, sin)
============================================================================================

Two single-column PRL figures (one per adaptation rule G ∈ {cos, sin}) built from the MEAN-FIELD
STEADY-STATE equations for R and Ā only (Eqs. 8 & 10; no closed-form weight variance here), overlaid
with microscopic simulations from ``weight_variance_bifurcation_micro.py``:

  row 1 : 2-D bifurcation diagram in the μ–Δ plane (μ on x, Δ on y). Fold curve Δ_SN(μ) and the
          transcritical line Δ=K/2, the cusp codim-2 point (μ=γ, Δ=K/2), the bistable region, and
          vertical cuts at the two μ values shown below. COS ONLY (sin has only a bare Δ=K/2 line).
  row 2 : R(Δ) 1-D bifurcation diagram, one column per μ — synchronized (stable) / saddle (unstable)
          / asynchronous branches, fold + transcritical markers, and the micro R at steady state.
  row 3 : Ā(Δ) 1-D bifurcation diagram, same layout (OMITTED for sin, where Ā≡1).
  row 4 : relative weight variance V_A/Ā²(Δ) — MICROSCOPIC ESTIMATE ONLY (markers), no mean field.

The cos figure therefore has 4 rows (2-D, R, Ā, V_A/Ā²); the sin figure has 2 rows (R, V_A/Ā²).

Mean-field fixed points: synchronized Ā₊ (stable) + saddle Ā₋ (unstable) for cos [Ā≡1 for sin], plus
the asynchronous branch R=0, Ā=1 (stable for Δ>K/2). Micro markers on the sync/async branches are
coloured to match their branch (see the paired micro script for the initialization).

    python weight_variance_bifurcation_rules.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from weight_variance_analysis import (branches, sync_delta_end,          # noqa: E402
                                      set_prl_style, _panel_label)

CONFIG = dict(
    K=1.0, gamma=0.01,
    mu_bi=0.05,       # bistable cut (μ > γ)   — cos only; for sin both cuts are monostable
    mu_mono=0.003,    # monostable cut (μ < γ)
    delta_min=1e-3, delta_max=1.2, n_delta=1400,
    mu_min=1e-3, mu_max=3e-1,                     # μ-axis of the 2-D plane (spans γ)
    rules=["cos", "sin"],
    micro_npz=dp.mpmf("weight_variance_bif_micro.npz"),
    out_dir=dp.MPMF, out_stem="weight_variance_bif",
)

C_ASYNC, C_SYNC = "0.55", "#1f77b4"
C_FOLD, C_TC, C_CUSP = "#2ca02c", "#ff7f0e", "k"
MARK = {"fold":          dict(marker="*", ms=10, mfc=C_FOLD, mec="k", mew=0.4),
        "transcritical": dict(marker="D", ms=5.0, mfc=C_TC, mec="k", mew=0.4),
        "cusp":          dict(marker="X", ms=7.0, mfc=C_CUSP, mec="k", mew=0.4)}
MSYNC = dict(marker="o", ms=3.3, mfc=C_SYNC, mec="k", mew=0.3, ls="none")
MASYNC = dict(marker="o", ms=3.3, mfc=C_ASYNC, mec="k", mew=0.3, ls="none")
Q_LABEL = {"R": r"phase coherence $\tilde R$", "Abar": r"avg. weight $\bar A$",
           "VA": r"rel. variance $V_A/\bar A^2$"}
RULE_TEX = {"cos": r"\cos", "sin": r"\sin"}


# ════════════════════════════════════════════════════════════════════════════
#  mean-field R, Ā branches (no weight-variance closure)
# ════════════════════════════════════════════════════════════════════════════
def ra_branches(rule, d, mu, g, K):
    """(R, Ā) arrays over Δ-grid `d` for the sync branch (`up`), the saddle branch (`lo`, cos only),
    and the asynchronous branch (`asy`; R=0, Ā=1)."""
    ones = np.ones_like(d)
    asy = dict(R=np.zeros_like(d), Abar=ones)
    if rule == "cos":
        br = branches(d, mu, K, g)
        up = dict(R=br["sync"]["R"], Abar=br["sync"]["A"])
        lo = dict(R=br["saddle"]["R"], Abar=br["saddle"]["A"])
        return up, lo, asy
    R2 = 1.0 - 2.0 * d / K                                        # sin: Ā ≡ 1
    phys = R2 > 0
    up = dict(R=np.where(phys, np.sqrt(np.clip(R2, 0, None)), np.nan),
              Abar=np.where(phys, 1.0, np.nan))
    nan = np.full_like(d, np.nan)
    return up, dict(R=nan, Abar=nan), asy


def fold_point(rule, mu, g, K):
    """(Δ_SN, R_SN, Ā_SN) of the saddle-node fold, or None (sin, or cos with μ≤γ)."""
    if not (rule == "cos" and mu > g):
        return None
    dSN = sync_delta_end(mu, K, g)
    A_SN = (g + mu) / (2 * g)
    R2_SN = max(1.0 - 2.0 * dSN / (K * A_SN), 0.0)
    return dSN, np.sqrt(R2_SN), A_SN


# ════════════════════════════════════════════════════════════════════════════
#  row 1: 2-D bifurcation plane in the μ–Δ plane (μ on x, Δ on y; fixed K)
# ════════════════════════════════════════════════════════════════════════════
def plot_2d(ax, cfg, rule):
    g, K = cfg["gamma"], cfg["K"]
    Dtc = K / 2.0
    mu = np.logspace(np.log10(cfg["mu_min"]), np.log10(cfg["mu_max"]), 600)
    ax.axhline(Dtc, color=C_TC, lw=1.2, ls="--", zorder=3)                       # transcritical (Δ=K/2)
    if rule == "cos":
        dSN = K * (g + mu) ** 2 / (8.0 * g * mu)                                 # saddle-node curve Δ_SN(μ)
        phys = mu > g
        ax.fill_between(mu[phys], Dtc, dSN[phys], color=C_SYNC, alpha=0.13, lw=0, zorder=0)
        ax.plot(mu[phys], dSN[phys], color=C_FOLD, lw=1.3, zorder=4)             # physical fold
        ax.plot(mu[~phys], dSN[~phys], color=C_FOLD, lw=0.9, ls=":", zorder=4)   # unphysical continuation
        ax.plot([g], [Dtc], zorder=6, ls="none", **MARK["cusp"])                 # cusp (μ=γ, Δ=K/2)
        ax.annotate("bistable", xy=(1.6 * g, 0.5 * (Dtc + min(cfg["delta_max"], 0.9))),
                    fontsize=5.4, color=C_SYNC, ha="left", rotation=90, va="center")
    for muv in (cfg["mu_bi"], cfg["mu_mono"]):
        ax.axvline(muv, color="0.5", lw=0.7, ls=":", zorder=1)
        ax.annotate(rf"$\mu={muv:g}$", xy=(muv, cfg["delta_max"]), xytext=(1, -1),
                    textcoords="offset points", ha="left", va="top", fontsize=5.2, color="0.35")
    ax.set_xscale("log")
    ax.set_xlim(cfg["mu_min"], cfg["mu_max"]); ax.set_ylim(0, cfg["delta_max"])
    ax.set_xlabel(r"adaptation rate $\mu$", labelpad=1)
    ax.set_ylabel(r"heterogeneity $\Delta$", labelpad=2)
    ax.set_title(r"$\mu$–$\Delta$ bifurcation diagram", fontsize=6.8, pad=3)
    if rule == "cos":
        handles = [Line2D([0], [0], color=C_FOLD, lw=1.3, label="fold"),
                   Line2D([0], [0], color=C_TC, lw=1.2, ls="--", label="transcrit."),
                   Line2D([0], [0], ls="none", label="cusp", **MARK["cusp"])]
    else:                                                                        # sin: only transcritical
        handles = [Line2D([0], [0], color=C_TC, lw=1.2, ls="--", label="transcrit.")]
    ax.legend(handles=handles, loc="upper right", fontsize=4.8, handlelength=1.4,
              labelspacing=0.2, borderaxespad=0.3)
    _panel_label(ax, "a", dx=-26)


# ════════════════════════════════════════════════════════════════════════════
#  row 2/3: 1-D bifurcation diagram of R or Ā vs Δ, at one μ, + micro markers
# ════════════════════════════════════════════════════════════════════════════
def plot_1d(ax, quant, mu, cfg, rule, mrule):
    g, K = cfg["gamma"], cfg["K"]
    Dtc = K / 2.0
    d = np.linspace(cfg["delta_min"], cfg["delta_max"], cfg["n_delta"])
    up, lo, asy = ra_branches(rule, d, mu, g, K)

    va = asy[quant]                                                              # asynchronous branch
    hi, low = d >= Dtc, d < Dtc
    ax.plot(d[hi], va[hi], color=C_ASYNC, lw=1.2, ls="-", zorder=2)              # stable  (Δ>K/2)
    ax.plot(d[low], va[low], color=C_ASYNC, lw=1.0, ls=":", zorder=2)           # unstable (Δ<K/2)
    ax.plot(d, up[quant], color=C_SYNC, lw=1.2, ls="-", zorder=3)               # sync (stable)
    ax.plot(d, lo[quant], color=C_SYNC, lw=1.0, ls=":", zorder=3)               # saddle (unstable)

    y_tc = 0.0 if quant == "R" else 1.0                                         # transcritical (Δ=K/2)
    ax.plot([Dtc], [y_tc], ls="none", zorder=6, clip_on=False, **MARK["transcritical"])
    fp = fold_point(rule, mu, g, K)
    if fp is not None:                                                          # saddle-node fold
        dSN, R_SN, A_SN = fp
        y_f = R_SN if quant == "R" else A_SN
        ax.plot([dSN], [y_f], ls="none", zorder=6, clip_on=False, **MARK["fold"])

    if mrule is not None:                                                       # micro markers per branch
        dd, msync, masy = mrule["deltas"], mrule[quant][:, 0], mrule[quant][:, 1]
        ax.plot(dd, msync, zorder=5, **MSYNC)
        ax.plot(dd, masy, zorder=5, **MASYNC)

    ax.set_xlim(0, cfg["delta_max"])
    if quant == "R":
        ax.set_ylim(-0.03, 1.05)
    else:
        top = np.nanmax(up[quant][np.isfinite(up[quant])]) if np.isfinite(up[quant]).any() else 1.0
        ax.set_ylim(0, 1.15 * max(top, 1.0))


# ════════════════════════════════════════════════════════════════════════════
#  row 4: microscopic V_A/Ā² vs Δ (no mean field)
# ════════════════════════════════════════════════════════════════════════════
def plot_micro_va(ax, cfg, mrule):
    if mrule is not None:
        dd, vsync, vasy = mrule["deltas"], mrule["VA"][:, 0], mrule["VA"][:, 1]
        ax.plot(dd, vsync, zorder=5, **MSYNC)
        ax.plot(dd, vasy, zorder=5, **MASYNC)
    ax.set_yscale("log")
    ax.set_xlim(0, cfg["delta_max"])


# ════════════════════════════════════════════════════════════════════════════
#  one full figure for a given rule
# ════════════════════════════════════════════════════════════════════════════
def micro_for(micro, rule, mu):
    """Slice the micro npz for one (rule, μ): dict of (deltas, R, Abar, VA) each (n_delta, 2 branches)."""
    if micro is None:
        return None
    rules = [str(r) for r in micro["rules"]]
    if rule not in rules:
        return None
    ri = rules.index(rule)
    mi = int(np.argmin(np.abs(micro["mus"] - mu)))
    return dict(deltas=micro["deltas"], R=micro["R"][ri, mi], Abar=micro["Abar"][ri, mi],
                VA=micro["VArel"][ri, mi])


def make_figure(cfg, rule, micro):
    has_2d = (rule == "cos")          # sin: drop the 2-D row (only a transcritical line — nothing happens)
    quants = ["R", "Abar", "VA"] if rule == "cos" else ["R", "VA"]              # sin: drop Ā row too
    nrow = len(quants) + (1 if has_2d else 0)
    mus = [cfg["mu_mono"], cfg["mu_bi"]]                                        # left: μ<γ, right: μ>γ

    hr = ([1.0] if has_2d else []) + [0.75] * len(quants)                       # 1-D rows: 3/4 of the 2-D row
    set_prl_style()
    fig = plt.figure(figsize=(3.4, 1.28 * sum(hr) + 0.35), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.05)
    gs = fig.add_gridspec(nrow, 2, height_ratios=hr)

    row0 = 0
    if has_2d:
        plot_2d(fig.add_subplot(gs[0, :]), cfg, rule)
        row0 = 1

    letters = iter("bcdefghij" if has_2d else "abcdefgh")
    first_1d = None
    row_axes = []
    for r, q in enumerate(quants):
        row_axes.append([])
        for c, mu in enumerate(mus):
            ax = fig.add_subplot(gs[r + row0, c])
            row_axes[r].append(ax)
            first_1d = first_1d or ax
            mrule = micro_for(micro, rule, mu)
            if q == "VA":
                plot_micro_va(ax, cfg, mrule)
            else:
                plot_1d(ax, q, mu, cfg, rule, mrule)
            if r == 0:
                ax.set_title(rf"$\mu={mu:g}$", fontsize=6.8, pad=3)
            if r == len(quants) - 1:
                ax.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
            if c == 0:
                ax.set_ylabel(Q_LABEL[q], labelpad=2)
            _panel_label(ax, next(letters))

    for axes in row_axes:                                                       # share the y-range across columns
        lims = [ax.get_ylim() for ax in axes]
        lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
        for c, ax in enumerate(axes):
            ax.set_ylim(lo, hi)
            if c > 0:
                ax.tick_params(labelleft=False)

    # branch/marker legend in the top-left 1-D panel (R, μ_mono)
    handles = [Line2D([0], [0], color=C_SYNC, lw=1.2, label="sync. (stable)"),
               Line2D([0], [0], color=C_SYNC, lw=1.0, ls=":", label="saddle (unstable)"),
               Line2D([0], [0], color=C_ASYNC, lw=1.2, label="async."),
               Line2D([0], [0], label="micro (sync)", **MSYNC),
               Line2D([0], [0], label="micro (async)", **MASYNC)]
    if rule == "cos":
        handles.insert(3, Line2D([0], [0], ls="none", label="fold", **MARK["fold"]))
    else:                                                                        # sin: no saddle/fold
        handles = [handles[0], handles[2], handles[3], handles[4]]
    first_1d.legend(handles=handles, loc="lower left", fontsize=4.4, handlelength=1.3,
                    labelspacing=0.18, borderaxespad=0.3)

    out = f"{cfg['out_dir']}/{cfg['out_stem']}_{rule}"
    os.makedirs(cfg["out_dir"], exist_ok=True)
    fig.savefig(out + ".svg"); fig.savefig(out + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {out}.svg / .png")


def main(cfg=CONFIG):
    micro = np.load(cfg["micro_npz"], allow_pickle=True) if os.path.exists(cfg["micro_npz"]) else None
    if micro is None:
        print(f"[warn] micro npz not found ({cfg['micro_npz']}); drawing mean field only")
    for rule in cfg["rules"]:
        make_figure(cfg, rule, micro)


if __name__ == "__main__":
    main()
