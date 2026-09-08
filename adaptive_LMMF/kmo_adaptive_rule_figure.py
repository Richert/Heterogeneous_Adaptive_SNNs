r"""
Adaptive Kuramoto rule sweep — relative weight variance V_A/Ā² vs. heterogeneity Δ
==================================================================================

Reads the .npz written by ``kmo_adaptive_rule_sweep.py`` and draws a 2×3 PRL figure:

    rows    : adaptation symmetry — symmetric (top) vs. antisymmetric (bottom)
    columns : adaptation-rule family (decay | logistic | pulse, i.e. the RULES registry)
    panels  : Δ  vs.  V_A/Ā² at the end of the simulation, one marker per trial plus the
              trial mean (a shaded band spans min…max over the random initial conditions)

The layout follows the sweep file: rows = ``symmetries``, columns = ``rules``, so a
rule family added to the sweep shows up as an extra column without touching this script.
The y-range is shared by the two rows of a column, not across columns — the families
differ by more than an order of magnitude in V_A/Ā². V_A and Ā are the off-diagonal weight
statistics stored by the sweep; the steady-state
value is averaged over the last ``tail_frac`` of the simulated time to remove the
residual jitter of the (still slowly drifting) weights.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_rule_figure.py \
        [--npz ...] [--out ...] [--log] [--free-y]
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
from prl_style import set_prl_style, panel_label, COL_DOUBLE
# ---------------------------------------------------------------------------
import argparse

import numpy as np
import matplotlib.pyplot as plt

CONFIG = dict(
    npz=dp.mpmf("kmo_adaptive_rule_sweep.npz"),
    out=dp.mpmf("kmo_adaptive_rule_variance"),
    tail_frac=0.1,                  # average the steady state over the last 10% of T
    log=False,                      # log-scaled V_A/Ā² axis (linear by default)
    share_y=True,                   # common y-range per COLUMN (rule family)
)

# panel titles / row labels (fall back to the raw key for families added later)
RULE_TITLE = {
    "decay":    r"$\dot A = \mu G + \gamma(1-A)$",
    "logistic": r"$\dot A = \mu G\,A(A_m-A)$",
    "pulse":    r"$\dot A = \mu_p G_p (A_m-A) - \mu_d G_d A$",
}
SYM_LABEL = {"sym": "symmetric", "asym": "antisymmetric"}

C_MEAN, C_TRIAL, C_BAND = "#1f77b4", "0.45", "#1f77b4"


def rel_variance(d, tail_frac):
    """(V_A/Ā²)[rule, sym, Δ, trial], averaged over the last `tail_frac` of the run.

    Floored at 0: the sweep evaluates V_A = ⟨A²⟩ − Ā², which cancels to round-off
    (±1e-16, coarsened by float32 storage) once the weights are all but identical —
    the perfectly locked, symmetric-rule limit — and can come out slightly negative."""
    n_t = d["VA"].shape[-1]
    k = max(1, int(round(tail_frac * n_t)))
    VA = d["VA"][..., -k:].mean(axis=-1)
    Abar = d["Abar"][..., -k:].mean(axis=-1)
    return np.maximum(VA / Abar ** 2, 0.0)


def main(cfg=CONFIG):
    d = np.load(cfg["npz"], allow_pickle=True)
    rules = [str(r) for r in d["rules"]]
    syms = [str(s) for s in d["symmetries"]]
    deltas = d["deltas"]
    rel = rel_variance(d, cfg["tail_frac"])                 # (n_rule, n_sym, n_delta, n_trial)
    nr, ns = len(rules), len(syms)

    set_prl_style()
    fig, axes = plt.subplots(ns, nr, figsize=(COL_DOUBLE, 1.55 * ns + 0.45),
                             squeeze=False, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)

    def column_ylim(r):
        """y-range shared by the two symmetries of one rule family (the columns differ
        by more than an order of magnitude, so a figure-wide range flattens most panels)."""
        v = rel[r][np.isfinite(rel[r]) & (rel[r] > 0)]
        if not v.size:
            return (1e-4, 1.0) if cfg["log"] else (-0.03, 1.0)
        if cfg["log"]:
            return 0.5 * v.min(), 2.0 * v.max()
        return -0.03 * v.max(), 1.08 * v.max()                  # linear axis anchored at 0

    ylims = [column_ylim(r) for r in range(nr)]

    letters = iter("abcdefghijkl")
    for s, sym in enumerate(syms):
        for r, rule in enumerate(rules):
            ax = axes[s][r]
            y = rel[r, s]                                   # (n_delta, n_trial)
            ax.fill_between(deltas, np.nanmin(y, axis=1), np.nanmax(y, axis=1),
                            color=C_BAND, alpha=0.18, lw=0, zorder=1)
            for k in range(y.shape[1]):                     # individual random ICs
                ax.plot(deltas, y[:, k], ls="none", marker="o", ms=1.8, mfc="none",
                        mec=C_TRIAL, mew=0.5, zorder=2)
            ax.plot(deltas, np.nanmean(y, axis=1), color=C_MEAN, lw=1.2, marker="o",
                    ms=2.6, mew=0.0, zorder=3)              # trial mean

            if cfg["log"]:
                ax.set_yscale("log")
            if cfg["share_y"]:
                ax.set_ylim(*ylims[r])
            ax.set_xlim(0, deltas.max() * 1.02)
            if s == 0:
                ax.set_title(RULE_TITLE.get(rule, rule), fontsize=7, pad=3)
            if s == ns - 1:
                ax.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
            if r == 0:                       # every column has its own scale -> keep all ticks
                ax.set_ylabel(rf"{SYM_LABEL.get(sym, sym)}" "\n" r"$V_A/\bar A^2$", labelpad=2)
            panel_label(ax, next(letters))

    n_trials = rel.shape[-1]
    axes[0][0].plot([], [], color=C_MEAN, lw=1.2, marker="o", ms=2.6,
                    label=f"mean of {n_trials} ICs")
    axes[0][0].plot([], [], ls="none", marker="o", ms=1.8, mfc="none", mec=C_TRIAL,
                    mew=0.5, label="single IC")
    axes[0][0].legend(loc="lower right", fontsize=5.2, handlelength=1.4, labelspacing=0.2,
                      borderaxespad=0.3)

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    fig.savefig(cfg["out"] + ".svg")
    fig.savefig(cfg["out"] + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {cfg['out']}.svg / .png   "
          f"({nr} rules × {ns} symmetries × {deltas.size} Δ × {n_trials} trials)")


def parse_args(cfg):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--npz", help="sweep file written by kmo_adaptive_rule_sweep.py")
    p.add_argument("--out", help="output stem (.svg/.png are appended)")
    p.add_argument("--tail-frac", type=float, dest="tail_frac",
                   help="fraction of T averaged for the steady state")
    p.add_argument("--log", action="store_true", dest="log", default=None,
                   help="log instead of linear V_A/Ā² axis")
    p.add_argument("--free-y", action="store_false", dest="share_y", default=None,
                   help="per-panel y-limits instead of one range per column")
    args = p.parse_args()
    return {**cfg, **{k: v for k, v in vars(args).items() if v is not None}}


if __name__ == "__main__":
    main(parse_args(CONFIG))
