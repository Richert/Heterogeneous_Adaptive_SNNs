"""
Plot K × n_terms Sweep Results — PRX style
============================================
Renders a 1×3 figure summarising a sweep produced by
``fourier_truncation_gridsearch.py``.  Each panel shows a different
quality-of-approximation metric as a function of the Fourier truncation
order ``n_terms``, with one line per value of the coupling strength K:

    Col 1 — log10(corr_A)   Pearson correlation between the KMO
                            coarse-grained coupling matrix and the OA
                            coupling matrix (log scale for visibility).
    Col 2 — rmse_sbar       RMSE between the empirical s̄_ml from the
                            KMO and the OA Fourier-truncated counterpart.
    Col 3 — nrmse_R         Normalised RMSE between the KMO and OA
                            macroscopic phase coherence R(t).

Markers indicate per-K means across trials and error bars show the
across-trial standard deviation (with the appropriate propagated
uncertainty for column 1).

The matplotlib style, figure-size convention, panel labels, and colour
scheme match `kuramoto_ensemble_fitting.py` so this plot fits in the same
Physical Review manuscript.

Usage
-----
    python plot_sweep_results.py --csv sweep_K_nterms_results.csv
    python plot_sweep_results.py --csv ... --K 1 3 5 --out fig.pdf
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── PRX figure style (mirrors kuramoto_ensemble_fitting.py) ─────────────────

def set_prx_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["STIXGeneral", "Times New Roman", "Times"],
        "mathtext.fontset":   "stix",
        "font.size":          12,
        "axes.titlesize":     12,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "axes.linewidth":     0.7,
        "lines.linewidth":    1.2,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "savefig.dpi":        300,
        "figure.dpi":         150,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


# Match the colour palette used in kuramoto_ensemble_fitting.py.
# With one line per K value in every panel, we use a 3-colour palette
# anchored at the same deep blue and muted red used throughout the manuscript.
K_PALETTE = ["#1f4e79", "#3a8e7c", "#c44e52"]   # blue → teal → red

def make_panel_label(ax, label, *, x=-0.22, y=1.04, fontsize=12):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            ha="left", va="bottom")


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate(df, K_values):
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    out = {}
    for K in K_values:
        sub = df[np.isclose(df["K"], K)]
        if sub.empty:
            print(f"  warning: no rows for K={K}, skipping", file=sys.stderr)
            continue
        agg = (sub.groupby("n_terms")
                  .agg(corr_A_mean      = ("corr_A",     "mean"),
                       corr_A_std       = ("corr_A",     "std"),
                       rmse_sbar_mean   = ("rmse_sbar",  "mean"),
                       rmse_sbar_std    = ("rmse_sbar",  "std"),
                       nrmse_R_mean     = ("nrmse_R",    "mean"),
                       nrmse_R_std      = ("nrmse_R",    "std"),
                       n_trials         = ("trial",      "count"))
                  .sort_index())

        # log10(corr_A): only defined for positive correlations.
        # Standard error propagation: d(log10 x)/dx = 1/(x ln 10), so
        # std(log10 X) ≈ std(X) / (X · ln 10) to first order.
        pos = agg["corr_A_mean"] > 0
        if not pos.all():
            bad = agg.index[~pos].tolist()
            print(f"  warning: K={K}: corr_A ≤ 0 at n_terms={bad}; "
                  "log10(corr_A) masked to NaN there", file=sys.stderr)
        agg["log_corr_A_mean"] = np.where(
            pos, np.log10(agg["corr_A_mean"].where(pos)), np.nan,
        )
        agg["log_corr_A_std"] = np.where(
            pos,
            agg["corr_A_std"]
            / (agg["corr_A_mean"].where(pos) * np.log(10.0)),
            np.nan,
        )

        out[K] = agg
    return out


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_sweep(agg, K_values, params, savepath):
    set_prx_style()

    K_list = [K for K in K_values if K in agg]
    if not K_list:
        raise RuntimeError("No K values to plot — check --K and the CSV.")

    # Map each K to a colour from the shared palette, padding/repeating if
    # the caller passed more or fewer than three K values.
    palette = K_PALETTE[:]
    while len(palette) < len(K_list):
        palette.extend(K_PALETTE)
    K_colour = {K: palette[i] for i, K in enumerate(K_list)}

    # Two-column PRX width = 7.0 in.  Single-row figure is shorter than the
    # 3×3 layout; ~2.9 in height keeps the panels approximately square.
    fig = plt.figure(figsize=(7.0, 2.9))
    gs = gridspec.GridSpec(
        nrows=1, ncols=3, figure=fig,
        wspace=0.45,
        left=0.085, right=0.985, top=0.88, bottom=0.20,
    )

    ax_corr = fig.add_subplot(gs[0, 0])
    ax_sbar = fig.add_subplot(gs[0, 1])
    ax_R    = fig.add_subplot(gs[0, 2])

    for K in K_list:
        a = agg[K]
        n_terms_axis = a.index.to_numpy()
        colour = K_colour[K]
        label  = fr"$K = {K:g}$"

        # ── Col 1: log10(corr_A) ─────────────────────────────────────────
        ax_corr.errorbar(
            n_terms_axis, a["log_corr_A_mean"], yerr=a["log_corr_A_std"],
            fmt="o-", color=colour, mfc=colour, mec=colour,
            capsize=2.5, lw=1.2, ms=4.0, label=label,
        )

        # ── Col 2: rmse_sbar ─────────────────────────────────────────────
        ax_sbar.errorbar(
            n_terms_axis, a["rmse_sbar_mean"], yerr=a["rmse_sbar_std"],
            fmt="s-", color=colour, mfc=colour, mec=colour,
            capsize=2.5, lw=1.2, ms=4.0, label=label,
        )

        # ── Col 3: nrmse_R ───────────────────────────────────────────────
        ax_R.errorbar(
            n_terms_axis, a["nrmse_R_mean"], yerr=a["nrmse_R_std"],
            fmt="^-", color=colour, mfc=colour, mec=colour,
            capsize=2.5, lw=1.2, ms=4.0, label=label,
        )

    ax_corr.set_ylabel(r"$\log_{10}\,\mathrm{corr}"
                       r"(A^{\mathrm{KO}},\, A^{\mathrm{OA}})$")
    ax_sbar.set_ylabel(r"$\mathrm{RMSE}(\bar s^{\mathrm{KO}},\,"
                       r"\bar s^{\mathrm{OA}})$")
    ax_R.set_ylabel(r"$\mathrm{NRMSE}(R^{\mathrm{KO}},\, R^{\mathrm{OA}})$")
    for ax in (ax_corr, ax_sbar, ax_R):
        ax.set_xlabel(r"$n_f$")

    # Lower bound of 0 for the (N)RMSE panels; let upper bound auto-scale.
    for ax in (ax_sbar, ax_R):
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(0.0, lo), hi)

    # Single shared legend in the first panel.
    ax_corr.legend(loc="lower right", frameon=False,
                   handlelength=1.8, borderaxespad=0.3,
                   labelspacing=0.25)

    # Integer x-ticks for n_terms across every panel, with a small left/right
    # margin so the leftmost tick label doesn't collide with the y-axis.
    all_nt = sorted({int(nt) for K in K_list for nt in agg[K].index.tolist()})
    for ax in (ax_corr, ax_sbar, ax_R):
        ax.set_xticks(all_nt)
        ax.margins(x=0.05)

    # Panel labels (a), (b), (c)
    for col, ax in enumerate((ax_corr, ax_sbar, ax_R)):
        make_panel_label(ax, f"({'abc'[col]})")

    fig.savefig(savepath, bbox_inches="tight")
    print(f"Figure saved → {savepath}")
    plt.show()
    return fig


# ─── CLI ─────────────────────────────────────────────────────────────────────

def pick_K_values(df, requested):
    all_K = sorted(df["K"].unique().tolist())
    if requested:
        chosen = [K for K in requested if any(np.isclose(K, k) for k in all_K)]
        missing = [K for K in requested
                   if not any(np.isclose(K, k) for k in all_K)]
        if missing:
            print(f"  warning: requested K values {missing} not in CSV; "
                  f"CSV has {all_K}", file=sys.stderr)
        return chosen or all_K[:3]
    if len(all_K) <= 3:
        return all_K
    idx = np.linspace(0, len(all_K) - 1, 3).round().astype(int)
    return [all_K[i] for i in idx]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", required=True,
                   help="Path to the sweep CSV produced by "
                        "fourier_truncation_gridsearch.py")
    p.add_argument("--K", type=float, nargs="+", default=None,
                   help="K values to use as columns "
                        "(default: three evenly spaced K's from the CSV)")
    p.add_argument("--out", default="sweep_summary.pdf",
                   help="Output figure path (PDF for PRX submission, "
                        "PNG also accepted)")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    if "status" in df.columns:
        n_ok = (df["status"] == "ok").sum()
        n_total = len(df)
        if n_ok < n_total:
            print(f"  {n_total - n_ok}/{n_total} rows have status != 'ok' "
                  "and will be dropped")

    K_values = pick_K_values(df, args.K)
    print(f"Plotting K = {K_values}")
    agg = aggregate(df, K_values)
    plot_sweep(agg, K_values, params="", savepath=args.out)


if __name__ == "__main__":
    main()