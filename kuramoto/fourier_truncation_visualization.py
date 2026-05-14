"""
Plot K × n_terms Sweep Results — PRX style
============================================
Renders a 3×3 figure summarising a sweep produced by
``fourier_truncation_gridsearch.py``.  Columns correspond to a chosen
subset of K values; rows show

    Row 1 — corr_A     (Pearson correlation between coupling matrices)
    Row 2 — rmse_sbar  (RMSE between empirical s̄_ml and OA truncation)
    Row 3 — mean phase coherence R for KMO and OA, with the temporal
            variance of R(t) shown as error bars

All curves are plotted vs. n_terms; markers indicate per-trial means and
error bars indicate across-trial std (rows 1 & 2) or the mean temporal
variance of R(t) reported by the sweep (row 3).

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


# Match the colour palette used in kuramoto_ensemble_fitting.py
C_KM = "#1f4e79"     # KMO  — deep blue
C_OA = "#c44e52"     # OA   — muted red
C_AUX = "#4c4c4c"    # grey accent (used for corr_A and rmse_sbar in this fig)
C_AUX2 = "#3a8e7c"    # grey accent (used for corr_A and rmse_sbar in this fig)

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
                       mean_R_km_mean   = ("mean_R_km",  "mean"),
                       mean_R_oa_mean   = ("mean_R_oa",  "mean"),
                       var_R_km_mean    = ("var_R_km",   "mean"),
                       var_R_oa_mean    = ("var_R_oa",   "mean"),
                       n_trials         = ("trial",      "count"))
                  .sort_index())
        out[K] = agg
    return out


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_sweep(agg, K_values, params, savepath):
    set_prx_style()

    K_list = [K for K in K_values if K in agg]
    if not K_list:
        raise RuntimeError("No K values to plot — check --K and the CSV.")
    n_cols = len(K_list)

    # Two-column PRX width = 7.0 in (used by the reference figure).  For the
    # 3×3 grid we keep that width and choose a height that gives roughly
    # square panels.
    fig = plt.figure(figsize=(7.0, 7.2))
    gs = gridspec.GridSpec(
        nrows=3, ncols=n_cols, figure=fig,
        hspace=0.45, wspace=0.45,
        left=0.10, right=0.97, top=0.93, bottom=0.085,
    )

    # Row-wide y-limits for rows 1 and 2 so columns are visually comparable
    corr_lo = min(agg[K]["corr_A_mean"].min() - agg[K]["corr_A_std"].max()
                  for K in K_list)
    corr_hi = max(agg[K]["corr_A_mean"].max() + agg[K]["corr_A_std"].max()
                  for K in K_list)
    rmse_lo = min((agg[K]["rmse_sbar_mean"] - agg[K]["rmse_sbar_std"]).min()
                  for K in K_list)
    rmse_hi = max((agg[K]["rmse_sbar_mean"] + agg[K]["rmse_sbar_std"]).max()
                  for K in K_list)
    corr_pad = 0.03 * (corr_hi - corr_lo) if corr_hi > corr_lo else 0.02
    rmse_pad = 0.03 * (rmse_hi - rmse_lo) if rmse_hi > rmse_lo else 0.02

    # Row-major panel labels: (a)(b)(c) on row 1, (d)(e)(f) on row 2, …
    letters = "abcdefghijklmnop"
    label_at = lambda row, col: letters[row * n_cols + col]

    for j, K in enumerate(K_list):
        a = agg[K]
        n_terms_axis = a.index.to_numpy()

        # ── Row 1: corr_A ────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, j])
        ax.errorbar(n_terms_axis, a["corr_A_mean"], yerr=a["corr_A_std"],
                    fmt="o-", color=C_AUX, mfc=C_AUX, mec=C_AUX,
                    capsize=2.5, lw=1.2, ms=4.0)
        ax.set_title(fr"$K = {K:g}$")
        if j == 0:
            ax.set_ylabel(r"$\mathrm{corr}(A^{\mathrm{KO}},"
                          r"\, A^{\mathrm{OA}})$")
        ax.set_ylim(corr_lo - corr_pad, min(1.01, corr_hi + corr_pad))
        make_panel_label(ax, f"({label_at(0, j)})")

        # ── Row 2: rmse_sbar ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, j])
        ax.errorbar(n_terms_axis, a["rmse_sbar_mean"], yerr=a["rmse_sbar_std"],
                    fmt="s-", color=C_AUX2, mfc=C_AUX2, mec=C_AUX2,
                    capsize=2.5, lw=1.2, ms=4.0)
        if j == 0:
            ax.set_ylabel(r"$\mathrm{RMSE}(\bar s^{\mathrm{KO}},\,"
                          r"\bar s^{\mathrm{OA}})$")
        ax.set_ylim(max(0.0, rmse_lo - rmse_pad), rmse_hi + rmse_pad)
        make_panel_label(ax, f"({label_at(1, j)})")

        # ── Row 3: mean R(t) with var(R) error bars ──────────────────────
        ax = fig.add_subplot(gs[2, j])
        ax.errorbar(n_terms_axis, a["mean_R_km_mean"],
                    yerr=a["var_R_km_mean"],
                    fmt="o-", color=C_KM, mfc=C_KM, mec=C_KM,
                    capsize=2.5, lw=1.2, ms=4.0,
                    label=r"KMO")
        ax.errorbar(n_terms_axis, a["mean_R_oa_mean"],
                    yerr=a["var_R_oa_mean"],
                    fmt="s--", color=C_OA, mfc=C_OA, mec=C_OA,
                    capsize=2.5, lw=1.2, ms=4.0,
                    label=r"OA")
        if j == 0:
            ax.set_ylabel(r"$\langle R(t) \rangle_t$")
            ax.legend(loc="lower right", frameon=False,
                      handlelength=2.2, borderaxespad=0.3)
        ax.set_xlabel(r"$n_f$")
        ax.set_ylim(0.0, 1.05)
        make_panel_label(ax, f"({label_at(2, j)})")

    # Integer x-ticks for n_terms across every panel
    all_nt = sorted({int(nt) for K in K_list for nt in agg[K].index.tolist()})
    for ax in fig.axes:
        ax.set_xticks(all_nt)

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