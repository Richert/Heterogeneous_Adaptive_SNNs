"""
Plot K × n_terms Sweep Results
================================
Renders a 3×3 figure summarising a sweep produced by
``fourier_truncation_gridsearch.py``.  Columns correspond to a chosen subset
of K values; rows show

    Row 1 — corr_A   (Pearson correlation between coupling matrices)
    Row 2 — rmse_sbar (RMSE between empirical s̄_ml and OA truncation)
    Row 3 — mean phase coherence R for KMO and OA, with variance of R(t)
            shown as error bars

All curves are plotted vs. n_terms; markers indicate per-trial means and
error bars indicate across-trial std (rows 1 & 2) or the mean temporal
variance of R(t) reported by the sweep (row 3, per the user spec).

Usage
-----
    python plot_sweep_results.py --csv sweep_K_nterms_results.csv
    python plot_sweep_results.py --csv path/to/results.csv --K 1 3 5

If --K is omitted, three evenly spaced K values are picked from the CSV.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate(df, K_values):
    """
    Returns a dict keyed by K of DataFrames indexed by n_terms with columns
        corr_A_mean, corr_A_std,
        rmse_sbar_mean, rmse_sbar_std,
        mean_R_km_mean, mean_R_oa_mean,
        var_R_km_mean,  var_R_oa_mean.

    Aggregation: across trials, treating each (K, n_terms) cell as a sample
    of size n_trials.  Failed rows (status != "ok") are dropped.
    """
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

def plot_sweep(agg, K_values, params, savepath=None):
    K_list = [K for K in K_values if K in agg]
    if not K_list:
        raise RuntimeError("No K values to plot — check --K and the CSV.")
    n_cols = len(K_list)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.0 * n_cols, 9.5),
                             sharex=True, constrained_layout=True)
    if n_cols == 1:
        axes = axes[:, None]   # keep 2D indexing

    # Pre-compute row-wide y-limits for rows 1 and 2 so columns are comparable
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

    for j, K in enumerate(K_list):
        a = agg[K]
        n_terms_axis = a.index.to_numpy()

        # ── Row 1: corr_A ────────────────────────────────────────────────
        ax = axes[0, j]
        ax.errorbar(n_terms_axis, a["corr_A_mean"], yerr=a["corr_A_std"],
                    fmt="o-", color="C0", capsize=3, lw=1.5)
        ax.set_title(fr"$K = {K:g}$")
        if j == 0:
            ax.set_ylabel(r"corr$(A^\mathrm{KMO}_\mathrm{cg}, A^\mathrm{OA})$")
        ax.set_ylim(corr_lo - corr_pad, min(1.01, corr_hi + corr_pad))
        ax.grid(True, alpha=0.3)

        # ── Row 2: rmse_sbar ─────────────────────────────────────────────
        ax = axes[1, j]
        ax.errorbar(n_terms_axis, a["rmse_sbar_mean"], yerr=a["rmse_sbar_std"],
                    fmt="s-", color="C2", capsize=3, lw=1.5)
        if j == 0:
            ax.set_ylabel(r"RMSE$(\bar s^\mathrm{KMO}_{ml}, "
                          r"\bar s^\mathrm{trunc}_{ml})$")
        ax.set_ylim(max(0.0, rmse_lo - rmse_pad), rmse_hi + rmse_pad)
        ax.grid(True, alpha=0.3)

        # ── Row 3: mean R(t) with var(R) error bars ──────────────────────
        ax = axes[2, j]
        ax.errorbar(n_terms_axis, a["mean_R_km_mean"],
                    yerr=a["var_R_km_mean"],
                    fmt="o-", color="C0", capsize=3, lw=1.5, label="KMO")
        ax.errorbar(n_terms_axis, a["mean_R_oa_mean"],
                    yerr=a["var_R_oa_mean"],
                    fmt="s--", color="C3", capsize=3, lw=1.5, label="OA")
        if j == 0:
            ax.set_ylabel(r"$\langle R(t)\rangle_t$  "
                          r"(error bars: $\mathrm{Var}_t[R]$)")
            ax.legend(loc="best", frameon=False, fontsize=9)
        ax.set_xlabel(r"$n_\mathrm{terms}$")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    # x-ticks: integer n_terms values across the whole CSV
    all_nt = sorted({nt for K in K_list for nt in agg[K].index.tolist()})
    for ax in axes[-1, :]:
        ax.set_xticks(all_nt)

    n_trials = int(agg[K_list[0]]["n_trials"].max())
    fig.suptitle(
        f"K × n_terms sweep summary  (n_trials = {n_trials}"
        + (f", {params}" if params else "")
        + ")",
        fontsize=11,
    )

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {savepath}")
    return fig


# ─── CLI ─────────────────────────────────────────────────────────────────────

def pick_K_values(df, requested):
    """Resolve --K argument; if None, pick three roughly evenly spaced K's."""
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
                   help="K values to use as the three columns "
                        "(default: three evenly spaced K's from the CSV)")
    p.add_argument("--out", default="sweep_summary.png",
                   help="Output figure path")
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

    # Pretty parameter string for the suptitle (best-effort; CSV-dependent)
    params = ""
    for col, label in [("dist", "dist"), ("M", "M"),
                       ("mu", "μ"), ("Delta", "Δ")]:
        if col in df.columns and df[col].nunique() == 1:
            params += f", {label}={df[col].iloc[0]}"

    plot_sweep(agg, K_values, params.lstrip(", "), savepath=args.out)


if __name__ == "__main__":
    main()