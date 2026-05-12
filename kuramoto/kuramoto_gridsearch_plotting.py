"""
Visualize KM vs OA Sweep Results
=================================
Reads in all sweep result files from a directory and produces 2D heatmaps of
the agreement metrics (RMSE_R and corr_A) in (μ, Δ₀) parameter space, with
one subplot column per value of d.

Expected input: parquet/CSV files written by sweep_kmo_oa.py, with columns
including at minimum:
    d, Delta0, mu, rmse_R, corr_A, trial

Trials are averaged for each (d, Δ₀, μ) cell.

Usage
-----
    python plot_sweep_heatmaps.py --in_dir sweep_results
    python plot_sweep_heatmaps.py --in_dir sweep_results --csv
    python plot_sweep_heatmaps.py --in_dir sweep_results --out heatmaps.png

The script EXPECTS exactly 3 distinct values of d. If your sweep covers more
or fewer, it will warn and use the first 3 (or whatever is present).
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 13.0


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_sweep_results(in_dir, use_csv=False):
    """Load and concatenate all sweep result files under in_dir."""
    in_dir = Path(in_dir)
    pattern = "*.csv" if use_csv else "*.parquet"
    files = sorted(glob.glob(str(in_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {in_dir}")

    reader = pd.read_csv if use_csv else pd.read_parquet
    dfs = [reader(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} files")
    return df


def aggregate(df):
    """Average metrics across trials for each (d, Δ₀, μ) cell."""
    # Only consider successful runs if a 'status' column exists
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    agg_cols = {
        "rmse_R": "mean",
        "corr_A": "mean",
    }
    # Include optional extras if present
    for extra in ("final_dR", "frob_A_blk", "mean_R_km", "mean_R_oa"):
        if extra in df.columns:
            agg_cols[extra] = "mean"

    grouped = df.groupby(["d", "Delta0", "mu"], as_index=False).agg(agg_cols)
    return grouped


def pivot_metric(df_d, metric):
    """
    Pivot one slice (fixed d) into a 2D table with Δ₀ as rows, μ as columns.
    Returns (table, mu_values, Delta_values).
    """
    table = df_d.pivot(index="Delta0", columns="mu", values=metric)
    # Ensure sorted axes
    table = table.sort_index(axis=0).sort_index(axis=1)
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmaps(df_agg, d_values, out_path=None):
    """
    Two rows × 3 columns: RMSE_R (top) and corr_A (bottom), one column per d.

    Each column shares a colour scale within its metric (so RMSE rows are
    comparable across d, and corr_A rows are comparable across d).
    """
    n_d = len(d_values)
    fig, axes = plt.subplots(2, n_d, figsize=(5.5 * n_d, 9), squeeze=False)

    # Pre-compute all pivot tables to determine shared colour ranges
    rmse_tables = []
    corr_tables = []
    for d in d_values:
        slc = df_agg[df_agg["d"] == d]
        rmse_tables.append(pivot_metric(slc, "rmse_R"))
        corr_tables.append(pivot_metric(slc, "corr_A"))

    # Shared colour scales
    rmse_vmin = min(t.min().min() for t in rmse_tables)
    rmse_vmax = max(t.max().max() for t in rmse_tables)
    corr_vmin = min(t.min().min() for t in corr_tables)
    corr_vmax = max(t.max().max() for t in corr_tables)
    # Pearson correlation is in [-1, 1]; clip vmin if all positive
    corr_vmax = min(1.0, corr_vmax)
    corr_vmin = max(-1.0, corr_vmin)

    for col, (d, rmse_tbl, corr_tbl) in enumerate(
            zip(d_values, rmse_tables, corr_tables)):

        # ── RMSE_R heatmap (top row) ─────────────────────────────────────────
        ax = axes[0, col]
        im = _draw_heatmap(ax, rmse_tbl,
                           vmin=rmse_vmin, vmax=rmse_vmax,
                           cmap="viridis")
        ax.set_title(f"RMSE of $R(t)$  |  $d={d}$", fontweight="bold")
        if col == 0:
            ax.set_ylabel(r"$\Delta_0$ (HWHM)")
        if col == n_d - 1:
            plt.colorbar(im, ax=ax, label=r"RMSE$_R$", shrink=0.85)
        else:
            plt.colorbar(im, ax=ax, shrink=0.85)

        # ── corr_A heatmap (bottom row) ──────────────────────────────────────
        ax = axes[1, col]
        im = _draw_heatmap(ax, corr_tbl,
                           vmin=corr_vmin, vmax=corr_vmax,
                           cmap="RdBu_r")
        ax.set_title(f"Pearson corr$(A^{{KM}}, A^{{OA}})$  |  $d={d}$",
                     fontweight="bold")
        ax.set_xlabel(r"$\mu$ (learning rate)")
        if col == 0:
            ax.set_ylabel(r"$\Delta_0$ (HWHM)")
        if col == n_d - 1:
            plt.colorbar(im, ax=ax, label=r"corr($A$)", shrink=0.85)
        else:
            plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("KM vs OA agreement in $(\\mu, \\Delta_0)$ parameter space",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    return fig


def _draw_heatmap(ax, table, vmin, vmax, cmap):
    """Render one pivot-table as a heatmap with proper tick labels."""
    mu_vals = table.columns.values
    Delta_vals = table.index.values
    data = table.values  # rows = Δ, cols = μ

    im = ax.imshow(data, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    # Tick labels — show all values
    ax.set_xticks(np.arange(len(mu_vals)))
    ax.set_yticks(np.arange(len(Delta_vals)))
    ax.set_xticklabels([_fmt(v) for v in mu_vals], rotation=0)
    ax.set_yticklabels([_fmt(v) for v in Delta_vals])

    # Annotate cell values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if _is_dark(im, v) else "black",
                        fontsize=9)

    return im


def _fmt(v):
    """Compact numeric tick label."""
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.3g}"


def _is_dark(im, v):
    """Heuristic: choose white text on dark cells, black on light."""
    norm = im.norm
    cmap = im.cmap
    rgba = cmap(norm(v))
    # Perceived brightness
    return (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]) < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="sweep_results",
                        help="directory containing sweep_*.parquet/.csv files")
    parser.add_argument("--csv", action="store_true",
                        help="read CSV instead of parquet")
    parser.add_argument("--out", default="km_vs_oa_heatmaps.png",
                        help="output figure path")
    args = parser.parse_args()

    df = load_sweep_results(args.in_dir, use_csv=args.csv)
    df_agg = aggregate(df)

    d_values = sorted(df_agg["d"].unique())
    if len(d_values) != 3:
        print(f"WARNING: expected 3 values of d, found {len(d_values)}: "
              f"{d_values}")
        d_values = d_values[:3] if len(d_values) > 3 else d_values

    if not d_values:
        raise SystemExit("No data after aggregation.")

    # Diagnostic: how many cells per d?
    for d in d_values:
        n_cells = (df_agg["d"] == d).sum()
        n_mu = df_agg[df_agg["d"] == d]["mu"].nunique()
        n_Delta = df_agg[df_agg["d"] == d]["Delta0"].nunique()
        print(f"  d={d}:  {n_cells} cells  ({n_mu} μ × {n_Delta} Δ₀)")

    plot_heatmaps(df_agg, d_values, out_path=args.out)
    plt.show()


if __name__ == "__main__":
    main()

