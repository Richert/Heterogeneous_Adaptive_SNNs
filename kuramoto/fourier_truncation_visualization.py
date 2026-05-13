"""
Visualize K-Sweep Results: KMO with |sin| Plasticity vs. OA Mean-Field
======================================================================
Reads in the CSV output of sweep_K_M.py (single-row schema with columns
including trial, K, M, mu, n_terms, rmse_R, corr_A, mean_r_km, mean_r_oa)
and produces a one-row, three-column figure:

  (a) Microscopic plasticity kernel |sin(φ)| vs. the truncated OA Fourier
      expansion, evaluated as a function of the phase difference φ at three
      representative squared phase coherences  R² = (r_m r_l)²  corresponding
      to the minimum, median, and maximum values of K in the sweep (using
      each K's average mean_r_km × mean_r_oa proxy).
  (b) Pearson correlation between the coupling matrices  corr_A  vs. K,
      with error bars (mean ± std across trials).
  (c) RMSE between the Kuramoto order parameter trajectories  rmse_R  vs. K,
      with error bars (mean ± std across trials).

Usage
-----
    python plot_K_sweep.py [--csv sweep_K_results.csv] [--out figure.pdf]
                           [--n_terms 5]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# PRX-style figure setup (matches the manuscript figures)
# ═══════════════════════════════════════════════════════════════════════════════

def set_prx_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["STIXGeneral", "Times New Roman", "Times"],
        "mathtext.fontset":   "stix",
        "font.size":          10,
        "axes.titlesize":     10,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    8,
        "axes.linewidth":     0.7,
        "lines.linewidth":    1.4,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "savefig.dpi":        300,
        "figure.dpi":         150,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def panel_label(ax, label, *, x=-0.20, y=1.02, fontsize=11):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            ha="left", va="bottom")


# ═══════════════════════════════════════════════════════════════════════════════
# Plasticity-rule plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def truncated_abs_sin(phi, R2, n_terms):
    """
    Truncated OA expansion of the |sin| plasticity rule:

        f_OA(φ, R²) = 2/π - (4/π) Σ_{n=1}^{n_terms} R²ⁿ cos(2n φ) / (4n² - 1)

    where R² = (r_m r_l)² serves as the effective ensemble-product coherence.
    Returns the same shape as the input phi.
    """
    out = np.full_like(phi, 2.0 / np.pi, dtype=float)
    for n in range(1, n_terms + 1):
        out += (-4.0 / (np.pi * (4 * n * n - 1))) * (R2 ** n) * np.cos(2 * n * phi)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate(df):
    """Average metrics across trials for each K (drop failed runs)."""
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    grouped = df.groupby("K").agg(
        rmse_R_mean    = ("rmse_R",    "mean"),
        rmse_R_std     = ("rmse_R",    "std"),
        corr_A_mean    = ("corr_A",    "mean"),
        corr_A_std     = ("corr_A",    "std"),
        mean_r_km_mean = ("mean_r_km", "mean"),
        mean_r_oa_mean = ("mean_r_oa", "mean"),
        n_trials       = ("rmse_R",    "count"),
    ).reset_index()

    return grouped


# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════

# Colours: shared blue for KMO, three discrete reds for OA at min/median/max K
C_KM   = "#1f4e79"
C_OA_3 = ["#f4a582", "#d6604d", "#9f1d20"]   # light → dark


def plot_figure(df_agg, n_terms, save_path):
    set_prx_style()

    fig, axes = plt.subplots(
        1, 3, figsize=(11.0, 3.4),
        gridspec_kw=dict(wspace=0.42, left=0.07, right=0.985,
                         top=0.91, bottom=0.18),
    )

    # ── (a) Plasticity kernel: |sin| vs truncated OA ──────────────────────
    ax = axes[0]
    phi = np.linspace(-np.pi, np.pi, 600)

    # Microscopic reference (independent of R)
    ax.plot(phi, np.abs(np.sin(phi)),
            color=C_KM, lw=1.6, label=r"$|\sin\varphi|$ (KMO)")

    # Pick 3 representative K values: min, median, max
    K_vals = df_agg["K"].values
    if len(K_vals) < 3:
        raise ValueError("Need at least 3 K values in the sweep to pick "
                         "min / median / max.")
    median_K = np.median(K_vals)
    K_pick   = [K_vals.min(),
                K_vals[np.argmin(np.abs(K_vals - median_K))],
                K_vals.max()]

    # For each picked K, compute the effective R² = (mean_r_km · mean_r_oa)
    # as a representative scalar.  Using the geometric KMO×OA proxy keeps the
    # comparison fair while still depending on K.
    for K, col in zip(K_pick, C_OA_3):
        row    = df_agg[df_agg["K"] == K].iloc[0]
        R_eff  = np.sqrt(row["mean_r_km_mean"] * row["mean_r_oa_mean"])
        R2_eff = R_eff ** 2
        y_oa   = truncated_abs_sin(phi, R2_eff, n_terms)
        ax.plot(phi, y_oa, color=col, lw=1.3, ls="--",
                label=fr"OA, $K={K:g}$  ($R^2 \approx {R2_eff:.2f}$)")

    ax.set_xlabel(r"phase difference  $\varphi$")
    ax.set_ylabel(r"$|\sin\varphi|$  and  $f_{\rm OA}(\varphi, R^2)$")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.legend(loc="upper center", frameon=False, handlelength=2.2,
              borderaxespad=0.3)
    panel_label(ax, "(a)", x=-0.17, y=1.02)

    # ── (b) corr_A vs K ───────────────────────────────────────────────────
    ax = axes[1]
    ax.errorbar(df_agg["K"], df_agg["corr_A_mean"],
                yerr=df_agg["corr_A_std"],
                color=C_KM, marker="o", ms=4.5, lw=1.3,
                capsize=3, capthick=0.8, elinewidth=0.8,
                label=r"corr$(A^{\rm KMO}, A^{\rm OA})$")
    ax.set_xlabel(r"coupling strength  $K$")
    ax.set_ylabel(r"corr$(A^{\rm KMO}, A^{\rm OA})$")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0.0, color="grey", lw=0.5, ls=":")
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")
    panel_label(ax, "(b)", x=-0.22, y=1.02)

    # ── (c) rmse_R vs K ───────────────────────────────────────────────────
    ax = axes[2]
    ax.errorbar(df_agg["K"], df_agg["rmse_R_mean"],
                yerr=df_agg["rmse_R_std"],
                color=C_KM, marker="o", ms=4.5, lw=1.3,
                capsize=3, capthick=0.8, elinewidth=0.8,
                label=r"RMSE$(R_{\rm KMO}, R_{\rm OA})$")
    ax.set_xlabel(r"coupling strength  $K$")
    ax.set_ylabel(r"RMSE$_R$")
    ax.set_ylim(bottom=0)
    panel_label(ax, "(c)", x=-0.22, y=1.02)

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved → {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="sweep_K_results.csv",
                        help="CSV file from sweep_K_M.py")
    parser.add_argument("--out", default="figure_K_sweep.pdf",
                        help="Output figure path")
    parser.add_argument("--n_terms", type=int, default=None,
                        help="Override Fourier truncation for panel (a). "
                             "By default reads from the CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    n_K = df["K"].nunique()
    n_trials = df["trial"].nunique() if "trial" in df.columns else "?"
    print(f"  K values: {n_K},  trials: {n_trials}")

    # Pull n_terms from CSV unless overridden
    if args.n_terms is None:
        if "n_terms" not in df.columns:
            raise ValueError("CSV has no 'n_terms' column; pass --n_terms.")
        n_terms_vals = df["n_terms"].unique()
        if len(n_terms_vals) != 1:
            raise ValueError(f"CSV contains multiple n_terms values: "
                             f"{n_terms_vals}.  Pass --n_terms to disambiguate.")
        n_terms = int(n_terms_vals[0])
    else:
        n_terms = args.n_terms
    print(f"  Fourier truncation: n_terms = {n_terms}")

    df_agg = aggregate(df)
    print("\nAggregated table:")
    print(df_agg.to_string(index=False))

    plot_figure(df_agg, n_terms=n_terms, save_path=args.out)


if __name__ == "__main__":
    main()