r"""
Weight-variance figure — coupling-weight variance vs heterogeneity (PRL "Weight Variance")
==========================================================================================

Reads the tidy CSV written by ``kmo_adaptive_single_sweep.py`` (cosine rule, γ=0.001, the
two cases μ=γ and μ=10γ) and assembles the manuscript's first figure. Two columns (one per
μ), three row-groups:

  (a,b) relative weight variance  V_A/Ā² vs Δ  — simulation markers + closed-form Eq. 37
        (imported from theory/weight_variance_analysis.py so the overlay matches exactly).
  (c,d) phase coherence R vs Δ — microscopic network (markers) vs the variance-free OA
        ensemble mean field (line): they agree for μ=γ and diverge badly for μ=10γ.
  (e–j) the final, frequency-sorted coupling matrix A at low / peak-variance / high Δ,
        showing the uniform → frequency-assortative → uniform-baseline progression.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_variance_figure.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "theory")))
from weight_variance_analysis import branches, S_order   # noqa: E402  (reuse Eq. 37 closed form)

CSV = "/home/rgast/data/mpmf_simulations/kmo_adaptive_single_sweep.csv"
OUT = "/home/rgast/data/mpmf_simulations/kmo_adaptive_variance"
K, GAMMA = 1.0, 0.001
TAIL_FRAC = 0.8                                           # steady state = last 20% of each trace
# μ cases and adaptation rules are auto-detected from the CSV (mus_in / rules_in).

C_MIC, C_MF, C_VAR, C_REF = "#e63946", "#1f77b4", "#c1121f", "0.55"
MAT_CMAP = "magma"


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 5.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def _panel_label(ax, letter, dx=-24, dy=4):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


# ════════════════════════════════════════════════════════════════════════════
#  data access
# ════════════════════════════════════════════════════════════════════════════
def load(csv=CSV):
    df = pd.read_csv(csv)
    tail = TAIL_FRAC * df["time"].dropna().max()
    return df, tail


def rules_in(df):
    """Adaptation rules present in the CSV, in canonical order."""
    present = set(df["G_A"].dropna().unique())
    return [r for r in ("cos", "sin", "|sin|") if r in present]


def mus_in(df):
    return sorted(df["mu"].dropna().unique())


def steady(df, tail, rule, mu, q):
    """Steady-state (tail-averaged) value of quantity `q` per Δ for rule `rule`, rate `mu`."""
    g = df[(df.quantity == q) & (df.G_A == rule) & (df.mu == mu) & (df.time >= tail)]
    s = g.groupby("Delta")["value"].mean().sort_index()
    return s.index.values, s.values


def final_matrix(df, rule, mu, D):
    af = df[(df.quantity == "A_final") & (df.G_A == rule) & (df.mu == mu) & (np.isclose(df.Delta, D))]
    n = int(af["row"].max()) + 1
    M = np.full((n, n), np.nan)
    M[af["row"].astype(int).values, af["col"].astype(int).values] = af["value"].values
    return M


def nearest(vals, target):
    vals = np.asarray(vals)
    return float(vals[np.argmin(np.abs(vals - target))])


_RULE_TEX = {"cos": r"\cos", "sin": r"\sin", "|sin|": r"|\!\sin|"}
_RULE_TAG = {"cos": "", "sin": "_sin", "|sin|": "_absin"}


def _mu_label(mu):
    r = mu / GAMMA
    return r"$\mu=\gamma$" if abs(r - 1.0) < 1e-6 else rf"$\mu={r:g}\,\gamma$"


# ════════════════════════════════════════════════════════════════════════════
#  figure
# ════════════════════════════════════════════════════════════════════════════
def make_figure(df, tail, rule, mus):
    """One figure per adaptation rule; columns = the μ cases. The cos-specific closed-form
    overlays (Eq. 37 via branches()/S_order) are drawn only for rule == 'cos'."""
    cos = rule == "cos"
    ncol = len(mus)
    letters = list("abcdefghijklmnop")
    fig = plt.figure(figsize=(3.5 * ncol, 5.6), constrained_layout=True)
    gs = GridSpec(3, 3 * ncol, figure=fig, height_ratios=[1.05, 1.05, 0.95])

    for c, mu in enumerate(mus):
        cols = slice(3 * c, 3 * c + 3)
        lab = letters[3 * c:3 * c + 3]                     # V_A, R, matrices for this column
        dsn = K * (GAMMA + mu) ** 2 / (8 * mu * GAMMA)
        d_end = dsn if mu > GAMMA else min(dsn, K / 2.0)   # cos branch endpoint (fold / transcritical)

        # ---- relative weight variance V_A/Ā² vs Δ -------------------------------------
        axv = fig.add_subplot(gs[0, cols])
        dD, VA = steady(df, tail, rule, mu, "VA_micro")
        _, Ab = steady(df, tail, rule, mu, "Abar_micro")
        rel = VA / Ab ** 2
        axv.plot(dD, rel, "o", ms=3.2, mfc="none", mec=C_MIC, mew=0.9,
                 label="microscopic", zorder=5)
        if cos:
            # MF prediction: Eq. 37 driven by the MF ensemble's OWN steady R and Ā (Eqs. 7-9),
            # so it follows the same attractor as the R panel (sync below / async above).
            dRf, Rf = steady(df, tail, rule, mu, "R_mf")
            _, Abf = steady(df, tail, rule, mu, "Abar_mf")
            S = S_order(Rf ** 2, Abf, dRf, K)
            axv.plot(dRf, mu ** 2 * (S ** 2 - Rf ** 4) / (2 * GAMMA ** 2) / Abf ** 2, "-",
                     color=C_MF, lw=1.1, label="mean field (Eq. 37)", zorder=4)
            # analytic synchronized branch: persists to the fold Δ_SN (the branch the
            # finite-IC mean field drops off of in the bistable range).
            dfit = np.linspace(0.01, 0.999 * d_end, 800)
            br = branches(dfit, mu, K, GAMMA)["sync"]
            axv.plot(dfit, br["VA"] / br["A"] ** 2, ":", color=C_REF, lw=0.9,
                     label="sync. branch", zorder=3)
            axv.axvline(d_end, color=C_REF, lw=0.6, ls=":", zorder=0)
        axv.set_ylim(bottom=0)
        axv.set_xlim(0, dD.max() * 1.02)
        axv.set_ylabel(r"rel. weight variance $V_A/\bar A^2$", labelpad=2)
        axv.set_title(rf"{_mu_label(mu)},   $G_A={_RULE_TEX[rule]}$", fontsize=7.2, pad=3)
        _panel_label(axv, lab[0])
        if c == 0:
            axv.legend(loc="upper left", fontsize=5.4, handlelength=1.5)

        # ---- phase coherence R: micro vs mean field ----------------------------------
        axr = fig.add_subplot(gs[1, cols])
        dR, Rm = steady(df, tail, rule, mu, "R_micro")
        _, Rmf = steady(df, tail, rule, mu, "R_mf")
        axr.plot(dR, Rmf, "-", color=C_MF, lw=1.1, label="mean field", zorder=3)
        axr.plot(dR, Rm, "o", ms=3.2, mfc="none", mec=C_MIC, mew=0.9,
                 label="microscopic", zorder=5)
        if cos:
            axr.axvline(d_end, color=C_REF, lw=0.6, ls=":", zorder=0)
        axr.set_ylim(-0.03, 1.05)
        axr.set_xlim(0, dR.max() * 1.02)
        axr.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
        axr.set_ylabel(r"phase coherence $R$", labelpad=2)
        _panel_label(axr, lab[1])
        if c == 0:
            axr.legend(loc="lower left", fontsize=5.4, handlelength=1.5)

        # ---- frequency-sorted coupling matrices at low / peak-variance / high Δ -------
        Ds = sorted(df[(df.quantity == "A_final") & (df.G_A == rule) & (df.mu == mu)].Delta.unique())
        d_peak = dD[int(np.nanargmax(rel))] if np.isfinite(rel).any() else Ds[len(Ds) // 2]
        picks = [Ds[0], nearest(Ds, d_peak), Ds[-1]]
        mats = [final_matrix(df, rule, mu, D) for D in picks]
        vmax = max(np.nanmax(M) for M in mats)
        axes_m = []
        for k, (D, M) in enumerate(zip(picks, mats)):
            axm = fig.add_subplot(gs[2, 3 * c + k])
            im = axm.imshow(M, cmap=MAT_CMAP, vmin=0, vmax=vmax, aspect="equal", origin="lower")
            axm.set_title(rf"$\Delta={D:.2f}$", fontsize=6.2, pad=1.5)
            axm.set_xticks([]); axm.set_yticks([])
            axes_m.append(axm)
            if k == 0:
                axm.set_ylabel(r"$\omega_j$ rank", fontsize=6, labelpad=1)
                _panel_label(axm, lab[2], dx=-14)
            if k == 1:
                axm.set_xlabel(r"$\omega_i$ rank", fontsize=6, labelpad=1)
        cb = fig.colorbar(im, ax=axes_m, location="right", fraction=0.10, pad=0.03, shrink=0.9)
        cb.set_label(r"$A_{ij}$", fontsize=6, labelpad=1)
        cb.ax.tick_params(labelsize=5.2)

    out = OUT + _RULE_TAG[rule]
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {out}.pdf / .png   (rule={rule})")


def main():
    set_prl_style()
    df, tail = load()
    mus = mus_in(df)
    for rule in rules_in(df):
        make_figure(df, tail, rule, mus)


if __name__ == "__main__":
    main()
