r"""
Adaptive Kuramoto — micro vs. single Ott–Antonsen ensemble: comparison figures
==============================================================================

Loads the sweep written by ``kmo_adaptive_single_sweep.py`` and renders:

1. A SUMMARY figure (single-column PRL style), 3 rows × 2 columns — one row per
   plasticity rule (cos, sin, |sin|). For each rule:
     col 1 — RMSE of the average phase-coherence dynamics  R_micro(t) vs R_MF(t)  vs Δ
     col 2 — RMSE of the final microscopic weights A_ij from the MF's final Ā      vs Δ
   with one coloured line per value of μ.

2. One PER-RULE figure (double-column PRL style), rows = the selected μ values,
   9 columns. For the minimum / median / maximum Δ, the micro & MF phase-coherence
   dynamics R(t) are overlaid in one panel spanning 2 columns, and the final
   microscopic coupling matrix is shown next to it (1 column): 3 Δ × 3 columns = 9.

Reads the tidy CSV (discriminated by the `quantity` column).

Run with any numpy/pandas/matplotlib env, e.g.:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_single_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "/home/rgast/data/mpmf_simulations/kmo_adaptive_single_sweep.csv"
OUT_SUMMARY = "/home/rgast/data/mpmf_simulations/kmo_adaptive_single_summary"
OUT_RULE = "/home/rgast/data/mpmf_simulations/kmo_adaptive_single_{tag}"

C_MICRO = "0.2"
C_MF = "#c1121f"
MU_CMAP = "viridis"           # colormap encoding μ in the summary lineplots
MATRIX_CMAP = "magma"         # colormap for the final coupling matrices
MU_PLOT = None                # μ values to plot per-rule (None = all in the data)

_TAG = {"cos": "cos", "sin": "sin", "|sin|": "absin"}   # filename-safe rule tags


# ════════════════════════════════════════════════════════════════════════════
#  PRL style
# ════════════════════════════════════════════════════════════════════════════
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.formatter.useoffset": False, "savefig.dpi": 300, "figure.dpi": 150,
    })


# ════════════════════════════════════════════════════════════════════════════
#  load + metrics
# ════════════════════════════════════════════════════════════════════════════
def load(csv):
    df = pd.read_csv(csv)
    rules = list(df["G_A"].dropna().unique())
    Deltas = sorted(df["Delta"].dropna().unique())
    mus = sorted(df["mu"].dropna().unique())

    traces = {}
    for q in ("R_micro", "R_mf", "Abar_micro", "Abar_mf"):
        for (rule, D, mu), g in df[df.quantity == q].groupby(["G_A", "Delta", "mu"]):
            g = g.sort_values("time")
            traces[(q, rule, float(D), float(mu))] = (g["time"].to_numpy(), g["value"].to_numpy())

    mats = {}
    for (rule, D, mu), g in df[df.quantity == "A_final"].groupby(["G_A", "Delta", "mu"]):
        nr, nc = int(g["row"].max()) + 1, int(g["col"].max()) + 1
        M = np.full((nr, nc), np.nan)
        M[g["row"].astype(int), g["col"].astype(int)] = g["value"].to_numpy()
        mats[(rule, float(D), float(mu))] = M
    return rules, Deltas, mus, traces, mats


def rmse_R_dynamics(traces, rule, D, mu):
    """RMSE over time between micro and MF phase coherence R(t)."""
    _, rm = traces[("R_micro", rule, D, mu)]
    _, rf = traces[("R_mf", rule, D, mu)]
    n = min(len(rm), len(rf))
    return float(np.sqrt(np.mean((rm[:n] - rf[:n]) ** 2)))


def rmse_final_weights(traces, mats, rule, D, mu):
    """RMSE of the final microscopic weights A_ij from the MF's final mean coupling Ā."""
    A = mats[(rule, D, mu)]
    _, af = traces[("Abar_mf", rule, D, mu)]
    return float(np.sqrt(np.nanmean((A - af[-1]) ** 2)))


# ════════════════════════════════════════════════════════════════════════════
#  figure 1 — summary lineplots (single column)
# ════════════════════════════════════════════════════════════════════════════
def make_summary(rules, Deltas, mus, traces, mats):
    cmap = plt.get_cmap(MU_CMAP)
    colors = [cmap(0.12 + 0.76 * i / max(1, len(mus) - 1)) for i in range(len(mus))]

    fig = plt.figure(figsize=(3.4, 1.25 * len(rules) + 0.4), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.05)
    gs = fig.add_gridspec(len(rules), 2)
    Dx = np.asarray(Deltas)

    for r, rule in enumerate(rules):
        axR = fig.add_subplot(gs[r, 0])
        axA = fig.add_subplot(gs[r, 1])
        for mi, mu in enumerate(mus):
            yR = [rmse_R_dynamics(traces, rule, D, mu) for D in Deltas]
            yA = [rmse_final_weights(traces, mats, rule, D, mu) for D in Deltas]
            axR.plot(Dx, yR, color=colors[mi], marker="o", ms=2.2, lw=0.9, label=f"{mu:g}")
            axA.plot(Dx, yA, color=colors[mi], marker="o", ms=2.2, lw=0.9)
        for ax in (axR, axA):
            ax.set_xscale("log")
        axR.set_ylabel(rf"$G_A={rule}$" + "\n" + r"$R(t)$ RMSE", labelpad=2)
        axA.set_ylabel(r"$A_{ij}{-}\bar A_{\rm MF}$ RMSE", labelpad=2)
        if r == 0:
            axR.set_title("coherence dynamics", fontsize=6.5, pad=3)
            axA.set_title("final weights", fontsize=6.5, pad=3)
            axR.legend(title=r"$\mu$", ncol=1, fontsize=5.5, title_fontsize=6,
                       handlelength=1.3, loc="best")
        if r == len(rules) - 1:
            axR.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
            axA.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)

    fig.savefig(OUT_SUMMARY + ".pdf")
    fig.savefig(OUT_SUMMARY + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT_SUMMARY}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
#  figure 2 — per-rule examples (double column)
# ════════════════════════════════════════════════════════════════════════════
def make_rule_figure(rule, Deltas, mus_plot, traces, mats):
    Dsel = [Deltas[0], Deltas[len(Deltas) // 2], Deltas[-1]]    # min / median / max Δ
    nmu = len(mus_plot)

    # Row height ≈ matrix-column width so the square coupling matrix fills its cell
    # and the R(t) panels (same row height) are no taller than the matrices.
    fig = plt.figure(figsize=(7.0, 0.62 * nmu + 0.5), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.015, wspace=0.02, hspace=0.04)
    # wider R(t) panels so the (square) coupling matrix fills its column tightly
    gs = fig.add_gridspec(nmu, 9, width_ratios=[1.4, 1.4, 1.0] * 3)

    for mi, mu in enumerate(mus_plot):
        for di, D in enumerate(Dsel):
            b = 3 * di
            ax_dyn = fig.add_subplot(gs[mi, b:b + 2])
            ax_mat = fig.add_subplot(gs[mi, b + 2])
            t, Rm = traces[("R_micro", rule, D, mu)]
            _, Rf = traces[("R_mf", rule, D, mu)]
            ax_dyn.plot(t, Rm, color=C_MICRO, lw=0.9, label="micro")
            ax_dyn.plot(t, Rf, color=C_MF, lw=0.9, ls="--", label="MF")
            ax_dyn.set_xlim(t[0], t[-1]); ax_dyn.set_ylim(-0.02, 1.02)
            ax_dyn.set_yticks([0, 0.5, 1.0])
            if mi == 0:
                ax_dyn.set_title(rf"$\Delta={D:g}$", fontsize=6.5, pad=2)
            if mi == nmu - 1:
                ax_dyn.set_xlabel(r"$t$", labelpad=1)
            else:
                ax_dyn.set_xticklabels([])
            if di == 0:
                ax_dyn.set_ylabel(rf"$\mu={mu:g}$" + "\n" + r"$R(t)$", labelpad=2)
            else:
                ax_dyn.set_yticklabels([])

            M = mats[(rule, D, mu)]
            im = ax_mat.imshow(M, origin="lower", aspect="equal", cmap=MATRIX_CMAP,
                               vmin=np.nanmin(M), vmax=np.nanmax(M), interpolation="nearest")
            ax_mat.set_xticks([]); ax_mat.set_yticks([])
            if mi == 0:
                ax_mat.set_title(r"$A_{ij}$", fontsize=6.0, pad=2)
            cb = fig.colorbar(im, ax=ax_mat, fraction=0.05, pad=0.02)
            cb.ax.tick_params(labelsize=4.5, pad=0.6)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_MICRO, lw=0.9, label="micro"),
               Line2D([0], [0], color=C_MF, lw=0.9, ls="--", label="mean field")]
    fig.legend(handles=handles, loc="outside upper right", ncol=2, fontsize=6)
    fig.suptitle(rf"$G_A = {rule}$", fontsize=8, x=0.01, ha="left")

    out = OUT_RULE.format(tag=_TAG.get(rule, rule))
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {out}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    rules, Deltas, mus, traces, mats = load(CSV)
    mus_plot = mus if MU_PLOT is None else [m for m in mus if m in MU_PLOT]

    set_prl_style()
    make_summary(rules, Deltas, mus, traces, mats)
    for rule in rules:
        make_rule_figure(rule, Deltas, mus_plot, traces, mats)


if __name__ == "__main__":
    main()
