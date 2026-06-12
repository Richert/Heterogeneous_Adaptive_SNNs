r"""
Adaptive Kuramoto — micro vs. single Ott–Antonsen ensemble: comparison figure
=============================================================================

Loads the sweep written by ``kmo_adaptive_single_sweep.py`` and renders a two-column
Physical-Review-Letters figure (2 rows × 9 columns). Each of the three plasticity
rules (cos, sin, |sin|) occupies a 2×3 block:

    column 1: two stacked Δ×μ heatmaps
        row 1 — RMSE of the average-coupling dynamics  Ā_micro(t) vs Ā_MF(t)
        row 2 — RMSE of the final microscopic weights A_ij from the MF's final Ā
    columns 2-3: two representative (Δ,μ) examples
        row 1 — average coupling weight Ā(t), micro vs MF
        row 2 — the final microscopic coupling matrix (frequency-sorted, block-avg)

Reads the tidy CSV (discriminated by the `quantity` column). Colormaps are top-level
parameters.

Run with any numpy/pandas/matplotlib env, e.g.:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_single_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

CSV = "/home/rgast/data/mpmf_simulations/kmo_adaptive_single_sweep.csv"
OUT = "/home/rgast/data/mpmf_simulations/kmo_adaptive_single_figure"

C_MICRO = "0.2"
C_MF = "#c1121f"
HEATMAP_CMAP = "viridis"      # colormap for the two RMSE heatmaps
MATRIX_CMAP = "magma"         # colormap for the final coupling matrices
N_EXAMPLES = 2                # representative (Δ,μ) points per rule


# ════════════════════════════════════════════════════════════════════════════
#  PRL style
# ════════════════════════════════════════════════════════════════════════════
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.5,
        "legend.fontsize": 5.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.formatter.useoffset": False,
        "savefig.dpi": 300, "figure.dpi": 150,
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
    for q in ("Abar_micro", "Abar_mf", "R_micro", "R_mf"):
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


def rmse_Abar_dynamics(traces, rule, D, mu):
    """RMSE over time between micro and MF average coupling Ā(t)."""
    _, am = traces[("Abar_micro", rule, D, mu)]
    _, af = traces[("Abar_mf", rule, D, mu)]
    n = min(len(am), len(af))
    return float(np.sqrt(np.mean((am[:n] - af[:n]) ** 2)))


def rmse_final_weights(traces, mats, rule, D, mu):
    """RMSE of the final microscopic weights A_ij from the MF's final mean coupling Ā."""
    A = mats[(rule, D, mu)]
    _, af = traces[("Abar_mf", rule, D, mu)]
    return float(np.sqrt(np.nanmean((A - af[-1]) ** 2)))


# ════════════════════════════════════════════════════════════════════════════
#  panels
# ════════════════════════════════════════════════════════════════════════════
def _heatmap(fig, ax, M, Deltas, mus, title, ylabel=True, xlabel=True):
    # M has shape (len(Deltas), len(mus)): Δ on the y-axis, μ on the x-axis
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=HEATMAP_CMAP)
    ax.set_xticks(range(len(mus)))
    ax.set_xticklabels([f"{m:g}" for m in mus])
    ax.set_yticks(range(len(Deltas)))
    ax.set_yticklabels([f"{d:g}" for d in Deltas] if ylabel else [])
    if xlabel:
        ax.set_xlabel(r"$\mu$", labelpad=1)
    if ylabel:
        ax.set_ylabel(r"$\Delta$", labelpad=1)
    ax.set_title(title, fontsize=6.0, pad=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cb.ax.tick_params(labelsize=4.5, pad=0.8)
    return im


def _abar_panel(ax, traces, rule, D, mu):
    t, am = traces[("Abar_micro", rule, D, mu)]
    _, af = traces[("Abar_mf", rule, D, mu)]
    ax.plot(t, am, color=C_MICRO, lw=0.9, label="micro")
    ax.plot(t, af, color=C_MF, lw=0.9, ls="--", label="MF")
    ax.set_xlim(t[0], t[-1])
    lo, hi = min(am.min(), af.min()), max(am.max(), af.max())
    if hi - lo < 1e-3:                              # near-constant (e.g. Ā for the sin rule)
        c = 0.5 * (lo + hi); ax.set_ylim(c - 0.5, c + 0.5)
    fmt = mticker.ScalarFormatter(useOffset=False)  # plain numbers, no offset / 1e-14 scale
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_ylabel(r"$\bar A(t)$", labelpad=1)
    ax.set_xlabel(r"$t$", labelpad=1)


def _matrix_panel(fig, ax, M):
    im = ax.imshow(M, origin="lower", aspect="equal", cmap=MATRIX_CMAP,
                   vmin=np.nanmin(M), vmax=np.nanmax(M), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(r"$j$", labelpad=0)
    ax.set_ylabel(r"$i$", labelpad=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=4.5, pad=0.8)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    rules, Deltas, mus, traces, mats = load(CSV)
    nrule = len(rules)

    set_prl_style()
    fig = plt.figure(figsize=(7.1, 3.2), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.015, wspace=0.03, hspace=0.05)
    gs = fig.add_gridspec(2, 3 * nrule)
    _stroke = [pe.withStroke(linewidth=1.0, foreground="black")]

    for r, rule in enumerate(rules):
        b = 3 * r
        # RMSE matrices over Δ (rows) × μ (cols)
        MM_dyn = np.full((len(Deltas), len(mus)), np.nan)
        MM_fin = np.full((len(Deltas), len(mus)), np.nan)
        for jj, D in enumerate(Deltas):
            for ii, mu in enumerate(mus):
                if ("Abar_micro", rule, D, mu) in traces:
                    MM_dyn[jj, ii] = rmse_Abar_dynamics(traces, rule, D, mu)
                    MM_fin[jj, ii] = rmse_final_weights(traces, mats, rule, D, mu)

        ax_h1 = fig.add_subplot(gs[0, b])
        ax_h2 = fig.add_subplot(gs[1, b])
        _heatmap(fig, ax_h1, MM_dyn, Deltas, mus, f"$G_A={rule}$\n$\\bar A(t)$ RMSE", xlabel=False)
        _heatmap(fig, ax_h2, MM_fin, Deltas, mus, r"$A_{ij}$–$\bar A_{\rm MF}$ RMSE")

        # two representative examples: best & worst Ā-dynamics RMSE
        keys = [(D, mu) for mu in mus for D in Deltas if ("Abar_micro", rule, D, mu) in traces]
        keys.sort(key=lambda k: rmse_Abar_dynamics(traces, rule, k[0], k[1]))
        idx = np.linspace(0, len(keys) - 1, N_EXAMPLES).round().astype(int)
        chosen = [keys[k] for k in idx]

        for c, (D, mu) in enumerate(chosen):
            D_idx, mu_idx = Deltas.index(D), mus.index(mu)
            for axh in (ax_h1, ax_h2):
                axh.text(mu_idx, D_idx, str(c + 1), ha="center", va="center", fontsize=5.5,
                         fontweight="bold", color="white", path_effects=_stroke,
                         bbox=dict(boxstyle="circle,pad=0.04", fc="0.1", ec="white", lw=0.6))
            ax_t = fig.add_subplot(gs[0, b + 1 + c])
            ax_m = fig.add_subplot(gs[1, b + 1 + c])
            _abar_panel(ax_t, traces, rule, D, mu)
            _matrix_panel(fig, ax_m, mats[(rule, D, mu)])
            ax_t.set_title(f"{c + 1}: $\\Delta={D:g}$, $\\mu={mu:g}$", fontsize=5.5, pad=2)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_MICRO, lw=0.9, label="micro"),
               Line2D([0], [0], color=C_MF, lw=0.9, ls="--", label="mean field")]
    fig.legend(handles=handles, loc="outside upper right", ncol=2, fontsize=6)

    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"[saved] {OUT}.pdf / .png")
    for rule in rules:
        keys = [(D, mu) for mu in mus for D in Deltas if ("Abar_micro", rule, D, mu) in traces]
        dyn = [rmse_Abar_dynamics(traces, rule, D, mu) for D, mu in keys]
        fin = [rmse_final_weights(traces, mats, rule, D, mu) for D, mu in keys]
        print(f"  {rule}: Ā-dyn RMSE [{min(dyn):.3f},{max(dyn):.3f}]  "
              f"final-weight RMSE [{min(fin):.3f},{max(fin):.3f}]")


if __name__ == "__main__":
    main()
