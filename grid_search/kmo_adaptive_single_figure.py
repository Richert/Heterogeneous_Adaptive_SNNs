r"""
Adaptive Kuramoto — micro vs. single Ott–Antonsen ensemble: comparison figure
=============================================================================

Loads the sweep written by ``kmo_adaptive_single_sweep.py`` and renders a
Physical-Review-Letters figure (one block per adaptation rule, cos on top, sin
below). For each rule:

  * column 1: a heatmap of the micro↔mean-field coherence mismatch
    RMSE_t[ R_micro(t) − R_MF(t) ] over the Δ×μ sweep grid, with three
    representative (Δ,μ) points marked (best / median / worst mismatch);
  * columns 2–4: those three examples, each shown as
      row 1: phase coherence R(t),   micro vs MF
      row 2: mean coupling Ā(t),     micro vs MF
      row 3: final microscopic coupling matrix A (frequency-sorted, block-averaged)

Reads the tidy CSV (discriminated by the `quantity` column). Colormaps are
top-level parameters.

Run with any numpy/pandas/matplotlib env, e.g.:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_single_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

CSV = "/home/rgast/data/mpmf_simulations/kmo_adaptive_single_sweep.csv"
OUT = "/home/rgast/data/mpmf_simulations/kmo_adaptive_single_figure"

C_MICRO = "0.2"
C_MF = "#c1121f"
MISMATCH_CMAP = "viridis"     # colormap for the R(t) mismatch heatmap
MATRIX_CMAP = "magma"         # colormap for the final coupling matrices
N_EXAMPLES = 3                # representative (Δ,μ) points per rule


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
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


# ════════════════════════════════════════════════════════════════════════════
#  load
# ════════════════════════════════════════════════════════════════════════════
def load(csv):
    df = pd.read_csv(csv)
    rules = list(df["G_A"].dropna().unique())
    Deltas = sorted(df["Delta"].dropna().unique())
    mus = sorted(df["mu"].dropna().unique())

    traces = {}                                          # (q, rule, Δ, μ) -> (t, value)
    for q in ("R_micro", "R_mf", "Abar_micro", "Abar_mf"):
        for (rule, D, mu), g in df[df.quantity == q].groupby(["G_A", "Delta", "mu"]):
            g = g.sort_values("time")
            traces[(q, rule, float(D), float(mu))] = (g["time"].to_numpy(), g["value"].to_numpy())

    mats = {}                                            # (rule, Δ, μ) -> A_final matrix
    for (rule, D, mu), g in df[df.quantity == "A_final"].groupby(["G_A", "Delta", "mu"]):
        nr, nc = int(g["row"].max()) + 1, int(g["col"].max()) + 1
        M = np.full((nr, nc), np.nan)
        M[g["row"].astype(int), g["col"].astype(int)] = g["value"].to_numpy()
        mats[(rule, float(D), float(mu))] = M

    return df, rules, Deltas, mus, traces, mats


def mismatch(traces, rule, D, mu):
    """Time-domain RMSE between micro and MF coherence R(t)."""
    _, Rm = traces[("R_micro", rule, D, mu)]
    _, Rf = traces[("R_mf", rule, D, mu)]
    n = min(len(Rm), len(Rf))
    return float(np.sqrt(np.mean((Rm[:n] - Rf[:n]) ** 2)))


# ════════════════════════════════════════════════════════════════════════════
#  panels
# ════════════════════════════════════════════════════════════════════════════
def _trace_panel(ax, traces, q_mic, q_mf, rule, D, mu, ylabel, ylim=None):
    t, vm = traces[(q_mic, rule, D, mu)]
    _, vf = traces[(q_mf, rule, D, mu)]
    ax.plot(t, vm, color=C_MICRO, lw=0.9, label="micro")
    ax.plot(t, vf, color=C_MF, lw=0.9, ls="--", label="MF")
    ax.set_xlim(t[0], t[-1])
    if ylim:
        ax.set_ylim(*ylim)
    else:
        lo = min(vm.min(), vf.min())
        hi = max(vm.max(), vf.max())
        if hi - lo < 1e-3:                  # near-constant (e.g. Ā for the sin rule)
            c = 0.5 * (lo + hi)
            ax.set_ylim(c - 0.5, c + 0.5)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_ylabel(ylabel, labelpad=2)


def _matrix_panel(fig, ax, M):
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    im = ax.imshow(M, origin="lower", aspect="equal", cmap=MATRIX_CMAP, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("osc. $j$ (sorted by $\\omega$)", labelpad=1)
    ax.set_ylabel("osc. $i$", labelpad=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    df, rules, Deltas, mus, traces, mats = load(CSV)
    nrule = len(rules)

    set_prl_style()
    ncol = 1 + N_EXAMPLES
    fig = plt.figure(figsize=(7.0, 3.0 * nrule), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    gs = fig.add_gridspec(3 * nrule, ncol)
    _stroke = [pe.withStroke(linewidth=1.0, foreground="black")]

    for r, rule in enumerate(rules):
        r0 = 3 * r
        # ── mismatch heatmap over Δ×μ (μ rows, Δ cols) ──────────────────────
        MM = np.full((len(mus), len(Deltas)), np.nan)
        for i, mu in enumerate(mus):
            for j, D in enumerate(Deltas):
                if ("R_micro", rule, D, mu) in traces:
                    MM[i, j] = mismatch(traces, rule, D, mu)
        axh = fig.add_subplot(gs[r0:r0 + 3, 0])
        im = axh.imshow(MM, origin="lower", aspect="auto", cmap=MISMATCH_CMAP)
        axh.set_xticks(range(len(Deltas)))
        axh.set_xticklabels([f"{d:g}" for d in Deltas])
        axh.set_yticks(range(len(mus)))
        axh.set_yticklabels([f"{m:g}" for m in mus])
        axh.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
        axh.set_ylabel(r"adaptation $\mu$", labelpad=2)
        axh.set_title(f"({'ab'[r]}) $G_A=\\,${rule}: R-mismatch", loc="left", fontsize=7, pad=3)
        cb = fig.colorbar(im, ax=axh, fraction=0.045, pad=0.012)
        cb.ax.tick_params(labelsize=5.5, pad=1.0)

        # representative points: best / median / worst mismatch
        keys = [(D, mu) for mu in mus for D in Deltas if ("R_micro", rule, D, mu) in traces]
        keys.sort(key=lambda k: mismatch(traces, rule, k[0], k[1]))
        idx = np.linspace(0, len(keys) - 1, N_EXAMPLES).round().astype(int)
        chosen = [keys[k] for k in idx]
        tags = ["best", "median", "worst"][:N_EXAMPLES]

        # ── examples (columns 2..) ──────────────────────────────────────────
        for c, ((D, mu), tag) in enumerate(zip(chosen, tags), start=1):
            j, i = Deltas.index(D), mus.index(mu)
            axh.text(j, i, str(c), ha="center", va="center", fontsize=6.5,
                     fontweight="bold", color="white", path_effects=_stroke,
                     bbox=dict(boxstyle="circle,pad=0.05", fc="0.1", ec="white", lw=0.7))
            ax_R = fig.add_subplot(gs[r0, c])
            ax_A = fig.add_subplot(gs[r0 + 1, c])
            ax_M = fig.add_subplot(gs[r0 + 2, c])
            _trace_panel(ax_R, traces, "R_micro", "R_mf", rule, D, mu, r"$R(t)$", ylim=(-0.02, 1.02))
            _trace_panel(ax_A, traces, "Abar_micro", "Abar_mf", rule, D, mu, r"$\bar A(t)$")
            _matrix_panel(fig, ax_M, mats[(rule, D, mu)])
            ax_R.set_title(f"{c}) {tag}: $\\Delta={D:g}$, $\\mu={mu:g}$\n"
                           f"RMSE$={mismatch(traces, rule, D, mu):.3f}$", fontsize=6.0, pad=2)
            ax_A.set_xlabel(r"time $t$", labelpad=1)

    # one shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_MICRO, lw=0.9, label="micro"),
               Line2D([0], [0], color=C_MF, lw=0.9, ls="--", label="mean field")]
    fig.legend(handles=handles, loc="outside upper right", ncol=2, fontsize=6)

    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"[saved] {OUT}.pdf / .png")
    for rule in rules:
        keys = [(D, mu) for mu in mus for D in Deltas if ("R_micro", rule, D, mu) in traces]
        mm = [mismatch(traces, rule, D, mu) for D, mu in keys]
        print(f"  {rule}: R-mismatch range [{min(mm):.3f}, {max(mm):.3f}]")


if __name__ == "__main__":
    main()
