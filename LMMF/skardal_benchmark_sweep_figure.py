r"""
Skardal-model benchmark — extended Fig. 2 (one figure per coupling regime)
==========================================================================

Loads the sweep produced by skardal_benchmark_sweep.py + the LMMF fits produced by
skardal_benchmark_lmmf.py, and assembles a single-column PRL figure for ONE coupling
regime (subcritical / critical / supercritical). Run once, it writes all three figures.

Layout (single column, 5 rows × 2 columns):
  * ROW 1 — two line plots vs the exponent n (trial-averaged):
      (a) number of mean-field equations,
      (b) RMSE of the AMPLITUDE SPECTRUM |FFT{R(t)}| between mean field and microscopic
          network (frequency-content mismatch; phase-insensitive — raw-trace RMSE would be
          Parseval-equivalent to the complex-FFT RMSE, so the magnitude spectrum is used).
    Two COLOURS distinguish the two reductions (Skardal vs. LMMF); LINE STYLE encodes N.
    (Skardal's dimension is exactly n, independent of N, so it is drawn once as a reference.)
  * ROWS 2–5 (shorter than row 1) — four representative examples, one per row, in the identical
    style as the original Fig. 2: LEFT the frequency density (histogram + fitted Lorentzian
    mixture + analytic g_n), RIGHT the R(t) comparison micro / Skardal / best fit.

Run in the ``pycobi`` conda env:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python skardal_benchmark_sweep_figure.py
    # optional: restrict to one regime -> python skardal_benchmark_sweep_figure.py critical
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
from prl_style import set_prl_style as _set_prl_style
from prl_style import panel_label
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skardal_benchmark_lmmf as LMMF                        # load_mixture + data dir

DATA_DIR = LMMF.CONFIG["in_dir"]
STEM = LMMF.CONFIG["in_stem"]
OUT_STEM = os.path.join(DATA_DIR, "skardal_sweep_figure")

REGIMES = ["subcritical", "critical", "supercritical"]

# four representative examples = 2 values of n × 2 values of N (rows ordered n outer, N inner)
EXAMPLE_n = [2, 128]
EXAMPLE_N = [200, 5000]

# colours (consistent with the original Fig. 2): micro grey, Skardal blue, LMMF/best-fit red
C_MICRO = "0.25"
C_SKARDAL = "#1f77b4"
C_ENS = "#c1121f"
C_COMP = "#2e6f95"
# line style per network size N
N_STYLE = {200: ":", 1000: "--", 5000: "-"}


set_prl_style = functools.partial(_set_prl_style, "prl_wide",
                                 **{})


_panel_label = functools.partial(panel_label, dx=-16, dy=4)


def _sweep_npz(n, regime, N):
    return np.load(os.path.join(DATA_DIR, f"{STEM}_n{n}_{regime}_N{N}.npz"), allow_pickle=False)


def _lmmf_npz(n, regime, N):
    return np.load(os.path.join(DATA_DIR, f"{STEM}_n{n}_{regime}_N{N}_lmmf.npz"), allow_pickle=False)


def _amp_spectrum(x):
    """One-sided amplitude spectrum |FFT{x}| / len(x) (DC bin = mean(x))."""
    return np.abs(np.fft.rfft(x)) / len(x)


def _fft_rmse(a, b):
    """RMSE between the amplitude spectra of two R(t) traces (phase-insensitive)."""
    L = min(len(a), len(b))
    return float(np.sqrt(np.mean((_amp_spectrum(a[:L]) - _amp_spectrum(b[:L])) ** 2)))


def load_metrics(regime):
    """Trial-averaged spectral RMSE (mean field vs micro) and M_lmmf per (N, n) for one regime.

    Recomputes the metric from the stored R(t) traces so the sweep/LMMF scripts stay untouched:
    rmse_sk = RMSE(|FFT R_skardal|, |FFT R_micro|); rmse_lmmf likewise for the LMMF ensemble."""
    sw = pd.read_csv(os.path.join(DATA_DIR, f"{STEM}_summary.csv"))
    combos = (sw[sw.regime == regime][["N", "n"]].drop_duplicates().itertuples(index=False))
    recs = []
    for N, n in combos:
        d, dl = _sweep_npz(int(n), regime, int(N)), _lmmf_npz(int(n), regime, int(N))
        for tr in range(int(d["n_trials"])):
            recs.append(dict(N=int(N), n=int(n),
                             rmse_sk=_fft_rmse(d["R_skardal"][tr], d["R_micro"][tr]),
                             rmse_lmmf=_fft_rmse(dl["R_ensemble"][tr], d["R_micro"][tr]),
                             M=int(dl["M"][tr])))
    return pd.DataFrame(recs).groupby(["N", "n"]).mean().reset_index()


# ════════════════════════════════════════════════════════════════════════════
#  figure
# ════════════════════════════════════════════════════════════════════════════
def _example_panels(fig, axL, axR, d, dl, show_dens_legend, show_R_legend,
                     xlabels, ylabel_dens, ylabel_R, R_yticklabels):
    """Draw one representative example (density on axL, R(t) on axR) in the original Fig.2 style."""
    Delta = float(d["Delta"]); M = int(dl["M"][0])
    w, Om, De = LMMF.load_mixture(dl, 0)

    # left: frequency distribution + Lorentzian-mixture fit (mixture line in the BACKGROUND)
    gx = np.linspace(-4 * Delta, 4 * Delta, 600)
    comps = w[None, :] * (De[None, :] / np.pi) / ((gx[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    axL.hist(d["omega"][0], bins=60, range=(-4 * Delta, 4 * Delta), density=True,
             color="0.85", edgecolor="none", zorder=0)
    axL.plot(gx, comps.sum(axis=1), color=C_ENS, lw=1.2, zorder=1, label="mixture")
    for k in range(M):
        axL.plot(gx, comps[:, k], color=C_COMP, lw=0.5, alpha=0.5, zorder=2)
    axL.plot(d["g_omega"], d["g_density"], color=C_SKARDAL, lw=1.1, zorder=3, label=r"$g_n(\omega)$")
    axL.set_xlim(-3 * Delta, 3 * Delta); axL.set_yticks([])
    if ylabel_dens:
        axL.set_ylabel(r"$\rho(\omega)$", labelpad=1.5)
    if xlabels:
        axL.set_xlabel(r"$\omega$", labelpad=1)
    if show_dens_legend:
        axL.legend(loc="upper left", fontsize=5.2)

    # right: R(t) comparison
    axR.plot(d["t"], d["R_micro"][0], color=C_MICRO, lw=1.0, ls="--", label="microscopic")
    axR.plot(d["t"], d["R_skardal"][0], color=C_SKARDAL, lw=1.1, ls="-", label="Skardal")
    axR.plot(dl["t"], dl["R_ensemble"][0], color=C_ENS, lw=1.1, ls="-", label="LMMF")
    axR.set_xlim(d["t"][10], d["t"][-1]); axR.set_ylim(0.0, 0.8)
    axR.set_yticks([0.0, 0.4, 0.8])
    if not R_yticklabels:
        axR.set_yticklabels([])
    if ylabel_R:
        axR.set_ylabel(r"$R(t)$", labelpad=1.5)
    if xlabels:
        axR.set_xlabel(r"time $t$", labelpad=1)
    if show_R_legend:
        axR.legend(loc="upper right", fontsize=5.2)
    return M


def make_figure(regime):
    set_prl_style()
    g = load_metrics(regime)
    Ns = sorted(g.N.unique())

    fig = plt.figure(figsize=(3.4, 6.6))
    # 5 rows: row 1 = metrics (a)+(b); rows 2-5 = the four examples (~3/4 the height of row 1)
    gs = fig.add_gridspec(5, 2, width_ratios=[1, 1.35],
                          height_ratios=[1, 0.75, 0.75, 0.75, 0.75],
                          left=0.135, right=0.975, top=0.94, bottom=0.055,
                          wspace=0.42, hspace=0.6)

    # ── ROW 1: metrics vs n [(a) # eqs, (b) spectral RMSE] ────────────────────
    ax_neq = fig.add_subplot(gs[0, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])
    for N in Ns:
        sub = g[g.N == N].sort_values("n")
        ls = N_STYLE.get(int(N), "-")
        ax_rmse.plot(sub.n, sub.rmse_sk, color=C_SKARDAL, ls=ls, lw=1.0, marker="o", ms=2.2)
        ax_rmse.plot(sub.n, sub.rmse_lmmf, color=C_ENS, ls=ls, lw=1.0, marker="o", ms=2.2)
        ax_neq.plot(sub.n, sub.M, color=C_ENS, ls=ls, lw=1.0, marker="o", ms=2.2)
    n_all = np.array(sorted(g.n.unique()))
    ax_neq.plot(n_all, n_all, color=C_SKARDAL, ls="-", lw=1.0)      # Skardal dim = n (N-independent)

    for ax in (ax_rmse, ax_neq):
        ax.set_xscale("log", base=2)
        ax.set_xlim(n_all.min() * 0.85, n_all.max() * 1.18)
        ax.set_xticks([1, 4, 16, 64, 256])
        ax.set_xticklabels([1, 4, 16, 64, 256])
        ax.set_xlabel(r"exponent $n$", labelpad=1)
    ax_rmse.set_yscale("log")
    ax_rmse.set_ylabel(r"RMSE $[\,|\hat{R}(f)|\,]$", labelpad=1.5)
    ax_neq.set_yscale("log")
    ax_neq.set_ylabel("# mean-field eqs.", labelpad=1.5)

    # both legends live on (a): line styles (N) upper-left, colours (model) lower-right
    model_handles = [Line2D([], [], color=C_SKARDAL, lw=1.2, label="Skardal"),
                     Line2D([], [], color=C_ENS, lw=1.2, label="LMMF")]
    N_handles = [Line2D([], [], color="0.35", lw=1.0, ls=N_STYLE.get(int(N), "-"),
                        label=rf"$N={int(N)}$") for N in Ns]
    N_leg = ax_neq.legend(handles=N_handles, loc="upper left", fontsize=5.8, handlelength=1.9)
    ax_neq.add_artist(N_leg)                                    # keep both legends on the same axes
    ax_neq.legend(handles=model_handles, loc="lower right", fontsize=5.8, handlelength=1.6)
    _panel_label(ax_neq, "a")
    _panel_label(ax_rmse, "b")

    # ── ROWS 2–5: four examples, one per row (density LEFT, R(t) RIGHT) ───────
    # ordered n outer, N inner -> (c) (d) (e) (f)
    examples = [(n, N) for n in EXAMPLE_n for N in EXAMPLE_N]
    for i, (n, N) in enumerate(examples):
        d, dl = _sweep_npz(n, regime, N), _lmmf_npz(n, regime, N)
        row = i + 1
        axL = fig.add_subplot(gs[row, 0])
        axR = fig.add_subplot(gs[row, 1])
        legend_here = (i == 0)                                  # first example = panel (c)
        M = _example_panels(fig, axL, axR, d, dl,
                             show_dens_legend=legend_here, show_R_legend=legend_here,
                             xlabels=(i == len(examples) - 1),
                             ylabel_dens=True, ylabel_R=True, R_yticklabels=True)
        # centred example title over the (density, R) pair + bold panel label on axL
        pL, pR = axL.get_position(), axR.get_position()
        fig.text((pL.x0 + pR.x1) / 2.0, pL.y1 + 0.012,
                 rf"$n={n}$, $N={N}$:  Skardal $M={n}$,  best fit $M={M}$",
                 ha="center", va="baseline", fontsize=6.5)
        _panel_label(axL, "cdef"[i])

    out = f"{OUT_STEM}_{regime}"
    fig.savefig(out + ".png", dpi=200)
    # fig.savefig(out + ".pdf")
    fig.savefig(out + ".svg")
    plt.close(fig)
    print(f"[saved] {os.path.basename(out)}.{{png,pdf,svg}}")


def main():
    regimes = sys.argv[1:] or REGIMES
    for regime in regimes:
        make_figure(regime)


if __name__ == "__main__":
    main()
