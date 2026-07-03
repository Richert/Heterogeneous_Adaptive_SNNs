r"""
Skardal-model benchmark — part 2: Lorentzian-mixture ensemble + comparison figure
=================================================================================

Loads the results written by skardal_benchmark_simulate.py (one .npz per exponent n),
and for each n:
  1. fits a Lorentzian mixture to the saved frequency samples with the CvM algorithm
     (theory/lorentzian_mixture.py) — this is the approach being benchmarked,
  2. simulates the multi-ensemble Ott–Antonsen model built from that mixture (same
     PyRates+solve_ivp ensemble model as kmo_lorentzian_fit_sweep), from the same
     coherent initial condition R(0)=R0,
  3. saves the ensemble dynamics, and
  4. plots the three-way comparison of the average phase-coherence dynamics R(t):
     microscopic Kuramoto vs. Skardal mean field vs. our Lorentzian-mixture ensemble.

The model-order contrast is annotated: the Skardal reduction needs n complex equations,
ours needs M (the fitted mixture size).

NOTE: the plotting layout here is PROVISIONAL — to be refined per the user's spec.

Run in the ``pycobi`` conda env:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python skardal_benchmark_figure.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
gs_path = os.path.join(_HERE, "..", "grid_search")
sys.path.insert(0, gs_path)
import kmo_lorentzian_fit_sweep as KFS          # reuse simulate_ensemble + LM (lorentzian_mixture)

DATA_DIR = "/home/rgast/data/mpmf_simulations"
IN_STEM = "skardal_benchmark_n"
OUT = os.path.join(DATA_DIR, "skardal_benchmark")

# Lorentzian-mixture fit (greedy CvM, penalized GoF selection)
DELTA_BOUNDS = (0.01, 1.5)       # width bounds on the ω-scale (g_n has spread Δ≈1)
M_MAX = 100
ALPHA = 0.001
LAMBDA_M = 1e-6
PATIENCE = 2
N_RESTARTS = 10
SEED = 1
FIT_METHOD = "slsqp"

C_MICRO = "0.25"
C_SKARDAL = "#1f77b4"
C_ENS = "#c1121f"
C_COMP = "#2e6f95"


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5, "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def _panel_label(ax, letter):
    """Bold PRL-style panel label OUTSIDE the axis box, above its top-left corner."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-14, 4),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


def main():
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.startswith(IN_STEM) and f.endswith(".npz")
                   and "ensemble" not in f)
    if not files:
        raise SystemExit(f"no {IN_STEM}*.npz found in {DATA_DIR} (run skardal_benchmark_simulate.py)")
    results = []
    for f in files:
        d = np.load(os.path.join(DATA_DIR, f), allow_pickle=False)
        n = int(d["n"]); K = float(d["K"]); R0 = float(d["R0"]); omega = d["omega"]
        cfg = {k: float(d[k]) for k in ("T", "dt", "dts", "rtol", "atol")}

        # (1) fit the Lorentzian mixture to the frequency samples
        res = KFS.LM.fit(omega, DELTA_BOUNDS, M_max=M_MAX, alpha=ALPHA, lambda_M=LAMBDA_M,
                         patience=PATIENCE, loss="cvm", n_restarts=N_RESTARTS, seed=SEED,
                         method=FIT_METHOD)
        m, M = res["model"], res["M"]
        # (2) simulate the ensemble OA model from the fitted mixture
        t_ens, R_ens = KFS.simulate_ensemble(m.w, m.Omega, m.Delta, K, R0, cfg, tag=f"skens{n}")
        print(f"n={n}: Skardal MF dim={n}, fitted mixture M={M} -> "
              f"micro R(end)={float(d['R_micro'][-1]):.3f}, Skardal={float(d['R_skardal'][-1]):.3f}, "
              f"ensemble={R_ens[-1]:.3f}")

        # (3) save ensemble results
        np.savez(os.path.join(DATA_DIR, f"{IN_STEM}{n}_ensemble.npz"),
                 n=np.int64(n), M=np.int64(M), t=t_ens, R_ensemble=R_ens,
                 weights=m.w, omega=m.Omega, delta=m.Delta)
        results.append(dict(n=n, d=d, m=m, M=M, t_ens=t_ens, R_ens=R_ens))

    # (4) single-column PRL figure — explicit layout reproduced from the hand-tuned figure2.svg.
    #     3 rows = n; cols = {frequency density (left), R(t) (right)}. All positions are in points.
    results.sort(key=lambda r: r["n"])
    set_prl_style()
    W, H = 254.88, 245.52                                   # figure size (pt), from figure2.svg
    fig = plt.figure(figsize=(W / 72.0, H / 72.0))
    COL_L = {"L": 21.2, "R": 148.6875}; WCOL = 98.9925      # column left edges + width (pt)
    ROWS = [(18.2475, 56.6425), (95.2875, 56.6425), (172.3275, 48.8925)]   # (y_top, height) per row
    TITLE_YB = [13.8756, 90.9156, 167.9556]                 # row-title baselines (pt from the top)
    ROW_CX = (COL_L["L"] + COL_L["R"] + WCOL) / 2.0 / W      # horizontal centre of a row (fig frac)

    def _ax(side, ytop, h):                                  # SVG (pt, top-down) -> add_axes rect
        return fig.add_axes([COL_L[side] / W, 1.0 - (ytop + h) / H, WCOL / W, h / H])

    for i, r in enumerate(results):
        n, d, m, M = r["n"], r["d"], r["m"], r["M"]
        Delta = float(d["Delta"]); last = (i == len(results) - 1)
        ytop, h = ROWS[i]
        axL, axR = _ax("L", ytop, h), _ax("R", ytop, h)
        # left column: frequency distribution + Lorentzian-mixture fit (mixture line in the BACKGROUND)
        gx = np.linspace(-4 * Delta, 4 * Delta, 600)
        comps = m.w[None, :] * (m.Delta[None, :] / np.pi) / ((gx[:, None] - m.Omega[None, :]) ** 2 + m.Delta[None, :] ** 2)
        axL.hist(d["omega"], bins=60, range=(-4 * Delta, 4 * Delta), density=True,
                 color="0.85", edgecolor="none", zorder=0)
        axL.plot(gx, comps.sum(axis=1), color=C_ENS, lw=1.2, zorder=1, label="mixture")  # background
        for k in range(M):
            axL.plot(gx, comps[:, k], color=C_COMP, lw=0.5, alpha=0.5, zorder=2)
        axL.plot(d["g_omega"], d["g_density"], color=C_SKARDAL, lw=1.1, zorder=3, label=r"$g_n(\omega)$")
        axL.set_xlim(-3 * Delta, 3 * Delta); axL.set_yticks([]); axL.set_ylabel(r"$\rho(\omega)$")
        if last:
            axL.set_xlabel(r"$\omega$", labelpad=1)
        if i == 0:
            axL.legend(loc="upper right", fontsize=5.5)
        # right column: R(t) comparison
        axR.plot(d["t"], d["R_micro"], color=C_MICRO, lw=1.0, label="microscopic")
        axR.plot(d["t"], d["R_skardal"], color=C_SKARDAL, lw=1.1, ls="-", label="Skardal")
        axR.plot(r["t_ens"], r["R_ens"], color=C_ENS, lw=1.1, ls="--", label="best fit")
        axR.set_xlim(d["t"][0], d["t"][-1]); axR.set_ylim(-0.02, 1.02); axR.set_ylabel(r"$R(t)$")
        if last:
            axR.set_xlabel(r"time $t$", labelpad=1)
        if i == 0:
            axR.legend(loc="upper right", fontsize=5.5)
        # centred row title (baseline from figure2.svg) + bold panel label outside axL's corner
        fig.text(ROW_CX, 1.0 - TITLE_YB[i] / H, rf"$n={n}$:  Skardal $M={n}$,  best fit $M={M}$",
                 ha="center", va="baseline", fontsize=8)
        _panel_label(axL, "abcdefgh"[i])

    fig.savefig(OUT + ".png", dpi=300)
    fig.savefig(OUT + ".svg")
    print(f"[saved] {OUT}.{{png,svg}}")


if __name__ == "__main__":
    main()
