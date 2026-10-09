r"""
Skardal benchmark — two-column Fig. 2 (subcritical regime): Skardal vs. two LMMF variants
=========================================================================================
Compares, for the rational frequency distributions g_n (skardal_benchmark_sweep.py), the
microscopic network with
  * the exact Skardal OA reduction (n complex equations),
  * the LMMF with the noise-floor cap (fit stops at D(M) <= 1/(6N): approximates the
    POPULATION distribution g_n; skardal_benchmark_lmmf.py, variant "floor"),
  * the LMMF without noise floor (former GoF acceptance: keeps fitting below the sampling noise,
    i.e. resolves the sample-specific structure of each network REALISATION; variant "nofloor").

Layout (PRL two-column, 4 rows × 3 columns):
  row 1: (a) number of mean-field equations vs n (all N), (b) spectral RMSE vs n at N_EX,
         (c) spectral RMSE vs N (averaged over n) — trial means ± s.e.m.
  rows 2-4: examples at N = N_EX for n in EXAMPLE_n: floor-capped fit | no-floor fit | R(t).

Run in the ``pycobi`` conda env after the sweep and both LMMF variants:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python skardal_benchmark_twocol_figure.py [regime]
"""
# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, re, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
from prl_style import set_prl_style as _set_prl_style
from prl_style import panel_label
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import skardal_benchmark_lmmf as LMMF

DATA_DIR, STEM = LMMF.CONFIG["in_dir"], LMMF.CONFIG["in_stem"]
REGIME = "subcritical"
EXAMPLE_n = [2, 16, 128]
N_EX = 1000
TRIAL_EX = 0

C_MICRO, C_SKARDAL, C_COMP = "0.25", "#1f77b4", "#2e6f95"
C_FLOOR, C_NOFLOOR = "#c1121f", "#ee8a00"
N_STYLE = {200: ":", 1000: "--", 5000: "-"}
MODELS = [("sk", "Skardal", C_SKARDAL), ("floor", "LMMF, noise floor", C_FLOOR),
          ("nofloor", "LMMF, no noise floor", C_NOFLOOR)]

set_prl_style = functools.partial(_set_prl_style, "prl")
_panel_label = functools.partial(panel_label, dx=-16, dy=4)


def _npz(n, regime, N, suffix=""):
    return np.load(os.path.join(DATA_DIR, f"{STEM}_n{n}_{regime}_N{N}{suffix}.npz"), allow_pickle=False)


def _fft_rmse(a, b):
    L = min(len(a), len(b))
    A, B = np.abs(np.fft.rfft(a[:L])) / L, np.abs(np.fft.rfft(b[:L])) / L
    return float(np.sqrt(np.mean((A - B) ** 2)))


def load_metrics(regime):
    """Per-trial spectral RMSE (vs. micro) and number of equations for all three models."""
    pat = re.compile(rf"^{re.escape(STEM)}_n(\d+)_{regime}_N(\d+)\.npz$")
    recs = []
    for f in sorted(os.listdir(DATA_DIR)):
        mt = pat.match(f)
        if not mt:
            continue
        n, N = int(mt.group(1)), int(mt.group(2))
        d = _npz(n, regime, N)
        fl, nf = _npz(n, regime, N, "_lmmf"), _npz(n, regime, N, "_lmmf_nofloor")
        for tr in range(int(d["n_trials"])):
            Rm = d["R_micro"][tr]
            recs.append(dict(N=N, n=n, trial=tr,
                             rmse_sk=_fft_rmse(d["R_skardal"][tr], Rm), M_sk=n,
                             rmse_floor=_fft_rmse(fl["R_ensemble"][tr], Rm), M_floor=int(fl["M"][tr]),
                             rmse_nofloor=_fft_rmse(nf["R_ensemble"][tr], Rm),
                             M_nofloor=int(nf["M"][tr])))
    return pd.DataFrame(recs)


def _mean_sem(df, by, col):
    g = df.groupby(by)[col]
    return g.mean(), g.sem()


def _density(ax, d, dl, color, ylabel, xlabel, legend):
    Delta = float(d["Delta"])
    w, Om, De = LMMF.load_mixture(dl, TRIAL_EX)
    gx = np.linspace(-4 * Delta, 4 * Delta, 800)
    comps = w[None, :] * (De[None, :] / np.pi) / ((gx[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    ax.hist(d["omega"][TRIAL_EX], bins=60, range=(-4 * Delta, 4 * Delta), density=True,
            color="0.85", edgecolor="none", zorder=0)
    for k in range(comps.shape[1]):
        ax.plot(gx, comps[:, k], color=C_COMP, lw=0.4, alpha=0.5, zorder=1)
    ax.plot(gx, comps.sum(1), color=color, lw=1.1, zorder=2, label="mixture")
    ax.plot(d["g_omega"], d["g_density"], color=C_SKARDAL, lw=1.0, zorder=3, label=r"$g_n(\omega)$")
    ax.set_xlim(-2.5 * Delta, 2.5 * Delta); ax.set_yticks([])
    if ylabel:
        ax.set_ylabel(r"$\rho(\omega)$", labelpad=1.5)
    if xlabel:
        ax.set_xlabel(r"$\omega$", labelpad=1)
    if legend:
        ax.legend(loc="upper left", fontsize=5.5, handlelength=1.4)
    return len(w)


def make_figure(regime=REGIME):
    set_prl_style()
    df = load_metrics(regime)
    Ns, ns = sorted(df.N.unique()), sorted(df.n.unique())

    fig = plt.figure(figsize=(7.0, 6.2))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.15, 1, 1, 1], left=0.07, right=0.985,
                          top=0.955, bottom=0.06, wspace=0.28, hspace=0.62)

    # ── row 1 ────────────────────────────────────────────────────────────────
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, j]) for j in range(3))
    for N in Ns:
        sub = df[df.N == N]
        for key, _, col in MODELS[1:]:
            m, _ = _mean_sem(sub, "n", f"M_{key}")
            ax_a.plot(m.index, m.values, color=col, ls=N_STYLE[N], lw=1.0, marker="o", ms=2.0)
    ax_a.plot(ns, ns, color=C_SKARDAL, lw=1.0)                        # Skardal: n equations
    sub = df[df.N == N_EX]
    for key, _, col in MODELS:
        m, s = _mean_sem(sub, "n", f"rmse_{key}")
        ax_b.plot(m.index, m.values, color=col, lw=1.0, marker="o", ms=2.2)
        ax_b.fill_between(m.index, m - s, m + s, color=col, alpha=0.18, lw=0)
        m, s = _mean_sem(df, "N", f"rmse_{key}")
        ax_c.plot(m.index, m.values, color=col, lw=1.0, marker="o", ms=2.6)
        ax_c.fill_between(m.index, m - s, m + s, color=col, alpha=0.18, lw=0)
    for ax in (ax_a, ax_b):
        ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64, 256])
        ax.set_xticklabels([1, 4, 16, 64, 256]); ax.set_xlabel(r"exponent $n$", labelpad=1)
    ax_a.set_yscale("log"); ax_a.set_ylabel("# mean-field equations", labelpad=1.5)
    ax_b.set_yscale("log"); ax_b.set_ylabel(r"spectral RMSE", labelpad=1.5)
    ax_b.set_title(rf"$N={N_EX}$", fontsize=7, pad=2)
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xticks(Ns); ax_c.set_xticklabels([str(N) for N in Ns]); ax_c.minorticks_off()
    ax_c.set_xlabel(r"network size $N$", labelpad=1); ax_c.set_ylabel("spectral RMSE", labelpad=1.5)
    ax_c.set_title(r"mean over $n$", fontsize=7, pad=2)
    for ax in (ax_b, ax_c):                    # < 1 decade on a log axis: explicit tick labels
        lo, hi = ax.get_ylim()
        ticks = [v for v in (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2) if lo <= v <= hi]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{v * 1e3:g}" for v in ticks])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylabel(r"spectral RMSE [$10^{-3}$]", labelpad=1.5)
    N_leg = ax_a.legend(handles=[Line2D([], [], color="0.35", ls=N_STYLE[N], lw=1.0, label=rf"$N={N}$")
                                 for N in Ns], loc="upper left", fontsize=5.8, handlelength=2.0)
    ax_a.add_artist(N_leg)
    ax_c.legend(handles=[Line2D([], [], color=c, lw=1.2, label=l) for _, l, c in MODELS],
                loc="upper right", fontsize=5.8, handlelength=1.6)
    for ax, lab in zip((ax_a, ax_b, ax_c), "abc"):
        _panel_label(ax, lab)

    # ── rows 2-4: examples at N = N_EX ──────────────────────────────────────────
    labels = iter("defghijkl")
    for i, n in enumerate(EXAMPLE_n):
        d = _npz(n, regime, N_EX)
        fl, nf = _npz(n, regime, N_EX, "_lmmf"), _npz(n, regime, N_EX, "_lmmf_nofloor")
        last = (i == len(EXAMPLE_n) - 1)
        ax1, ax2, ax3 = (fig.add_subplot(gs[i + 1, j]) for j in range(3))
        M1 = _density(ax1, d, fl, C_FLOOR, ylabel=True, xlabel=last, legend=(i == 0))
        M2 = _density(ax2, d, nf, C_NOFLOOR, ylabel=False, xlabel=last, legend=False)
        ax1.set_title(rf"$n={n}$: noise floor, $M={M1}$", fontsize=6.5, pad=2)
        ax2.set_title(rf"$n={n}$: no noise floor, $M={M2}$", fontsize=6.5, pad=2)
        t = d["t"]
        ax3.plot(t, d["R_micro"][TRIAL_EX], color=C_MICRO, lw=1.0, ls="--", label="network")
        ax3.plot(t, d["R_skardal"][TRIAL_EX], color=C_SKARDAL, lw=1.0, label=rf"Skardal ($M={n}$)")
        ax3.plot(fl["t"], fl["R_ensemble"][TRIAL_EX], color=C_FLOOR, lw=1.0, label="LMMF, noise floor")
        ax3.plot(nf["t"], nf["R_ensemble"][TRIAL_EX], color=C_NOFLOOR, lw=1.0, label="LMMF, no floor")
        ax3.set_xlim(t[0], t[-1]); ax3.set_ylim(0, 0.92); ax3.set_yticks([0, 0.4, 0.8])
        ax3.set_ylabel(r"$R(t)$", labelpad=1.5)
        ax3.set_title(rf"$n={n}$, $N={N_EX}$", fontsize=6.5, pad=2)
        if last:
            ax3.set_xlabel(r"time $t$", labelpad=1)
        if i == 0:
            ax3.legend(loc="upper right", fontsize=5.5, handlelength=1.6)
        for ax in (ax1, ax2, ax3):
            _panel_label(ax, next(labels))

    out = os.path.join(DATA_DIR, f"skardal_twocol_figure_{regime}")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[saved] {out}.{{png,svg,pdf}}")
    summ = df.groupby("N")[["M_floor", "M_nofloor", "rmse_sk", "rmse_floor", "rmse_nofloor"]].mean()
    print(summ.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    make_figure(sys.argv[1] if len(sys.argv) > 1 else REGIME)
