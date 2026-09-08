r"""
Multi-(σ, c) probe: microscopic coherence dynamics + coupling matrix per combination
====================================================================================

Simulates the frequency-assortative Kuramoto network (A_ij = c²(ω_i−ω̄)(ω_j−ω̄) + N(1,σ)) at N=200
for a HAND-PICKED list of (σ, c) combinations and tiles them — in the style of the summary figure's
example rows — so the candidate sweep settings can be compared at a glance:
each row = one (σ, c) with the average phase-coherence dynamics R(t) (micro + mean-only LMMF +
analytic corr.-aware LMMF) next to the (square) microscopic coupling matrix A_ij.

A single shared ω-sample + LMMF fit is reused across all combinations (so only σ, c differ); the
mean-only MF is identical for all rows and the corr.-aware MF depends on c only.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_random_coupling_single.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grid_search"))
import kmo_lorentzian_fit_sweep as KFS          # noqa: E402  (in ../grid_search)
import kmo_random_coupling_sweep as S           # noqa: E402  (in ../grid_search)

# ── what to simulate ─────────────────────────────────────────────────────────
N = 200
SEED = 1
COMBOS = [               # (σ, c) candidates to compare — edit to probe the sweep space
    (0.1, 0.3),
    (0.1, 0.6),
    (0.5, 0.3),
    (0.5, 0.6),
]

C_MICRO = "0.2"
MODEL_STYLE = {                                   # (colour, linestyle, label)
    "mean": ("#c1121f", "--", "mean-only LMMF"),
    "corr": ("#1f77b4", ":",  "corr.-aware LMMF"),
}
MATRIX_CMAP = "magma"
OUT = "/home/rgast/data/mpmf_simulations/kmo_random_coupling_single"


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def main():
    cfg = dict(S.CONFIG, N=N)
    rng = np.random.default_rng(SEED)
    K, mu = cfg["K"], 1.0

    # shared ω-sample + LMMF fit (reused by every (σ, c) so only the coupling differs)
    omega = np.sort(rng.uniform(-cfg["omega_max"], cfg["omega_max"], N))
    theta0 = rng.normal(0.0, cfg["sigma0"], N)
    R0 = float(np.abs(np.exp(1j * theta0).mean()))
    model = KFS.LM.fit(omega, cfg["delta_bounds"], M_max=cfg["fit_M_max"], alpha=cfg["fit_alpha"],
                       lambda_M=cfg["fit_lambda"], patience=3, loss=cfg["fit_loss"],
                       n_restarts=cfg["fit_restarts"], seed=cfg["seed"], method=cfg["fit_method"])["model"]
    wbar = float(omega.mean())
    mu_m = model.Omega - wbar
    om = omega - wbar

    # mean-only MF: identical for all rows; corr.-aware MF: depends on c only (cache it)
    _, R_mean = KFS.simulate_ensemble(model.w, model.Omega, model.Delta, K * mu, R0, cfg, tag="single_mf")
    corr_cache = {}

    rows = []
    for (sigma, c) in COMBOS:
        A = S.build_coupling(omega, c, sigma, rng)               # A_ij = c²(ω_i−ω̄)(ω_j−ω̄) + N(1, σ)
        t, R_mic = S.simulate_micro(omega, A, theta0, cfg)
        if c not in corr_cache:
            corr_cache[c] = S.simulate_ensemble_analytic(model, mu_m, K, c, R0, cfg)[1]
        R_corr = corr_cache[c]
        c_real = S._pearson(A.ravel(), np.outer(om, om).ravel())
        print(f"σ={sigma:<4} c={c:<5}(c_real={c_real:+.2f}) -> "
              f"R_mic={R_mic[-1]:.3f}, mean={R_mean[-1]:.3f}, corr={R_corr[-1]:.3f}")
        rows.append((sigma, c, c_real, t, R_mic, R_corr, A))

    # ── figure: one row per (σ, c) = [R(t) (wide) | square coupling matrix | colourbar] ──────────
    set_prl_style()
    nr = len(rows)
    fig = plt.figure(figsize=(3.4, 0.95 * nr + 0.35), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.06)
    gs = fig.add_gridspec(nr, 3, width_ratios=[2.25, 1.0, 0.08])

    for ri, (sigma, c, c_real, t, R_mic, R_corr, A) in enumerate(rows):
        ax_dyn = fig.add_subplot(gs[ri, 0])
        ax_mat = fig.add_subplot(gs[ri, 1])
        cax = fig.add_subplot(gs[ri, 2])

        ax_dyn.plot(t, R_mic, color=C_MICRO, lw=1.0, zorder=5)
        for mo, Rf in (("mean", R_mean), ("corr", R_corr)):
            col, ls, _ = MODEL_STYLE[mo]
            ax_dyn.plot(t, Rf, color=col, lw=0.9, ls=ls)
        ax_dyn.set_xlim(t[0], t[-1]); ax_dyn.set_ylim(-0.02, 1.02); ax_dyn.set_yticks([0, 0.5, 1.0])
        ax_dyn.set_ylabel(r"$R(t)$", labelpad=2)
        ax_dyn.annotate(rf"$\sigma={sigma:g}$,  $c={c:g}$  ($c_{{\rm real}}{{=}}{c_real:+.2f}$)",
                        xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom", fontsize=5.6)
        if ri == 0:
            handles = [Line2D([0], [0], color=C_MICRO, lw=1.0, label="micro")]
            handles += [Line2D([0], [0], color=MODEL_STYLE[mo][0], lw=0.9, ls=MODEL_STYLE[mo][1],
                               label=MODEL_STYLE[mo][2]) for mo in ("mean", "corr")]
            ax_dyn.legend(handles=handles, loc="lower left", fontsize=5.0, handlelength=1.3,
                          borderaxespad=0.3, labelspacing=0.2)
        if ri == nr - 1:
            ax_dyn.set_xlabel(r"$t$", labelpad=1)
        else:
            ax_dyn.set_xticklabels([])

        im = ax_mat.imshow(A, origin="lower", aspect="equal", cmap=MATRIX_CMAP,
                           vmin=A.min(), vmax=A.max(), interpolation="nearest")
        ax_mat.set_xticks([]); ax_mat.set_yticks([])
        if ri == 0:
            ax_mat.set_title(r"$A_{ij}$", fontsize=6.5, pad=2)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=4.5, pad=0.6, length=1.5)

    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
