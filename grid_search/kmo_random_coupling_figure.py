r"""
Kuramoto structured coupling vs. mean-only & correlation-aware LMMF — figures
=============================================================================

Loads the sweep written by ``kmo_random_coupling_sweep.py`` (coupling A_ij = k_i k_j + N(0,σ),
k_i = 1 + c·p_i linear in the ω-sorted index) and compares the micro network against TWO
mean-field models that share one Lorentzian-mixture fit:
    mean — mean-only global LMMF (knows only μ);    corr — correlation-aware LMMF.

1. SUMMARY: time-domain RMSE between R_micro(t) and each MF R(t) as a function of the strength
   slope c — a line plot with one line per σ (mean ± std over trials).  One panel per MF model.

2. EXAMPLES: for representative (σ, c), the shared Lorentzian-mixture fit (top), plus the micro &
   both MF R(t) and the coupling matrix A_ij for a representative (median-RMSE) trial.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_random_coupling_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = "/home/rgast/data/mpmf_simulations/kmo_random_coupling_sweep.csv"
OUT_SUMMARY = "/home/rgast/data/mpmf_simulations/kmo_random_coupling_summary"
OUT_EXAMPLES = "/home/rgast/data/mpmf_simulations/kmo_random_coupling_examples"

C_MICRO = "0.2"
MODELS = ["mean", "corr"]
MODEL_STYLE = {                                   # (colour, linestyle, label)
    "mean": ("#c1121f", "--", "mean-only LMMF"),
    "corr": ("#1f77b4", ":",  "corr.-aware LMMF"),
}
SIGMA_CMAP = "viridis"           # encodes σ in the summary line plots
MATRIX_CMAP = "magma"
MIX_TRIAL = 0                    # which trial's (shared) mixture fit to show in the examples
SIGMA_PLOT = None                # σ rows for the examples figure (None = all σ)
C_PLOT = None                    # c columns for the examples (None = min / median / max c)


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
    df = pd.read_csv(csv, low_memory=False)
    sigmas = sorted(df["sigma"].dropna().unique())
    cs = sorted(df["c"].dropna().unique())
    trials = sorted(int(t) for t in df["trial"].dropna().unique())
    models = [m for m in MODELS if m in set(df["model"].dropna().unique())]
    omega_max = float(df["omega_max"].dropna().iloc[0])

    # shared LMMF mixture, refit per trial → show one trial's fit (MIX_TRIAL)
    mix_all = df[df.quantity == "mixture"]
    mt = MIX_TRIAL if (mix_all["trial"] == MIX_TRIAL).any() else int(mix_all["trial"].dropna().min())
    g = mix_all[mix_all["trial"] == mt].sort_values("Omega")
    mixture = (g["w"].to_numpy(), g["Omega"].to_numpy(), g["Delta"].to_numpy())
    M = len(g)

    R_mf = {}
    for (mo, s, c, tr), gg in df[df.quantity == "R_mf"].groupby(["model", "sigma", "c", "trial"]):
        gg = gg.sort_values("time")
        R_mf[(mo, float(s), float(c), int(tr))] = (gg["time"].to_numpy(), gg["value"].to_numpy())
    R_mic, corr = {}, {}
    for (s, c, tr), gg in df[df.quantity == "R_micro"].groupby(["sigma", "c", "trial"]):
        gg = gg.sort_values("time")
        R_mic[(float(s), float(c), int(tr))] = (gg["time"].to_numpy(), gg["value"].to_numpy())
    for (s, c, tr), gg in df[df.quantity == "corr"].groupby(["sigma", "c", "trial"]):
        corr[(float(s), float(c), int(tr))] = float(gg["c_real"].iloc[0])
    mats = {}
    for (s, c, tr), gg in df[df.quantity == "A_final"].groupby(["sigma", "c", "trial"]):
        nr, nc = int(gg["row"].max()) + 1, int(gg["col"].max()) + 1
        Mx = np.full((nr, nc), np.nan)
        Mx[gg["row"].astype(int), gg["col"].astype(int)] = gg["value"].to_numpy()
        mats[(float(s), float(c), int(tr))] = Mx
    return sigmas, cs, trials, models, mixture, M, mt, omega_max, R_mic, R_mf, mats, corr


def rmse(R_mic, R_mf, model, s, c, tr):
    """Time-domain RMSE between micro and a given MF model's R(t) for one trial."""
    _, rm = R_mic[(s, c, tr)]
    _, rf = R_mf[(model, s, c, tr)]
    n = min(len(rm), len(rf))
    return float(np.sqrt(np.mean((rm[:n] - rf[:n]) ** 2)))


def rmse_stats(R_mic, R_mf, model, s, c, trials):
    vals = np.array([rmse(R_mic, R_mf, model, s, c, tr) for tr in trials], float)
    vals = vals[np.isfinite(vals)]
    return (float(vals.mean()), float(vals.std())) if vals.size else (np.nan, np.nan)


def representative_trial(R_mic, R_mf, s, c, trials, model="mean"):
    order = sorted(trials, key=lambda tr: rmse(R_mic, R_mf, model, s, c, tr))
    return order[len(order) // 2]


def _plot_mixture(ax, mixture, omega_max, M, mt):
    w, Om, De = mixture
    xg = np.linspace(-1.15 * omega_max, 1.15 * omega_max, 400)
    for k in range(len(w)):
        ax.plot(xg, w[k] * (De[k] / np.pi) / ((xg - Om[k]) ** 2 + De[k] ** 2),
                color="0.75", lw=0.4, zorder=1)
    dens = (w[None, :] * (De[None, :] / np.pi) / ((xg[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)).sum(1)
    ax.plot(xg, dens, color="#1f77b4", lw=1.3, zorder=3)
    uni = 1.0 / (2 * omega_max)
    ax.plot([-omega_max, -omega_max, omega_max, omega_max], [0, uni, uni, 0],
            color="0.35", ls="--", lw=0.8, zorder=2)
    ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$\omega$", labelpad=1); ax.set_ylabel(r"$\rho(\omega)$", labelpad=2)
    ax.set_title(rf"shared LMMF mixture (trial {mt}, $M{{=}}{M}$)", fontsize=6.5, pad=2)


# ════════════════════════════════════════════════════════════════════════════
#  figure 1 — summary: RMSE vs strength slope c, one line per σ, one panel per MF model
# ════════════════════════════════════════════════════════════════════════════
def make_summary(sigmas, cs, trials, models, M, R_mic, R_mf):
    cmap = plt.get_cmap(SIGMA_CMAP)
    scol = [cmap(0.12 + 0.76 * i / max(1, len(sigmas) - 1)) for i in range(len(sigmas))]
    cx = np.asarray(cs)

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 2.6),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for mi, mo in enumerate(models):
        ax = axes[mi]
        for si, s in enumerate(sigmas):
            ms = np.array([rmse_stats(R_mic, R_mf, mo, s, c, trials) for c in cs])
            ax.errorbar(cx, ms[:, 0], yerr=ms[:, 1], color=scol[si], marker="o", ms=2.6, lw=1.0,
                        capsize=1.5, elinewidth=0.6, label=f"{s:g}")
        ax.set_title(rf"{MODEL_STYLE[mo][2]}  ($M{{=}}{M}$)", fontsize=7, pad=3)
        ax.set_xlabel(r"strength slope $c$", labelpad=1)
        ax.margins(x=0.08)
        if mi == 0:
            ax.set_ylabel(r"$R(t)$ RMSE", labelpad=2)
            ax.legend(title=r"$\sigma$", ncol=1, fontsize=5.8, title_fontsize=6, handlelength=1.4,
                      loc="best")

    fig.savefig(OUT_SUMMARY + ".pdf")
    fig.savefig(OUT_SUMMARY + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT_SUMMARY}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
#  figure 2 — representative examples: mixture fit + R(t) (micro vs 2 MF) + coupling matrix
# ════════════════════════════════════════════════════════════════════════════
def make_examples(sigmas, cs, trials, models, mixture, M, mt, omega_max, R_mic, R_mf, mats, corr):
    sig_plot = list(SIGMA_PLOT) if SIGMA_PLOT is not None else list(sigmas)
    c_plot = list(C_PLOT) if C_PLOT is not None else [cs[0], cs[len(cs) // 2], cs[-1]]
    nrow, ncc = len(sig_plot) + 1, len(c_plot)

    fig = plt.figure(figsize=(7.0, 1.05 * len(sig_plot) + 1.2), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.02, wspace=0.02, hspace=0.07)
    gs = fig.add_gridspec(nrow, 3 * ncc, height_ratios=[0.75] + [1.0] * len(sig_plot))

    # top row: the shared Lorentzian-mixture fit (centred block)
    _plot_mixture(fig.add_subplot(gs[0, ncc:2 * ncc]), mixture, omega_max, M, mt)

    for ri, s in enumerate(sig_plot):
        for ci_i, c in enumerate(c_plot):
            b = 3 * ci_i
            ax_dyn = fig.add_subplot(gs[ri + 1, b:b + 2])
            ax_mat = fig.add_subplot(gs[ri + 1, b + 2])
            tr = representative_trial(R_mic, R_mf, s, c, trials)
            t, Rm = R_mic[(s, c, tr)]
            ax_dyn.plot(t, Rm, color=C_MICRO, lw=1.0, label="micro", zorder=5)
            for mo in models:
                col, ls, _ = MODEL_STYLE[mo]
                tf, Rf = R_mf[(mo, s, c, tr)]
                ax_dyn.plot(tf, Rf, color=col, lw=0.9, ls=ls)
            ax_dyn.set_xlim(t[0], t[-1]); ax_dyn.set_ylim(-0.02, 1.02); ax_dyn.set_yticks([0, 0.5, 1.0])
            if ri == 0:
                ax_dyn.set_title(rf"$c={c:g}$ ($c_{{\rm real}}{{=}}{corr.get((s, c, tr), c):+.2f}$)",
                                 fontsize=6.0, pad=2)
            if ri == len(sig_plot) - 1:
                ax_dyn.set_xlabel(r"$t$", labelpad=1)
            else:
                ax_dyn.set_xticklabels([])
            if ci_i == 0:
                ax_dyn.set_ylabel(rf"$\sigma={s:g}$" + "\n" + r"$R(t)$", labelpad=2)
            else:
                ax_dyn.set_yticklabels([])

            Mx = mats[(s, c, tr)]
            im = ax_mat.imshow(Mx, origin="lower", aspect="equal", cmap=MATRIX_CMAP,
                               vmin=np.nanmin(Mx), vmax=np.nanmax(Mx), interpolation="nearest")
            ax_mat.set_xticks([]); ax_mat.set_yticks([])
            if ri == 0:
                ax_mat.set_title(r"$A_{ij}$", fontsize=6.0, pad=2)
            cb = fig.colorbar(im, ax=ax_mat, fraction=0.05, pad=0.02)
            cb.ax.tick_params(labelsize=4.5, pad=0.6)

    handles = [Line2D([0], [0], color=C_MICRO, lw=1.0, label="micro")]
    handles += [Line2D([0], [0], color=MODEL_STYLE[mo][0], lw=0.9, ls=MODEL_STYLE[mo][1],
                       label=MODEL_STYLE[mo][2]) for mo in models]
    fig.legend(handles=handles, loc="outside upper right", ncol=3, fontsize=6)

    fig.savefig(OUT_EXAMPLES + ".pdf")
    fig.savefig(OUT_EXAMPLES + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT_EXAMPLES}.pdf / .png")


def main():
    (sigmas, cs, trials, models, mixture, M, mt, omega_max,
     R_mic, R_mf, mats, corr) = load(CSV)
    set_prl_style()
    make_summary(sigmas, cs, trials, models, M, R_mic, R_mf)
    make_examples(sigmas, cs, trials, models, mixture, M, mt, omega_max, R_mic, R_mf, mats, corr)


if __name__ == "__main__":
    main()
