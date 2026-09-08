r"""
Kuramoto structured coupling vs. mean-only & correlation-aware LMMF — journal figure
====================================================================================

Loads the sweep written by ``kmo_random_coupling_sweep.py`` (coupling A_ij = c²(ω_i−ω̄)(ω_j−ω̄) +
N(1,σ)) and compares the micro network against TWO mean-field models that share one
Lorentzian-mixture fit:
    mean — mean-only global LMMF (knows only μ);    corr — analytic correlation-aware LMMF.

ONE single-column PRL figure (layout hand-tuned to manuscript figure1.svg):
  * row 1, left   — spectral RMSE(R_micro, R_mf) vs coupling coefficient c, BOTH MF models in one axis
                    (colour = model, line transparency = noise σ: σ=0 opaque → more σ more transparent);
                    the σ legend sits ABOVE the axes, the model legend inside upper-left;
  * row 1, right  — a representative shared Lorentzian-mixture fit over its empirical ω-sample
                    (skardal_benchmark_figure.py style: histogram + components + sum + uniform box);
  * rows 2–4      — three sweep examples (intermediate σ at c≈0 and max c; max σ at max c), each row a
                    micro-vs-MF R(t) panel (σ/c annotated upper-right) + the square coupling matrix A_ij.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_random_coupling_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from kmo_lorentzian_fit_figure import spectral_rmse   # Fourier-amplitude RMSE (robust to phase shifts)

CSV = "/home/rgast/data/mpmf_simulations/kmo_random_coupling_sweep.csv"
OUT = "/home/rgast/data/mpmf_simulations/kmo_random_coupling"

C_MICRO = "0.2"
MODELS = ["mean", "corr"]
MODEL_STYLE = {                                   # (colour, linestyle, label)
    "mean": ("#c1121f", "--", "mean-only LMMF"),
    "corr": ("#1f77b4", "--", "corr.-aware LMMF"),
}
MICRO_ALPHA = 0.7
ALPHA_MIN = 0.4                  # most-noisy σ → this transparency (σ=0 → 1.0)
MATRIX_CMAP = "magma"
MIX_TRIAL = 0                    # which trial's (shared) mixture fit + ω-sample to show in row 1
EXAMPLES = None                  # list of (σ, c) example rows; None = auto (see _example_pairs)


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


def _panel_label(ax, letter, dx=-20, dy=4):
    """Bold PRL-style panel label OUTSIDE the axis box, above its top-left corner."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


def _alpha_for_sigma(i, n):
    """σ=0 (i=0) opaque; larger σ more transparent, floored at ALPHA_MIN."""
    return 1.0 - (1.0 - ALPHA_MIN) * (i / max(1, n - 1))


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

    # shared LMMF mixture + empirical ω-sample, refit per trial → show one trial's fit (MIX_TRIAL)
    mix_all = df[df.quantity == "mixture"]
    mt = MIX_TRIAL if (mix_all["trial"] == MIX_TRIAL).any() else int(mix_all["trial"].dropna().min())
    g = mix_all[mix_all["trial"] == mt].sort_values("Omega")
    mixture = (g["w"].to_numpy(), g["Omega"].to_numpy(), g["Delta"].to_numpy())
    M = len(g)
    om = df[df.quantity == "omega"]
    omega_samp = om[om["trial"] == mt].sort_values("idx")["value"].to_numpy()

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
    return sigmas, cs, trials, models, mixture, M, mt, omega_samp, omega_max, R_mic, R_mf, mats, corr


def rmse(R_mic, R_mf, model, s, c, tr):
    """Spectral RMSE (Fourier-amplitude) between micro and a given MF model's R(t) for one trial —
    robust to small phase shifts between the models."""
    _, rm = R_mic[(s, c, tr)]
    _, rf = R_mf[(model, s, c, tr)]
    return spectral_rmse(rm, rf)


def rmse_stats(R_mic, R_mf, model, s, c, trials):
    vals = np.array([rmse(R_mic, R_mf, model, s, c, tr) for tr in trials], float)
    vals = vals[np.isfinite(vals)]
    return (float(vals.mean()), float(vals.std())) if vals.size else (np.nan, np.nan)


def representative_trial(R_mic, R_mf, s, c, trials, model="mean"):
    order = sorted(trials, key=lambda tr: rmse(R_mic, R_mf, model, s, c, tr))
    return order[len(order) // 2]


def _example_pairs(sigmas, cs):
    """Three (σ, c) example rows: (1) c=0 at intermediate σ, (2) the largest c>0 at the same σ,
    (3) that same c at the largest σ."""
    if EXAMPLES is not None:
        return [tuple(e) for e in EXAMPLES]
    s_int = sigmas[len(sigmas) // 2]                       # intermediate σ
    s_max = sigmas[-1]                                     # largest σ
    c0 = min(cs, key=abs)                                  # c closest to 0
    c_pos = max(cs)                                        # largest c > 0
    return [(s_int, c0), (s_int, c_pos), (s_max, c_pos)]


# ════════════════════════════════════════════════════════════════════════════
#  row-1 panels
# ════════════════════════════════════════════════════════════════════════════
def _plot_rmse(ax, sigmas, cs, trials, models, R_mic, R_mf):
    """RMSE vs strength slope c — colour = MF model, line transparency = noise σ."""
    cx = np.asarray(cs)
    ytop = 0.0
    for mo in models:
        col = MODEL_STYLE[mo][0]
        for si, s in enumerate(sigmas):
            ms = np.array([rmse_stats(R_mic, R_mf, mo, s, c, trials) for c in cs])
            ax.errorbar(cx, ms[:, 0], yerr=ms[:, 1], color=col, alpha=_alpha_for_sigma(si, len(sigmas)),
                        marker="o", ms=2.2, lw=0.9, capsize=1.2, elinewidth=0.5)
            ytop = max(ytop, np.nanmax(ms[:, 0] + ms[:, 1]))
    ax.set_xlabel(r"coupling coefficient $c$", labelpad=1)
    ax.set_ylabel(r"$R(t)$ spectral RMSE", labelpad=2)
    ax.margins(x=0.08)
    ax.set_ylim(0.0, 1.2 * ytop)             # modest headroom for the (upper-left) model legend

    mod_handles = [Line2D([0], [0], color=MODEL_STYLE[mo][0], lw=1.4, label=MODEL_STYLE[mo][2])
                   for mo in models]
    sig_handles = [Line2D([0], [0], color="0.25", lw=1.4, alpha=_alpha_for_sigma(si, len(sigmas)),
                          label=f"{s:g}") for si, s in enumerate(sigmas)]
    leg1 = ax.legend(handles=mod_handles, loc="upper left", fontsize=5.6, handlelength=1.4,
                     borderaxespad=0.3, labelspacing=0.25)
    ax.add_artist(leg1)
    # σ legend ABOVE the axes box (horizontal, centred), title on top
    ax.legend(handles=sig_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), title=r"$\sigma$",
              fontsize=5.4, title_fontsize=5.6, handlelength=1.2, borderaxespad=0.2, labelspacing=0.2,
              columnspacing=1.0, ncol=len(sigmas))


def _plot_mixture(ax, mixture, omega_samp, omega_max, M, mt):
    """Shared Lorentzian-mixture fit over its empirical ω-sample (skardal_benchmark style)."""
    w, Om, De = mixture
    xg = np.linspace(-1.2 * omega_max, 1.2 * omega_max, 500)
    comps = w[None, :] * (De[None, :] / np.pi) / ((xg[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    if omega_samp is not None and omega_samp.size:
        ax.hist(omega_samp, bins=20, range=(-1.2 * omega_max, 1.2 * omega_max), density=True,
                color="0.85", edgecolor="none", zorder=0)
    for k in range(len(w)):
        ax.plot(xg, comps[:, k], color="#2e6f95", lw=0.4, alpha=0.5, zorder=2)
    dens = comps.sum(axis=1)
    ax.plot(xg, dens, color="#222222", lw=1.2, zorder=3, label="fit")
    uni = 1.0 / (2 * omega_max)
    ax.plot([-omega_max, -omega_max, omega_max, omega_max], [0, uni, uni, 0],
            color="0.4", ls="--", lw=0.8, zorder=4, label=r"$g(\omega)$")
    ax.set_xlim(xg[0], xg[-1])
    ax.set_ylim(0.0, 1.32 * max(dens.max(), uni))    # headroom so the legend clears the peaks
    ax.set_xlabel(r"$\omega$", labelpad=1); ax.set_ylabel(r"$\rho(\omega)$", labelpad=2)
    ax.set_title(rf"shared LMMF fit ($M{{=}}{M}$)", fontsize=6.5, pad=2)
    ax.legend(loc="upper center", fontsize=5.4, handlelength=1.2, borderaxespad=0.2, ncol=2,
              columnspacing=1.0)


# ════════════════════════════════════════════════════════════════════════════
#  the single combined figure
# ════════════════════════════════════════════════════════════════════════════
def make_figure(sigmas, cs, trials, models, mixture, M, mt, omega_samp, omega_max,
                R_mic, R_mf, mats, corr):
    ex_pairs = _example_pairs(sigmas, cs)
    n_ex = len(ex_pairs)

    fig = plt.figure(figsize=(3.4, 1.45 + 0.95 * n_ex), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.05)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.0, 0.86 * n_ex], hspace=-0.06)

    # ── row 1 : RMSE line plot  |  representative mixture fit ─────────────────
    gst = sf_top.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax_rmse = sf_top.add_subplot(gst[0, 0])
    ax_mix = sf_top.add_subplot(gst[0, 1])
    _plot_rmse(ax_rmse, sigmas, cs, trials, models, R_mic, R_mf)
    _plot_mixture(ax_mix, mixture, omega_samp, omega_max, M, mt)
    _panel_label(ax_rmse, "a", dx=-22)
    _panel_label(ax_mix, "b", dx=-20)

    # ── rows 2..(1+n_ex) : R(t) comparison (wide)  +  square coupling matrix + colourbar ──
    gsb = sf_bot.add_gridspec(n_ex, 3, width_ratios=[2.25, 1.0, 0.07])
    for ri, (s_ex, c) in enumerate(ex_pairs):
        ax_dyn = sf_bot.add_subplot(gsb[ri, 0])
        ax_mat = sf_bot.add_subplot(gsb[ri, 1])
        cax = sf_bot.add_subplot(gsb[ri, 2])
        tr = representative_trial(R_mic, R_mf, s_ex, c, trials)

        t, Rm = R_mic[(s_ex, c, tr)]
        ax_dyn.plot(t, Rm, color=C_MICRO, lw=1.0, alpha=MICRO_ALPHA, label="micro")  # micro first, underneath
        for mo in models:
            col, ls, _ = MODEL_STYLE[mo]
            tf, Rf = R_mf[(mo, s_ex, c, tr)]
            ax_dyn.plot(tf, Rf, color=col, lw=0.9, ls=ls)
        ax_dyn.set_xlim(t[0], t[-1]); ax_dyn.set_ylim(-0.02, 1.02); ax_dyn.set_yticks([0, 0.5, 1.0])
        ax_dyn.set_ylabel(r"$R(t)$", labelpad=2)
        # parameter values go in the panel title (frees the plotting area for the legend)
        ax_dyn.set_title(rf"$\sigma={s_ex:g}$,  $c={c:g}$  ($c_{{\rm real}}{{=}}{corr.get((s_ex, c, tr), c):+.2f}$)",
                         fontsize=6.0, pad=2)
        if ri == 0:
            trace_handles = [Line2D([0], [0], color=C_MICRO, lw=1.0, alpha=MICRO_ALPHA, label="micro")]
            trace_handles += [Line2D([0], [0], color=MODEL_STYLE[mo][0], lw=0.9,
                                     ls=MODEL_STYLE[mo][1], label=MODEL_STYLE[mo][2]) for mo in models]
            ax_dyn.legend(handles=trace_handles, loc="upper right", fontsize=5.0, handlelength=1.3,
                          borderaxespad=0.3, labelspacing=0.2)
        if ri == n_ex - 1:
            ax_dyn.set_xlabel(r"$t$", labelpad=1)
        else:
            ax_dyn.set_xticklabels([])
        _panel_label(ax_dyn, "cde"[ri] if ri < 3 else "", dx=-22)

        Mx = mats[(s_ex, c, tr)]
        im = ax_mat.imshow(Mx, origin="lower", aspect="equal", cmap=MATRIX_CMAP,
                           vmin=np.nanmin(Mx), vmax=np.nanmax(Mx), interpolation="nearest")
        ax_mat.set_xticks([]); ax_mat.set_yticks([])
        if ri == 0:
            ax_mat.set_title(r"$A_{ij}$", fontsize=6.5, pad=2)
        cb = fig.colorbar(im, cax=cax)           # dedicated gridspec column → not clipped
        cb.ax.tick_params(labelsize=4.5, pad=0.6, length=1.5)

    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {OUT}.svg / .png")


def main():
    (sigmas, cs, trials, models, mixture, M, mt, omega_samp, omega_max,
     R_mic, R_mf, mats, corr) = load(CSV)
    set_prl_style()
    make_figure(sigmas, cs, trials, models, mixture, M, mt, omega_samp, omega_max,
                R_mic, R_mf, mats, corr)


if __name__ == "__main__":
    main()
