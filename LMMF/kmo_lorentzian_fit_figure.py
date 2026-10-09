r"""
Kuramoto micro vs. Lorentzian-ensemble mean-field — fit-quality figure
======================================================================

Loads the sweep results written by ``kmo_lorentzian_fit_sweep.py`` and

  1. computes, for every (λ, M_max) sweep point, the RMSE between the *Fourier
     amplitude spectra* of the average phase-coherence dynamics R(t) of the
     microscopic Kuramoto network and the ensemble mean field, and
  2. renders a Physical-Review-Letters double-column figure (2 rows × 4 columns):

       column 1: (a) heatmap of the spectral RMSE over the (λ, M_max) sweep, and
                 (b) heatmap of the mean selected M* over resampled data sets as a function
                     of λ and the sample size N (kmo_lorentzian_M_stability.py), with ranges
       columns 2-4: three representative sweep points (best / median / worst RMSE);
         each column shows the micro-vs-MF frequency distribution (top, Gaussian-
         mixture-demo style) and the R(t) dynamics (bottom).

The spectral RMSE uses the amplitude spectrum |FFT(R)| (phase-independent), so it
measures the mismatch of the coherence level + oscillation content between the two
models. Reads the tidy CSV (discriminated by the `quantity` column).

Run with any numpy/scipy/pandas/matplotlib env, e.g.:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_lorentzian_fit_figure.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
from prl_style import set_prl_style as _set_prl_style
from prl_style import panel_label
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

CSV = dp.mpmf("kmo_lorentzian_sweep.csv")
STAB_CSV = dp.mpmf("kmo_lorentzian_M_stability.csv")   # kmo_lorentzian_M_stability.py
OUT = dp.mpmf("kmo_lorentzian_fit_figure")

C_MICRO = "0.2"
C_MF = "#c1121f"
C_COMP = "#2e6f95"
HEATMAP_CMAP = "Reds"     # colormap for the spectral-RMSE heatmap
STAB_CMAP = "Blues"       # colormap for the M*-stability heatmap

# ════════════════════════════════════════════════════════════════════════════
#  PRL single-column style
# ════════════════════════════════════════════════════════════════════════════
set_prl_style = functools.partial(_set_prl_style, "prl",
                                 **{"xtick.major.size": 2.0, "ytick.major.size": 2.0})


# ════════════════════════════════════════════════════════════════════════════
#  load + analysis
# ════════════════════════════════════════════════════════════════════════════
def load(csv):
    df = pd.read_csv(csv)
    omega = df.loc[df.quantity == "omega", "value"].to_numpy()
    rm = df[df.quantity == "R_micro"].sort_values("time")
    t_mic, R_mic = rm["time"].to_numpy(), rm["value"].to_numpy()

    mf, mix = df[df.quantity == "R_mf"], df[df.quantity == "mixture"]
    points = {}
    for (lm, Mmax), g in mf.groupby(["lambda", "M_max"]):
        g = g.sort_values("time")
        mg = mix[(np.isclose(mix["lambda"], lm)) & (mix["M_max"] == Mmax)].sort_values("idx")
        points[(float(lm), int(Mmax))] = dict(
            t=g["time"].to_numpy(), R=g["value"].to_numpy(),
            Mstar=int(g["M_star"].iloc[0]),
            # λ in units of the noise floor 1/(6N) (sweeps after the floor-rule change)
            kappa=(float(g["kappa"].iloc[0]) if "kappa" in g and g["kappa"].notna().any()
                   else None),
            w=mg["w"].to_numpy(), Omega=mg["Omega"].to_numpy(), Delta=mg["Delta"].to_numpy())
    return omega, t_mic, R_mic, points


def spectral_rmse(R_ref, R):
    """RMSE between the Fourier amplitude spectra of two coherence time series."""
    n = min(len(R_ref), len(R))
    Fa = np.abs(np.fft.rfft(R_ref[:n])) / n
    Fb = np.abs(np.fft.rfft(R[:n])) / n
    return float(np.sqrt(np.mean((Fb - Fa) ** 2)))


def _sci(x):
    """1e-05 -> $10^{-5}$ (exact powers of ten), else %g."""
    e = np.log10(x)
    return f"$10^{{{int(round(e))}}}$" if np.isclose(e, round(e)) else f"{x:g}"


def lorentzian_pdf(x, w, Om, De):
    return (w[None, :] * (De[None, :] / np.pi)
            / ((x[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2))


# ════════════════════════════════════════════════════════════════════════════
#  panels
# ════════════════════════════════════════════════════════════════════════════
def plot_distribution(ax, omega, p, gx):
    ax.hist(omega, bins=60, range=(gx[0], gx[-1]), density=True, color="0.82",
            label="micro", zorder=0)
    comps = lorentzian_pdf(gx, p["w"], p["Omega"], p["Delta"])
    for k in range(p["Mstar"]):
        ax.plot(gx, comps[:, k], lw=0.6, color=C_COMP, alpha=0.7, zorder=2)
    ax.plot(gx, comps.sum(axis=1), lw=1.0, color=C_MF, label="MF mixture", zorder=3)
    ax.set_xlim(gx[0], gx[-1])
    ax.set_yticks([])
    ax.set_xlabel(r"$\omega$", labelpad=1)
    ax.set_ylabel(r"$\rho(\omega)$", labelpad=2)


def plot_dynamics(ax, t_mic, R_mic, p):
    ax.plot(t_mic, R_mic, color=C_MICRO, lw=0.9, label="micro")
    ax.plot(p["t"], p["R"], color=C_MF, lw=0.9, ls="--", label="MF")
    ax.set_xlim(t_mic[0], t_mic[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"time $t$", labelpad=1)
    ax.set_ylabel(r"$R(t)$", labelpad=2)


_panel_label = functools.partial(panel_label, dx=-14, dy=4)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    omega, t_mic, R_mic, points = load(CSV)

    # spectral RMSE over the (M_max, λ) sweep grid
    Mmaxs = sorted({k[1] for k in points})
    lams = sorted({k[0] for k in points})
    RMSE = np.full((len(Mmaxs), len(lams)), np.nan)
    Mstar = np.zeros_like(RMSE, dtype=int)
    rmse_pt = {}
    for (lm, Mmax), p in points.items():
        i, j = Mmaxs.index(Mmax), lams.index(lm)
        RMSE[i, j] = rmse_pt[(lm, Mmax)] = spectral_rmse(R_mic, p["R"])
        Mstar[i, j] = p["Mstar"]

    # three representative sweep points: best / median / worst spectral RMSE
    order = sorted(points, key=lambda k: rmse_pt[k])
    chosen = [order[0], order[len(order) // 2], order[-1]]
    labels = ["best", "median", "worst"]

    gx = np.linspace(np.percentile(omega, 0.5), np.percentile(omega, 99.5), 700)

    # ── figure: 4 columns; column 1 = two heatmaps + shared legend ──────────
    set_prl_style()
    fig = plt.figure(figsize=(7.0, 2.7), layout="constrained")   # PRL double column
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1])
    col1 = gs[0:2, 0].subgridspec(3, 1, height_ratios=[5, 5, 1.4], hspace=0.08)
    axh = fig.add_subplot(col1[0])
    axs = fig.add_subplot(col1[1])
    ax_leg = fig.add_subplot(col1[2]); ax_leg.axis("off")

    kap = {lm: next(p["kappa"] for (l2, _), p in points.items() if l2 == lm) for lm in lams}
    in_floor_units = all(k is not None for k in kap.values())

    def _lam_ticks(ax):
        ax.set_yticks(range(len(lams)))
        ax.set_yticklabels([f"{kap[b]:g}" if in_floor_units else _sci(b) for b in lams])
        ax.set_ylabel(r"penalty $\lambda$ $[1/(6N)]$" if in_floor_units else r"penalty $\lambda$",
                      labelpad=1)

    def _annotate(ax, im, A, txt, fs=5.5):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isnan(A[i, j]):
                    continue
                r, g, b, _ = im.cmap(im.norm(A[i, j]))
                lum = 0.299 * r + 0.587 * g + 0.114 * b      # perceived luminance
                ax.text(j, i, txt(i, j), ha="center", va="center", linespacing=0.9,
                        fontsize=fs, color="black" if lum > 0.5 else "white")

    # (a) spectral RMSE: rows = λ, columns = M_max; cell text = selected M*
    im = axh.imshow(RMSE.T, origin="lower", aspect="auto", cmap=HEATMAP_CMAP)
    _lam_ticks(axh)
    axh.set_xticks(range(len(Mmaxs)))
    axh.set_xticklabels([str(m) for m in Mmaxs])
    axh.set_xlabel(r"max. ensembles $M_{\max}$", labelpad=1)
    axh.set_title("spectral RMSE", fontsize=6.5, pad=2)
    _panel_label(axh, "a")
    _annotate(axh, im, RMSE.T, lambda i, j: f"{Mstar.T[i, j]}")
    cb = fig.colorbar(im, ax=axh, fraction=0.06, pad=0.02)
    cb.ax.tick_params(labelsize=5.0, pad=1.0)
    _stroke = [pe.withStroke(linewidth=1.0, foreground="black")]
    for (lm, Mmax), lab in zip(chosen, "cde"):          # mark the three example points
        i, j = lams.index(lm), Mmaxs.index(Mmax)
        axh.text(j + 0.32, i + 0.28, lab, ha="center", va="center", fontsize=6.0,
                 fontweight="bold", color="white", path_effects=_stroke,
                 bbox=dict(boxstyle="circle,pad=0.05", fc="0.1", ec="white", lw=0.7))

    # (b) sampling stability of M*: mean over resampled data sets, rows = λ, columns = N;
    #     cell text = mean (top) and observed range (bottom)
    sel = pd.read_csv(STAB_CSV)
    sel = sel[sel.quantity == "selection"]              # fits with M_max = 16
    Ns = sorted(sel.N.unique())
    MEAN = np.full((len(lams), len(Ns)), np.nan); LO = np.zeros_like(MEAN); HI = np.zeros_like(MEAN)
    for i, lm in enumerate(lams):
        for j, N in enumerate(Ns):
            g = sel[np.isclose(sel["lambda"], lm) & (sel.N == N)].M_star
            MEAN[i, j], LO[i, j], HI[i, j] = g.mean(), g.min(), g.max()
    n_seeds = int(sel.groupby(["lambda", "N"]).size().min())
    im2 = axs.imshow(MEAN, origin="lower", aspect="auto", cmap=STAB_CMAP,
                     vmin=1, vmax=np.nanmax(HI))
    _lam_ticks(axs)
    axs.set_xticks(range(len(Ns)))
    axs.set_xticklabels([f"{n / 1000:g}k" if n >= 1000 else str(n) for n in Ns])
    axs.set_xlabel(r"sample size $N$", labelpad=1)
    axs.set_title(rf"mean $M^*$ ({n_seeds} samples each)", fontsize=6.5, pad=2)
    _panel_label(axs, "b")
    _annotate(axs, im2, MEAN, lambda i, j: (f"{MEAN[i, j]:.1f}\n"
                                            + (f"{LO[i, j]:.0f}" if LO[i, j] == HI[i, j]
                                               else f"{LO[i, j]:.0f}–{HI[i, j]:.0f}")),
              fs=4.8)
    if len(omega) in Ns:                                 # outline the N used in (a) and (c-e)
        j = Ns.index(len(omega))
        axs.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(lams), fill=False, ec="0.1", lw=0.8))
    cb2 = fig.colorbar(im2, ax=axs, fraction=0.06, pad=0.02)
    cb2.ax.tick_params(labelsize=5.0, pad=1.0)

    # remaining three columns: representative examples (dist on top, R(t) below)
    block_cells = [(gs[0, 1], gs[1, 1]), (gs[0, 2], gs[1, 2]), (gs[0, 3], gs[1, 3])]
    for (lm, Mmax), (top_gs, bot_gs), lab, tag in zip(chosen, block_cells, "cde", labels):
        p = points[(lm, Mmax)]
        rm = rmse_pt[(lm, Mmax)]
        ax_d = fig.add_subplot(top_gs)
        ax_t = fig.add_subplot(bot_gs)
        plot_distribution(ax_d, omega, p, gx)
        plot_dynamics(ax_t, t_mic, R_mic, p)
        lam_txt = (f"${p['kappa']:g}/(6N)$" if p["kappa"] is not None else _sci(lm))
        ax_d.set_title(f"{tag}: $M^*={p['Mstar']}$, $\\lambda=${lam_txt}\n"
                       f"RMSE$={rm:.3f}$", fontsize=6.0, pad=2)
        _panel_label(ax_d, lab)

    # one shared legend (distribution + dynamics) in the strip below the heatmap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(fc="0.82", label="micro $\\rho(\\omega)$"),
               Line2D([0], [0], color=C_MF, lw=1.3, label="MF mixture"),
               Line2D([0], [0], color=C_MICRO, lw=0.9, label="micro $R(t)$"),
               Line2D([0], [0], color=C_MF, lw=0.9, ls="--", label="MF $R(t)$")]
    ax_leg.legend(handles=handles, loc="center", ncol=2, fontsize=5.5,
                  handlelength=1.4, columnspacing=1.0, handletextpad=0.4,
                  borderaxespad=0.0)

    fig.savefig(OUT + ".svg", bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {OUT}.svg / .png")
    print("chosen examples (λ, M_max) -> RMSE:",
          [(c, round(rmse_pt[c], 4)) for c in chosen])


if __name__ == "__main__":
    main()
