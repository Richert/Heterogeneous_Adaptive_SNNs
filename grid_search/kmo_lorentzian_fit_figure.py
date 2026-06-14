r"""
Kuramoto micro vs. Lorentzian-ensemble mean-field — fit-quality figure
======================================================================

Loads the sweep results written by ``kmo_lorentzian_fit_sweep.py`` and

  1. computes, for every (λ, M_max) sweep point, the RMSE between the *Fourier
     amplitude spectra* of the average phase-coherence dynamics R(t) of the
     microscopic Kuramoto network and the ensemble mean field, and
  2. renders a Physical-Review-Letters double-column figure (2 rows × 4 columns):

       column 1 (both rows): heatmap of the spectral RMSE over the whole sweep
       columns 2-4: three representative sweep points (best / median / worst RMSE);
         each column shows the micro-vs-MF frequency distribution (top, Gaussian-
         mixture-demo style) and the R(t) dynamics (bottom).

The spectral RMSE uses the amplitude spectrum |FFT(R)| (phase-independent), so it
measures the mismatch of the coherence level + oscillation content between the two
models. Reads the tidy CSV (discriminated by the `quantity` column).

Run with any numpy/scipy/pandas/matplotlib env, e.g.:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_lorentzian_fit_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

CSV = "/home/rgast/data/mpmf_simulations/kmo_lorentzian_sweep.csv"
OUT = "/home/rgast/data/mpmf_simulations/kmo_lorentzian_fit_figure"

C_MICRO = "0.2"
C_MF = "#c1121f"
C_COMP = "#2e6f95"
HEATMAP_CMAP = "Reds"     # colormap for the spectral-RMSE heatmap

# ════════════════════════════════════════════════════════════════════════════
#  PRL single-column style
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
            w=mg["w"].to_numpy(), Omega=mg["Omega"].to_numpy(), Delta=mg["Delta"].to_numpy())
    return omega, t_mic, R_mic, points


def spectral_rmse(R_ref, R):
    """RMSE between the Fourier amplitude spectra of two coherence time series."""
    n = min(len(R_ref), len(R))
    Fa = np.abs(np.fft.rfft(R_ref[:n])) / n
    Fb = np.abs(np.fft.rfft(R[:n])) / n
    return float(np.sqrt(np.mean((Fb - Fa) ** 2)))


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

    # ── figure: 4×2, single column ──────────────────────────────────────────
    set_prl_style()
    # Double-column; height reduced (the old bottom legend strip is gone). The
    # shared legend now lives in a strip below the heatmap within column 1, so the
    # example columns (distribution + dynamics) keep their full, unchanged height.
    fig = plt.figure(figsize=(7.0, 2.0), layout="constrained")   # PRL double column
    # minimal but non-overlapping whitespace (pads in inches, *space as fractions)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    gs = fig.add_gridspec(2, 4)

    # column 1: heatmap on top, shared legend in the freed strip beneath it
    col1 = gs[0:2, 0].subgridspec(2, 1, height_ratios=[6, 1], hspace=0.04)
    axh = fig.add_subplot(col1[0])
    ax_leg = fig.add_subplot(col1[1]); ax_leg.axis("off")
    im = axh.imshow(RMSE, origin="lower", aspect="auto", cmap=HEATMAP_CMAP)
    axh.set_xticks(range(len(lams)))
    axh.set_xticklabels([f"{b:g}" for b in lams])
    axh.set_yticks(range(len(Mmaxs)))
    axh.set_yticklabels([str(m) for m in Mmaxs])
    axh.set_xlabel(r"penalty $\lambda$", labelpad=1)
    axh.set_ylabel(r"max. ensembles $M_{\max}$", labelpad=2)
    axh.set_title("(a) spectral RMSE", fontsize=7, pad=3)
    _stroke = [pe.withStroke(linewidth=1.0, foreground="black")]
    # M* per cell: plain fill (no stroke), black on light cells / white on dark cells,
    # chosen from the cell's background luminance.
    for i in range(len(Mmaxs)):
        for j in range(len(lams)):
            if np.isnan(RMSE[i, j]):
                continue
            r, g, b, _ = im.cmap(im.norm(RMSE[i, j]))
            lum = 0.299 * r + 0.587 * g + 0.114 * b      # perceived luminance
            axh.text(j, i, f"{Mstar[i, j]}", ha="center", va="center",
                     fontsize=6, color="black" if lum > 0.5 else "white")
    cb = fig.colorbar(im, ax=axh, fraction=0.045, pad=0.012)
    cb.ax.tick_params(labelsize=5.5, pad=1.0)

    # mark the three chosen points on the heatmap (offset to the cell corner so
    # they don't sit on the M* annotation)
    for (lm, Mmax), lab in zip(chosen, "bcd"):
        i, j = Mmaxs.index(Mmax), lams.index(lm)
        axh.text(j + 0.30, i + 0.30, lab, ha="center", va="center", fontsize=6.5,
                 fontweight="bold", color="white", path_effects=_stroke,
                 bbox=dict(boxstyle="circle,pad=0.05", fc="0.1", ec="white", lw=0.7))

    # remaining three columns: representative examples (dist on top, R(t) below)
    block_cells = [(gs[0, 1], gs[1, 1]), (gs[0, 2], gs[1, 2]), (gs[0, 3], gs[1, 3])]
    for (lm, Mmax), (top_gs, bot_gs), lab, tag in zip(chosen, block_cells, "bcd", labels):
        p = points[(lm, Mmax)]
        rm = rmse_pt[(lm, Mmax)]
        ax_d = fig.add_subplot(top_gs)
        ax_t = fig.add_subplot(bot_gs)
        plot_distribution(ax_d, omega, p, gx)
        plot_dynamics(ax_t, t_mic, R_mic, p)
        ax_d.set_title(f"({lab}) {tag}: $M^*={p['Mstar']}$, $\\lambda={lm:g}$\n"
                       f"RMSE$={rm:.3f}$", fontsize=6.0, pad=2)

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

    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"[saved] {OUT}.pdf / .png")
    print("chosen examples (λ, M_max) -> RMSE:",
          [(c, round(rmse_pt[c], 4)) for c in chosen])


if __name__ == "__main__":
    main()
