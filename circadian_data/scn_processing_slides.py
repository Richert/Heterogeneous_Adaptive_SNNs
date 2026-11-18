r"""
Slide figures: from raw PER2::LUC traces to R(t) and intrinsic frequencies (epoch-wise peak pipeline)
====================================================================================================

Figure 1  scn<i>_processing_steps   one example cell, step by step, and the population result:
    (a) provided trace X and the epoch-wise polynomial baseline (pre-TTX | TTX | washout fitted separately)
    (b) detrended + normalised trace (per epoch), Gaussian-smoothed, with the local-threshold peaks
    (c) peak phase θ(t) mod 2π (linear between peaks, cycles counted across gaps)
    (d) cos θ_i(t) of all cells, sorted by lobe and intrinsic frequency
    (e) R(t) of the slice and both lobes, with the measurement ceiling c(t) and the masked ±24 h windows
Figure 2  scn<i>_frequency_estimation   how the intrinsic frequencies are estimated:
    (a) phase relative to a 24-h clock during TTX for example cells (5–95 % frequency quantiles); the slope of the
        least-squares line over the TTX epoch without its first 24 h is ω_i
    (b) histogram of ω_i per lobe (stacked) with the pooled noise-floor mixture and the single Lorentzian

Uses the cached targets (scn_targets) and pooled mixtures (scn_pycobi_bifurcation); no simulation.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_processing_slides.py [--slice 1] [--out DIR]
Output: data_paths.MPMF/circadian_data/writeup/ (or --out)
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scn_common as S
import scn_epoch_processing as EP
import scn_targets as TG
import scn_pycobi_bifurcation as B

C_A, C_B = "#2f7fd8", "#ef6a35"          # lobe colours (as in scn_overview)
C_TTX = "0.9"
CPD = 24 / (2 * np.pi)                   # rad/h -> cycles/day
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})


def shade(ax, g, days=True, mask=False):
    f = 1 / 24 if days else 1
    ax.axvspan(g["ttx"] * f, g["wash"] * f, color=C_TTX, lw=0, zorder=0)
    if mask:
        for tc in (g["ttx"], g["wash"]):
            ax.axvspan((tc - g["mask_h"]) * f, (tc + g["mask_h"]) * f, facecolor="none", edgecolor="0.6",
                       hatch="////", lw=0, zorder=0)


def example_cell(g, X):
    """Cell (index into the target set) with the median TTX-epoch amplitude relative to its pre-TTX amplitude."""
    a, b = g["epochs"][1]; p0, p1 = g["epochs"][0]
    Xc = X[:, g["cells"]]
    amp = lambda seg: np.subtract(*np.percentile(seg, [95, 5], axis=0))
    rel = amp(Xc[a:b]) / amp(Xc[p0:p1])
    return int(np.argsort(rel)[rel.size // 2])


def steps_figure(i, out):
    g = TG.load(i); X, _ = S.load(i)
    T = int(g["T"]); t = np.arange(T); td = t / 24
    k = example_cell(g, X); x = X[:, g["cells"][k]]
    bnd = EP.bounds(T, int(g["ttx"]), int(g["wash"]))
    base = np.empty(T); y = np.empty(T); peaks = []
    for a, b in bnd:
        seg = x[a:b, None]
        bl = EP.baseline(seg, TG.PIPE["baseline_kind"])
        yn = EP.normalise(seg - bl, TG.PIPE["norm"])
        base[a:b] = bl[:, 0]; y[a:b] = yn[:, 0]
        peaks.append(a + EP.segment_peaks(yn[:, 0]))
    peaks = np.concatenate(peaks)
    ys = np.concatenate([gaussian_filter1d(y[a:b], EP.SMOOTH_H) for a, b in bnd])
    th = g["PH"][:, k]

    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(3, 2, width_ratios=(1, 1.05), hspace=0.38, wspace=0.22,
                          left=0.07, right=0.97, top=0.93, bottom=0.1)
    ax = [fig.add_subplot(gs[r, 0]) for r in range(3)]
    for a_ in ax[1:]:
        a_.sharex(ax[0])
    gr = gs[:, 1].subgridspec(2, 1, hspace=0.3, height_ratios=(1.15, 1))
    axr = [fig.add_subplot(gr[0]), fig.add_subplot(gr[1])]
    axr[1].sharex(axr[0])

    shade(ax[0], g); ax[0].plot(td, x, color="0.15", lw=0.9, label="provided trace")
    ax[0].plot(td, base, color="#e08a00", lw=1.6, label="epoch-wise baseline")
    for a, b in bnd[1:]:
        ax[0].axvline(a / 24, color="0.5", lw=0.6, ls=":")
    ax[0].set_ylabel("PER2::LUC (a.u.)"); ax[0].legend(loc="upper right", frameon=False, ncol=2)
    ax[0].set_title(f"(a) one cell of SCN{i}: raw trace and baseline", loc="left")

    shade(ax[1], g); ax[1].plot(td, y, color="0.7", lw=0.7, label="detrended + normalised")
    ax[1].plot(td, ys, color="0.15", lw=1.0, label="smoothed")
    ax[1].plot(peaks / 24, np.interp(peaks, t, ys), "v", color="#b2182b", ms=5, label="peaks")
    ax[1].set_ylabel("normalised"); ax[1].legend(loc="upper right", frameon=False, ncol=3)
    ax[1].set_title("(b) per-epoch detrending, normalisation and peak detection", loc="left")

    shade(ax[2], g)
    ok = np.isfinite(th)
    w = np.where(ok, np.mod(th, 2 * np.pi), np.nan); w[1:][np.abs(np.diff(w)) > np.pi] = np.nan
    ax[2].plot(td, w, color="0.15", lw=1.0)
    ax[2].plot(peaks / 24, np.zeros_like(peaks), "v", color="#b2182b", ms=4)
    ax[2].set_yticks([0, np.pi, 2 * np.pi]); ax[2].set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax[2].set_ylabel(r"phase $\theta$"); ax[2].set_xlabel("time (days)")
    ax[2].set_title("(c) phase: linear between peaks", loc="left")
    for a_ in ax[:2]:
        plt.setp(a_.get_xticklabels(), visible=False)
    ax[0].set_xlim(0, T / 24)

    # population: phase relative to the slice mean phase, sorted by lobe, then frequency
    lobe, om = g["lobe"], g["om"]
    order = np.lexsort((om, lobe))
    Psi = np.angle(TG.mean_field(g["PH"]))
    P = np.angle(np.exp(1j * (g["PH"][:, order] - Psi[:, None]))).T
    im = axr[0].imshow(P, aspect="auto", cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, interpolation="nearest",
                       extent=(0, T / 24, P.shape[0], 0))
    nA = int((lobe == 0).sum())
    axr[0].axhline(nA, color="k", lw=0.8)
    for tc in (g["ttx"], g["wash"]):
        axr[0].axvline(tc / 24, color="k", lw=0.8, ls="--")
    axr[0].set_ylabel("cell (lobe A | B,\nsorted by frequency)")
    axr[0].set_yticks([nA / 2, nA + (P.shape[0] - nA) / 2]); axr[0].set_yticklabels(["A", "B"])
    for tl, c in zip(axr[0].get_yticklabels(), (C_A, C_B)):
        tl.set_color(c); tl.set_fontweight("bold")
    cb = fig.colorbar(im, ax=axr[0], pad=0.01, fraction=0.04, ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"]); cb.set_label(r"$\theta_i - \Psi$")
    axr[0].set_title(f"(d) all {P.shape[0]} cells, phase relative to the slice mean phase $\\Psi$", loc="left")
    plt.setp(axr[0].get_xticklabels(), visible=False)

    shade(axr[1], g, mask=True)
    axr[1].plot(td, g["c"], color="0.4", lw=0.9, ls=":", label=r"ceiling $c(t)$")
    axr[1].plot(td, g["RA"], color=C_A, lw=1.0, label="lobe A")
    axr[1].plot(td, g["RB"], color=C_B, lw=1.0, label="lobe B")
    axr[1].plot(td, g["R"], color="k", lw=1.6, label="slice")
    axr[1].set_ylim(0, 1.02); axr[1].set_ylabel(r"coherence $R(t)$"); axr[1].set_xlabel("time (days)")
    axr[1].legend(loc="lower left", frameon=False, ncol=2)
    axr[1].set_title(r"(e) $R(t) = |\langle e^{i\theta_j}\rangle_j|$ (hatched: masked $\pm$24 h)", loc="left")
    axr[1].set_xlim(0, T / 24)
    # align the raster width with the R(t) axis (the colorbar takes space from the raster only)
    fig.canvas.draw()
    p0, p1 = axr[0].get_position(), axr[1].get_position()
    axr[1].set_position([p1.x0, p1.y0, p0.width, p1.height])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"scn{i}_processing_steps.{ext}"), dpi=200)
    plt.close(fig)
    print(f"[steps] SCN{i}: example cell {g['cells'][k]} (target index {k}), {peaks.size} peaks")


def frequency_figure(i, out):
    g = TG.load(i)
    T = int(g["T"]); t = np.arange(T)
    a, b = g["epochs"][1]                       # TTX epoch without its first 24 h (as in scn_targets)
    om = g["om"]; lobe = g["lobe"]
    q = np.quantile(om, (0.05, 0.25, 0.5, 0.75, 0.95))
    ex = [int(np.argmin(np.abs(om - v))) for v in q]
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9), gridspec_kw=dict(width_ratios=(1.15, 1), wspace=0.25,
                                                                         left=0.07, right=0.985, top=0.9,
                                                                         bottom=0.15))
    t0 = int(g["ttx"])
    sl = slice(t0 - 24, int(g["wash"]) + 24)
    ax[0].axvspan(0, (a - t0) / 24, color="0.85", lw=0, zorder=0)
    ax[0].text((a - t0) / 48, 0.98, "excluded\n(first 24 h)", transform=ax[0].get_xaxis_transform(), ha="center",
               va="top", fontsize=8, color="0.35")
    labs = []
    for n, c in enumerate(ex):
        th = g["PH"][:, c]
        rel = (th - 2 * np.pi * t / 24) / (2 * np.pi)          # cycles relative to a 24-h clock
        ref = rel[t0] if np.isfinite(rel[t0]) else np.nanmean(rel[t0:t0 + 24])
        col = cmap(n / (len(ex) - 1) * 0.9)
        ax[0].plot((t[sl] - t0) / 24, rel[sl] - ref, color=col, lw=1.4)
        okk = np.isfinite(th[a:b])
        p = np.polyfit(t[a:b][okk], rel[a:b][okk] - ref, 1)
        tt = np.array([a, b])
        ax[0].plot((tt - t0) / 24, np.polyval(p, tt), color=col, lw=0.9, ls="--")
        labs.append([np.polyval(p, b), f"{om[c] * CPD:.3f} c/d", col])
    ys_ = np.array([l[0] for l in labs]); o = np.argsort(ys_); gap = 0.07
    for k_ in range(1, o.size):
        ys_[o[k_]] = max(ys_[o[k_]], ys_[o[k_ - 1]] + gap)
    for y_, (_, txt, col) in zip(ys_, labs):
        ax[0].text((b - t0) / 24 + 0.12, y_, txt, color=col, va="center", fontsize=8.5)
    ax[0].axvline((int(g["wash"]) - t0) / 24, color="k", lw=0.8, ls="--")
    ax[0].set_xlim(-1, (int(g["wash"]) - t0) / 24 + 1.6)
    ax[0].set_xlabel("days since TTX application"); ax[0].set_ylabel("phase − 24-h clock (cycles)")
    ax[0].set_title(r"(a) $\omega_i$ = least-squares phase slope under TTX", loc="left")

    f = om * CPD
    bins = np.linspace(0.75, 1.25, 41)
    ax[1].hist([f[lobe == 0], f[lobe == 1]], bins=bins, stacked=True, color=(C_A, C_B), density=True,
               label=("lobe A", "lobe B"), alpha=0.85)
    x = np.linspace(0.75, 1.25, 800) / CPD
    for kind, col, ls, lab in (("pfloor", "k", "-", "noise-floor mixture"), ("psingle", "0.35", "--",
                                                                             "single Lorentzian")):
        mod = B.pooled_mixture(i, kind)
        w = np.asarray(mod["w"], float); w = w / w.sum()
        Om = np.asarray(mod["Om"], float); De = np.asarray(mod["De"], float)
        pdf = np.sum(w[:, None] * De[:, None] / np.pi / ((x[None] - Om[:, None]) ** 2 + De[:, None] ** 2), 0)
        ax[1].plot(x * CPD, pdf / CPD, color=col, ls=ls, lw=1.8, label=f"{lab} (M = {w.size})")
    ax[1].set_xlim(0.75, 1.25); ax[1].set_xlabel("intrinsic frequency (cycles/day)"); ax[1].set_ylabel("density")
    ax[1].legend(frameon=False, loc="upper right")
    ax[1].set_title(f"(b) SCN{i}: {f.size} cells, Lorentzian-mixture fit", loc="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"scn{i}_frequency_estimation.{ext}"), dpi=200)
    plt.close(fig)
    print(f"[freq] SCN{i}: example frequencies {np.round(om[ex] * CPD, 3).tolist()} c/d")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    i = arg("--slice", 1, int)
    out = arg("--out", os.path.join(S.OUT, "writeup"))
    os.makedirs(out, exist_ok=True)
    steps_figure(i, out)
    frequency_figure(i, out)
