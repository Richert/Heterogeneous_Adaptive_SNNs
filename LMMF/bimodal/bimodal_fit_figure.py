r"""
Manuscript figure (PNAS two-column): LMMF fits of the bimodal Gaussian mixture across sample sizes, M_max and λ
=============================================================================================================

One ROW per sample size N (bimodal_fit_ensemble.N_LIST), five columns:
  (1) sample histogram (seed SEED_EX), true density, and two no-floor LMMF fits of that sample: few components
      (selection at (M_max, λ) = LOW) and many components (HIGH)
  (2) network R(t) along the backward K-ramp (α = 0; 100 time units per K step, K strip on top) with the bifurcation
      diagrams of the two example fits overlaid as continuous lines (K mapped onto the ramp time, each grid K at the
      centre of its window; PyCoBi, bimodal_fit_examples.py): STABLE equilibria (incl. the incoherent R = 0 below the
      onset K) and R min / max of STABLE limit cycles; unstable solutions are not shown. Markers where a drawn state
      changes stability: Hopf (square), fold (circle), torus (diamond), period doubling (X), and the Hopf bifurcation of
      the incoherent state (triangle; the critical mode has a cluster frequency, Ω_c ≈ ±1.5-2.2, for all examples).
      Light bands: ramp windows where the example's simulated attractor is not a drawn stable state (no stable
      equilibrium / limit cycle at that K, or a quasi-periodic oscillation); R range from the example's own ramp.
      The legend lists only the marker types drawn.
  (3) spectral RMSE (LMMF vs network ramp): colour = mean over the samples, text = SD
  (4) fit loss in units of the noise floor, D / (1/(6N)): colour = mean, text = SD
  (5) number of Lorentzians M*: colour = mean, text = range (min–max)
Heatmaps: rows M_max, columns λ; all fits WITHOUT a noise floor. Outlined cells: the no-floor fit equals the fit the
noise-floor rule would have returned for that sample (LM.fit floor_c = 1, λ = 0, M_max = 16) in ≥ 50 % of the samples;
if no cell qualifies (the floor fit's M* lies between grid cells, e.g. N = 10k: M* = 9), the cell whose mean M* is
closest to the floor fit's is outlined dashed.
Colour scales are shared across the rows (RMSE, D/floor: logarithmic).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_fit_figure.py [--show]
Output: /home/rgast/OneDrive/manuscripts/PRL_2026/LMMF/figures/bimodal_distributions.{png,svg,pdf}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import numpy as np
import bimodal_fit_ensemble as BE
import bimodal_fit_examples as BX
import lorentzian_mixture as LM
from kmo_lorentzian_fit_sweep import CONFIG as FIG1

MS_FIGURES = "/home/rgast/OneDrive/manuscripts/PRL_2026/LMMF/figures"
C_EX = {"low": "#7b2cbf", "high": "#2a78d6"}
C_NET, C_HIST, C_TRUE = "0.5", "0.85", "0.3"
FLOOR_FRAC = 0.5
MARKERS = {"HB": "s", "LP": "o", "TR": "D", "PD": "X"}      # bifurcation markers (Hopf of the incoherent state: "^")


def true_pdf(x):
    mu, sd, cw = (np.asarray(FIG1[k], float) for k in ("gmm_means", "gmm_stds", "gmm_weights"))
    cw = cw / cw.sum()
    return (cw[:, None] / (np.sqrt(2 * np.pi) * sd[:, None])
            * np.exp(-0.5 * ((x[None] - mu[:, None]) / sd[:, None]) ** 2)).sum(0)


def diagram_on_ramp(ax, ex, Ks_desc, Tw, col):
    """STABLE states of one example's bifurcation diagram as continuous lines on the ramp's time axis: K is mapped
    linearly onto time, each K of the ramp grid at the centre of its hold window (window i <-> Ks_desc[i]); unstable
    equilibria and unstable limit cycles are not drawn. Lines: equilibria (incl. the incoherent R = 0 below the onset
    K) and the R min / R max of stable limit cycles."""
    import matplotlib.patheffects as pe
    K_hi, dK = Ks_desc[0], Ks_desc[0] - Ks_desc[1]
    K_lo = Ks_desc[-1] - 0.5 * dK
    tK = lambda K: ((K_hi - np.asarray(K, float)) / dK + 0.5) * Tw
    inr = lambda K: (np.asarray(K) >= K_lo) & (np.asarray(K) <= K_hi + 0.5 * dK)
    kw = dict(color=col, lw=1.4, zorder=4, path_effects=[pe.withStroke(linewidth=2.3, foreground="w")])
    K_on = ex["model"]["K_onset"]
    K0 = np.linspace(K_lo, min(K_on, K_hi + 0.5 * dK), 50)
    if K0.size and K0[-1] > K_lo:
        ax.plot(tK(K0), np.zeros_like(K0), **kw)                     # stable incoherent state
    eq = ex["eq"]
    if eq is not None and eq["K"].size:
        y = np.where(eq["stable"] & inr(eq["K"]), eq["R"], np.nan)
        ax.plot(tK(eq["K"]), y, **kw)
    for lc in ex["lc"]:
        ok = lc["stable"].astype(bool) & inr(lc["K"])
        for key in ("Rmin", "Rmax"):
            ax.plot(tK(lc["K"]), np.where(ok, lc[key], np.nan), **kw)
    # ramp windows where the simulated attractor is not a drawn stable state: (i) no stable equilibrium / incoherent
    # state / limit cycle exists at that K, or (ii) the window oscillates but no stable continued cycle matches it
    # (quasi-periodic). Drawn as the simulated R range (non-stationary dynamics, from the example's own ramp)
    import bimodal_fit_examples as BX_
    rp = ex.get("ramp")

    def stable_here(K):
        if K < K_on:
            return True
        for br in ([eq] if eq is not None and eq["K"].size else []) + list(ex["lc"]):
            Kb, st = br["K"], br["stable"].astype(bool)
            if np.any(st[:-1] & st[1:] & (np.minimum(Kb[:-1], Kb[1:]) <= K) & (np.maximum(Kb[:-1], Kb[1:]) >= K)):
                return True
        return False

    band = [] if rp is None else [i for i, (K, sd, hi) in enumerate(zip(rp["K"], rp["Rsd"], rp["Rmax"]))
                                  if not stable_here(K) or (sd > BX_.OSC_MIN and not BX_.covered(ex["lc"], K, hi))]
    for i in band:
        K = rp["K"][i]
        ax.fill_between([tK(K) - 0.5 * Tw, tK(K) + 0.5 * Tw], rp["Rmin"][i], rp["Rmax"][i], color=col, alpha=0.22,
                        lw=0, zorder=3)
    # bifurcation markers where the stability of a drawn state changes, and the Hopf point of the incoherent state
    used = set()
    mk = dict(ms=4.2, mfc="w", mec=col, mew=1.0, zorder=6, ls="none")
    if K_lo <= K_on <= K_hi + 0.5 * dK:
        ax.plot(tK(K_on), 0.0, marker="^", **mk); used.add("HB0")
    for br, keys in ((eq, ("R",)), *((lc, ("Rmin", "Rmax")) for lc in ex["lc"])):
        if br is None or not br["K"].size:
            continue
        st = br["stable"].astype(bool)
        for j, b in enumerate(br["bif"]):
            if b not in MARKERS or not inr(br["K"][j]):
                continue
            lo, hi = max(j - 1, 0), min(j + 1, st.size - 1)
            if st[lo] == st[hi]:                                  # no change of stability: not drawn
                continue
            used.add(b)
            for key in keys:
                ax.plot(tK(br["K"][j]), br[key][j], marker=MARKERS[b], **mk)
    return used, bool(band)


def _vD(stats):
    v = np.log10(stats["D"][0][np.isfinite(stats["D"][0])])
    return max(abs(v.min()), abs(v.max()))


def main(show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from prl_style import set_prl_style, pnas_panel_label, PNAS_DOUBLE
    set_prl_style("pnas")
    S = np.load(os.path.join(BE.OUT, "summary.npz"))
    Ns, Mmax, lam = S["N"], S["M_max"], S["lam"]
    rmse, D, M = S["rmse"], S["D"] / S["noise_floor"][:, None, None, None], S["M"]
    same = np.nanmean(S["same_floor"].astype(float), axis=1)                   # (N, M_max, λ)
    stats = dict(rmse=(np.nanmean(rmse, 1), np.nanstd(rmse, 1)), D=(np.nanmean(D, 1), np.nanstd(D, 1)),
                 M=(np.nanmean(M, 1), np.nanmin(M, 1), np.nanmax(M, 1)))
    norms = dict(rmse=LogNorm(np.nanmin(stats["rmse"][0]), np.nanmax(stats["rmse"][0])),
                 D=LogNorm(10 ** -_vD(stats), 10 ** _vD(stats)), M=Normalize(1, BE.M_TRACE))
    cmaps = dict(rmse="Reds", D="BrBG_r", M="Purples")
    H = 1.75 * len(Ns) + 0.75
    fig = plt.figure(figsize=(PNAS_DOUBLE, H))
    gs = fig.add_gridspec(len(Ns) + 1, 5, height_ratios=[1.0] * len(Ns) + [0.3],
                          width_ratios=[1.05, 1.75, 0.82, 0.82, 0.82], left=0.06, right=0.99, top=1 - 0.25 / H,
                          bottom=0.02, wspace=0.42, hspace=0.38)
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ramp_axes = []; used_all = set(); band_any = False; closest_used = False
    ims = {}
    for r, N in enumerate(Ns):
        ex = {t: np.load(BX.out_file(N, t), allow_pickle=True).item() for t in ("low", "high")}
        d, _ = BE.load_run(N, BX.SEED_EX)
        ax = [fig.add_subplot(gs[r, j]) for j in range(5)]
        # (1) sample, true density, example fits
        a = ax[0]; x = d["omega"]; g = np.linspace(-4.0, 4.0, 1201)
        a.hist(x, bins=min(60, max(15, int(np.sqrt(x.size) * 2))), range=(-4, 4), density=True, color=C_HIST,
               edgecolor="none", zorder=1)
        a.plot(g, true_pdf(g), color=C_TRUE, lw=1.0, ls=(0, (3, 1.5)), zorder=2)
        for t in ("low", "high"):
            m = ex[t]["model"]
            a.plot(g, LM.LorentzianMixture(m["w"], m["Om"], m["De"]).pdf(g), color=C_EX[t], lw=1.0, zorder=3)
        a.legend(handles=[Line2D([], [], color=C_EX[t], lw=1.0, label=f"$M = {ex[t]['model']['M']}$")
                          for t in ("low", "high")], loc="upper right", fontsize=6, handlelength=1.2,
                 handletextpad=0.4, labelspacing=0.2, borderaxespad=0.2, frameon=False)
        hv, _ = np.histogram(x, bins=40, range=(-4, 4), density=True)
        a.set_ylim(0, 1.35 * max(true_pdf(g).max(), hv.max()))
        fM = S["floor_M"][r]
        a.text(0.5, 0.42, "noise-floor fit:\n" + (f"$M^* = {np.nanmin(fM):.0f}$" if np.nanmin(fM) == np.nanmax(fM)
                                                    else f"$M^* = {np.nanmin(fM):.0f}$–${np.nanmax(fM):.0f}$"),
               transform=a.transAxes, ha="center", va="top", fontsize=5.5, color="0.25", zorder=6,
               bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.8))
        a.set_xlim(-4, 4); a.set_yticks([]); a.set_ylabel(f"$N = {N:,}$".replace(",", "\\,") + "\n"
                                                          + r"$\rho(\omega)$", labelpad=2)
        # (2) network ramp + diagrams
        c = ax[1]; ramp_axes.append(c)
        Ks_desc = d["K"][::-1]; ns = d["R_net"].shape[-1]; Tw = (ns - 1) * float(d["dts"])
        t_ = np.concatenate([i * Tw + np.arange(ns - 1) * float(d["dts"]) for i in range(Ks_desc.size)])
        seg = lambda arr: np.concatenate([arr[ik, :-1] for ik in range(d["K"].size - 1, -1, -1)])
        c.plot(t_, seg(d["R_net"]), color=C_NET, lw=0.35, zorder=2)
        for t in ("low", "high"):
            u_, b_ = diagram_on_ramp(c, ex[t], Ks_desc, Tw, C_EX[t]); used_all |= u_; band_any |= b_
        c.set_xlim(0, t_[-1]); c.set_ylim(-0.03, 1.12); c.set_ylabel(r"$R$", labelpad=2)
        strip = c.inset_axes([0.0, 0.905, 1.0, 0.095])
        strip.imshow(Ks_desc[None, :], aspect="auto", cmap="Greys", extent=[0, t_[-1], 0, 1],
                     norm=Normalize(-0.4 * Ks_desc.max(), 1.25 * Ks_desc.max()), interpolation="nearest")
        for Kl in (2.5, 1.5, 0.5):
            i = int(np.argmin(np.abs(Ks_desc - Kl)))
            strip.text((i + 0.5) * Tw, 0.5, rf"$K = {Kl:g}$", ha="center", va="center", fontsize=5.5,
                       color="w" if Kl > 1.6 else "k")
        for Kb in (2.0, 1.0):
            strip.axvline(int(np.argmin(np.abs(Ks_desc - Kb))) * Tw, color="0.5", lw=0.4)
        strip.set_xlim(0, t_[-1]); strip.set_xticks([]); strip.set_yticks([])
        for sp in strip.spines.values():
            sp.set_linewidth(0.4)
        # (3)-(5) heatmaps: rows M_max, columns λ
        # noise-floor outline: cells whose no-floor fit equals the floor fit in >= FLOOR_FRAC of the samples; if there is
        # none (the floor M* lies between grid cells), the cell with the mean M* closest to the floor fit's (dashed)
        exact_any = bool(np.any(same[r] >= FLOOR_FRAC))
        dM = np.abs(stats["M"][0][r] - np.nanmean(S["floor_M"][r]))
        closest = np.unravel_index(np.nanargmin(dM), dM.shape)
        closest_used = closest_used or not exact_any
        for j, key in zip((2, 3, 4), ("rmse", "D", "M")):
            h = ax[j]; mean = stats[key][0][r]
            ims[key] = h.imshow(mean, origin="lower", aspect="auto", cmap=cmaps[key], norm=norms[key])
            for i in range(Mmax.size):
                for k in range(lam.size):
                    if key == "M":
                        lo, hi = stats["M"][1][r, i, k], stats["M"][2][r, i, k]
                        txt = f"{lo:.0f}" if lo == hi else f"{lo:.0f}–{hi:.0f}"
                    else:
                        sd = stats[key][1][r, i, k]
                        txt = f"{1e3 * sd:.1f}" if key == "rmse" else (f"{sd:.1g}" if sd < 10 else f"{sd:.0f}")
                    rgb = ims[key].cmap(ims[key].norm(mean[i, k]))[:3]
                    h.text(k, i, txt, ha="center", va="center", fontsize=5,
                           color="k" if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 0.5 else "w")
                    if same[r, i, k] >= FLOOR_FRAC or (not exact_any and (i, k) == closest):
                        h.add_patch(Rectangle((k - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec="#f2c14e", lw=1.3,
                                              ls="-" if exact_any else (0, (2.5, 1.2)), zorder=5,
                                              path_effects=[pe.withStroke(linewidth=2.2, foreground="k")]))
            h.set_xticks(range(lam.size)); h.set_xticklabels([f"$10^{{{int(np.log10(l))}}}$" for l in lam])
            h.set_yticks(range(Mmax.size)); h.set_yticklabels([str(m) for m in Mmax])
            if j == 2:
                h.set_ylabel(r"$M_{\max}$", labelpad=1)
            else:
                h.tick_params(labelleft=False)
            if r == len(Ns) - 1:
                h.set_xlabel(r"penalty $\lambda$", labelpad=1)
            else:
                h.tick_params(labelbottom=False)
        if r == len(Ns) - 1:
            ax[0].set_xlabel(r"$\omega$", labelpad=1); c.set_xlabel("time along the backward $K$-ramp", labelpad=1)
        else:
            ax[0].tick_params(labelbottom=False); c.tick_params(labelbottom=False)
        if r == 0:
            for j, ttl in enumerate(("sample and LMMF fits", r"network $R(t)$ and LMMF bifurcations",
                                     "spectral RMSE", r"loss $D\,/\,\frac{1}{6N}$", r"components $M^*$")):
                ax[j].set_title(ttl, pad=3)
        for j, axx in enumerate(ax):
            pnas_panel_label(axx, next(letters), dx=-14 if j in (0, 3, 4) else -20, dy=3)
    # legend + colour bars
    lg = fig.add_subplot(gs[-1, :2]); lg.axis("off")
    hd = [Patch(fc=C_HIST, label="sample"), Line2D([], [], color=C_TRUE, lw=1.0, ls=(0, (3, 1.5)), label="true $\\rho$"),
          Line2D([], [], color=C_NET, lw=1.0, label="network $R(t)$"),
          Line2D([], [], color=C_EX["low"], lw=1.2, label="LMMF, few components"),
          Line2D([], [], color=C_EX["high"], lw=1.2, label="LMMF, many components"),
          Line2D([], [], color="0.3", lw=1.4, label="LMMF stable states\n(limit cycles: $R$ min, max)"),
          Patch(fc="0.3", alpha=0.22, label="non-stationary (simulation)"),
          Rectangle((0, 0), 1, 1, fill=False, ec="#f2c14e", lw=1.3, label="noise-floor fit ($\\geq$50 % of samples)")]
    if closest_used:
        hd.append(Rectangle((0, 0), 1, 1, fill=False, ec="#f2c14e", lw=1.3, ls=(0, (2.5, 1.2)),
                            label="closest cell to it"))
    if not band_any:
        hd = [h_ for h_ in hd if h_.get_label() != "non-stationary (simulation)"]
    lg.legend(handles=hd, loc="center left", ncol=3, frameon=False, fontsize=6, handlelength=1.6, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.0)
    mk = dict(ls="none", ms=4.2, mfc="w", mec="0.3", mew=1.0)          # bifurcation-marker key in the last ramp panel
    keys = [("HB0", "^", "Hopf (incoherent state)"), ("HB", "s", "Hopf"), ("LP", "o", "fold"), ("TR", "D", "torus"),
            ("PD", "X", "period doubling")]
    ramp_axes[-1].legend(handles=[Line2D([], [], marker=m_, label=l_, **mk) for k_, m_, l_ in keys if k_ in used_all],
                         loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=5.8, frameon=True, framealpha=0.9,
                         edgecolor="none", handletextpad=0.3, labelspacing=0.25, borderpad=0.3)
    for j, key, lab in ((2, "rmse", "mean RMSE\n" + r"(text: SD $\times 10^{3}$)"),
                        (3, "D", r"mean $D\,/\,\frac{1}{6N}$" + "\n" + r"(text: SD; $<1$: below floor)"),
                        (4, "M", "mean $M^*$\n(text: range)")):
        host = fig.add_subplot(gs[-1, j]); host.axis("off")
        cax = host.inset_axes([0.05, 0.55, 0.9, 0.18])
        cb = fig.colorbar(ims[key], cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=6, labelpad=1); cb.ax.tick_params(labelsize=5.5)
    os.makedirs(MS_FIGURES, exist_ok=True)
    out = os.path.join(MS_FIGURES, "bimodal_distributions")
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(out + ext, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"[saved] {out}.png/.svg/.pdf")
    if show:
        plt.show()


if __name__ == "__main__":
    main(show="--show" in sys.argv)
