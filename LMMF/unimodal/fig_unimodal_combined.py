r"""
Combined unimodal figure — PRL two-column style (7.0 in wide, STIX serif, bold panel labels outside the
top-left axis corner, shared legend, no suptitle).

One ROW per distribution (default: Gaussian, rational g_8, double Gaussian), four columns:
  (1) true ρ (dark grey), N = 1000 sample histogram (light grey), and the three LMMF fits:
      true ρ without floor / sample with noise floor / sample without floor          [fit_scenarios.py]
  (2) classification of the N = 1000 network (K-ramps, forward vs backward) in the (K, α) plane, with the
      exact incoherence boundary of the true ρ (dark grey, dashed) and the full LMMF incoherence
      boundaries of the three fits (fit colours); dotted lines mark the α of columns (3)-(4)
  (3) spectral RMSE (LMMF vs network, mean over both ramp directions) vs K at α = 0, one colour per fit;
      grey: finite-size reference = RMSE between the network and the infinite-N mean field of the true ρ
  (4) as (3), at α = α2 (tunable, --alpha2). Columns (3)-(4) share their y-limits within each row.
Reads only saved results; runtime < 1 min.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python fig_unimodal_combined.py [--alpha2 0.7]
            [--dists gauss,g8,dgauss] [--mmax 64] [--width-k 10]
Default (main figure): M_max = 64 with the width floor k = 10 for the sample fits -> fig_unimodal_combined.*;
other fit sets are written with a tag (e.g. _mmax100).
--illustration [--ramp-alpha 0.7]: the revised figure (main_illus) for the setup of LMMF/illustration.py
(seed-searched N = 1000 samples, λ = 1e-6): fits with / without the noise floor, network locked fraction,
and R(t) along the backward ramp -> illustration/fig_unimodal_illustration.*
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import functools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
from prl_style import set_prl_style, panel_label
import lorentzian_mixture as LM
import unimodal_kalpha as UK
import unimodal_kalpha_figure as UF
import fit_scenarios as FS
import unimodal_n1000 as N1

LABELS = {"gauss": "Gaussian", "dgauss": "double Gaussian", "uniform": "uniform"}
C_TRUE, C_HIST, C_REF = "0.15", "0.85", "0.6"
SCEN = ("pop", "floor", "nofloor")
C_FIT = {"pop": "#2a78d6", "floor": "#eb6834", "nofloor": "#7b2cbf"}
L_FIT = {"pop": r"LMMF fit to true $\rho$ (no floor)", "floor": "LMMF fit to sample (noise floor)",
         "nofloor": "LMMF fit to sample (no floor)"}
CLS_COLORS = ["0.88", "#c6dcf7"]                              # async, sync
CLS_NAMES = ["async.", "sync."]
ALPHA2_DEFAULT = 0.7
MMAX_MAIN, WIDTH_K_MAIN = 64, 10                             # fit set of the main figure (untagged file name)
R_SYNC, R_OSC = 0.1, 0.05                                     # N = 1000: larger finite-size jitter
ELL_FLOOR = 1e-3                                              # lower end of the log10 ℓ colour scale
_lab = functools.partial(panel_label, dx=-20, dy=3, fontsize=8)


def label(name):
    return LABELS.get(name, rf"rational $g_{{{name[1:]}}}$" if name.startswith("g") else name)


def classify(R):
    tail = R[..., R.shape[-1] // 2:]
    m, sd = tail.mean(-1), tail.std(-1)
    return np.where(sd > R_OSC, 2, np.where(m > R_SYNC, 1, 0))


def classify_binary(net):
    """0 async / 1 sync: time-averaged R (second half of each hold window), averaged over the forward
    and backward ramps, above R_SYNC. Multistable and oscillatory points are thereby assigned to
    whichever side their mean coherence falls on."""
    m = 0.5 * sum(R[..., R.shape[-1] // 2:].mean(-1) for R in (net["fwd"], net["bwd"]))
    return (m > R_SYNC).astype(int)


def assemble(name, kind, key, mmax=16, width_k=0):
    """Concatenate the α chunks per direction (ordered by α index). kind 'lmmf' with mmax != 16 reads
    the LMMF-only files of unimodal_n1000.py; with mmax = 16 the LMMF traces live in the net files."""
    out = {}
    for d in UK.DIRECTIONS:
        if kind == "lmmf" and (mmax != 16 or width_k):
            prefix = f"{name}_lmmf_mmax{mmax}" + (f"_wk{width_k:g}" if width_k else "") + f"_{d}_c"
        else:
            prefix = f"{name}_{'net' if kind == 'lmmf' else kind}_{d}_c"
        P = [np.load(os.path.join(N1.OUT, f)) for f in sorted(os.listdir(N1.OUT)) if f.startswith(prefix)]
        ia = np.concatenate([p["ia"] for p in P]); o = np.argsort(ia)
        axis = 1 if key == "R_lmmf" else 0
        out[d] = np.concatenate([p[key] for p in P], axis=axis).take(o, axis=axis).astype(float)
    return out


def rmse_row(net, other, ia, k=None):
    sel = (lambda d: other[d][ia]) if k is None else (lambda d: other[d][k, ia])
    return 0.5 * (UF.spectral_rmse(net["fwd"][ia], sel("fwd")) + UF.spectral_rmse(net["bwd"][ia], sel("bwd")))


def load_row(name, illus, mmax, width_k):
    """Everything a row needs: fits (with stability map 'G'), sample, ramps of network / fits / continuum,
    and the (K, α) grid of the stability maps."""
    if illus:
        IL = _illustration()
        loci = np.load(IL.loci_file(name), allow_pickle=True).item()
        fits = {s: dict(IL.load_fit(name, s), G=loci[s][0]) for s in SCEN}
        R = {d: IL.load_ramps(name, d) for d in UK.DIRECTIONS}
        return dict(fits=fits, x=IL.sample(name), net={d: R[d]["R_net"] for d in R}, lm={d: R[d]["R_fit"] for d in R},
                    co={d: R[d]["R_cont"] for d in R}, KF=loci["K"], AF=loci["A"])
    fits = {s: np.load(FS.fpath(name, s, i, N1.N_NET, mmax, width_k), allow_pickle=True).item()
            for s, i in N1.SCEN_ORDER}
    has_cont = any(f.startswith(f"{name}_cont_") for f in os.listdir(N1.OUT))
    return dict(fits=fits, x=N1.network_sample(name), net=assemble(name, "net", "R_net"),
                lm=assemble(name, "lmmf", "R_lmmf", mmax, width_k),
                co=assemble(name, "cont", "R_cont") if has_cont else None, KF=UK.K_FINE, AF=UK.A_FINE)


def _illustration():
    sys.path.insert(0, os.path.join(_HERE, ".."))
    import illustration as IL
    return IL


def main(names, alpha2, mmax, width_k, illus=False):
    set_prl_style("prl")
    fig = plt.figure(figsize=(7.0, 1.5 * len(names) + 0.5))
    gs = fig.add_gridspec(len(names) + 1, 4, height_ratios=[1.0] * len(names) + [0.24],
                          width_ratios=[0.85, 1.0, 0.9, 0.9], left=0.075, right=0.99, top=0.955,
                          bottom=0.01, wspace=0.42, hspace=0.3)
    cmap = ListedColormap(CLS_COLORS); norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    ext = [UK.K_GRID[0], UK.K_GRID[-1], UK.A_GRID[0], UK.A_GRID[-1]]
    alphas = (0.0, alpha2)
    letters = iter("abcdefghijklmnop")
    for r, name in enumerate(names):
        ax = [fig.add_subplot(gs[r, j]) for j in range(4)]
        D = load_row(name, illus, mmax, width_k)
        fits = D["fits"]
        # (1) distribution, sample, fits
        a = ax[0]
        x = D["x"]
        lim = 2.5 if name == "dgauss" else 2.0 if name.startswith("g") else 3.0
        g = np.linspace(-lim, lim, 1201)
        a.hist(x, bins=50, range=(-lim, lim), density=True, color=C_HIST, edgecolor="none", zorder=1)
        # the no-floor sample fit is omitted here (too crowded); drawing order: true ρ, then the fit to
        # the true ρ, then the noise-floor sample fit in front
        a.plot(g, np.asarray(UK.dists()[name]["pdf"](g), float).ravel(), color=C_TRUE, lw=1.1, zorder=2)
        for s, z in (("pop", 3), ("floor", 4)):
            f = fits[s]
            a.plot(g, LM.LorentzianMixture(f["w"], f["Om"], f["De"]).pdf(g), color=C_FIT[s], lw=0.8, zorder=z)
        a.set_xlim(-lim, lim); a.set_yticks([])
        a.set_ylabel(label(name) + "\n" + r"$\rho(\omega)$", labelpad=2)
        a.text(0.97, 0.97, r"$M$: " + "/".join(str(fits[s]["M"]) for s in SCEN), transform=a.transAxes,
               ha="right", va="top", fontsize=5.5, color="0.25")
        # (2) network classification + incoherence boundaries
        b = ax[1]
        net = D["net"]
        if illus:                                # locked fraction of the network (backward ramp)
            L = _illustration().locking_maps(name, "bwd")
            ext_l = [UK.K_GRID[0] - 0.05, UK.K_GRID[-1] + 0.05, UK.A_GRID[0] - 0.025, UK.A_GRID[-1] + 0.025]
            im_l = b.imshow(np.log10(np.maximum(L["ell_net"], ELL_FLOOR)), origin="lower", aspect="auto", extent=ext_l,
                            cmap="viridis", norm=Normalize(np.log10(ELL_FLOOR), 0.0), interpolation="nearest")
            ex = np.load(UK._f(name, "boundary_exact") + ".npy"); ex = ex[np.isfinite(ex[:, 0])]
            b.plot(ex[:, 0], np.abs(ex[:, 1]), "--", color="w", lw=1.0)
            for s in SCEN:
                b.contour(D["KF"], D["AF"], fits[s]["G"], levels=[0.0], colors=[C_FIT[s]],
                          linewidths=0.9).set_path_effects([pe.withStroke(linewidth=1.8, foreground="w")])
            for a_ in alphas:
                b.axhline(a_, color="w", lw=0.5, ls=(0, (1, 2)))
            b.set_xlim(ext[:2]); b.set_ylim(ext[2:]); b.set_ylabel(r"phase lag $\alpha$", labelpad=2)
        cls = classify_binary(net)
        n4 = np.bincount(UF.combine(classify(net["fwd"]), classify(net["bwd"])).ravel() * 2 + cls.ravel(), minlength=8)
        print(f"{name}: 4-class -> (async, sync): mono-async {n4[0:2]}, mono-sync {n4[2:4]}, "
              f"multistable {n4[4:6]}, oscillatory {n4[6:8]}")
        if not illus:
            b.imshow(cls, origin="lower", aspect="auto", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
            ex = np.load(UK._f(name, "boundary_exact") + ".npy"); ex = ex[np.isfinite(ex[:, 0])]
            b.plot(ex[:, 0], np.abs(ex[:, 1]), "--", color=C_TRUE, lw=1.0)
            for s in SCEN:
                b.contour(D["KF"], D["AF"], fits[s]["G"], levels=[0.0], colors=[C_FIT[s]], linewidths=0.85)
            for a_ in alphas:
                b.axhline(a_, color="k", lw=0.5, ls=(0, (1, 2)))
            b.set_xlim(ext[:2]); b.set_ylim(ext[2:]); b.set_ylabel(r"phase lag $\alpha$", labelpad=2)
        # (3)-(4) RMSE vs K, shared y-limits within the row
        lm, co = D["lm"], D["co"]
        vals = []
        for c, a_ in zip(ax[2:], alphas):
            ia = int(np.argmin(np.abs(UK.A_GRID - a_)))
            if co is not None:
                y = rmse_row(net, co, ia); vals.append(y); c.semilogy(UK.K_GRID, y, color=C_REF, lw=1.0)
            for k, s in enumerate(SCEN):
                y = rmse_row(net, lm, ia, k); vals.append(y); c.semilogy(UK.K_GRID, y, color=C_FIT[s], lw=0.85)
            c.set_xlim(ext[:2])
            if r == 0:
                c.set_title(rf"$\alpha = {UK.A_GRID[ia]:g}$", pad=3)
        v = np.concatenate(vals); v = v[np.isfinite(v) & (v > 0)]
        lo, hi = 10 ** np.floor(np.log10(v.min())), 10 ** np.ceil(np.log10(v.max()))
        for c in ax[2:]:
            c.set_ylim(lo, hi)
            c.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            c.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
            c.yaxis.set_minor_formatter(NullFormatter())
        ax[3].set_yticklabels([])
        ax[2].set_ylabel("spectral RMSE", labelpad=2)
        for j, axx in enumerate(ax):
            if r == len(names) - 1:
                axx.set_xlabel((r"$\omega$", r"coupling $K$", r"coupling $K$", r"coupling $K$")[j], labelpad=1)
            else:
                axx.tick_params(labelbottom=False)
            _lab(axx, next(letters))
        if r == 0:
            ax[0].set_title(r"distribution, $N=1000$ sample, fits", pad=3)
            ax[1].set_title(r"network ($N=1000$): locked fraction" if illus else
                            r"network ($N=1000$) and thresholds", pad=3)
    # shared legend
    lg = fig.add_subplot(gs[-1, :]); lg.axis("off")
    h = [Line2D([], [], color=C_TRUE, lw=1.1, label=r"true $\rho$"), Patch(fc=C_HIST, label=r"sample ($N=1000$)")]
    h += [Line2D([], [], color=C_FIT[s], lw=0.9, label=L_FIT[s]) for s in SCEN]
    h += [Line2D([], [], color=C_TRUE, ls="--", lw=1.0, label=r"exact threshold (true $\rho$)"),
          Line2D([], [], color=C_REF, lw=1.0, label=r"network vs. true-$\rho$ mean field")]
    if illus:
        lg.legend(handles=h, loc="center left", ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2,
                  handletextpad=0.5, borderaxespad=0.0, bbox_to_anchor=(0.0, 0.5))
        cax = lg.inset_axes([0.80, 0.30, 0.18, 0.16])
        cb = fig.colorbar(im_l, cax=cax, orientation="horizontal")
        cb.set_label(r"$\log_{10}$ locked fraction $\ell$ (network)", fontsize=6, labelpad=2)
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=5.5)
    else:
        h += [Patch(fc=c_, ec="0.6", lw=0.3, label=l_) for c_, l_ in zip(CLS_COLORS, CLS_NAMES)]
        lg.legend(handles=h, loc="center", ncol=4, frameon=False, handlelength=1.6, columnspacing=1.2,
                  handletextpad=0.5, borderaxespad=0.0)
    tag = ("" if (mmax, width_k) == (MMAX_MAIN, WIDTH_K_MAIN) else
           f"_mmax{mmax}" + (f"_wk{width_k:g}" if width_k else ""))
    out = (os.path.join(_illustration().OUT, "fig_unimodal_illustration") if illus else
           os.path.join(UK.OUT, f"fig_unimodal_combined{tag}"))
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


C_ILL = {"floor": "#eb6834", "nofloor": "#2a78d6"}          # revised figure: noise floor orange, no floor blue
L_ILL = {"floor": "LMMF, noise floor", "nofloor": "LMMF, no floor"}
RAMP_ALPHA = {"gauss": 0.7, "g8": 0.4, "dgauss": 1.1}   # α of the ramp panel, per distribution


def main_illus(names, ramp_alpha=RAMP_ALPHA):
    """Revised unimodal figure (illustration setup). Per row: (1) true ρ, N = 1000 sample, LMMF fits with
    (orange) and without (blue) the noise floor; (2) network locked fraction (backward ramp) with the exact
    threshold (white, dashed) and the two LMMF thresholds; (3) R(t) along the whole backward K-ramp at
    α = ramp_alpha[dist] for network and both LMMFs (LMMF traces recomputed with the finite-size kick applied
    only when R < N^{-1/2}, i.e. near incoherence: no spikes in the coherent stretches, correct re-growth),
    with a K strip inside the top of the panel."""
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as pe
    IL = _illustration()
    set_prl_style("prl")
    fig = plt.figure(figsize=(7.0, 1.5 * len(names) + 0.5))
    gs = fig.add_gridspec(len(names) + 1, 3, height_ratios=[1.0] * len(names) + [0.24],
                          width_ratios=[0.85, 1.0, 1.9], left=0.075, right=0.99, top=0.955,
                          bottom=0.01, wspace=0.32, hspace=0.3)
    ext = [UK.K_GRID[0] - 0.05, UK.K_GRID[-1] + 0.05, UK.A_GRID[0] - 0.025, UK.A_GRID[-1] + 0.025]
    letters = iter("abcdefghijklmnop")
    kinds = ("floor", "nofloor")
    stroke = [pe.withStroke(linewidth=1.8, foreground="w")]
    for r, name in enumerate(names):
        ax = [fig.add_subplot(gs[r, j]) for j in range(3)]
        fits = {k: IL.load_fit(name, k) for k in kinds}
        loci = np.load(IL.loci_file(name), allow_pickle=True).item()
        # (1) distribution
        a = ax[0]; x = IL.sample(name)
        lim = 2.5 if name == "dgauss" else 2.0 if name.startswith("g") else 3.0
        g = np.linspace(-lim, lim, 1201)
        a.hist(x, bins=50, range=(-lim, lim), density=True, color=C_HIST, edgecolor="none", zorder=1)
        a.plot(g, np.asarray(UK.dists()[name]["pdf"](g), float).ravel(), color=C_TRUE, lw=1.1, zorder=2)
        for k, z in (("nofloor", 3), ("floor", 4)):
            f = fits[k]
            a.plot(g, LM.LorentzianMixture(f["w"], f["Om"], f["De"]).pdf(g), color=C_ILL[k], lw=0.8, zorder=z)
        a.set_xlim(-lim, lim); a.set_yticks([])
        a.set_ylabel(label(name) + "\n" + r"$\rho(\omega)$", labelpad=2)
        a.text(0.97, 0.97, r"$M$: " + "/".join(str(fits[k]["M"]) for k in kinds), transform=a.transAxes,
               ha="right", va="top", fontsize=5.5, color="0.25")
        # (2) network locked fraction + thresholds
        b = ax[1]
        L = IL.locking_maps(name, "bwd")
        im_l = b.imshow(np.log10(np.maximum(L["ell_net"], ELL_FLOOR)), origin="lower", aspect="auto", extent=ext,
                        cmap="viridis", norm=Normalize(np.log10(ELL_FLOOR), 0.0), interpolation="nearest")
        ex = np.load(UK._f(name, "boundary_exact") + ".npy"); ex = ex[np.isfinite(ex[:, 0])]
        b.plot(ex[:, 0], np.abs(ex[:, 1]), "--", color="w", lw=1.0)
        for k in kinds:
            b.contour(loci["K"], loci["A"], loci[k][0], levels=[0.0], colors=[C_ILL[k]],
                      linewidths=0.9).set_path_effects(stroke)
        a_r = ramp_alpha[name] if isinstance(ramp_alpha, dict) else ramp_alpha
        b.axhline(a_r, color="w", lw=0.5, ls=(0, (1, 2)))
        b.set_xlim(ext[:2]); b.set_ylim(ext[2:]); b.set_ylabel(r"phase lag $\alpha$", labelpad=2)
        b.set_yticks(np.arange(0.0, 1.41, 0.2))
        # (3) backward ramp at α = ramp_alpha
        c = ax[2]
        R = IL.load_ramps(name, "bwd"); kk = list(R["kinds"])
        ia = int(np.argmin(np.abs(R["alpha"] - a_r)))
        Ks = R["K"][::-1]                                         # ramp order: K decreasing
        ns = R["R_net"].shape[-1]; Tw = (ns - 1) * R["dts"]
        t = np.concatenate([i * Tw + np.arange(ns - 1) * R["dts"] for i in range(Ks.size)])
        seg = lambda arr: np.concatenate([arr[ik, :-1] for ik in range(R["K"].size - 1, -1, -1)])
        c.plot(t, seg(R["R_net"][ia]), color=C_TRUE, lw=0.35, zorder=2)
        sys.path.insert(0, os.path.join(_HERE, ".."))
        import illustration_ramps as IR                          # LMMF traces: kick only near incoherence
        Rnk = IR.lmmf_ramp_nokick(name, float(R["alpha"][ia]), kinds)
        for j, k in enumerate(kinds):
            c.plot(t, seg(Rnk[j]), color=C_ILL[k], lw=0.6, zorder=3)
        c.text(0.012, 0.04, rf"$\alpha = {R['alpha'][ia]:g}$", transform=c.transAxes, ha="left", va="bottom",
               fontsize=6.5, bbox=dict(fc="w", ec="none", pad=1.0, alpha=0.85), zorder=5)
        c.set_xlim(0, t[-1]); c.set_ylim(0, 1.12)
        c.set_ylabel(r"$R(t)$", labelpad=2)
        # K strip inside the top of the panel
        strip = c.inset_axes([0.0, 0.905, 1.0, 0.095])
        strip.imshow(Ks[None, :], aspect="auto", cmap="Greys", extent=[0, t[-1], 0, 1],
                     norm=Normalize(-0.4 * Ks.max(), 1.25 * Ks.max()), interpolation="nearest")
        for Kl in (3.5, 2.5, 1.5, 0.5):
            i = int(np.argmin(np.abs(Ks - Kl)))
            strip.text((i + 0.5) * Tw, 0.5, rf"$K\!=\!{Ks[i]:g}$", ha="center", va="center", fontsize=5,
                       color="w" if Ks[i] > 2.0 else "k")
        for Kb in (3.0, 2.0, 1.0):                               # K = 3, 2, 1 boundaries as thin ticks
            strip.axvline((int(np.argmin(np.abs(Ks - Kb)))) * Tw, color="0.5", lw=0.4)
        strip.set_xlim(0, t[-1]); strip.set_xticks([]); strip.set_yticks([])
        for sp in strip.spines.values():
            sp.set_linewidth(0.4)
        if r == 0:
            c.set_title(r"backward $K$-ramp ($100$ time units per $K$ step)", pad=3)
            ax[0].set_title(r"distribution, $N=1000$ sample, fits", pad=3)
            ax[1].set_title(r"network ($N=1000$): locked fraction", pad=3)
        for j, axx in enumerate(ax):
            if r == len(names) - 1:
                axx.set_xlabel((r"$\omega$", r"coupling $K$", r"time along the ramp")[j], labelpad=1)
            else:
                axx.tick_params(labelbottom=False)
            _lab(axx, next(letters))
    lg = fig.add_subplot(gs[-1, :]); lg.axis("off")
    h = [Line2D([], [], color=C_TRUE, lw=1.1, label=r"true $\rho$ / network $R(t)$"),
         Patch(fc=C_HIST, label=r"sample ($N=1000$)")]
    h += [Line2D([], [], color=C_ILL[k], lw=0.9, label=L_ILL[k]) for k in kinds]
    h += [Line2D([], [], color="w", ls="--", lw=1.0, label=r"exact threshold (true $\rho$)",
                 path_effects=[pe.withStroke(linewidth=2.0, foreground="0.5")])]
    lg.legend(handles=h, loc="center left", ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2,
              handletextpad=0.5, borderaxespad=0.0, bbox_to_anchor=(0.0, 0.5))
    cax = lg.inset_axes([0.80, 0.30, 0.18, 0.16])
    cb = fig.colorbar(im_l, cax=cax, orientation="horizontal")
    cb.set_label(r"$\log_{10}$ locked fraction $\ell$ (network)", fontsize=6, labelpad=2)
    cb.ax.xaxis.set_label_position("top"); cb.ax.tick_params(labelsize=5.5)
    out = os.path.join(IL.OUT, "fig_unimodal_illustration")
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


if __name__ == "__main__":
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    if "--illustration" in sys.argv:
        main_illus(arg("--dists", f"gauss,g{UK.N_RATIONAL},dgauss").split(","), arg("--ramp-alpha", RAMP_ALPHA, float))
        # (--ramp-alpha A sets one α for all rows; default: RAMP_ALPHA per distribution)
    else:
        main(arg("--dists", f"gauss,g{UK.N_RATIONAL},dgauss").split(","), arg("--alpha2", ALPHA2_DEFAULT, float),
             arg("--mmax", MMAX_MAIN, int), arg("--width-k", WIDTH_K_MAIN, float))
