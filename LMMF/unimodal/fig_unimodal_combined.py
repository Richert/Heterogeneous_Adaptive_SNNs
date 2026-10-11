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


def main(names, alpha2, mmax, width_k):
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
        fits = {s: np.load(FS.fpath(name, s, i, N1.N_NET, mmax, width_k), allow_pickle=True).item()
                for s, i in N1.SCEN_ORDER}
        # (1) distribution, sample, fits
        a = ax[0]
        x = N1.network_sample(name)
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
        net = assemble(name, "net", "R_net")
        cls = classify_binary(net)
        n4 = np.bincount(UF.combine(classify(net["fwd"]), classify(net["bwd"])).ravel() * 2 + cls.ravel(), minlength=8)
        print(f"{name}: 4-class -> (async, sync): mono-async {n4[0:2]}, mono-sync {n4[2:4]}, "
              f"multistable {n4[4:6]}, oscillatory {n4[6:8]}")
        b.imshow(cls, origin="lower", aspect="auto", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
        ex = np.load(UK._f(name, "boundary_exact") + ".npy"); ex = ex[np.isfinite(ex[:, 0])]
        b.plot(ex[:, 0], np.abs(ex[:, 1]), "--", color=C_TRUE, lw=1.0)
        for s in SCEN:
            b.contour(UK.K_FINE, UK.A_FINE, fits[s]["G"], levels=[0.0], colors=[C_FIT[s]], linewidths=0.85)
        for a_ in alphas:
            b.axhline(a_, color="k", lw=0.5, ls=(0, (1, 2)))
        b.set_xlim(ext[:2]); b.set_ylim(ext[2:]); b.set_ylabel(r"phase lag $\alpha$", labelpad=2)
        # (3)-(4) RMSE vs K, shared y-limits within the row
        lm = assemble(name, "lmmf", "R_lmmf", mmax, width_k)
        has_cont = any(f.startswith(f"{name}_cont_") for f in os.listdir(N1.OUT))
        co = assemble(name, "cont", "R_cont") if has_cont else None
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
            ax[1].set_title(r"network ($N=1000$) and thresholds", pad=3)
    # shared legend
    lg = fig.add_subplot(gs[-1, :]); lg.axis("off")
    h = [Line2D([], [], color=C_TRUE, lw=1.1, label=r"true $\rho$"), Patch(fc=C_HIST, label=r"sample ($N=1000$)")]
    h += [Line2D([], [], color=C_FIT[s], lw=0.9, label=L_FIT[s]) for s in SCEN]
    h += [Line2D([], [], color=C_TRUE, ls="--", lw=1.0, label=r"exact threshold (true $\rho$)"),
          Line2D([], [], color=C_REF, lw=1.0, label=r"network vs. true-$\rho$ mean field")]
    h += [Patch(fc=c_, ec="0.6", lw=0.3, label=l_) for c_, l_ in zip(CLS_COLORS, CLS_NAMES)]
    lg.legend(handles=h, loc="center", ncol=4, frameon=False, handlelength=1.6, columnspacing=1.2,
              handletextpad=0.5, borderaxespad=0.0)
    tag = ("" if (mmax, width_k) == (MMAX_MAIN, WIDTH_K_MAIN) else
           f"_mmax{mmax}" + (f"_wk{width_k:g}" if width_k else ""))
    out = os.path.join(UK.OUT, f"fig_unimodal_combined{tag}")
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


if __name__ == "__main__":
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    main(arg("--dists", f"gauss,g{UK.N_RATIONAL},dgauss").split(","), arg("--alpha2", ALPHA2_DEFAULT, float),
         arg("--mmax", MMAX_MAIN, int), arg("--width-k", WIDTH_K_MAIN, float))
