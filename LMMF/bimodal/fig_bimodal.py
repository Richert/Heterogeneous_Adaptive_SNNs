r"""
Bimodal figure (PRL two-column): network landscape, LMMF error map, error vs distance across restarts.
=======================================================================================================

  (a) (K, α) classes of the N = 5000 network from the backward K-ramp (async / sync / oscillatory;
      second half of every holding window: oscillatory if std R > R_OSC, else sync if mean R > R_SYNC),
      with the exact incoherence boundary of ρ and that of the LMMF (M = 6).
      The forward ramp agrees except for isolated boundary cells (no bistability at ΔK = 0.1), so the
      backward ramp alone is used.
  (b) spectral RMSE between LMMF and network R(t) on the same grid (backward ramp, per holding window).
  (c) spectral RMSE vs CvM distance to the true ρ for all fits across restarts (M ≤ 7) at
      K = (0.8, 1, 1.2) K_H, α = 0 (bimodal_distance_scatter.py), with a log–log least-squares line per K.
  --illustration: the revised figure (main_illus): locked-fraction maps (a)-(c) + examples (d)-(g).
  (d)–(g) example holding windows of the backward ramp (network vs LMMF R(t)) at the (K, α) points
      EXAMPLES, marked by their letters in (a) and (b).
LMMF boundaries: zero contours of the largest (first mode) and second-largest (second mode)
eigenvalue real part at z = 0.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python fig_bimodal.py [--ref net|cont] [--restarts 40]
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "unimodal"),
                os.path.join(_HERE, "..", "..", "shared"), os.path.join(_HERE, "..", "continuity")]
import functools
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from prl_style import set_prl_style, panel_label
import unimodal_kalpha_figure as UF                  # spectral_rmse
import bimodal_kalpha as BK
import bimodal_distance_scatter as DS

R_SYNC, R_OSC = 0.1, 0.04
CLS_COLORS = ["0.88", "#c6dcf7", "#e87ba4"]
CLS_NAMES = ["async.", "sync.", "oscillatory"]
C_EXACT, C_LMMF, C_NOFLOOR = "0.1", "#eb6834", "#7b2cbf"
C_K = ["#9ec5ef", "#4a8fd9", "#0d3d7a"]
C_VAR = {"floor": "#eb6834", "nofloor": "#7b2cbf"}
C_NET = "0.15"
EXAMPLES = [(1.0, 0.0), (3.0, 0.0), (5.0, 0.0), (3.0, 1.0)]       # (K, α) of panels (d)-(g)
T_SHOW = 25.0                                                    # last T_SHOW time units of the window
_lab = functools.partial(panel_label, dx=-22, dy=3, fontsize=8)


def load_ramp(direction="bwd"):
    P = [np.load(f) for f in sorted(glob.glob(os.path.join(BK.OUT, f"ramp_{direction}_c*.npz")))]
    ia = np.concatenate([p["ia"] for p in P]); o = np.argsort(ia)
    take = lambda k: np.concatenate([p[k] for p in P])[o].astype(float)
    return take("R_net"), take("R_lmmf")


def classify(R):
    tail = R[..., R.shape[-1] // 2:]
    m, sd = tail.mean(-1), tail.std(-1)
    return np.where(sd > R_OSC, 2, np.where(m > R_SYNC, 1, 0))


def boundary_segments(ex, kmax, amax=1.45, jump=0.3):
    """Exact boundary points (K, α) inside the window, split where consecutive points jump."""
    K, a = ex[:, 0], ex[:, 1]
    ok = (K <= kmax) & (a >= 0) & (a <= amax)
    segs, cur = [], []
    for i in range(K.size):
        if ok[i] and (not cur or (abs(K[i] - cur[-1][0]) < jump and abs(a[i] - cur[-1][1]) < jump)):
            cur.append((K[i], a[i]))
        else:
            if len(cur) > 1:
                segs.append(np.array(cur))
            cur = [(K[i], a[i])] if ok[i] else []
    if len(cur) > 1:
        segs.append(np.array(cur))
    return segs


def draw_boundaries(ax, ex, sts):
    """Exact boundary (dashed) and the LMMF first/second-mode boundaries of every variant in sts."""
    for s in boundary_segments(ex, BK.K_GRID[-1]):
        ax.plot(s[:, 0], s[:, 1], "--", color=C_EXACT, lw=1.0, zorder=3)
    for v, st in sts.items():
        for g in ("G", "G2"):
            if g in st:
                ax.contour(st["K"], st["A"], st[g], levels=[0.0], colors=[C_VAR[v]], linewidths=0.9, zorder=4)


def mark_examples(ax, letters, examples=None):
    for (K, a), l in zip(examples or EXAMPLES, letters):
        ax.plot(K, a, "o", ms=6.5, mfc="white", mec="0.1", mew=0.6, zorder=6, clip_on=False)
        ax.text(K, a, l, ha="center", va="center", fontsize=5, fontweight="bold", zorder=7, clip_on=False)


def main(ref="net", n_restarts=DS.N_RESTARTS, illus=False, xdist="true"):
    set_prl_style("prl")
    fig = plt.figure(figsize=(7.0, 4.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.16, 1.0], left=0.065, right=0.99, top=0.955,
                             bottom=0.07, hspace=0.3)
    gs = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.08, 1.0], wspace=0.62)
    ax = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ext = [BK.K_GRID[0] - 0.05, BK.K_GRID[-1] + 0.05, BK.A_GRID[0] - 0.025, BK.A_GRID[-1] + 0.025]
    ex = np.load(os.path.join(BK.OUT, "boundary_exact.npy"))
    if illus:                                    # revised setup: LMMF/illustration.py (N = 1000, seed 18)
        sys.path.insert(0, os.path.join(_HERE, ".."))
        import illustration as IL
        loci = np.load(IL.loci_file("bimodal"), allow_pickle=True).item()
        sts = {v: dict(K=loci["K"], A=loci["A"], G=loci[v][0], G2=loci[v][1], fit=IL.load_fit("bimodal", v))
               for v in ("floor", "nofloor")}
        R = IL.load_ramps("bimodal", "bwd")
        kinds = list(R["kinds"])
        Rn, Rl, Rl_nf = R["R_net"], R["R_fit"][kinds.index("floor")], R["R_fit"][kinds.index("nofloor")]
        DS.OUT = os.path.join(DS.OUT, "illustration")
    else:
        sts = {v: np.load(BK.stability_file(v), allow_pickle=True).item() for v in BK.VARIANTS
               if os.path.exists(BK.stability_file(v))}
        Rn, Rl = load_ramp("bwd")
        nf = BK.lmmf_ramp_file("nofloor", "bwd")
        Rl_nf = np.load(nf)["R_lmmf"].astype(float) if os.path.exists(nf) else None
    N_NET = 1000 if illus else 5000
    M_var = {v: int(st["fit"]["M"]) for v, st in sts.items()}

    # (a) network classes
    cls = classify(Rn)
    ax[0].imshow(cls, origin="lower", aspect="auto", extent=ext, interpolation="nearest",
                 cmap=ListedColormap(CLS_COLORS), norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    draw_boundaries(ax[0], ex, sts)
    ax[0].set_title(rf"network ($N={N_NET}$)", pad=3)

    # (b) spectral RMSE LMMF vs network
    E = UF.spectral_rmse(Rn, Rl)
    im = ax[1].imshow(E, origin="lower", aspect="auto", extent=ext, interpolation="nearest", cmap="Blues",
                      norm=LogNorm(vmin=max(E[E > 0].min(), 1e-5), vmax=E.max()))
    draw_boundaries(ax[1], ex, {"floor": sts["floor"]})               # the map is for the floor fit
    cb = fig.colorbar(im, ax=ax[1], pad=0.02, fraction=0.06)
    cb.set_label("spectral RMSE", labelpad=2)
    cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: rf"${v / 10 ** np.floor(np.log10(v)):.0f}\times10^{{{int(np.floor(np.log10(v)))}}}$"))
    cb.ax.tick_params(labelsize=5)
    ax[1].set_title(rf"LMMF (noise floor, $M={M_var['floor']}$) vs. network", pad=3)
    for a in ax[:2]:
        a.set_xlim(ext[:2]); a.set_ylim(ext[2:]); a.set_xlabel(r"coupling $K$", labelpad=1)
        a.axhline(DS.ALPHA, color="k", lw=0.5, ls=(0, (1, 2)))
    ax[0].set_ylabel(r"phase lag $\alpha$", labelpad=2)
    for a in ax[:2]:
        mark_examples(a, "defg")
    print(f"[a] class counts (async, sync, osc): {np.bincount(cls.ravel(), minlength=3)}; "
          f"[b] RMSE range {E.min():.2e}–{E.max():.2e}, median {np.median(E):.2e}"
          + (f"; nofloor median {np.median(UF.spectral_rmse(Rn, Rl_nf)):.2e}" if Rl_nf is not None else ""))

    # (c) error vs distance: restart clouds of N_SAMPLES samples + the selected floor / no-floor fits
    c = ax[2]
    path = os.path.join(DS.OUT, f"sims_r{n_restarts}{DS.tag()}.npz")
    spaths = sorted(glob.glob(os.path.join(DS.OUT, f"sample*_sims_r{n_restarts}{DS.tag()}.npz")))
    if spaths:
        cont = np.load(os.path.join(DS.OUT, f"cont{DS.tag()}.npz"))
        S = [np.load(f_) for f_ in spaths]
        for k in range(cont["K"].size):
            X, Y, KIND = [], [], []
            for d in S:
                R_ref = d["R_net"][k] if ref == "net" else cont["R_cont"][k]
                Rf = d["R_fit"][:, k].astype(float)
                X.append(d["D_true"] if xdist == "true" else d["D_sample"]); Y.append(UF.spectral_rmse(Rf, np.broadcast_to(R_ref, Rf.shape)))
                KIND.append(d["kind"])
            x, y, kind = np.concatenate(X), np.concatenate(Y), np.concatenate(KIND)
            ok = (x > 0) & (y > 0)
            cl = ok & (kind == "cloud")
            c.loglog(x[cl], y[cl], "o", ms=1.6, color=C_K[k], alpha=0.45, mew=0, zorder=2)
            for kd, mk in (("floor", "o"), ("nofloor", "^")):
                b = ok & (kind == kd)
                c.loglog(x[b], y[b], mk, ms=3.0, mfc="none", mec=C_K[k], mew=0.7, zorder=3)
            p = np.polyfit(np.log(x[cl]), np.log(y[cl]), 1)
            xs = np.geomspace(x[cl].min(), x[cl].max(), 50)
            c.loglog(xs, np.exp(np.polyval(p, np.log(xs))), "-", color=C_K[k], lw=0.9, zorder=4)
            r = np.corrcoef(np.log(x[cl]), np.log(y[cl]))[0, 1]
            med = {kd: (np.median(x[kind == kd]), np.median(y[kind == kd])) for kd in ("floor", "nofloor")}
            print(f"[c] K = {cont['kfacs'][k]:g} K_H: {cl.sum()} cloud fits of {len(S)} samples, slope {p[0]:.2f}, "
                  f"r = {r:.2f}; median (D, RMSE) floor {med['floor'][0]:.1e}, {med['floor'][1]:.2e} | "
                  f"nofloor {med['nofloor'][0]:.1e}, {med['nofloor'][1]:.2e}")
        c.invert_xaxis()
        mmax = int(max(int(d_["M_nominal"].max()) for d_ in S if "M_nominal" in d_.files) if illus else DS.M_MAX)
        mmax = min(mmax, 15 if illus else DS.M_MAX)
        c.set_title(rf"$\alpha = {float(cont['alpha']):g}$, {len(S)} samples, restarts ($M\leq{mmax}$)", pad=3)
    elif os.path.exists(path):
        d = np.load(path)
        R_ref = d["R_net"] if ref == "net" else d["R_cont"]
        for k in range(d["K"].size):
            x = d["D_true"]; y = UF.spectral_rmse(d["R_fit"][:, k].astype(float),
                                                  np.broadcast_to(R_ref[k], d["R_fit"][:, k].shape))
            ok = (x > 0) & (y > 0)
            c.loglog(x[ok], y[ok], "o", ms=1.8, color=C_K[k], alpha=0.6, mew=0, zorder=2)
            b = d["is_best"] & ok
            c.loglog(x[b], y[b], "o", ms=3.2, mfc="none", mec=C_K[k], mew=0.8, zorder=3)
            p = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
            xs = np.geomspace(x[ok].min(), x[ok].max(), 50)
            c.loglog(xs, np.exp(np.polyval(p, np.log(xs))), "-", color=C_K[k], lw=0.9, zorder=4)
            r = np.corrcoef(np.log(x[ok]), np.log(y[ok]))[0, 1]
            print(f"[c] K = {d['kfacs'][k]:g} K_H: {ok.sum()} fits, slope {p[0]:.2f}, r = {r:.2f}")
        c.invert_xaxis()
        c.set_title(rf"$\alpha = {float(d['alpha']):g}$, fits across restarts ($M\leq{DS.M_MAX}$)", pad=3)
    else:
        c.text(0.5, 0.5, "scatter data pending\n(bimodal_distance_scatter.py)", transform=c.transAxes,
               ha="center", va="center", fontsize=6, color="0.4")
    c.set_xlabel(r"CvM distance $D$ to true $\rho$" if xdist == "true" else r"CvM distance $D$ to the sample", labelpad=1)
    c.set_ylabel("spectral RMSE" + ("" if ref == "net" else r" (vs. true-$\rho$ MF)"), labelpad=2)
    for a, l in zip(ax, "abc"):
        _lab(a, l)

    # (d)-(g) example holding windows (backward ramp)
    gex = outer[2].subgridspec(2, 2, wspace=0.12, hspace=0.55)
    dts = BK.DTS; ns = Rn.shape[-1]; t = np.arange(ns) * dts
    sel = t >= t[-1] - T_SHOW
    for e, ((K, a), l) in enumerate(zip(EXAMPLES, "defg")):
        axe = fig.add_subplot(gex[e // 2, e % 2])
        ia = int(np.argmin(np.abs(BK.A_GRID - a))); ik = int(np.argmin(np.abs(BK.K_GRID - K)))
        axe.plot(t[sel], Rn[ia, ik, sel], color=C_NET, lw=0.8, label="network")
        if Rl_nf is not None:
            axe.plot(t[sel], Rl_nf[ia, ik, sel], color=C_NOFLOOR, lw=0.8, label="LMMF (no floor)")
        axe.plot(t[sel], Rl[ia, ik, sel], color=C_LMMF, lw=0.8, label="LMMF")
        axe.set_ylim(0, 1); axe.set_xlim(t[sel][0], t[-1])
        axe.set_title(rf"$K = {BK.K_GRID[ik]:g}$, $\alpha = {BK.A_GRID[ia]:g}$: {CLS_NAMES[cls[ia, ik]]}, "
                      rf"spectral RMSE ${E[ia, ik] * 1e3:.1f}\times10^{{-3}}$", pad=2, fontsize=6)
        if e % 2:
            axe.set_yticklabels([])
        else:
            axe.set_ylabel(r"$R(t)$", labelpad=2)
        if e // 2:
            axe.set_xlabel(r"time within holding window", labelpad=1)
        else:
            axe.set_xticklabels([])
        _lab(axe, l)

    # shared legend
    lg = fig.add_subplot(outer[1]); lg.axis("off")
    h = [Patch(fc=c_, ec="0.6", lw=0.3, label=l_) for c_, l_ in zip(CLS_COLORS, CLS_NAMES)]
    h += [Line2D([], [], color=C_EXACT, ls="--", lw=1.0, label=r"exact boundary (true $\rho$)"),
          Line2D([], [], color=C_LMMF, lw=0.9, label=rf"LMMF, noise floor ($M={M_var.get('floor', '?')}$)"),
          Line2D([], [], color=C_NOFLOOR, lw=0.9, label=rf"LMMF, no floor ($M={M_var.get('nofloor', '?')}$)"),
          Line2D([], [], color=C_NET, lw=0.9, label=r"network $R(t)$")]
    h += [Line2D([], [], ls="none", marker="o", ms=2.5, color=C_K[k], label=rf"$K = {f:g}\,K_H$")
          for k, f in enumerate(DS.KFACS)]
    h += [Line2D([], [], ls="none", marker="o", ms=1.6, color="0.4", label="restart optima"),
          Line2D([], [], ls="none", marker="o", ms=3.0, mfc="none", mec="0.3", label="selected, noise floor"),
          Line2D([], [], ls="none", marker="^", ms=3.0, mfc="none", mec="0.3", label="selected, no floor")]
    lg.legend(handles=h, loc="center", ncol=6, frameon=False, handlelength=1.6, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.0, fontsize=6)
    out = (os.path.join(IL.OUT, "fig_bimodal_illustration") if illus else os.path.join(BK.OUT, "fig_bimodal")) \
        + ("" if ref == "net" else "_cont") + ("" if xdist == "true" else "_dsample")
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


ELL_FLOOR, ELL_C = 1e-3, 0.05
C_ILL = {"floor": "#eb6834", "nofloor": "#2a78d6"}      # revised figures: noise floor orange, no floor blue
EXAMPLES_ILL = [(1.7, 0.0), (3.0, 0.0), (4.5, 0.0), (3.0, 0.8)]   # (K, α) of panels (d)-(g), revised figure


def main_illus():
    """Revised bimodal figure (illustration setup, N = 1000, seed 18):
    (a)-(c) log10 locked fraction (backward ramp) of the network, the noise-floor LMMF and the floorless
    LMMF, with the exact critical lines of ρ (white, dashed); LMMF panels also show that LMMF's own
    critical lines (cluster-resolved: zero contours of the largest Re λ at z = 0 among the modes with
    Im λ < 0 (left cluster) and Im λ ≥ 0 (right cluster), the analogue of the two exact lines) and
    the network's ℓ = ELL_C contour (black). (d)-(g) example holding windows as before."""
    import matplotlib.patheffects as pe
    from matplotlib.colors import Normalize
    sys.path.insert(0, os.path.join(_HERE, ".."))
    import illustration as IL
    set_prl_style("prl")
    fig = plt.figure(figsize=(7.0, 4.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.13, 1.0], left=0.065, right=0.895, top=0.955,
                             bottom=0.07, hspace=0.3)
    gs = outer[0].subgridspec(1, 3, wspace=0.12)
    ax = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ex = np.load(os.path.join(BK.OUT, "boundary_exact.npy"))
    loci = np.load(IL.loci_file("bimodal"), allow_pickle=True).item()
    fits = {v: IL.load_fit("bimodal", v) for v in ("floor", "nofloor")}
    R = IL.load_ramps("bimodal", "bwd"); kinds = list(R["kinds"])
    Rn = R["R_net"]; Rl = {v: R["R_fit"][kinds.index(v)] for v in fits}
    L = IL.locking_maps("bimodal", "bwd")
    ell = {"net": L["ell_net"], **{v: L["ell_fit"][kinds.index(v)] for v in fits}}
    K, A = L["K"], L["alpha"]
    ext = [K[0] - 0.05, K[-1] + 0.05, A[0] - 0.025, A[-1] + 0.025]
    norm = Normalize(np.log10(ELL_FLOOR), 0.0)
    stroke = [pe.withStroke(linewidth=1.8, foreground="w")]
    titles = {"net": r"network ($N=1000$)", "floor": rf"LMMF, noise floor ($M={fits['floor']['M']}$)",
              "nofloor": rf"LMMF, no floor ($M={fits['nofloor']['M']}$)"}
    for a, key in zip(ax, ("net", "floor", "nofloor")):
        im = a.imshow(np.log10(np.maximum(ell[key], ELL_FLOOR)), origin="lower", aspect="auto", extent=ext,
                      cmap="viridis", norm=norm, interpolation="nearest")
        for sg in boundary_segments(ex, K[-1]):
            a.plot(sg[:, 0], sg[:, 1], "--", color="w", lw=1.0, zorder=3)
        if key != "net":
            for g in (2, 3):                                     # Gm (left cluster), Gp (right cluster)
                a.contour(loci["K"], loci["A"], loci[key][g], levels=[0.0], colors=[C_ILL[key]], linewidths=0.9,
                          zorder=4).set_path_effects(stroke)
            a.contour(K, A, ell["net"], levels=[ELL_C], colors=["k"], linewidths=0.7, zorder=4)
            a.set_yticklabels([])
        else:
            a.set_ylabel(r"phase lag $\alpha$", labelpad=2)
        a.axhline(0.0, color="w", lw=0.5, ls=(0, (1, 2)))
        mark_examples(a, "defg", EXAMPLES_ILL)
        a.set_xlim(ext[:2]); a.set_ylim(ext[2:]); a.set_xlabel(r"coupling $K$", labelpad=1)
        a.set_title(titles[key], pad=3)
    cax = fig.add_axes([0.91, 0.62, 0.012, 0.32])
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"$\log_{10}$ locked fraction $\ell$", labelpad=2)
    cb.ax.tick_params(labelsize=5.5)
    band = (ell["net"] > 0.01) & (ell["net"] < 0.5)
    print("[a-c] mean |ℓ_fit − ℓ_net| grid / transition band: "
          + ", ".join(f"{v} {np.abs(ell[v] - ell['net']).mean():.3f} / {np.abs(ell[v] - ell['net'])[band].mean():.3f}"
                      for v in fits))
    for a, l in zip(ax, "abc"):
        _lab(a, l)

    # (d)-(g) example holding windows (backward ramp)
    gex = outer[2].subgridspec(2, 2, wspace=0.12, hspace=0.55)
    ns = Rn.shape[-1]; t = np.arange(ns) * R["dts"]; sel = t >= t[-1] - T_SHOW
    for e, ((Kx, ax_), l) in enumerate(zip(EXAMPLES_ILL, "defg")):
        axe = fig.add_subplot(gex[e // 2, e % 2])
        ia = int(np.argmin(np.abs(A - ax_))); ik = int(np.argmin(np.abs(K - Kx)))
        axe.plot(t[sel], Rn[ia, ik, sel], color=C_NET, lw=0.8)
        axe.plot(t[sel], Rl["nofloor"][ia, ik, sel], color=C_ILL["nofloor"], lw=0.8)
        axe.plot(t[sel], Rl["floor"][ia, ik, sel], color=C_ILL["floor"], lw=0.8)
        axe.set_ylim(0, 1); axe.set_xlim(t[sel][0], t[-1])
        E = {v: UF.spectral_rmse(Rn[ia, ik], Rl[v][ia, ik]) * 1e3 for v in fits}
        axe.set_title(rf"$K = {K[ik]:g}$, $\alpha = {A[ia]:g}$: network $\ell = {ell['net'][ia, ik]:.2f}$; "
                      rf"RMSE ($\times10^{{-3}}$) floor {E['floor']:.1f}, no floor {E['nofloor']:.1f}", pad=2, fontsize=6)
        if e % 2:
            axe.set_yticklabels([])
        else:
            axe.set_ylabel(r"$R(t)$", labelpad=2)
        if e // 2:
            axe.set_xlabel(r"time within holding window", labelpad=1)
        else:
            axe.set_xticklabels([])
        _lab(axe, l)

    lg = fig.add_subplot(outer[1]); lg.axis("off")
    h = [Line2D([], [], color="w", ls="--", lw=1.0, label=r"exact critical lines (true $\rho$)",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="0.5")]),
         Line2D([], [], color=C_ILL["floor"], lw=0.9, label="LMMF, noise floor: critical lines / $R(t)$"),
         Line2D([], [], color=C_ILL["nofloor"], lw=0.9, label="LMMF, no floor: critical lines / $R(t)$"),
         Line2D([], [], color="k", lw=0.7, label=rf"network $\ell={ELL_C:g}$ / network $R(t)$")]
    lg.legend(handles=h, loc="center", ncol=4, frameon=False, handlelength=1.8, columnspacing=1.2,
              handletextpad=0.4, borderaxespad=0.0, fontsize=6)
    out = os.path.join(IL.OUT, "fig_bimodal_illustration")
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


if __name__ == "__main__":
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    if "--illustration" in sys.argv:
        main_illus()
    else:
        main(arg("--ref", "net"), arg("--restarts", DS.N_RESTARTS, int), False, arg("--xdist", "true"))
