r"""
Main figure (single-column PRL): the level of detail of the frequency distribution and the coherence dynamics
=============================================================================================================

Same dynamics in all models (setup C; scn_model_fit "hr-noise+transient+local"), frequency distribution per slice and
lobe captured by: no floor (λ = 1e-5) | noise floor | single Lorentzian (M_max = 1).
FROZEN comparison (default): the dynamical parameters are those of the fit with the EMPIRICAL distribution (one narrow
Lorentzian per cell; scn_model_fit kind "empirical") and are NOT refitted for the mixtures (scn_model_fit frozen), so
every difference between the models is due to the frequency distribution alone. --refit: each mixture with its own
fitted parameters instead.

  (a) SCN3: intrinsic-frequency histogram of all cells (lobes pooled) and the three model distributions (slice level:
      lobe mixtures weighted by the lobe fractions)
  (b) SCN3: coherence R(t), data and c(t)|Z(t)| of the empirical reference and the three models
  (c) approximation error vs distribution error, per slice and model:
      x  Cramér–von Mises distance D = W²/N between each lobe's model distribution and its cells' frequencies,
         averaged over the lobes (cell-weighted)
      y  frozen: RMSE of c|Z| vs the empirical reference's c|Z_ref| over the mask (slice and both lobes combined);
         dotted: the reference's own error vs the data (mean over slices); --refit: RMSE of c|Z| vs R
      grey lines connect the three models of a slice
  (d) coherence LEVELS relative to the data, per slice and model (incl. the empirical reference): R_model − R_data
      before TTX (mean over [start, t_ttx − 24 h)), the minimum under TTX (in [t_ttx, t_wash + 24 h)) and after
      washout (mean over [t_wash + 72 h, end)); model = c|Z| (frozen), data = R. (The lobe-specific
      desynchronisation rates, the previous panel (d), are moved to the supplement.)
  (e) SCN3: steady-state coherence R*(K) of the three mixtures with ONE global coupling K (no lag, noise or local
      coupling; scn_bifurcation_mixtures, synchronised branch: mean and oscillation envelope), log K in [0.01, 0.5];
      horizontal lines: the ceiling-corrected data coherence before TTX, after washout and its minimum (lowest R/c
      in [t_ttx, t_wash + 24 h]); markers: the coupling at which each mixture's R* matches each level
  (f) these inferred couplings for SCN1–4 (circle: pre-TTX, square: post-washout, triangle: minimum), per mixture
  (needs scn_bifurcation_mixtures.py to have been run)

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_main_figure.py [--example 3] [--refit] [--e-variant log|logz|lin|rel|1-R|rel1-R]
    python scn_main_figure.py --e-variants          (sheet with all x-axis variants of panel (e))
Output: data_paths.MPMF/circadian_data/writeup/scn_lmmf_main.{pdf,png}, scn_lmmf_main_stats.json
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from scipy.optimize import curve_fit
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_model_fit as F
import scn_supp_figure as SF
import scn_bifurcation_mixtures as BM

OUT = dp.ensure(os.path.join(S.OUT, "writeup"))
SLICES = (1, 2, 3, 4)
MARK = ["o", "s", "D", "^"]
CPD = 24 / (2 * np.pi)
VARIANTS = SF.VARIANTS                                      # (kind, label, colour, linestyle)
REF = ("empirical", "empirical", "#7b3294", "-")
FIT = os.path.join(S.OUT, "model_fit", "shapes")
WIN = "hr-noise+transient+local"
FROZEN = "--refit" not in sys.argv


def mlabel(i, kind, lab):
    """Legend label with the number of Lorentzians per lobe (M = a/b if the lobes differ); "M = 1" for single."""
    if kind == "single":
        return "M = 1"
    if kind == "empirical":
        return lab
    mod = F.mixture(i, kind, TG.load(i)); n = np.bincount(mod["lobe"])
    return f"{lab} (M = {n[0]})" if n[0] == n[1] else f"{lab} (M = {n[0]}/{n[1]})"


def res_of(i, kind):
    """Fit result of slice i: frozen evaluation at the empirical reference (default) or the refit (--refit)."""
    if not FROZEN:
        return SF.load(i, kind)
    f = os.path.join(FIT, f"scn{i}_{kind}_frozen-empirical-{WIN}.npy")
    return np.load(f, allow_pickle=True).item() if os.path.exists(f) else None


def shown():
    """Models in panels b–f: the reference (frozen only) and the three mixtures."""
    return ([REF] if FROZEN else []) + list(VARIANTS)


# ─── statistics ──────────────────────────────────────────────────────────────────────────────────
def model_R(r, d):
    ti = r["tm"].astype(int)
    return ti.astype(float), d["c"][ti] * np.abs(r["Zg"]), d["cA"][ti] * np.abs(r["ZA"]), d["cB"][ti] * np.abs(r["ZB"])


def kappa_d(t, R, ttx):
    sel = (t >= ttx + 24) & (t < ttx + 72) & np.isfinite(R) & (R > 0)
    return -np.polyfit(t[sel], np.log(R[sel]), 1)[0] * 24.0


def resync_rate(t, R, wash):
    """1/w (1/day) of a logistic fit R = a + b σ((t − t0)/w) on [wash − 72 h, wash + 96 h]; t0 free."""
    sel = (t >= wash - 72) & (t < wash + 96) & np.isfinite(R)
    tt, RR = t[sel], R[sel]
    a0, b0 = RR.min(), RR.max() - RR.min()
    t_half = tt[np.argmax(RR >= a0 + 0.5 * b0)]
    f = lambda x, a, b, t0, w: a + b / (1 + np.exp(-(x - t0) / w))
    try:
        p, _ = curve_fit(f, tt, RR, p0=[a0, b0, t_half, 6.0],
                         bounds=([0, 0, wash - 96, 0.5], [1, 1.2, wash + 72, 96]), maxfev=20000)
        return 24.0 / p[3], p[2] - wash
    except RuntimeError:
        return np.nan, np.nan


def cvm(om, mod, k):
    """Cramér–von Mises distance D = W²/N between the cells' frequencies om and lobe k of the mixture."""
    sel = mod["lobe"] == k; w = mod["w"][sel] / mod["w"][sel].sum()
    x = np.sort(om); n = x.size
    Fm = (w[None] * (0.5 + np.arctan((x[:, None] - mod["Om"][sel][None]) / mod["De"][sel][None]) / np.pi)).sum(1)
    return (np.sum((Fm - (np.arange(n) + 0.5) / n) ** 2) + 1 / (12 * n)) / n


def K_crit(mod, alpha, K_loc, gamma, K_hi=20.0):
    """Smallest K_s ≥ 0 at which the incoherent state of the linearised LMMF loses stability (bisection)."""
    Om, De, w, cl = mod["Om"], mod["De"], mod["w"], mod["lobe"]
    p = np.array([w[cl == k].sum() for k in (0, 1)])
    B = np.where(cl[:, None] == cl[None, :], w[None, :] / p[cl][:, None], 0.0)
    D0 = np.diag(1j * Om - De - gamma) + 0.5 * K_loc * B
    G = 0.5 * np.exp(-1j * alpha) * np.outer(np.ones(Om.size), w)
    growth = lambda K: np.max(np.linalg.eigvals(D0 + K * G).real)
    if growth(0.0) >= 0:
        return 0.0
    if growth(K_hi) < 0:
        return np.nan
    lo, hi = 0.0, K_hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if growth(mid) < 0 else (lo, mid)
    return hi


LEVELS = [("pre", ":", "o"), ("post", "-.", "s"), ("min", "--", "v")]     # data coherence levels


E_VARIANTS = {"log":    "log K, [0.01, 0.5]",
              "logz":   "log K, [0.015, 0.25]",
              "lin":    "linear K, [0, 0.2]",
              "rel":    r"log $K/K_c$ (each mixture), [1, 30]",
              "1-R":    r"log K vs log $(1 - R^*)$ (approach to full sync)",
              "rel1-R": r"log $K/K_c$ vs log $(1 - R^*)$"}
E_VARIANT = "log"


def draw_e(a, ex, bif, variant="log"):
    """Panel (e): synchronised branch R*(K) of the three mixtures with the data levels, x-axis variant."""
    rel = variant.startswith("rel"); comp = variant.endswith("1-R")
    for kind, lab, col, ls in VARIANTS:
        z = np.load(os.path.join(BM.OUT, f"scn{ex}_{kind}_bif.npz"))
        Kc = BM.K_crit(F.mixture(ex, kind, TG.load(ex))) if rel else 1.0
        x = z["Ks"] / Kc
        Y = (lambda r: 1 - r) if comp else (lambda r: r)
        if not comp:
            a.fill_between(x, Y(z["sync_min"]), Y(z["sync_max"]), color=col, alpha=0.15, lw=0)
        a.plot(x, Y(z["sync_mean"]), color=col, ls=ls, lw=0.9)
        for lev, lls, mk in LEVELS:
            Kx = bif[ex][kind]["K_" + lev]
            if np.isfinite(Kx):
                a.plot(Kx / Kc, Y(bif[ex]["R_" + lev]), mk, color=col, ms=3, mec="k", mew=0.3, zorder=3)
    for lev, lls, mk in LEVELS:
        a.axhline((1 - bif[ex]["R_" + lev]) if comp else bif[ex]["R_" + lev], color="k", lw=0.5, ls=lls, zorder=1)
    if variant == "lin":
        a.set_xlim(0, 0.2)
    else:
        a.set_xscale("log")
        a.set_xlim(*({"log": (0.01, 0.5), "logz": (0.015, 0.25), "rel": (1, 30), "1-R": (0.01, 0.5),
                      "rel1-R": (1, 30)}[variant]))
    if comp:
        a.set_yscale("log"); a.set_ylim(5e-3, 1.0); a.set_ylabel(r"$1 - R^*$")
    else:
        a.set_ylim(0, 1.02); a.set_ylabel(r"steady state $R^*$")
    a.set_xlabel(r"$K / K_c$" if rel else r"global coupling $K$ (rad/h)")
    a.set_title(f"SCN{ex}: coherence vs coupling")


def e_variants(ex):
    """Sheet with all x-axis variants of panel (e), for choosing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prl_style import set_prl_style, COL_DOUBLE
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center"})
    bif = bifurcation_levels()
    fig, axs = plt.subplots(2, 3, figsize=(COL_DOUBLE, 4.0),
                            gridspec_kw=dict(hspace=0.6, wspace=0.4, left=0.07, right=0.99, top=0.92, bottom=0.1))
    for a, (v, desc) in zip(axs.flat, E_VARIANTS.items()):
        draw_e(a, ex, bif, v); a.set_title(f"[{v}] {desc}", fontsize=5.5)
    out = os.path.join(OUT, "scn_lmmf_main_e_variants")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"[saved] {out}.png/.pdf")


def bifurcation_levels():
    """Per slice: ceiling-corrected data coherence before TTX, after washout and its MINIMUM (lowest R/c in
    [t_ttx, t_wash + 24 h]), and for each mixture the global coupling at which its synchronised steady state R*(K)
    (scn_bifurcation_mixtures) equals each level."""
    out = {}
    for i in sorted(set(SLICES) | {3}):
        d = TG.load(i); T = int(d["T"]); ttx, wash = int(d["ttx"]), int(d["wash"]); v0, v1 = S.valid_window(T)
        Rc = d["R"] / d["c"]
        lev = dict(pre=float(np.nanmean(Rc[TG.start_hour(d, v0):ttx - 24])),
                   post=float(np.nanmean(Rc[wash + 72:v1])), min=float(np.nanmin(Rc[ttx:wash + 24])))
        o = {f"R_{k}": v for k, v in lev.items()}
        for kind, *_ in VARIANTS:
            z = np.load(os.path.join(BM.OUT, f"scn{i}_{kind}_bif.npz"))
            o[kind] = {f"K_{k}": BM.invert(z["Ks"], z["sync_mean"], min(v, 0.999)) for k, v in lev.items()}
        out[i] = o
    return out


def stats():
    out = {}
    for i in SLICES:
        d = TG.load(i); ttx, wash = int(d["ttx"]), int(d["wash"]); t = np.arange(int(d["T"]), dtype=float)
        v0, v1 = S.valid_window(int(d["T"])); t0 = TG.start_hour(d, v0)
        levels = lambda tt, R: dict(pre=float(np.nanmean(R[(tt >= t0) & (tt < ttx - 24)])),
                                    min=float(np.nanmin(R[(tt >= ttx) & (tt < wash + 24)])),
                                    post=float(np.nanmean(R[(tt >= wash + 72) & (tt < v1)])))
        o = {"data": dict(lev=levels(t, d["R"]), kd=kappa_d(t, d["R"], ttx), kr=resync_rate(t, d["R"], wash),
                          kd_lobes=[kappa_d(t, d["RA"], ttx), kappa_d(t, d["RB"], ttx)]),
             "N": [int((d["lobe"] == k).sum()) for k in (0, 1)]}
        for kind, lab, *_ in shown():
            r = res_of(i, kind)
            mod = r["model"] if r is not None else F.mixture(i, kind, d)
            Dk = [cvm(d["om"][d["lobe"] == k], mod, k) for k in (0, 1)]
            o[kind] = dict(D=float(np.average(Dk, weights=o["N"])), D_lobes=Dk)
            if r is not None:
                tm, Rg, RA, RB = model_R(r, d)
                o[kind].update(lev=levels(tm, Rg), kd=kappa_d(tm, Rg, ttx), kr=resync_rate(tm, Rg, wash),
                               kd_lobes=[kappa_d(tm, RA, ttx), kappa_d(tm, RB, ttx)],
                               rmse=float(np.sqrt((r["rmse"] ** 2 + r["rmse_A"] ** 2 + r["rmse_B"] ** 2) / 3)),
                               rmse_ref=float(r.get("rmse_ref_comb", np.nan)), loss=float(r["loss"]))
        out[i] = o
    floor = float(np.mean([1 / (6 * n) for i in SLICES for n in out[i]["N"]]))
    return out, floor


# ─── figure ──────────────────────────────────────────────────────────────────────────────────────
def figure(st, floor, ex, plane):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from prl_style import set_prl_style, panel_label, COL_SINGLE
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center"})
    fig, axs = plt.subplots(3, 2, figsize=(COL_SINGLE, 4.6),
                            gridspec_kw=dict(hspace=0.62, wspace=0.42, left=0.13, right=0.98, top=0.97, bottom=0.08))
    d = TG.load(ex); T = int(d["T"]); v0, v1 = S.valid_window(T)
    # (a) frequency distribution, lobes pooled
    a = axs[0, 0]; f = d["om"] * CPD
    g = np.linspace(f.min() - 0.02, f.max() + 0.02, 800)
    a.hist(f, bins=30, density=True, color="0.82", lw=0)
    for kind, lab, col, ls in VARIANTS:
        r = SF.load(ex, kind); mod = r["model"] if r is not None else F.mixture(ex, kind, d)
        pdf = (mod["w"][:, None] * mod["De"][:, None] / np.pi
               / ((g[None] / CPD - mod["Om"][:, None]) ** 2 + mod["De"][:, None] ** 2)).sum(0) / CPD
        a.plot(g, pdf, color=col, ls=ls, lw=0.9, label=mlabel(ex, kind, lab))
    a.set_yticks([]); a.set_xlabel("frequency (cycles/day)"); a.set_ylabel("density")
    a.legend(loc="upper right", fontsize=4.6, handlelength=1.6, borderaxespad=0.1)
    a.text(0.03, 0.95, f"SCN{ex}", transform=a.transAxes, va="top", fontsize=6, fontweight="bold")
    # (b) coherence
    a = axs[0, 1]; td = np.arange(T) / 24
    a.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", lw=0, zorder=0)
    a.plot(td, np.where((np.arange(T) >= v0) & (np.arange(T) < v1), d["R"], np.nan), color="k", lw=1.0, zorder=3)
    for kind, lab, col, ls in shown():
        r = res_of(ex, kind)
        if r is not None:
            tm, Rg, *_ = model_R(r, d)
            a.plot(tm / 24, Rg, color=col, ls=ls, lw=1.1 if kind == "empirical" else 0.8, zorder=2,
                   alpha=0.8 if kind == "empirical" else 1.0, label=mlabel(ex, kind, lab))
    if FROZEN:
        a.legend(loc="lower right", fontsize=4.4, handlelength=1.6, borderaxespad=0.1)
    a.set_ylim(0, 1.02); a.set_xlabel("time (days)"); a.set_ylabel(r"coherence $R$")
    # (c) approximation error vs distribution error (old panel e)
    a = axs[1, 0]
    yk = "rmse_ref" if FROZEN else "rmse"
    for j, i in enumerate(SLICES):
        pts = [(st[i][k]["D"], st[i][k][yk]) for k, *_ in VARIANTS if yk in st[i][k]]
        if len(pts) > 1:
            a.plot(*zip(*pts), color="0.75", lw=0.5, zorder=1)
        for kind, _, col, _ in VARIANTS:
            if yk in st[i][kind]:
                a.plot(st[i][kind]["D"], st[i][kind][yk], MARK[j], color=col, ms=3.2, mec="k", mew=0.3, zorder=2)
    if FROZEN:
        e_ref = np.nanmean([st[i]["empirical"]["rmse"] for i in SLICES if "rmse" in st[i]["empirical"]])
        a.axhline(e_ref, color=REF[2], ls=":", lw=0.7)
        a.text(0.97, e_ref, "reference vs data ", transform=a.get_yaxis_transform(), ha="right", va="bottom",
               fontsize=4.4,
               color=REF[2])
        a.set_yscale("log")
    a.set_xscale("log"); a.set_xlabel(r"distribution error $D$")
    a.set_ylabel("approximation error" if FROZEN else "coherence RMSE")
    a.legend(handles=[Line2D([], [], marker=MARK[j], color="0.4", ls="none", ms=3, label=f"SCN{i}")
                      for j, i in enumerate(SLICES)], loc="upper left", fontsize=4.4, handletextpad=0.1,
             borderaxespad=0.2, ncol=2, columnspacing=0.6)
    # (d) coherence before TTX, minimum under TTX, after washout: model − data, per slice and model
    a = axs[1, 1]; mods = shown(); offs = np.linspace(-0.27, 0.27, len(mods))
    for j, i in enumerate(SLICES):
        for (kind, lab, col, ls), o in zip(mods, offs):
            if "lev" not in st[i][kind]:
                continue
            ys = [st[i][kind]["lev"][lev] - st[i]["data"]["lev"][lev] for lev, *_ in LEVELS]
            a.plot([j + o] * 3, ys, color=col, lw=0.5, alpha=0.6, zorder=1)
            for (lev, lls, mk), y in zip(LEVELS, ys):
                a.plot(j + o, y, mk, color=col, ms=3, mec="k", mew=0.3, zorder=2)
    a.axhline(0, color="0.5", lw=0.5, zorder=0)
    a.set_xticks(range(len(SLICES))); a.set_xticklabels([f"SCN{i}" for i in SLICES])
    a.set_ylabel(r"$R_{\rm model} - R_{\rm data}$"); a.set_title("coherence levels")
    a.legend(handles=[Line2D([], [], marker=mk, color="0.3", ls="none", ms=3, label=nm)
                      for (lev, lls, mk), nm in zip(LEVELS, ("pre-TTX", "post-washout", "minimum"))],
             loc="lower left", fontsize=4.4, handletextpad=0.1, borderaxespad=0.2)
    # (e) steady-state coherence R*(K) of the three mixtures (global coupling only), SCN example
    a = axs[2, 0]; bif = bifurcation_levels()
    draw_e(a, ex, bif, E_VARIANT)
    # (f) inferred couplings per slice: pre-TTX (circle), post-washout (square), coherence minimum (triangle)
    a = axs[2, 1]
    offs = {k: o for (k, *_), o in zip(VARIANTS, (-0.18, 0.0, 0.18))}
    for j, i in enumerate(SLICES):
        for kind, lab, col, ls in VARIANTS:
            ys = [bif[i][kind]["K_" + lev] for lev, *_ in LEVELS]
            a.plot([j + offs[kind]] * 3, ys, color=col, lw=0.5, alpha=0.6, zorder=1)
            for (lev, lls, mk), y in zip(LEVELS, ys):
                a.plot(j + offs[kind], y, mk, color=col, ms=3, mec="k", mew=0.3, zorder=2)
    a.set_xticks(range(len(SLICES))); a.set_xticklabels([f"SCN{i}" for i in SLICES])
    a.set_yscale("log"); a.set_ylabel(r"inferred $K$ (rad/h)"); a.set_title("inferred coupling")
    a.legend(handles=[Line2D([], [], marker=mk, color="0.3", ls="none", ms=3, label=nm)
                      for (lev, lls, mk), nm in zip(LEVELS, ("pre-TTX", "post-washout", "minimum"))],
             loc="upper right", fontsize=4.4, handletextpad=0.1, borderaxespad=0.2)
    axs[2, 0].legend(handles=[Line2D([], [], color=c, ls=ls, lw=0.9, label=mlabel(ex, k, l))
                              for k, l, c, ls in VARIANTS] +
                     [Line2D([], [], color="k", lw=0.5, ls=lls, label=f"data {nm}")
                      for (lev, lls, mk), nm in zip(LEVELS, ("pre-TTX", "post-washout", "minimum"))],
                     loc="center right", fontsize=4.0, handlelength=1.6, borderaxespad=0.2)
    for a_, l in zip(axs.flat, "abcdef"):
        panel_label(a_, l, dx=-20, dy=3)
    out = os.path.join(OUT, "scn_lmmf_main" + ("" if FROZEN else "_refit"))
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=400, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")


def main():
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    ex = arg("--example", 3, int); plane = arg("--plane", "gamma")
    global E_VARIANT
    E_VARIANT = arg("--e-variant", E_VARIANT)
    if "--e-variants" in sys.argv:
        e_variants(ex)
        return
    st, floor = stats()
    for i, o in st.items():
        print(f"SCN{i}: data κ_d {o['data']['kd']:.3f}, κ_r {o['data']['kr'][0]:.2f}/day (t0 {o['data']['kr'][1]:+.0f} h) | "
              + " | ".join(f"{k}: D {o[k]['D']:.2e}" + (f", RMSE {o[k]['rmse']:.3f}, vs ref {o[k]['rmse_ref']:.3f}, "
                                                        f"κ_d {o[k]['kd']:.3f}, κ_r {o[k]['kr'][0]:.2f}"
                                                        if 'rmse' in o[k] else "")
                           for k, *_ in shown()))
    print(f"sampling noise floor of D (mean over lobes): {floor:.2e}")
    with open(os.path.join(OUT, "scn_lmmf_main_stats" + ("" if FROZEN else "_refit") + ".json"), "w") as fh:
        json.dump(dict(stats={str(k): v for k, v in st.items()}, noise_floor=floor), fh, indent=1, default=float)
    figure(st, floor, ex, plane)


if __name__ == "__main__":
    main()
