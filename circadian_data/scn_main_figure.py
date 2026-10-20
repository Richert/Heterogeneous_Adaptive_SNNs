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
         averaged over the lobes (cell-weighted); the dashed line is the SAMPLING NOISE FLOOR E[D] = 1/(6N),
         averaged over the lobes of SCN1–4 (expected distance of an exact model, finite N)
      y  frozen: RMSE of c|Z| vs the empirical reference's c|Z_ref| over the mask (slice and both lobes combined);
         dotted: the reference's own error vs the data (mean over slices); --refit: RMSE of c|Z| vs R
      grey lines connect the three models of a slice
  (d) LOBE-SPECIFIC DESYNCHRONISATION rate κ_d = −d ln R_L/dt over [t_ttx + 24 h, t_ttx + 72 h] (1/day), model vs
      data, SCN1–4; filled = lobe A, open = lobe B; colour = model (incl. the empirical reference)
  (e) SCN3: steady-state coherence R*(K) of the three mixtures with ONE global coupling K (no lag, noise or local
      coupling; scn_bifurcation_mixtures, synchronised branch: mean and oscillation envelope), log K in [0.01, 0.5];
      horizontal lines: the ceiling-corrected data coherence before TTX, after washout and its minimum (lowest R/c
      in [t_ttx, t_wash + 24 h]); markers: the coupling at which each mixture's R* matches each level
  (f) these inferred couplings for SCN1–4 (circle: pre-TTX, square: post-washout, triangle: minimum), per mixture
  (needs scn_bifurcation_mixtures.py to have been run)

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_main_figure.py [--example 3] [--plane gamma|alpha] [--refit]
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
        o = {"data": dict(kd=kappa_d(t, d["R"], ttx), kr=resync_rate(t, d["R"], wash),
                          kd_lobes=[kappa_d(t, d["RA"], ttx), kappa_d(t, d["RB"], ttx)]),
             "N": [int((d["lobe"] == k).sum()) for k in (0, 1)]}
        for kind, lab, *_ in shown():
            r = res_of(i, kind)
            mod = r["model"] if r is not None else F.mixture(i, kind, d)
            Dk = [cvm(d["om"][d["lobe"] == k], mod, k) for k in (0, 1)]
            o[kind] = dict(D=float(np.average(Dk, weights=o["N"])), D_lobes=Dk)
            if r is not None:
                tm, Rg, RA, RB = model_R(r, d)
                o[kind].update(kd=kappa_d(tm, Rg, ttx), kr=resync_rate(tm, Rg, wash),
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
        a.plot(g, pdf, color=col, ls=ls, lw=0.9, label=lab)
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
                   alpha=0.8 if kind == "empirical" else 1.0, label=lab)
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
    a.axvline(floor, color="0.3", ls="--", lw=0.6)
    a.text(floor, 0.03, "noise floor ", transform=a.get_xaxis_transform(), ha="right", va="bottom", fontsize=4.6,
           color="0.3")
    if FROZEN:
        e_ref = np.nanmean([st[i]["empirical"]["rmse"] for i in SLICES if "rmse" in st[i]["empirical"]])
        a.axhline(e_ref, color=REF[2], ls=":", lw=0.7)
        a.text(0.03, e_ref, " reference vs data", transform=a.get_yaxis_transform(), va="bottom", fontsize=4.4,
               color=REF[2])
        a.set_yscale("log")
    a.set_xscale("log"); a.set_xlabel(r"distribution error $D$")
    a.set_ylabel("approximation error" if FROZEN else "coherence RMSE")
    a.legend(handles=[Line2D([], [], marker=MARK[j], color="0.4", ls="none", ms=3, label=f"SCN{i}")
                      for j, i in enumerate(SLICES)], loc="upper left", fontsize=4.4, handletextpad=0.1,
             borderaxespad=0.2, ncol=2, columnspacing=0.6)
    # (d) lobe-specific desynchronisation rates: filled = lobe A, open = lobe B
    a = axs[1, 1]; hi = 0.0
    for j, i in enumerate(SLICES):
        xd = st[i]["data"]["kd_lobes"]
        for kind, _, col, _ in shown():
            if "kd_lobes" not in st[i][kind]:
                continue
            for k_ in (0, 1):
                y = st[i][kind]["kd_lobes"][k_]
                a.plot(xd[k_], y, MARK[j], color=col, mfc=col if k_ == 0 else "white", ms=3.2, mec=col if k_ else "k",
                       mew=0.3 if k_ == 0 else 0.7, zorder=3 - k_)
                hi = max(hi, xd[k_], y)
    hi = 1.08 * hi
    a.plot([0, hi], [0, hi], color="0.6", lw=0.5, zorder=0); a.set_xlim(0, hi); a.set_ylim(0, hi)
    a.set_xlabel(r"data $\kappa_d$ (1/day)"); a.set_ylabel(r"model $\kappa_d$ (1/day)")
    a.set_title("desynchronisation (lobes)")
    a.legend(handles=[Line2D([], [], marker="o", color="0.3", ls="none", ms=3, label="lobe A"),
                      Line2D([], [], marker="o", color="0.3", mfc="white", ls="none", ms=3, label="lobe B")],
             loc="upper left", fontsize=4.4, handletextpad=0.1, borderaxespad=0.2)
    # (e) steady-state coherence R*(K) of the three mixtures (global coupling only), SCN example
    a = axs[2, 0]; bif = bifurcation_levels()
    for kind, lab, col, ls in VARIANTS:
        z = np.load(os.path.join(BM.OUT, f"scn{ex}_{kind}_bif.npz"))
        a.fill_between(z["Ks"], z["sync_min"], z["sync_max"], color=col, alpha=0.15, lw=0)
        a.plot(z["Ks"], z["sync_mean"], color=col, ls=ls, lw=0.9, label=lab)
    for lev, lls, mk in LEVELS:
        a.axhline(bif[ex]["R_" + lev], color="k", lw=0.5, ls=lls, zorder=1)
        for kind, lab, col, ls in VARIANTS:
            Kx = bif[ex][kind]["K_" + lev]
            if np.isfinite(Kx):
                a.plot(Kx, bif[ex]["R_" + lev], mk, color=col, ms=3, mec="k", mew=0.3, zorder=3)
    a.set_xscale("log"); a.set_xlim(0.01, 0.5); a.set_ylim(0, 1.02)
    a.set_xlabel(r"global coupling $K$ (rad/h)"); a.set_ylabel(r"steady state $R^*$")
    a.set_title(f"SCN{ex}: coherence vs coupling")
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
    axs[2, 0].legend(handles=[Line2D([], [], color=c, ls=ls, lw=0.9, label=l) for _, l, c, ls in VARIANTS] +
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
