r"""
Main figure (single-column PRL): the level of detail of the frequency distribution and the coherence dynamics
=============================================================================================================

Same dynamics in all three models (setup C; scn_model_fit "hr-noise+transient+local"), frequency distribution per
slice and lobe captured by: no floor (λ = 1e-5) | noise floor | single Lorentzian (M_max = 1).

  (a) SCN3: intrinsic-frequency histogram of all cells (lobes pooled) and the three model distributions (slice level:
      lobe mixtures weighted by the lobe fractions)
  (b) SCN3: coherence R(t), data and c(t)|Z(t)| of the three models
  (c) DESYNCHRONISATION rate κ_d = −d ln R/dt over [t_ttx + 24 h, t_ttx + 72 h] (1/day): model vs data, SCN1–4
  (d) RESYNCHRONISATION rate 1/w (1/day) of a logistic fit R(t) = a + b σ((t − t_0)/w) over [t_wash − 72 h,
      t_wash + 96 h]; the onset t_0 is free, so the rate does not depend on when resynchronisation starts. The data
      rise lies largely inside the masked window, where the peak pipeline smears abrupt changes over about ±1 cycle:
      data rates are lower bounds.
  (e) error of the coherence fit vs error of the frequency-distribution fit, per slice and model:
      x  Cramér–von Mises distance D = W²/N between each lobe's model distribution and its cells' frequencies,
         averaged over the lobes (cell-weighted); the dashed line is the SAMPLING NOISE FLOOR E[D] = 1/(6N),
         averaged over the lobes of SCN1–4 (expected distance of an exact model, finite N)
      y  RMSE of c|Z| vs R over the mask, slice and both lobes combined (the fitted quantity)
      grey lines connect the three models of a slice
  (f) SCN3: critical coupling of the incoherent state, K_c(γ), for each model's distribution (with its fitted lag α_s
      and lobe-local coupling K_loc): smallest K_s at which the linearised LMMF
          ż_m = (iΩ_m − Δ_m − γ) z_m + ½ [K_s e^{−iα_s} Σ_n w_n z_n + K_loc Σ_{n ∈ L(m)} w_n z_n / p_L(m)]
      has an eigenvalue with Re λ = 0; and the fitted trajectory (K_s(t), γ u(t)) of each model, with the pre-TTX
      state (circle), the TTX plateau (triangle) and the post-washout state (square). --plane alpha: (K_s, α) instead.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_main_figure.py [--example 3] [--plane gamma|alpha]
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

OUT = dp.ensure(os.path.join(S.OUT, "writeup"))
SLICES = (1, 2, 3, 4)
MARK = ["o", "s", "D", "^"]
CPD = 24 / (2 * np.pi)
VARIANTS = SF.VARIANTS                                      # (kind, label, colour, linestyle)


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


def stats():
    out = {}
    for i in SLICES:
        d = TG.load(i); ttx, wash = int(d["ttx"]), int(d["wash"]); t = np.arange(int(d["T"]), dtype=float)
        o = {"data": dict(kd=kappa_d(t, d["R"], ttx), kr=resync_rate(t, d["R"], wash)),
             "N": [int((d["lobe"] == k).sum()) for k in (0, 1)]}
        for kind, lab, *_ in VARIANTS:
            r = SF.load(i, kind)
            mod = r["model"] if r is not None else F.mixture(i, kind, d)
            Dk = [cvm(d["om"][d["lobe"] == k], mod, k) for k in (0, 1)]
            o[kind] = dict(D=float(np.average(Dk, weights=o["N"])), D_lobes=Dk)
            if r is not None:
                tm, Rg, *_ = model_R(r, d)
                o[kind].update(kd=kappa_d(tm, Rg, ttx), kr=resync_rate(tm, Rg, wash),
                               rmse=float(np.sqrt((r["rmse"] ** 2 + r["rmse_A"] ** 2 + r["rmse_B"] ** 2) / 3)))
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
    fig, axs = plt.subplots(3, 2, figsize=(COL_SINGLE, 4.5),
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
    for kind, lab, col, ls in VARIANTS:
        r = SF.load(ex, kind)
        if r is not None:
            tm, Rg, *_ = model_R(r, d); a.plot(tm / 24, Rg, color=col, ls=ls, lw=0.8, zorder=2)
    a.set_ylim(0, 1.02); a.set_xlabel("time (days)"); a.set_ylabel(r"coherence $R$")
    # (c), (d) rates
    for a, key, lab, lims in ((axs[1, 0], "kd", r"$\kappa_d$", (0, 0.8)), (axs[1, 1], "kr", r"$\kappa_r$", "log")):
        xs, ys = [], []
        for j, i in enumerate(SLICES):
            xd = st[i]["data"][key]; xd = xd[0] if isinstance(xd, (tuple, list)) else xd
            for kind, _, col, _ in VARIANTS:
                if key in st[i][kind]:
                    y = st[i][kind][key]; y = y[0] if isinstance(y, (tuple, list)) else y
                    a.plot(xd, y, MARK[j], color=col, ms=3.2, mec="k", mew=0.3); xs.append(xd); ys.append(y)
        if lims == "log":                                  # resync rates span an order of magnitude
            v = np.array(xs + ys, float); v = v[np.isfinite(v) & (v > 0)]
            lo_, hi = 0.7 * v.min(), 1.4 * v.max()
            a.set_xscale("log"); a.set_yscale("log"); a.plot([lo_, hi], [lo_, hi], color="0.6", lw=0.5, zorder=0)
            a.set_xlim(lo_, hi); a.set_ylim(lo_, hi)
        else:
            hi = lims[1]
            a.plot([0, hi], [0, hi], color="0.6", lw=0.5, zorder=0); a.set_xlim(0, hi); a.set_ylim(0, hi)
        a.set_xlabel(f"data {lab} (1/day)"); a.set_ylabel(f"model {lab} (1/day)")
    axs[1, 0].set_title("desynchronisation"); axs[1, 1].set_title("resynchronisation")
    axs[1, 0].legend(handles=[Line2D([], [], marker=MARK[j], color="0.4", ls="none", ms=3, label=f"SCN{i}")
                              for j, i in enumerate(SLICES)], loc="upper left", fontsize=4.6, handletextpad=0.1,
                     borderaxespad=0.2)
    # (e) error vs error
    a = axs[2, 0]
    for j, i in enumerate(SLICES):
        pts = [(st[i][k]["D"], st[i][k]["rmse"]) for k, *_ in VARIANTS if "rmse" in st[i][k]]
        if len(pts) > 1:
            a.plot(*zip(*pts), color="0.75", lw=0.5, zorder=1)
        for kind, _, col, _ in VARIANTS:
            if "rmse" in st[i][kind]:
                a.plot(st[i][kind]["D"], st[i][kind]["rmse"], MARK[j], color=col, ms=3.2, mec="k", mew=0.3, zorder=2)
    a.axvline(floor, color="0.3", ls="--", lw=0.6)
    a.text(floor, 0.03, "noise floor ", transform=a.get_xaxis_transform(), ha="right", va="bottom", fontsize=4.6,
           color="0.3")
    a.set_xscale("log"); a.set_xlabel(r"distribution error $D$"); a.set_ylabel(r"coherence RMSE")
    # (f) critical coupling and trajectories
    a = axs[2, 1]
    g_fit = [SF.load(ex, k)["params"]["gamma"] for k, *_ in VARIANTS if SF.load(ex, k) is not None]
    gam = np.linspace(0, max(2.5 * max(g_fit + [0.0]), 0.005), 61) if plane == "gamma" else None
    al = np.linspace(-0.6, 0.6, 61) if plane == "alpha" else None
    for kind, lab, col, ls in VARIANTS:
        r = SF.load(ex, kind); mod = r["model"] if r is not None else F.mixture(ex, kind, d)
        p = r["params"] if r is not None else dict(alpha_s=0.0, K_i=0.0)
        if plane == "gamma":
            Kc = [K_crit(mod, p["alpha_s"], p["K_i"], x) for x in gam]; a.plot(Kc, gam * 1e3, color=col, ls=ls, lw=0.9)
        else:
            Kc = [K_crit(mod, x, p["K_i"], 0.0) for x in al]; a.plot(Kc, al, color=col, ls=ls, lw=0.9)
        if r is not None:
            tt = np.arange(v0, v1, 0.5); K = F.Kfun(p, d)(tt); u = F.ufun(p, d)(tt)
            y = p["gamma"] * u * 1e3 if plane == "gamma" else np.full(tt.size, p["alpha_s"])
            a.plot(np.maximum(K, 1e-4), y, color=col, lw=0.5, alpha=0.6)
            for t_, mk in ((d["ttx"] - 24, "o"), (d["wash"] - 1, "v"), (v1 - 1, "s")):
                k_ = np.argmin(np.abs(tt - t_)); a.plot(max(K[k_], 1e-4), y[k_], mk, color=col, ms=3, mec="k", mew=0.3)
    a.set_xscale("log"); a.set_xlim(1e-3, 3)
    a.set_xlabel(r"coupling $K_s$ (rad/h)")
    a.set_ylabel(r"noise $\gamma u$ ($10^{-3}$ rad/h)" if plane == "gamma" else r"phase lag $\alpha_s$ (rad)")
    a.set_title("critical coupling")
    for a_, l in zip(axs.flat, "abcdef"):
        panel_label(a_, l, dx=-20, dy=3)
    out = os.path.join(OUT, "scn_lmmf_main")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=400, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")


def main():
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    ex = arg("--example", 3, int); plane = arg("--plane", "gamma")
    st, floor = stats()
    for i, o in st.items():
        print(f"SCN{i}: data κ_d {o['data']['kd']:.3f}, κ_r {o['data']['kr'][0]:.2f}/day (t0 {o['data']['kr'][1]:+.0f} h) | "
              + " | ".join(f"{k}: D {o[k]['D']:.2e}" + (f", RMSE {o[k]['rmse']:.3f}, κ_d {o[k]['kd']:.3f}, "
                                                        f"κ_r {o[k]['kr'][0]:.2f}" if 'rmse' in o[k] else "")
                           for k, *_ in VARIANTS))
    print(f"sampling noise floor of D (mean over lobes): {floor:.2e}")
    with open(os.path.join(OUT, "scn_lmmf_main_stats.json"), "w") as fh:
        json.dump(dict(stats={str(k): v for k, v in st.items()}, noise_floor=floor), fh, indent=1, default=float)
    figure(st, floor, ex, plane)


if __name__ == "__main__":
    main()
