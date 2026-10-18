r"""
Two-column PRL figure + statistics of the SCN LMMF fits (write-up)
==================================================================

Winning dynamics (setup C): homeostatic TTX-sensitive coupling + TTX noise γ u(t) + lobe-specific transient noise
after TTX application + TTX-insensitive lobe-local coupling, in the identifiable parametrisation (scn_model_fit
"hr-noise+transient+local": block depth ρ_min, plateau ρ_ttx, recovery time t_hom). Three FREQUENCY DISTRIBUTIONS
with the same dynamics, at decreasing LEVEL OF DETAIL (scn_model_fit.mixture kinds; per slice and lobe):
  no floor           Lorentzian mixture without the noise floor (most components; "nofloor")
  noise floor        Lorentzian mixture with the noise floor (the winning model; "floor")
  single Lorentzian  one Lorentzian (M_max = 1; "single")
How the level of detail at which the frequency distribution is captured translates into the predicted coherence
dynamics. (Before 2026-10-17 the figure compared the winning model with the minimal
3-level coupling model; --legacy restores that comparison.)

Panels:
  (a) intrinsic (TTX-epoch) frequency distributions per slice: histogram of all cells, mixture density
      (black) and its lobe parts (lobe colours, weighted by the lobe fractions)
  (b) slice coherence R(t): data (epoch-peak pipeline) vs c(t)|Z| of the three models
  (c) K_s(t)/K_pre of the noise-floor model vs time since TTX application (washout marked)
  (d) DESYNCHRONISATION rate κ_d = −d ln R/dt over [t_ttx + 24 h, t_ttx + 72 h] (1/day), data vs model,
      slice and lobes
  (e) RESYNCHRONISATION time t_½: time relative to washout at which R first exceeds the midpoint between its
      minimum (from 4 days before to 12 h after washout) and the post-washout plateau (h); data vs model
  (f) post-washout plateau relative to pre-TTX coherence, R_post / R_pre, data vs model
Statistics on stdout and in writeup/scn_lmmf_fit_stats.json.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_writeup_figure.py [--legacy] [--fit-dir DIR]
Output: data_paths.MPMF/circadian_data/writeup/{scn_lmmf_fits_prl.pdf/.png, scn_lmmf_fit_stats.json}
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_model_fit as F

OUT = dp.ensure(os.path.join(S.OUT, "writeup"))
FIT = os.path.join(S.OUT, "model_fit", "shapes")
SLICES = tuple(int(v) for v in sys.argv[sys.argv.index("--slices") + 1].split(",")) if "--slices" in sys.argv else (1, 2, 3, 4)
WIN = "hr-noise+transient+local"
SHOW = [("nofloor", WIN, "no floor", "#2166ac", "--"), ("floor", WIN, "noise floor", "#c0392b", "-"),
        ("single", WIN, "single Lorentzian", "0.45", ":")]
LEGACY = [("floor", "hf-noise+transient+local", "winning", "#c0392b", "-"), ("floor", "base", "minimal", "0.5", "--")]
MARK = ["o", "s", "D", "^"]                                  # slices 1–4 in the statistics panels
C_SL = ["#2a6fdb", "#e0782b", "#3a9e5c", "#b8418f"]          # slices 1–4 (fixed order)
C_LOBE = ["#1b7837", "#762a83"]
CPD = 24 / (2 * np.pi)


def load(i, name, kind="floor"):
    f = os.path.join(FIT, f"scn{i}_{kind}_{name}.npy")
    return np.load(f, allow_pickle=True).item() if os.path.exists(f) else None


def model_R(r, d):
    """Model observables c(t)|Z| on the hourly grid of the fit (slice, lobe A, lobe B)."""
    ti = r["tm"].astype(int)
    return ti, d["c"][ti] * np.abs(r["Zg"]), d["cA"][ti] * np.abs(r["ZA"]), d["cB"][ti] * np.abs(r["ZB"])


def kappa(t, R, a, b):
    sel = (t >= a) & (t < b) & np.isfinite(R) & (R > 0)
    return -np.polyfit(t[sel], np.log(R[sel]), 1)[0] * 24.0


def resync(t, R, wash, v1):
    lo = (t >= wash - 96) & (t < wash + 12) & np.isfinite(R)
    k = np.where(lo)[0][np.argmin(R[lo])]
    R_hi = np.nanmean(R[(t >= wash + 72) & (t < v1)])
    thr = 0.5 * (R[k] + R_hi)
    up = np.where((np.arange(t.size) > k) & (R >= thr))[0]
    return (t[up[0]] - wash if up.size else np.nan), t[k] - wash, R[k]


def stats(i):
    d = TG.load(i); T = int(d["T"]); ttx, wash = int(d["ttx"]), int(d["wash"]); v0, v1 = S.valid_window(T)
    t = np.arange(T, dtype=float)
    series = {"data": (t, d["R"], d["RA"], d["RB"])}
    fits = {}
    for kind, name, lab, *_ in SHOW:
        r = load(i, name, kind)
        if r is None:
            continue
        fits[lab] = r; ti, Rg, RA, RB = model_R(r, d); series[lab] = (ti.astype(float), Rg, RA, RB)
    out = {}
    for lab, (tt, Rg, RA, RB) in series.items():
        th, tmin, Rmin = resync(tt, Rg, wash, v1)
        pre = np.nanmean(Rg[(tt >= v0) & (tt < ttx - 24)]); post = np.nanmean(Rg[(tt >= wash + 72) & (tt < v1)])
        out[lab] = dict(kappa=[kappa(tt, X, ttx + 24, ttx + 72) for X in (Rg, RA, RB)],
                        t_half=th, t_min=tmin, R_min=Rmin, R_pre=pre, R_post=post, ratio=post / pre)
    out["fit"] = {}
    for lab, r in fits.items():
        p = r["params"]
        out["fit"][lab] = dict(rmse=[r["rmse"], r["rmse_A"], r["rmse_B"]], loss=r["loss"], n_free=r["n_free"],
                               M=int(r["model"]["Om"].size),
                               params=dict(K_pre=p["K_pre"], alpha_s=p["alpha_s"], gamma=p["gamma"], u_res=p["u_res"],
                                           tau_off=p["tau_off"], g_trA=p["g_trA"], g_trB=p["g_trB"],
                                           tau_tr=p["tau_tr"], K_loc=p["K_i"],
                                           **(F.identifiable(p, d) if p.get("shape") == "homeo" else {})))
    return out


def figure(st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prl_style import set_prl_style, panel_label, COL_DOUBLE
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center"})
    fig = plt.figure(figsize=(COL_DOUBLE, 5.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.0, 1.0], hspace=0.55, wspace=0.38,
                          left=0.07, right=0.985, top=0.95, bottom=0.08)
    # (a) frequency distributions: data histogram and the three model distributions
    for j, i in enumerate(SLICES):
        a = fig.add_subplot(gs[0, j]); d = TG.load(i)
        f = d["om"] * CPD
        a.hist(f, bins=28, density=True, color="0.85", lw=0)
        g = np.linspace(f.min() - 0.03, f.max() + 0.03, 600) / CPD
        for kind, name, lab, col, ls in SHOW:
            r = load(i, name, kind)
            mod = r["model"] if r is not None else F.mixture(i, kind, d)
            pdf = (mod["w"][:, None] * mod["De"][:, None] / np.pi
                   / ((g[None] - mod["Om"][:, None]) ** 2 + mod["De"][:, None] ** 2)).sum(0) / CPD
            a.plot(g * CPD, pdf, color=col, lw=1.0, ls=ls, label=lab)
        a.set_title(f"SCN{i}  (N = {f.size})"); a.set_yticks([]); a.set_xlabel("frequency (cycles/day)")
        if j == 0:
            a.set_ylabel("density"); panel_label(a, "a", dx=-14, dy=6)
            a.legend(loc="upper left", fontsize=5, handlelength=1.6)
    # (b) R(t)
    axb = []
    for j, i in enumerate(SLICES):
        a = fig.add_subplot(gs[1, j], sharey=axb[0] if axb else None); axb.append(a)
        d = TG.load(i); T = int(d["T"]); v0, v1 = S.valid_window(T); td = np.arange(T) / 24
        a.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", lw=0, zorder=0)
        for tc in (d["ttx"], d["wash"]):
            a.axvspan((tc - 24) / 24, (tc + 24) / 24, color="none", hatch="////", ec="0.8", lw=0, zorder=0)
        R = np.where((np.arange(T) >= v0) & (np.arange(T) < v1), d["R"], np.nan)
        a.plot(td, R, color="k", lw=1.1, label="data", zorder=3)
        txt = []
        for kind, name, lab, col, ls in SHOW:
            r = load(i, name, kind)
            if r is None:
                continue
            ti, Rg, *_ = model_R(r, d)
            a.plot(ti / 24, Rg, color=col, lw=0.9, ls=ls, label=lab, zorder=2); txt.append(f"{r['rmse']:.3f}")
        a.text(0.5, 0.97, "RMSE " + " / ".join(txt), transform=a.transAxes, ha="center", va="top", fontsize=5.2)
        a.set_ylim(0, 1.02); a.set_xlabel("time (days)"); a.set_title(f"SCN{i}")
        if j == 0:
            a.set_ylabel(r"coherence $R(t)$"); panel_label(a, "b", dx=-14, dy=6)
            a.legend(loc="lower left", fontsize=5, handlelength=1.6)
        else:
            a.tick_params(labelleft=False)
    # (c) K_s(t) / K_pre
    a = fig.add_subplot(gs[2, 0])
    for j, i in enumerate(SLICES):
        d = TG.load(i); r = load(i, WIN, "floor")
        if r is None:
            continue
        p = r["params"]; tt = np.arange(0, int(d["T"]), 0.25)
        K = F.Kfun(p, d)(tt) / p["K_pre"]
        a.plot((tt - d["ttx"]) / 24, np.maximum(K, 1e-4), color=C_SL[j], lw=0.9, label=f"SCN{i}")
        a.plot([(d["wash"] - d["ttx"]) / 24] * 2, [1.6e-4, 3e-4], color=C_SL[j], lw=1.2)
    a.set_yscale("log"); a.set_ylim(1e-4, 20); a.set_xlim(-1, 9)
    a.set_xlabel("days since TTX"); a.set_ylabel(r"$K_s(t)/K_{\rm pre}$"); a.set_title("homeostatic coupling")
    a.legend(loc="upper right", fontsize=5, ncol=2, handlelength=1.2); panel_label(a, "c", dx=-22, dy=6)

    def scatter(a, key, label, letter, lims, unit, lobes=False):
        """Data vs model: colour = model, marker = slice; lobes as small, faint markers (if lobes)."""
        for j, i in enumerate(SLICES):
            for kind, name, lab, col, ls in SHOW:
                if lab not in st[i]:
                    continue
                xs, ys = st[i]["data"][key], st[i][lab][key]
                xs, ys = (xs, ys) if isinstance(xs, list) else ([xs], [ys])
                for k_ in range(len(xs) if lobes else 1):
                    a.plot(xs[k_], ys[k_], MARK[j], ms=4 if k_ == 0 else 2.6, color=col,
                           alpha=1.0 if k_ == 0 else 0.55, mew=0.5, mec="k" if k_ == 0 else col, ls="none")
        a.plot(lims, lims, color="0.6", lw=0.6, zorder=0)
        a.set_xlim(lims); a.set_ylim(lims); a.set_xlabel(f"data {unit}"); a.set_ylabel(f"model {unit}")
        a.set_title(label); panel_label(a, letter, dx=-22, dy=6)
    a = fig.add_subplot(gs[2, 1])
    scatter(a, "kappa", "desynchronisation", "d", [0, 0.9], r"$\kappa_d$ (1/day)", lobes=True)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker=MARK[j], color="0.3", ls="none", ms=3.5, label=f"SCN{i}")
                      for j, i in enumerate(SLICES)] +
             [Line2D([], [], marker="o", color="0.3", alpha=0.55, ls="none", ms=2.6, label="lobes")],
             loc="upper left", fontsize=5, handletextpad=0.2, ncol=1)
    a = fig.add_subplot(gs[2, 2])
    scatter(a, "t_half", "resynchronisation", "e", [-40, 40], r"$t_{1/2}$ (h)")
    a.axhline(0, color="0.85", lw=0.5, zorder=0); a.axvline(0, color="0.85", lw=0.5, zorder=0)
    a = fig.add_subplot(gs[2, 3])
    scatter(a, "ratio", "post-washout plateau", "f", [0.85, 1.14], r"$R_{\rm post}/R_{\rm pre}$")
    out = os.path.join(OUT, "scn_lmmf_fits_prl")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")


def main():
    global SHOW, FIT, OUT
    if "--legacy" in sys.argv:
        SHOW = LEGACY
    if "--fit-dir" in sys.argv:                            # e.g. a smoke-test directory
        FIT = sys.argv[sys.argv.index("--fit-dir") + 1]; OUT = dp.ensure(os.path.join(FIT, "writeup"))
    st = {i: stats(i) for i in SLICES}
    for i, s_ in st.items():
        print(f"\nSCN{i}")
        for lab, v in s_.items():
            if lab == "fit":
                continue
            print(f"  {lab:12s} κ_d slice/A/B {np.round(v['kappa'], 3).tolist()} 1/day | t_½ {v['t_half']:+.0f} h "
                  f"(min R {v['R_min']:.2f} at {v['t_min']:+.0f} h) | R_pre {v['R_pre']:.3f} R_post {v['R_post']:.3f} "
                  f"ratio {v['ratio']:.3f}")
        for lab, f in s_["fit"].items():
            print(f"  fit {lab:12s} M = {f['M']}, RMSE slice/A/B {np.round(f['rmse'], 3).tolist()}, loss {f['loss']:.5f}"
                  f" | " + ", ".join(f"{k} {v:.3g}" for k, v in f["params"].items()))
    with open(os.path.join(OUT, "scn_lmmf_fit_stats.json"), "w") as fh:
        json.dump({str(k): v for k, v in st.items()}, fh, indent=1, default=float)
    figure(st)


if __name__ == "__main__":
    main()
