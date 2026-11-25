r"""
Deconvolving Lorentzian-mixture fit of the intrinsic (TTX) frequencies with the noise-floor criterion
=====================================================================================================

Observation model for cell i (frequencies in cycles/day; peak method `epoch` or `wavelet`, scn_frequency_figure):
    x_i = f_i + e_i,   f_i ~ Σ_m w_m Lorentz(μ_m, γ_m)                         (intrinsic frequency distribution)
    e_i ~ (1 − π_o) N(0, (c · SE_i)²) + π_o Cauchy(0, σ_o)                     (measurement error)
SE_i is the cell's regression SE (scn_frequency_figure.frequencies); (c, π_o, σ_o) are CALIBRATED once per peak method
on the ground-truth validation (synthetic traces with known frequencies; maximum likelihood on the errors, pooled over
slices): c rescales the Gaussian core, the Cauchy component captures cycle slips. The predictive CDF of cell i is
    F_i(x) = Σ_m w_m [ (1 − π_o) E_{e∼N(0,(c SE_i)²)} L_m(x − e) + π_o L(x; μ_m, γ_m + σ_o) ]
(Gauss–Hermite quadrature for the Voigt part; Lorentz ⊗ Cauchy is again a Lorentzian).

FIT (deconvolved): minimise the Cramér–von Mises distance of the probability-integral transforms u_i = F_i(x_i) from the
uniform distribution, D = [1/(12N) + Σ_k (u_(k) − (2k−1)/(2N))²] / N — under the true model the u_i are i.i.d. uniform
even though every cell has its own error, so E[D] = 1/(6N) and the NOISE FLOOR of the LMMF method carries over
unchanged. Greedy in M (warm start from the M − 1 fit + a component at the largest PIT residual, plus random
restarts), M* = argmin_{visited M} D(M) + λ M up to the first M with D ≤ 1/(6N) (λ = LAM, absolute, as in
shared/lorentzian_mixture). NAIVE fit: the same with no error model (c = π_o = 0), i.e. the usual noise-floor fit of the
noisy x_i.

Steps:
  calib     error-model calibration per method (validation errors of scn_frequency_figure.validate, TTX condition)
  fit       per slice and method: naive and deconvolved fits of the TTX frequencies + bootstrap over cells (B_BOOT)
  recover   synthetic recovery: KNOWN intrinsic distributions (unimodal broad / bimodal / narrow) → per-cell TTX
            frequencies → synthetic traces with a slice's trend, envelope and noise (templates REC_TEMPLATES) → the
            same peak method → naive vs deconvolved fits; recovered M* and widths vs the truth (REC_SEEDS repeats)
  figure    diagnostic figure

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_deconvolution.py calib fit recover figure
        [--methods epoch,wavelet] [--slices 1,2,3,4,5] [--procs 8] [--boot 30] [--seeds 4]
Output: data_paths.MPMF/circadian_data/deconvolution/
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_frequency_figure as FF

OUT = dp.ensure(os.path.join(S.OUT, "deconvolution"))
LAM = 1e-5
M_MAX = 5
N_RESTART = 6
GH_T, GH_W = np.polynomial.hermite.hermgauss(24)
GH_W = GH_W / np.sqrt(np.pi)
G_MIN = 1e-4                                  # smallest width in standardised units
B_BOOT = 30
NAIVE_ONLY = "--naive-only" in sys.argv     # 2026-11-25: the user prefers the naive (no error model) fit
REC_TEMPLATES = (1, 2, 4)
REC_SEEDS = 4
SCENARIOS = {                                 # intrinsic TTX distributions (c/d): (weights, centres, half-widths)
    "broad unimodal": ([1.0], [0.96], [0.025]),
    "bimodal": ([0.55, 0.45], [0.92, 1.01], [0.012, 0.012]),
    "narrow unimodal": ([1.0], [0.965], [0.004]),
}


# ─── error model ─────────────────────────────────────────────────────────────────────────────────
def error_pdf(e, se, c, po, so):
    g = np.exp(-0.5 * (e / (c * se)) ** 2) / (np.sqrt(2 * np.pi) * c * se)
    return (1 - po) * g + po * so / (np.pi * (e ** 2 + so ** 2))


def calibrate(method, slices):
    E, SE = [], []
    for i in slices:
        v = FF.validate(i, method)["ttx"]
        m = np.isfinite(v["est"]) & np.isfinite(v["se"])
        E.append((v["est"] - v["true"])[m]); SE.append(v["se"][m])
    E, SE = np.concatenate(E), np.concatenate(SE)
    nll = lambda p: -np.sum(np.log(error_pdf(E, SE, np.exp(p[0]), 1 / (1 + np.exp(-p[1])), np.exp(p[2])) + 1e-300))
    r = minimize(nll, [0.0, -3.0, np.log(0.1)], method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6))
    c, po, so = float(np.exp(r.x[0])), float(1 / (1 + np.exp(-r.x[1]))), float(np.exp(r.x[2]))
    z = E / (c * SE)
    out = dict(c=c, pi_o=po, sigma_o=so, n=int(E.size), robust_z_sd=float(1.4826 * np.median(np.abs(z))))
    json.dump(out, open(os.path.join(OUT, f"error_model_{method}.json"), "w"), indent=1)
    print(f"[calib {method}] c = {c:.3f}, π_o = {po:.3f}, σ_o = {so:.3f} c/d (n = {E.size})", flush=True)
    return out


def load_calib(method):
    return json.load(open(os.path.join(OUT, f"error_model_{method}.json")))


# ─── mixture CDF, PIT and fitting ────────────────────────────────────────────────────────────────
def unpack(p, M):
    mu = p[:M]; g = G_MIN + np.exp(p[M:2 * M]); w = np.exp(np.r_[0.0, p[2 * M:]]); return w / w.sum(), mu, g


def pit(p, M, xs, ss, po, so_s):
    """u_i = F_i(x_i) in standardised units (xs, Gaussian error SDs ss = c SE_i / scale, outlier scale so_s)."""
    w, mu, g = unpack(p, M)
    d = xs[:, None, None] - mu[None, :, None] - np.sqrt(2) * ss[:, None, None] * GH_T[None, None, :]
    core = np.sum(GH_W * (0.5 + np.arctan(d / g[None, :, None]) / np.pi), axis=2)        # (N, M)
    out = 0.5 + np.arctan((xs[:, None] - mu[None]) / (g[None] + so_s)) / np.pi
    return np.clip(((1 - po) * core + po * out) @ w, 0.0, 1.0)


def cvm(u):
    n = u.size; k = np.arange(1, n + 1)
    return (1 / (12 * n) + np.sum((np.sort(u) - (2 * k - 1) / (2 * n)) ** 2)) / n


def fit_mixture(x, se, calib, rng, deconv=True):
    """Greedy noise-floor fit; returns dict(M, w, mu, gamma (c/d), D, D_path, floor)."""
    ok = np.isfinite(x) & np.isfinite(se); x, se = x[ok], se[ok]
    loc = np.median(x); scale = 1.4826 * np.median(np.abs(x - loc))
    xs = (x - loc) / scale
    if deconv:
        ss = calib["c"] * se / scale; po = calib["pi_o"]; so_s = calib["sigma_o"] / scale
    else:
        ss = np.zeros_like(xs); po = 0.0; so_s = 0.0
    n = xs.size; floor = 1 / (6 * n)
    obj = lambda p, M: cvm(pit(p, M, xs, ss, po, so_s))
    lo_mu, hi_mu = np.quantile(xs, 0.01), np.quantile(xs, 0.99)
    path, best_sel, prev = [], None, None
    for M in range(1, M_MAX + 1):
        inits = []
        if prev is not None:
            u = pit(prev, M - 1, xs, ss, po, so_s); res = np.sort(u) - (2 * np.arange(1, n + 1) - 1) / (2 * n)
            j = np.argsort(xs)[int(np.argmax(np.abs(res)))]
            pw = prev[2 * (M - 1):]
            inits.append(np.r_[prev[:M - 1], xs[j], prev[M - 1:2 * (M - 1)], np.log(0.3), pw, np.log(1.0 / M)])
        for _ in range(N_RESTART):
            inits.append(np.r_[np.sort(rng.uniform(lo_mu, hi_mu, M)), np.log(rng.uniform(0.1, 0.8, M)),
                               rng.normal(0, 0.5, M - 1)])
        bounds = [(lo_mu, hi_mu)] * M + [(np.log(1e-6), np.log(5.0))] * M + [(-6, 6)] * (M - 1)
        best = None
        for p0 in inits:
            p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
            r = minimize(obj, p0, args=(M,), method="L-BFGS-B", bounds=bounds, options=dict(maxiter=400))
            if best is None or r.fun < best.fun:
                best = r
        prev = best.x; D = float(best.fun); path.append((M, D, best.x))
        if D <= floor:
            break
    sel = min(path, key=lambda q: q[1] + LAM * q[0])
    M, D, p = sel; w, mu, g = unpack(p, M)
    return dict(M=int(M), w=w.tolist(), mu=(loc + scale * mu).tolist(), gamma=(scale * g).tolist(), D=D,
                floor=floor, D_path=[q[1] for q in path], n=int(n))


# ─── jobs ────────────────────────────────────────────────────────────────────────────────────────
def measured_ttx(i, method):
    r = FF.analyse(i, method)
    return r["f"]["ttx"], r["se"]["ttx"]


def _fit_job(args):
    i, method, b, deconv, x, se, calib = args
    rng = np.random.default_rng(1000 * i + b)
    if b > 0:
        idx = rng.integers(0, x.size, x.size); x, se = x[idx], se[idx]
    return (i, method, b, deconv, fit_mixture(x, se, calib, rng, deconv))


def fit_all(methods, slices, procs, n_boot):
    jobs = []
    for method in methods:
        calib = load_calib(method)
        for i in slices:
            x, se = measured_ttx(i, method)
            for b in range(n_boot + 1):
                for deconv in ((False,) if NAIVE_ONLY else (False, True)):
                    jobs.append((i, method, b, deconv, x, se, calib))
    t_ = time.time()
    with Pool(procs) as pool:
        R = pool.map(_fit_job, jobs, chunksize=1)
    res = {}
    for i, method, b, deconv, f in R:
        res.setdefault(method, {}).setdefault(str(i), {}).setdefault("deconv" if deconv else "naive", {})[str(b)] = f
    json.dump(res, open(os.path.join(OUT, "fits.json"), "w"), indent=1)
    for method in methods:
        for i in slices:
            for kind in (("naive",) if NAIVE_ONLY else ("naive", "deconv")):
                F = res[method][str(i)][kind]; f0 = F["0"]
                Ms = np.array([F[str(b)]["M"] for b in range(1, n_boot + 1)])
                print(f"[fit {method}] SCN{i} {kind:6s}: M* = {f0['M']} (bootstrap " +
                      ", ".join(f"M={m}: {np.mean(Ms == m):.0%}" for m in sorted(set(Ms))) + ") | μ " +
                      str(np.round(f0["mu"], 3).tolist()) + " γ " + str(np.round(f0["gamma"], 4).tolist()) +
                      " w " + str(np.round(f0["w"], 2).tolist()) + f" | D/floor {f0['D'] / f0['floor']:.2f}",
                      flush=True)
                bs = [F[str(b)] for b in range(1, n_boot + 1)]
                gmax = np.array([np.max(q["gamma"]) for q in bs]); g1 = np.array([q["gamma"][0] for q in bs if q["M"] == 1])
                ci = lambda v: f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]" if len(v) else "—"
                sep = np.array([np.ptp(q["mu"]) for q in bs if q["M"] >= 2])
                print(f"            95% CI: width (M=1 resamples) {ci(g1)}; widest component {ci(gmax)}; "
                      f"centre separation (M≥2 resamples) {ci(sep)}", flush=True)
    print(f"[fit] {len(jobs)} fits in {time.time() - t_:.0f} s", flush=True)


def _rec_job(args):
    method, scen, tmpl, seed, calib = args
    import scn_pipeline_validation as V
    rng = np.random.default_rng(10_000 + 100 * tmpl + seed + 7 * list(SCENARIOS).index(scen))
    g = TG.load(tmpl); d = S.process(tmpl); sel = np.zeros(d["N"], bool); sel[g["cells"]] = True
    w, mu, ga = SCENARIOS[scen]; n = int(sel.sum())
    comp = rng.choice(len(w), n, p=w)
    f_true = np.array(mu)[comp] + np.array(ga)[comp] * np.tan(np.pi * (rng.uniform(size=n) - 0.5))
    f_true = np.clip(f_true, 0.75, 1.3)                             # keep periods inside the 18–32 h band
    per = dict(d["period"]); pt = per["ttx"].copy(); pt[sel] = 24.0 / f_true; per["ttx"] = pt
    d2 = dict(d); d2["period"] = per
    trend, A, noise, kick, Theta = V.ground_truth_inputs(d2, sel, float(g["noise_scale"]), rng)
    th = V.true_phases(d2, sel, Theta, "dephase", kick)
    Xs = trend + A * np.cos(th) + noise
    tps = FF.PEAKS[method](Xs, int(g["ttx"]), int(g["wash"]))
    W = FF.windows(g)
    x, se, _ = FF.frequencies(tps, *W["ttx"])
    out = {}
    for deconv in ((False,) if NAIVE_ONLY else (False, True)):
        out["deconv" if deconv else "naive"] = fit_mixture(x, se, calib, rng, deconv)
    out["n_excluded"] = int(np.sum(~np.isfinite(x)))
    return method, scen, tmpl, seed, out


def recover(methods, procs, n_seeds):
    jobs = [(m, sc, t, s, load_calib(m)) for m in methods for sc in SCENARIOS for t in REC_TEMPLATES
            for s in range(n_seeds)]
    t_ = time.time()
    with Pool(procs) as pool:
        R = pool.map(_rec_job, jobs, chunksize=1)
    res = {}
    for m, sc, t, s, out in R:
        res.setdefault(m, {}).setdefault(sc, []).append(dict(template=t, seed=s, **out))
    json.dump(res, open(os.path.join(OUT, "recovery.json"), "w"), indent=1)
    for m in methods:
        for sc, (w, mu, ga) in SCENARIOS.items():
            L = res[m][sc]
            for kind in (("naive",) if NAIVE_ONLY else ("naive", "deconv")):
                Ms = np.array([q[kind]["M"] for q in L])
                gmin = [np.min(q[kind]["gamma"]) for q in L]
                print(f"[recover {m}] {sc:16s} (true M = {len(w)}, γ = {ga}) {kind:6s}: M* " +
                      ", ".join(f"{k}: {np.mean(Ms == k):.0%}" for k in sorted(set(Ms))) +
                      f" | median smallest γ {np.median(gmin):.4f} c/d | median widest γ "
                      f"{np.median([np.max(q[kind]['gamma']) for q in L]):.4f} | excluded cells "
                      f"{np.median([q['n_excluded'] for q in L]):.0f}", flush=True)
    print(f"[recover] {len(jobs)} runs in {time.time() - t_:.0f} s", flush=True)


# ─── figure ──────────────────────────────────────────────────────────────────────────────────────
def figure(methods, slices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prl_style import set_prl_style, PNAS_DOUBLE
    set_prl_style("pnas")
    res = json.load(open(os.path.join(OUT, "fits.json")))
    rec = json.load(open(os.path.join(OUT, "recovery.json"))) if os.path.exists(os.path.join(OUT, "recovery.json")) \
        else {}
    fig, ax = plt.subplots(len(methods) + 1, len(slices), figsize=(PNAS_DOUBLE, 2.0 * (len(methods) + 1)),
                           squeeze=False, gridspec_kw=dict(hspace=0.6, wspace=0.3, left=0.07, right=0.98, top=0.93,
                                                           bottom=0.08))
    g = np.linspace(0.75, 1.25, 600)
    lor = lambda f: sum(w * gm / np.pi / ((g - m) ** 2 + gm ** 2) for w, m, gm in zip(f["w"], f["mu"], f["gamma"]))
    for r_, method in enumerate(methods):
        for k, i in enumerate(slices):
            a = ax[r_, k]; x, se = measured_ttx(i, method); x = x[np.isfinite(x)]
            a.hist(x, bins=np.linspace(0.75, 1.25, 41), density=True, color="0.85")
            f0 = res[method][str(i)]
            a.plot(g, lor(f0["naive"]["0"]), color="#b2182b", lw=1.2, label=f"naive (M={f0['naive']['0']['M']})")
            if "deconv" in f0:
                a.plot(g, lor(f0["deconv"]["0"]), color="0.3", lw=1.0, ls="--",
                       label=f"deconvolved (M={f0['deconv']['0']['M']})")
            a.set_xlim(0.75, 1.25); a.set_yticks([]); a.legend(frameon=False, fontsize=5)
            a.set_title(f"SCN{i}, {method}", fontsize=7)
            if r_ == len(methods) - 1:
                a.set_xlabel("TTX frequency (c/d)")
    # recovery summary: fraction of runs recovering the true M, per scenario and method
    for k, method in enumerate(methods):
        a = ax[-1, k]
        if method not in rec:
            a.axis("off"); continue
        names = list(SCENARIOS); xpos = np.arange(len(names))
        for dx, kind, col in ((-0.18, "naive", "#b2182b"), (0.18, "deconv", "0.5")):
            if kind not in rec[method][names[0]][0]:
                continue
            fr = [np.mean([q[kind]["M"] == len(SCENARIOS[sc][0]) for q in rec[method][sc]]) for sc in names]
            a.bar(xpos + dx, fr, width=0.34, color=col, label=kind)
        a.set_xticks(xpos); a.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=5.5)
        a.set_ylim(0, 1); a.set_ylabel("true M recovered"); a.set_title(f"synthetic recovery, {method}", fontsize=7)
        a.legend(frameon=False, fontsize=5)
    for k in range(len(methods), len(slices)):
        ax[-1, k].axis("off")
    fig.savefig(os.path.join(OUT, "scn_deconvolution.png"), dpi=250); fig.savefig(os.path.join(OUT,
                                                                                    "scn_deconvolution.pdf"))


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    methods = arg("--methods", ["epoch", "wavelet"], lambda s_: s_.split(","))
    slices = arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")])
    procs = arg("--procs", os.cpu_count(), int)
    if "calib" in sys.argv:
        for m in methods:
            calibrate(m, slices)
    if "fit" in sys.argv:
        fit_all(methods, slices, procs, arg("--boot", B_BOOT, int))
    if "recover" in sys.argv:
        recover(methods, procs, arg("--seeds", REC_SEEDS, int))
    if "figure" in sys.argv:
        figure(methods, slices)
