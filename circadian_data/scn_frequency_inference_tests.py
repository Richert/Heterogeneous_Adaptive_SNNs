r"""
Premise tests for a revised inference of the intrinsic frequencies (SCN_REVIEW.md §12 follow-up)
================================================================================================

Peaks of the fit-target cells (scn_targets) are recomputed with the epoch-wise peak pipeline (scn_epoch_processing).

T1  intervals   phase diffusion vs peak-timing error. Model per cell (stationary period T_i within an epoch):
                    P_k = T_i + d_k + e_{k+1} − e_k,   d_k ~ (0, σ_D²) (accumulating: phase diffusion),
                                                     e_k ~ (0, σ_m²) (non-accumulating: peak-timing error)
                Mean-free moment estimators (no bias from subtracting a per-cell mean over few cycles):
                    E1 = ⟨(P_k − P_{k+1})²⟩ = 2σ_D² + 6σ_m²,   E2 = ⟨(P_k − P_{k+2})²⟩ = 2σ_D² + 4σ_m²
                    ⇒ σ_m² = (E1 − E2)/2,  σ_D² = (E2 − 4σ_m²)/2;  D = 2π² σ_D² / T³ (rad²/h)
                pooled over the cells of a slice, per epoch (pre-TTX, TTX without its first 24 h, washout +72 h),
                intervals > GAP × the cell's median skipped (missed peak). Per cell under TTX: ω̂_i = 2π n / (t_last −
                t_first) (first and last peak in the window) with SE_i² = 2D/T_w + 2 ω̂_i² σ_m² / T_w². Then a
                Lorentzian-mixture fit with the per-cell GAUSSIAN errors convolved in (Voigt likelihood, M = 1, 2;
                pooled slice): widths with (deconvolved) and without the errors.
T2  locking     the coupled-state premise: per cell the circular-mean phase relative to the slice mean phase,
                φ_i = arg⟨e^{i(θ_i − Ψ)}⟩ over the pre-TTX ([start, t_ttx − 24 h)) and post-washout ([t_wash + 72 h,
                end − 72 h)) windows (> 0: cell ahead). Spearman ρ(φ_pre, ω̂), ρ(φ_post, ω̂), ρ(φ_pre, φ_post); and the
                Kuramoto–Sakaguchi locked-state relation sin φ_i ≈ (ω_i − Ω)/(K R) (α ≈ 0): OLS slope of sin φ on
                ω̂ − mean → implied K = 1/(slope · R) (errors in ω̂ attenuate the slope, so K is an UPPER bound).
T3  core/shell  per slice, cells split by distance from their lobe centroid into the inner and outer tercile (per lobe):
                R_in(t), R_out(t) and the phase difference arg(Z_out Z̄_in) (h); desynchronisation rate κ_d (days
                1–3 of TTX), minimum R under TTX, resynchronisation half-time after washout. No ceiling correction
                (the ceiling is computed for the whole slice only).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_frequency_inference_tests.py [--slices 1,2,3,4,5]
Output: data_paths.MPMF/circadian_data/freq_inference/ (tests.json, scn_core_shell_R.{png,pdf})
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from scipy.stats import spearmanr
from scipy.special import voigt_profile
from scipy.optimize import minimize
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_epoch_processing as EP

OUT = dp.ensure(os.path.join(S.OUT, "freq_inference"))
CPD = 24 / (2 * np.pi)
GAP = 1.5


def peak_times(i, g):
    """Sub-hour peak times (list per target cell), epoch-wise pipeline as in scn_targets."""
    X, _ = S.load(i); X = X[:, g["cells"]]; T = X.shape[0]
    tps = [[] for _ in range(X.shape[1])]
    for a, b in EP.bounds(T, int(g["ttx"]), int(g["wash"])):
        seg = X[a:b]
        y = EP.normalise(seg - EP.baseline(seg, TG.PIPE["baseline_kind"]), TG.PIPE["norm"])
        for c in range(X.shape[1]):
            tps[c].append(a + EP.segment_peaks(y[:, c]))
    return [np.concatenate(t) for t in tps]


def interval_moments(tps, a, b):
    """Pooled E1, E2, mean period, and the per-cell (ω̂, T_w) over [a, b)."""
    d1, d2, per, cells = [], [], [], []
    for tp in tps:
        t = tp[(tp >= a) & (tp < b)]
        if t.size < 3:
            cells.append((np.nan, np.nan)); continue
        P = np.diff(t); med = np.median(P)
        ok = P < GAP * med
        # consecutive valid intervals only
        for k in range(P.size - 1):
            if ok[k] and ok[k + 1]:
                d1.append((P[k] - P[k + 1]) ** 2)
        for k in range(P.size - 2):
            if ok[k] and ok[k + 1] and ok[k + 2]:          # P_k and P_{k+2} share no peak, but need P_{k+1} valid
                d2.append((P[k] - P[k + 2]) ** 2)
        per += list(P[ok])
        n = int(np.round((t[-1] - t[0]) / med))            # cycles between first and last peak (gaps counted)
        cells.append((2 * np.pi * n / (t[-1] - t[0]), t[-1] - t[0]))
    E1, E2, Tm = np.mean(d1), np.mean(d2), np.mean(per)
    sm2 = (E1 - E2) / 2; sD2 = (E2 - 4 * sm2) / 2
    return dict(E1=E1, E2=E2, T=Tm, sigma_m=float(np.sqrt(max(sm2, 0))), sigma_D=float(np.sqrt(max(sD2, 0))),
                sm2=float(sm2), sD2=float(sD2), D=float(2 * np.pi ** 2 * max(sD2, 0) / Tm ** 3),
                n1=len(d1), n2=len(d2)), np.array(cells)


def voigt_fit(x, se, M):
    """ML Lorentzian mixture with per-point Gaussian errors se (Voigt likelihood); se = 0 → plain Lorentzian.
    Returns (w, Ω, Δ, negloglik)."""
    mu, sd = np.median(x), 1.4826 * np.median(np.abs(x - np.median(x)))
    xs, ss = (x - mu) / sd, np.maximum(se / sd, 1e-9)

    def unpack(p):
        Om = p[:M]; De = np.exp(p[M:2 * M]); w = np.exp(np.r_[0.0, p[2 * M:]]); return w / w.sum(), Om, De

    def nll(p):
        w, Om, De = unpack(p)
        f = sum(w[m] * voigt_profile(xs - Om[m], ss, De[m]) for m in range(M))
        return -np.sum(np.log(np.maximum(f, 1e-300)))
    best = None
    rng = np.random.default_rng(0)
    for s in range(12):
        p0 = np.r_[np.sort(rng.choice(xs, M)), np.log(rng.uniform(0.1, 0.8, M)), np.zeros(M - 1)]
        r = minimize(nll, p0, method="Nelder-Mead", options=dict(maxiter=4000 * M, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    w, Om, De = unpack(best.x)
    o = np.argsort(Om)
    return w[o], mu + sd * Om[o], sd * De[o], float(best.fun)


def circ_phase(PH, Psi, a, b, i_ok=None):
    rel = np.exp(1j * (PH[a:b] - Psi[a:b, None]))
    return np.angle(np.nanmean(np.where(np.isfinite(rel), rel, np.nan), axis=0))


def lobe_radius(xy, lobe):
    r = np.zeros(lobe.size); terc = np.zeros(lobe.size, int)        # 0 inner, 1 middle, 2 outer
    for L in (0, 1):
        m = lobe == L; c = xy[m].mean(0); rr = np.linalg.norm(xy[m] - c, axis=1); r[m] = rr
        q1, q2 = np.quantile(rr, (1 / 3, 2 / 3)); terc[m] = np.where(rr <= q1, 0, np.where(rr >= q2, 2, 1))
    return r, terc


def main(slices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    fig, ax = plt.subplots(2, len(slices), figsize=(3.0 * len(slices), 5.2), sharex="col", squeeze=False,
                           gridspec_kw=dict(hspace=0.15, wspace=0.3, left=0.06, right=0.99, top=0.9, bottom=0.1))
    print(f"{'':6s}{'epoch':7s}{'T (h)':>7s}{'σ_D (h)':>9s}{'σ_m (h)':>9s}{'D (rad²/h)':>12s}{'D → κ (1/d)':>13s}"
          f"{'n':>6s}")
    for col, i in enumerate(slices):
        g = TG.load(i); _, locs = S.load(i); tps = peak_times(i, g)
        T = int(g["T"]); ttx, wash = int(g["ttx"]), int(g["wash"]); v0, v1 = S.valid_window(T)
        r = {}
        # ── T1
        wins = {"pre": (v0, ttx), "ttx": (ttx + S.TRANSIENT_H, wash), "wash": (wash + 72, v1)}
        for ep, (a, b) in wins.items():
            mom, cells = interval_moments(tps, a, b)
            r["T1_" + ep] = mom
            print(f"SCN{i:<3d}{ep:7s}{mom['T']:7.2f}{mom['sigma_D']:9.2f}{mom['sigma_m']:9.2f}{mom['D']:12.5f}"
                  f"{mom['D'] * 24:13.3f}{mom['n1']:6d}")
            if ep == "ttx":
                om_hat, Tw = cells[:, 0], cells[:, 1]
        mt = r["T1_ttx"]
        se = np.sqrt(2 * mt["D"] / Tw + 2 * om_hat ** 2 * mt["sm2"] / Tw ** 2)
        ok = np.isfinite(om_hat)
        x = om_hat[ok] * CPD; sx = se[ok] * CPD
        sp = spearmanr(om_hat[ok], g["om"][ok])[0]
        fits = {}
        for M in (1, 2):
            for lab, s_ in (("plain", np.zeros_like(sx)), ("deconv", sx)):
                w, Om, De, nll = voigt_fit(x, s_, M)
                fits[f"M{M}_{lab}"] = dict(w=w.round(3).tolist(), Om=Om.round(4).tolist(), De=De.round(4).tolist(),
                                           nll=nll)
        r["T1_cells"] = dict(n=int(ok.sum()), se_median_cpd=float(np.median(sx)),
                             spread_cauchy_cpd=float(0.5 * np.subtract(*np.percentile(x, [75, 25]))),
                             spearman_vs_lsq=float(sp), fits=fits)
        print(f"      TTX per-cell SE median {np.median(sx):.4f} c/d; spread (half-IQR) {r['T1_cells']['spread_cauchy_cpd']:.4f}"
              f" c/d; ρ(ω̂, LSQ slope) = {sp:.2f}")
        for k, v in fits.items():
            print(f"      {k:10s} Ω {v['Om']}  Δ {v['De']}  w {v['w']}  (−logL {v['nll']:.1f})")
        # ── T2
        Zs = TG.mean_field(g["PH"]); Psi = np.unwrap(np.angle(Zs))
        Psi = np.where(np.isfinite(Psi), Psi, 0.0)
        phi_pre = circ_phase(g["PH"], Psi, v0, ttx - 24); phi_post = circ_phase(g["PH"], Psi, wash + 72, v1)
        R_pre = float(np.nanmean(np.abs(Zs[v0:ttx - 24]))); R_post = float(np.nanmean(np.abs(Zs[wash + 72:v1])))
        t2 = {}
        for lab, phi, Rl in (("pre", phi_pre, R_pre), ("post", phi_post, R_post)):
            m = ok & np.isfinite(phi)
            rho, p = spearmanr(phi[m], om_hat[m])
            b = np.polyfit(om_hat[m] - om_hat[m].mean(), np.sin(phi[m]), 1)[0]
            t2[lab] = dict(rho=float(rho), p=float(p), slope=float(b), R=Rl, K_implied=float(1 / (b * Rl)) if b > 0
                           else None, sd_phi_h=float(np.std(phi[m]) * 24 / (2 * np.pi)))
        m = np.isfinite(phi_pre) & np.isfinite(phi_post)
        t2["pre_vs_post_rho"] = float(spearmanr(phi_pre[m], phi_post[m])[0])
        r["T2"] = t2
        print(f"      locking: ρ(φ_pre, ω̂) = {t2['pre']['rho']:.2f} (p {t2['pre']['p']:.1g}), ρ(φ_post, ω̂) = "
              f"{t2['post']['rho']:.2f} (p {t2['post']['p']:.1g}), ρ(φ_pre, φ_post) = {t2['pre_vs_post_rho']:.2f}; "
              f"SD φ pre/post {t2['pre']['sd_phi_h']:.1f}/{t2['post']['sd_phi_h']:.1f} h; implied K pre/post "
              f"{t2['pre']['K_implied']} / {t2['post']['K_implied']} rad/h")
        # ── T3
        _, terc = lobe_radius(locs[g["cells"]].astype(float), g["lobe"])
        E = np.where(np.isfinite(g["PH"]), np.exp(1j * np.nan_to_num(g["PH"])), np.nan)
        Zin, Zout = np.nanmean(E[:, terc == 0], 1), np.nanmean(E[:, terc == 2], 1)
        td = np.arange(T) / 24
        t3 = {}
        for lab, Zr in (("inner", Zin), ("outer", Zout)):
            R = np.abs(Zr); a, b = ttx + 24, ttx + 72
            kd = -np.polyfit(np.arange(a, b), np.log(R[a:b]), 1)[0] * 24
            seg = R[ttx:wash + 24]; imin = ttx + int(np.nanargmin(seg)); Rmin = float(R[imin])
            plat = float(np.nanmean(R[wash + 72:v1])); half = Rmin + 0.5 * (plat - Rmin)
            after = np.where(R[imin:] >= half)[0]
            t3[lab] = dict(kappa_d=float(kd), R_min=Rmin, t_min_h=imin - wash,
                           t_half_h=float(imin + after[0] - wash) if after.size else None, plateau=plat)
        dpsi = np.angle(Zout * np.conj(Zin)) * 24 / (2 * np.pi)
        t3["dpsi_pre_h"] = float(np.nanmean(dpsi[v0:ttx - 24])); t3["dpsi_post_h"] = float(np.nanmean(dpsi[wash + 72:v1]))
        r["T3"] = t3
        print(f"      core/shell: κ_d in/out {t3['inner']['kappa_d']:.2f}/{t3['outer']['kappa_d']:.2f} 1/d; R_min "
              f"{t3['inner']['R_min']:.2f}/{t3['outer']['R_min']:.2f}; t½ after washout {t3['inner']['t_half_h']}/"
              f"{t3['outer']['t_half_h']} h; outer − inner phase pre/post {t3['dpsi_pre_h']:+.2f}/{t3['dpsi_post_h']:+.2f} h")
        a0 = ax[0, col]; a1 = ax[1, col]
        for a_ in (a0, a1):
            a_.axvspan(ttx / 24, wash / 24, color="0.9", lw=0)
        a0.plot(td, np.abs(Zin), color="#08519c", lw=1.3, label="inner tercile")
        a0.plot(td, np.abs(Zout), color="#e6550d", lw=1.3, label="outer tercile")
        a0.set_ylim(0, 1.02); a0.set_title(f"SCN{i}"); a1.plot(td, dpsi, color="k", lw=1)
        a1.axhline(0, color="0.5", lw=0.6); a1.set_ylim(-6, 6); a1.set_xlabel("time (days)")
        if col == 0:
            a0.set_ylabel("R(t)"); a1.set_ylabel("outer − inner phase (h)"); a0.legend(frameon=False, fontsize=8)
        res[str(i)] = r
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"scn_core_shell_R.{ext}"), dpi=200)
    json.dump(res, open(os.path.join(OUT, "tests.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    main(arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")]))
