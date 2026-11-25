r"""
Intrinsic-frequency figure (two-column PNAS): frequencies, their errors and their predictive value in the three
conditions (pre-TTX, TTX, post-washout), for one PEAK METHOD, with a ground-truth validation
====================================================================================================================

Peak methods (fit-target cells of scn_targets; peaks per epoch, parabolic sub-hour refinement):
  epoch     our pipeline: per epoch a polynomial baseline and scalar normalisation, Gaussian smoothing, peaks ≥ 16 h
            apart with a LOCAL prominence threshold (scn_epoch_processing.segment_peaks)
  wavelet   Abel et al. (2016, Fig. S1): discrete wavelet transform of the whole trace (Daubechies-8, 4 levels);
            only the level-4 detail is kept, i.e. the dyadic 16–32 h band (removes the slow baseline and the fast
            noise); peaks = maxima of this smooth band signal ≥ 16 h apart, no prominence threshold (a threshold
            relative to the epoch's range rejects real TTX peaks: the whole-trace band leaks the large pre-TTX and
            post-washout swings into the edges of the low-amplitude TTX epoch)
Conditions (h): pre = [0, t_ttx), TTX = [t_ttx + 24, t_wash), post = [t_wash + 24, T) (peak-based frequencies have no
filter edges; the first 24 h after a treatment change are excluded).

A  frequency distributions: per cell and condition, the period T_i from a least-squares fit of the peak times to the
   cycle number, f_i = 24 / T_i (cycles/day). CYCLE-SLIP HANDLING (2026-11-25): cycles are counted against a SLICE-WIDE
   reference period P_ref (median peak interval of all cells in the condition), not the cell's own median (which is
   itself ~48 h when half the peaks are missed): a peak closer than EXTRA_FRAC · P_ref to its predecessor is dropped
   (extra peak), an interval of k · P_ref counts as round(k) cycles (missed peaks); cells whose corrected period lies
   outside S.BAND_H (18–32 h) are excluded (≥ 2 peaks needed)
B  frequency errors: standard error of f_i from the residual scatter of the peak times (non-accumulating peak-timing
   error, scn_frequency_inference_tests), the residual variance shrunk towards the slice and condition pool with
   PRIOR_DOF pseudo-degrees of freedom (the pre-TTX epoch has 2–3 cycles per cell); dashed: the spread (half the
   interquartile range) of the TTX frequencies
C  (NOT in this figure since 2026-11-25; --with-skill: for the later dynamics figure) phase prediction with ONE minimal model, the Kuramoto network θ̇_i = ν_B + Δω_i + K R sin(Ψ − θ_i), Δω_i = ω_i^A −
   ⟨ω^A⟩ (frequencies from condition A → phases in the SECOND HALF of condition B; diagonal: ω from the FIRST half of B):
     B = TTX (uncoupled, dispersing): integrated from the measured phases at the midpoint of B, ν_B = the measured
         collective frequency, K fitted on a grid (0 included); error = wrapped θ_model − θ_data over the hours
     B = pre / post (coupled, locked): its locked fixed point, relative phase ψ_i = arcsin(Δω_i / KR) with KR fitted,
         vs the measured time-averaged relative phase ψ_i = arg⟨e^{i(θ_i − Ψ)}⟩ (integrating from measured phases
         would carry the locked pattern over and leave nothing for the frequencies to predict)
   Skill = 1 − MSE(per-cell frequencies) / MSE(identical frequencies; for TTX with its own best K, for pre / post all
   cells at Ψ). Skill > 0: the frequencies of A carry per-cell information about the phases in B. The per-cell
   frequencies are SHRUNK towards the mean by their errors (empirical Bayes, ω_i ← ω̄ + λ_i (ω̂_i − ω̄), λ_i = v /
   (v + SE_i²), v = max(Var ω̂ − ⟨SE²⟩, 0)), the best linear predictor given the measurement errors.
C  noise-floor Lorentzian-mixture fit (naive, scn_deconvolution.py fit --naive-only) of the TTX frequencies: histogram,
   fit (line) and 95 % bootstrap band; M* and its bootstrap support (shown if deconvolution/fits.json exists)
D–G  validation with synthetic traces (scn_pipeline_validation: each cell's measured trend, envelope and calibrated
   noise; phases: collective before TTX, constant per-cell frequencies under TTX, relaxation after washout) through
   the same method: D estimated vs true TTX frequency; E robust error SD vs predicted SE (calibration); F z = error / SE
   vs N(0, 1); G true vs estimated TTX distribution (SCN1).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_frequency_figure.py [--methods epoch,wavelet]
        [--slices 1,2,3,4,5]
Output: data_paths.MPMF/circadian_data/writeup/scn_frequency_inference_<method>.{png,pdf,json}
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import pywt
from scipy.stats import spearmanr, norm
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_epoch_processing as EP

OUT = dp.ensure(os.path.join(S.OUT, "writeup"))
CONDS = ("pre", "ttx", "post")
LABEL = dict(pre="pre-TTX", ttx="TTX", post="post-washout")
COL = dict(pre="#4d4d4d", ttx="#b2182b", post="#2166ac")
PRIOR_DOF = 2.0
EXTRA_FRAC = 0.6                          # peaks closer than this × P_ref are extra (dropped)
WAVELET, LEVEL = "db8", 4
K_GRID = np.r_[0.0, np.geomspace(1e-3, 1.0, 30)]
DT = 0.25
SEED = 11
WITH_SKILL = "--with-skill" in sys.argv


def windows(g):
    T = int(g["T"]); v0, v1 = S.valid_window(T); ttx, wash = int(g["ttx"]), int(g["wash"])
    return dict(pre=(0, ttx), ttx=(ttx + S.TRANSIENT_H, wash), post=(wash + S.TRANSIENT_H, T))


# ─── peak methods ────────────────────────────────────────────────────────────────────────────────
def peaks_epoch(X, ttx, wash):
    T = X.shape[0]; tps = [[] for _ in range(X.shape[1])]
    for a, b in EP.bounds(T, ttx, wash):
        seg = X[a:b]
        y = EP.normalise(seg - EP.baseline(seg, TG.PIPE["baseline_kind"]), TG.PIPE["norm"])
        for c in range(X.shape[1]):
            tps[c].append(a + EP.segment_peaks(y[:, c]))
    return [np.sort(np.concatenate(t)) for t in tps]


def wavelet_band(x):
    """Level-LEVEL detail of a Daubechies DWT of the whole trace (dyadic 16–32 h band for hourly samples)."""
    co = pywt.wavedec(x, WAVELET, mode="symmetric", level=LEVEL)
    keep = [np.zeros_like(c) for c in co]; keep[1] = co[1]           # [cA_L, cD_L, ..., cD_1]
    return pywt.waverec(keep, WAVELET, mode="symmetric")[:x.size]


def peaks_wavelet(X, ttx, wash):
    from scipy.signal import find_peaks
    T = X.shape[0]; Bw = np.column_stack([wavelet_band(X[:, c]) for c in range(X.shape[1])])
    tps = [[] for _ in range(X.shape[1])]
    for a, b in EP.bounds(T, ttx, wash):
        for c in range(X.shape[1]):
            y = Bw[a:b, c]; pk, _ = find_peaks(y, distance=EP.MIN_PEAK_DIST)
            tps[c].append(a + np.array([EP.refine(y, p) for p in pk]))
    return [np.sort(np.concatenate(t)) for t in tps]


PEAKS = dict(epoch=peaks_epoch, wavelet=peaks_wavelet)


# ─── frequencies and errors ──────────────────────────────────────────────────────────────────────
def reference_period(tps, a, b):
    P = np.concatenate([np.diff(tp[(tp >= a) & (tp < b)]) for tp in tps])
    P = P[(P > S.BAND_H[0]) & (P < S.BAND_H[1])]
    return float(np.median(P)) if P.size else 24.0


def clean_peaks(t, p_ref):
    """Drop extra peaks (closer than EXTRA_FRAC · p_ref to the previous kept peak)."""
    keep = [t[0]] if t.size else []
    for x in t[1:]:
        if x - keep[-1] >= EXTRA_FRAC * p_ref:
            keep.append(x)
    return np.array(keep)


def regress(tps, a, b):
    """Per cell: period T (h), residual variance s², dof, Σ(n − n̄)² from peak time vs cycle number in [a, b);
    cycles counted against the slice reference period (cycle-slip handling); out-of-band periods → NaN."""
    out = np.full((len(tps), 4), np.nan)
    p_ref = reference_period(tps, a, b)
    for c, tp in enumerate(tps):
        t = clean_peaks(tp[(tp >= a) & (tp < b)], p_ref)
        if t.size < 2:
            continue
        P = np.diff(t)
        n = np.r_[0.0, np.cumsum(np.maximum(1.0, np.round(P / p_ref)))]
        coef = np.polyfit(n, t, 1); r = t - np.polyval(coef, n); dof = t.size - 2
        if not (S.BAND_H[0] < coef[0] < S.BAND_H[1]):
            continue
        out[c] = (coef[0], np.sum(r ** 2) / dof if dof > 0 else np.nan, dof, np.sum((n - n.mean()) ** 2))
    return out


def frequencies(tps, a, b):
    """f (c/d), SE_f (c/d), ω (rad/h) per cell; residual variance shrunk towards the pool (PRIOR_DOF)."""
    R = regress(tps, a, b); T, s2, dof, Sxx = R.T
    ok = np.isfinite(s2) & (dof > 0)
    pool = np.sum(dof[ok] * s2[ok]) / np.sum(dof[ok]) if ok.any() else np.nan
    s2e = np.where(ok, (dof * np.nan_to_num(s2) + PRIOR_DOF * pool) / (np.nan_to_num(dof) + PRIOR_DOF), pool)
    se_T = np.sqrt(s2e / Sxx)
    f = 24.0 / T
    return f, 24.0 * se_T / T ** 2, 2 * np.pi / T


def phases(tps, T):
    return np.column_stack([EP.peak_phase(tp, T) for tp in tps])


# ─── phase prediction ────────────────────────────────────────────────────────────────────────────
def collective_freq(PH):
    E = np.where(np.isfinite(PH), np.exp(1j * np.nan_to_num(PH)), np.nan)
    Z = np.nanmean(E, axis=1); ph = np.unwrap(np.angle(Z))
    return np.polyfit(np.arange(ph.size), ph, 1)[0]


def best_mse(theta0, omega, target):
    """Kuramoto runs for all K_GRID at once; MSE of the wrapped phase error over the finite target samples, min over
    K (and the K)."""
    nh = target.shape[0]; sub = int(round(1 / DT))
    th = np.repeat(theta0[None], K_GRID.size, 0); se = np.zeros(K_GRID.size); cnt = 0
    fin = np.isfinite(target)
    for h in range(1, nh):
        for _ in range(sub):
            Z = np.exp(1j * th).mean(1, keepdims=True)
            th = th + DT * (omega[None] + K_GRID[:, None] * np.imag(Z * np.exp(-1j * th)))
        m = fin[h]
        if m.any():
            se += np.sum(np.angle(np.exp(1j * (th[:, m] - target[h, m][None]))) ** 2, axis=1); cnt += m.sum()
    mse = se / max(cnt, 1); k = int(np.argmin(mse))
    return float(mse[k]), float(K_GRID[k])


KR_GRID = np.geomspace(1e-4, 1.0, 81)


def relative_phase(PH):
    """Time-averaged phase of each cell relative to the mean phase Ψ(t) of the cells (rad)."""
    E = np.where(np.isfinite(PH), np.exp(1j * np.nan_to_num(PH)), np.nan)
    Psi = np.angle(np.nanmean(E, axis=1))
    return np.angle(np.nanmean(E * np.exp(-1j * Psi)[:, None], axis=0))


def locked_skill(psi, dev):
    """Locked fixed point of the Kuramoto model: ψ_i = arcsin(Δω_i / KR) (KR fitted) vs ψ_i = 0 for all."""
    m0 = np.mean(psi ** 2)
    mse = [np.mean(np.angle(np.exp(1j * (psi - np.arcsin(np.clip(dev / kr, -1, 1))))) ** 2) for kr in KR_GRID]
    k = int(np.argmin(mse))
    return 1 - mse[k] / m0, float(KR_GRID[k])


def skill_matrix(PH, tps, W):
    S_ = np.full((3, 3), np.nan); Kb = np.full((3, 3), np.nan)
    for jb, B in enumerate(CONDS):
        a, b = W[B]; tm = (a + b) // 2
        target = PH[tm:b]
        for ia, A in enumerate(CONDS):
            src = (a, tm) if A == B else W[A]
            f_, se_, om = frequencies(tps, *src)
            ok = np.isfinite(om) & np.isfinite(se_) & np.isfinite(PH[tm]) & (np.isfinite(target).sum(0) > 0)
            if ok.sum() < 20:
                continue
            se_w = se_[ok] * 2 * np.pi / 24                         # c/d -> rad/h
            v = max(np.var(om[ok]) - np.mean(se_w ** 2), 0.0)
            dev = (om[ok] - om[ok].mean()) * v / (v + se_w ** 2)
            if B == "ttx":
                nu = collective_freq(target[:, ok])
                m1, k1 = best_mse(PH[tm, ok], nu + dev, target[:, ok])
                m0, _ = best_mse(PH[tm, ok], np.full(ok.sum(), nu), target[:, ok])
                S_[ia, jb] = 1 - m1 / m0; Kb[ia, jb] = k1
            else:
                S_[ia, jb], Kb[ia, jb] = locked_skill(relative_phase(target[:, ok]), dev)
    return S_, Kb


# ─── per slice ───────────────────────────────────────────────────────────────────────────────────
def analyse(i, method, skill=False):
    g = TG.load(i); X, _ = S.load(i); X = X[:, g["cells"]]; T = X.shape[0]; W = windows(g)
    tps = PEAKS[method](X, int(g["ttx"]), int(g["wash"]))
    res = dict(f={}, se={})
    for c in CONDS:
        f, se, _ = frequencies(tps, *W[c]); res["f"][c] = f; res["se"][c] = se
    if skill:
        res["skill"], res["K"] = skill_matrix(phases(tps, T), tps, W)
    return res


def validate(i, method):
    import scn_pipeline_validation as V
    g = TG.load(i); d = S.process(i); sel = np.zeros(d["N"], bool); sel[g["cells"]] = True
    W = windows(g); t = np.arange(d["T"])
    rng = np.random.default_rng(SEED + 100 * i)
    trend, A, noise, kick, Theta = V.ground_truth_inputs(d, sel, float(g["noise_scale"]), rng)
    th = V.true_phases(d, sel, Theta, "dephase", kick)
    Xs = trend + A * np.cos(th) + noise
    tps = PEAKS[method](Xs, int(g["ttx"]), int(g["wash"]))
    out = {}
    for c in CONDS:
        a, b = W[c]
        f_est, se, _ = frequencies(tps, a, b)
        f_true = np.polyfit(t[a:b], th[a:b], 1)[0] * 24 / (2 * np.pi)
        out[c] = dict(est=f_est, se=se, true=f_true)
    return out


# ─── figure ──────────────────────────────────────────────────────────────────────────────────────
def figure(method, R, Vd, slices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from prl_style import set_prl_style, PNAS_DOUBLE, pnas_panel_label
    set_prl_style("pnas")
    fits = None
    fp = os.path.join(S.OUT, "deconvolution", "fits.json")
    if os.path.exists(fp):
        fits = json.load(open(fp)).get(method)
    nrow = 4 if fits else 3
    fig = plt.figure(figsize=(PNAS_DOUBLE, 7.7 if fits else 5.9))
    gs = GridSpec(nrow, 1, figure=fig, height_ratios=(1, 1, 1, 1.15)[-nrow:] if fits else (1, 1, 1.15),
                  hspace=0.55, left=0.075, right=0.975, top=0.965, bottom=0.06)
    n = len(slices)
    rowA = gs[0].subgridspec(1, n, wspace=0.28); rowB = gs[1].subgridspec(1, n, wspace=0.28)
    rowM = gs[2].subgridspec(1, n, wspace=0.28) if fits else None
    rowD = gs[nrow - 1].subgridspec(1, 4, wspace=0.42)
    gx = np.linspace(bins_f[0] if False else 0.75, 1.25, 600)
    lor = lambda f: sum(w * gm / np.pi / ((gx - m) ** 2 + gm ** 2) for w, m, gm in zip(f["w"], f["mu"], f["gamma"]))
    bins_f = np.linspace(0.75, 1.25, 41); bins_se = np.geomspace(2e-3, 0.3, 31)
    for k, i in enumerate(slices):
        r = R[i]
        a = fig.add_subplot(rowA[k])
        for c in CONDS:
            f = r["f"][c]; f = f[np.isfinite(f)]
            a.hist(f, bins=bins_f, density=True, histtype="step", color=COL[c], lw=1.0, label=LABEL[c])
        a.set_title(f"SCN{i}", fontweight="bold"); a.set_xlim(bins_f[0], bins_f[-1]); a.set_yticks([])
        a.set_xlabel("frequency (c/d)")
        if k == 0:
            a.set_ylabel("density"); pnas_panel_label(a, "A", dx=-14)
            a.legend(frameon=False, loc="upper left", handlelength=1.2, fontsize=5.5)
        b_ = fig.add_subplot(rowB[k])
        for c in CONDS:
            s = r["se"][c]; s = s[np.isfinite(s)]
            b_.hist(s, bins=bins_se, histtype="step", color=COL[c], lw=1.0)
        ft = r["f"]["ttx"]; ft = ft[np.isfinite(ft)]
        b_.axvline(0.5 * np.subtract(*np.percentile(ft, [75, 25])), color=COL["ttx"], ls="--", lw=0.8)
        b_.set_xscale("log"); b_.set_xlabel("frequency SE (c/d)"); b_.set_yticks([])
        if k == 0:
            b_.set_ylabel("cells"); pnas_panel_label(b_, "B", dx=-14)
        if fits:
            F = fits[str(i)]["naive"]; f0 = F["0"]; boots = [F[b] for b in F if b != "0"]
            m_ = fig.add_subplot(rowM[k])
            m_.hist(ft, bins=bins_f, density=True, color="0.85", lw=0)
            P = np.array([lor(q) for q in boots])
            m_.fill_between(gx, np.percentile(P, 2.5, 0), np.percentile(P, 97.5, 0), color=COL["ttx"], alpha=0.25,
                            lw=0)
            m_.plot(gx, lor(f0), color=COL["ttx"], lw=1.1)
            Ms = np.array([q["M"] for q in boots])
            m_.text(0.03, 0.96, f"M* = {f0['M']}\n({np.mean(Ms == f0['M']):.0%} of resamples)",
                    transform=m_.transAxes, va="top", fontsize=5.5)
            m_.set_xlim(bins_f[0], bins_f[-1]); m_.set_yticks([]); m_.set_xlabel("TTX frequency (c/d)")
            if k == 0:
                m_.set_ylabel("density"); pnas_panel_label(m_, "C", dx=-14)
    # validation
    dA = fig.add_subplot(rowD[0]); dE = fig.add_subplot(rowD[1]); dF = fig.add_subplot(rowD[2])
    dG = fig.add_subplot(rowD[3])
    est = np.concatenate([Vd[i]["ttx"]["est"] for i in slices]); tru = np.concatenate([Vd[i]["ttx"]["true"]
                                                                                     for i in slices])
    ok = np.isfinite(est) & np.isfinite(tru)
    dA.plot(tru[ok], est[ok], ".", ms=1.5, color=COL["ttx"], alpha=0.5)
    lim = (0.75, 1.25); dA.plot(lim, lim, color="k", lw=0.6); dA.set_xlim(lim); dA.set_ylim(lim)
    rho = spearmanr(tru[ok], est[ok])[0]
    dA.text(0.04, 0.96, f"ρ = {rho:.2f}\nmedian |err| {np.median(np.abs(est[ok] - tru[ok])):.3f} c/d",
            transform=dA.transAxes, va="top", fontsize=5.5)
    dA.set_xlabel("true TTX frequency (c/d)"); dA.set_ylabel("estimated (c/d)"); pnas_panel_label(dA, "D" if fits else "C", dx=-22)
    zs = {}
    for c in CONDS:
        e = np.concatenate([Vd[i][c]["est"] - Vd[i][c]["true"] for i in slices])
        s = np.concatenate([Vd[i][c]["se"] for i in slices]); m = np.isfinite(e) & np.isfinite(s)
        e, s = e[m], s[m]; zs[c] = e / s
        qb = np.quantile(s, np.linspace(0, 1, 7))
        xc, yc = [], []
        for lo, hi in zip(qb[:-1], qb[1:]):
            q = (s >= lo) & (s <= hi)
            if q.sum() > 10:
                xc.append(np.median(s[q])); yc.append(1.4826 * np.median(np.abs(e[q])))
        dE.plot(xc, yc, "o-", color=COL[c], ms=2.5, lw=0.8, label=LABEL[c])
    lo_, hi_ = 3e-3, 0.3
    dE.plot([lo_, hi_], [lo_, hi_], color="k", lw=0.6); dE.set_xscale("log"); dE.set_yscale("log")
    dE.set_xlim(lo_, hi_); dE.set_ylim(lo_, hi_)
    dE.set_xlabel("predicted SE (c/d)"); dE.set_ylabel("robust error SD (c/d)"); dE.legend(frameon=False, fontsize=5.5)
    pnas_panel_label(dE, "E" if fits else "D", dx=-24)
    zb = np.linspace(-5, 5, 41); xx = np.linspace(-5, 5, 200)
    for c in CONDS:
        z = zs[c][np.isfinite(zs[c])]
        dF.hist(z, bins=zb, density=True, histtype="step", color=COL[c], lw=1.0)
    dF.plot(xx, norm.pdf(xx), color="k", lw=0.6, ls="--", label="N(0, 1)")
    dF.set_xlim(-5, 5); dF.set_xlabel("error / SE"); dF.set_ylabel("density"); dF.legend(frameon=False, fontsize=5.5)
    pnas_panel_label(dF, "F" if fits else "E", dx=-22)
    i0 = slices[0]
    for arr, lab, col, ls in ((Vd[i0]["ttx"]["true"], "true", "k", "-"), (Vd[i0]["ttx"]["est"], "estimated",
                                                                            COL["ttx"], "-")):
        a_ = arr[np.isfinite(arr)]
        dG.hist(a_, bins=bins_f, density=True, histtype="step", color=col, ls=ls, lw=1.0, label=lab)
    dG.set_xlabel(f"SCN{i0} TTX frequency (c/d)"); dG.set_yticks([]); dG.set_ylim(top=dG.get_ylim()[1] * 1.3)
    dG.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2)
    pnas_panel_label(dG, "G" if fits else "F", dx=-10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"scn_frequency_inference_{method}.{ext}"), dpi=300)
    plt.close(fig)


def main(methods, slices):
    for method in methods:
        R = {i: analyse(i, method, skill=WITH_SKILL) for i in slices}
        Vd = {i: validate(i, method) for i in slices}
        summ = {}
        print(f"\n=== {method} ===")
        for i in slices:
            r = R[i]; s = {}
            for c in CONDS:
                f = r["f"][c]; se = r["se"][c]; m = np.isfinite(f)
                s[c] = dict(n=int(m.sum()), median_f=float(np.median(f[m])),
                            spread_halfIQR=float(0.5 * np.subtract(*np.percentile(f[m], [75, 25]))),
                            median_se=float(np.nanmedian(se)))
                v = Vd[i][c]; e = v["est"] - v["true"]; mm = np.isfinite(e) & np.isfinite(v["se"])
                z = e[mm] / v["se"][mm]
                s[c].update(val_rms_err=float(np.sqrt(np.mean(e[mm] ** 2))),
                            val_med_abs_err=float(np.median(np.abs(e[mm]))),
                            val_med_se=float(np.median(v["se"][mm])),
                            val_z_sd=float(np.std(z)), val_z_robust_sd=float(1.4826 * np.median(np.abs(z))),
                            val_frac_abs_z_gt3=float(np.mean(np.abs(z) > 3)))
            if WITH_SKILL:
                s["skill"] = r["skill"].round(3).tolist(); s["K"] = r["K"].round(4).tolist()
            summ[str(i)] = s
            print(f"SCN{i}: " + " | ".join(f"{c}: n {s[c]['n']} f̃ {s[c]['median_f']:.3f} spread {s[c]['spread_halfIQR']:.4f}"
                                          f" SE {s[c]['median_se']:.4f} (val med|err| {s[c]['val_med_abs_err']:.4f} vs "
                                          f"med SE {s[c]['val_med_se']:.4f}, robust z-SD {s[c]['val_z_robust_sd']:.2f}, "
                                          f"|z|>3 {s[c]['val_frac_abs_z_gt3']:.2f})" for c in CONDS))
            if WITH_SKILL:
                print("      skill (rows: from pre/TTX/post; cols: in pre/TTX/post):", np.round(r["skill"], 2).tolist())
        json.dump(summ, open(os.path.join(OUT, f"scn_frequency_inference_{method}.json"), "w"), indent=1)
        figure(method, R, Vd, slices)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    main(arg("--methods", ["epoch", "wavelet"], lambda s_: s_.split(",")),
         arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")]))
