r"""
Do the intrinsic frequencies change during TTX? Drift and noise of single-cell rhythms in the TTX window
========================================================================================================

Per slice, cells rhythmic under TTX (scn_common), window W = [TTX + TRANSIENT_H, washout):

  cycle analysis   detrended trace (24 h moving average) smoothed with a Gaussian (σ = SMOOTH_H); peaks ≥ 16 h
                   apart (scn_common.peaks), refined to sub-hour resolution by a parabola through the three samples
                   around each maximum; troughs = minima between consecutive peaks. Cells with ≥ MIN_PERIODS
                   peak-to-peak periods in W are analysed.
  slow drift       per cell: OLS slope of the cycle periods vs time (h per day); population: mean slope, fraction of
                   cells with |slope| significant (two-sided t-test, p < 0.05), early-vs-late period difference
                   (first vs second half of each cell's cycles), and the across-cell spread of the periods in the
                   first vs the last cycles (convergence would indicate residual coupling).
  phase noise      SD of the peak times around a constant-period fit (h) and the cycle-to-cycle period jitter
                   (SD of the period residuals around the cell's linear trend, h).
  amplitude        cycle amplitude = peak − mean of the neighbouring troughs, relative to the 24 h trend (baseline);
                   damping rate = slope of log amplitude vs time (1/day); amplitude noise = SD of the log-amplitude
                   residuals around that trend.
  NULL MODEL       per cell, a STATIONARY fit to its detrended TTX segment: x(t) = A e^{−κ t} cos(ω t + φ) + c with
                   AR(1) residual noise matched to the cell's residual SD and lag-1 autocorrelation (constant
                   frequency by construction); N_SURR surrogates per cell are analysed with the identical pipeline.
                   Every statistic is reported for data AND null, so estimator bias / resolution is visible.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_ttx_frequency_drift.py [--slices 1,2,3,4,5] [--surr 5]
Output: data_paths.MPMF/circadian_data/ttx_drift/{scn_ttx_drift.png/.pdf, scn<i>_ttx_drift.npz}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import least_squares
from scipy.stats import t as student_t
import data_paths as dp
import scn_common as S

OUT = dp.ensure(os.path.join(S.OUT, "ttx_drift"))
MIN_PERIODS = 3
N_SURR = 5
SEED = 7


# ─── cycle analysis of one trace segment ─────────────────────────────────────────────────────────
def cycles(raw):
    """Peak times (h, sub-hour), periods, period midpoints, relative cycle amplitudes of one raw segment
    (the segment carries 24 h of margin on both sides for the trend; peaks are taken inside the margins)."""
    trend = uniform_filter1d(raw, 24, mode="nearest")
    y = gaussian_filter1d(raw - trend, S.SMOOTH_H)
    pk = S.peaks(y)
    if pk.size < 2:
        return None
    tp = []
    for p in pk:                                               # parabolic refinement of the maximum
        if 0 < p < y.size - 1:
            a, b, c = y[p - 1], y[p], y[p + 1]
            den = a - 2 * b + c
            tp.append(p + (0.5 * (a - c) / den if den != 0 else 0.0))
        else:
            tp.append(float(p))
    tp = np.array(tp)
    amp = []
    for k, p in enumerate(pk):                                 # peak − mean of neighbouring troughs
        lo = [y[pk[k - 1]:p].min()] if k > 0 else []
        hi = [y[p:pk[k + 1]].min()] if k < pk.size - 1 else []
        tr = lo + hi
        amp.append((y[p] - np.mean(tr)) / max(trend[p], 1e-12) if tr else np.nan)
    return dict(tp=tp, per=np.diff(tp), tmid=0.5 * (tp[1:] + tp[:-1]), amp=np.array(amp))


def ols(x, y):
    """Slope, its standard error and two-sided p-value of y ~ a + b x."""
    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(n), x - x.mean()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (n - 2)
    se = np.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    p = 2 * student_t.sf(abs(beta[1] / se), n - 2) if se > 0 else np.nan
    return beta[1], se, p


def cell_stats(c):
    """Drift, jitter, phase scatter, amplitude damping / noise of one cell's cycles (None if too few)."""
    if c is None or c["per"].size < MIN_PERIODS:
        return None
    per, tm, tp = c["per"], c["tmid"], c["tp"]
    slope, se, p = ols(tm / 24.0, per)                       # h per day
    X = np.column_stack([np.ones(per.size), tm - tm.mean()])
    jitter = np.std(per - X @ np.linalg.lstsq(X, per, rcond=None)[0], ddof=2) if per.size > 2 else np.nan
    k = np.arange(tp.size)
    phase_sd = np.std(tp - np.polyval(np.polyfit(k, tp, 1), k), ddof=2) if tp.size > 2 else np.nan
    h = per.size // 2
    early, late = per[:max(h, 1)].mean(), per[-max(h, 1):].mean()
    a = c["amp"]; ok = np.isfinite(a) & (a > 0)
    damp = amp_noise = np.nan
    if ok.sum() >= 3:
        cf = np.polyfit(tp[ok] / 24.0, np.log(a[ok]), 1)
        damp = -cf[0]
        amp_noise = np.std(np.log(a[ok]) - np.polyval(cf, tp[ok] / 24.0), ddof=2)
    return dict(slope=slope, se=se, p=p, jitter=jitter, phase_sd=phase_sd, early=early, late=late,
                mean_per=per.mean(), first=per[0], last=per[-1], damp=damp, amp_noise=amp_noise,
                n_per=per.size, amp_mean=np.nanmean(a))


# ─── stationary null model ───────────────────────────────────────────────────────────────────────
def null_surrogates(raw, a0, a1, n_surr, rng):
    """Surrogates of raw[a0:a1] from a stationary damped-sinusoid fit + matched AR(1) noise; the 24 h margins
    outside [a0, a1) are kept from the data so that the detrending is identical."""
    seg = raw[a0:a1]; t = np.arange(seg.size, dtype=float)
    trend = uniform_filter1d(raw, 24, mode="nearest")[a0:a1]
    det = seg - trend
    best = None
    for P0 in (22.0, 24.0, 26.0, 28.0):
        x0 = [np.std(det) * np.sqrt(2), 0.0, 2 * np.pi / P0, 0.0]
        f = lambda q: q[0] * np.exp(-q[1] * t / 24) * np.cos(q[2] * t + q[3]) - det
        r = least_squares(f, x0, bounds=([0, -2, 2 * np.pi / 34, -np.inf], [np.inf, 5, 2 * np.pi / 16, np.inf]))
        if best is None or r.cost < best.cost:
            best = r
    A, kap, om, ph = best.x
    model = A * np.exp(-kap * t / 24) * np.cos(om * t + ph)
    res = det - model
    sd = res.std(); rho = np.clip(np.corrcoef(res[:-1], res[1:])[0, 1], -0.99, 0.99) if res.size > 2 else 0.0
    out = []
    for _ in range(n_surr):
        e = np.empty(seg.size); e[0] = rng.normal(0, sd)
        innov = rng.normal(0, sd * np.sqrt(1 - rho ** 2), seg.size)
        for k in range(1, seg.size):
            e[k] = rho * e[k - 1] + innov[k]
        s = raw.copy(); s[a0:a1] = trend + model + e
        out.append(s)
    return out, 2 * np.pi / om


# ─── per slice ───────────────────────────────────────────────────────────────────────────────────
def analyse(i, n_surr):
    d = S.process(i)
    r = d["rhythmic"]["ttx"]; X = d["X"][:, r]
    a0, a1 = d["epochs"]["ttx"]
    m0, m1 = max(a0 - 24, 0), min(a1 + 24, d["T"])            # 24 h margins for the trend filter
    rng = np.random.default_rng([SEED, i])
    data, null = [], []
    for c in range(X.shape[1]):
        raw = X[m0:m1, c]
        cyc = cycles(raw)
        if cyc is not None:                                    # keep only peaks inside the TTX window
            keep = (cyc["tp"] >= a0 - m0) & (cyc["tp"] < a1 - m0)
            cyc = cycles_subset(cyc, keep)
        st = cell_stats(cyc)
        if st is None:
            continue
        data.append(st)
        surr, P_fit = null_surrogates(raw, a0 - m0, a1 - m0, n_surr, rng)
        for s in surr:
            cs = cycles(s)
            if cs is not None:
                cs = cycles_subset(cs, (cs["tp"] >= a0 - m0) & (cs["tp"] < a1 - m0))
            ss = cell_stats(cs)
            if ss is not None:
                null.append(ss)
    tab = lambda L, k: np.array([x[k] for x in L], float)
    keys = list(data[0]) if data else []
    res = {f"d_{k}": tab(data, k) for k in keys}
    res.update({f"n_{k}": tab(null, k) for k in keys})
    res.update(n_rhythmic=int(r.sum()), window=np.array([a0, a1]))
    np.savez_compressed(os.path.join(OUT, f"scn{i}_ttx_drift.npz"), **res)
    return res


def cycles_subset(c, keep):
    """Restrict a cycles() result to the peaks in `keep` (consecutive peaks only define periods)."""
    if c is None or keep.sum() < 2:
        return None
    idx = np.where(keep)[0]
    tp = c["tp"][idx]; amp = c["amp"][idx]
    return dict(tp=tp, per=np.diff(tp), tmid=0.5 * (tp[1:] + tp[:-1]), amp=amp)


def summary(i, R):
    D = lambda k: R[f"d_{k}"]; N = lambda k: R[f"n_{k}"]
    sig = lambda pv: np.mean(pv < 0.05)
    per0, perL = D("first"), D("last")
    print(f"\nSCN{i}: {D('slope').size} of {R['n_rhythmic']} rhythmic cells with ≥ {MIN_PERIODS} periods in the TTX "
          f"window ({(R['window'][1] - R['window'][0]) / 24:.1f} d); null: {N('slope').size} surrogate cells")
    rows = [("mean period (h)", "mean_per", np.mean), ("period drift (h/day), mean", "slope", np.mean),
            ("period drift (h/day), SD across cells", "slope", np.std),
            ("cycle-to-cycle jitter (h), median", "jitter", np.nanmedian),
            ("peak-time scatter (h), median", "phase_sd", np.nanmedian),
            ("amplitude damping (1/day), median", "damp", np.nanmedian),
            ("log-amplitude noise, median", "amp_noise", np.nanmedian)]
    print(f"  {'statistic':42s} {'data':>9s} {'null':>9s}")
    for lab, k, f in rows:
        print(f"  {lab:42s} {f(D(k)):9.3f} {f(N(k)):9.3f}")
    print(f"  {'fraction of cells with significant drift':42s} {sig(D('p')):9.2f} {sig(N('p')):9.2f}")
    print(f"  {'late − early period (h), mean':42s} {np.mean(D('late') - D('early')):9.3f} {np.mean(N('late') - N('early')):9.3f}")
    print(f"  {'across-cell period SD: first / last cycle':42s} {np.std(per0):5.2f} / {np.std(perL):4.2f}"
          f"   null {np.std(N('first')):5.2f} / {np.std(N('last')):4.2f}")
    print(f"  {'corr(drift, mean period)':42s} {np.corrcoef(D('slope'), D('mean_per'))[0, 1]:9.2f} "
          f"{np.corrcoef(N('slope'), N('mean_per'))[0, 1]:9.2f}")


def figure(slices, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CD, CN = "#2a78d6", "0.6"
    fig, axs = plt.subplots(5, len(slices), figsize=(3.0 * len(slices), 12), layout="constrained", squeeze=False)
    for j, i in enumerate(slices):
        R = results[i]; D = lambda k: R[f"d_{k}"]; N = lambda k: R[f"n_{k}"]
        a = axs[0, j]                                            # drift distribution
        b = np.linspace(-3, 3, 41)
        a.hist(np.clip(N("slope"), -3, 3), bins=b, density=True, color=CN, alpha=0.6, label="null")
        a.hist(np.clip(D("slope"), -3, 3), bins=b, density=True, histtype="step", color=CD, lw=1.3, label="data")
        a.axvline(np.mean(D("slope")), color=CD, ls="--", lw=0.8)
        a.set_title(f"SCN{i}: period drift (n = {D('slope').size})", fontsize=8); a.set_xlabel("h / day")
        a.legend(fontsize=6, frameon=False)
        a = axs[1, j]                                            # early vs late
        a.scatter(D("early"), D("late"), s=5, color=CD, alpha=0.6)
        lim = [min(D("early").min(), D("late").min()), max(D("early").max(), D("late").max())]
        a.plot(lim, lim, color="0.3", lw=0.7); a.set_xlabel("period, early cycles (h)"); a.set_ylabel("late cycles (h)")
        a.set_title("early vs late period", fontsize=8)
        a = axs[2, j]                                            # drift vs mean period (convergence?)
        a.scatter(D("mean_per"), D("slope"), s=5, color=CD, alpha=0.6, label="data")
        a.scatter(N("mean_per"), N("slope"), s=3, color=CN, alpha=0.3, label="null")
        a.axhline(0, color="0.3", lw=0.6); a.set_ylim(-4, 4)
        a.set_xlabel("mean period (h)"); a.set_ylabel("drift (h/day)"); a.set_title("drift vs period", fontsize=8)
        a = axs[3, j]                                            # phase noise
        b = np.linspace(0, 6, 31)
        a.hist(np.clip(N("jitter"), 0, 6), bins=b, density=True, color=CN, alpha=0.6)
        a.hist(np.clip(D("jitter"), 0, 6), bins=b, density=True, histtype="step", color=CD, lw=1.3)
        a.set_xlabel("cycle-to-cycle jitter (h)"); a.set_title("period jitter", fontsize=8)
        a = axs[4, j]                                            # amplitude damping and noise
        a.scatter(N("damp"), N("amp_noise"), s=3, color=CN, alpha=0.3)
        a.scatter(D("damp"), D("amp_noise"), s=5, color=CD, alpha=0.6)
        a.set_xlabel("amplitude damping (1/day)"); a.set_ylabel("log-amplitude noise")
        a.set_xlim(-0.5, 1.5); a.set_ylim(0, 1.0); a.set_title("amplitude", fontsize=8)
    out = os.path.join(OUT, "scn_ttx_drift")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"\n[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", list(S.SLICES), lambda s_: [int(v) for v in s_.split(",")])
    n_surr = arg("--surr", N_SURR, int)
    results = {}
    for i in slices:
        results[i] = analyse(i, n_surr)
        summary(i, results[i])
    figure(slices, results)
