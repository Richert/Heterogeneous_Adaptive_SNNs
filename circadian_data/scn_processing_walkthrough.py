r"""
Walk-through of the SCN preprocessing: what each step does to single cells, and which parts of R(t) it creates
===============================================================================================================

The phase-coherence target of the LMMF fits (scn_lmmf_fit / scn_syngap_fit / scn_lmmf_sim) is
    R(t) = |⟨exp(iθ_i(t))⟩|,  i ∈ cells rhythmic under TTX,
with θ_i the Hilbert phase of the processed trace. The steps (scn_common.process(method="hilbert")):
  0. provided trace X_i(t): hourly PER2::LUC, already smoothed and rescaled by the authors (not raw counts)
  1. detrend         D = X − (24 h centred moving average of X)
  2. band-pass       B = 4th-order Butterworth, periods BAND_H = 18–32 h, zero phase (sosfiltfilt; non-causal)
  3. Hilbert         θ = unwrap(arg(B + i H[B])), envelope A = |B + i H[B]|
  4. selection       rhythmic under TTX: band fraction var(B)/var(D) ≥ RHYTHM_MIN in the TTX epoch and the
                     epoch frequency (phase slope) inside the band
  5. windows         first EDGE_START_H / last EDGE_END_H hours dropped (filter edges); the first TRANSIENT_H
                     hours after a treatment change are dropped from frequency estimates (not from R(t))

FIT TARGET since 2026-10-16: the epoch-wise peak pipeline (scn_epoch_processing, method="peak": epoch-wise
polynomial baseline and scalar normalisation, local-threshold sub-hour peaks, phase interpolated across the epochs
with cycle counting), chosen in scn_pipeline_validation.py; the legacy global Hilbert pipeline above is shown for
comparison. (The old global-threshold peak phase, scn_common.peaks/phase, slips cycles under TTX and is no
longer used.)

Figure 1 (--slice): example cells of high / median / low rhythmicity under TTX, step by step: raw + trend;
    detrended + band-passed + envelope + epoch-wise peaks; legacy Hilbert vs epoch-peak phase and the
    instantaneous Hilbert frequency.
Figure 2 (--slice): population statistics through the steps: envelope relative to the trend (median, IQR)
    and SNR; band fraction per epoch vs the threshold; spread of instantaneous frequencies; R(t) of the fit
    target vs the legacy pipeline and its variants (bands, detrending window); cell sets and weighting.
Figure 3 (--slices): GROUND-TRUTH test of the FIT-TARGET pipeline (epoch peak). Synthetic traces with KNOWN
    phases are put through it, for the same cells:
        x_i = trend_i + a_i(t) cos θ_i(t) + n_i(t)
    trend_i: the cell's measured 24 h trend; a_i: the cell's measured envelope (24 h smoothed; under TTX it
    contains the noise envelope too, so the SNR is, if anything, overestimated); n_i: noise surrogate of the
    cell's out-of-band residual D − B per epoch (Fourier surrogate, in-band level interpolated from the 12–18 h
    and 32–48 h flanks; × --noise-scale). Ground-truth phases:
        sync      θ_i = Θ(t) for all cells (R_true ≡ 1);  Θ = piecewise-linear collective phase of the data
        dephase   θ_i = Θ(t) before TTX; Θ(t_ttx) + ω_i (t − t_ttx) under TTX (ω_i = the cell's measured TTX
                  frequency, no coupling); after washout the offsets relax back, δ_i e^{−(t−t_wash)/TAU_RESYNC}
    each at three levels: filter only (constant amplitude, no noise) | + measured amplitude | + noise.
    Shown: R_true, R_processed and the phase-estimation error (circular SD of θ_processed − θ_true across cells).
    Any difference between R_processed and R_true is a property of the processing / the signal quality, not of
    the dynamics, and must not be fitted.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_processing_walkthrough.py --slice 1 [--slices 1,2,3,4,5]
        [--cells i,j,k] [--noise-scale 1.0] [--seed 3] [--show]
Output: data_paths.MPMF/circadian_data/processing/{scn<slice>_steps_traces, scn<slice>_steps_stats,
        scn_ground_truth}.{png,pdf}
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import argparse
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, butter, sosfiltfilt
import data_paths as dp
import scn_common as S
import scn_epoch_processing as EP

OUT = dp.ensure(os.path.join(S.OUT, "processing"))
TAU_RESYNC = 24.0                 # h, relaxation of the phase offsets after washout (dephase scenario)
H = 24 / (2 * np.pi)              # rad -> h
CPD = 24 / (2 * np.pi)            # rad/h -> cycles/day
C_EP = {"pre": "#4d4d4d", "ttx": "#b2182b", "wash": "#2166ac"}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice", type=int, default=1, choices=S.SLICES, help="slice for figures 1 and 2")
    p.add_argument("--slices", type=str, default=",".join(map(str, S.SLICES)), help="slices for figure 3")
    p.add_argument("--cells", type=str, default=None, help="example cell indices (default: by rhythmicity)")
    p.add_argument("--noise-scale", type=float, default=1.0, help="scale of the synthetic noise surrogates")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


# ─── pipeline (identical to scn_common.process, parametrised for the alternatives) ──────────────────
def pipeline(X, band=S.BAND_H, detrend_h=24):
    D = X - uniform_filter1d(X, size=detrend_h, axis=0, mode="nearest")
    sos = butter(4, [1.0 / band[1], 1.0 / band[0]], btype="band", fs=1.0, output="sos")
    B = sosfiltfilt(sos, D, axis=0)
    Za = hilbert(B, axis=0)
    return D, B, np.unwrap(np.angle(Za), axis=0), np.abs(Za)


def R_of(PH, w=None):
    E = np.exp(1j * PH)
    if w is None:
        return np.abs(E.mean(1))
    return np.abs((w * E).sum(1) / w.sum(1))


def circ_sd(x, axis=1):
    """Circular SD along `axis`, ignoring NaN."""
    ok = np.isfinite(x)
    L = np.abs(np.where(ok, np.exp(1j * np.nan_to_num(x)), 0).sum(axis)) / np.maximum(ok.sum(axis), 1)
    return np.sqrt(-2 * np.log(np.clip(L, 1e-12, 1)))


def epoch_peak(X, d):
    """Fit-target phases (epoch peak pipeline) and the per-epoch peak times of each cell."""
    r = EP.process(X, d["ttx"], d["wash"], method="peak", baseline_kind="poly", norm="scalar")
    tps = [np.concatenate([a + EP.segment_peaks(r["Y"][a:b, c]) for a, b in EP.bounds(d["T"], d["ttx"], d["wash"])])
           for c in range(X.shape[1])]
    return r["PH"], tps


def masked(R, T):
    v0, v1 = S.valid_window(T); R = np.array(R, float); R[:v0] = np.nan; R[v1:] = np.nan
    return R


# ─── ground truth ──────────────────────────────────────────────────────────────────────────────────
def collective_phase(d, sel):
    """Piecewise-linear collective phase Θ(t): slope = data collective frequency in pre / wash epochs and the
    mean measured TTX frequency under TTX; continuous at the boundaries, anchored to the data at the start."""
    T = d["T"]; t = np.arange(T); ttx, wash = d["ttx"], d["wash"]
    Th = np.unwrap(np.angle(np.exp(1j * d["PH"][:, sel]).mean(1)))
    e = d["epochs"]
    sl = lambda a, b: np.polyfit(t[a:b], Th[a:b], 1)[0]
    w_pre, w_wash = sl(*e["pre"]), sl(*e["wash"])
    w_ttx = np.mean(2 * np.pi / d["period"]["ttx"][sel])
    a0 = Th[e["pre"][0]] - w_pre * e["pre"][0]
    Theta = np.where(t < ttx, a0 + w_pre * t, 0.0)
    Theta = np.where((t >= ttx) & (t < wash), a0 + w_pre * ttx + w_ttx * (t - ttx), Theta)
    Theta = np.where(t >= wash, a0 + w_pre * ttx + w_ttx * (wash - ttx) + w_wash * (t - wash), Theta)
    return Theta


def true_phases(d, sel, Theta, scenario):
    T = d["T"]; t = np.arange(T)[:, None]; ttx, wash = d["ttx"], d["wash"]
    n = int(sel.sum())
    if scenario == "sync":
        return np.repeat(Theta[:, None], n, 1)
    om = 2 * np.pi / d["period"]["ttx"][sel][None]
    th = np.repeat(Theta[:, None], n, 1)
    under = (t >= ttx) & (t < wash)
    th = np.where(under, Theta[ttx] + om * (t - ttx), th)
    delta = np.angle(np.exp(1j * (Theta[ttx] + om[0] * (wash - ttx) - Theta[wash])))   # offsets at washout
    after = t >= wash
    return np.where(after, Theta[:, None] + delta[None] * np.exp(-(t - wash) / TAU_RESYNC), th)


def noise_surrogate(r, rng):
    """Fourier surrogate of the out-of-band residual r (1-D); the in-band amplitude spectrum (BAND_H) is
    filled with the geometric mean of the flanks (12–18 h and 32–48 h)."""
    n = r.size
    F = np.fft.rfft(r - r.mean()); f = np.fft.rfftfreq(n, 1.0)
    amp = np.abs(F)
    band = (f >= 1 / S.BAND_H[1]) & (f <= 1 / S.BAND_H[0])
    flank = ((f >= 1 / 48) & (f < 1 / S.BAND_H[1])) | ((f > 1 / S.BAND_H[0]) & (f <= 1 / 12))
    if flank.any():
        amp[band] = np.exp(np.mean(np.log(amp[flank] + 1e-300)))
    ph = rng.uniform(0, 2 * np.pi, f.size); ph[0] = 0
    return np.fft.irfft(amp * np.exp(1j * ph), n)


def synthetic(d, sel, scenario, level, noise_scale, rng):
    """Synthetic traces for the cells `sel` at level 'filter' | 'amplitude' | 'noise'; returns X_syn, θ_true."""
    X = d["X"][:, sel]; T = d["T"]
    trend = uniform_filter1d(X, size=24, axis=0, mode="nearest")
    th = true_phases(d, sel, collective_phase(d, sel), scenario)
    D, B, _, A = pipeline(X)
    if level == "filter":
        a = np.repeat(np.median(A, 0)[None], T, 0)
    else:
        a = uniform_filter1d(A, size=24, axis=0, mode="nearest")
    Xs = trend + a * np.cos(th)
    if level == "noise":
        R = D - B
        cuts = [0, d["ttx"], d["wash"], T]
        for c in range(X.shape[1]):
            for a0, a1 in zip(cuts[:-1], cuts[1:]):
                Xs[a0:a1, c] += noise_scale * noise_surrogate(R[a0:a1, c], rng)
    return Xs, th


def ground_truth(i, noise_scale, seed):
    d = S.process(i); sel = d["rhythmic"]["ttx"]; T = d["T"]
    rng = np.random.default_rng(seed + i)
    out = dict(T=T, ttx=d["ttx"], wash=d["wash"], R_data=masked(R_of(d["PH"][:, sel]), T))
    out["R_data"] = masked(EP.coherence(epoch_peak(d["X"][:, sel], d)[0]), T)
    for sc in ("sync", "dephase"):
        for lv in ("filter", "amplitude", "noise"):
            Xs, th = synthetic(d, sel, sc, lv, noise_scale, rng)
            PHs = epoch_peak(Xs, d)[0]
            out[sc, lv] = dict(R=masked(EP.coherence(PHs), T), err=masked(circ_sd(PHs - th), T))
        out[sc, "true"] = masked(R_of(th), T)
    return out


# ─── figures ─────────────────────────────────────────────────────────────────────────────────────
def shade(ax, d, T):
    ax.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
    v0, v1 = S.valid_window(T)
    for a, b in ((0, v0), (v1, T)):
        ax.axvspan(a / 24, b / 24, color="none", hatch="////", ec="0.75", lw=0, zorder=0)
    for t0 in (d["ttx"], d["wash"]):
        ax.axvspan(t0 / 24, (t0 + S.TRANSIENT_H) / 24, color="#fde0c5", alpha=0.6, lw=0, zorder=0)


def fig_traces(d, cells, plt):
    T = d["T"]; td = np.arange(T) / 24
    D, B, PH, A = pipeline(d["X"])
    PHp, TP = epoch_peak(d["X"][:, cells], d)
    trend = uniform_filter1d(d["X"], size=24, axis=0, mode="nearest")
    fig, axs = plt.subplots(3, len(cells), figsize=(3.2 * len(cells), 6.6), layout="constrained", sharex=True,
                            squeeze=False)
    for j, c in enumerate(cells):
        bf = {e: d["bandfrac"][e][c] for e in d["epochs"]}
        a = axs[0, j]; shade(a, d, T)
        a.plot(td, d["X"][:, c], color="0.2", lw=0.8, label="provided trace X")
        a.plot(td, trend[:, c], color="#e08214", lw=1.0, label="24 h trend")
        a.set_title(f"cell {c}: band fraction pre/TTX/wash {bf['pre']:.2f}/{bf['ttx']:.2f}/{bf['wash']:.2f}"
                    f"{'' if d['rhythmic']['ttx'][c] else ' (excluded)'}", fontsize=7)
        a = axs[1, j]; shade(a, d, T)
        a.plot(td, D[:, c], color="0.6", lw=0.7, label="1. detrended D")
        a.plot(td, B[:, c], color="#2166ac", lw=1.0, label="2. band-passed B")
        a.plot(td, A[:, c], color="#5aae61", lw=0.8, label="3. Hilbert envelope A")
        a.plot(td, -A[:, c], color="#5aae61", lw=0.8)
        pk = TP[j]; a.plot(pk / 24, np.interp(pk, np.arange(T), D[:, c]), "v", color="#b2182b", ms=3,
                           label="epoch-wise peaks (fit target)")
        a = axs[2, j]; shade(a, d, T)
        dif = np.angle(np.exp(1j * (PH[:, c] - PHp[:, j]))) * H
        a.plot(td, dif, color="0.3", lw=0.8, label="legacy Hilbert − epoch-peak phase (h)")
        a.set_ylim(-12, 12); a.set_xlabel("time (days)")
        b2 = a.twinx()
        b2.plot(td[1:], np.diff(PH[:, c]) * CPD, color="#762a83", lw=0.8, label="inst. frequency (cyc/day)")
        b2.set_ylim(0.6, 1.4); b2.tick_params(labelsize=6)
        if j == 0:
            axs[0, j].set_ylabel("X (a.u.)"); axs[1, j].set_ylabel("D, B, ±A")
            a.set_ylabel("phase difference (h)")
            for k in range(2):
                axs[k, j].legend(fontsize=5.5, frameon=False, loc="upper right")
            h1, l1 = a.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
            a.legend(h1 + h2, l1 + l2, fontsize=5.5, frameon=False, loc="lower left")
    fig.suptitle("grey = TTX, orange = post-change transient (dropped from frequencies), hatched = edges dropped "
                 "from R(t)", fontsize=7)
    return fig


def fig_stats(i, d, plt):
    T = d["T"]; td = np.arange(T) / 24; sel = d["rhythmic"]["ttx"]
    D, B, PH, A = pipeline(d["X"])
    trend = uniform_filter1d(d["X"], size=24, axis=0, mode="nearest")
    noise = np.sqrt(uniform_filter1d((D - B) ** 2, size=48, axis=0, mode="nearest"))
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6), layout="constrained")
    # (a) envelope relative to the trend, and SNR
    a = axs[0, 0]; shade(a, d, T)
    rel = A[:, sel] / np.maximum(trend[:, sel], 1e-300)
    q = np.nanpercentile(rel, [25, 50, 75], axis=1)
    a.fill_between(td, q[0], q[2], color="#5aae61", alpha=0.3, lw=0); a.plot(td, q[1], color="#1b7837", lw=1.1,
                                                                            label="envelope A / trend")
    a.set_ylabel("relative amplitude"); a.set_yscale("log")
    b2 = a.twinx(); snr = np.nanmedian(A[:, sel] / np.maximum(noise[:, sel], 1e-300), 1)
    b2.plot(td, snr, color="0.3", lw=0.9, ls="--", label="SNR: A / out-of-band SD"); b2.set_yscale("log")
    b2.tick_params(labelsize=6)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc="lower left")
    a.set_title(f"SCN{i}: amplitude (median, IQR; {sel.sum()} cells rhythmic under TTX)", fontsize=7)
    # (b) band fractions per epoch
    a = axs[0, 1]; bins = np.linspace(0, 1, 41)
    for e, c in C_EP.items():
        a.hist(d["bandfrac"][e], bins=bins, histtype="step", color=c, lw=1.2,
               label=f"{e}: {d['rhythmic'][e].sum()}/{d['N']} rhythmic")
    a.axvline(S.RHYTHM_MIN, color="k", lw=0.8, ls=":"); a.set_xlabel("band fraction var(B)/var(D)")
    a.set_ylabel("cells"); a.legend(fontsize=6, frameon=False, loc="upper left")
    a.set_title("4. selection: rhythmicity per epoch (threshold dotted)", fontsize=7)
    # (c) instantaneous frequency spread
    a = axs[0, 2]; shade(a, d, T)
    f = np.diff(PH[:, sel], axis=0) * CPD
    fs = uniform_filter1d(f, size=24, axis=0, mode="nearest")
    q = np.percentile(fs, [10, 25, 50, 75, 90], axis=1)
    a.fill_between(td[1:], q[0], q[4], color="#762a83", alpha=0.15, lw=0)
    a.fill_between(td[1:], q[1], q[3], color="#762a83", alpha=0.3, lw=0); a.plot(td[1:], q[2], color="#762a83", lw=1)
    for e, (a0, a1) in d["epochs"].items():
        pe = 24 / d["period"][e][sel]
        a.hlines(np.percentile(pe, [25, 75]), a0 / 24, a1 / 24, color=C_EP[e], lw=0.8, ls="--")
    a.set_ylim(0.75, 1.25); a.set_ylabel("frequency (cycles/day)")
    a.set_title("inst. frequency (24 h avg; 10/25/50/75/90%) vs epoch IQR (dashed)", fontsize=7)
    # (d) fit target vs the legacy pipeline and its variants
    a = axs[1, 0]; shade(a, d, T)
    PHe = epoch_peak(d["X"], d)[0]                          # all cells; subsets below
    alts = [("epoch peak (fit target)", EP.coherence(PHe[:, sel]), "k", 1.4),
            ("legacy: 18–32 h, 24 h detrend", R_of(PH[:, sel]), "0.5", 1.1)]
    for band, c in (((20, 28), "#e08214"), ((16, 36), "#5aae61")):
        alts.append((f"legacy, band {band[0]}–{band[1]} h", R_of(pipeline(d["X"][:, sel], band=band)[2]), c, 0.8))
    alts.append(("legacy, 48 h detrend", R_of(pipeline(d["X"][:, sel], detrend_h=48)[2]), "#2166ac", 0.8))
    for lab, R, c, lw in alts:
        a.plot(td, masked(R, T), color=c, lw=lw, label=lab)
    a.set_ylim(0, 1.02); a.set_ylabel("R(t)"); a.set_xlabel("time (days)")
    a.legend(fontsize=5.5, frameon=False, loc="lower left"); a.set_title("R(t): fit target vs legacy", fontsize=7)
    # (e) cell sets and weighting
    a = axs[1, 1]; shade(a, d, T)
    allr = d["rhythmic"]["pre"] & d["rhythmic"]["ttx"] & d["rhythmic"]["wash"]
    Ee = np.where(np.isfinite(PHe), np.exp(1j * np.nan_to_num(PHe)), 0)
    Rw = np.abs((A[:, sel] * Ee[:, sel]).sum(1)) / np.maximum((A[:, sel] * np.isfinite(PHe[:, sel])).sum(1), 1e-300)
    for lab, R, c, lw in (("rhythmic under TTX (fit target)", EP.coherence(PHe[:, sel]), "k", 1.4),
                          (f"all cells ({d['N']})", EP.coherence(PHe), "#e08214", 0.9),
                          (f"rhythmic in all epochs ({allr.sum()})", EP.coherence(PHe[:, allr]), "#5aae61", 0.9),
                          ("weighted by the legacy envelope", Rw, "#2166ac", 0.9)):
        a.plot(td, masked(R, T), color=c, lw=lw, label=lab)
    a.set_ylim(0, 1.02); a.set_xlabel("time (days)"); a.legend(fontsize=5.5, frameon=False, loc="lower left")
    a.set_title("R(t), epoch peak: cell selection and weighting", fontsize=7)
    # (f) finite-size floor
    a = axs[1, 2]; shade(a, d, T)
    a.plot(td, masked(EP.coherence(PHe[:, sel]), T), color="k", lw=1.4, label="data (epoch peak)")
    rng = np.random.default_rng(0)
    Rr = [EP.coherence(PHe[:, sel] + rng.uniform(0, 2 * np.pi, sel.sum())[None]) for _ in range(20)]
    a.plot(td, masked(np.mean(Rr, 0), T), color="0.5", lw=0.9,
           label=f"random constant offsets (≈ 1/√N = {1 / np.sqrt(sel.sum()):.2f})")
    a.set_ylim(0, 1.02); a.set_xlabel("time (days)"); a.legend(fontsize=5.5, frameon=False, loc="lower left")
    a.set_title("finite-size floor of R", fontsize=7)
    return fig


def fig_ground_truth(GT, plt):
    sl = list(GT)
    fig, axs = plt.subplots(3, len(sl), figsize=(2.9 * len(sl), 6.6), layout="constrained", squeeze=False,
                            sharey="row")
    C_LV = {"filter": "#5aae61", "amplitude": "#2166ac", "noise": "#b2182b"}
    for j, i in enumerate(sl):
        g = GT[i]; T = g["T"]; td = np.arange(T) / 24; dd = dict(ttx=g["ttx"], wash=g["wash"])
        for row, sc in enumerate(("sync", "dephase")):
            a = axs[row, j]; shade(a, dd, T)
            a.plot(td, g["R_data"], color="0.6", lw=1.0, label="data (epoch peak)")
            a.plot(td, g[sc, "true"], color="k", lw=1.3, ls=":", label="ground truth")
            for lv, c in C_LV.items():
                a.plot(td, g[sc, lv]["R"], color=c, lw=0.9, label=f"processed: {lv}" if lv == "filter"
                       else f"+ {lv}")
            a.set_ylim(0, 1.02); a.set_title(f"SCN{i}: {sc}", fontsize=8)
            if j == 0:
                a.set_ylabel("R(t)"); a.legend(fontsize=5.5, frameon=False, loc="lower left")
        a = axs[2, j]; shade(a, dd, T)
        for sc, ls in (("sync", "-"), ("dephase", "--")):
            for lv, c in C_LV.items():
                a.plot(td, g[sc, lv]["err"] * H, color=c, lw=0.9, ls=ls,
                       label=f"{sc}, {lv}" if j == 0 else None)
        a.set_ylim(0, 8); a.set_xlabel("time (days)")
        if j == 0:
            a.set_ylabel("phase error, circ. SD (h)"); a.legend(fontsize=5, frameon=False, ncol=2, loc="upper left")
    return fig


def gt_table(GT):
    print("\nGround truth: epoch means of R (true / processed filter / +amplitude / +noise) and data")
    for i, g in GT.items():
        e = S.epochs(i, g["T"])
        for sc in ("sync", "dephase"):
            row = []
            for name, (a0, a1) in e.items():
                m = lambda R: np.nanmean(R[a0:a1])
                row.append(f"{name} {m(g[sc, 'true']):.2f}/{m(g[sc, 'filter']['R']):.2f}/"
                           f"{m(g[sc, 'amplitude']['R']):.2f}/{m(g[sc, 'noise']['R']):.2f}")
            print(f"  SCN{i} {sc:8s} " + " | ".join(row))
        print(f"  SCN{i} data     " + " | ".join(f"{n} {np.nanmean(g['R_data'][a0:a1]):.2f}"
                                               for n, (a0, a1) in e.items()))
        # first day after TTX: the window of the rapid drop
        a0 = g["ttx"]; a1 = g["ttx"] + 36
        print(f"  SCN{i} first 36 h of TTX: data {np.nanmean(g['R_data'][a0:a1]):.2f}, sync+noise "
              f"{np.nanmean(g['sync', 'noise']['R'][a0:a1]):.2f}, dephase true / +noise "
              f"{np.nanmean(g['dephase', 'true'][a0:a1]):.2f} / {np.nanmean(g['dephase', 'noise']['R'][a0:a1]):.2f}")


def main():
    a = parse()
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = S.process(a.slice)
    if a.cells:
        cells = [int(v) for v in a.cells.split(",")]
    else:                                                    # high / median / low rhythmicity under TTX
        bf = d["bandfrac"]["ttx"]; o = np.argsort(bf)
        cells = [int(o[-1]), int(o[o.size // 2]), int(o[int(0.1 * o.size)])]
    saved = []
    for name, fig in ((f"scn{a.slice}_steps_traces", fig_traces(d, cells, plt)),
                      (f"scn{a.slice}_steps_stats", fig_stats(a.slice, d, plt))):
        out = os.path.join(OUT, name); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); saved.append(out)
    GT = {i: ground_truth(i, a.noise_scale, a.seed) for i in (int(v) for v in a.slices.split(","))}
    gt_table(GT)
    out = os.path.join(OUT, "scn_ground_truth" + ("" if a.noise_scale == 1 else f"_ns{a.noise_scale:g}"))
    fig = fig_ground_truth(GT, plt); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); saved.append(out)
    for s_ in saved:
        print(f"[saved] {s_}.png/.pdf")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
