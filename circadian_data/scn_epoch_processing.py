r"""
Epoch-wise preprocessing of the SCN traces: nothing is filtered across a treatment change
=========================================================================================

Each recording is split at TTX application and washout into the epochs pre = [0, t_ttx), ttx = [t_ttx, t_wash),
wash = [t_wash, T). Within each epoch SEPARATELY:

  1. baseline     'poly': least-squares polynomial of degree clip(round(L / POLY_H), 1, POLY_MAX) (L = epoch
                  length, h); 'ma': 24 h centred moving average with mirror padding at the epoch edges.
                  The bioluminescence baseline steps at a medium change, so a baseline fitted across a
                  boundary (the legacy pipeline) leaks the step and the neighbouring epoch's rhythm.
  2. normalise    'scalar': divide by the epoch's robust half-amplitude ((p95 − p5) / 2);
                  'envelope': divide by the 24 h-smoothed Hilbert envelope of the band-passed epoch.
                  The Hilbert phase is invariant to a constant scale and nearly invariant to a slow envelope;
                  normalisation matters for the peak threshold (step 3b) and for amplitude-weighted coherence.
  3a. 'hilbert'   band-pass (4th-order Butterworth, periods `band`, zero phase) and Hilbert phase on the epoch
                  mirror-padded by PAD_H hours; crop; phases of the three epochs concatenated and unwrapped.
                  Cost: a filter edge (~ one cycle) at every epoch boundary.
  3b. 'peak'      Gaussian smoothing (σ = SMOOTH_H), peaks ≥ MIN_PEAK_DIST h apart with prominence ≥ PROM of
                  the normalised amplitude (per epoch, i.e. LOCAL threshold), parabolic sub-hour refinement;
                  the peaks of all epochs are merged and the phase is interpolated linearly between consecutive
                  peaks ACROSS the boundaries (2π × the number of cycles in the interval, so a missed peak is
                  counted as a cycle when the interval is ≥ GAP_FRAC × the cell's median period). NaN before the
                  first and after the last peak. No filter edge; the phase is piecewise linear (one frequency per
                  cycle), so a change within a cycle is spread over that cycle.

process(X, t_ttx, t_wash, ...) works on any (T, N) array (data or synthetic traces); process_slice(i, ...) on a
slice. Phase 0 = PER2 maximum in both methods.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks, butter, sosfiltfilt, hilbert

SMOOTH_H, MIN_PEAK_DIST, PROM = 2.0, 16, 0.3
POLY_H, POLY_MAX = 60.0, 4
PAD_H = 48
GAP_FRAC = 1.5


def bounds(T, t_ttx, t_wash):
    return [(0, t_ttx), (t_ttx, t_wash), (t_wash, T)]


def baseline(x, kind="poly"):
    """Baseline (L, N) of one epoch segment x (L, N)."""
    L = x.shape[0]
    if kind == "poly":
        deg = int(np.clip(round(L / POLY_H), 1, POLY_MAX))
        s = np.linspace(-1, 1, L)
        V = np.polynomial.legendre.legvander(s, deg)
        coef, *_ = np.linalg.lstsq(V, x, rcond=None)
        return V @ coef
    if kind == "ma":
        p = min(24, L - 1)
        xp = np.pad(x, ((p, p), (0, 0)), mode="reflect")
        return uniform_filter1d(xp, size=24, axis=0, mode="nearest")[p:p + L]
    raise ValueError(kind)


def bandpass_hilbert(y, band):
    """Band-passed segment and its analytic signal, computed on the mirror-padded segment and cropped."""
    L = y.shape[0]; p = min(PAD_H, L - 1)
    yp = np.pad(y, ((p, p), (0, 0)), mode="reflect")
    sos = butter(4, [1.0 / band[1], 1.0 / band[0]], btype="band", fs=1.0, output="sos")
    B = sosfiltfilt(sos, yp, axis=0)
    Za = hilbert(B, axis=0)
    return B[p:p + L], Za[p:p + L]


def normalise(d, kind="scalar", band=(18.0, 32.0)):
    if kind == "none":
        return d
    if kind == "scalar":
        a = 0.5 * np.subtract(*np.percentile(d, [95, 5], axis=0))
        return d / np.maximum(a, 1e-300)[None]
    if kind == "envelope":
        _, Za = bandpass_hilbert(d, band)
        env = uniform_filter1d(np.abs(Za), size=24, axis=0, mode="nearest")
        return d / np.maximum(env, 1e-300 + 1e-6 * env.max(0))
    raise ValueError(kind)


def refine(y, p):
    if 0 < p < y.size - 1:
        a, b, c = y[p - 1], y[p], y[p + 1]
        den = a - 2 * b + c
        return p + (0.5 * (a - c) / den if den != 0 else 0.0)
    return float(p)


def segment_peaks(y):
    """Sub-hour peak times of one normalised epoch segment (1-D), with a LOCAL prominence threshold."""
    ys = gaussian_filter1d(y, SMOOTH_H)
    amp = np.subtract(*np.percentile(ys, [95, 5]))
    pk, _ = find_peaks(ys, distance=MIN_PEAK_DIST, prominence=PROM * amp)
    return np.array([refine(ys, p) for p in pk])


def peak_phase(tp, T):
    """Unwrapped phase (T,) from merged peak times tp; cycles counted over long gaps; NaN outside."""
    ph = np.full(T, np.nan)
    if tp.size < 2:
        return ph
    dt = np.diff(tp)
    P = np.median(dt)
    ncyc = np.where(dt >= GAP_FRAC * P, np.maximum(1, np.round(dt / P)), 1)
    cyc = np.concatenate([[0], np.cumsum(ncyc)])
    t = np.arange(T); inside = (t >= tp[0]) & (t <= tp[-1])
    ph[inside] = np.interp(t[inside], tp, 2 * np.pi * cyc)
    return ph


def process(X, t_ttx, t_wash, method="hilbert", baseline_kind="poly", norm="scalar", band=(18.0, 32.0)):
    """Phases (T, N) (unwrapped; NaN where undefined), normalised detrended traces (T, N) and, for 'hilbert',
    the normalised envelope (T, N)."""
    X = np.asarray(X, float)
    T, N = X.shape
    Yn = np.empty_like(X); PH = np.full((T, N), np.nan); ENV = np.full((T, N), np.nan)
    tps = [[] for _ in range(N)]
    for a, b in bounds(T, t_ttx, t_wash):
        seg = X[a:b]
        d = seg - baseline(seg, baseline_kind)
        y = normalise(d, norm, band)
        Yn[a:b] = y
        if method == "hilbert":
            _, Za = bandpass_hilbert(y, band)
            PH[a:b] = np.angle(Za); ENV[a:b] = np.abs(Za)
        elif method == "peak":
            for c in range(N):
                tps[c].append(a + segment_peaks(y[:, c]))
        else:
            raise ValueError(method)
    if method == "hilbert":
        PH = np.unwrap(PH, axis=0)
    else:
        for c in range(N):
            PH[:, c] = peak_phase(np.concatenate(tps[c]), T)
    return dict(PH=PH, Y=Yn, ENV=ENV)


def process_slice(i, cells=None, **kw):
    import scn_common as S
    X, _ = S.load(i)
    if cells is not None:
        X = X[:, cells]
    return process(X, S.TTX_TIME[i], S.WASH_TIME[i], **kw)


def coherence(PH):
    """R(t) over the cells with a defined phase at t."""
    ok = np.isfinite(PH)
    Z = np.where(ok, np.exp(1j * np.nan_to_num(PH)), 0).sum(1)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(Z) / np.where(n > 0, n, np.nan)
