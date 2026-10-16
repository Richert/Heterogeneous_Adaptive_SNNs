r"""
Shared loading and preprocessing of the Abel et al. (2016) SCN PER2::LUC data
=============================================================================

Data (download_scn_data.py): 5 SCN slices; per slice
  full_time_trajectories/scn<i>_full_data.csv   T x N hourly bioluminescence of the N cells tracked
                                                through ALL conditions (pre-TTX, TTX, washout)
  full_time_trajectories/scn<i>_full_locs.csv   N x 2 pixel coordinates of those cells
  ttx_wash_trajectories/...                     a larger cell set, post-washout segment only (not used here)
  raw_mic_scores/mic_indiv_scn<i>.csv           pairwise MIC of the ttx_wash cell set
TTX is applied at TTX_TIME[i] h and washed out at WASH_TIME[i] h (README of the repository).

Preprocessing of one trace x(t) (hourly):
  1. detrend: x − 24 h centred moving average (removes the slow bioluminescence decay)
  2. smooth: Gaussian, σ = SMOOTH_H hours
  3. peaks: local maxima ≥ MIN_PEAK_DIST h apart with prominence ≥ PROM_FRAC × the trace's robust amplitude
  4. phase: 2π-increments between consecutive peaks, linear in between (NaN outside the first/last peak)
Periods: median peak-to-peak interval within an epoch (≥ MIN_CYCLES intervals, else NaN).
Epochs exclude the first EDGE_START_H and last EDGE_END_H hours of the recording (filter edge effects of the
band-pass / Hilbert phase), and the first TRANSIENT_H hours after TTX application and washout.

Continuous phase (preferred, process(..., method="hilbert")): the detrended trace is band-passed to periods
BAND_H (4th-order Butterworth, zero phase) and the Hilbert phase is unwrapped; a cell's frequency in an epoch
is the least-squares slope of its unwrapped phase (sub-hour period resolution, robust to low amplitude).
Rhythmicity: fraction of the detrended signal's variance inside the band, ≥ RHYTHM_MIN in that epoch.
"""
import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks, butter, sosfiltfilt, hilbert

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))
import data_paths as dp

RAW = dp.mpmf("circadian_data", "raw", "scn-resynchronization-data-2016-master")
OUT = dp.ensure(dp.mpmf("circadian_data"))
SLICES = (1, 2, 3, 4, 5)
TTX_TIME = {1: 90, 2: 109, 3: 82, 4: 107, 5: 113}
WASH_TIME = {1: 234, 2: 252, 3: 224, 4: 240, 5: 242}
SMOOTH_H, MIN_PEAK_DIST, PROM_FRAC, MIN_CYCLES = 2.0, 16, 0.2, 2
TRANSIENT_H = 24            # hours after a treatment change excluded from period estimates
EDGE_START_H, EDGE_END_H = 24, 72   # recording edges excluded from period estimates and R(t) (filter edges)
BAND_H = (18.0, 32.0)       # circadian band (period, h) for the Hilbert phase
RHYTHM_MIN = 0.5            # min fraction of detrended variance in the band for a cell to count as rhythmic


def load(i):
    """(X, locs): bioluminescence (T h, N cells) and pixel coordinates (N, 2) of slice i."""
    X = np.loadtxt(os.path.join(RAW, "full_time_trajectories", f"scn{i}_full_data.csv"), delimiter=",")
    locs = np.loadtxt(os.path.join(RAW, "full_time_trajectories", f"scn{i}_full_locs.csv"), delimiter=",")
    return X, locs


def epochs(i, T):
    """Hour ranges [start, stop) of the three conditions, excluding the first TRANSIENT_H h after a change."""
    t1, t2 = TTX_TIME[i], WASH_TIME[i]
    return {"pre": (EDGE_START_H, t1), "ttx": (t1 + TRANSIENT_H, t2), "wash": (t2 + TRANSIENT_H, T - EDGE_END_H)}


def valid_window(T):
    """Hours [start, stop) used for R(t): the recording without its first / last edge hours."""
    return EDGE_START_H, T - EDGE_END_H


def detrend_smooth(X):
    trend = uniform_filter1d(X, size=24, axis=0, mode="nearest")
    return gaussian_filter1d(X - trend, SMOOTH_H, axis=0)


def peaks(y):
    amp = np.subtract(*np.nanpercentile(y, [95, 5]))
    p, _ = find_peaks(y, distance=MIN_PEAK_DIST, prominence=PROM_FRAC * amp)
    return p


def phase(y, pk=None):
    """Unwrapped phase from peak times (2π per cycle, linear between peaks; NaN outside)."""
    pk = peaks(y) if pk is None else pk
    ph = np.full(y.size, np.nan)
    if pk.size >= 2:
        t = np.arange(y.size)
        inside = (t >= pk[0]) & (t <= pk[-1])
        ph[inside] = np.interp(t[inside], pk, 2 * np.pi * np.arange(pk.size))
    return ph


def hilbert_phase(D):
    """Unwrapped Hilbert phase (T, N) of the band-passed detrended traces D, and the band-passed traces."""
    sos = butter(4, [1.0 / BAND_H[1], 1.0 / BAND_H[0]], btype="band", fs=1.0, output="sos")
    B = sosfiltfilt(sos, D, axis=0)
    return np.unwrap(np.angle(hilbert(B, axis=0)), axis=0), B


def process(i, method="hilbert"):
    """Everything per slice: smoothed traces, peaks, phases, per-epoch periods and amplitudes."""
    X, locs = load(i)
    Y = detrend_smooth(X)
    T, N = Y.shape
    eps = epochs(i, T)
    if method == "hilbert":
        D = X - uniform_filter1d(X, size=24, axis=0, mode="nearest")
        PH, B = hilbert_phase(D)
        t = np.arange(T)
        per, rhythmic, bandfrac = {}, {}, {}
        for e, (a, b) in eps.items():
            sl = slice(a, b)
            slope = np.polyfit(t[sl] - t[sl].mean(), PH[sl], 1)[0]         # rad / h, per cell
            per[e] = 2 * np.pi / slope
            bandfrac[e] = B[sl].var(0) / np.maximum(D[sl].var(0), 1e-300)
            rhythmic[e] = (bandfrac[e] >= RHYTHM_MIN) & (per[e] > BAND_H[0]) & (per[e] < BAND_H[1])
        return dict(X=X, Y=Y, locs=locs, PH=PH, B=B, period=per, rhythmic=rhythmic, bandfrac=bandfrac,
                    epochs=eps, T=T, N=N, ttx=TTX_TIME[i], wash=WASH_TIME[i])
    PH = np.full((T, N), np.nan); per = {e: np.full(N, np.nan) for e in eps}
    amp = {e: np.full(N, np.nan) for e in eps}; PK = []
    for c in range(N):
        pk = peaks(Y[:, c]); PK.append(pk)
        PH[:, c] = phase(Y[:, c], pk)
        for e, (a, b) in eps.items():
            q = pk[(pk >= a) & (pk < b)]
            if q.size - 1 >= MIN_CYCLES:
                per[e][c] = np.median(np.diff(q))
            seg = Y[a:b, c]
            amp[e][c] = np.subtract(*np.percentile(seg, [95, 5])) / max(np.std(X[a:b, c]), 1e-12) if seg.size else np.nan
    return dict(X=X, Y=Y, locs=locs, PH=PH, peaks=PK, period=per, amp=amp, epochs=eps, T=T, N=N,
                ttx=TTX_TIME[i], wash=WASH_TIME[i])


def lobes(locs):
    """Left/right SCN assignment: 2-means on the pixel coordinates (fixed seed), labels ordered along the
    axis that separates the two centres (0 = smaller coordinate), so labels are reproducible."""
    from scipy.cluster.vq import kmeans2
    cen, lab = kmeans2(np.asarray(locs, float), 2, seed=3, minit="++")
    ax = int(np.argmax(np.abs(cen[0] - cen[1])))
    return lab if cen[0, ax] < cen[1, ax] else 1 - lab


def order_parameter(PH, cells=None):
    """R(t) = |<exp(iθ)>| over the given cells (cells with a defined phase at t only) and their number."""
    P = PH if cells is None else PH[:, cells]
    ok = np.isfinite(P)
    Z = np.where(ok, np.exp(1j * np.nan_to_num(P)), 0).sum(1)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(Z) / np.where(n > 0, n, np.nan), n
