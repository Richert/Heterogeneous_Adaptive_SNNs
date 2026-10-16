r"""
Fit targets of the SCN slices from the validated epoch-wise peak pipeline
=========================================================================

Per slice (cells rhythmic under TTX, scn_common.process — the same cell set as before):
  phases   epoch peak pipeline (scn_epoch_processing: epoch-wise polynomial baseline, scalar normalisation,
           local-threshold sub-hour peaks, phase interpolated across epochs with cycle counting); validated
           against ground truth in scn_pipeline_validation.py (best of legacy / global peak / epoch peak).
  ω_i      intrinsic frequencies: least-squares slope of the peak phase over the TTX epoch without its first
           TRANSIENT_H hours (rad/h); cells whose period leaves S.BAND_H are dropped.
  R        coherence of the slice and of the two lobes (scn_common.lobes), hourly.
  ceiling  measurement ceiling c(t): R of PERFECTLY synchronised synthetic traces (each cell's measured trend,
           envelope and noise; scn_pipeline_validation "sync") through the same pipeline, averaged over
           N_CEIL noise realisations; slice and lobes. The model is compared as c(t) |Z(t)|: the loss of
           coherence that the measurement itself causes (low amplitude / SNR under TTX, peak-timing jitter) is
           not attributed to the dynamics. CALIBRATION: the data bound the ceiling from below (c ≥ R), so the
           noise surrogates are scaled by the largest s in NOISE_GRID for which the masked mean of c is ≥ that
           of R in the coupled epochs (pre-TTX and post-washout); s = 1 is the uncalibrated surrogate (whose
           in-band level, interpolated from the spectral flanks, can overestimate the noise).
  mask     1 inside scn_common.valid_window, 0 within ±MASK_H of TTX application and washout (the peak phase
           is piecewise linear and spreads a change over about ±1 cycle) and where R is undefined.
  PH       per-cell phases (for the initial state of the simulations; NaN before the first peak).
Cached in data_paths.MPMF/circadian_data/targets/scn<i>_target.npz (delete to rebuild).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_targets.py [--slices 1,2,3,4,5]   # build / refresh cache
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import data_paths as dp
import scn_common as S
import scn_epoch_processing as EP

OUT = dp.ensure(os.path.join(S.OUT, "targets"))
MASK_H = 24
N_CEIL = 3
NOISE_GRID = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.0)
SEED = 11
PIPE = dict(method="peak", baseline_kind="poly", norm="scalar")


def _lobe_R(PH, lobe):
    return EP.coherence(PH), EP.coherence(PH[:, lobe == 0]), EP.coherence(PH[:, lobe == 1])


def build(i):
    import scn_pipeline_validation as V                      # ground-truth generator (heavy imports)
    d = S.process(i); sel0 = d["rhythmic"]["ttx"]; T = d["T"]; ttx, wash = d["ttx"], d["wash"]
    PH = EP.process(d["X"][:, sel0], ttx, wash, **PIPE)["PH"]
    a, b = d["epochs"]["ttx"]; t = np.arange(T)
    om = np.full(PH.shape[1], np.nan)
    for c in range(PH.shape[1]):
        ok = np.isfinite(PH[a:b, c])
        if ok.sum() > 24:
            om[c] = np.polyfit(t[a:b][ok], PH[a:b, c][ok], 1)[0]
    keep = np.isfinite(om) & (2 * np.pi / np.where(om > 0, om, np.nan) > S.BAND_H[0]) & \
        (2 * np.pi / np.where(om > 0, om, np.nan) < S.BAND_H[1])
    cells = np.where(sel0)[0][keep]; PH = PH[:, keep]; om = om[keep]
    sel = np.zeros(d["N"], bool); sel[cells] = True
    lobe = S.lobes(d["locs"])[sel]
    R, RA, RB = _lobe_R(PH, lobe)
    # measurement ceiling (sync ground truth through the same pipeline), noise scale calibrated (c ≥ R)
    v0, v1 = S.valid_window(T)
    base = ((t >= v0) & (t < v1) & (np.abs(t - ttx) >= MASK_H) & (np.abs(t - wash) >= MASK_H)
            & np.isfinite(R) & np.isfinite(RA) & np.isfinite(RB))
    coupled = base & ((t < ttx) | (t >= wash))
    inputs = []
    for k in range(N_CEIL):
        trend, A, noise, kick, Theta = V.ground_truth_inputs(d, sel, 1.0, np.random.default_rng(SEED + 100 * i + k))
        inputs.append((trend + A * np.cos(V.true_phases(d, sel, Theta, "sync", kick)), noise))
    for s in NOISE_GRID:
        C = np.array([_lobe_R(EP.process(xs + s * nz, ttx, wash, **PIPE)["PH"], lobe) for xs, nz in inputs])
        c, cA, cB = np.nanmean(C, axis=0)
        ok = all(np.nanmean(cc[coupled & (t < ttx)]) >= np.nanmean(rr[coupled & (t < ttx)]) and
                 np.nanmean(cc[coupled & (t >= wash)]) >= np.nanmean(rr[coupled & (t >= wash)])
                 for cc, rr in ((c, R), (cA, RA), (cB, RB)))
        if ok:
            break
    noise_scale = s
    mask = (base & np.isfinite(c)).astype(float)
    out = dict(slice=i, T=T, ttx=ttx, wash=wash, cells=cells, lobe=lobe, om=om, PH=PH, R=R, RA=RA, RB=RB,
               c=c, cA=cA, cB=cB, mask=mask, epochs=np.array([d["epochs"][e] for e in ("pre", "ttx", "wash")]),
               valid=np.array([v0, v1]), mask_h=MASK_H, n_dropped=int((~keep).sum()), noise_scale=noise_scale)
    np.savez_compressed(os.path.join(OUT, f"scn{i}_target.npz"), **out)
    return out


def start_hour(d, t_min=None, frac=0.9):
    """First hour ≥ t_min (default: start of the valid window) at which ≥ frac of the cells have a defined phase
    (the peak phase is NaN before a cell's first peak); simulations start there from the measured state."""
    t_min = S.valid_window(int(d["T"]))[0] if t_min is None else int(t_min)
    f = np.isfinite(d["PH"]).mean(1)
    ok = np.where((np.arange(f.size) >= t_min) & (f >= frac))[0]
    return int(ok[0]) if ok.size else t_min


def load(i, rebuild=False):
    f = os.path.join(OUT, f"scn{i}_target.npz")
    if rebuild or not os.path.exists(f):
        return build(i)
    z = np.load(f)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    for i in arg("--slices", list(S.SLICES), lambda s_: [int(v) for v in s_.split(",")]):
        g = build(i)
        e = {n: tuple(g["epochs"][k]) for k, n in enumerate(("pre", "ttx", "wash"))}
        m = lambda x, a, b: np.nanmean(np.where(g["mask"][a:b] > 0, x[a:b], np.nan))
        print(f"SCN{i}: {g['cells'].size} cells ({g['n_dropped']} dropped: period outside band), noise scale "
              f"{g['noise_scale']:g}, lobes "
              f"{np.bincount(g['lobe']).tolist()}; masked-epoch means R / ceiling: "
              + " | ".join(f"{n} {m(g['R'], *ab):.2f}/{m(g['c'], *ab):.2f}" for n, ab in e.items()))
