r"""
Which phase-estimation pipeline can we trust? Ground-truth validation of R(t) for the SCN fits
==============================================================================================

Pipelines (cells rhythmic under TTX, identical for all):
  legacy (global Hilbert)  global 24 h moving-average detrend + global 18–32 h band-pass + Hilbert (current fit
                           target; scn_common.process)
  global peak              global 24 h moving-average detrend + Gaussian smoothing (σ = 2 h) over the whole
                           recording; peaks on integer hours with a GLOBAL prominence threshold (0.2 × the whole
                           trace's p95 − p5); phase linear between consecutive peaks, 2π per interval (no
                           missed-cycle counting) (scn_common.peaks/phase; the "peak-based phase" curve of
                           scn_processing_walkthrough)
  epoch peak               epoch-wise baseline (BASE_DEFAULT) + normalisation (NORM_DEFAULT) + local-threshold
                           sub-hour peaks, phase interpolated across epochs with cycle counting
                           (scn_epoch_processing)

Ground truth, per slice and for the same cells: x_i = trend_i + a_i(t) cos θ_i(t) + n_i(t) with
  trend_i   per-epoch 48 h mirror-padded moving average of the cell's trace (steps at the medium changes, and
            different from both baseline estimators)
  a_i       per-epoch, 24 h-smoothed Hilbert envelope of the cell (its amplitude steps / decays as measured)
  n_i       per-epoch Fourier surrogate of the cell's out-of-band residual, in-band level interpolated from the
            flanks (scn_processing_walkthrough.noise_surrogate), × --noise-scale; ONE realisation per slice,
            shared by all scenarios and pipelines
and the phase scenarios
  sync      θ_i = Θ(t): R_true ≡ 1 — the measurement ceiling c(t) of each pipeline
  dephase   free-running at the cell's measured TTX frequency from t_ttx, offsets relax after washout (τ = 24 h)
  reset     dephase + an instantaneous wrapped-Cauchy phase kick of scale RESET_SIGMA at t_ttx (R_true drops
            by e^{−σ} at once): tests how fast each pipeline can follow an abrupt change
Metrics over the R(t) window (scn_common.valid_window): RMSE of R_processed − R_true overall and in the 48 h after
TTX application / washout, mean bias under TTX, and the median phase error (circular SD of θ_processed − θ_true).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_pipeline_validation.py [--slices 1,2,3,4,5]
        [--noise-scale 1.0] [--seed 3] [--show]
Output: data_paths.MPMF/circadian_data/pipeline_validation/{validation_R, validation_error, validation_zoom}
        .{png,pdf}, validation_metrics.csv, scn<i>_targets.npz (data R and ceiling c(t) per pipeline)
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import argparse
import csv
import numpy as np
from scipy.ndimage import uniform_filter1d
import data_paths as dp
import scn_common as S
import scn_epoch_processing as EP
import scn_processing_walkthrough as W

OUT = dp.ensure(os.path.join(S.OUT, "pipeline_validation"))
RESET_SIGMA = 0.7                       # rad, wrapped-Cauchy kick scale at TTX (R × e^{−0.7} ≈ 0.5)
BASE_DEFAULT, NORM_DEFAULT = "poly", "scalar"
H = 24 / (2 * np.pi)
MAIN = {"legacy (global Hilbert)": "0.45", "global peak": "#e08214", "epoch peak": "#b2182b"}
SCEN = ("sync", "dephase", "reset")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slices", type=str, default=",".join(map(str, S.SLICES)))
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def global_peak_phase(X):
    """The old peak phase: global detrend + smoothing, global prominence threshold, 2π per peak interval."""
    Y = S.detrend_smooth(X)
    return np.column_stack([S.phase(Y[:, c]) for c in range(X.shape[1])])


def pipelines():
    """name -> callable(X, ttx, wash) -> PH."""
    return {"legacy (global Hilbert)": lambda X, a, b: W.pipeline(X)[2],
            "global peak": lambda X, a, b: global_peak_phase(X),
            "epoch peak": lambda X, a, b: EP.process(X, a, b, method="peak", baseline_kind=BASE_DEFAULT,
                                                      norm=NORM_DEFAULT)["PH"]}


# ─── ground truth ──────────────────────────────────────────────────────────────────────────────────
def epochwise(X, ttx, wash, f):
    out = np.empty_like(X)
    for a, b in EP.bounds(X.shape[0], ttx, wash):
        out[a:b] = f(X[a:b])
    return out


def ma_mirror(x, size):
    p = min(size, x.shape[0] - 1)
    xp = np.pad(x, ((p, p), (0, 0)), mode="reflect")
    return uniform_filter1d(xp, size=size, axis=0, mode="nearest")[p:p + x.shape[0]]


def true_phases(d, sel, Theta, scenario, kick):
    if scenario == "sync":
        return np.repeat(Theta[:, None], int(sel.sum()), 1)
    T = d["T"]; t = np.arange(T)[:, None]; ttx, wash = d["ttx"], d["wash"]
    om = 2 * np.pi / d["period"]["ttx"][sel][None]
    k = kick[None] if scenario == "reset" else 0.0
    th = np.repeat(Theta[:, None], om.size, 1)
    th = np.where((t >= ttx) & (t < wash), Theta[ttx] + k + om * (t - ttx), th)
    delta = np.angle(np.exp(1j * (Theta[ttx] + k + om * (wash - ttx) - Theta[wash])))
    return np.where(t >= wash, Theta[:, None] + delta * np.exp(-(t - wash) / W.TAU_RESYNC), th)


def ground_truth_inputs(d, sel, noise_scale, rng):
    X = d["X"][:, sel]; T = d["T"]; ttx, wash = d["ttx"], d["wash"]
    trend = epochwise(X, ttx, wash, lambda s: ma_mirror(s, 48))
    D = epochwise(X, ttx, wash, lambda s: s - EP.baseline(s, "ma"))
    B = np.empty_like(D); A = np.empty_like(D)
    for a, b in EP.bounds(T, ttx, wash):
        Bs, Za = EP.bandpass_hilbert(D[a:b], S.BAND_H)
        B[a:b] = Bs; A[a:b] = ma_mirror(np.abs(Za), 24)
    noise = np.zeros_like(X)
    for c in range(X.shape[1]):
        for a, b in EP.bounds(T, ttx, wash):
            noise[a:b, c] = noise_scale * W.noise_surrogate((D - B)[a:b, c], rng)
    u = rng.uniform(size=X.shape[1])
    kick = np.angle(np.exp(1j * RESET_SIGMA * np.tan(np.pi * (u - 0.5))))
    return trend, A, noise, kick, W.collective_phase(d, sel)


# ─── per slice ───────────────────────────────────────────────────────────────────────────────────
def run_slice(i, P, noise_scale, seed):
    d = S.process(i); sel = d["rhythmic"]["ttx"]; T = d["T"]; ttx, wash = d["ttx"], d["wash"]
    rng = np.random.default_rng(seed + 10 * i)
    trend, A, noise, kick, Theta = ground_truth_inputs(d, sel, noise_scale, rng)
    v0, v1 = S.valid_window(T)
    def msk(R):
        R = np.array(R, float); R[:v0] = np.nan; R[v1:] = np.nan
        return R
    res = dict(T=T, ttx=ttx, wash=wash, data={}, truth={}, R={}, err={}, metrics=[])
    Xd = d["X"][:, sel]
    for name, f in P.items():
        res["data"][name] = msk(EP.coherence(f(Xd, ttx, wash)))
    for sc in SCEN:
        th = true_phases(d, sel, Theta, sc, kick)
        Xs = trend + A * np.cos(th) + noise
        Rt = msk(EP.coherence(th)); res["truth"][sc] = Rt
        for name, f in P.items():
            PH = f(Xs, ttx, wash)
            Rp = msk(EP.coherence(PH))
            e = PH - th; ok = np.isfinite(e)
            L = np.abs(np.where(ok, np.exp(1j * np.nan_to_num(e)), 0).sum(1)) / np.maximum(ok.sum(1), 1)
            err = msk(np.sqrt(-2 * np.log(np.clip(L, 1e-12, 1))))
            res["R"][sc, name] = Rp; res["err"][sc, name] = err
            dR = Rp - Rt
            win = lambda a, b: np.sqrt(np.nanmean(dR[a:b] ** 2))
            res["metrics"].append(dict(slice=i, scenario=sc, pipeline=name, rmse=np.sqrt(np.nanmean(dR ** 2)),
                                       rmse_ttx48=win(ttx, ttx + 48), rmse_wash48=win(wash, wash + 48),
                                       bias_ttx=np.nanmean(dR[ttx:wash]), phase_err_h=np.nanmedian(err) * H))
    np.savez_compressed(os.path.join(OUT, f"scn{i}_targets.npz"),
                        t=np.arange(T), cells=np.where(sel)[0],
                        **{f"R_data|{k}": v for k, v in res["data"].items()},
                        **{f"ceiling|{k}": res["R"]["sync", k] for k in P})
    return res


# ─── output ──────────────────────────────────────────────────────────────────────────────────────
def table(M, P):
    print(f"\nMean over slices (R_processed − R_true): RMSE all | 48 h after TTX | 48 h after wash | bias under TTX "
          f"| median phase error (h)")
    for sc in SCEN:
        print(f"  ── {sc}")
        for name in P:
            r = [m for m in M if m["scenario"] == sc and m["pipeline"] == name]
            g = lambda k: np.nanmean([m[k] for m in r])
            print(f"    {name:32s} {g('rmse'):6.3f} {g('rmse_ttx48'):7.3f} {g('rmse_wash48'):7.3f} "
                  f"{g('bias_ttx'):+7.3f} {g('phase_err_h'):6.2f}")
    print("\nPer slice, main pipelines, RMSE all (sync / dephase / reset) and reset RMSE 48 h after TTX")
    for i in sorted({m["slice"] for m in M}):
        row = []
        for name in MAIN:
            g = {m["scenario"]: m for m in M if m["slice"] == i and m["pipeline"] == name}
            row.append(f"{name} {g['sync']['rmse']:.2f}/{g['dephase']['rmse']:.2f}/{g['reset']['rmse']:.2f} "
                       f"({g['reset']['rmse_ttx48']:.2f})")
        print(f"  SCN{i}: " + " | ".join(row))


def figures(results, plt):
    sl = list(results)
    shade = lambda a, g: W.shade(a, dict(ttx=g["ttx"], wash=g["wash"]), g["T"])
    # R(t): data + three scenarios
    fig1, axs = plt.subplots(len(sl), 4, figsize=(11, 2.1 * len(sl)), layout="constrained", squeeze=False,
                             sharey=True)
    for r, i in enumerate(sl):
        g = results[i]; td = np.arange(g["T"]) / 24
        for c, sc in enumerate(("data",) + SCEN):
            a = axs[r, c]; shade(a, g)
            if sc != "data":
                a.plot(td, g["truth"][sc], color="k", lw=1.4, ls=":", label="ground truth")
            for name, col in MAIN.items():
                R = g["data"][name] if sc == "data" else g["R"][sc, name]
                a.plot(td, R, color=col, lw=0.9, label=name)
            a.set_ylim(0, 1.02); a.set_title(f"SCN{i}: {sc}", fontsize=8)
            if r == len(sl) - 1:
                a.set_xlabel("time (days)")
        axs[r, 0].set_ylabel("R(t)")
    axs[0, 1].legend(fontsize=6, frameon=False, loc="lower left")
    # phase error
    fig2, axs = plt.subplots(len(sl), 3, figsize=(8.5, 2.0 * len(sl)), layout="constrained", squeeze=False,
                             sharey=True)
    for r, i in enumerate(sl):
        g = results[i]; td = np.arange(g["T"]) / 24
        for c, sc in enumerate(SCEN):
            a = axs[r, c]; shade(a, g)
            for name, col in MAIN.items():
                a.plot(td, g["err"][sc, name] * H, color=col, lw=0.9, label=name)
            a.set_ylim(0, 6); a.set_title(f"SCN{i}: {sc}", fontsize=8)
        axs[r, 0].set_ylabel("phase error (h)")
    axs[0, 0].legend(fontsize=6, frameon=False, loc="upper left")
    for a in axs[-1]:
        a.set_xlabel("time (days)")
    # zoom on the transitions
    fig3, axs = plt.subplots(len(sl), 4, figsize=(11, 2.1 * len(sl)), layout="constrained", squeeze=False,
                             sharey=True)
    for r, i in enumerate(sl):
        g = results[i]; td = np.arange(g["T"]) / 24
        for c, (sc, t0) in enumerate((("reset", g["ttx"]), ("data", g["ttx"]), ("dephase", g["wash"]),
                                      ("data", g["wash"]))):
            a = axs[r, c]; shade(a, g)
            if sc != "data":
                a.plot(td, g["truth"][sc], color="k", lw=1.4, ls=":", label="ground truth")
            for name, col in MAIN.items():
                a.plot(td, g["data"][name] if sc == "data" else g["R"][sc, name], color=col, lw=1.0, label=name)
            a.set_xlim(t0 / 24 - 2, t0 / 24 + 3); a.set_ylim(0, 1.02)
            a.set_title(f"SCN{i}: {sc} at {'TTX' if t0 == g['ttx'] else 'washout'}", fontsize=8)
            if r == len(sl) - 1:
                a.set_xlabel("time (days)")
        axs[r, 0].set_ylabel("R(t)")
    axs[0, 0].legend(fontsize=6, frameon=False, loc="lower left")
    return {"validation_R": fig1, "validation_error": fig2, "validation_zoom": fig3}


def main():
    a = parse()
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = pipelines()
    results = {}
    for i in (int(v) for v in a.slices.split(",")):
        results[i] = run_slice(i, P, a.noise_scale, a.seed)
        print(f"[SCN{i}] done")
    M = [m for g in results.values() for m in g["metrics"]]
    table(M, P)
    with open(os.path.join(OUT, "validation_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(M[0])); w.writeheader(); w.writerows(M)
    for name, fig in figures(results, plt).items():
        out = os.path.join(OUT, name); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
        print(f"[saved] {out}.png/.pdf")
    print(f"[saved] {os.path.join(OUT, 'validation_metrics.csv')}, scn<i>_targets.npz")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
