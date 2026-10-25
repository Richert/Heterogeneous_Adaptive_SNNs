r"""
Bimodal Gaussian mixture: LMMF fit ensembles over (N, M_max, λ) — fit loss, M*, Hopf point and ramp RMSE
=========================================================================================================

Distribution: the Fig. 1 Gaussian mixture (kmo_lorentzian_fit_sweep.CONFIG: means ±2, SD 0.6, equal weights).
For every sample size N ∈ N_LIST and N_SEEDS independent samples (seeds SEED0 …):

  trace   ONE greedy Lorentzian-mixture fit WITHOUT any acceptance rule (LM.fit floor_c = 0, λ = 0, patience =
          M_TRACE, M = 1 … M_TRACE; CvM loss, slsqp, warm start, widths in DELTA_BOUNDS). The fixed-M fits it visits
          are λ-independent, so every (M_max, λ) selection of the no-floor rule is an exact replay of this trace
          (select_nofloor: penalised argmin of D + λ m with patience PATIENCE over M_nominal ≤ M_max — LM.fit's
          floor_c = 0 logic; checked against LM.fit in `selfcheck`). The NOISE-FLOOR reference of the sample is
          the fit LM.fit returns with the floor cap (floor_c = 1: stop at the first D ≤ 1/(6N); λ = 0, M_max =
          M_TRACE) — select_floor. The fits shown and scored all use NO floor; the floor fit only marks
          where it would land in the (M_max, λ) plane.
  onset   for every trace model: the smallest K at which the incoherent state z = 0 loses stability (α = 0;
          eigenvalues of diag(iΩ − Δ) + K/2 · 1 wᵀ) and the frequency Ω_c of the critical mode RELATIVE to the
          weighted mean frequency. Type: |Ω_c| > HOPF_TOL -> Hopf bifurcation to standing waves (two counter-rotating
          clusters; bimodal); |Ω_c| ≈ 0 -> steady onset of ONE synchronised cluster (a pitchfork of the incoherent
          state in the frame rotating with the mean frequency; e.g. a single Lorentzian at K = 2Δ) — NOT a Hopf point
          (the imaginary part of the lab-frame eigenvalue is only the common rotation)
  ramp    backward quasi-static K-ramp, α = 0, K = K_GRID (0.1 … 3, 40 steps; as the unimodal PNAS figure), HOLD time
          units per step, state carried, start from a wrapped-Cauchy state with R0 = 0.9:
            network  the N-oscillator Kuramoto network with the sample frequencies (RK4, dt = DT)
            LMMF     all trace models at once (Ott–Antonsen ensembles, same start R0); finite-size seed N^{-1/2} added
                     at a K step only if the model's R < N^{-1/2} (as illustration_ramps.lmmf_ramp_nokick)
          spectral RMSE per K window = RMS difference of the |FFT| of R(t) (unimodal_kalpha_figure.spectral_rmse);
          the scalar RMSE of a model is the mean over the K windows.
Output: data_paths.MPMF/bimodal_fit_ensemble/ — run_N<N>_s<seed>.npz (sample, trace models, Hopf, ramps, RMSE),
summary.npz (per N, seed, M_max, λ: selected M, D, RMSE, onset K and type, K_Hopf (NaN where the onset is not a
Hopf point); floor selection) via `summary`.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_fit_ensemble.py selfcheck
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_fit_ensemble.py run [--procs 8] [--N 100,1000,10000]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_fit_ensemble.py summary
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import time
import numpy as np
from multiprocessing import Pool
import data_paths as dp
import lorentzian_mixture as LM
from kmo_lorentzian_fit_sweep import CONFIG as FIG1, sample_gaussian_mixture

OUT = dp.ensure(dp.mpmf("bimodal_fit_ensemble"))
N_LIST = (100, 1000, 10000)
N_SEEDS, SEED0 = 20, 300
M_TRACE = 16
M_MAX_GRID = (1, 2, 4, 8, 16)
LAMBDA_GRID = (1e-6, 1e-5, 1e-4, 1e-3)
HOPF_TOL = 0.1                                                # |Ω_c| above which the onset is a Hopf point
PATIENCE = FIG1["patience"]                                   # 3
DELTA_BOUNDS = FIG1["delta_bounds"]                           # (1e-4, 1e2)
FIT_KW = dict(loss=FIG1["loss"], n_restarts=FIG1["n_restarts"], method=FIG1["method"])
K_GRID = np.round(np.linspace(0.1, 3.0, 40), 4)
HOLD, DT, DTS, R0 = 100.0, 0.05, 0.1, 0.9


def sample(N, seed):
    return sample_gaussian_mixture(FIG1["gmm_means"], FIG1["gmm_stds"], FIG1["gmm_weights"], N,
                                   np.random.default_rng(seed))


# ─── selection replays ───────────────────────────────────────────────────────────────────────────
def select_nofloor(trace, M_max, lam, patience=PATIENCE):
    """LM.fit with floor_c = 0: penalised argmin of D + λ m with patience, M_nominal ≤ M_max. Returns trace index."""
    best, best_tot, stall = None, np.inf, 0
    for k, e in enumerate(trace):
        if e["M_nominal"] > M_max:
            break
        tot = e["D"] + lam * e["M"]
        if tot < best_tot:
            best, best_tot, stall = k, tot, 0
        else:
            stall += 1
            if stall >= patience:
                break
    return best


def select_floor(trace, N, lam=0.0, M_max=M_TRACE, patience=PATIENCE):
    """LM.fit with floor_c = 1: as select_nofloor, but stop at the first D ≤ 1/(6N) (keeping the argmin so far)."""
    floor = LM.noise_floor(N)
    best, best_tot, stall = None, np.inf, 0
    for k, e in enumerate(trace):
        if e["M_nominal"] > M_max:
            break
        tot = e["D"] + lam * e["M"]
        if tot < best_tot:
            best, best_tot, stall = k, tot, 0
        else:
            stall += 1
        if e["D"] <= floor or stall >= patience:
            break
    return best


# ─── Hopf point of the incoherent state ──────────────────────────────────────────────────────────
def onset(w, Om, De, K_hi=10.0):
    """Smallest K (α = 0) at which z = 0 is unstable, and the critical mode's frequency relative to Σ w Ω."""
    A0 = np.diag(1j * Om - De); P = 0.5 * np.outer(np.ones(w.size), w)
    lead = lambda K: np.linalg.eigvals(A0 + K * P)
    g = lambda K: lead(K).real.max()
    if g(K_hi) < 0:
        return np.nan, np.nan
    lo, hi = 0.0, K_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    ev = lead(hi)
    return hi, float(ev[np.argmax(ev.real)].imag - np.sum(w * Om))


# ─── ramps ───────────────────────────────────────────────────────────────────────────────────────
def _wrapped_cauchy(R, n, seed):
    return (-np.log(R)) * np.tan(np.pi * (np.random.default_rng(seed).random(n) - 0.5))


def ramps(om, models, N):
    """Backward K-ramp of the network (frequencies om) and of all LMMF models at once.
    Returns R_net (nK, ns), R_lmmf (n_models, nK, ns); K ascending (index ik ↔ K_GRID[ik])."""
    n_steps, stride = int(round(HOLD / DT)), int(round(DTS / DT)); ns = n_steps // stride + 1
    nK = K_GRID.size; kick = 1.0 / np.sqrt(N)
    th = _wrapped_cauchy(R0, om.size, 100); R0_obs = float(np.abs(np.exp(1j * th).mean()))
    # LMMF: concatenated components, per-model segment sums
    Ms = np.array([m["M"] for m in models]); off = np.r_[0, np.cumsum(Ms)[:-1]]
    w = np.concatenate([m["w"] for m in models]); a = np.concatenate([1j * m["Om"] - m["De"] for m in models])
    seg = np.repeat(np.arange(len(models)), Ms)
    z = np.full(w.size, R0_obs + 0j)
    Zm = lambda z_: np.add.reduceat(w * z_, off)
    R_net = np.zeros((nK, ns)); R_l = np.zeros((len(models), nK, ns))

    def f_net(th_, K):
        Z = np.exp(1j * th_).mean()
        return om + K * np.imag(Z * np.exp(-1j * th_))

    def f_l(z_, K):
        H = K * Zm(z_)[seg]
        return a * z_ + 0.5 * (H - np.conj(H) * z_ * z_)

    for ik in range(nK - 1, -1, -1):
        K = K_GRID[ik]
        low = np.abs(Zm(z)) < kick                                   # seed only near incoherence
        if low.any():
            z = z + kick * low[seg]
            r = np.abs(z); z = np.where(r > 1, z / np.maximum(r, 1e-300), z)
        R_net[ik, 0] = abs(np.exp(1j * th).mean()); R_l[:, ik, 0] = np.abs(Zm(z))
        for s in range(1, n_steps + 1):
            k1 = f_net(th, K); k2 = f_net(th + 0.5 * DT * k1, K); k3 = f_net(th + 0.5 * DT * k2, K)
            k4 = f_net(th + DT * k3, K); th = th + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            q1 = f_l(z, K); q2 = f_l(z + 0.5 * DT * q1, K); q3 = f_l(z + 0.5 * DT * q2, K)
            q4 = f_l(z + DT * q3, K); z = z + DT / 6 * (q1 + 2 * q2 + 2 * q3 + q4)
            if s % stride == 0:
                R_net[ik, s // stride] = abs(np.exp(1j * th).mean()); R_l[:, ik, s // stride] = np.abs(Zm(z))
    return R_net, R_l


def spectral_rmse(a, b):
    n = a.shape[-1]
    return np.sqrt(np.mean((np.abs(np.fft.rfft(a, axis=-1)) - np.abs(np.fft.rfft(b, axis=-1))) ** 2, -1)) / n


# ─── one (N, seed) ───────────────────────────────────────────────────────────────────────────────
def run_file(N, seed):
    return os.path.join(OUT, f"run_N{N}_s{seed}.npz")


def job(args):
    N, seed = args
    out = run_file(N, seed)
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    om = sample(N, seed)
    r = LM.fit(om, DELTA_BOUNDS, M_max=M_TRACE, lambda_M=0.0, patience=M_TRACE, floor_c=0.0, seed=seed, **FIT_KW)
    models = [dict(M=int(e["model"].M), w=np.asarray(e["model"].w, float), Om=np.asarray(e["model"].Omega, float),
                   De=np.asarray(e["model"].Delta, float)) for e in r["trace"]]
    Mn = np.array([e["M_nominal"] for e in r["trace"]]); Me = np.array([m["M"] for m in models])
    D = np.array([e["data_loss"] for e in r["trace"]])
    ons = np.array([onset(m["w"], m["Om"], m["De"]) for m in models])
    t_fit = time.time() - t0
    R_net, R_l = ramps(om, models, N)
    rmse_K = spectral_rmse(R_net[None], R_l)                          # (n_models, nK)
    mx = max(Me)
    pad = lambda key: np.array([np.pad(m[key], (0, mx - m["M"]), constant_values=np.nan) for m in models])
    np.savez_compressed(out, N=N, seed=seed, omega=om.astype(np.float32), M_nominal=Mn, M=Me, D=D,
                        w=pad("w"), Om=pad("Om"), De=pad("De"), K_onset=ons[:, 0], Om_onset=ons[:, 1],
                        is_hopf=np.abs(ons[:, 1]) > HOPF_TOL,
                        K=K_GRID, R_net=R_net.astype(np.float32), R_lmmf=R_l.astype(np.float32), rmse_K=rmse_K,
                        rmse=rmse_K.mean(1), hold=HOLD, dts=DTS, t_fit=t_fit, t_total=time.time() - t0)
    return out, time.time() - t0


def load_run(N, seed):
    d = np.load(run_file(N, seed))
    trace = [dict(M_nominal=int(mn), M=int(m), D=float(D)) for mn, m, D in zip(d["M_nominal"], d["M"], d["D"])]
    return d, trace


def model(d, k):
    m = int(d["M"][k])
    return dict(M=m, w=d["w"][k, :m], Om=d["Om"][k, :m], De=d["De"][k, :m])


# ─── summary over (N, seed, M_max, λ) ────────────────────────────────────────────────────────────
def summary():
    seeds = list(range(SEED0, SEED0 + N_SEEDS))
    shape = (len(N_LIST), N_SEEDS, len(M_MAX_GRID), len(LAMBDA_GRID))
    sel, M, D, RM, KO, OO = (np.full(shape, np.nan) for _ in range(6))
    floor_k = np.full(shape[:2], -1); floor_M = np.full(shape[:2], np.nan)
    for a, N in enumerate(N_LIST):
        for b, s in enumerate(seeds):
            if not os.path.exists(run_file(N, s)):
                continue
            d, trace = load_run(N, s)
            fk = select_floor(trace, N); floor_k[a, b] = fk; floor_M[a, b] = trace[fk]["M"]
            for i, Mm in enumerate(M_MAX_GRID):
                for j, lam in enumerate(LAMBDA_GRID):
                    k = select_nofloor(trace, Mm, lam)
                    sel[a, b, i, j] = k; M[a, b, i, j] = trace[k]["M"]; D[a, b, i, j] = trace[k]["D"]
                    RM[a, b, i, j] = d["rmse"][k]; KO[a, b, i, j] = d["K_onset"][k]
                    OO[a, b, i, j] = d["Om_onset"][k]
    same_floor = sel == floor_k[:, :, None, None]
    np.savez(os.path.join(OUT, "summary.npz"), N=np.array(N_LIST), seeds=np.array(seeds), M_max=np.array(M_MAX_GRID),
             lam=np.array(LAMBDA_GRID), sel=sel, M=M, D=D, rmse=RM, K_onset=KO, Om_onset=OO,
             K_hopf=np.where(np.abs(OO) > HOPF_TOL, KO, np.nan), floor_k=floor_k, floor_M=floor_M,
             same_floor=same_floor, noise_floor=np.array([LM.noise_floor(N) for N in N_LIST]))
    for a, N in enumerate(N_LIST):
        print(f"N = {N}: floor-rule M* = {np.nanmean(floor_M[a]):.1f} "
              f"({np.nanmin(floor_M[a]):.0f}–{np.nanmax(floor_M[a]):.0f}); mean RMSE per (M_max × λ):")
        print(np.round(np.nanmean(RM[a], 0), 4))


def selfcheck(N=1000, seed=7):
    """select_nofloor / select_floor must reproduce LM.fit."""
    om = sample(N, seed)
    r = LM.fit(om, DELTA_BOUNDS, M_max=M_TRACE, lambda_M=0.0, patience=M_TRACE, floor_c=0.0, seed=seed, **FIT_KW)
    trace = [dict(M_nominal=e["M_nominal"], M=e["M"], D=e["data_loss"]) for e in r["trace"]]
    for Mm in M_MAX_GRID:
        for lam in LAMBDA_GRID:
            q = LM.fit(om, DELTA_BOUNDS, M_max=Mm, lambda_M=lam, patience=PATIENCE, floor_c=0.0, seed=seed, **FIT_KW)
            assert trace[select_nofloor(trace, Mm, lam)]["M"] == q["M"], ("nofloor", Mm, lam)
    q = LM.fit(om, DELTA_BOUNDS, M_max=M_TRACE, lambda_M=0.0, patience=PATIENCE, floor_c=1.0, seed=seed, **FIT_KW)
    assert trace[select_floor(trace, N)]["M"] == q["M"], "floor"
    print(f"[selfcheck] replays == LM.fit (N = {N}, seed = {seed})", flush=True)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    if "selfcheck" in sys.argv:
        selfcheck()
    if "run" in sys.argv:
        Ns = arg("--N", list(N_LIST), lambda s_: [int(x) for x in s_.split(",")])
        jobs = sorted([(N, s) for N in Ns for s in range(SEED0, SEED0 + N_SEEDS)], key=lambda j: -j[0])
        t0 = time.time()
        with Pool(arg("--procs", os.cpu_count(), int)) as p:
            for k, (out, dt) in enumerate(p.imap_unordered(job, jobs, chunksize=1), 1):
                el = time.time() - t0
                print(f"[run] {k}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; {el / 60:.1f} min elapsed)",
                      flush=True)
    if "summary" in sys.argv:
        summary()
