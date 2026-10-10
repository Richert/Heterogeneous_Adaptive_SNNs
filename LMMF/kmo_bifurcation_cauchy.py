r"""
Does a Cauchy-transform loss term fix the LMMF bifurcation errors? + RMSE vs. distance to bifurcations
======================================================================================================

Fits: the three Fig. 1 fits of the last column of Fig. 1(a) (M_max = 16; λ = 1e-5, 1e-4, 1e-3 →
M = 6, 4, 2), read from the Fig. 1 sweep CSV, and for each the SAME M refitted with an augmented loss

    J = D_CvM / D_CvM^0  +  β · L_ε / L_ε^0,
    L_ε(ρ_N, ρ_M) = ∫ |Φ_N(ε+iν) − Φ_M(ε+iν)|² dν = 2π ∫_0^∞ |φ_N(s) − φ_M(s)|² e^{−2εs} ds,

where Φ is the Cauchy transform (closed form, see continuity/spectral_loss_prototype.py) and
D_CvM^0, L_ε^0 are the values at the CvM fit (so β = 1 weights both terms equally). The incoherence
Hopf point is set by Φ near the imaginary axis (1 = (K/2) Φ(iν)), which L_ε controls at resolution ε.
Default declared before looking at results: ε = 0.1, β = 1; a small (ε, β) grid is reported as a
sensitivity check.

Bifurcation points: K_H from the eigenvalues of the linearisation at z = 0 (identical to the PyCoBi
Hopf points, kmo_bifurcation_accuracy.py), K_SN from the PyCoBi continuation of the locked state.

Dynamics: K sampled at distances δ ∈ DELTAS on both sides of the TRUE K_H and K_SN; for every K,
N_TRIALS network realisations (Fig. 1 frequencies, independent wrapped-Cauchy initial phases), the
all LMMF fits from the trial's R(0), and the true-ρ continuum once per K (trial-averaged R(0));
spectral RMSE (Fig. 1 metric) on [0, T]. Cost: ~10 s per network realisation, ~60 s per continuum
run -> ~1.2 CPU-hours in total; parallelised over (K, trial) jobs.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_bifurcation_cauchy.py fits
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_bifurcation_cauchy.py bif     # sequential Auto
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_bifurcation_cauchy.py sims [--procs N]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_bifurcation_cauchy_figure.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared"), os.path.join(_HERE, "continuity")]
import subprocess
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import lorentzian_mixture as LM
import kmo_bifurcation_accuracy as BA
from spectral_loss_prototype import spectral_loss, spectral_const, spectral_loss_grad

OUT = BA.OUT
FITS_NPZ = os.path.join(OUT, "cauchy_fits.npz")
SIMS_NPZ = os.path.join(OUT, "cauchy_sims.npz")
FIG1_LAMBDAS = {1e-5: 6, 1e-4: 4, 1e-3: 2}
EPS_DEFAULT, BETA_DEFAULT = 0.1, 1.0
GRID = [(0.1, 1.0), (0.1, 10.0), (0.3, 1.0), (0.3, 10.0)]     # (ε, β) sensitivity grid
DELTAS = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
N_TRIALS, T, DTS = 10, 100.0, 0.1


# ════════════════════════════════════════════════════════════════════════════
#  fits
# ════════════════════════════════════════════════════════════════════════════
def fig1_fits():
    df = pd.read_csv(BA.dp.mpmf("kmo_lorentzian_sweep.csv"))
    mix = df[(df.quantity == "mixture") & (df.M_max == 16)]
    out = []
    for lam, M in FIG1_LAMBDAS.items():
        g = mix[np.isclose(mix["lambda"], lam)].sort_values("idx")
        assert len(g) == M, (lam, len(g))
        out.append(dict(name=f"CvM M={M}", kind="cvm", M=M, eps=np.nan, beta=np.nan,
                        w=g.w.to_numpy(), Om=g.Omega.to_numpy(), De=g.Delta.to_numpy()))
    return out


def fit_cauchy(omega, f0, eps, beta, n_jitter=4, seed=0, dbounds=(1e-4, 1e2)):
    """Fixed-M fit of J = D/D0 + β L_ε/L0, started from the CvM fit f0 (+ jittered restarts)."""
    xs = np.sort(omega); n = xs.size; u = (np.arange(n) + 0.5) / n; M = f0["M"]
    const = spectral_const(omega, eps)                     # O(N^2) sample term: once, not per call
    D = lambda p: LM._cvm_obj_natural(p, M, xs, n, u)[0]
    L = lambda p: spectral_loss(p[:M], p[M:2 * M], p[2 * M:], omega, eps, const=const)
    p0 = np.concatenate([f0["w"], f0["Om"], f0["De"]])
    D0, L0 = D(p0), L(p0)

    def J(p):                                              # value + analytic gradient
        d, gd = LM._cvm_obj_natural(p, M, xs, n, u)
        l = spectral_loss(p[:M], p[M:2 * M], p[2 * M:], omega, eps, const=const)
        gl = spectral_loss_grad(p[:M], p[M:2 * M], p[2 * M:], omega, eps)
        return d / D0 + beta * l / L0, gd / D0 + beta * gl / L0
    rng = np.random.default_rng(seed)
    starts = [p0] + [np.concatenate([f0["w"], f0["Om"] + rng.normal(0, 0.05, M),
                                     f0["De"] * np.exp(rng.normal(0, 0.2, M))]) for _ in range(n_jitter)]
    bounds = [(0, 1)] * M + [(None, None)] * M + [dbounds] * M
    eq = dict(type="eq", fun=lambda p: p[:M].sum() - 1)
    best = None
    for s in starts:
        r = minimize(J, s, method="SLSQP", jac=True, bounds=bounds, constraints=[eq],
                     options=dict(maxiter=1000, ftol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    w = np.clip(p[:M], 0, None); w /= w.sum()
    return dict(w=w, Om=p[M:2 * M], De=p[2 * M:], D=D(p), L=L(p), D0=D0, L0=L0)


def make_fits():
    omega, _ = BA.fig1_sample()
    fits = fig1_fits()
    out = list(fits)
    for f0 in fits:
        for eps, beta in GRID:
            r = fit_cauchy(omega, f0, eps, beta)
            name = f"CvM+Cauchy M={f0['M']} (ε={eps:g}, β={beta:g})"
            out.append(dict(name=name, kind="cauchy", M=f0["M"], eps=eps, beta=beta,
                            w=r["w"], Om=r["Om"], De=r["De"]))
            print(f"[fits] {name}: D/D0={r['D'] / r['D0']:.3f}  L/L0={r['L'] / r['L0']:.3f}", flush=True)
    np.savez(FITS_NPZ, name=np.array([f["name"] for f in out]), kind=np.array([f["kind"] for f in out]),
             M=np.array([f["M"] for f in out]), eps=np.array([f["eps"] for f in out]),
             beta=np.array([f["beta"] for f in out]),
             w=np.concatenate([f["w"] for f in out]), Om=np.concatenate([f["Om"] for f in out]),
             De=np.concatenate([f["De"] for f in out]))


def load_fits():
    d = np.load(FITS_NPZ)
    off = np.concatenate([[0], np.cumsum(d["M"])])
    return [dict(idx=i, name=str(d["name"][i]), kind=str(d["kind"][i]), M=int(d["M"][i]),
                 eps=float(d["eps"][i]), beta=float(d["beta"][i]),
                 w=d["w"][off[i]:off[i + 1]], Om=d["Om"][off[i]:off[i + 1]], De=d["De"][off[i]:off[i + 1]])
            for i in range(d["M"].size)]


# ════════════════════════════════════════════════════════════════════════════
#  bifurcation points
# ════════════════════════════════════════════════════════════════════════════
def run_bif_one(idx):
    f = load_fits()[idx]
    os.chdir(BA.dp.ensure(os.path.join(OUT, "auto_work")))
    res = BA.auto_locked(f)
    res["K_H"] = BA.eig_KH(f)
    np.save(os.path.join(OUT, f"cauchy_bif_{idx:02d}.npy"), res, allow_pickle=True)
    print(f"[bif] {f['name']}: K_H={res['K_H']:.4f}  K_SN={res['K_SN']:.4f}", flush=True)


def run_bif():
    for f in load_fits():                                   # sequential: one Auto session at a time
        out = os.path.join(OUT, f"cauchy_bif_{f['idx']:02d}.npy")
        if os.path.exists(out):
            continue
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "bif_one", str(f["idx"])],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("[bif]")]
        print(line[-1] if line else f"[bif] {f['name']}: FAILED\n{r.stderr[-800:]}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  dynamics near the bifurcations
# ════════════════════════════════════════════════════════════════════════════
def k_points():
    tr = np.load(BA.TRUTH_NPZ)
    rows = []
    for bif, Kb in (("hopf", float(tr["K_H"])), ("fold", float(tr["K_SN"]))):
        for side, sgn in (("left", -1), ("right", 1)):
            for d in DELTAS:
                rows.append((bif, side, float(d), Kb + sgn * d))
    return rows


GAMMA0 = 0.5 * BA.FIG1["sigma0"] ** 2


def _theta0(n, trial):
    rng = np.random.default_rng(10_000 + trial)
    return GAMMA0 * np.tan(np.pi * (rng.random(n) - 0.5))


def _job_trial(args):
    """One network realisation at one K, plus every LMMF fit from that realisation's R(0)."""
    k, K, trial, fits_sel = args
    om, _ = BA.fig1_sample()
    th0 = _theta0(om.size, trial)
    R0 = float(np.abs(np.exp(1j * th0).mean()))
    R_net = BA._network(om, th0, K, T, DTS)
    R_fit = np.array([BA._riccati(f["Om"] + 1j * f["De"], f["w"], R0, K, T, DTS) for f in fits_sel])
    return k, trial, R_net, R_fit


def _job_cont(args):
    """True-ρ continuum (20k quantile nodes) ONCE per K, from the trial-averaged R(0): the
    realisations differ in R(0) by ~0.01, which only affects the first few time units."""
    k, K, R0 = args
    nodes = BA.quantile_nodes(20000)
    return k, BA._riccati(nodes, np.full(nodes.size, 1 / nodes.size), R0, K, T, DTS)


def run_sims(n_proc=None):
    from multiprocessing import Pool
    fits = load_fits()
    sel = [f for f in fits if f["kind"] == "cvm" or (f["eps"] == EPS_DEFAULT and f["beta"] == BETA_DEFAULT)]
    pts = k_points()
    om, _ = BA.fig1_sample()
    R0_mean = float(np.mean([np.abs(np.exp(1j * _theta0(om.size, tr)).mean()) for tr in range(N_TRIALS)]))
    nt = int(round(T / DTS)) + 1
    R_net = np.zeros((len(pts), N_TRIALS, nt)); R_fit = np.zeros((len(pts), N_TRIALS, len(sel), nt))
    R_cont = np.zeros((len(pts), nt))
    jobs = [("trial", (k, K, tr, sel)) for k, (*_, K) in enumerate(pts) for tr in range(N_TRIALS)]
    jobs += [("cont", (k, K, R0_mean)) for k, (*_, K) in enumerate(pts)]
    jobs.sort(key=lambda j: j[0] != "cont")               # long continuum jobs first (load balance)
    n_proc = n_proc or os.cpu_count()
    t0 = time.time()
    with Pool(n_proc) as p:
        it = p.imap_unordered(_dispatch, jobs, chunksize=1)
        for i, res in enumerate(it, 1):
            if res[0] == "cont":
                R_cont[res[1]] = res[2]
            else:
                _, k, tr, Rn, Rf = res
                R_net[k, tr] = Rn; R_fit[k, tr] = Rf
            if i % 20 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"[sims] {i}/{len(jobs)} jobs  {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
    np.savez(SIMS_NPZ, bif=np.array([b for b, *_ in pts]), side=np.array([s_ for _, s_, *_ in pts]),
             delta=np.array([d for *_, d, _ in pts]), K=np.array([K for *_, K in pts]),
             R_net=R_net, R_cont=R_cont, R_fit=R_fit, fit_idx=np.array([f["idx"] for f in sel]),
             R0_cont=R0_mean)
    print(f"[sims] {len(pts)} K values x {N_TRIALS} trials in {(time.time() - t0) / 60:.1f} min -> {SIMS_NPZ}")


def _dispatch(job):
    kind, args = job
    if kind == "cont":
        return ("cont",) + _job_cont(args)
    return ("trial",) + _job_trial(args)


if __name__ == "__main__":
    # usage: ... fits | bif | sims [--procs N] | bif_one <idx>
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mode = sys.argv[1]
    if mode == "sims":
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else None
        run_sims(procs)
    elif mode == "bif_one":
        run_bif_one(int(sys.argv[2]))
    else:
        {"fits": make_fits, "bif": run_bif}[mode]()
