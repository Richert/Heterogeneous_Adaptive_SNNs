r"""
Which sample size (and seed) illustrates sample-specific dynamics that the LMMF can capture?
=============================================================================================

For a distribution ρ, sample size N and seed, at a few (K, α) points near the transitions (all from the
same partially synchronous start, R0 = 0.5; RK4 dt = 0.05, T = 200, R sampled every 0.1):
  net    the Kuramoto–Sakaguchi network on the sample (wrapped-Cauchy phases, independent of ω)
  smf    the SAMPLE mean field: OA on the N sample frequencies as zero-width nodes (the sample's
         frequency structure without finite-size phase noise)
  tmf    the TRUE-ρ mean field (OA on N_Q quantile nodes of ρ)
  floor / nofloor   the LMMF fits of the sample with / without the noise floor (figure settings)
Per point (spectral RMSE of R(t) on [0, T], unimodal_kalpha_figure.spectral_rmse):
  capturable deviation   dev   = RMSE(smf, tmf)       what a sample fit could gain over ρ
  finite-size noise      noise = RMSE(net, smf)       what no mean field reproduces
  captured share         1 − RMSE(LMMF, smf) / dev    per fit
  and RMSE(LMMF, net), RMSE(tmf, net) for reference.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python sample_size_scan.py run [--N 500,1000,2000,5000]
                                                       [--seeds 0-9] [--dist gauss,bimodal] [--procs P]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python sample_size_scan.py summary [same filters]
Results: data_paths.MPMF/sample_size_scan/
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "unimodal"), os.path.join(_HERE, "bimodal"),
                os.path.join(_HERE, "..", "shared"), os.path.join(_HERE, "continuity")]
import time
import numpy as np
from multiprocessing import Pool
import data_paths as dp
import lorentzian_mixture as LM
import unimodal_kalpha as UK
import unimodal_kalpha_figure as UF
import fit_scenarios as FS
import kmo_bifurcation_accuracy as KBA

OUT = dp.ensure(dp.mpmf("sample_size_scan"))
R0, T_END, DT, DTS = 0.5, 200.0, 0.05, 0.1
N_Q = 5000
POINTS = {                                   # (K, α) near the transitions
    "gauss": [(1.6, 0.7), (1.8, 0.7), (2.0, 0.7), (2.2, 0.7), (1.6, 0.0), (2.0, 0.0)],
    "bimodal": [(1.8, 0.8), (2.0, 0.8), (2.2, 0.8), (2.0, 0.0), (3.0, 0.0)],
    "g8": [(0.9, 0.7), (1.0, 0.7), (1.1, 0.7), (1.2, 0.7), (1.2, 0.0), (1.4, 0.0)],        # K_c 1.03 / 1.28
    "dgauss": [(0.35, 0.7), (0.42, 0.7), (0.5, 0.7), (0.6, 0.7), (0.3, 0.0), (0.4, 0.0)],  # K_c 0.43 / 0.35
}
FIT = {                                      # --lam L: λ = L, M_max = 64, width floor k = 10 for both
    "gauss": dict(lambda_M=1e-7, M_max=64, width_k=10),   # (defaults: the settings of the stage-1 scan)
    "bimodal": dict(lambda_M=1e-5, M_max=16, width_k=0),
    "g8": dict(lambda_M=1e-7, M_max=64, width_k=10),
    "dgauss": dict(lambda_M=1e-7, M_max=64, width_k=10),
}
LAM = None                                   # None: FIT as above; else the uniform setting (files in lam<L>/)


def sample(dist, N, seed):
    rng = np.random.default_rng([2027, seed, N])
    if dist == "gauss":
        return rng.normal(0.0, 1.0, N)
    if dist in ("g8", "dgauss"):
        return np.asarray(UK.dists()[dist]["sample"](N, rng), float)
    return KBA.sample_gaussian_mixture(KBA.FIG1["gmm_means"], KBA.FIG1["gmm_stds"], KBA.FIG1["gmm_weights"], N, rng)


def quantiles(dist, n):
    return KBA.quantile_nodes(n) if dist == "bimodal" else FS.quantile_nodes(dist, n)


def _kc(dist):
    return np.array([K * np.exp(-1j * a) for K, a in POINTS[dist]])


def simulate_mf(Om, De, w, Kc):
    """OA ensembles (Lorentzian or zero-width nodes) for all points at once -> R (P, ns)."""
    UK.DT = DT
    f = dict(Om=np.asarray(Om), De=np.asarray(De), w=np.asarray(w), M=len(w))
    n, stride = int(round(T_END / DT)), int(round(DTS / DT))
    z0 = np.full((Kc.size, f["M"]), R0 + 0j)
    return UK._rk4_carry(UK._lmmf_rhs(f, Kc[:, None]), z0, n, stride, lambda y: np.abs(y @ f["w"]))


def simulate_net(om, Kc):
    UK.DT = DT
    n, stride = int(round(T_END / DT)), int(round(DTS / DT))
    th0 = (-np.log(R0)) * np.tan(np.pi * (np.random.default_rng(100).random(om.size) - 0.5))
    th = np.broadcast_to(th0, (Kc.size, om.size)).copy()
    return UK._rk4_carry(UK._net_rhs(om, Kc[:, None]), th, n, stride, lambda y: np.abs(np.exp(1j * y).mean(1)))


def tmf_file(dist):
    return os.path.join(OUT, f"{dist}_tmf.npy")


def ensure_tmf(dist):
    if not os.path.exists(tmf_file(dist)):
        q = quantiles(dist, N_Q)
        np.save(tmf_file(dist), simulate_mf(q, np.zeros(N_Q), np.full(N_Q, 1.0 / N_Q), _kc(dist)).astype(np.float32))


def job_file(dist, N, seed):
    d = OUT if LAM is None else dp.ensure(os.path.join(OUT, f"lam{LAM:g}"))
    return os.path.join(d, f"{dist}_N{N}_s{seed:02d}.npz")


def _fit(x, dist, floor_c):
    c = FIT[dist]
    dmin = FS.width_floor(x, c["width_k"]) if c["width_k"] else 1e-4
    r = LM.fit(x, (dmin, 1e2), lambda_M=c["lambda_M"], M_max=c["M_max"], floor_c=floor_c, patience=3,
               n_restarts=10, seed=1, method="slsqp", large_M=16)
    m = r["model"]
    return m, r["data_loss"]


def _job(args):
    dist, N, seed = args
    out = job_file(dist, N, seed)
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    x = sample(dist, N, seed); Kc = _kc(dist)
    res = dict(R_net=simulate_net(x, Kc), R_smf=simulate_mf(x, np.zeros(N), np.full(N, 1.0 / N), Kc))
    q = quantiles(dist, N_Q); u = (np.arange(N_Q) + 0.5) / N_Q
    for kind, fc in (("floor", 1.0), ("nofloor", 0.0)):
        m, D = _fit(x, dist, fc)
        res[f"R_{kind}"] = simulate_mf(m.Omega, m.Delta, m.w, Kc)
        res[f"M_{kind}"] = m.M; res[f"D_{kind}"] = D
        res[f"Dtrue_{kind}"] = float(np.mean((m.cdf(q) - u) ** 2))
        res[f"fit_{kind}"] = np.concatenate([m.w, m.Omega, m.Delta])
    np.savez_compressed(out, **{k: (v.astype(np.float32) if isinstance(v, np.ndarray) and k.startswith("R_") else v)
                                for k, v in res.items()}, x=x, points=np.array(POINTS[dist]))
    return out, time.time() - t0


def run(dists, Ns, seeds, procs):
    for d in dists:
        ensure_tmf(d)
    cost = lambda j: j[1] * (1 if j[0] == "bimodal" else 3)          # gauss fits (λ = 1e-7) are slowest
    jobs = sorted([(d, N, s) for d in dists for N in Ns for s in seeds], key=cost, reverse=True)
    t0 = time.time()
    with Pool(min(procs, len(jobs))) as p:
        for k, (out, dt) in enumerate(p.imap_unordered(_job, jobs, chunksize=1), 1):
            el = time.time() - t0
            print(f"[scan] {k}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; {el / 60:.1f} min elapsed, "
                  f"~{el / k * (len(jobs) - k) / 60:.1f} min left)", flush=True)


def metrics(dist, N, seed):
    d = np.load(job_file(dist, N, seed)); tmf = np.load(tmf_file(dist)).astype(float)
    E = lambda a, b: UF.spectral_rmse(np.asarray(a, float), np.asarray(b, float))   # per point
    smf = d["R_smf"]
    m = dict(dev=E(smf, tmf), noise=E(d["R_net"], smf), tmf_net=E(tmf, d["R_net"]))
    for kind in ("floor", "nofloor"):
        m[f"err_{kind}"] = E(d[f"R_{kind}"], smf)
        m[f"net_{kind}"] = E(d[f"R_{kind}"], d["R_net"])
        m[f"M_{kind}"] = int(d[f"M_{kind}"]); m[f"Dtrue_{kind}"] = float(d[f"Dtrue_{kind}"])
    return m


def summary(dists, Ns, seeds):
    for dist in dists:
        print(f"\n=== {dist}: points (K, α) = {POINTS[dist]}; medians over seeds of the point-averaged values ===")
        print(f"{'N':>5} {'n':>3} {'dev':>8} {'noise':>8} {'dev/noise':>9} | {'captured share':>16} | "
              f"{'RMSE vs network':>32} | {'M floor/nofloor':>15}")
        print(f"{'':>36} | {'floor':>7} {'nofloor':>8} | {'tmf':>8} {'floor':>8} {'nofloor':>8}   |")
        for N in Ns:
            ms = [metrics(dist, N, s) for s in seeds if os.path.exists(job_file(dist, N, s))]
            if not ms:
                continue
            med = lambda f: float(np.median([f(m) for m in ms]))
            dev = med(lambda m: m["dev"].mean()); noise = med(lambda m: m["noise"].mean())
            cap = {k: med(lambda m: 1 - m[f"err_{k}"].mean() / m["dev"].mean()) for k in ("floor", "nofloor")}
            print(f"{N:5d} {len(ms):3d} {dev:8.2e} {noise:8.2e} {dev / noise:9.2f} | {cap['floor']:7.2f} "
                  f"{cap['nofloor']:8.2f} | {med(lambda m: m['tmf_net'].mean()):8.2e} "
                  f"{med(lambda m: m['net_floor'].mean()):8.2e} {med(lambda m: m['net_nofloor'].mean()):8.2e}   | "
                  f"{med(lambda m: m['M_floor']):5.0f} / {med(lambda m: m['M_nofloor']):3.0f}")


def rank(dists, N, seeds):
    """Per-seed table, sorted by the captured deviation dev × captured share (floorless)."""
    for dist in dists:
        rows = []
        for s in seeds:
            if not os.path.exists(job_file(dist, N, s)):
                continue
            m = metrics(dist, N, s); dev = m["dev"].mean()
            cf, cn = 1 - m["err_floor"].mean() / dev, 1 - m["err_nofloor"].mean() / dev
            rows.append((dev * cn, s, dev, m["noise"].mean(), cf, cn, m["tmf_net"].mean(), m["net_nofloor"].mean(),
                         m["M_floor"], m["M_nofloor"], m["Dtrue_nofloor"]))
        rows.sort(reverse=True)
        print(f"\n=== {dist}, N = {N}: seeds ranked by dev × captured share (floorless) ===")
        print(f"{'seed':>4} {'dev':>8} {'noise':>8} {'cap floor':>9} {'cap nofl':>8} {'tmf-net':>8} {'nofl-net':>8} "
              f"{'M fl/nofl':>9} {'D_true nofl':>11}")
        for r in rows:
            print(f"{r[1]:4d} {r[2]:8.2e} {r[3]:8.2e} {r[4]:9.2f} {r[5]:8.2f} {r[6]:8.2e} {r[7]:8.2e} {r[8]:4d}/{r[9]:<4d} "
                  f"{r[10]:11.2e}")
        caps = np.array([r[5] for r in rows]); devs = np.array([r[2] for r in rows])
        print(f"  median over {len(rows)} seeds: dev {np.median(devs):.2e}, captured floorless {np.median(caps):.2f}")


def _range(s_):
    a, b = (s_.split("-") + [None])[:2]
    return list(range(int(a), int(b) + 1)) if b else [int(v) for v in s_.split(",")]


if __name__ == "__main__":
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    dists = arg("--dist", ["gauss", "bimodal"], lambda s_: s_.split(","))
    Ns = arg("--N", [500, 1000, 2000, 5000], lambda s_: [int(v) for v in s_.split(",")])
    seeds = arg("--seeds", list(range(10)), _range)
    LAM = arg("--lam", None, float)
    if LAM is not None:
        FIT = {d: dict(lambda_M=LAM, M_max=64, width_k=10) for d in FIT}
    if "run" in sys.argv:
        run(dists, Ns, seeds, arg("--procs", os.cpu_count(), int))
    if "summary" in sys.argv or "run" in sys.argv:
        summary(dists, Ns, seeds)
    if "rank" in sys.argv:
        rank(dists, Ns[0], seeds)
