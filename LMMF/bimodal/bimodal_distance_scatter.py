r"""
Bimodal distribution: LMMF spectral error vs distance to the true distribution, across fit restarts
====================================================================================================

Fits of the Fig. 1 frequency sample (N = 5000, seed 1) for M = 1, …, M_MAX (= 7; the selected fits
use M = 6–7). For every M, LM.fit_fixed_M runs N_RESTARTS random initialisations plus the two warm
starts grown from the best (M−1)-fit (as in the greedy algorithm), and EVERY local optimum is kept
(return_all) — the restarts populate a cloud of fits at different distances from ρ. Each optimum is
pruned to its effective order (LM._prune_mixture) and de-duplicated. Per fit:
  D_true    CvM distance to the TRUE ρ:  mean_i (F_fit(q_i) − u_i)², q_i quantile nodes of ρ (N_Q nodes)
  D_sample  CvM loss to the sample (the fitting objective, as LM.fit reports it)
  K_H       incoherence threshold of the fit at α (first unstable eigenvalue at z = 0)

Dynamics at α = ALPHA and K = KFACS · K_H(ρ) (sub-critical, critical, super-critical around the Hopf
bifurcation that creates the standing-wave limit cycle; K_H = 1.939 at α = 0, locking at K_SN = 4.35):
the N = 5000 network (Fig. 1 sample, Fig. 1 wrapped-Cauchy initial phases), the true-ρ mean field
(OA on N_CONT quantile nodes) and every fit, all from the same R(0). RK4 dt = 0.05, R sampled every
0.1, T = T_END. The figure script computes the spectral RMSE against the network (or the true-ρ mean
field) from the stored traces.

Stages (results in data_paths.MPMF/bimodal_kalpha/distance_scatter):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_distance_scatter.py fits [--procs P] [--restarts R]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_distance_scatter.py sims [--procs P]
                                                              [--alpha 0] [--kfacs 0.8,1,1.2]
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "unimodal"),
                os.path.join(_HERE, "..", "..", "shared"), os.path.join(_HERE, "..", "continuity")]
import time
import numpy as np
from multiprocessing import Pool
from scipy.stats import cramervonmises
import data_paths as dp
import lorentzian_mixture as LM
import unimodal_kalpha as UK
import kmo_bifurcation_accuracy as KBA
import bimodal_kalpha as BK

OUT = dp.ensure(os.path.join(BK.OUT, "distance_scatter"))
M_MAX, N_RESTARTS = 7, 40
FIT_KW = dict(delta_bounds=(1e-4, 1e2), method="slsqp", seed=1)
N_Q, N_CONT = 20000, 20000
ALPHA = 0.0
KFACS = (0.8, 1.0, 1.2)
T_END, DT, DTS = 200.0, 0.05, 0.1


def kh_true(alpha=None):
    """K_H(α) of the true ρ: smallest K on the exact boundary at this α (α = 0: the truth file)."""
    alpha = ALPHA if alpha is None else alpha
    if alpha == 0:
        return BK.truth()["K_H"]
    ex = BK.exact_boundary(); K, a = ex[:, 0], ex[:, 1] - alpha
    i = np.where(np.sign(a[:-1]) != np.sign(a[1:]))[0]
    return float(min(K[j] - a[j] * (K[j + 1] - K[j]) / (a[j + 1] - a[j]) for j in i))


def d_true(m, q, u):
    return float(np.mean((m.cdf(q) - u) ** 2))


def d_sample(m, xs):
    n = xs.size
    return (float(cramervonmises(xs, m.cdf).statistic) - 1.0 / (12.0 * n)) / n


# ─── fits ─────────────────────────────────────────────────────────────────────────────────────────
def _fit_M(args):
    """All local optima at order M (random restarts + warm starts from the best (M−1)-fit, which is
    recomputed here so that every M is an independent job)."""
    M, n_restarts = args
    out = os.path.join(OUT, f"fits_M{M}_r{n_restarts}.npy")
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    xs = np.sort(BK.network_sample())
    prev = None
    for m in range(1, M):                                # the greedy chain up to M − 1 (best fits only)
        prev, _ = LM.fit_fixed_M(xs, m, n_restarts=10, init_model=prev, **FIT_KW)
        prev = LM._prune_mixture(prev)
    best, _, allm = LM.fit_fixed_M(xs, M, n_restarts=n_restarts, init_model=prev, return_all=True, **FIT_KW)
    q = KBA.quantile_nodes(N_Q); u = (np.arange(N_Q) + 0.5) / N_Q
    rows = []
    for mod, obj, kind in allm:
        p = LM._prune_mixture(mod)
        f = dict(w=p.w, Om=p.Omega, De=p.Delta, M=p.M)
        rows.append(dict(f, M_nominal=M, kind=kind, obj=obj, D_sample=d_sample(p, xs), D_true=d_true(p, q, u),
                         K_H=UK.lmmf_Kc(f, ALPHA)[0], is_best=False))
    b = LM._prune_mixture(best)
    for r in rows:                                       # flag the optimum the algorithm would keep
        r["is_best"] = bool(r["M"] == b.M and np.allclose(np.sort(r["Om"]), np.sort(b.Omega), atol=1e-8))
    np.save(out, dict(rows=rows, M=M, n_restarts=n_restarts), allow_pickle=True)
    return out, time.time() - t0


def load_fits(n_restarts=N_RESTARTS, dedupe=True):
    rows = []
    for M in range(1, M_MAX + 1):
        rows += np.load(os.path.join(OUT, f"fits_M{M}_r{n_restarts}.npy"), allow_pickle=True).item()["rows"]
    if dedupe:                                           # identical optima (same order, same loss)
        seen, keep = set(), []
        for r in rows:
            key = (r["M"], round(r["D_sample"] * 1e10))
            if key not in seen:
                seen.add(key); keep.append(r)
        rows = keep
    return rows


def run_fits(procs, n_restarts):
    jobs = [(M, n_restarts) for M in range(M_MAX, 0, -1)]   # largest M first (slowest)
    t0 = time.time()
    with Pool(min(procs, len(jobs))) as p:
        for k, (out, dt) in enumerate(p.imap_unordered(_fit_M, jobs, chunksize=1), 1):
            print(f"[fits] {k}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; "
                  f"{(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
    rows = load_fits(n_restarts)
    print(f"[fits] {len(rows)} distinct optima; per M: "
          + ", ".join(f"{M}: {sum(r['M_nominal'] == M for r in rows)}" for M in range(1, M_MAX + 1)))


# ─── sims ─────────────────────────────────────────────────────────────────────────────────────────
def couplings():
    return kh_true() * np.array(KFACS)


def tag():
    return f"_a{ALPHA:g}_k" + "-".join(f"{f:g}" for f in KFACS)


def _simulate(f, Kc, R0):
    n_steps, stride = int(round(T_END / DT)), int(round(DTS / DT))
    z0 = np.full((Kc.size, f["M"]), R0 + 0j)
    return UK._rk4_carry(UK._lmmf_rhs(f, Kc[:, None]), z0, n_steps, stride, lambda y: np.abs(y @ f["w"]))


def _sim_chunk(args):
    rows, Kc, R0 = args
    return np.array([_simulate(r, Kc, R0) for r in rows], dtype=np.float32)


def run_sims(procs, n_restarts):
    UK.DT = DT
    t0 = time.time()
    om, th0 = KBA.fig1_sample()
    R0 = float(np.abs(np.exp(1j * th0).mean()))
    K = couplings(); Kc = K * np.exp(-1j * ALPHA)
    n_steps, stride = int(round(T_END / DT)), int(round(DTS / DT))
    R_net = UK._rk4_carry(UK._net_rhs(om, Kc[:, None]), np.broadcast_to(th0, (K.size, om.size)).copy(),
                          n_steps, stride, lambda y: np.abs(np.exp(1j * y).mean(1)))
    nodes = KBA.quantile_nodes(N_CONT)
    R_cont = _simulate(dict(Om=nodes, De=np.zeros(N_CONT), w=np.full(N_CONT, 1.0 / N_CONT), M=N_CONT), Kc, R0)
    print(f"[sims] network + true-ρ mean field ({time.time() - t0:.0f} s)", flush=True)
    rows = load_fits(n_restarts)
    chunks = [c for c in np.array_split(np.arange(len(rows)), procs) if c.size]
    with Pool(min(procs, len(chunks))) as p:
        parts = p.map(_sim_chunk, [([rows[i] for i in c], Kc, R0) for c in chunks])
    R_fit = np.concatenate(parts)                        # (n_fits, nK, ns)
    keys = ("M", "M_nominal", "D_true", "D_sample", "K_H", "obj")
    out = os.path.join(OUT, f"sims_r{n_restarts}{tag()}.npz")
    np.savez_compressed(out, K=K, K_H_true=kh_true(), alpha=ALPHA, kfacs=np.array(KFACS), R0=R0, dts=DTS,
                        R_net=R_net.astype(np.float32), R_cont=R_cont.astype(np.float32), R_fit=R_fit,
                        kind=np.array([r["kind"] for r in rows]), is_best=np.array([r["is_best"] for r in rows]),
                        **{k: np.array([r[k] for r in rows]) for k in keys})
    print(f"[sims] {len(rows)} fits × {K.size} K ({(time.time() - t0) / 60:.1f} min) -> {out}")


if __name__ == "__main__":
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    procs = arg("--procs", os.cpu_count(), int)
    n_restarts = arg("--restarts", N_RESTARTS, int)
    ALPHA = arg("--alpha", ALPHA, float)
    KFACS = arg("--kfacs", KFACS, lambda s_: tuple(float(v) for v in s_.split(",")))
    if "fits" in sys.argv:
        run_fits(procs, n_restarts)
    if "sims" in sys.argv:
        print(f"K_H(α={ALPHA:g}) = {kh_true():.4f}; K = {np.round(couplings(), 4)}")
        run_sims(procs, n_restarts)
