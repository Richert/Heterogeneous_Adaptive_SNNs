r"""
Illustration setup shared by the unimodal and bimodal figures (revision, 2026-10)
=================================================================================

One network sample per distribution, N = 1000, chosen by the seed search of sample_size_scan.py
(typical sample deviation from ρ, largest contrast between the floorless and the noise-floor fit):
    gauss 17,  g8 24,  dgauss 4,  bimodal 18      (sampled exactly as in sample_size_scan.sample)
Fits (λ = 1e-6, warm start, M_max = 64, slsqp, 10 restarts, patience 3):
    pop      the TRUE ρ (N_Q quantile nodes), no floor, widths ≥ 1e-4
    floor    the sample with the noise-floor cap, widths ≥ k/(N ρ̂_max), k = 10
    nofloor  the sample without the floor, same width floor
Loci: largest / second-largest Re eigenvalue at z = 0 on a fine (α, K) grid for every fit (zero contours =
LMMF incoherence boundaries; verified against PyCoBi Hopf continuations, bimodal/lmmf_hopf_check.py), and the
cluster-resolved maps (largest Re λ among modes with Im λ < 0 / ≥ 0), stored as loci[kind] = (G1, G2, Gm, Gp).
Exact boundaries of ρ: unimodal_kalpha (gauss, g8, dgauss) and bimodal_kalpha (bimodal) files.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python illustration.py fits [--procs P] [--dist a,b]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python illustration.py loci [--dist a,b]
Ramps: illustration_ramps.py. Results: data_paths.MPMF/illustration/
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
import fit_scenarios as FS
import kmo_bifurcation_accuracy as KBA
import sample_size_scan as SS

OUT = dp.ensure(dp.mpmf("illustration"))
N = 1000
SEEDS = {"gauss": 17, "g8": 24, "dgauss": 4, "bimodal": 18}
DISTS = tuple(SEEDS)
LAM, WIDTH_K, M_MAX, N_Q = 1e-6, 10, 64, 5000
KINDS = ("pop", "floor", "nofloor")
FIT_KW = dict(lambda_M=LAM, M_max=M_MAX, patience=3, n_restarts=10, seed=1, method="slsqp", large_M=16)
# (K, α) grids of the ramps and stability maps
A_GRID = UK.A_GRID                                          # 0, 0.05, …, 1.4
K_GRID = {d: (np.round(np.arange(0.1, 6.0001, 0.1), 3) if d == "bimodal" else UK.K_GRID) for d in DISTS}
A_FINE = np.linspace(0.0, 1.45, 146)
K_FINE = {d: np.linspace(0.02, 6.0 if d == "bimodal" else 4.0, 300) for d in DISTS}


def sample(dist):
    return SS.sample(dist, N, SEEDS[dist])


def quantiles(dist, n):
    return SS.quantiles(dist, n)


def pdf(dist, w):
    w = np.asarray(w, float)
    return KBA.rho_true(w) if dist == "bimodal" else np.asarray(UK.dists()[dist]["pdf"](w), float).ravel()


def exact_boundary(dist):
    """Exact incoherence boundary of ρ: columns K, α, Ω."""
    if dist == "bimodal":
        return np.load(os.path.join(dp.mpmf("bimodal_kalpha"), "boundary_exact.npy"))
    return np.load(UK._f(dist, "boundary_exact") + ".npy")


# ─── fits ─────────────────────────────────────────────────────────────────────────────────────────
def fit_file(dist, kind):
    return os.path.join(dp.ensure(os.path.join(OUT, "fits")), f"{dist}_{kind}.npy")


def _fit_job(args):
    dist, kind = args
    out = fit_file(dist, kind)
    if os.path.exists(out):
        return out, 0.0, None
    t0 = time.time()
    if kind == "pop":
        x, bounds, fc = quantiles(dist, N_Q), (1e-4, 1e2), 0.0
    else:
        x = sample(dist); bounds = (FS.width_floor(x, WIDTH_K), 1e2); fc = 1.0 if kind == "floor" else 0.0
    r = LM.fit(x, bounds, floor_c=fc, **FIT_KW)
    m = r["model"]
    q = quantiles(dist, N_Q); u = (np.arange(N_Q) + 0.5) / N_Q
    res = dict(w=m.w, Om=m.Omega, De=m.Delta, M=int(m.M), D=float(r["data_loss"]),
               D_true=float(np.mean((m.cdf(q) - u) ** 2)), delta_min=bounds[0], lam=LAM, floor_c=fc,
               trace_M=np.array([e["M"] for e in r["trace"]]), trace_D=np.array([e["data_loss"] for e in r["trace"]]))
    np.save(out, res, allow_pickle=True)
    return out, time.time() - t0, res


def run_fits(dists, procs):
    jobs = [(d, k) for d in dists for k in KINDS]
    with Pool(min(procs, len(jobs))) as p:
        for out, dt, res in p.imap_unordered(_fit_job, jobs, chunksize=1):
            info = "" if res is None else f"M = {res['M']}, D = {res['D']:.2e}, D_true = {res['D_true']:.2e}"
            print(f"[fits] {os.path.basename(out)}: {info} ({dt:.0f} s)", flush=True)


def load_fit(dist, kind):
    d = np.load(fit_file(dist, kind), allow_pickle=True).item()
    return dict(w=np.asarray(d["w"]), Om=np.asarray(d["Om"]), De=np.asarray(d["De"]), M=int(d["M"]),
                D=d["D"], D_true=d["D_true"])


def mixture_quantiles(f, n):
    """n quantile nodes (i + ½)/n of a Lorentzian mixture (test-oscillator frequencies)."""
    m = LM.LorentzianMixture(f["w"], f["Om"], f["De"])
    u = (np.arange(n) + 0.5) / n
    lo, hi = np.full(n, f["Om"].min() - 1e4), np.full(n, f["Om"].max() + 1e4)
    for _ in range(80):                                            # bisection, vectorised
        mid = 0.5 * (lo + hi); below = m.cdf(mid) < u
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# ─── loci ─────────────────────────────────────────────────────────────────────────────────────────
def stability_maps(f, K, A):
    """G1, G2: largest and second-largest Re λ at z = 0. Gm, Gp: largest Re λ among the modes with
    negative / positive frequency Im λ (cluster-resolved, for the bimodal case: the LMMF analogue of the
    two exact critical lines, whose critical frequencies Ω lie in the left / right cluster)."""
    A0 = np.diag(1j * f["Om"] - f["De"]); P = 0.5 * np.outer(np.ones(f["M"]), f["w"])
    G1 = np.empty((A.size, K.size)); G2, Gm, Gp = (np.empty_like(G1) for _ in range(3))
    for i, a in enumerate(A):
        for j, k in enumerate(K):
            ev = np.linalg.eigvals(A0 + k * np.exp(-1j * a) * P)
            re = np.sort(ev.real)[::-1]
            G1[i, j], G2[i, j] = re[0], re[1]
            neg, pos = ev.real[ev.imag < 0], ev.real[ev.imag >= 0]
            Gm[i, j] = neg.max() if neg.size else -np.inf
            Gp[i, j] = pos.max() if pos.size else -np.inf
    return G1, G2, Gm, Gp


def loci_file(dist):
    return os.path.join(OUT, "fits", f"{dist}_loci.npy")


def run_loci(dists):
    for d in dists:
        t0 = time.time()
        res = dict(K=K_FINE[d], A=A_FINE)
        for k in KINDS:
            res[k] = stability_maps(load_fit(d, k), K_FINE[d], A_FINE)
        np.save(loci_file(d), res, allow_pickle=True)
        print(f"[loci] {d} ({time.time() - t0:.0f} s)", flush=True)


# ─── ramp data and frequency locking ──────────────────────────────────────────────────────────────
S_MAX, K_NB, M_MIN = 0.2, 8, 10          # plateau detection: local slope dω̃/dω < S_MAX over ±K_NB
                                         # neighbours, runs of ≥ M_MIN oscillators
MACRO = 0.05                             # a locked cluster is macroscopic if it holds ≥ 5 % of the oscillators
TOL_EDGE = 5e-3                          # edge growth: |ω̃ − plateau frequency| < TOL_EDGE


def load_ramps(dist, direction):
    """All α chunks of illustration_ramps.py for one direction, ordered by α. Arrays: R_net (nα, nK, ns),
    R_fit (nkind, nα, nK, ns), R_cont (nα, nK, ns), W_net (nα, nK, N), W_test (nkind, nα, nK, N_TEST)."""
    d = os.path.join(OUT, "ramps")
    P = [np.load(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith(f"{dist}_{direction}_c")]
    ia = np.concatenate([p["ia"] for p in P]); o = np.argsort(ia)
    out = {}
    for k, ax in (("R_net", 0), ("R_fit", 1), ("R_cont", 0), ("W_net", 0), ("W_test", 1)):
        out[k] = np.concatenate([p[k] for p in P], axis=ax).take(o, axis=ax).astype(float)
    out.update(alpha=A_GRID[ia[o]], K=P[0]["K"], dts=float(P[0]["dts"]), kinds=[str(k) for k in P[0]["kinds"]])
    return out


def plateaus(om_sorted, w_sorted):
    """Locked runs of ω̃ vs ω (both sorted by ω): (start, stop) index pairs, non-overlapping.
    An oscillator is in a plateau if, over ±K_NB neighbours, the slope dω̃/dω < S_MAX OR the effective
    frequencies differ by < TOL_EDGE (the latter handles (near-)identical natural frequencies, where the
    slope is ill-defined; it is also the frequency resolution of the measurement window)."""
    n = om_sorted.size
    s = np.full(n, np.inf); dw = np.full(n, np.inf)
    i = np.arange(K_NB, n - K_NB)
    dw[i] = np.abs(w_sorted[i + K_NB] - w_sorted[i - K_NB])
    s[i] = dw[i] / np.maximum(om_sorted[i + K_NB] - om_sorted[i - K_NB], 1e-12)
    m = np.concatenate([[False], (s < S_MAX) | (dw < TOL_EDGE), [False]]).astype(int)
    d = np.diff(m); starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
    runs = []
    for a, b in zip(starts, stops):
        if b - a < M_MIN:
            continue
        f = np.median(w_sorted[a:b])                      # grow the run over the edge oscillators that the
        while a > 0 and abs(w_sorted[a - 1] - f) < TOL_EDGE:   # ±K_NB slope window cuts off
            a -= 1
        while b < n and abs(w_sorted[b] - f) < TOL_EDGE:
            b += 1
        runs.append((a, b))
    return runs


def locking(om, W):
    """Locked fraction ℓ and number of macroscopic clusters for every (…, N) row of effective
    frequencies W (natural frequencies om, same for all rows). ℓ is the fraction of oscillators in the
    UNION of the plateaus (grown runs may overlap; nobody is counted twice). Locked oscillators whose ω̃
    agree within 0.02 form one cluster (the slope test can split a plateau)."""
    o = np.argsort(om); om_s = om[o]
    flat = W.reshape(-1, W.shape[-1])[:, o]
    ell = np.zeros(flat.shape[0]); ncl = np.zeros(flat.shape[0], int)
    for r in range(flat.shape[0]):
        mask = np.zeros(om.size, bool)
        for a, b in plateaus(om_s, flat[r]):
            mask[a:b] = True
        ell[r] = mask.mean()
        if mask.any():
            f = np.sort(flat[r][mask])
            groups = np.split(f, np.where(np.diff(f) > 0.02)[0] + 1)
            ncl[r] = sum(g.size >= MACRO * om.size for g in groups)
    return ell.reshape(W.shape[:-1]), ncl.reshape(W.shape[:-1])


def largest_cluster(om, W):
    """Fraction of oscillators in the LARGEST locked cluster (locked oscillators grouped by ω̃ within 0.02,
    as in locking()) for every (…, N) row of W."""
    o = np.argsort(om); om_s = om[o]
    flat = W.reshape(-1, W.shape[-1])[:, o]
    out = np.zeros(flat.shape[0])
    for r in range(flat.shape[0]):
        mask = np.zeros(om.size, bool)
        for a, b in plateaus(om_s, flat[r]):
            mask[a:b] = True
        if mask.any():
            f = np.sort(flat[r][mask])
            out[r] = max(g.size for g in np.split(f, np.where(np.diff(f) > 0.02)[0] + 1)) / om.size
    return out.reshape(W.shape[:-1])


def locking_maps(dist, direction="bwd"):
    """ℓ and macroscopic-cluster counts for the network and every fit's test oscillators (cached)."""
    cache = os.path.join(OUT, "ramps", f"{dist}_{direction}_locking.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    r = load_ramps(dist, direction)
    ell_net, ncl_net = locking(sample(dist), r["W_net"])
    n_test = r["W_test"].shape[-1]
    ell_fit, ncl_fit = [], []
    for j, k in enumerate(r["kinds"]):
        e, c = locking(mixture_quantiles(load_fit(dist, k), n_test), r["W_test"][j])
        ell_fit.append(e); ncl_fit.append(c)
    res = dict(ell_net=ell_net, ncl_net=ncl_net, ell_fit=np.array(ell_fit), ncl_fit=np.array(ncl_fit),
               K=r["K"], alpha=r["alpha"])
    np.savez_compressed(cache, **res)
    return res


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    dists = arg("--dist", list(DISTS), lambda s_: s_.split(","))
    if "fits" in sys.argv:
        run_fits(dists, arg("--procs", os.cpu_count(), int))
    if "loci" in sys.argv:
        run_loci(dists)
