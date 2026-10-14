r"""
Bimodal distribution (Fig. 1 example) in the (K, α) plane: network dynamics and LMMF accuracy
==============================================================================================

ρ = ½ N(−2, 0.6²) + ½ N(2, 0.6²), the N = 5000 frequency sample of Fig. 1 (seed 1). At α = 0 the
incoherent state loses stability in a Hopf bifurcation at K_H = 1.939 (standing-wave limit cycle,
ν* = 1.90) and the locked state appears at the fold K_SN = 4.35 (kmo_bifurcation_accuracy.py truth).

Kuramoto–Sakaguchi network  dθ_i/dt = ω_i + Im(e^{−iθ_i} H),  H = K e^{−iα} Z;
LMMF  dz_m/dt = (iω̄_m − Δ_m) z_m + ½ (H − H̄ z_m²),  H = K e^{−iα} Σ_l w_l z_l.
LMMF variants (λ = 1e-5, warm start, M_max = 16):
  floor    the Fig. 1 sample fit with the noise floor (M = 6; kmo_bif_accuracy/fits.npz) — network ramps;
  nofloor  the same sample fitted without the floor (floor_c = 0; stage fit_nofloor) — LMMF-only ramps.

Stages (resumable; results in data_paths.MPMF/bimodal_kalpha):
  loci     exact incoherence boundary of ρ (e^{iα}/K = J(Ω), both signs of Ω) and the LMMF stability
           maps (largest and second-largest Re eigenvalue at z = 0 on a fine (α, K) grid; their zero
           contours = first and second incoherence boundary)                                (~20 s)
  sims     quasi-static K-ramps of network + LMMF for every α row (as unimodal_kalpha.py): forward
           from an incoherent start (R0 = 0.05), backward from a coherent start (R0 = 0.9), HOLD per
           step, state carried, LMMF finite-size kick N^{-1/2}. RK4, dt = 0.05, R sampled every
           DTS = 0.1 (the standing wave oscillates at ~2ν* ≈ 3.8). One job per (direction, α chunk).
  time     timing probe: one ramp step (all α rows) for network + LMMF
  fit_nofloor   the floorless fit of the Fig. 1 sample                                     (~1 min)
  lmmf_ramps    LMMF-only ramps of the nofloor fit (same grid, kick and initial R as the network ramps)

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_kalpha.py loci
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_kalpha.py sims [--procs P] [--chunks C] [--hold H]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_kalpha.py time
Classification and the (K, α) spectral-RMSE map are computed in the figure script from the ramp files.
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
from scipy.integrate import quad
import data_paths as dp
import unimodal_kalpha as UK                     # RK4 integrators, LMMF rhs, finite-size kick
import kmo_bifurcation_accuracy as KBA           # Fig. 1 sample, true ρ, fits, K_H / K_SN

OUT = dp.ensure(dp.mpmf("bimodal_kalpha"))
N = 5000
K_GRID = np.round(np.arange(0.1, 6.0001, 0.1), 3)        # 60 couplings (K_H = 1.94, K_SN = 4.35 at α = 0)
A_GRID = UK.A_GRID                                       # α = 0, 0.05, …, 1.4
A_FINE = np.linspace(0.0, 1.45, 146)
K_FINE = np.linspace(0.02, 6.0, 300)
DT, DTS, HOLD = 0.05, 0.1, 100.0
DIRECTIONS = {"fwd": 0.05, "bwd": 0.9}                   # ramp direction -> R0 of the wrapped-Cauchy start
KICK = 1.0 / np.sqrt(N)
LAM_FIT = 1e-5


def network_sample():
    om, _ = KBA.fig1_sample()
    return om


VARIANTS = ("floor", "nofloor")
FIT_NOFLOOR = os.path.join(OUT, "fit_nofloor.npy")


def lmmf_fit(variant="floor", lam=LAM_FIT):
    """floor: the Fig. 1 sample fit at λ (noise floor, warm start; kmo_bifurcation_accuracy fits).
    nofloor: the same sample and λ without the floor (run_fit_nofloor)."""
    if variant == "nofloor":
        d = np.load(FIT_NOFLOOR, allow_pickle=True).item()
        return dict(w=d["w"], Om=d["Om"], De=d["De"], M=d["M"])
    f = [g for g in KBA.load_fits() if g["source"] == "sample" and np.isclose(g["lam"], lam)][0]
    return dict(w=f["w"], Om=f["Om"], De=f["De"], M=f["M"])


def fit_settings():
    c = KBA.CFG
    return dict(M_max=c["M_max"], lambda_M=LAM_FIT, patience=c["patience"], n_restarts=c["n_restarts"], seed=1,
                method=c["method"])


def run_fit_nofloor():
    t0 = time.time()
    r = KBA.LM.fit(network_sample(), KBA.CFG["delta_bounds"], floor_c=0.0, **fit_settings())
    m = r["model"]
    np.save(FIT_NOFLOOR, dict(w=m.w, Om=m.Omega, De=m.Delta, M=m.M, data_loss=r["data_loss"],
                              trace_M=np.array([e["M"] for e in r["trace"]]),
                              trace_D=np.array([e["data_loss"] for e in r["trace"]])), allow_pickle=True)
    print(f"[fit_nofloor] M = {m.M}, D = {r['data_loss']:.2e} (floor 1/(6N) = {KBA.LM.noise_floor(N):.2e}; "
          f"floor fit M = {lmmf_fit('floor')['M']}) ({time.time() - t0:.0f} s)")


def truth():
    d = np.load(KBA.TRUTH_NPZ)
    return {k: float(d[k]) for k in ("K_H", "nu_star", "K_SN", "r_SN")}


# ─── loci ─────────────────────────────────────────────────────────────────────────────────────────
def exact_boundary(n_om=1201, om_max=6.0):
    """e^{iα}/K = J(Ω) = (π/2) ρ(Ω) + (i/2) PV∫ ρ(ω)/(ω − Ω) dω over Ω ∈ [−om_max, om_max]
    (both signs: ρ is bimodal, the critical mode sits near a cluster). Columns K, α, Ω."""
    out = []
    for Om in np.linspace(-om_max, om_max, n_om):
        pv = quad(KBA._rho_scalar, -30, 30, weight="cauchy", wvar=Om, limit=400)[0]
        J = 0.5 * np.pi * KBA._rho_scalar(Om) + 0.5j * pv
        if abs(J) > 0:
            out.append((1 / abs(J), np.angle(J), Om))
    return np.array(out)


def stability_map(f):
    """Largest and second-largest real part of the linearisation at z = 0 on (A_FINE × K_FINE). The
    coupling is rank one, so only the collective modes (roots of 1 = (K/2) e^{−iα} Σ w/(λ − iΩ + Δ))
    can become unstable: the zero contour of G1 is the first incoherence boundary, that of G2 the
    instability of the second mode (the second frequency cluster)."""
    A0 = np.diag(1j * f["Om"] - f["De"]); P = 0.5 * np.outer(np.ones(f["M"]), f["w"])
    G1 = np.empty((A_FINE.size, K_FINE.size)); G2 = np.empty_like(G1)
    for i, a in enumerate(A_FINE):
        Pa = np.exp(-1j * a) * P
        for j, K in enumerate(K_FINE):
            ev = np.sort(np.linalg.eigvals(A0 + K * Pa).real)[::-1]
            G1[i, j], G2[i, j] = ev[0], ev[1]
    return G1, G2


def stability_file(variant="floor"):
    return os.path.join(OUT, "lmmf_stability" + ("" if variant == "floor" else f"_{variant}") + ".npy")


def run_loci(variants=VARIANTS):
    t0 = time.time()
    if not os.path.exists(os.path.join(OUT, "boundary_exact.npy")):
        np.save(os.path.join(OUT, "boundary_exact.npy"), exact_boundary())
    for v in variants:
        if v == "nofloor" and not os.path.exists(FIT_NOFLOOR):
            print("[loci] nofloor fit missing (run fit_nofloor first)"); continue
        f = lmmf_fit(v)
        G1, G2 = stability_map(f)
        np.save(stability_file(v), dict(G=G1, G2=G2, A=A_FINE, K=K_FINE, fit=f, truth=truth()), allow_pickle=True)
        print(f"[loci] LMMF {v} (M = {f['M']}) stability maps ({time.time() - t0:.0f} s)")


# ─── ramps ────────────────────────────────────────────────────────────────────────────────────────
def ramp(direction, ia_idx, hold=HOLD):
    """Quasi-static ramp over K_GRID for the α rows ia_idx (vectorised). Returns R_net (nα, nK, ns)
    and R_lmmf (nα, nK, ns), K ascending, plus the observed R(0)."""
    alphas = A_GRID[ia_idx][:, None]
    om = network_sample()
    R0 = DIRECTIONS[direction]
    th = (-np.log(R0)) * np.tan(np.pi * (np.random.default_rng(100).random(N) - 0.5))
    R0_obs = float(np.abs(np.exp(1j * th).mean()))
    th = np.broadcast_to(th, (alphas.size, N)).copy()
    f = lmmf_fit()
    zs = [np.full((alphas.size, f["M"]), R0_obs + 0j)]
    order = range(K_GRID.size) if direction == "fwd" else range(K_GRID.size - 1, -1, -1)
    n_steps, stride = int(round(hold / DT)), int(round(DTS / DT))
    ns = n_steps // stride + 1
    Rn = np.zeros((alphas.size, K_GRID.size, ns)); Rl = np.zeros_like(Rn)
    for ik in order:
        Kc = K_GRID[ik] * np.exp(-1j * alphas)
        Rn[:, ik] = UK._rk4_carry(UK._net_rhs(om, Kc), th, n_steps, stride,
                                  lambda y: np.abs(np.exp(1j * y).mean(1)))
        th = UK._LAST[0]
        Rl[:, ik] = UK._lmmf_step([f], zs, Kc, n_steps, stride, kick=KICK)[0]
    return Rn, Rl, R0_obs


def ramp_file(direction, c, hold=HOLD):
    return os.path.join(OUT, f"ramp_{direction}_c{c:02d}" + ("" if hold == HOLD else f"_hold{hold:g}") + ".npz")


def _ramp_job(args):
    direction, c, idx, hold = args
    out = ramp_file(direction, c, hold)
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    Rn, Rl, R0 = ramp(direction, idx, hold)
    np.savez_compressed(out, R_net=Rn.astype(np.float32), R_lmmf=Rl.astype(np.float32), ia=idx,
                        alpha=A_GRID[idx], K=K_GRID, R0=R0, hold=hold, dts=DTS)
    return out, time.time() - t0


def run_sims(procs, n_chunks, hold=HOLD):
    UK.DT = DT                                           # _rk4_carry reads the step from UK
    chunks = [c for c in np.array_split(np.arange(A_GRID.size), n_chunks) if c.size]
    jobs = [(d, c, idx, hold) for d in DIRECTIONS for c, idx in enumerate(chunks)]
    t0 = time.time()
    with Pool(min(procs, len(jobs))) as p:
        for k, (out, dt) in enumerate(p.imap_unordered(_ramp_job, jobs, chunksize=1), 1):
            el = time.time() - t0
            print(f"[sims] {k}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; {el / 60:.1f} min elapsed, "
                  f"~{el / k * (len(jobs) - k) / 60:.1f} min left)", flush=True)


def lmmf_ramp_file(variant, direction):
    return os.path.join(OUT, f"ramp_lmmf_{variant}_{direction}.npz")


def run_lmmf_ramps(variant="nofloor"):
    """LMMF-only ramps (all α rows at once) with the same initial R, kick and grid as the network ramps."""
    UK.DT = DT
    f = lmmf_fit(variant)
    alphas = A_GRID[:, None]
    n_steps, stride = int(round(HOLD / DT)), int(round(DTS / DT))
    for direction, R0 in DIRECTIONS.items():
        t0 = time.time()
        th = (-np.log(R0)) * np.tan(np.pi * (np.random.default_rng(100).random(N) - 0.5))
        R0_obs = float(np.abs(np.exp(1j * th).mean()))                     # as in ramp()
        zs = [np.full((alphas.size, f["M"]), R0_obs + 0j)]
        Rl = np.zeros((alphas.size, K_GRID.size, n_steps // stride + 1))
        order = range(K_GRID.size) if direction == "fwd" else range(K_GRID.size - 1, -1, -1)
        for ik in order:
            Rl[:, ik] = UK._lmmf_step([f], zs, K_GRID[ik] * np.exp(-1j * alphas), n_steps, stride, kick=KICK)[0]
        np.savez_compressed(lmmf_ramp_file(variant, direction), R_lmmf=Rl.astype(np.float32), alpha=A_GRID,
                            K=K_GRID, R0=R0_obs, hold=HOLD, dts=DTS, M=f["M"])
        print(f"[lmmf_ramps] {variant} {direction} (M = {f['M']}) ({time.time() - t0:.0f} s)", flush=True)


def run_time_probe():
    """One ramp step (all α rows, HOLD) for network + LMMF -> estimated ramp cost."""
    UK.DT = DT
    om = network_sample(); f = lmmf_fit()
    th = np.zeros((A_GRID.size, N)); Kc = 2.0 * np.exp(-1j * A_GRID[:, None])
    n_steps, stride = int(round(HOLD / DT)), int(round(DTS / DT))
    t0 = time.time()
    UK._rk4_carry(UK._net_rhs(om, Kc), th, n_steps, stride, lambda y: np.abs(np.exp(1j * y).mean(1)))
    tn = time.time() - t0
    t0 = time.time()
    UK._lmmf_step([f], [np.full((A_GRID.size, f["M"]), 0.5 + 0j)], Kc, n_steps, stride, kick=KICK)
    tl = time.time() - t0
    full = 2 * K_GRID.size * (tn + tl)                   # both directions, all α rows, one process
    print(f"[time] one ramp step ({A_GRID.size} α rows, N = {N}): network {tn:.1f} s, LMMF {tl:.2f} s -> "
          f"total {full / 60:.0f} CPU-min (2 directions × {K_GRID.size} K); split over P processes ≈ total / P")


if __name__ == "__main__":
    mode = sys.argv[1]
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    if mode == "loci":
        run_loci()
    elif mode == "fit_nofloor":
        run_fit_nofloor()
    elif mode == "lmmf_ramps":
        run_lmmf_ramps(arg("--variant", "nofloor"))
    elif mode == "sims":
        run_sims(arg("--procs", os.cpu_count(), int), arg("--chunks", 4, int), arg("--hold", HOLD, float))
    elif mode == "time":
        run_time_probe()
    else:
        raise SystemExit(__doc__)
