r"""
N = 1000 network K-ramps for the combined unimodal figure (fig_unimodal_combined.py)
===================================================================================

For every distribution, the network is built on sample 0 of fit_scenarios.py (N = 1000), i.e. the same
sample that is shown as histogram and fitted in scenarios (2) "sample, floor" and (3) "sample, no floor".
Ramped identically (quasi-static K-ramps over all α rows, forward from an incoherent and backward from a
coherent wrapped-Cauchy start, holding time unimodal_kalpha.HOLD per step):
  * the network (N = 1000),
  * the three LMMF fits: (1) true ρ, no floor; (2) sample 0, floor; (3) sample 0, no floor,
  * the infinite-N mean field of the TRUE ρ ("continuum": the OA equation on N_CONT quantile nodes of ρ),
    the finite-size reference for the RMSE row,
with the finite-size seed KICK = N^{-1/2} (N = 1000) for all mean-field models.

Jobs: one per (distribution, direction, α chunk) for network + LMMF fits, and one per (distribution,
direction, α chunk) for the continuum; resumable output files in unimodal_kalpha/n1000/.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_n1000.py [--procs P] [--chunks C] [--kind net|cont|lmmf] [--dist a,b] [--mmax M] [--width-k K]
        kind lmmf: recompute only the LMMF traces for the scenario fits with M_max = M
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import time
import numpy as np
from multiprocessing import Pool
import unimodal_kalpha as UK
import fit_scenarios as FS

N_NET = 1000
KICK = 1.0 / np.sqrt(N_NET)
N_CONT = 2000
SCEN_ORDER = [("pop", 0), ("floor", 0), ("nofloor", 0)]
MMAX = 16                       # which scenario fits to use (FS.fpath tag); set via --mmax
WIDTH_K = 0                     # width-floor tag of the sample fits; set via --width-k
OUT = UK.dp.ensure(os.path.join(UK.OUT, "n1000"))


def lmmf_file(name, direction, ci, mmax=None):
    """LMMF-only ramp traces for the scenario fits with M_max = mmax (mmax = 16: stored in the net files)."""
    wk = f"_wk{WIDTH_K:g}" if WIDTH_K else ""
    return os.path.join(OUT, f"{name}_lmmf_mmax{MMAX if mmax is None else mmax}{wk}_{direction}_c{ci:02d}.npz")


def load_scenario_fits(name, mmax=None):
    fits = []
    for scen, i in SCEN_ORDER:
        r = np.load(FS.fpath(name, scen, i, N_NET, MMAX if mmax is None else mmax, WIDTH_K),
                    allow_pickle=True).item()
        fits.append(dict(w=r["w"], Om=r["Om"], De=r["De"], M=r["M"]))
    return fits


def network_sample(name):
    return FS.samples(name, N_NET)[0]


def _start(direction, n):
    R0 = UK.DIRECTIONS[direction]
    th = (-np.log(R0)) * np.tan(np.pi * (np.random.default_rng(100).random(n) - 0.5))
    return th, float(np.abs(np.exp(1j * th).mean()))


def _order(direction):
    return range(UK.K_GRID.size) if direction == "fwd" else range(UK.K_GRID.size - 1, -1, -1)


def _job(args):
    kind, name, direction, ci, idx = args
    out = lmmf_file(name, direction, ci) if kind == "lmmf" else os.path.join(OUT, f"{name}_{kind}_{direction}_c{ci:02d}.npz")
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    alphas = UK.A_GRID[idx][:, None]
    n_steps, stride = int(round(UK.HOLD / UK.DT)), int(round(UK.DTS / UK.DT))
    ns = n_steps // stride + 1
    th0, R0 = _start(direction, N_NET)
    if kind == "net":                                   # network + the three LMMF fits
        om = network_sample(name)
        th = np.broadcast_to(th0, (alphas.size, N_NET)).copy()
        fits = load_scenario_fits(name)
        zs = [np.full((alphas.size, f["M"]), R0 + 0j) for f in fits]
        Rn = np.zeros((alphas.size, UK.K_GRID.size, ns)); Rl = np.zeros((len(fits),) + Rn.shape)
        for ik in _order(direction):
            Kc = UK.K_GRID[ik] * np.exp(-1j * alphas)
            Rn[:, ik] = UK._rk4_carry(UK._net_rhs(om, Kc), th, n_steps, stride,
                                      lambda y: np.abs(np.exp(1j * y).mean(1)))
            th = UK._LAST[0]
            Rl[:, :, ik] = UK._lmmf_step(fits, zs, Kc, n_steps, stride, kick=KICK)
        np.savez_compressed(out, R_net=Rn.astype(np.float32), R_lmmf=Rl.astype(np.float32), ia=idx, R0=R0,
                            scenarios=np.array([s for s, _ in SCEN_ORDER]))
    elif kind == "lmmf":                                # LMMF fits only (network/continuum reused)
        fits = load_scenario_fits(name)
        zs = [np.full((alphas.size, f["M"]), R0 + 0j) for f in fits]
        Rl = np.zeros((len(fits), alphas.size, UK.K_GRID.size, ns))
        for ik in _order(direction):
            Rl[:, :, ik] = UK._lmmf_step(fits, zs, UK.K_GRID[ik] * np.exp(-1j * alphas), n_steps, stride, kick=KICK)
        np.savez_compressed(out, R_lmmf=Rl.astype(np.float32), ia=idx, R0=R0, mmax=MMAX,
                            scenarios=np.array([s for s, _ in SCEN_ORDER]))
    else:                                               # continuum mean field of the true ρ
        nodes = FS.quantile_nodes(name, N_CONT)
        f = dict(Om=nodes, De=np.zeros(N_CONT), w=np.full(N_CONT, 1.0 / N_CONT), M=N_CONT)
        zs = [np.full((alphas.size, N_CONT), R0 + 0j)]
        Rc = np.zeros((alphas.size, UK.K_GRID.size, ns))
        for ik in _order(direction):
            Rc[:, ik] = UK._lmmf_step([f], zs, UK.K_GRID[ik] * np.exp(-1j * alphas), n_steps, stride, kick=KICK)[0]
        np.savez_compressed(out, R_cont=Rc.astype(np.float32), ia=idx, R0=R0, n_nodes=N_CONT)
    return out, time.time() - t0


def run(procs, n_chunks, kinds=("net", "cont"), names=None):
    chunks = [np.array(c) for c in np.array_split(np.arange(UK.A_GRID.size), n_chunks)]
    jobs = [(k, n, d, ci, idx) for k in kinds for n in (names or UK.dists()) for d in UK.DIRECTIONS
            for ci, idx in enumerate(chunks)]
    t0 = time.time()
    with Pool(procs) as p:
        for i, (out, dt) in enumerate(p.imap_unordered(_job, jobs, chunksize=1), 1):
            el = time.time() - t0
            print(f"[n1000] {i}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; {el / 60:.1f} min elapsed, "
                  f"~{el / i * (len(jobs) - i) / 60:.1f} min left)", flush=True)


def time_probe():
    """One ramp step (all α rows) for the network (N = 1000) and the continuum (N_CONT nodes)."""
    om = network_sample("gauss"); th = np.zeros((UK.A_GRID.size, N_NET))
    Kc = 2.0 * np.exp(-1j * UK.A_GRID[:, None]); n_steps, stride = int(round(UK.HOLD / UK.DT)), int(round(UK.DTS / UK.DT))
    t0 = time.time(); UK._rk4_carry(UK._net_rhs(om, Kc), th, n_steps, stride, lambda y: np.abs(np.exp(1j * y).mean(1)))
    tn = time.time() - t0
    nodes = FS.quantile_nodes("gauss", N_CONT); f = dict(Om=nodes, De=np.zeros(N_CONT), w=np.full(N_CONT, 1 / N_CONT), M=N_CONT)
    t0 = time.time(); UK._lmmf_step([f], [np.full((UK.A_GRID.size, N_CONT), 0.5 + 0j)], Kc, n_steps, stride)
    tc = time.time() - t0
    nK = UK.K_GRID.size
    print(f"[time] one ramp step: network {tn:.1f}s, continuum {tc:.1f}s -> per (dist, direction): "
          f"network job {nK * tn / 60:.1f} min, continuum job {nK * tc / 60:.1f} min; "
          f"all 16 jobs = {16 * nK * (tn + tc) / 2 / 60:.0f} CPU-min (/8 cores ≈ {16 * nK * (tn + tc) / 2 / 60 / 8:.0f} min)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    if "time" in sys.argv[1:2]:
        time_probe()
    else:
        MMAX = arg("--mmax", MMAX, int)
        WIDTH_K = arg("--width-k", WIDTH_K, float)
        run(arg("--procs", os.cpu_count(), int), arg("--chunks", 1, int),
            tuple(arg("--kind", "net,cont").split(",")), arg("--dist", None, lambda s_: s_.split(",")))
