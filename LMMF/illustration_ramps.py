r"""
(K, α) ramps for the illustration samples: network, LMMF fits, true-ρ continuum, frequency locking
===================================================================================================

For every distribution of illustration.py (N = 1000 sample) and every α row of illustration.A_GRID,
quasi-static K-ramps over illustration.K_GRID[dist] — forward from an incoherent start (R0 = 0.05),
backward from a coherent start (R0 = 0.9), wrapped-Cauchy phases, holding time HOLD per step, state
carried — of
  net    the Kuramoto–Sakaguchi network (N = 1000)
  fits   the LMMF fits pop / floor / nofloor (finite-size kick N^{-1/2} per step, |z| ≤ 1)
  cont   the true-ρ mean field (OA on N_CONT quantile nodes of ρ, same kick)
R(t) is sampled every DTS. Frequency locking (both sides measured the same way):
  W_net    effective frequency of every network oscillator over the second half of each window,
           ω̃_i = [θ_i(T) − θ_i(T/2)] / (T/2) (unwrapped phases)
  W_test   the same for N_TEST test oscillators per fit: frequencies = quantile nodes of the fitted
           mixture, driven WITHOUT feedback by that LMMF's own field H(t) = K e^{−iα} Σ w z(t)
           (exact in the large-N limit; also valid for oscillatory states)
RK4, dt = DT. One job per (distribution, direction, α chunk); resumable files in illustration/ramps/.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python illustration_ramps.py time
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python illustration_ramps.py run [--dist a,b] [--procs P] [--chunks C]
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import time
import numpy as np
from multiprocessing import Pool
import illustration as IL

OUT = IL.dp.ensure(os.path.join(IL.OUT, "ramps"))
DT, DTS, HOLD = 0.05, 0.1, 100.0
DIRECTIONS = {"fwd": 0.05, "bwd": 0.9}
N_TEST, N_CONT = 1000, 2000
KICK = 1.0 / np.sqrt(IL.N)


def _wrapped_cauchy(R0, n, seed):
    return (-np.log(R0)) * np.tan(np.pi * (np.random.default_rng(seed).random(n) - 0.5))


def _project(z):
    r = np.abs(z)
    return np.where(r > 1.0, z / np.maximum(r, 1e-300), z)


def ramp(dist, direction, ia_idx, hold=HOLD, k_steps=None):
    """Returns dict of R_net (nα, nK, ns), R_fit (nkind, nα, nK, ns), R_cont, W_net (nα, nK, N),
    W_test (nkind, nα, nK, N_TEST); K ascending. k_steps limits the number of K steps (timing)."""
    alphas = IL.A_GRID[ia_idx][:, None]; na = alphas.shape[0]
    Ks = IL.K_GRID[dist]; nK = Ks.size
    om = IL.sample(dist)
    R0 = DIRECTIONS[direction]
    th0 = _wrapped_cauchy(R0, IL.N, 100); R0_obs = float(np.abs(np.exp(1j * th0).mean()))
    th = np.broadcast_to(th0, (na, IL.N)).copy()
    fits = [IL.load_fit(dist, k) for k in IL.KINDS]
    q = IL.quantiles(dist, N_CONT)
    cont = dict(Om=q, De=np.zeros(N_CONT), w=np.full(N_CONT, 1.0 / N_CONT), M=N_CONT)
    systems = fits + [cont]                                          # OA systems (last: continuum)
    a_ = [1j * f["Om"] - f["De"] for f in systems]
    zs = [np.full((na, f["M"]), R0_obs + 0j) for f in systems]
    nodes = [IL.mixture_quantiles(f, N_TEST) for f in fits]
    phis = [np.broadcast_to(_wrapped_cauchy(R0, N_TEST, 101), (na, N_TEST)).copy() for _ in fits]
    n_steps, stride = int(round(hold / DT)), int(round(DTS / DT)); half = n_steps // 2
    ns = n_steps // stride + 1
    nf = len(fits)
    R_net = np.zeros((na, nK, ns)); R_sys = np.zeros((len(systems), na, nK, ns))
    W_net = np.zeros((na, nK, IL.N), np.float32); W_test = np.zeros((nf, na, nK, N_TEST), np.float32)
    order = list(range(nK)) if direction == "fwd" else list(range(nK - 1, -1, -1))
    if k_steps:
        order = order[:k_steps]

    for ik in order:
        Kc = Ks[ik] * np.exp(-1j * alphas)                              # (nα, 1)
        zs = [_project(z + KICK) for z in zs]

        def rhs(state):
            th_, zs_, ph_ = state
            Z = np.exp(1j * th_).mean(1, keepdims=True)
            d_th = om[None, :] + np.imag(np.exp(-1j * th_) * (Kc * Z))
            d_z, d_ph = [], []
            for j, (z, f) in enumerate(zip(zs_, systems)):
                H = Kc * (z @ f["w"])[:, None]
                d_z.append(a_[j] * z + 0.5 * (H - np.conj(H) * z * z))
                if j < nf:
                    d_ph.append(nodes[j][None, :] + np.imag(np.exp(-1j * ph_[j]) * H))
            return d_th, d_z, d_ph

        def add(s, k, c):
            return (s[0] + c * k[0], [x + c * y for x, y in zip(s[1], k[1])], [x + c * y for x, y in zip(s[2], k[2])])

        state = (th, zs, phis)
        obs = lambda s: (np.abs(np.exp(1j * s[0]).mean(1)), [np.abs(z @ f["w"]) for z, f in zip(s[1], systems)])
        r_n, r_s = obs(state); R_net[:, ik, 0] = r_n
        for j in range(len(systems)):
            R_sys[j, :, ik, 0] = r_s[j]
        for s in range(1, n_steps + 1):
            k1 = rhs(state); k2 = rhs(add(state, k1, 0.5 * DT)); k3 = rhs(add(state, k2, 0.5 * DT))
            k4 = rhs(add(state, k3, DT))
            state = (state[0] + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                     [x + DT / 6 * (a1 + 2 * a2 + 2 * a3 + a4) for x, a1, a2, a3, a4 in zip(state[1], k1[1], k2[1], k3[1], k4[1])],
                     [x + DT / 6 * (a1 + 2 * a2 + 2 * a3 + a4) for x, a1, a2, a3, a4 in zip(state[2], k1[2], k2[2], k3[2], k4[2])])
            if s == half:
                th_half, ph_half = state[0].copy(), [p.copy() for p in state[2]]
            if s % stride == 0:
                r_n, r_s = obs(state); R_net[:, ik, s // stride] = r_n
                for j in range(len(systems)):
                    R_sys[j, :, ik, s // stride] = r_s[j]
        T2 = (n_steps - half) * DT
        W_net[:, ik] = (state[0] - th_half) / T2
        for j in range(nf):
            W_test[j, :, ik] = (state[2][j] - ph_half[j]) / T2
        th, zs, phis = state
    return dict(R_net=R_net, R_fit=R_sys[:nf], R_cont=R_sys[nf], W_net=W_net, W_test=W_test, R0=R0_obs)


def lmmf_ramp_nokick(dist, alpha, kinds=("floor", "nofloor"), direction="bwd", seed_below=True):
    """LMMF-only ramp at one α for visualisation, without the step-wise spikes of the finite-size kick:
    seed_below=True adds the kick N^{-1/2} at a K step ONLY if the LMMF's R has fallen below N^{-1/2} (near
    incoherence, where a finite network is never exactly at R = 0 and the seed is needed for re-growth,
    e.g. re-entrant synchrony); seed_below=False: no kick at all (re-growth from ~0 is then delayed).
    Same start R(0), K grid, DT, DTS and HOLD as ramp(). Cached. Returns R (nkind, nK, ns), K ascending."""
    out = os.path.join(OUT, f"{dist}_{direction}_{'seedkick' if seed_below else 'nokick'}_a{alpha:g}.npz")
    if os.path.exists(out):
        return np.load(out)["R"]
    th0 = _wrapped_cauchy(DIRECTIONS[direction], IL.N, 100); R0_obs = float(np.abs(np.exp(1j * th0).mean()))
    Ks = IL.K_GRID[dist]
    n_steps, stride = int(round(HOLD / DT)), int(round(DTS / DT)); ns = n_steps // stride + 1
    order = list(range(Ks.size)) if direction == "fwd" else list(range(Ks.size - 1, -1, -1))
    R = np.zeros((len(kinds), Ks.size, ns))
    for j, k in enumerate(kinds):
        f = IL.load_fit(dist, k); a = 1j * f["Om"] - f["De"]; w = f["w"]
        z = np.full(f["M"], R0_obs + 0j)
        for ik in order:
            Kc = Ks[ik] * np.exp(-1j * alpha)
            if seed_below and abs(z @ w) < KICK:
                z = _project(z + KICK)
            F = lambda z_: a * z_ + 0.5 * (Kc * (z_ @ w) - np.conj(Kc * (z_ @ w)) * z_ * z_)
            R[j, ik, 0] = abs(z @ w)
            for s in range(1, n_steps + 1):
                k1 = F(z); k2 = F(z + 0.5 * DT * k1); k3 = F(z + 0.5 * DT * k2); k4 = F(z + DT * k3)
                z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if s % stride == 0:
                    R[j, ik, s // stride] = abs(z @ w)
    np.savez_compressed(out, R=R.astype(np.float32), K=Ks, alpha=alpha, kinds=np.array(kinds))
    return R


def ramp_file(dist, direction, c):
    return os.path.join(OUT, f"{dist}_{direction}_c{c:02d}.npz")


def _job(args):
    dist, direction, c, idx = args
    out = ramp_file(dist, direction, c)
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    r = ramp(dist, direction, idx)
    np.savez_compressed(out, ia=idx, alpha=IL.A_GRID[idx], K=IL.K_GRID[dist], kinds=np.array(IL.KINDS),
                        R0=r["R0"], hold=HOLD, dts=DTS, n_test=N_TEST,
                        **{k: r[k].astype(np.float32) for k in ("R_net", "R_fit", "R_cont", "W_net", "W_test")})
    return out, time.time() - t0


def run(dists, procs, n_chunks):
    chunks = [c for c in np.array_split(np.arange(IL.A_GRID.size), n_chunks) if c.size]
    cost = lambda j: IL.K_GRID[j[0]].size * len(j[3])
    jobs = sorted([(d, r, c, idx) for d in dists for r in DIRECTIONS for c, idx in enumerate(chunks)],
                  key=cost, reverse=True)
    t0 = time.time()
    with Pool(min(procs, len(jobs))) as p:
        for k, (out, dt) in enumerate(p.imap_unordered(_job, jobs, chunksize=1), 1):
            el = time.time() - t0
            print(f"[ramps] {k}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; {el / 60:.1f} min elapsed, "
                  f"~{el / k * (len(jobs) - k) / 60:.1f} min left)", flush=True)


def time_probe(dists):
    total = 0.0
    for d in dists:
        t0 = time.time()
        ramp(d, "fwd", np.array([10]), k_steps=1)
        dt = time.time() - t0
        cpu = dt * IL.A_GRID.size * IL.K_GRID[d].size * len(DIRECTIONS)
        total += cpu
        print(f"[time] {d}: one α row, one K step {dt:.1f} s -> {cpu / 60:.0f} CPU-min for the full grid", flush=True)
    print(f"[time] total {total / 60:.0f} CPU-min; wall ≈ total / P (P processes, chunks ≥ P / 2 per direction)")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    dists = arg("--dist", list(IL.DISTS), lambda s_: s_.split(","))
    if "time" in sys.argv:
        time_probe(dists)
    if "run" in sys.argv:
        run(dists, arg("--procs", os.cpu_count(), int), arg("--chunks", IL.A_GRID.size, int))
