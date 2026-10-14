r"""
Network locked fraction in the two parameter planes of Fig. 3 (heterogeneity control)
=====================================================================================

The N = 5000 network of Fig. 3 (the stored sample of kmo_het_fit.npz, the network of panel d), ramped
quasi-statically from a coherent start (wrapped-Cauchy phases, R0 = 0.9), holding time HOLD per step,
state carried; frequency locking measured as in LMMF/illustration.py (effective frequencies over the
second half of every holding window, plateaus of ω̃ vs ω).

  hK   (h, K) plane, panel (e): ω_i(h) = ω̄ + h (ω_i⁰ − ω̄) (as in panel d, ω̄ = sample mean);
       for every h, K is ramped DOWN from K_MAX to K_MIN.
  DC   (h_Δ, h_ω̄) plane at K = CONFIG K, panel (f): every oscillator is assigned to its most likely
       fitted Lorentzian component m(i) (largest w_m L(ω_i; Ω_m, Δ_m)) and
           ω_i(h_Δ, h_ω̄) = Ω̄ + h_ω̄ (Ω_m − Ω̄) + h_Δ (ω_i⁰ − Ω_m)
       (centres scaled by h_ω̄, the spread within each component by h_Δ — the micro analogue of the
       mean-field knobs of kmo_heterogeneity_bifurcation.py); for every h_Δ, h_ω̄ is ramped UP from
       the homogeneous-centre limit.
Output: <plane>_locking.npz in data_paths.MPMF/kmo_het_locking/ with x, y grids and ell, n_clusters,
R_mean, R_std arrays of shape (ny, nx).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_heterogeneity_locking.py [hK] [DC] [--procs P]
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import time
import numpy as np
from multiprocessing import Pool
import data_paths as dp

OUT = dp.ensure(dp.mpmf("kmo_het_locking"))
FIT_NPZ = dp.mpmf("kmo_het_fit.npz")
K_FIX = 1.0                                       # kmo_heterogeneity_sim.CONFIG["K"]
DT, HOLD, R0_START = 0.05, 100.0, 0.9
N_GRID = 30
GRIDS = {   # (row parameter = x axis, step parameter = y axis), step order = ramp direction
    "hK": dict(x=np.linspace(0.05, 1.3, N_GRID), y=np.linspace(0.2, 3.0, N_GRID), y_order="down"),
    "DC": dict(x=np.linspace(0.02, 1.3, N_GRID), y=np.linspace(0.02, 1.3, N_GRID), y_order="up"),
}


def load():
    d = np.load(FIT_NPZ, allow_pickle=False)
    return (np.asarray(d["weights"], float), np.asarray(d["omega"], float), np.asarray(d["delta"], float),
            float(d["omega_bar"]), np.asarray(d["samples"], float))


def frequencies(plane, xs, y):
    """Natural frequencies (len(xs), N) of the rows xs at step value y."""
    w, Om, De, Ombar, om0 = load()
    if plane == "hK":
        ombar = om0.mean()
        return ombar + xs[:, None] * (om0 - ombar)[None, :]
    resp = w[None, :] * De[None, :] / ((om0[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    m = np.argmax(resp, axis=1)
    return Ombar + y * (Om[m] - Ombar)[None, :] + xs[:, None] * (om0 - Om[m])[None, :]


def _job(args):
    plane, ix = args
    g = GRIDS[plane]; xs = g["x"][ix]
    ys = g["y"] if g["y_order"] == "up" else g["y"][::-1]
    n = int(round(HOLD / DT)); half = n // 2
    N = load()[4].size
    th = np.broadcast_to((-np.log(R0_START)) * np.tan(np.pi * (np.random.default_rng(100).random(N) - 0.5)),
                         (xs.size, N)).copy()
    nx, ny = xs.size, ys.size
    W = np.zeros((ny, nx, N), np.float32); OM = np.zeros((ny, nx, N), np.float32)
    Rm = np.zeros((ny, nx)); Rs = np.zeros((ny, nx))
    for j, y in enumerate(ys):
        om = frequencies(plane, xs, y)
        K = y if plane == "hK" else K_FIX
        rhs = lambda th_: om + K * np.imag(np.exp(-1j * th_) * np.exp(1j * th_).mean(1, keepdims=True))
        Rt = []
        for s in range(1, n + 1):
            k1 = rhs(th); k2 = rhs(th + 0.5 * DT * k1); k3 = rhs(th + 0.5 * DT * k2); k4 = rhs(th + DT * k3)
            th = th + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if s == half:
                th_half = th.copy()
            if s > half and s % 2 == 0:
                Rt.append(np.abs(np.exp(1j * th).mean(1)))
        jj = j if g["y_order"] == "up" else ny - 1 - j            # store in ascending y
        W[jj] = (th - th_half) / ((n - half) * DT); OM[jj] = om
        Rt = np.array(Rt); Rm[jj] = Rt.mean(0); Rs[jj] = Rt.std(0)
    return plane, ix, W, OM, Rm, Rs


def run(planes, procs):
    sys.path.insert(0, _HERE)
    import illustration as IL                                     # plateau detection (locking)
    for plane in planes:
        out = os.path.join(OUT, f"{plane}_locking.npz")
        if os.path.exists(out):
            print(f"[{plane}] exists: {out}"); continue
        g = GRIDS[plane]
        chunks = [c for c in np.array_split(np.arange(g["x"].size), procs) if c.size]
        t0 = time.time()
        ell = np.zeros((g["y"].size, g["x"].size)); ncl = np.zeros_like(ell, int)
        Rm = np.zeros_like(ell); Rs = np.zeros_like(ell)
        Wall = np.zeros((g["y"].size, g["x"].size, load()[4].size), np.float32)
        with Pool(min(procs, len(chunks))) as p:
            for k, (_, ix, W, OM, rm, rs) in enumerate(p.imap_unordered(_job, [(plane, c) for c in chunks]), 1):
                Wall[:, ix] = W
                for a in range(W.shape[0]):
                    for b in range(W.shape[1]):
                        e, c = IL.locking(OM[a, b].astype(float), W[a, b][None].astype(float))
                        ell[a, ix[b]], ncl[a, ix[b]] = e[0], c[0]
                Rm[:, ix], Rs[:, ix] = rm, rs
                print(f"[{plane}] {k}/{len(chunks)} chunks ({(time.time() - t0) / 60:.1f} min)", flush=True)
        np.savez_compressed(out, x=g["x"], y=g["y"], ell=ell, n_clusters=ncl, R_mean=Rm, R_std=Rs,
                            y_order=g["y_order"], hold=HOLD, K_fix=K_FIX)
        np.savez_compressed(os.path.join(OUT, f"{plane}_W.npz"), W=Wall, x=g["x"], y=g["y"])   # raw ω̃ (re-analysis)
        print(f"[{plane}] saved {out}  (ℓ range {ell.min():.3f}–{ell.max():.3f})", flush=True)


def add_largest_cluster(planes=("hK", "DC")):
    """Add 'ell_max' (fraction in the largest locked cluster) to <plane>_locking.npz from the stored ω̃."""
    sys.path.insert(0, _HERE)
    import illustration as IL
    for plane in planes:
        f = os.path.join(OUT, f"{plane}_locking.npz"); d = dict(np.load(f))
        Wd = np.load(os.path.join(OUT, f"{plane}_W.npz")); W = Wd["W"]
        lm = np.zeros(W.shape[:2])
        for iy, y in enumerate(d["y"]):
            om = frequencies(plane, d["x"], y)
            for ix in range(d["x"].size):
                lm[iy, ix] = IL.largest_cluster(om[ix], W[iy, ix][None].astype(float))[0]
        d["ell_max"] = lm
        np.savez_compressed(f, **d)
        print(f"[{plane}] ell_max added (range {lm.min():.3f}–{lm.max():.3f})")


if __name__ == "__main__":
    if "largest" in sys.argv:
        add_largest_cluster(); sys.exit(0)
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    planes = [p for p in ("hK", "DC") if p in sys.argv] or ["hK", "DC"]
    run(planes, arg("--procs", os.cpu_count(), int))
