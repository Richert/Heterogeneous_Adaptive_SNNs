r"""
Network sweep over the Δ–β plane of the adaptation phase lag, G_β(φ) = sin(φ + β)
===================================================================================

Companion of weight_variance_beta_plane.py (mean field + quasi-static self-consistency). For each
(β, Δ) on a grid and two initial conditions — coherent (θ_i = 0, uniform weights at the mean-field value
Ā = 1 + κ sin β for R = 1) and asynchronous (uniform random phases, A_ij = 1) — the N = 500 network is
integrated for T = 5000 and characterized over the final 20 %:
  * R, collective frequency Ω (linear fit to the unwrapped mean phase), Ā, V_A/Ā², ⟨|Z_2|⟩, ⟨|Z_2 − Z²|⟩;
  * per-oscillator mean frequencies → largest frequency-locked fraction (oscillators within ±lock_tol
    of a common frequency) and the phase coherence of that cluster at the final time, which separates
    compact locked clusters from phase-dispersed frequency-locked states.

    python weight_variance_beta_grid.py --workers 6      # → weight_variance_beta_grid.npz (~12-15 min)
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import argparse, time
import numpy as np

CONFIG = dict(N=500, K=1.0, gamma=0.01, mu=0.03, dt=0.05, T=5000.0, rec_every=10, lock_tol=5e-3,
              betas=np.linspace(0.0, np.pi / 2, 13), deltas=np.round(np.linspace(0.05, 1.2, 15), 4),
              inits=("coherent", "async"), seed=21,
              out=dp.mpmf("weight_variance_beta_grid.npz"))
#: refined grid: Δ step 0.04, plus a third initial condition on the quasi-static state (structured
#: weights A_ij = 1 + κ Im(e^{iβ} c_j c_i*)) within `qs_window` of the end of the quasi-static branch
CONFIG_FINE = dict(CONFIG, deltas=np.round(np.arange(0.02, 1.1801, 0.04), 4), inits=("coherent", "async", "qs"),
                   qs_window=0.2, plane_npz=dp.mpmf("weight_variance_beta_plane.npz"),
                   out=dp.mpmf("weight_variance_beta_grid_fine.npz"))


def _worker_init():
    import numba
    numba.set_num_threads(1)


def run_one(task):
    import weight_variance_rule_micro as base
    beta, D, init, seed, cfg, qs_state = task
    N, K, g, mu, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["mu"], cfg["dt"]
    rng = np.random.default_rng(seed)
    omega = base.lorentzian_quantiles(N, D)
    if init == "qs":                                                  # on the quasi-static state
        import weight_variance_quasistatic as qs
        R0, Om0, W0 = qs_state                                    # re-solve from (R, Ω, |W|)
        st = qs.solve_beta(omega, mu / g, K, beta, (R0, Om0, complex(W0)))
        if st is None:
            st = qs.state_beta(omega, R0, Om0, complex(W0), mu / g, K, beta)
        c = st["c"]
        theta = np.angle(c).copy()
        dr = ~st["locked"]
        r = np.clip(np.abs(c[dr]), 1e-9, 1 - 1e-9)              # wrapped Cauchy with first moment c_i
        u = rng.uniform(-np.pi, np.pi, dr.sum())
        theta[dr] = np.angle(c[dr]) + 2.0 * np.arctan(((1 - r) / (1 + r)) * np.tan(u / 2.0))
        A = 1.0 + mu / g * np.imag(np.exp(1j * beta) * np.outer(np.conj(c), c))
    else:
        if init == "coherent":
            theta, A0 = np.zeros(N), 1.0 + mu / g * np.sin(beta)
        else:
            theta, A0 = rng.uniform(-np.pi, np.pi, N), 1.0
        A = np.full((N, N), A0)
    A = np.ascontiguousarray(A)
    np.fill_diagonal(A, 0.0)
    sb, cb = np.sin(beta), np.cos(beta)
    mu_eff, c_cos = mu * (sb + cb), sb / (sb + cb)
    n = int(cfg["T"] / dt)
    n0 = int(0.8 * n)
    e0 = np.empty(0)
    base._simulate_numba_z(theta, A, omega, K, mu_eff, g, dt, n0, np.zeros(n0, np.bool_), c_cos,
                           e0, e0, e0, e0.astype(complex), e0.astype(complex))
    th0 = theta.copy()
    n1 = n - n0
    rec = np.zeros(n1, np.bool_)
    rec[::cfg["rec_every"]] = True
    m = rec.sum()
    Rs, Ab, VA = np.empty(m), np.empty(m), np.empty(m)
    Z1, Z2 = np.empty(m, complex), np.empty(m, complex)
    base._simulate_numba_z(theta, A, omega, K, mu_eff, g, dt, n1, rec, c_cos, Rs, Ab, VA, Z1, Z2)
    Tw = n1 * dt
    t = np.arange(m) * dt * cfg["rec_every"]
    Om = np.polyfit(t, np.unwrap(np.angle(Z1)), 1)[0]
    w = (theta - th0) / Tw                                            # per-oscillator mean frequency
    ws = np.sort(w)
    hi = np.searchsorted(ws, ws + 2 * cfg["lock_tol"], side="right")
    k = int(np.argmax(hi - np.arange(N)))                             # densest frequency window
    w_c = ws[k:hi[k]].mean()
    members = np.abs(w - w_c) <= cfg["lock_tol"]
    R_cl = np.abs(np.exp(1j * theta[members]).mean()) if members.any() else np.nan
    return dict(R=Rs.mean(), R_std=Rs.std(), Omega=Om, Abar=Ab.mean(), VArel=(VA / Ab ** 2).mean(),
                Z2=np.abs(Z2).mean(), oa_dist=np.abs(Z2 - Z1 ** 2).mean(), lock_frac=members.mean(),
                lock_freq=w_c, R_cluster=R_cl)


def main(cfg=CONFIG, workers=6):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    betas, deltas, inits = cfg["betas"], cfg["deltas"], cfg["inits"]
    tasks, idx = [], []
    plane = np.load(cfg["plane_npz"]) if "qs" in inits else None
    for a, b in enumerate(betas):
        for j, D in enumerate(deltas):
            for c, init in enumerate(inits):
                qs_state = None
                if init == "qs":                                      # last quasi-static state ≤ Δ
                    ok = np.isfinite(plane["qs_R"][a]) & (plane["deltas"] <= D + 1e-9)
                    end = plane["qs_last"][a]
                    if not ok.any() or not (end - cfg["qs_window"] <= D <= end + 0.08):
                        continue
                    k = np.nonzero(ok)[0][-1]
                    qs_state = (plane["qs_R"][a, k], plane["qs_Omega"][a, k], plane["qs_Wabs"][a, k])
                tasks.append((b, D, init, cfg["seed"] + 1000 * a + 10 * j + c, cfg, qs_state))
                idx.append((a, j, c))
    keys = ("R", "R_std", "Omega", "Abar", "VArel", "Z2", "oa_dist", "lock_frac", "lock_freq", "R_cluster")
    out = {k: np.full((betas.size, deltas.size, len(inits)), np.nan) for k in keys}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_worker_init) as ex:
        for n, ((a, j, c), res) in enumerate(zip(idx, ex.map(run_one, tasks, chunksize=1))):
            for k in keys:
                out[k][a, j, c] = res[k]
            if (n + 1) % 60 == 0:
                print(f"[{n + 1}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    np.savez(cfg["out"], betas=betas, deltas=deltas, inits=np.array(inits), mu=cfg["mu"], gamma=cfg["gamma"],
             K=cfg["K"], N=cfg["N"], T=cfg["T"], **out)
    print(f"saved {cfg['out']} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fine", action="store_true", help="refined Δ grid + quasi-static initial condition")
    args = ap.parse_args()
    main(CONFIG_FINE if args.fine else CONFIG, workers=args.workers)
