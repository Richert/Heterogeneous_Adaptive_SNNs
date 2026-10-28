r"""
Network validation of the quasi-static synchronized branch beyond the mean-field fold (G = cos)
==============================================================================================

For each μ and Δ between (just below) the mean-field fold and (just beyond) the quasi-static fold, the
N = 500 network is initialized ON the quasi-static state of weight_variance_cos_quasistatic.npz:
locked oscillators at their quasi-static phases, drifting oscillators drawn from the wrapped Cauchy
density with first moment c_i (which realizes c_n = c^n), and slaved weights A_ij = 1 + κ Re(c_j c_i*).
Beyond the quasi-static fold (no state to start from) the last available state is used.
The network is then integrated with the Euler kernel of weight_variance_rule_micro.py (dt = 0.05,
T = 5000) and R, V_A/Ā² and ⟨|Z_2 − Z²|⟩ are measured over the final 20 %.

    python weight_variance_cos_quasistatic_validate.py      # → ..._validate.npz (~8 min)
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import time, warnings
import numpy as np

import weight_variance_quasistatic as qs
import weight_variance_rule_micro as base
from weight_variance_bifurcation_rules import fold_point

CONFIG = dict(N=500, K=1.0, gamma=0.01, dt=0.05, T=5000.0, rec_every=10, seed=7,
              mus=[0.02, 0.04, 0.05], step=0.05, below=0.1, beyond=0.1,
              qs_npz=dp.mpmf("weight_variance_cos_quasistatic.npz"),
              out=dp.mpmf("weight_variance_cos_quasistatic_validate.npz"))


def qs_initial_state(omega, R, W, kap, K, rng):
    """Phases and weights realizing the quasi-static state (R, W)."""
    st = qs.state_RW(omega, R, W, kap, K)
    c = st["c"]
    theta = np.angle(c).copy()
    dr = ~st["locked"]
    r = np.clip(np.abs(c[dr]), 1e-9, 1 - 1e-9)                 # wrapped Cauchy: modulus r, mean angle
    u = rng.uniform(-np.pi, np.pi, dr.sum())
    theta[dr] = np.angle(c[dr]) + 2.0 * np.arctan(((1 - r) / (1 + r)) * np.tan(u / 2.0))
    A = 1.0 + kap * np.real(np.outer(np.conj(c), c))              # A_ij = 1 + κ Re(c_j c_i*)
    np.fill_diagonal(A, 0.0)
    return theta, np.ascontiguousarray(A)


def run(D, mu, R0, W0, cfg, rng):
    N, K, g, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["dt"]
    kap = mu / g
    omega = base.lorentzian_quantiles(N, D)
    theta, A = qs_initial_state(omega, R0, W0, kap, K, rng)
    n = int(cfg["T"] / dt)
    rec = np.zeros(n, np.bool_)
    rec[int(0.8 * n)::cfg["rec_every"]] = True
    m = rec.sum()
    Rs, Ab, VA = np.empty(m), np.empty(m), np.empty(m)
    Z1, Z2 = np.empty(m, complex), np.empty(m, complex)
    base._simulate_numba_z(theta, A, omega, K, mu, g, dt, n, rec, 1.0, Rs, Ab, VA, Z1, Z2)
    return Rs.mean(), (VA / Ab ** 2).mean(), Ab.mean(), np.abs(Z2 - Z1 ** 2).mean()


def main(cfg=CONFIG):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    q = np.load(cfg["qs_npz"])
    K, g = cfg["K"], cfg["gamma"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for mu in cfg["mus"]:
        m = int(np.argmin(np.abs(q["mus"] - mu)))
        ok = np.isfinite(q["R"][m])
        d_qs = q["deltas"][ok].max()
        d_mf = fold_point("cos", mu, g, K)[0]
        for D in np.round(np.arange(d_mf - cfg["below"], d_qs + cfg["beyond"] + 1e-9, cfg["step"]), 4):
            j = int(np.argmin(np.abs(np.where(ok, q["deltas"], np.inf) - min(D, d_qs))))
            t0 = time.time()
            R, rel, Ab, dist = run(D, mu, q["R"][m, j], q["W"][m, j], cfg, rng)
            qsR = q["R"][m, j] if D <= d_qs + 1e-9 else np.nan
            rows.append((mu, D, R, rel, Ab, dist, qsR, q["VArel"][m, j] if np.isfinite(qsR) else np.nan))
            print(f"μ={mu} Δ={D:.3f}: net R={R:.3f} VA/Ā²={rel:.3g} dist={dist:.3f} | QS R={qsR:.3f} "
                  f"VA/Ā²={rows[-1][-1]:.3g} | MF fold {d_mf:.3f}, QS fold {d_qs:.2f} ({time.time() - t0:.0f}s)",
                  flush=True)
    rows = np.array(rows)
    np.savez(cfg["out"], mu=rows[:, 0], delta=rows[:, 1], R=rows[:, 2], VArel=rows[:, 3], Abar=rows[:, 4],
             oa_dist=rows[:, 5], qs_R=rows[:, 6], qs_VArel=rows[:, 7], N=cfg["N"], T=cfg["T"])
    print(f"saved {cfg['out']}")


if __name__ == "__main__":
    main()
