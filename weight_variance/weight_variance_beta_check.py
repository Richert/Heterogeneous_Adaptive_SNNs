r"""
Check of the general-β theory at an intermediate adaptation phase lag, G_β(φ) = sin(φ + β)
=========================================================================================

β = π/2 and β = 0 are the symmetric (cos) and antisymmetric (sin) rules of the manuscript. At an
intermediate β, the drive contains both a pull (∝ sin β) and a torque (∝ cos β). For each (μ, Δ),
the N = 500 network (Euler kernel of weight_variance_rule_micro.py with the mixed drive
μ(sin β cos φ + cos β sin φ), uniform-weight MF-fixed-point init) is compared with
  * the mean-field closure for general β (Ā = 1 + κR² sin β, V_A of Eq. va_ss_full), and
  * the quasi-static self-consistency for general β (R, Ω, Ā, V_A/Ā² with self-consistent S, W).

    python weight_variance_beta_check.py      # → weight_variance_beta_check.npz (~2 min)
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
import weight_variance_rule_micro_oainit as oai

CONFIG = dict(N=500, K=1.0, gamma=0.01, dt=0.05, T=5000.0, rec_every=10, seed=3,
              beta=np.pi / 4, mus=[0.003, 0.01, 0.02], deltas=[0.1, 0.3, 0.6],
              out=dp.mpmf("weight_variance_beta_check.npz"))


def network(D, mu, beta, cfg, rng):
    N, K, g, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["dt"]
    omega = base.lorentzian_quantiles(N, D)
    mf = qs.meanfield_beta(D, mu, g, K, beta)
    if mf is None:                                                 # asynchronous initial state
        theta, A0 = rng.uniform(-np.pi, np.pi, N), 1.0
    else:
        theta, A0 = oai.oa_phases(omega, mf["R"], mf["Abar"], K, rng), mf["Abar"]
    A = np.full((N, N), A0)
    np.fill_diagonal(A, 0.0)
    sb, cb = np.sin(beta), np.cos(beta)
    mu_eff, c_cos = mu * (sb + cb), sb / (sb + cb)               # μ(sb cos + cb sin) as one drive
    n = int(cfg["T"] / dt)
    rec = np.zeros(n, np.bool_)
    rec[int(0.8 * n)::cfg["rec_every"]] = True
    m = rec.sum()
    Rs, Ab, VA = np.empty(m), np.empty(m), np.empty(m)
    Z1, Z2 = np.empty(m, complex), np.empty(m, complex)
    base._simulate_numba_z(np.ascontiguousarray(theta, float), A, omega, K, mu_eff, g, dt, n, rec,
                           c_cos, Rs, Ab, VA, Z1, Z2)
    t = np.arange(m) * dt * cfg["rec_every"]
    Om = np.polyfit(t, np.unwrap(np.angle(Z1)), 1)[0]
    return dict(R=Rs.mean(), Omega=Om, Abar=Ab.mean(), VArel=(VA / Ab ** 2).mean(),
                oa_dist=np.abs(Z2 - Z1 ** 2).mean()), mf, omega


def main(cfg=CONFIG):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    K, g, beta = cfg["K"], cfg["gamma"], cfg["beta"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for mu in cfg["mus"]:
        kap = mu / g
        for D in cfg["deltas"]:
            t0 = time.time()
            net, mf, omega = network(D, mu, beta, cfg, rng)
            R0 = mf["R"] if mf is not None else max(net["R"], 0.3)
            st = qs.solve_beta(omega, kap, K, beta, (R0, 0.0, complex(R0 ** 2)))
            q = (dict(R=st["R"], Omega=st["Omega"], **qs.weight_stats_beta(st, kap, mu, g, D, beta))
                 if st is not None else dict(R=np.nan, Omega=np.nan, Abar=np.nan, VArel=np.nan))
            m = mf if mf is not None else dict(R=0.0, Abar=1.0, VArel=np.nan)
            rows.append([mu, D, net["R"], net["Omega"], net["Abar"], net["VArel"], net["oa_dist"],
                         m["R"], m["Abar"], m["VArel"], q["R"], q["Omega"], q["Abar"], q["VArel"]])
            print(f"μ={mu} κ={kap:g} Δ={D}: R net/MF/QS {net['R']:.3f}/{m['R']:.3f}/{q['R']:.3f} | "
                  f"Ω net/QS {net['Omega']:+.4f}/{q['Omega']:+.4f} | Ā net/MF/QS {net['Abar']:.3f}/"
                  f"{m['Abar']:.3f}/{q['Abar']:.3f} | VA/Ā² net/MF/QS {net['VArel']:.3g}/{m['VArel']:.3g}/"
                  f"{q['VArel']:.3g} ({time.time() - t0:.0f}s)", flush=True)
    cols = ["mu", "delta", "net_R", "net_Omega", "net_Abar", "net_VArel", "net_oa_dist",
            "mf_R", "mf_Abar", "mf_VArel", "qs_R", "qs_Omega", "qs_Abar", "qs_VArel"]
    rows = np.array(rows)
    np.savez(cfg["out"], beta=beta, **{c: rows[:, k] for k, c in enumerate(cols)})
    print(f"saved {cfg['out']}")


if __name__ == "__main__":
    main()
