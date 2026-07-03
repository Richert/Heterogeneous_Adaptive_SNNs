r"""
Adaptive-coupling Kuramoto: V_A(t) micro-vs-mean-field sweep over (K, μ)
=======================================================================

Focused sweep comparing the microscopic weight-variance V_A(t) against the mean-field V_A(t) of the
PRL_2026 "Weight Variance" closure, at fixed Δ=1 and γ=0.001, over a densely sampled global coupling
K and μ ∈ {0.001, 0.003, 0.01}.  For every (K, μ) the mean field is run in TWO versions:
    * "full" — C_A = C_S + C_F  (Eqs. 71+72),
    * "cs"   — C_A = C_S only   (C_F dropped),
both driven by R(t) taken from the microscopic simulation.  The micro simulation, the fixed
order-parameter S(x=b/Δ) table, and the MF closure are reused from ``weight_variance_meanfield``.

Saves one .npz (V_A(t) traces for micro + both MF versions, plus parameters) for a separate
plotting script.  NB: this is a heavy one-time run (one N×N micro sim per (K, μ)).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_VA_sweep.py
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

import weight_variance_meanfield as W          # reuse simulate_micro + order_parameter_S (+ S table)

CONFIG = dict(
    delta=1.0, gamma=0.001,                     # fixed
    mus=[0.001, 0.003, 0.01],                   # one sweep per μ
    k_min=1.0, k_max=4.0, n_k=15,               # densely sampled global coupling K
    # microscopic network + integration (per (K, μ))
    N=500, T=5000.0, dt=0.05, dts=1.0, sigma0=0.3, seed=1,
    out_dir="/home/richard/data/kmo_adaptive",    # ("/home/data/kmo_adaptive" → /home/rgast/data/...)
    out_name="weight_variance_VA_sweep.npz",
)


def simulate_meanfield_VA(cfg, t_micro, R_micro, use_CF):
    """Mean-field V_A(t) (Eqs. 10, 71–75; fixed S) driven by micro R(t).
    use_CF=False holds C_F at 0, i.e. C_A = C_S. Returns the V_A trace."""
    K, g, mu, delta = cfg["K"], cfg["gamma"], cfg["mu"], cfg["delta"]
    cF_on = 1.0 if use_CF else 0.0

    def rhs(t, y):
        A, CS, CF, V = y
        R = np.interp(t, t_micro, R_micro)
        S = W.order_parameter_S(R, A, K, delta)
        sS2 = 0.5 * (S ** 2 - R ** 4)                        # Eq. 74
        sF2 = 0.5 * (1.0 - S ** 2)                           # Eq. 75
        dA = mu * R ** 2 + g * (1.0 - A)                     # Eq. 10
        dCS = -g * CS + mu * sS2                             # Eq. 71
        dCF = cF_on * (-(g + 2.0 * delta) * CF + mu * sF2)   # Eq. 72 (0 if use_CF=False)
        dV = 2.0 * mu * (CS + CF) - 2.0 * g * V              # Eq. 73
        return [dA, dCS, dCF, dV]

    sol = solve_ivp(rhs, (t_micro[0], t_micro[-1]), [1.0, 0.0, 0.0, 0.0], t_eval=t_micro,
                    method="RK45", rtol=1e-7, atol=1e-9, max_step=cfg["dt"])
    return sol.y[3]


def main(cfg=CONFIG):
    Ks = np.linspace(cfg["k_min"], cfg["k_max"], cfg["n_k"])
    mus = np.asarray(cfg["mus"], float)
    n_mu, n_K = len(mus), len(Ks)
    print(f"V_A(t) sweep — Δ={cfg['delta']}, γ={cfg['gamma']}, μ={list(mus)}, "
          f"K∈[{cfg['k_min']},{cfg['k_max']}] ({n_K} pts) → {n_mu * n_K} micro sims")

    t_ref = None
    VA_micro = VA_full = VA_cs = None
    for im, mu in enumerate(mus):
        for ik, K in enumerate(Ks):
            rc = dict(cfg, mu=float(mu), K=float(K))
            mic = W.simulate_micro(rc)
            va_full = simulate_meanfield_VA(rc, mic["t"], mic["R"], use_CF=True)
            va_cs = simulate_meanfield_VA(rc, mic["t"], mic["R"], use_CF=False)
            if t_ref is None:                                # allocate on first run
                t_ref = mic["t"]
                shape = (n_mu, n_K, t_ref.size)
                VA_micro = np.full(shape, np.nan)
                VA_full = np.full(shape, np.nan)
                VA_cs = np.full(shape, np.nan)
            VA_micro[im, ik] = mic["VA"]
            VA_full[im, ik] = va_full
            VA_cs[im, ik] = va_cs
            print(f"  μ={mu:<6} K={K:5.2f} -> V_A(end) micro={mic['VA'][-1]:.4f} "
                  f"full={va_full[-1]:.4f} cs={va_cs[-1]:.4f}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], cfg["out_name"])
    np.savez(out, t=t_ref, Ks=Ks, mus=mus,
             VA_micro=VA_micro, VA_full=VA_full, VA_cs=VA_cs,
             delta=cfg["delta"], gamma=cfg["gamma"],
             N=cfg["N"], T=cfg["T"], dt=cfg["dt"], dts=cfg["dts"],
             sigma0=cfg["sigma0"], seed=cfg["seed"])
    print(f"[saved] {out}  (VA arrays {VA_micro.shape})")


if __name__ == "__main__":
    main()
