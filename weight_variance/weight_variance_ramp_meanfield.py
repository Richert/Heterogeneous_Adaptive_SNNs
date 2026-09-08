r"""
Adaptive-coupling Kuramoto: K-ramp mean-field simulations (data generation)
==========================================================================

Slow K-ramp of the reduced mean-field system (PRL_2026 "Weight Variance"; R,Ā via Eqs. 8, 10 and
V_A via Eqs. 71–75), matching the microscopic ramp in ``weight_variance_ramp_micro.py`` (shared
parameters/helpers imported from there).  Forward and backward, per μ, state carried across steps:
  * FORWARD  IC: R≈0, Ā=1;   BACKWARD IC: R=1, Ā=Ā*(K_start).
The mean-field R is re-seeded to ≥ 1/√N at the start of each K-hold (`mf_seed_floor`): the noiseless
MF has R=0 as an invariant, so without this it stays stuck on the async branch even once K>2Δ; the
finite-size incoherent level 1/√N supplies the physical ignition seed the deterministic MF lacks.

Cheap (ODE integration only).  Saves one .npz (mean-field R(t), V_A(t), C_A(t), Ā(t), per-step
coherence + parameters) for the plotting scripts (scale-free ratios V_A/Ā², C_A/Ā² formed downstream).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp_meanfield.py
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

from weight_variance_meanfield import order_parameter_S            # on-manifold S = ⟨|c|²⟩
import weight_variance_ramp_micro as M                             # shared params + _Ksteps + _abar_ss

CONFIG = dict(M.CONFIG,                                            # inherit the shared ramp parameters
              mf_seed_floor=True,                                  # re-seed MF R ≥ 1/√N per step
              out_name="weight_variance_ramp_meanfield.npz")


# ════════════════════════════════════════════════════════════════════════════
#  mean-field K-ramp: Eqs. 8, 10 (R, Ā) + 71–75 for V_A/Ā²; state carried across steps
# ════════════════════════════════════════════════════════════════════════════
def mf_ramp(mu, direction, cfg):
    g, delta = cfg["gamma"], cfg["delta"]
    te = np.arange(0.0, cfg["tau_d"], cfg["dts"])
    Ksteps = M._Ksteps(cfg, direction)
    if direction == "fwd":
        y = [0.01, 1.0, 0.0, 0.0, 0.0]                    # R≈0, Ā=1
    else:
        y = [1.0, M._abar_ss(Ksteps[0], mu, g, delta), 0.0, 0.0, 0.0]   # R=1, Ā=Ā*(K_start)

    def rhs(t, y, K):
        R, A, CS, CF, V = y
        S = float(order_parameter_S(R, A, K, delta))
        sS2, sF2 = 0.5 * (S ** 2 - R ** 4), 0.5 * (1.0 - S ** 2)
        return [-delta * R + (K * A / 2.0) * R * (1.0 - R ** 2),      # Eq. 8
                mu * R ** 2 + g * (1.0 - A),                          # Eq. 10
                -g * CS + mu * sS2,                                   # Eq. 71
                -(g + 2.0 * delta) * CF + mu * sF2,                   # Eq. 72
                2.0 * mu * (CS + CF) - 2.0 * g * V]                   # Eq. 73

    R_floor = 1.0 / np.sqrt(cfg["N"]) if cfg.get("mf_seed_floor", True) else 0.0
    tcur, T, Rt, VA, CA, Ab, Rpts = 0.0, [], [], [], [], [], []
    for K in Ksteps:
        y[0] = max(y[0], R_floor)                         # finite-size coherence seed (see docstring)
        sol = solve_ivp(lambda t, yy: rhs(t, yy, K), (0.0, cfg["tau_d"]), y, t_eval=te,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=1.0)
        R, A, CS, CF, V = sol.y
        T.append(tcur + te); Rt.append(R)
        VA.append(V); CA.append(CS + CF); Ab.append(A)                # V_A, C_A, Ā
        Rpts.append(float(R[-1])); y = list(sol.y[:, -1]); tcur += cfg["tau_d"]
    return dict(T=np.concatenate(T), R=np.concatenate(Rt), VA=np.concatenate(VA),
                CA=np.concatenate(CA), A=np.concatenate(Ab), Rpts=np.array(Rpts))


def main(cfg=CONFIG):
    mus = cfg["mus"]
    Ks = np.arange(*cfg["k_ramp"])
    n_ramp = Ks.size
    print(f"mean-field K-ramp — Δ={cfg['delta']}, γ={cfg['gamma']}, μ={mus}, "
          f"{n_ramp} K-steps × τ_d={cfg['tau_d']} (mf_seed_floor={cfg['mf_seed_floor']})")

    Tf = Rf = VA = CA = A = Rp_f = None
    for i, mu in enumerate(mus):
        for di, d in enumerate(("fwd", "bwd")):
            print(f"  μ={mu} {d} ...")
            m = mf_ramp(mu, d, cfg)
            if Rf is None:
                Tf = m["T"]
                shape = (len(mus), 2, Tf.size)
                Rf, VA, CA, A = (np.full(shape, np.nan) for _ in range(4))
                Rp_f = np.full((len(mus), 2, n_ramp), np.nan)
            Rf[i, di], VA[i, di], CA[i, di], A[i, di] = m["R"], m["VA"], m["CA"], m["A"]
            Rp_f[i, di] = m["Rpts"] if d == "fwd" else m["Rpts"][::-1]   # ascending-K order

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], cfg["out_name"])
    np.savez(out, mus=np.asarray(mus, float), Ks=Ks, n_ramp=n_ramp, Tf=Tf,
             Rf=Rf, VA=VA, CA=CA, A=A, Rp_f=Rp_f,
             gamma=cfg["gamma"], delta=cfg["delta"], k_min=cfg["k_min"], k_max=cfg["k_max"],
             tau_d=cfg["tau_d"], N=cfg["N"], dts=cfg["dts"], seed=cfg["seed"],
             mf_seed_floor=cfg["mf_seed_floor"])
    print(f"[saved] {out}  (Rf {Rf.shape}, directions 0=fwd/1=bwd)")


if __name__ == "__main__":
    main()
