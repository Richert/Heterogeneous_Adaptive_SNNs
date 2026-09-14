r"""
Adaptive-coupling Kuramoto: Δ-ramp mean-field simulations (data generation)
==========================================================================

Slow HETEROGENEITY ramp of the reduced mean-field system (PRL_2026 "Weight Variance"; R, Ā via
Eqs. 8, 10 and V_A via Eqs. 71–75), matching the microscopic ramp in
``weight_variance_ramp_micro.py`` (shared parameters/helpers imported from there):

    Ṙ   = −ΔR + (KĀ/2) R(1−R²)
    Ā̇   = μ R² G(0) + γ(1−Ā)                 G(0) = 1 (cos) / 0 (sin, odd rule ⇒ Ā → 1)
    Ċ_S = −γ C_S + μ σ_S²,   Ċ_F = −(γ+2Δ) C_F + μ σ_F²,   V̇_A = 2μ (C_S+C_F) − 2γ V_A

with σ_S² = ½(S²−R⁴), σ_F² = ½(1−S²) and S = ⟨|c|²⟩ (on-manifold, tabulated). Forward (Δ increasing)
and backward (Δ decreasing) per rule and μ, state carried across steps, ICs mirroring the micro ramp
(fwd: R=1, Ā=1; bwd: R≈0, Ā=1; both with V_A = C_S = C_F = 0 — uniform initial weights).

The mean-field R is re-seeded to ≥ 1/√N at the start of each Δ-hold (`mf_seed_floor`): the noiseless
MF has R=0 as an invariant, so on the backward ramp it would stay stuck on the async branch even once
Δ < K/2; the finite-size incoherent level 1/√N supplies the physical ignition seed it lacks.

Cheap (ODE integration only). Saves one .npz (R(t), Ā(t), V_A(t), C_A(t) + per-step values) for
``weight_variance_ramp_figure.py``; the scale-free ratio V_A/Ā² is formed downstream.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp_meanfield.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
# ---------------------------------------------------------------------------
import os
import numpy as np
from scipy.integrate import solve_ivp

from weight_variance_meanfield import order_parameter_S            # on-manifold S = ⟨|c|²⟩
import weight_variance_ramp_micro as M                             # shared params + _dsteps

CONFIG = dict(M.CONFIG,                                            # inherit the shared ramp parameters
              mf_seed_floor=True,                                  # re-seed MF R ≥ 1/√N per Δ step
              out_name="weight_variance_ramp_meanfield.npz")


# ════════════════════════════════════════════════════════════════════════════
#  mean-field Δ-ramp: Eqs. 8, 10 (R, Ā) + 71–73 (C_S, C_F, V_A)
# ════════════════════════════════════════════════════════════════════════════
def mf_ramp(rule, mu, direction, cfg):
    g, K = cfg["gamma"], cfg["K"]
    G0 = 1.0 if rule == "cos" else 0.0                    # ⟨G⟩ at zero phase lag
    te = np.arange(0.0, cfg["tau_d"], cfg["dts"])
    R0 = 1.0 if direction == "fwd" else 0.01              # mirrors the micro IC
    y = [R0, 1.0, 0.0, 0.0, 0.0]                          # R, Ā=1, C_S, C_F, V_A (uniform weights)

    def rhs(t, y, delta):
        R, A, CS, CF, V = y
        Rc = min(max(R, 0.0), 1.0)
        S = float(order_parameter_S(Rc, A, K, delta))
        sS2, sF2 = 0.5 * (S ** 2 - Rc ** 4), 0.5 * (1.0 - S ** 2)
        return [-delta * Rc + (K * A / 2.0) * Rc * (1.0 - Rc ** 2),   # Eq. 8
                mu * Rc ** 2 * G0 + g * (1.0 - A),                    # Eq. 10
                -g * CS + mu * sS2,                                   # Eq. 71
                -(g + 2.0 * delta) * CF + mu * sF2,                   # Eq. 72
                2.0 * mu * (CS + CF) - 2.0 * g * V]                   # Eq. 73

    R_floor = 1.0 / np.sqrt(cfg["N"]) if cfg.get("mf_seed_floor", True) else 0.0
    tcur, T, Rt, VA, CA, Ab, Rpts, VRpts = 0.0, [], [], [], [], [], [], []
    m0 = int((1.0 - cfg["meas_frac"]) * te.size)
    for D in M._dsteps(cfg, direction):
        y[0] = max(y[0], R_floor)                         # finite-size coherence seed (see docstring)
        sol = solve_ivp(lambda t, yy: rhs(t, yy, D), (0.0, cfg["tau_d"]), y, t_eval=te,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=1.0)
        R, A, CS, CF, V = sol.y
        T.append(tcur + te); Rt.append(R); VA.append(V); CA.append(CS + CF); Ab.append(A)
        Rpts.append(float(R[m0:].mean()))
        VRpts.append(float((V[m0:] / A[m0:] ** 2).mean()))
        y = list(sol.y[:, -1]); tcur += cfg["tau_d"]
    return dict(T=np.concatenate(T), R=np.concatenate(Rt), VA=np.concatenate(VA),
                CA=np.concatenate(CA), A=np.concatenate(Ab),
                Rpts=np.array(Rpts), VRpts=np.array(VRpts))


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rules, mus = cfg["rules"], cfg["mus"]
    deltas = np.arange(*cfg["delta_ramp"])
    nr, nm, nd = len(rules), len(mus), deltas.size
    print(f"mean-field Δ-ramp — K={cfg['K']}, γ={cfg['gamma']}, μ={mus}, rules={rules}, "
          f"{nd} Δ-steps ∈ {np.round(deltas, 2)} × τ_d={cfg['tau_d']:g} "
          f"(mf_seed_floor={cfg['mf_seed_floor']})")

    Tf = Rf = VA = CA = A = Rp_f = VRp_f = None
    for ri, rule in enumerate(rules):
        for i, mu in enumerate(mus):
            for di, d in enumerate(("fwd", "bwd")):
                m = mf_ramp(rule, mu, d, cfg)
                if Rf is None:
                    Tf = m["T"]
                    Rf, VA, CA, A = (np.full((nr, nm, 2, Tf.size), np.nan) for _ in range(4))
                    Rp_f, VRp_f = (np.full((nr, nm, 2, nd), np.nan) for _ in range(2))
                Rf[ri, i, di], VA[ri, i, di] = m["R"], m["VA"]
                CA[ri, i, di], A[ri, i, di] = m["CA"], m["A"]
                asc = slice(None) if d == "fwd" else slice(None, None, -1)   # ascending in Δ
                Rp_f[ri, i, di], VRp_f[ri, i, di] = m["Rpts"][asc], m["VRpts"][asc]
                print(f"  G={rule:<4} μ={mu:<6g} {d}: R {m['R'][0]:.3f}→{m['R'][-1]:.3f}  "
                      f"Ā(end)={m['A'][-1]:.3f}  V_A/Ā²(end)={m['VA'][-1] / m['A'][-1] ** 2:.4f}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], cfg["out_name"])
    np.savez(out, rules=np.array(rules), mus=np.asarray(mus, float), deltas=deltas, n_ramp=nd,
             Tf=Tf, Rf=Rf, VA=VA, CA=CA, A=A, Rp_f=Rp_f, VRp_f=VRp_f,
             K=cfg["K"], gamma=cfg["gamma"], delta_min=cfg["delta_min"], delta_max=cfg["delta_max"],
             tau_d=cfg["tau_d"], N=cfg["N"], dts=cfg["dts"], seed=cfg["seed"],
             mf_seed_floor=cfg["mf_seed_floor"])
    print(f"[saved] {out}  (Rf {Rf.shape}, directions 0=fwd/1=bwd)")


if __name__ == "__main__":
    main()
