r"""
Adaptive-coupling Kuramoto: K-ramp / hysteresis simulations (data generation)
=============================================================================

Runs slow K-ramp simulations of the microscopic adaptive Kuramoto network (N×N) and of the reduced
mean-field system (PRL_2026 "Weight Variance"; R,Ā via Eqs. 8, 10 and V_A via Eqs. 71–75), for
μ ∈ {0.001, 0.01} at fixed Δ=1, γ=0.001, forward and backward.  K is stepped through the values
np.arange(0.5, 3.0, 0.3), holding each for `tau_d`, state carried across steps to expose hysteresis:
  * FORWARD ramp  IC: uniform phases (R≈0) and identical weights A_ij = 1;
  * BACKWARD ramp IC: identical phases (R=1) and weights A_ij = Ā* (the mean-field steady-state
    coupling at the starting/highest K).

Saves one .npz (R(t) traces for micro & mean field, the mean-field V_A/Ā²(t), per-step ramp
measurements + parameters) for the plotting script ``weight_variance_ramp_figure.py``.
NB: heavy one-time run (one N×N micro ramp per μ and direction).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp.py
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

from weight_variance_meanfield import order_parameter_S            # on-manifold S = ⟨|c|²⟩

CONFIG = dict(
    gamma=0.001, delta=1.0,
    mus=[0.001, 0.01],                 # monostable/cusp (μ=γ) and bistable (μ>γ)
    k_ramp=(0.5, 3.0, 0.3),            # microscopic ramp K values: np.arange(start, stop, step)
    k_min=0.3, k_max=3.0,              # K-range for the analytic branches (figure only)
    tau_d=5000.0,                      # dwell time per K step (≳ 1/γ for quasi-static tracking)
    N=500, dt=0.05, dts=2.0, meas_frac=0.3, seed=1,
    out_dir="/home/rgast/data/kmo_adaptive",
    out_name="weight_variance_ramp.npz",
)


def _Ksteps(cfg, direction):
    Ks = np.arange(*cfg["k_ramp"])
    return Ks[::-1] if direction == "bwd" else Ks


def _abar_ss(K, mu, g, delta):
    """Upper (stable) synchronous-branch coupling Ā* at coupling K (Eq. 81); 1 if no sync root."""
    b = 1.0 + mu / g
    disc = 0.25 * b ** 2 - 2.0 * mu * delta / (g * K)
    return 0.5 * b + np.sqrt(disc) if disc > 0 else 1.0


# ════════════════════════════════════════════════════════════════════════════
#  microscopic K-ramp: full N×N adaptive network (Euler), state carried across steps
# ════════════════════════════════════════════════════════════════════════════
def micro_ramp(mu, direction, cfg):
    N, g, delta, dt = cfg["N"], cfg["gamma"], cfg["delta"], cfg["dt"]
    n_hold = int(cfg["tau_d"] / dt)
    rec = max(1, int(cfg["dts"] / dt))
    m0 = int((1.0 - cfg["meas_frac"]) * n_hold)
    rng = np.random.default_rng(cfg["seed"])
    p = (np.arange(N) + 0.5) / N
    omega = delta * np.tan(np.pi * (p - 0.5))
    Ksteps = _Ksteps(cfg, direction)
    if direction == "fwd":                                # uniform phases (R≈0), A_ij = 1
        theta = rng.uniform(0.0, 2 * np.pi, N)
        A = np.ones((N, N))
    else:                                                 # identical phases (R=1), A_ij = Ā*(K_start)
        theta = np.zeros(N)
        A = np.full((N, N), _abar_ss(Ksteps[0], mu, g, delta))
    tcur, T, Rt, Rpts = 0.0, [], [], []
    for K in Ksteps:
        KinvN, acc = K / N, []
        for k in range(n_hold):
            e = np.exp(1j * theta)
            r = np.abs(e.mean())
            if k % rec == 0:
                T.append(tcur); Rt.append(r)
            if k >= m0:
                acc.append(r)
            theta = theta + dt * (omega + KinvN * np.imag(np.conj(e) * (A @ e)))
            A = A + dt * (mu * np.real(np.conj(e)[:, None] * e[None, :]) + g * (1.0 - A))
            tcur += dt
        Rpts.append(float(np.mean(acc)))
    return np.array(T), np.array(Rt), np.array(Rpts)


# ════════════════════════════════════════════════════════════════════════════
#  mean-field K-ramp: Eqs. 8, 10 (R, Ā) + 71–75 for V_A/Ā²; state carried across steps
# ════════════════════════════════════════════════════════════════════════════
def mf_ramp(mu, direction, cfg):
    g, delta = cfg["gamma"], cfg["delta"]
    te = np.arange(0.0, cfg["tau_d"], cfg["dts"])
    Ksteps = _Ksteps(cfg, direction)
    if direction == "fwd":
        y = [0.01, 1.0, 0.0, 0.0, 0.0]                    # R≈0, Ā=1
    else:
        y = [1.0, _abar_ss(Ksteps[0], mu, g, delta), 0.0, 0.0, 0.0]   # R=1, Ā=Ā*(K_start)

    def rhs(t, y, K):
        R, A, CS, CF, V = y
        S = float(order_parameter_S(R, A, K, delta))
        sS2, sF2 = 0.5 * (S ** 2 - R ** 4), 0.5 * (1.0 - S ** 2)
        return [-delta * R + (K * A / 2.0) * R * (1.0 - R ** 2),      # Eq. 8
                mu * R ** 2 + g * (1.0 - A),                          # Eq. 10
                -g * CS + mu * sS2,                                   # Eq. 71
                -(g + 2.0 * delta) * CF + mu * sF2,                   # Eq. 72
                2.0 * mu * (CS + CF) - 2.0 * g * V]                   # Eq. 73

    tcur, T, Rt, ratio, Rpts = 0.0, [], [], [], []
    for K in Ksteps:
        sol = solve_ivp(lambda t, yy: rhs(t, yy, K), (0.0, cfg["tau_d"]), y, t_eval=te,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=1.0)
        R, A, CS, CF, V = sol.y
        T.append(tcur + te); Rt.append(R); ratio.append(V / A ** 2)
        Rpts.append(float(R[-1])); y = list(sol.y[:, -1]); tcur += cfg["tau_d"]
    return np.concatenate(T), np.concatenate(Rt), np.concatenate(ratio), np.array(Rpts)


# ════════════════════════════════════════════════════════════════════════════
#  run all ramps and save
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    mus = cfg["mus"]
    Ks = np.arange(*cfg["k_ramp"])                                 # ascending ramp K values
    n_ramp = Ks.size
    print(f"K-ramp simulations — Δ={cfg['delta']}, γ={cfg['gamma']}, μ={mus}, "
          f"N={cfg['N']}, {n_ramp} K-steps ∈ {np.round(Ks, 2)} × τ_d={cfg['tau_d']}")

    Tm = Tf = Rm = Rf = ratio = Rp_m = Rp_f = None
    for i, mu in enumerate(mus):
        for di, d in enumerate(("fwd", "bwd")):
            print(f"  μ={mu} {d}: micro + mean field ...")
            tm, rm, rpm = micro_ramp(mu, d, cfg)
            tf, rf, rat, rpf = mf_ramp(mu, d, cfg)
            if Rm is None:
                Tm, Tf = tm, tf
                Rm = np.full((len(mus), 2, tm.size), np.nan)
                Rf = np.full((len(mus), 2, tf.size), np.nan)
                ratio = np.full((len(mus), 2, tf.size), np.nan)
                Rp_m = np.full((len(mus), 2, n_ramp), np.nan)
                Rp_f = np.full((len(mus), 2, n_ramp), np.nan)
            Rm[i, di], Rf[i, di], ratio[i, di] = rm, rf, rat
            Rp_m[i, di] = rpm if d == "fwd" else rpm[::-1]         # store in ascending-K order
            Rp_f[i, di] = rpf if d == "fwd" else rpf[::-1]

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], cfg["out_name"])
    np.savez(out, mus=np.asarray(mus, float), Ks=Ks, n_ramp=n_ramp, Tm=Tm, Tf=Tf,
             Rm=Rm, Rf=Rf, ratio=ratio, Rp_m=Rp_m, Rp_f=Rp_f,
             gamma=cfg["gamma"], delta=cfg["delta"], k_min=cfg["k_min"], k_max=cfg["k_max"],
             tau_d=cfg["tau_d"], N=cfg["N"], dt=cfg["dt"], dts=cfg["dts"], seed=cfg["seed"])
    print(f"[saved] {out}  (Rm {Rm.shape}, directions 0=fwd/1=bwd)")


if __name__ == "__main__":
    main()
