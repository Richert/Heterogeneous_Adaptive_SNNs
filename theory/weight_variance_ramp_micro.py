r"""
Adaptive-coupling Kuramoto: K-ramp microscopic simulations (data generation)
============================================================================

Slow K-ramp simulations of the microscopic adaptive Kuramoto network (full N×N), for μ ∈ {0.001,
0.005} at fixed Δ=1, γ=0.001, forward and backward.  K is stepped through np.arange(0.5, 3.0, 0.3),
holding each for `tau_d`, with the network state carried across steps to expose hysteresis:
  * FORWARD ramp  IC: uniform phases (R≈0) and identical weights A_ij = 1;
  * BACKWARD ramp IC: identical phases (R=1) and weights A_ij = Ā* (mean-field steady-state coupling
    at the starting/highest K).

Saves one .npz (micro R(t) traces + per-step coherence estimates + parameters) for the plotting
scripts.  This is the HEAVY part of the ramp pipeline (one N×N ramp per μ and direction); the cheap
mean-field ramp is generated separately by ``weight_variance_ramp_meanfield.py`` (which imports the
shared parameters and helpers from here).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp_micro.py
"""
import os
import numpy as np

CONFIG = dict(
    gamma=0.001, delta=1.0,
    mus=[0.001, 0.005],                # monostable/cusp (μ=γ) and bistable (μ>γ)
    k_ramp=(0.5, 3.0, 0.3),            # ramp K values: np.arange(start, stop, step)
    k_min=0.3, k_max=3.0,              # K-range for the analytic branches (figures only)
    tau_d=5000.0,                      # dwell time per K step (≳ 1/γ for quasi-static tracking)
    N=500, dt=0.05, dts=2.0, meas_frac=0.3, seed=1,
    out_dir="/home/rgast/data/kmo_adaptive",
    out_name="weight_variance_ramp_micro.npz",
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


def main(cfg=CONFIG):
    mus = cfg["mus"]
    Ks = np.arange(*cfg["k_ramp"])
    n_ramp = Ks.size
    print(f"micro K-ramp — Δ={cfg['delta']}, γ={cfg['gamma']}, μ={mus}, N={cfg['N']}, "
          f"{n_ramp} K-steps ∈ {np.round(Ks, 2)} × τ_d={cfg['tau_d']}")

    Tm = Rm = Rp_m = None
    for i, mu in enumerate(mus):
        for di, d in enumerate(("fwd", "bwd")):
            print(f"  μ={mu} {d} ...")
            tm, rm, rpm = micro_ramp(mu, d, cfg)
            if Rm is None:
                Tm = tm
                Rm = np.full((len(mus), 2, tm.size), np.nan)
                Rp_m = np.full((len(mus), 2, n_ramp), np.nan)
            Rm[i, di] = rm
            Rp_m[i, di] = rpm if d == "fwd" else rpm[::-1]         # store in ascending-K order

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], cfg["out_name"])
    np.savez(out, mus=np.asarray(mus, float), Ks=Ks, n_ramp=n_ramp, Tm=Tm, Rm=Rm, Rp_m=Rp_m,
             gamma=cfg["gamma"], delta=cfg["delta"], k_min=cfg["k_min"], k_max=cfg["k_max"],
             tau_d=cfg["tau_d"], N=cfg["N"], dt=cfg["dt"], dts=cfg["dts"], seed=cfg["seed"])
    print(f"[saved] {out}  (Rm {Rm.shape}, directions 0=fwd/1=bwd)")


if __name__ == "__main__":
    main()
