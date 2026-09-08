r"""
Adaptive-coupling Kuramoto: (K, μ) sweep of phase coherence vs. weight statistics
=================================================================================

Microscopic-simulation sweep over the global coupling K and adaptation rate μ (fixed Δ=1, γ=0.001)
of the adaptive Kuramoto network
    θ̇_i = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i),   Ȧ_ij = μ cos(θ_j − θ_i) + γ(1 − A_ij).
For every (K, μ) it records, per time step, the phase coherence R = |⟨e^{iθ}⟩|, the average coupling
weight Ā, the weight variance V_A, and the covariance C_A = Cov(A_ij, cos(θ_j−θ_i)) (the latter three
over off-diagonal pairs).

A window of size `window` also sets the transient: statistics start at the first full window
(t ≥ window).  Windows of size `window`, stepped forward by window/2 (50 % overlap), tile the
post-transient run; R, Ā, V_A, C_A are time-averaged within each window and then averaged over
windows.

Saves one .npz (R, Ā, V_A, C_A per (K, μ), plus parameters) for a downstream figure that plots the
scale-free weight statistics V_A/Ā² and C_A/Ā² against the phase coherence R.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_coherence_sweep.py
"""
import os
import numpy as np

CONFIG = dict(
    gamma=0.001, delta=1.0,
    k_min=0.5, k_max=3.0, n_k=10,                   # K sweep (linear)
    mu_min=0.001, mu_max=0.01, n_mu=3, mu_log=True,  # μ sweep (log by default)
    window=500.0, T_total=5000.0,                   # averaging window; total sim time
    N=500, dt=0.05, dts=1.0, sigma0=0.3, seed=1,     # coherent IC θ_i(0) ~ N(0, sigma0)
    out_dir="/home/rgast/data/kmo_adaptive",
    out_name="weight_variance_coherence_sweep.npz",
)


def simulate_and_measure(K, mu, cfg):
    """Run the micro network; return window-averaged phase coherence R, average coupling weight Ā,
    weight variance V_A, and coupling–phase covariance C_A (means over 50 %-overlap windows)."""
    N, g, delta, dt = cfg["N"], cfg["gamma"], cfg["delta"], cfg["dt"]
    dts, W = cfg["dts"], cfg["window"]
    nsteps = int(cfg["T_total"] / dt)
    rec = max(1, int(round(dts / dt)))
    W_rec = int(round(W / dts))                      # records per window
    rng = np.random.default_rng(cfg["seed"])
    p = (np.arange(N) + 0.5) / N
    omega = delta * np.tan(np.pi * (p - 0.5))        # deterministic Lorentzian quantiles (ω̄=0)
    theta = rng.normal(0.0, cfg["sigma0"], N)        # coherent IC
    A = np.ones((N, N))
    KinvN, n_off = K / N, N * N - N

    n_rec = nsteps // rec + 1
    starts = np.arange(W_rec, n_rec - W_rec + 1, W_rec // 2)     # window start record-indices

    R_t, Abar_t, VA_t, CA_t = [], [], [], []
    for k in range(nsteps + 1):
        e = np.exp(1j * theta)
        G = np.real(np.conj(e)[:, None] * e[None, :])            # cos(θ_j − θ_i)  (symmetric)
        if k % rec == 0:
            dgA = np.diagonal(A)
            Abar = (A.sum() - dgA.sum()) / n_off
            VA = ((A * A).sum() - (dgA * dgA).sum()) / n_off - Abar ** 2
            Gbar = (G.sum() - N) / n_off
            AGbar = ((A * G).sum() - dgA.sum()) / n_off
            R_t.append(abs(e.mean())); Abar_t.append(Abar); VA_t.append(VA)
            CA_t.append(AGbar - Abar * Gbar)
        if k == nsteps:
            break
        theta = theta + dt * (omega + KinvN * np.imag(np.conj(e) * (A @ e)))
        A = A + dt * (mu * G + g * (1.0 - A))

    R_t, Abar_t, VA_t, CA_t = map(np.asarray, (R_t, Abar_t, VA_t, CA_t))
    r_w, ab_w, va_w, ca_w = [], [], [], []
    for j0 in starts:
        sl = slice(int(j0), int(j0) + W_rec)
        r_w.append(R_t[sl].mean()); ab_w.append(Abar_t[sl].mean())
        va_w.append(VA_t[sl].mean()); ca_w.append(CA_t[sl].mean())

    return dict(R=float(np.mean(r_w)), Abar=float(np.mean(ab_w)),
                VA=float(np.mean(va_w)), CA=float(np.mean(ca_w)))


def main(cfg=CONFIG):
    Ks = np.linspace(cfg["k_min"], cfg["k_max"], cfg["n_k"])
    if cfg["mu_log"]:
        mus = np.logspace(np.log10(cfg["mu_min"]), np.log10(cfg["mu_max"]), cfg["n_mu"])
    else:
        mus = np.linspace(cfg["mu_min"], cfg["mu_max"], cfg["n_mu"])
    print(f"(K,μ) coherence sweep — Δ={cfg['delta']}, γ={cfg['gamma']}, N={cfg['N']}, "
          f"window={cfg['window']}, T_total={cfg['T_total']}, {len(mus)}×{len(Ks)} points")

    shape = (len(mus), len(Ks))
    R, Abar, VA, CA = (np.full(shape, np.nan) for _ in range(4))
    for i, mu in enumerate(mus):
        for j, K in enumerate(Ks):
            r = simulate_and_measure(float(K), float(mu), cfg)
            R[i, j], Abar[i, j], VA[i, j], CA[i, j] = r["R"], r["Abar"], r["VA"], r["CA"]
            print(f"  μ={mu:7.4f} K={K:5.2f} -> R={r['R']:.3f} Ā={r['Abar']:.3f} "
                  f"V_A/Ā²={r['VA']/r['Abar']**2:.3f} C_A/Ā²={r['CA']/r['Abar']**2:.3f}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], cfg["out_name"])
    np.savez(out, Ks=Ks, mus=mus, R=R, Abar=Abar, VA=VA, CA=CA,
             gamma=cfg["gamma"], delta=cfg["delta"], N=cfg["N"], window=cfg["window"],
             T_total=cfg["T_total"], dt=cfg["dt"], dts=cfg["dts"], sigma0=cfg["sigma0"], seed=cfg["seed"])
    print(f"[saved] {out}  (grid {shape})")


if __name__ == "__main__":
    main()
