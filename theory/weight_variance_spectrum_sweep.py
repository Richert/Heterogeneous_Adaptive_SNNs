r"""
Adaptive-coupling Kuramoto: (K, μ) sweep of weight statistics + eigenvalue spectra
==================================================================================

Microscopic-simulation sweep over the global coupling K and adaptation rate μ (fixed Δ=1, γ=0.001)
of the adaptive Kuramoto network
    θ̇_i = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i),   Ȧ_ij = μ cos(θ_j − θ_i) + γ(1 − A_ij).
For every (K, μ) it records, per time step, the average coupling weight Ā, the weight variance V_A,
and the covariance C_A = Cov(A_ij, cos(θ_j−θ_i)) (all over off-diagonal pairs), and computes:
  * the eigenvalue spectrum of the (symmetric) coupling matrix A;
  * the eigenvalue spectrum of the "system dynamics" = eigenvalues of the covariance matrix of the
    wrapped phases s_i(t) = sin(θ_i(t)), evaluated over a time window of size `window`.

The `window` also sets the transient: statistics start at the first window (t ≥ window).  Windows of
size `window`, stepped forward by window/2 (50 % overlap), tile the post-transient run; Ā, V_A, C_A
are time-averaged within each window and then averaged over windows (the eigenvalue spectra are
window-level averages by construction — the dynamics spectrum via its covariance, the coupling
spectrum via A snapshotted at each window centre — and are likewise averaged over windows).

Saves one .npz (Ā, V_A, C_A, both eigenvalue spectra, and the participation ratios PR = (Σλ)²/Σλ²
of the phase dynamics and the coupling weights, per (K, μ)) for a downstream figure that plots
V_A/Ā² and C_A/Ā² against the participation ratios.  HEAVY (one N×N ramp-free run per (K, μ)).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_spectrum_sweep.py
"""
import os
import numpy as np

CONFIG = dict(
    gamma=0.001, delta=1.0,
    k_min=0.5, k_max=3.0, n_k=10,                   # K sweep (linear)
    mu_min=0.001, mu_max=0.01, n_mu=3, mu_log=True,  # μ sweep (log by default)
    window=500.0, T_total=5000.0,                   # covariance/averaging window; total sim time
    N=500, dt=0.05, dts=1.0, sigma0=0.3, seed=1,     # coherent IC θ_i(0) ~ N(0, sigma0)
    out_dir="/home/rgast/data/kmo_adaptive",
    out_name="weight_variance_spectrum_sweep.npz",
)


def _participation_ratio(ev):
    """PR = (Σλ)² / Σλ²  — effective number of modes of an eigenvalue spectrum."""
    s2 = float(np.sum(ev ** 2))
    return float(np.sum(ev) ** 2 / s2) if s2 > 0 else 0.0


def simulate_and_measure(K, mu, cfg):
    """Run the micro network; return window-averaged Ā, V_A, C_A, mean sorted eigenvalue spectra of
    the coupling matrix and the sin(θ) covariance, and the two participation ratios."""
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
    centres = set(int(s + W_rec // 2) for s in starts)          # A-snapshot record-indices

    S, Abar_t, VA_t, CA_t = [], [], [], []
    cpl_specs = []                                              # coupling eigenvalues at window centres
    jrec = 0
    for k in range(nsteps + 1):
        e = np.exp(1j * theta)
        G = np.real(np.conj(e)[:, None] * e[None, :])            # cos(θ_j − θ_i)  (symmetric)
        if k % rec == 0:
            dgA = np.diagonal(A)
            Abar = (A.sum() - dgA.sum()) / n_off
            VA = ((A * A).sum() - (dgA * dgA).sum()) / n_off - Abar ** 2
            Gbar = (G.sum() - N) / n_off
            AGbar = ((A * G).sum() - dgA.sum()) / n_off
            S.append(np.sin(theta)); Abar_t.append(Abar); VA_t.append(VA)
            CA_t.append(AGbar - Abar * Gbar)
            if jrec in centres:                                  # coupling spectrum at window centre
                cpl_specs.append(np.sort(np.linalg.eigvalsh(A))[::-1])
            jrec += 1
        if k == nsteps:
            break
        theta = theta + dt * (omega + KinvN * np.imag(np.conj(e) * (A @ e)))
        A = A + dt * (mu * G + g * (1.0 - A))

    S = np.asarray(S)                                            # (n_rec, N)
    Abar_t, VA_t, CA_t = map(np.asarray, (Abar_t, VA_t, CA_t))
    ab_w, va_w, ca_w, dyn_specs = [], [], [], []
    for j0 in starts:
        sl = slice(int(j0), int(j0) + W_rec)
        ab_w.append(Abar_t[sl].mean()); va_w.append(VA_t[sl].mean()); ca_w.append(CA_t[sl].mean())
        cov = np.cov(S[sl], rowvar=False)                        # N×N covariance of sin(θ)
        dyn_specs.append(np.sort(np.linalg.eigvalsh(cov))[::-1])
    dyn_specs = np.asarray(dyn_specs)                            # (n_win, N)
    cpl_specs = np.asarray(cpl_specs)                            # (n_win, N)

    return dict(
        Abar=float(np.mean(ab_w)), VA=float(np.mean(va_w)), CA=float(np.mean(ca_w)),
        dyn_spec=dyn_specs.mean(0), cpl_spec=cpl_specs.mean(0),
        PR_dyn=float(np.mean([_participation_ratio(s) for s in dyn_specs])),
        PR_cpl=float(np.mean([_participation_ratio(s) for s in cpl_specs])),
    )


def main(cfg=CONFIG):
    Ks = np.linspace(cfg["k_min"], cfg["k_max"], cfg["n_k"])
    if cfg["mu_log"]:
        mus = np.logspace(np.log10(cfg["mu_min"]), np.log10(cfg["mu_max"]), cfg["n_mu"])
    else:
        mus = np.linspace(cfg["mu_min"], cfg["mu_max"], cfg["n_mu"])
    N = cfg["N"]
    print(f"(K,μ) spectrum sweep — Δ={cfg['delta']}, γ={cfg['gamma']}, N={N}, "
          f"window={cfg['window']}, T_total={cfg['T_total']}, {len(mus)}×{len(Ks)} points")

    shape = (len(mus), len(Ks))
    Abar, VA, CA, PR_dyn, PR_cpl = (np.full(shape, np.nan) for _ in range(5))
    dyn_eigs = np.full(shape + (N,), np.nan)
    cpl_eigs = np.full(shape + (N,), np.nan)
    for i, mu in enumerate(mus):
        for j, K in enumerate(Ks):
            r = simulate_and_measure(float(K), float(mu), cfg)
            Abar[i, j], VA[i, j], CA[i, j] = r["Abar"], r["VA"], r["CA"]
            PR_dyn[i, j], PR_cpl[i, j] = r["PR_dyn"], r["PR_cpl"]
            dyn_eigs[i, j], cpl_eigs[i, j] = r["dyn_spec"], r["cpl_spec"]
            print(f"  μ={mu:7.4f} K={K:5.2f} -> Ā={r['Abar']:.3f} V_A/Ā²={r['VA']/r['Abar']**2:.3f} "
                  f"C_A/Ā²={r['CA']/r['Abar']**2:.3f} PR_dyn={r['PR_dyn']:.1f} PR_cpl={r['PR_cpl']:.1f}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], cfg["out_name"])
    np.savez(out, Ks=Ks, mus=mus, Abar=Abar, VA=VA, CA=CA, PR_dyn=PR_dyn, PR_cpl=PR_cpl,
             dyn_eigs=dyn_eigs, cpl_eigs=cpl_eigs,
             gamma=cfg["gamma"], delta=cfg["delta"], N=N, window=cfg["window"],
             T_total=cfg["T_total"], dt=cfg["dt"], dts=cfg["dts"], sigma0=cfg["sigma0"], seed=cfg["seed"])
    print(f"[saved] {out}  (grid {shape}, spectra dim N={N})")


if __name__ == "__main__":
    main()
