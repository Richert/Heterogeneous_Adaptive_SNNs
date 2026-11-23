r"""
Frequency shift induced by fluctuating coupling weights in the asynchronous state
==================================================================================

For freely rotating oscillators (the assumption behind the C_g closure), the weight fluctuations
δA_ij(t) = μ ∫ e^{−γτ} sin(φ_ij(t−τ) + β) dτ are correlated with the coupling term sin φ_ij(t), and
the time-averaged force shifts the frequency of oscillator i by

    δω_i = (K μ / 2) ⟨(γ cos β + Ω_ij sin β) / (γ² + Ω_ij²)⟩_j ,   Ω_ij = ω_j − ω_i,

which for Lorentzian frequencies evaluates to δω_i = (Kμ/2)[(Δ+γ)cos β − ω_i sin β] / [(Δ+γ)² + ω_i²]:
frequency compression (∝ sin β) and a forward push (∝ cos β). Uniform coupling alone also shifts the
frequencies in the asynchronous state (at O(K²), independent of μ), so every adaptive run is compared
with a μ = 0 control run with identical frequencies and initial phases.

    python weight_variance_async_shift.py      # → weight_variance_async_shift.npz (~1.5 min)
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import time
import numpy as np

import weight_variance_rule_micro as base

CONFIG = dict(N=500, K=1.0, gamma=0.01, dt=0.05, T=5000.0, Tw=2000.0, Delta=1.0, mu=0.04,
              betas={"cos": np.pi / 2, "pi4": np.pi / 4, "sin": 0.0}, seeds=[11, 12],
              out=dp.mpmf("weight_variance_async_shift.npz"))


def shift_theory(omega, K, mu, g, beta, Delta):
    """Finite-N sum and Lorentzian closed form of δω_i."""
    Om = omega[None, :] - omega[:, None]
    ker = (g * np.cos(beta) + Om * np.sin(beta)) / (g ** 2 + Om ** 2)
    np.fill_diagonal(ker, 0.0)
    finite = 0.5 * K * mu * ker.sum(axis=1) / omega.size
    cont = 0.5 * K * mu * ((Delta + g) * np.cos(beta) - omega * np.sin(beta)) / ((Delta + g) ** 2 + omega ** 2)
    return finite, cont


def mean_freqs(omega, mu, beta, seed, cfg):
    N, K, g, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["dt"]
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, N)
    A = np.ones((N, N))
    np.fill_diagonal(A, 0.0)
    sb, cb = np.sin(beta), np.cos(beta)
    mu_eff, c_cos = (mu * (sb + cb), sb / (sb + cb)) if mu > 0 else (0.0, 1.0)
    e0 = np.empty(0)
    n0, n1 = int((cfg["T"] - cfg["Tw"]) / dt), int(cfg["Tw"] / dt)
    rec = np.zeros(n0, np.bool_)
    base._simulate_numba_z(theta, A, omega, K, mu_eff, g, dt, n0, rec, c_cos, e0, e0, e0,
                           e0.astype(complex), e0.astype(complex))
    th0 = theta.copy()
    rec = np.zeros(n1, np.bool_)
    rec[::20] = True
    m = rec.sum()
    Rs, Ab, VA = np.empty(m), np.empty(m), np.empty(m)
    Z1, Z2 = np.empty(m, complex), np.empty(m, complex)
    base._simulate_numba_z(theta, A, omega, K, mu_eff, g, dt, n1, rec, c_cos, Rs, Ab, VA, Z1, Z2)
    return (theta - th0) / cfg["Tw"], Rs.mean()


def main(cfg=CONFIG):
    K, g, mu, D = cfg["K"], cfg["gamma"], cfg["mu"], cfg["Delta"]
    omega = base.lorentzian_quantiles(cfg["N"], D)
    out = dict(omega=omega, Delta=D, mu=mu, gamma=g, K=K)
    for s, seed in enumerate(cfg["seeds"]):
        t0 = time.time()
        w0, R0 = mean_freqs(omega, 0.0, 0.0, seed, cfg)
        out[f"w_ctrl_{s}"] = w0
        print(f"seed {seed}: μ=0 control R={R0:.3f} ({time.time() - t0:.0f}s)", flush=True)
        for name, beta in cfg["betas"].items():
            t0 = time.time()
            w, R = mean_freqs(omega, mu, beta, seed, cfg)
            out[f"w_{name}_{s}"] = w
            fin, cont = shift_theory(omega, K, mu, g, beta, D)
            out[f"th_{name}"], out[f"thc_{name}"] = fin, cont
            dw = w - w0
            core = np.abs(omega) < 3.0
            print(f"seed {seed} {name} (β={beta:.3f}): R={R:.3f} | mean δω meas {dw.mean():+.5f} vs theory "
                  f"{fin.mean():+.5f} (cont. ⟨δω⟩ = Kμcosβ/(2(2Δ+γ)) = {K * mu * np.cos(beta) / (2 * (2 * D + g)):+.5f}) | "
                  f"corr(meas, theory) on |ω|<3: {np.corrcoef(dw[core], fin[core])[0, 1]:.3f} | "
                  f"slope {np.polyfit(fin[core], dw[core], 1)[0]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(cfg["out"], **out)
    print(f"saved {cfg['out']}")


if __name__ == "__main__":
    main()
