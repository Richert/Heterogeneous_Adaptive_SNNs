r"""
Skardal-model benchmark — part 1: microscopic Kuramoto + Skardal mean field
===========================================================================

Benchmark of the Lorentzian-mixture ensemble approach against the low-dimensional
reduction of Skardal, Phys. Rev. E 98, 022207 (2018), for the family of RATIONAL
frequency distributions (Eq. 14):

    g_n(ω) = n·sin(π/2n)·Δ^{2n-1} / [ π (ω^{2n} + Δ^{2n}) ] ,   n = 1, 2, 3, ...

(n=1 = Cauchy; n→∞ = uniform on [−Δ,Δ]). g_n is rational with 2n simple poles on the
circle |ω|=Δ; the n poles in the lower half-plane,
    ω̂_k = Δ·exp(−i(2k+1)π/2n),  k = 0,…,n−1,
give an OA reduction of dimension n (complex). With residue weights
    c_k = −2πi·Res(g_n; ω̂_k) = i·sin(π/2n)·ω̂_k/Δ      (Σ_k c_k = 1),
the Skardal mean field is  α̇_k = −i ω̂_k α_k + (K/2)(z* − z α_k²),  z* = Σ_k c_k α_k,
z = conj(z*), and the order parameter is R(t) = |z|.  (For n=1 this is the usual
single-Lorentzian OA equation.)

This script, for each n: draws N microscopic frequencies from g_n, simulates the
microscopic Kuramoto network (same PyRates+solve_ivp model as kmo_lorentzian_fit_sweep)
and the Skardal mean field from a shared coherent initial condition, and saves the R(t)
dynamics, the frequency distribution (samples + analytic g_n), and the Skardal MF dimension.
Part 2 (skardal_benchmark_figure.py) fits the Lorentzian mixture to the saved samples,
simulates the ensemble model, and plots the three-way comparison.

Run in the ``pycobi`` conda env (dev PyRates 1.2.x):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python skardal_benchmark_simulate.py
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
gs_path = os.path.join(_HERE, "..", "grid_search")
sys.path.insert(0, gs_path)
import kmo_lorentzian_fit_sweep as KFS          # reuse simulate_micro (PyRates Kuramoto net)


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    n_exponents=[2, 20, 200],     # exponents n of Eq. 14 to benchmark
    Delta=1.0,                    # spread parameter of g_n
    K=1.2,                        # global coupling
    N=5000,                       # microscopic oscillators
    sigma0=0.5,                   # coherent initial condition θ_i(0) ~ N(0, σ0); z(0)=R0
    # integration (shared by micro / Skardal / ensemble; solve_ivp RK45)
    T=40.0, dt=1e-2, dts=0.1, rtol=1e-6, atol=1e-8,
    seed=1,
    out_dir="/home/rgast/data/mpmf_simulations",
    out_stem="skardal_benchmark_n",            # -> <out_dir>/skardal_benchmark_n<n>.npz
)


# ════════════════════════════════════════════════════════════════════════════
#  rational distribution g_n: sampling, density, Skardal poles
# ════════════════════════════════════════════════════════════════════════════
def gn_density(omega, n, Delta):
    return n * np.sin(np.pi / (2 * n)) * Delta ** (2 * n - 1) / (np.pi * (omega ** (2 * n) + Delta ** (2 * n)))


def sample_gn(n, Delta, N, rng):
    """Draw N i.i.d. frequencies from g_n (Eq. 14). n=1 is exact Cauchy; n≥2 via numeric
    inverse-CDF (g_n decays like ω^{-2n}, so a finite grid captures essentially all mass)."""
    if n == 1:
        return Delta * np.tan(np.pi * (rng.random(N) - 0.5))
    L = 14.0 * Delta
    grid = np.linspace(-L, L, 400001)
    g = gn_density(grid, n, Delta)
    cdf = np.cumsum(g); cdf /= cdf[-1]
    return np.interp(rng.random(N), cdf, grid)


def skardal_poles(n, Delta):
    """Lower-half-plane poles ω̂_k and residue weights c_k = −2πi·Res(g_n; ω̂_k)."""
    k = np.arange(n)
    omega_hat = Delta * np.exp(-1j * (2 * k + 1) * np.pi / (2 * n))
    c = 1j * np.sin(np.pi / (2 * n)) * omega_hat / Delta          # Σ_k c_k = 1
    return omega_hat, c


def simulate_skardal(n, Delta, K, R0, cfg, tag=""):
    """Skardal mean field: n complex OA equations at the lower-half poles. Returns t, R(t)."""
    omega_hat, c = skardal_poles(n, Delta)
    a0 = np.full(n, R0 + 0.0j)                                    # α_k(0)=R0 (coherent IC, z(0)=R0)

    def rhs(t, y):
        a = y[:n] + 1j * y[n:]
        zc = c @ a                                               # z* = Σ_k c_k α_k
        da = -1j * omega_hat * a + 0.5 * K * (zc - np.conj(zc) * a ** 2)
        return np.concatenate([da.real, da.imag])

    t_eval = np.arange(0.0, cfg["T"], cfg["dts"])
    y0 = np.concatenate([a0.real, a0.imag])
    sol = solve_ivp(rhs, (0.0, cfg["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["dt"])
    a = sol.y[:n] + 1j * sol.y[n:]
    R = np.abs(c @ a)                                            # |z| = |z*|
    return t_eval, R


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    Delta, K = cfg["Delta"], cfg["K"]
    gx = np.linspace(-5 * Delta, 5 * Delta, 1000)                # analytic g_n grid (for plotting)

    for n in cfg["n_exponents"]:
        omega = sample_gn(n, Delta, cfg["N"], rng)
        theta0 = rng.normal(0.0, cfg["sigma0"], cfg["N"])
        R0 = float(np.abs(np.exp(1j * theta0).mean()))
        print(f"n={n}: N={cfg['N']}, K={K}, Δ={Delta}, R(0)={R0:.3f}, "
              f"Skardal MF dim={n} (complex), ω∈[{omega.min():.1f},{omega.max():.1f}]")

        t_mic, R_mic = KFS.simulate_micro(omega, K, theta0, cfg, tag=f"sk_micro{n}")
        t_sk, R_sk = simulate_skardal(n, Delta, K, R0, cfg, tag=f"sk{n}")
        print(f"   micro R(end)={R_mic[-1]:.3f}   Skardal R(end)={R_sk[-1]:.3f}")

        out = os.path.join(cfg["out_dir"], f"{cfg['out_stem']}{n}.npz")
        np.savez(out,
                 n=np.int64(n), Delta=float(Delta), K=float(K), N=np.int64(cfg["N"]),
                 sigma0=float(cfg["sigma0"]), R0=float(R0),
                 T=float(cfg["T"]), dt=float(cfg["dt"]), dts=float(cfg["dts"]),
                 rtol=float(cfg["rtol"]), atol=float(cfg["atol"]),
                 t=t_mic, R_micro=R_mic, R_skardal=R_sk,
                 n_mf_skardal=np.int64(n),                       # # mean-field equations (complex)
                 omega=omega,                                    # frequency samples (for the fit)
                 g_omega=gx, g_density=gn_density(gx, n, Delta)) # analytic g_n
        print(f"   [saved] {os.path.basename(out)}")


if __name__ == "__main__":
    main()
