"""
Kuramoto Microscopic vs. Ott-Antonsen Macroscopic — Side-by-Side Comparison
=============================================================================
Runs two parallel simulations that share the same initial conditions:

  1. Microscopic (KM):  N individual Kuramoto oscillators with adaptive weights
         dθ_k/dt  = ω_k + (1/N) Σ_l A_kl sin(θ_l - θ_k)
         dA_kl/dt = μ cos(θ_l - θ_k) - γ A_kl

  2. Macroscopic (OA):  M = N/d populations, each on the OA manifold
         ṙ_I      = -Δ_I r_I + (1-r_I²)/2 · Σ_J A_IJ r_J cos(ψ_J-ψ_I)
         ψ̇_I      = ω_I + (1+r_I²)/(2r_I) · Σ_J A_IJ r_J sin(ψ_J-ψ_I)
         Ȧ_IJ     = μ r_I r_J cos(ψ_J-ψ_I) - γ A_IJ

Note: the coupling in the KM equation uses a factor of 1/N (not K/N) so that
the total drive on each oscillator scales identically to the OA equation when
A_IJ represents a population-averaged weight.

Initial condition matching (N oscillators → M populations of size d)
----------------------------------------------------------------------
  Frequencies:
    ω_k ~ Lorentzian(ω₀, Δ₀)  drawn once for all N oscillators.
    Population I uses oscillators k ∈ [Id, (I+1)d).
    ω_I = mean of {ω_k}_{k∈pop_I}  (centre frequency)
    Δ_I = π⁻¹ · sample HWHM estimated from {ω_k}_{k∈pop_I}
          (for small d, Δ_I is set to the global Δ₀ to avoid noise)

  Phases:
    θ_k ~ Uniform(-π, π)  drawn once.
    r_I⁰ = |mean_{k∈pop_I} exp(iθ_k)|   (empirical intra-population sync)
    ψ_I⁰ = arg(mean_{k∈pop_I} exp(iθ_k)) (empirical mean phase)

  Weights:
    A_kl⁰ = 0  for all k,l  (both simulations start from zero weights)
    A_IJ⁰ = 0  (consistent with block average of zero)

Comparison outputs
------------------
  Figure 1 — Global order parameter R(t) for both models
  Figure 2 — Final coupling matrices: full KM matrix + coarse-grained block
             average (M×M) alongside the OA weight matrix A_IJ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from config.utility_functions import uniform, lorentzian2, lorentzian
from numba import njit, prange
_EPS = 1e-9
plt.rcParams["font.size"] = 14.0

# ═══════════════════════════════════════════════════════════════════════════════
# Shared initial-condition factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_initial_conditions(N, d, omega0, Delta0, dist, seed):
    """
    Draw shared initial conditions for N oscillators grouped into M=N//d
    populations of size d.

    Returns
    -------
    omega_micro : (N,)   individual frequencies
    theta0      : (N,)   individual initial phases
    omega_pop   : (M,)   population centre frequencies
    delta_pop   : (M,)   population half-widths
    r0          : (M,)   initial OA order-parameter magnitudes
    psi0        : (M,)   initial OA mean phases
    """
    assert N % d == 0, "N must be divisible by d"
    M = N // d
    rng = np.random.default_rng(seed)

    # Individual frequencies — uniform
    if dist == "uniform":
        omega_pop = uniform(M, omega0, Delta0)
        delta_pop = uniform(M, Delta0/M, 0.0)
    elif dist == "lorentzian":
        n = np.arange(1, M+1)
        omega_pop = omega0 + Delta0*np.tan(0.5*np.pi*(2*n-M-1)/(M+1))
        delta_pop = Delta0*(np.tan(0.5*np.pi*(2*n-M-0.5)/(M+1))-np.tan(0.5*np.pi*(2*n-M-1.5)/(M+1)))
    else:
        raise ValueError(f"Invalid distribution argument (dist={dist}).")

    # Population-level parameters derived from the micro sample
    omega_micro = np.empty(N)
    r0 = np.empty(M)
    psi0 = np.empty(M)
    theta0 = np.empty(N)

    for I in range(M):
        th = rng.uniform(-np.pi, np.pi, d)
        idx = slice(I * d, (I + 1) * d)
        omega_micro[idx] = lorentzian2(d, omega_pop[I], delta_pop[I]) #lb=omega_pop[I]-5*Delta1, ub=omega_pop[I]+5*Delta1)
        theta0[idx] = th
        z_mean = np.mean(np.exp(1j * th))
        psi0[I] = np.angle(z_mean)
        r0[I] = np.abs(z_mean)

    return omega_micro, theta0, omega_pop, delta_pop, r0, psi0


# ═══════════════════════════════════════════════════════════════════════════════
# Microscopic Kuramoto ODE  (N oscillators + N² weights)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True)
def km_ode(t, theta, K, omega):
    N = len(omega)
    interaction = np.zeros_like(theta)
    for i in prange(N):
        interaction[i] = np.sum(np.sin(theta-theta[i]))
    dtheta = omega + (K / N) * interaction

    return dtheta


def km_order_parameter(theta):
    """Global KM order parameter R(t) from individual phases (N, steps)."""
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


# ═══════════════════════════════════════════════════════════════════════════════
# Macroscopic OA ODE  (M populations: r, ψ, M² weights)
# ═══════════════════════════════════════════════════════════════════════════════

# @njit
def oa_ode(t, y, K, omega, delta):
    M = len(omega)
    r = y[:M]
    psi = y[M:]

    dpsi = psi[np.newaxis, :] - psi[:, np.newaxis]  # ψ_J - ψ_I
    Ar = r[np.newaxis, :]  # broadcasts r_J across rows
    w_cos = np.sum(Ar * np.cos(dpsi), axis=1)
    w_sin = np.sum(Ar * np.sin(dpsi), axis=1)

    dy = np.zeros_like(y)
    dy[:M] = -delta * r + 0.5 * (1.0 - r ** 2) * w_cos * K/M
    dy[M:] = omega + 0.5 * (1.0 + r ** 2) / r * w_sin * K/M

    return dy


def oa_order_parameter(r, psi):
    """Global OA order parameter R(t) = |mean_I z_I| = |mean_I r_I e^{iψ_I}|."""
    return np.abs(np.mean(r * np.exp(1j * psi), axis=0))


# ═══════════════════════════════════════════════════════════════════════════════
# Combined simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(
        N=60,  # total number of microscopic oscillators
        d=6,  # population size  →  M = N/d populations
        T=80.0,  # simulation duration
        K = 1.0,  # global coupling strength
        omega0=0.0,  # centre of the Lorentzian
        Delta0=0.2,  # HWHM of the Lorentzian
        dist="uniform",
        seed=42,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
):
    assert N % d == 0, "N must be divisible by d"
    M = N // d

    print(f"N={N} oscillators,  d={d} per population,  M={M} populations")

    # ── Shared initial conditions ────────────────────────────────────────────
    omega_micro, theta0, omega_pop, delta_pop, r0, psi0 = \
        make_initial_conditions(N, d, omega0, Delta0, dist, seed)

    # ── KM simulation ────────────────────────────────────────────────────────
    print("\nRunning KM simulation …")
    sol_km = solve_ivp(km_ode, (0, T), theta0, method=method,
                       args=(K, omega_micro),
                       rtol=rtol, atol=atol, dense_output=False)
    if not sol_km.success:
        raise RuntimeError(f"KM solve_ivp failed: {sol_km.message}")
    print(f"  Done — {sol_km.t.size} steps, {sol_km.nfev} evaluations")

    t_km = sol_km.t
    theta = sol_km.y[:N]  # (N, steps_km)
    R_km = km_order_parameter(theta)  # (steps_km,)

    # ── OA simulation ────────────────────────────────────────────────────────
    y0_oa = np.concatenate([r0, psi0])
    print("Running OA simulation …")
    sol_oa = solve_ivp(oa_ode, (0, T), y0_oa, method=method,
                       args=(K, omega_pop, delta_pop),
                       rtol=rtol, atol=atol, dense_output=False)
    if not sol_oa.success:
        raise RuntimeError(f"OA solve_ivp failed: {sol_oa.message}")
    print(f"  Done — {sol_oa.t.size} steps, {sol_oa.nfev} evaluations")

    t_oa = sol_oa.t
    r_oa = sol_oa.y[:M]  # (M, steps_oa)
    psi_oa = sol_oa.y[M:2 * M]  # (M, steps_oa)
    R_oa = oa_order_parameter(r_oa, psi_oa)  # (steps_oa,)

    return dict(
        # KM
        t_km=t_km, theta=theta, R_km=R_km,
        omega_micro=omega_micro,
        # OA
        t_oa=t_oa, r_oa=r_oa, psi_oa=psi_oa, R_oa=R_oa,
        omega_pop=omega_pop, delta_pop=delta_pop,
        # shared
        N=N, M=M, d=d, T=T,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(res):
    N, M, d = res["N"], res["M"], res["d"]

    t_km, R_km = res["t_km"], res["R_km"]
    t_oa, R_oa = res["t_oa"], res["R_oa"]

    # ── Figure 1: Global order parameter ─────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(20, 8))
    ax.plot(t_km, R_km, color="steelblue", lw=1.8,
            label=f"KM  (N={N} oscillators)")
    ax.plot(t_oa, R_oa, color="crimson", lw=1.8, ls="--",
            label=f"OA  (M={M} populations, d={d})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Global order parameter $R(t)$")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_title(
        f"Global Kuramoto order parameter: KM vs OA\n"
        f"N={N}, d={d}, M={M}"
    )
    ax.legend(fontsize=11)
    fig1.tight_layout()

    return fig1


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    d = 100
    CONFIG = dict(
        N=4*d,  # total oscillators  (must be divisible by d)
        d=d,  # oscillators per population  →  M = N/d = 10
        T=200.0,  # simulation time
        K=3.0,  # global coupling strength
        omega0=1.0,  # Lorentzian centre frequency
        Delta0=1.0,  # Lorentzian HWHM
        #   smaller Δ → less incoherence damping → higher r
        dist="uniform",
        seed=42,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    res = simulate(**CONFIG)

    fig1 = plot_comparison(res)
    plt.show()

