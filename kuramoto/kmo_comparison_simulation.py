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
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from config.utility_functions import uniform, lorentzian2, lorentzian
_EPS = 1e-9
plt.rcParams["font.size"] = 16.0


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
        omega_micro[idx] = lorentzian2(d, omega_pop[I], delta_pop[I]) #, lb=omega_pop[I]-10*delta_pop[I], ub=omega_pop[I]+10*delta_pop[I])
        theta0[idx] = th
        z_mean = np.mean(np.exp(1j * th))
        psi0[I] = np.angle(z_mean)
        r0[I] = np.abs(z_mean)

    return omega_micro, theta0, omega_pop, delta_pop, r0, psi0


# ═══════════════════════════════════════════════════════════════════════════════
# Microscopic Kuramoto ODE  (N oscillators + N² weights)
# ═══════════════════════════════════════════════════════════════════════════════

def hebbian(x):
    return np.cos(x)

def antihebbian(x):
    return np.sin(x)

def km_ode(t, y, K, omega, mu, gamma, f):
    N = len(omega)
    theta = y[:N]
    A = y[N:].reshape(N, N)

    diff = theta[np.newaxis, :] - theta[:, np.newaxis]  # (N,N): θ_l-θ_k
    interaction = np.sum(A * np.sin(diff), axis=1)  # (N,)
    dtheta = omega + (K/N) * interaction

    dA = mu * f(diff) - gamma * A
    np.fill_diagonal(dA, 0.0)  # no self-coupling

    return np.concatenate([dtheta, dA.ravel()])


def km_order_parameter(theta):
    """Global KM order parameter R(t) from individual phases (N, steps)."""
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


def km_coarse_grain(A_fine, d):
    """
    Coarse-grain the N×N weight matrix into M×M by block-averaging.
    Block (I,J) = mean of the d×d sub-matrix A[Id:(I+1)d, Jd:(J+1)d].
    """
    N = A_fine.shape[0]
    M = N // d
    A_cg = np.zeros((M, M))
    for I in range(M):
        for J in range(M):
            A_cg[I, J] = A_fine[I * d:(I + 1) * d, J * d:(J + 1) * d].mean()
    return A_cg


# ═══════════════════════════════════════════════════════════════════════════════
# Macroscopic OA ODE  (M populations: r, ψ, M² weights)
# ═══════════════════════════════════════════════════════════════════════════════

def oa_ode(t, y, K, omega, delta, mu, gamma, f):
    M = len(omega)
    r = np.clip(y[:M], _EPS, 1.0 - _EPS)
    psi = y[M:2*M]
    A = y[2*M:].reshape(M, M)

    dpsi = psi[np.newaxis, :] - psi[:, np.newaxis]  # ψ_J - ψ_I
    Ar = A * r[np.newaxis, :]  # A_IJ * r_J
    w_cos = np.sum(Ar * np.cos(dpsi), axis=1)
    w_sin = np.sum(Ar * np.sin(dpsi), axis=1)

    dr = -delta*r + 0.5*(1.0 - r**2) * w_cos * K/M
    dpsi_ = omega + 0.5*(1.0 + r**2) / r * w_sin * K/M

    rr =  r[np.newaxis, :] * r[:, np.newaxis]
    dA = mu * rr * f(dpsi) - gamma * A

    return np.concatenate([dr, dpsi_, dA.ravel()])


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
        mu=0.4,  # Hebbian learning rate
        gamma=0.15,  # weight decay rate
        omega0=0.0,  # centre of the Lorentzian
        Delta0=0.2,  # HWHM of the Lorentzian
        plasticity = "antihebbian",   # hebbian or antihebbian
        dist = "uniform",
        seed=42,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
):
    assert N % d == 0, "N must be divisible by d"
    M = N // d

    print(f"N={N} oscillators,  d={d} per population,  M={M} populations")
    print(f"μ={mu}, γ={gamma}, Δ₀={Delta0},  T={T}")
    if gamma > 0.0:
        print(f"Steady-state weight bound: |A*| ≤ μ/γ = {mu / gamma:.3f}")

    # ── Shared initial conditions ────────────────────────────────────────────
    omega_micro, theta0, omega_pop, delta_pop, r0, psi0 = \
        make_initial_conditions(N, d, omega0, Delta0, dist, seed)

    A_km0 = np.ones((N, N))  # both models start with 1.0 weights
    A_oa0 = np.ones((M, M))

    f = hebbian if plasticity == "hebbian" else antihebbian

    # ── KM simulation ────────────────────────────────────────────────────────
    y0_km = np.concatenate([theta0, A_km0.ravel()])
    print("\nRunning KM simulation …")
    sol_km = solve_ivp(km_ode, (0, T), y0_km, method=method,
                       args=(K, omega_micro, mu, gamma, f),
                       rtol=rtol, atol=atol, dense_output=False)
    if not sol_km.success:
        raise RuntimeError(f"KM solve_ivp failed: {sol_km.message}")
    print(f"  Done — {sol_km.t.size} steps, {sol_km.nfev} evaluations")

    t_km = sol_km.t
    theta = sol_km.y[:N]  # (N, steps_km)
    A_km = sol_km.y[N:].reshape(N, N, -1)  # (N, N, steps_km)
    R_km = km_order_parameter(theta)  # (steps_km,)

    # ── OA simulation ────────────────────────────────────────────────────────
    y0_oa = np.concatenate([r0, psi0, A_oa0.ravel()])
    print("Running OA simulation …")
    sol_oa = solve_ivp(oa_ode, (0, T), y0_oa, method=method,
                       args=(K, omega_pop, delta_pop, mu, gamma, f),
                       rtol=rtol, atol=atol, dense_output=False)
    if not sol_oa.success:
        raise RuntimeError(f"OA solve_ivp failed: {sol_oa.message}")
    print(f"  Done — {sol_oa.t.size} steps, {sol_oa.nfev} evaluations")

    t_oa = sol_oa.t
    r_oa = sol_oa.y[:M]  # (M, steps_oa)
    psi_oa = sol_oa.y[M:2 * M]  # (M, steps_oa)
    A_oa = sol_oa.y[2 * M:].reshape(M, M, -1)  # (M, M, steps_oa)
    R_oa = oa_order_parameter(r_oa, psi_oa)  # (steps_oa,)

    return dict(
        # KM
        t_km=t_km, theta=theta, A_km=A_km, R_km=R_km,
        omega_micro=omega_micro,
        # OA
        t_oa=t_oa, r_oa=r_oa, psi_oa=psi_oa, A_oa=A_oa, R_oa=R_oa,
        omega_pop=omega_pop, delta_pop=delta_pop,
        # shared
        N=N, M=M, d=d, mu=mu, gamma=gamma, T=T,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(res):
    N, M, d = res["N"], res["M"], res["d"]
    mu, gamma = res["mu"], res["gamma"]

    t_km, R_km = res["t_km"], res["R_km"]
    t_oa, R_oa = res["t_oa"], res["R_oa"]

    A_km_final = res["A_km"][:, :, -1]  # (N, N)
    A_oa_final = res["A_oa"][:, :, -1]  # (M, M)
    A_km_cg = km_coarse_grain(A_km_final, d)  # (M, M)

    # ── Figure 1: Global order parameter ─────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(10, 4))
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
        f"N={N}, d={d}, M={M},  μ={mu}, γ={gamma}"
    )
    ax.legend(fontsize=11)
    fig1.tight_layout()

    # ── Figure 2: Coupling matrices ───────────────────────────────────────────
    # Three heatmaps: full KM (N×N), block-averaged KM (M×M), OA (M×M)
    vmax_full = np.abs(A_km_final).max() or 1.0
    vmax_cg = max(np.abs(A_km_cg).max(), np.abs(A_oa_final).max()) or 1.0

    fig2 = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)

    # -- Full KM matrix (N×N)
    ax0 = fig2.add_subplot(gs[0])
    im0 = ax0.imshow(A_km_final, cmap="RdBu_r",
                     vmin=-vmax_full, vmax=vmax_full,
                     interpolation="nearest", aspect="auto")
    plt.colorbar(im0, ax=ax0, label="$A_{kl}$", shrink=0.85)
    # Draw population-boundary grid lines
    for tick in range(0, N + 1, d):
        ax0.axhline(tick - 0.5, color="k", lw=0.4)
        ax0.axvline(tick - 0.5, color="k", lw=0.4)
    ax0.set_title(f"KM weight matrix $A^{{KM}}$  (N×N = {N}×{N})")
    ax0.set_xlabel("Oscillator $l$")
    ax0.set_ylabel("Oscillator $k$")

    # -- Coarse-grained KM matrix (M×M)
    ax1 = fig2.add_subplot(gs[1])
    im1 = ax1.imshow(A_km_cg, cmap="RdBu_r",
                     vmin=-vmax_cg, vmax=vmax_cg,
                     interpolation="nearest", aspect="equal")
    plt.colorbar(im1, ax=ax1, label="$\\bar{A}_{IJ}$", shrink=0.85)
    ax1.set_xticks(range(M));
    ax1.set_yticks(range(M))
    ax1.set_title(f"KM block-averaged  $\\bar{{A}}^{{KM}}$  (M×M = {M}×{M})")
    ax1.set_xlabel("Population $J$")
    ax1.set_ylabel("Population $I$")

    # -- OA weight matrix (M×M)
    ax2 = fig2.add_subplot(gs[2])
    im2 = ax2.imshow(A_oa_final, cmap="RdBu_r",
                     vmin=-vmax_cg, vmax=vmax_cg,
                     interpolation="nearest", aspect="equal")
    plt.colorbar(im2, ax=ax2, label="$A_{IJ}$", shrink=0.85)
    ax2.set_xticks(range(M));
    ax2.set_yticks(range(M))
    ax2.set_title(f"OA weight matrix $A^{{OA}}$  (M×M = {M}×{M})")
    ax2.set_xlabel("Population $J$")
    ax2.set_ylabel("Population $I$")

    fig2.suptitle(
        f"Final coupling matrices at T={res['T']:.0f}  "
        f"(μ={mu}, γ={gamma}, d={d})",
        fontsize=12, fontweight="bold"
    )
    fig2.tight_layout()
    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    d = 20
    CONFIG = dict(
        N=10*d,  # total oscillators  (must be divisible by d)
        d=d,  # oscillators per population  →  M = N/d = 10
        T=100.0,  # simulation time
        K=0.5,  # global coupling strength
        mu=0.01,  # Hebbian learning rate
        gamma=0.0,  # weight decay  →  |A*| ≤ μ/γ ≈ 2.67
        omega0=0.0,  # Lorentzian centre frequency
        Delta0=0.3,  # Lorentzian HWHM
        #   smaller Δ → less incoherence damping → higher r
        plasticity="antihebbian",
        dist="uniform",
        seed=42,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    res = simulate(**CONFIG)

    fig1, fig2 = plot_comparison(res)

    fig1.savefig("comparison_order_parameter.png", dpi=150, bbox_inches="tight")
    fig2.savefig("comparison_coupling_matrices.png", dpi=150, bbox_inches="tight")
    print("\nFigures saved:")
    print("  comparison_order_parameter.png")
    print("  comparison_coupling_matrices.png")

    plt.show()

