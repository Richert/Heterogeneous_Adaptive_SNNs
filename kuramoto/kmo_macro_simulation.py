"""
Coupled Kuramoto Populations — Ott-Antonsen Reduction with Adaptive Weights
=============================================================================
Simulates M coupled oscillator populations. Each population i is described
by its complex Kuramoto order parameter z_i = r_i * exp(i*ψ_i), governed by
the Ott-Antonsen (OA) equations for a Lorentzian frequency distribution.
Inter-population coupling weights A_ij co-evolve with the order parameters.

Model
-----
Phase amplitude (order parameter magnitude):
    ṙ_i = -Δ_i r_i + (1 - r_i²)/2 * Σ_j A_ij r_j cos(ψ_j - ψ_i)

Mean phase:
    ψ̇_i = ω_i + (1 + r_i²)/(2 r_i) * Σ_j A_ij r_j sin(ψ_j - ψ_i)

Adaptive coupling weights (Hebbian + decay):
    Ȧ_ij = μ r_i r_j cos(ψ_j - ψ_i) - γ A_ij

where for each population i:
    ω_i   : centre frequency of the Lorentzian distribution
    Δ_i   : half-width at half-maximum (HWHM) of the distribution (> 0)
    r_i   : synchrony within population i  (0 = incoherent, 1 = fully sync)
    ψ_i   : mean phase of population i
    A_ij  : adaptive coupling weight from population j to population i

State vector layout (length 2M + M²):
    y = [r_0, …, r_{M-1}, ψ_0, …, ψ_{M-1}, A_00, A_01, …, A_{M-1,M-1}]

Notes
-----
* Diagonal weights A_ii represent self-coupling within a population and are
  included in the dynamics; set mu=0 or A0_scale=0 to start from zero.
* r_i is clipped to [ε, 1-ε] during integration via a soft barrier in the ODE
  to avoid the singularity in the ψ̇ equation at r_i = 0.
* The weight matrix is real-valued and unbounded; the steady-state magnitude
  is |A*_ij| ≤ μ/γ (achieved when populations are fully synchronised).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ── Constants ─────────────────────────────────────────────────────────────────

_EPS = 1e-9   # guard against r_i → 0 in the ψ̇ equation


# ── State packing / unpacking ─────────────────────────────────────────────────

def pack(r, psi, A):
    """Flatten r (M,), ψ (M,), A (M,M) → 1-D state vector (2M + M²,)."""
    return np.concatenate([r, psi, A.ravel()])


def unpack(y, M):
    """Recover r (M,), ψ (M,), A (M,M) from the flat state vector."""
    r   = y[:M]
    psi = y[M:2*M]
    A   = y[2*M:].reshape(M, M)
    return r, psi, A


# ── ODE right-hand side ───────────────────────────────────────────────────────

def oa_hebbian_ode(t, y, K, omega, delta, mu, gamma):
    """
    RHS of the Ott-Antonsen + adaptive-weight system with hebbian plasticity.

    Parameters
    ----------
    t     : float       Current time (required by solve_ivp).
    y     : (2M+M²,)    Flat state vector [r, ψ, A.ravel()].
    K     : float       Global coupling strength
    omega : (M,)        Population centre frequencies.
    delta : (M,)        Population half-widths (HWHM), must be > 0.
    mu    : float       Hebbian learning rate.
    gamma : float       Weight decay rate.

    Returns
    -------
    dy_dt : (2M+M²,)
    """
    M = len(omega)
    r, psi, A = unpack(y, M)

    # Guard r against the singularity at 0 and the OA boundary at 1
    r_safe = np.clip(r, _EPS, 1.0 - _EPS)

    # Phase differences:  dpsi[i,j] = ψ_j - ψ_i
    dpsi = psi[np.newaxis, :] - psi[:, np.newaxis]   # (M, M)

    # Weighted sums entering both equations
    # w_cos[i] = Σ_j A_ij r_j cos(ψ_j - ψ_i)
    # w_sin[i] = Σ_j A_ij r_j sin(ψ_j - ψ_i)
    Ar = A * r_safe[np.newaxis, :]                    # (M, M): A_ij * r_j
    w_cos = K/M * np.sum(Ar * np.cos(dpsi), axis=1)         # (M,)
    w_sin = K/M * np.sum(Ar * np.sin(dpsi), axis=1)         # (M,)

    # ── Ott-Antonsen equations ─────────────────────────────────────────────
    dr_dt  = -delta * r_safe + 0.5 * (1.0 - r_safe**2) * w_cos
    dps_dt = omega  + 0.5 * (1.0 + r_safe**2) / r_safe * w_sin

    # ── Adaptive weight dynamics  Ȧ_ij = μ r_i r_j cos(Δψ) - γ A_ij ──────
    rr = r_safe[:, np.newaxis] * r_safe[np.newaxis, :]   # (M, M): r_i * r_j
    dA_dt = mu * rr * np.cos(dpsi) - gamma * A

    return pack(dr_dt, dps_dt, dA_dt)

def oa_antihebbian_ode(t, y, K, omega, delta, mu, gamma):
    """
    RHS of the Ott-Antonsen + adaptive-weight system with hebbian plasticity.

    Parameters
    ----------
    t     : float       Current time (required by solve_ivp).
    y     : (2M+M²,)    Flat state vector [r, ψ, A.ravel()].
    K     : float       Global coupling strength
    omega : (M,)        Population centre frequencies.
    delta : (M,)        Population half-widths (HWHM), must be > 0.
    mu    : float       Hebbian learning rate.
    gamma : float       Weight decay rate.

    Returns
    -------
    dy_dt : (2M+M²,)
    """
    M = len(omega)
    r, psi, A = unpack(y, M)

    # Guard r against the singularity at 0 and the OA boundary at 1
    r_safe = np.clip(r, _EPS, 1.0 - _EPS)

    # Phase differences:  dpsi[i,j] = ψ_j - ψ_i
    dpsi = psi[np.newaxis, :] - psi[:, np.newaxis]   # (M, M)

    # Weighted sums entering both equations
    # w_cos[i] = Σ_j A_ij r_j cos(ψ_j - ψ_i)
    # w_sin[i] = Σ_j A_ij r_j sin(ψ_j - ψ_i)
    Ar = A * r_safe[np.newaxis, :]                    # (M, M): A_ij * r_j
    w_cos = K/M * np.sum(Ar * np.cos(dpsi), axis=1)         # (M,)
    w_sin = K/M * np.sum(Ar * np.sin(dpsi), axis=1)         # (M,)

    # ── Ott-Antonsen equations ─────────────────────────────────────────────
    dr_dt  = -delta * r_safe + 0.5 * (1.0 - r_safe**2) * w_cos
    dps_dt = omega  + 0.5 * (1.0 + r_safe**2) / r_safe * w_sin

    # ── Adaptive weight dynamics  Ȧ_ij = μ r_i r_j cos(Δψ) - γ A_ij ──────
    rr = r_safe[:, np.newaxis] * r_safe[np.newaxis, :]   # (M, M): r_i * r_j
    dA_dt = mu * rr * np.abs(np.sin(dpsi)) - gamma * A

    return pack(dr_dt, dps_dt, dA_dt)

# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(
    M=10,
    T=100.0,
    K=1.0,
    mu=0.3,
    gamma=0.1,
    omega_mean=0.0,
    omega_std=0.5,
    delta_mean=0.1,
    delta_std=0.02,
    r0_mean=0.5,
    r0_std=0.1,
    A0_scale=1.0,
    plasticity="hebbian",
    seed=42,
    method="RK45",
    rtol=1e-7,
    atol=1e-9,
):
    """
    Run the OA + adaptive-coupling simulation.

    Parameters
    ----------
    M          : int    Number of populations.
    T          : float  Simulation duration.
    K          : float  Global coupling strength.
    mu         : float  Hebbian learning rate for weights.
    gamma      : float  Weight decay rate.
    omega_mean : float  Mean centre frequency across populations.
    omega_std  : float  Std dev of centre frequencies (Gaussian spread).
    delta_mean : float  Mean HWHM (incoherence damping) across populations.
    delta_std  : float  Std dev of HWHM values.
    r0_mean    : float  Mean initial order-parameter magnitude.
    r0_std     : float  Std dev of initial r values.
    A0_scale   : float  Std dev of random initial weights (0 → start at zero).
    plasticity : str    'hebbian' or 'antihebbian'.
    seed       : int    Random seed for reproducibility.
    method     : str    solve_ivp solver.
    rtol, atol : float  Solver tolerances.

    Returns
    -------
    t      : (steps,)      Time array.
    r      : (M, steps)    Order-parameter magnitudes.
    psi    : (M, steps)    Mean phases.
    A_traj : (M, M, steps) Coupling weight matrices.
    omega  : (M,)          Centre frequencies used.
    delta  : (M,)          Half-widths used.
    """
    rng = np.random.default_rng(seed)

    omega = rng.uniform(-omega_std, omega_std, M) + omega_mean
    delta = np.maximum(0.0, rng.uniform(-delta_std, delta_std, M) + delta_mean)  # must be > 0

    # Initial order parameters: r ∈ (0,1), ψ ∈ [−π, π]
    r0   = np.clip(rng.uniform(-r0_std, r0_std, M) + r0_mean, _EPS, 1.0 - _EPS)
    psi0 = rng.uniform(-np.pi, np.pi, M)

    # Initial weights
    if A0_scale > 0:
        A0 = rng.normal(0, A0_scale, (M, M))
        A0 = (A0 + A0.T) / 2   # start symmetric
    else:
        A0 = np.zeros((M, M))

    y0 = pack(r0, psi0, A0)

    print(f"OA adaptive simulation  —  M={M} populations, T={T}")
    print(f"  μ={mu}, γ={gamma}  →  |A*_ij| ≤ {mu/gamma:.3f}")
    print(f"  Δ ∈ [{delta.min():.3f}, {delta.max():.3f}]  "
          f"ω ∈ [{omega.min():.3f}, {omega.max():.3f}]")
    print(f"  Solver: {method}  (rtol={rtol}, atol={atol})")

    sol = solve_ivp(
        fun=oa_hebbian_ode if plasticity == "hebbian" else oa_antihebbian_ode,
        t_span=(0.0, T),
        y0=y0,
        method=method,
        args=(K, omega, delta, mu, gamma),
        dense_output=False,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    print(f"  Done — {sol.t.size} steps, {sol.nfev} RHS evaluations")

    t      = sol.t
    r      = sol.y[:M, :]
    psi    = sol.y[M:2*M, :]
    A_traj = sol.y[2*M:, :].reshape(M, M, -1)

    return t, r, psi, A_traj, omega, delta


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(t, r, psi, A_traj, omega, delta, mu, gamma,
                 title="OA Adaptive Kuramoto"):
    """Five-panel summary plot."""
    M = r.shape[0]

    # Global order parameter: magnitude of the mean complex order parameter
    z_mean = np.mean(r * np.exp(1j * psi), axis=0)
    R_global = np.abs(z_mean)

    # Mean and std of r across populations
    r_mean = r.mean(axis=0)
    r_std  = r.std(axis=0)

    # Mean coupling weight statistics (all entries including diagonal)
    mean_w  = A_traj.mean(axis=(0, 1))
    mean_aw = np.abs(A_traj).mean(axis=(0, 1))
    theoretical_max = mu / gamma

    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    shared_axes = axes[:4]   # time-series panels share x-axis
    for ax in shared_axes[1:]:
        ax.sharex(shared_axes[0])

    # 1. Per-population order parameter magnitude r_i(t)
    ax = axes[0]
    for i in range(M):
        ax.plot(t, r[i], lw=0.8, alpha=0.7, label=f"pop {i}" if M <= 8 else None)
    ax.set_ylabel("$r_i(t)$")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-population order parameter magnitude")
    if M <= 8:
        ax.legend(fontsize=8, ncol=M, loc="lower right")

    # 2. Mean r ± std and global R
    ax = axes[1]
    ax.fill_between(t, r_mean - r_std, r_mean + r_std, alpha=0.25, color="steelblue")
    ax.plot(t, r_mean,   color="steelblue", lw=1.5, label=r"$\langle r_i \rangle \pm \sigma$")
    ax.plot(t, R_global, color="crimson",   lw=1.5, ls="--",
            label=r"$R = |\langle z_i \rangle|$ (global)")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Order parameter")
    ax.set_title("Mean within-population sync vs. global inter-population sync")
    ax.legend(fontsize=9)

    # 3. Mean phases ψ_i(t) wrapped to [−π, π]
    ax = axes[2]
    psi_wrapped = np.mod(psi.T + np.pi, 2 * np.pi) - np.pi
    ax.plot(t, psi_wrapped, lw=0.8, alpha=0.7)
    ax.set_ylabel(r"$\psi_i$ (rad)")
    ax.set_title("Mean phases wrapped to [−π, π]")

    # 4. Mean coupling weight over time
    ax = axes[3]
    ax.plot(t, mean_w,  lw=1.5, label=r"$\langle A_{ij} \rangle$")
    ax.plot(t, mean_aw, lw=1.5, ls="--", label=r"$\langle |A_{ij}| \rangle$")
    ax.axhline(theoretical_max, color="grey", lw=0.8, ls=":",
               label=rf"$\mu/\gamma = {theoretical_max:.2f}$")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Time")
    ax.set_title("Mean coupling weight over time")
    ax.legend(fontsize=9)

    # 5. Final weight matrix heatmap (separate x-axis)
    ax = axes[4]
    A_final = A_traj[:, :, -1]
    vmax = np.abs(A_final).max() or 1.0
    im = ax.imshow(A_final, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, label="$A_{ij}$")
    ax.set_xticks(range(M))
    ax.set_yticks(range(M))
    ax.set_xlabel("Population j")
    ax.set_ylabel("Population i")
    ax.set_title(f"Final weight matrix $A(T={t[-1]:.0f})$")

    plt.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CONFIG = dict(
        M=20,           # number of populations
        T=500.0,        # simulation duration
        K=1.0,           # global coupling strength
        mu=0.1,         # Hebbian learning rate
                        #   larger → weights grow faster toward μ/γ
        gamma=0.01,      # weight decay rate
                        #   steady-state bound: |A*_ij| ≤ μ/γ = 3.0 here
        omega_mean=0.4, # mean centre frequency
        omega_std=0.2,  # spread of centre frequencies across populations
        delta_mean=0.005, # mean HWHM (incoherence damping within a population)
                        #   smaller Δ → populations can reach higher r
        delta_std=0.0, # heterogeneity in HWHM values
        r0_mean=0.0,    # initial order-parameter magnitude (mean)
        r0_std=0.1,     # initial order-parameter magnitude (spread)
        A0_scale=1.0,   # initial weight noise (0 → all weights start at zero)
        plasticity="antihebbian",
        seed=42,
        method="RK45",  # "RK45" | "RK23" | "DOP853" | "Radau" | "BDF" | "LSODA"
        rtol=1e-7,
        atol=1e-9,
    )

    t, r, psi, A_traj, omega, delta = simulate(**CONFIG)

    fig = plot_results(
        t, r, psi, A_traj, omega, delta,
        mu=CONFIG["mu"], gamma=CONFIG["gamma"],
        title=(f"OA Adaptive Kuramoto  |  M={CONFIG['M']} populations, "
               f"T={CONFIG['T']},  μ={CONFIG['mu']}, γ={CONFIG['gamma']}"),
    )

    plt.savefig("oa_adaptive_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved to oa_adaptive_results.png")
    plt.show()
