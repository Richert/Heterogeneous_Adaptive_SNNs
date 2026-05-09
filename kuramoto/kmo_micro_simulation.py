"""
Kuramoto Oscillator Network with Adaptive Coupling
====================================================
Simulates N coupled Kuramoto oscillators whose connection weights co-evolve
with the oscillator phases, using scipy's solve_ivp.

Model (coupled ODEs):
    dθ_i/dt  = ω_i + (K/N) * Σ_j A_ij * sin(θ_j - θ_i)
    dA_ij/dt = μ * cos(θ_j - θ_i) - γ * A_ij

where:
    θ_i   : phase of oscillator i
    ω_i   : natural frequency of oscillator i
    K     : global coupling strength scaling the phase dynamics
    A_ij  : continuous (real-valued) coupling weight between i and j
    μ     : Hebbian learning rate  (drives A_ij toward phase-coherent pairs)
    γ     : decay / forgetting rate (prevents unbounded growth of weights)

The state vector passed to solve_ivp is the concatenation:
    y = [θ_0, ..., θ_{N-1}, A_00, A_01, ..., A_{N-1,N-1}]
        length N + N²

Notes
-----
* Self-coupling is excluded: A_ii is forced to zero in the initial condition
  and its derivative is zero (cos(0) - γ*0 = 1 ≠ 0 in general, so we mask
  the diagonal explicitly in the ODE).
* The weight matrix is NOT constrained to be symmetric during integration;
  asymmetric adaptive networks are a valid and interesting case. If you want
  a symmetric network, set `symmetrise=True` in CONFIG (achieved by
  initialising A symmetrically and noting that the ODE preserves symmetry
  when phase differences are symmetric, which they are).
* A_ij is real-valued and unbounded by default. A fixed point satisfies
  A*_ij = (μ/γ) * cos(Δθ*_ij), so weights saturate naturally near ±μ/γ.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ── State packing / unpacking ─────────────────────────────────────────────────

def pack(theta, A):
    """Flatten θ (N,) and A (N,N) into a single 1-D state vector."""
    return np.concatenate([theta, A.ravel()])


def unpack(y, N):
    """Recover θ (N,) and A (N,N) from the flat state vector."""
    theta = y[:N]
    A = y[N:].reshape(N, N)
    return theta, A


# ── Coupled ODE (phases + weights) ───────────────────────────────────────────

def hebbian_kuramoto_ode(t, y, omega, K, mu, gamma, diag_mask):
    """
    RHS of the adaptive Kuramoto system with Hebbian plasticity.

    Parameters
    ----------
    t         : float         Current time (unused directly).
    y         : (N + N²,)     Flat state [θ, A.ravel()].
    omega     : (N,)          Natural frequencies.
    K         : float         Phase-coupling strength.
    mu        : float         Hebbian learning rate for weights.
    gamma     : float         Weight decay rate.
    diag_mask : (N,N) bool    True on off-diagonal entries (precomputed).

    Returns
    -------
    dy_dt : (N + N²,) ndarray
    """
    N = len(omega)
    theta, A = unpack(y, N)

    # ── Phase differences  diff[i,j] = θ_j - θ_i ──────────────────────────
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]   # (N, N)

    # ── Phase dynamics ──────────────────────────────────────────────────────
    interaction = np.sum(A * np.sin(diff), axis=1)        # (N,)
    dtheta_dt = omega + (K / N) * interaction

    # ── Weight dynamics  dA_ij/dt = μ cos(Δθ) - γ A_ij  (off-diagonal) ───
    dA_dt = diag_mask * (mu * np.cos(diff) - gamma * A)  # diagonal stays 0

    return pack(dtheta_dt, dA_dt)

def antihebbian_kuramoto_ode(t, y, omega, K, mu, gamma, diag_mask):
    """
    RHS of the adaptive Kuramoto system with anti-Hebbian plasticity.

    Parameters
    ----------
    t         : float         Current time (unused directly).
    y         : (N + N²,)     Flat state [θ, A.ravel()].
    omega     : (N,)          Natural frequencies.
    K         : float         Phase-coupling strength.
    mu        : float         Hebbian learning rate for weights.
    gamma     : float         Weight decay rate.
    diag_mask : (N,N) bool    True on off-diagonal entries (precomputed).

    Returns
    -------
    dy_dt : (N + N²,) ndarray
    """
    N = len(omega)
    theta, A = unpack(y, N)

    # ── Phase differences  diff[i,j] = θ_j - θ_i ──────────────────────────
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]   # (N, N)

    # ── Phase dynamics ──────────────────────────────────────────────────────
    interaction = np.sum(A * np.sin(diff), axis=1)        # (N,)
    dtheta_dt = omega + (K / N) * interaction

    # ── Weight dynamics  dA_ij/dt = μ cos(Δθ) - γ A_ij  (off-diagonal) ───
    dA_dt = diag_mask * (mu * np.abs(np.sin(diff)) - gamma * A)  # diagonal stays 0

    return pack(dtheta_dt, dA_dt)

# ── Order parameter ───────────────────────────────────────────────────────────

def order_parameter(theta):
    """
    Kuramoto order parameter r(t) = |mean(exp(i·θ))|.

    r ≈ 0 → incoherent,  r ≈ 1 → fully synchronised.
    Accepts θ of shape (N,) or (N, steps).
    """
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(
    N=30,
    T=50.0,
    K=1.5,
    mu=0.5,
    gamma=0.2,
    omega_mean=0.0,
    omega_std=1.0,
    A0_scale=0.0,
    plasticity="hebbian",
    symmetrise=True,
    seed=42,
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
):
    """
    Run the adaptive Kuramoto simulation.

    Parameters
    ----------
    N          : int    Number of oscillators.
    T          : float  Total simulation time.
    K          : float  Phase-coupling strength.
    mu         : float  Hebbian learning rate (weight growth toward coherence).
    gamma      : float  Weight decay rate (prevents unbounded growth).
    omega_mean : float  Mean natural frequency.
    omega_std  : float  Width of the Lorentzian frequency distribution.
    A0_scale   : float  Std dev of random initial weights (0 → start from zero).
    plasticity : str    'hebbian' or 'antihebbian'
    symmetrise : bool   If True, symmetrise the initial weight matrix.
    seed       : int    Random seed.
    method     : str    solve_ivp solver (e.g. 'RK45', 'DOP853', 'Radau').
    rtol, atol : float  Solver tolerances.

    Returns
    -------
    t      : (steps,)   Time points.
    theta  : (N, steps) Phase trajectories.
    A_traj : (N, N, steps) Weight matrix over time.
    omega  : (N,)       Natural frequencies used.
    """
    rng = np.random.default_rng(seed)

    # Natural frequencies
    #omega = rng.standard_cauchy(N) * omega_std + omega_mean
    omega = rng.uniform(-omega_std, +omega_std, N) + omega_mean

    # Initial phases — uniform on [−π, π]
    theta0 = rng.uniform(-np.pi, np.pi, N)

    # Initial weights
    if A0_scale > 0:
        A0 = rng.normal(0, A0_scale, (N, N))
        if symmetrise:
            A0 = (A0 + A0.T) / 2
    else:
        A0 = np.zeros((N, N))
    np.fill_diagonal(A0, 0.0)   # no self-coupling

    # Off-diagonal mask (precomputed, passed into ODE to avoid repeated alloc)
    diag_mask = ~np.eye(N, dtype=bool)

    y0 = pack(theta0, A0)

    print(f"Adaptive Kuramoto simulation — N={N}, T={T}")
    print(f"  K={K}, μ={mu}, γ={gamma}  →  steady-state |A_ij| ≤ {mu/gamma:.3f}")
    print(f"  Solver: {method}  (rtol={rtol}, atol={atol})")

    sol = solve_ivp(
        fun=hebbian_kuramoto_ode if plasticity == 'hebbian' else antihebbian_kuramoto_ode,
        t_span=(0.0, T),
        y0=y0,
        method=method,
        args=(omega, K, mu, gamma, diag_mask),
        dense_output=False,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    print(f"  Done — {sol.t.size} steps, {sol.nfev} RHS evaluations")

    t = sol.t
    theta = sol.y[:N, :]           # (N, steps)
    A_traj = sol.y[N:, :].reshape(N, N, -1)   # (N, N, steps)

    return t, theta, A_traj, omega


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(t, theta, A_traj, omega, mu, gamma, title="Adaptive Kuramoto"):
    """Four-panel summary plot."""
    N = theta.shape[0]
    phases = np.mod(theta + np.pi, 2 * np.pi) - np.pi
    r = order_parameter(phases)

    # Mean weight and mean |weight| over time
    # Exclude diagonal (always zero)
    mask = ~np.eye(N, dtype=bool)
    mean_w  = np.array([A_traj[:, :, k][mask].mean() for k in range(len(t))])
    mean_aw = np.array([np.abs(A_traj[:, :, k][mask]).mean() for k in range(len(t))])
    theoretical_max = mu / gamma

    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 1. Wrapped phases
    ax0 = axes[0]
    ax0.plot(t, phases.T, lw=0.5, alpha=0.5)
    ax0.set_ylabel("Phase mod 2π (rad)")
    ax0.set_title("Oscillator phases wrapped to [−π, π]")

    # 2. Order parameter
    ax = axes[1]
    ax.plot(t, r, color="crimson", lw=2)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("r(t)")
    ax.set_title("Synchronisation order parameter")
    ax.sharex(ax0)

    # 3. Mean coupling weight
    ax = axes[2]
    ax.plot(t, mean_w,  lw=1.5, label=r"$\langle A_{ij} \rangle$")
    ax.plot(t, mean_aw, lw=1.5, ls="--", label=r"$\langle |A_{ij}| \rangle$")
    ax.axhline(theoretical_max, color="grey", lw=0.8, ls=":",
               label=rf"$\mu/\gamma = {theoretical_max:.2f}$")
    ax.set_ylabel("Weight")
    ax.set_title("Mean coupling weight over time")
    ax.legend(fontsize=9)
    ax.sharex(ax0)

    # 4. Final weight matrix heatmap
    ax = axes[3]
    A_final = A_traj[:, :, -1]
    vmax = np.abs(A_final).max() or 1.0
    im = ax.imshow(A_final, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, label="$A_{ij}$")
    ax.set_title(f"Final weight matrix A(T={t[-1]:.1f})")
    ax.set_xlabel("Oscillator j")
    ax.set_ylabel("Oscillator i")
    # sharex links x-axis to time plots, so we override xlabel for the heatmap
    ax.set_xlabel("Oscillator j")
    ax.set_xlim(-0.5, A_final.shape[1] - 0.5)

    plt.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CONFIG = dict(
        N=200,           # number of oscillators
        T=200.0,         # simulation duration (seconds)
        K=1.0,          # phase-coupling strength
        mu=0.0,         # Hebbian learning rate  — larger → weights grow faster
        gamma=0.0,      # weight decay rate       — larger → weights stay small
                        # steady-state bound: |A_ij| ≤ μ/γ
        omega_mean=1.0,
        omega_std=2.0,
        A0_scale=1.0,   # initial weight noise (0 → all weights start at zero)
        symmetrise=True,
        seed=42,
        method="RK45",  # "RK45" | "RK23" | "DOP853" | "Radau" | "BDF" | "LSODA"
        rtol=1e-6,
        atol=1e-8,
        plasticity="antihebbian"
    )

    t, theta, A_traj, omega = simulate(**CONFIG)

    fig = plot_results(
        t, theta, A_traj, omega,
        mu=CONFIG["mu"], gamma=CONFIG["gamma"],
        title=(f"Adaptive Kuramoto  |  N={CONFIG['N']}, K={CONFIG['K']}, "
               f"μ={CONFIG['mu']}, γ={CONFIG['gamma']}"),
    )

    plt.savefig("kuramoto_adaptive_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved to kuramoto_adaptive_results.png")
    plt.show()

