"""
Minimal single-population theta neuron vs. OA mean-field comparison
====================================================================
All-to-all coupling, A_ij = 1 for all i,j, no plasticity.

Microscopic model (N theta neurons):
    dθ_k/dt = (1 - cos θ_k) + (1 + cos θ_k) * [η_k + I(t)]
    I(t)    = J/N * Σ_l s(n, θ_l)
    s(n,θ)  = c_n * (1 - cos θ)^n,   c_n = 2^n(n!)^2/(2n)!

OA mean-field (complex Z):
    Ż = ½ [(iη̄ + iI(t) - Δ)(1+Z)² + i(1-Z)²]

Three variants of I(t) are run for the OA model:
    (a) OA-exact:   I(t) = J/N * Σ_l s(n, θ_l(t))  — same empirical mean as TN
    (b) OA-manifold: I(t) = J * <s(n,θ)>_Z           — OA manifold approximation
    (c) OA-micro:   feeds the microscopic I(t) trajectory into Ż post-hoc
                    (diagnostic only — not a coupled ODE)

This isolates three possible error sources independently:
    - If (a) matches TN but (b) does not → error is in <s>_Z approximation
    - If (a) does not match TN           → error is in the OA equation itself
    - (c) is a consistency check: if (c) matches TN, the OA equation is correct
      but I(t) is being computed wrongly; if (c) also mismatches, the OA
      equation itself is wrong
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import factorial, comb

plt.rcParams["font.size"] = 14.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def coupling_norm(n):
    return (2 ** n * factorial(n) ** 2) / factorial(2 * n)


def s_kernel(n_pulse, theta, cn):
    """Microscopic synaptic kernel s(n, θ) = c_n (1-cosθ)^n."""
    return cn * (1.0 - np.cos(theta)) ** n_pulse


def fourier_coeffs_s(n):
    """Real Fourier cosine coefficients of (1-cosθ)^n (verified previously)."""
    c = np.zeros(2 * n + 1, dtype=complex)
    for k in range(n + 1):
        binom_nk = comb(n, k) * (-1) ** k / 2 ** k
        for j in range(k + 1):
            p = 2 * j - k
            c[p + n] += binom_nk * comb(k, j)
    s_hat = np.zeros(n + 1)
    s_hat[0] = c[n].real
    for p in range(1, n + 1):
        s_hat[p] = 2.0 * c[p + n].real
    return s_hat


def oa_synaptic_mean(Z, n_pulse, s_hat, cn):
    """
    <s(n,θ)>_Z on the OA manifold.
    Z scalar or array; returns real scalar or array.
    """
    S = np.full_like(Z, s_hat[0], dtype=complex)
    Zp = Z.copy().astype(complex)
    for p in range(1, n_pulse + 1):
        S += s_hat[p] * Zp
        Zp *= Z
    return cn * S.real


def lorentzian_quantile(N, eta0, Delta):
    """N quantile-spaced samples from Lorentzian(eta0, Delta)."""
    k = np.arange(1, N + 1)
    return eta0 + Delta * np.tan(np.pi * (k / (N + 1) - 0.5))


# ─────────────────────────────────────────────────────────────────────────────
# Microscopic ODE
# ─────────────────────────────────────────────────────────────────────────────

def tn_ode(t, theta, eta, J, n_pulse, cn):
    """All-to-all theta neuron network, A_ij=1, no plasticity."""
    I = J / len(eta) * np.sum(s_kernel(n_pulse, theta, cn))
    cm = np.cos(theta)
    return (1.0 - cm) + (1.0 + cm) * (eta + I)


# ─────────────────────────────────────────────────────────────────────────────
# OA ODEs  (three variants)
# ─────────────────────────────────────────────────────────────────────────────

def oa_ode_manifold(t, y, eta0, Delta, J, n_pulse, s_hat, cn):
    """
    OA with I(t) = J * <s>_Z  (OA manifold approximation).
    State: y = [Re(Z), Im(Z)]
    """
    Z = y[0] + 1j * y[1]
    I = J * oa_synaptic_mean(np.array([Z]), n_pulse, s_hat, cn)[0]
    Zdot = -0.5 * ((Delta - 1j * (eta0 + I)) * (1 + Z) ** 2
                  + 1j * (1 - Z) ** 2)
    return [Zdot.real, Zdot.imag]


def oa_ode_exact(t, y, eta0, Delta, J, n_pulse, cn, theta_interp):
    """
    OA with I(t) = exact empirical mean from the microscopic trajectory.
    theta_interp(t) returns the current microscopic phase array.
    State: y = [Re(Z), Im(Z)]
    """
    theta_now = theta_interp(t)
    I = J / len(theta_now) * np.sum(s_kernel(n_pulse, theta_now, cn))
    Zdot = -0.5 * ((Delta - 1j * (eta0 + I)) * (1 + y[0] + 1j * y[1]) ** 2
                  + 1j * (1 - y[0] - 1j * y[1]) ** 2)
    return [Zdot.real, Zdot.imag]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
        N=1000,
        T=100.0,
        J=1.0,
        eta0=-0.5,
        Delta=0.5,
        n_pulse=1,
        seed=42,
        method="RK45",
        rtol=1e-8,
        atol=1e-9,
):
    cn = coupling_norm(n_pulse)
    s_hat = fourier_coeffs_s(n_pulse)

    print(f"N={N}, J={J}, η₀={eta0}, Δ={Delta}, n_pulse={n_pulse}")
    print(f"c_n={cn:.6f},  s_hat={s_hat}")

    rng = np.random.default_rng(seed)
    eta = lorentzian_quantile(N, eta0, Delta)
    theta0_micro = rng.uniform(-np.pi, np.pi, N)

    # Initial OA state: empirical mean of e^{iθ}
    Z0 = np.mean(np.exp(1j * theta0_micro))
    y0_oa = [Z0.real, Z0.imag]

    # ── 1. Microscopic simulation ─────────────────────────────────────────────
    print("\nRunning microscopic TN …")
    sol_tn = solve_ivp(
        tn_ode, (0, T), theta0_micro, method=method,
        args=(eta, J, n_pulse, cn),
        rtol=rtol, atol=atol, dense_output=True,  # dense for interpolation
    )
    assert sol_tn.success, sol_tn.message
    print(f"  Done — {sol_tn.t.size} steps")

    # Empirical order parameter and mean synaptic input from microscopic run
    t_tn = sol_tn.t
    theta_t = sol_tn.y  # (N, steps)
    Z_micro = np.mean(np.exp(1j * theta_t), axis=0)
    R_tn = np.abs(Z_micro)
    I_micro = J / N * np.sum(s_kernel(n_pulse, theta_t, cn), axis=0)

    # ── 2. OA — manifold approximation for I(t) ──────────────────────────────
    print("Running OA (manifold I) …")
    sol_oa_mf = solve_ivp(
        oa_ode_manifold, (0, T), y0_oa, method=method,
        args=(eta0, Delta, J, n_pulse, s_hat, cn),
        rtol=rtol, atol=atol, dense_output=False,
    )
    assert sol_oa_mf.success, sol_oa_mf.message
    print(f"  Done — {sol_oa_mf.t.size} steps")

    Z_mf = sol_oa_mf.y[0] + 1j * sol_oa_mf.y[1]
    R_mf = np.abs(Z_mf)

    # ── 3. OA — exact empirical I(t) from microscopic trajectory ─────────────
    print("Running OA (exact empirical I) …")

    def theta_interp(t):
        return sol_tn.sol(t)  # dense output from solve_ivp

    sol_oa_ex = solve_ivp(
        oa_ode_exact, (0, T), y0_oa, method=method,
        args=(eta0, Delta, J, n_pulse, cn, theta_interp),
        rtol=rtol, atol=atol, dense_output=False,
    )
    assert sol_oa_ex.success, sol_oa_ex.message
    print(f"  Done — {sol_oa_ex.t.size} steps")

    Z_ex = sol_oa_ex.y[0] + 1j * sol_oa_ex.y[1]
    R_ex = np.abs(Z_ex)

    return dict(
        t_tn=t_tn, R_tn=R_tn, Z_micro=Z_micro, I_micro=I_micro,
        t_mf=sol_oa_mf.t, R_mf=R_mf, Z_mf=Z_mf,
        t_ex=sol_oa_ex.t, R_ex=R_ex, Z_ex=Z_ex,
        eta0=eta0, Delta=Delta, J=J, N=N, n_pulse=n_pulse,
    )


def plot(res):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)
    fig.suptitle(
        f"Single-population theta neuron vs OA  |  "
        f"N={res['N']}, J={res['J']}, η₀={res['eta0']}, "
        f"Δ={res['Delta']}, n={res['n_pulse']}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: |Z(t)|
    ax = axes[0]
    ax.plot(res["t_tn"], res["R_tn"],
            color="steelblue", lw=1.5, label="TN microscopic")
    ax.plot(res["t_mf"], res["R_mf"],
            color="crimson", lw=1.5, ls="--", label="OA (manifold $I$)")
    ax.plot(res["t_ex"], res["R_ex"],
            color="forestgreen", lw=1.5, ls=":", label="OA (exact empirical $I$)")
    ax.set_ylabel("$|Z(t)|$")
    ax.set_ylim(0, 1.05)
    ax.set_title("Order parameter magnitude")
    ax.legend(fontsize=11)

    # Panel 2: Re(Z) and Im(Z) for TN vs OA-manifold
    ax = axes[1]
    ax.plot(res["t_tn"], res["Z_micro"].real,
            color="steelblue", lw=1.2, label=r"TN  Re$(Z)$")
    ax.plot(res["t_tn"], res["Z_micro"].imag,
            color="steelblue", lw=1.2, ls="--", label=r"TN  Im$(Z)$")
    ax.plot(res["t_mf"], res["Z_mf"].real,
            color="crimson", lw=1.2, label=r"OA  Re$(Z)$")
    ax.plot(res["t_mf"], res["Z_mf"].imag,
            color="crimson", lw=1.2, ls="--", label=r"OA  Im$(Z)$")
    ax.set_ylabel("$Z(t)$")
    ax.set_title("Real and imaginary parts of $Z$")
    ax.legend(fontsize=10, ncol=2)

    # Panel 3: mean synaptic input I(t)
    ax = axes[2]
    ax.plot(res["t_tn"], res["I_micro"],
            color="steelblue", lw=1.2, label="$I(t)$ microscopic")
    # OA manifold I(t) — recompute from Z_mf
    cn = coupling_norm(res["n_pulse"])
    s_hat = fourier_coeffs_s(res["n_pulse"])
    I_mf = res["J"] * oa_synaptic_mean(res["Z_mf"], res["n_pulse"], s_hat, cn)
    ax.plot(res["t_mf"], I_mf,
            color="crimson", lw=1.2, ls="--", label="$I(t)$ OA manifold")
    ax.set_ylabel("$I(t)$")
    ax.set_xlabel("Time")
    ax.set_title("Mean synaptic input")
    ax.legend(fontsize=11)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test both n=1 (exact OA closure) and n=2 (higher harmonics)
    for n in [1, 2]:
        res = run(
            N=2000,
            T=100.0,
            J=1.0,
            eta0=-0.5,
            Delta=0.5,
            n_pulse=n,
            seed=42,
            rtol=1e-8,
            atol=1e-9,
        )
        fig = plot(res)
        fig.savefig(f"minimal_oa_comparison_n{n}.png", dpi=150, bbox_inches="tight")
        print(f"Saved minimal_oa_comparison_n{n}.png\n")
    plt.show()

