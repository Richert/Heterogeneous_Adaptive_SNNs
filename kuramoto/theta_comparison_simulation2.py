"""
Theta Neuron Network: Microscopic vs. OA Complex-Z Macroscopic Comparison
==========================================================================
Compares two parallel simulations with matched initial conditions:

  1. Microscopic (TN):  N theta neurons
         dθ_k/dt = 1 - cos(θ_k) + [1 + cos(θ_k)] * [η_k + J/N Σ_l A_kl s(n,θ_l)]
         dA_kl/dt = μ f(θ_l - θ_k) - γ A_kl
         s(n,θ)  = c_n (1 - cos θ)^n,    c_n = 2^n (n!)² / (2n)!

  2. Macroscopic (OA):  M = N/d populations, Z_m = x_m + i y_m integrated directly
         Ż_m = ½ [(iη̄_m + iJ S_m - Δ_m)(1 + Z_m)² + i(1 - Z_m)²]
         Ȧ_mn = μ R_m R_n f_OA(Ψ_n - Ψ_m) - γ A_mn

         where  S_m   = Σ_n A_mn · <s(n_pulse, θ)>_{Z_n}   (mean-field synaptic input)
                R_m   = |Z_m|,   Ψ_m = arg(Z_m)

  The mean-field average <s(n,θ)>_Z is computed via the Fourier expansion of
  (1-cosθ)^n on the OA (Poisson kernel) manifold where <e^{ipθ}> = Z^p:

         <s(n,θ)>_Z = c_n · Re[s_hat[0] + Σ_{p=1}^{n} s_hat[p] · Z^p]

  where s_hat[p] are the (complex) Fourier coefficients of (1-cosθ)^n.
  Note: since (1-cosθ)^n is real and even, all s_hat[p] are real, and
  Re[s_hat[p] Z^p] = s_hat[p] R^p cos(pΨ), recovering the previous formula.
  However, keeping Z complex avoids all manual trig separation.

Key advantage over the (R, Ψ) formulation
------------------------------------------
  The complex Z ODE is integrated directly without any separation into
  amplitude/phase components, eliminating the 1/R singularity at R→0 and
  removing any risk of sign errors in the trig expansion.

  The global order parameter is recovered as:
         R_global(t) = |mean_m Z_m(t)|

Initial condition matching
--------------------------
  Z_m^0 = empirical mean complex phase of block m:
         Z_m^0 = (1/d) Σ_{k∈m} e^{iθ_k}
  This is consistent with the OA manifold definition Z = <e^{iθ}>.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from math import factorial, comb
from config.utility_functions import uniform, lorentzian2

_EPS = 1e-12
plt.rcParams["font.size"] = 16.0


# ═══════════════════════════════════════════════════════════════════════════════
# Synaptic coupling kernel and OA mean-field average
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_norm(n):
    """c_n = 2^n (n!)^2 / (2n)!"""
    return (2 ** n * factorial(n) ** 2) / factorial(2 * n)


def fourier_coeffs_s(n):
    """
    Real Fourier cosine coefficients of (1 - cos θ)^n.

    Returns s_hat of length n+1:
        s_hat[0] = DC coefficient
        s_hat[p] = amplitude of cos(pθ) for p = 1,...,n

    All coefficients are real because (1-cosθ)^n is even and real.
    Verified by verify_fourier_coeffs().
    """
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


def verify_fourier_coeffs(n, cn, s_hat, n_quad=10000):
    """Print numerical vs analytical Fourier coefficients of s(n,θ)."""
    theta = np.linspace(-np.pi, np.pi, n_quad, endpoint=False)
    s_num = cn * (1.0 - np.cos(theta)) ** n
    print(f"\n--- Fourier verification (n_pulse={n}) ---")
    print(f"  p=0: numerical={s_num.mean():.8f},  "
          f"analytical={cn * s_hat[0]:.8f}")
    for p in range(1, min(n + 1, 5)):
        amp_num = 2.0 * (s_num * np.cos(p * theta)).mean()
        print(f"  p={p}: numerical={amp_num:.8f},  "
              f"analytical={cn * s_hat[p]:.8f}")
    print()


def oa_synaptic_mean_complex(n_pulse, Z, s_hat, cn):
    """
    Mean-field average of s(n_pulse, θ) for an array of complex order
    parameters Z (shape (M,)).

    On the OA manifold <e^{ipθ}> = Z^p, so:

        <s> = c_n * Re[ s_hat[0] + Σ_{p=1}^{n} s_hat[p] * Z^p ]

    Since s_hat[p] is real and Z^p = R^p e^{ipΨ}, this equals
    c_n * [s_hat[0] + Σ_p s_hat[p] R^p cos(pΨ)] — same as before
    but computed without explicitly extracting R and Ψ.
    """
    S = np.full(len(Z), s_hat[0], dtype=complex)
    Zp = Z.copy()  # Z^1
    for p in range(1, n_pulse + 1):
        S += s_hat[p] * Zp
        Zp *= Z  # Z^{p+1}
    return cn * S.real  # imaginary part vanishes by symmetry


def s_micro(n_pulse, theta, cn):
    return cn * (1.0 - np.cos(theta)) ** n_pulse


# ═══════════════════════════════════════════════════════════════════════════════
# Shared initial conditions
# ═══════════════════════════════════════════════════════════════════════════════

def make_initial_conditions(N, d, eta0, Delta0, dist, seed):
    assert N % d == 0, "N must be divisible by d"
    M = N // d
    rng = np.random.default_rng(seed)

    if dist == "uniform":
        eta_pop = uniform(M, eta0, Delta0)
        delta_pop = np.full(M, Delta0 / M)
    elif dist == "lorentzian":
        n_idx = np.arange(1, M + 1)
        eta_pop = eta0 + Delta0 * np.tan(0.5 * np.pi * (2 * n_idx - M - 1) / (M + 1))
        delta_pop = Delta0 * (np.tan(0.5 * np.pi * (2 * n_idx - M - 0.5) / (M + 1))
                              - np.tan(0.5 * np.pi * (2 * n_idx - M - 1.5) / (M + 1)))
    else:
        raise ValueError(f"Invalid dist='{dist}'")

    eta_micro = np.empty(N)
    theta0 = np.empty(N)
    Z0 = np.empty(M, dtype=complex)  # complex initial OA state

    for I in range(M):
        th = rng.uniform(-np.pi, np.pi, d)
        idx = slice(I * d, (I + 1) * d)
        eta_micro[idx] = lorentzian2(d, eta_pop[I], delta_pop[I])
        theta0[idx] = th
        # Z_m^0 = (1/d) Σ_{k∈m} e^{iθ_k}  — normalised empirical order parameter
        Z0[I] = np.mean(np.exp(1j * th))

    return eta_micro, theta0, eta_pop, delta_pop, Z0


# ═══════════════════════════════════════════════════════════════════════════════
# Plasticity rules
# ═══════════════════════════════════════════════════════════════════════════════

def hebbian(x):
    return np.cos(x)


def antihebbian(x):
    return np.sin(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Microscopic theta neuron ODE  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def tn_ode(t, y, eta, J, mu, gamma, n_pulse, cn, f):
    N = len(eta)
    theta = y[:N]
    A = y[N:].reshape(N, N)

    s_vec = s_micro(n_pulse, theta, cn)
    I_syn = (J / N) * (A @ s_vec)

    cm = np.cos(theta)
    dtheta = (1.0 - cm) + (1.0 + cm) * (eta + I_syn)

    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    dA = mu * f(diff) - gamma * A
    np.fill_diagonal(dA, 0.0)

    return np.concatenate([dtheta, dA.ravel()])


def tn_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


def tn_coarse_grain(A_fine, d):
    N = A_fine.shape[0]
    M = N // d
    Ac = np.zeros((M, M))
    for I in range(M):
        for Jj in range(M):
            Ac[I, Jj] = A_fine[I * d:(I + 1) * d, Jj * d:(Jj + 1) * d].mean()
    return Ac


# ═══════════════════════════════════════════════════════════════════════════════
# Macroscopic OA ODE — complex Z formulation
# ═══════════════════════════════════════════════════════════════════════════════

def oa_ode_complex(t, y, eta_pop, delta_pop, J, mu, gamma, n_pulse, s_hat, cn, f_oa):
    """
    OA mean-field ODE integrated in complex Z representation.

    State vector (all real):
        y = [x_0,...,x_{M-1}, y_0,...,y_{M-1}, A_00,...,A_{M-1,M-1}]
    where Z_m = x_m + i y_m.

    ODE:
        Ż_m = ½ [(iη̄_m + iJ S_m - Δ_m)(1+Z_m)² + i(1-Z_m)²]
        Ȧ_mn = μ |Z_m| |Z_n| f_OA(arg(Z_n) - arg(Z_m)) - γ A_mn
    """
    M = len(eta_pop)
    x = y[:M]
    yy = y[M:2 * M]
    A = y[2 * M:].reshape(M, M)

    Z = x + 1j * yy  # (M,) complex

    # Clip magnitude to avoid runaway (OA manifold requires |Z| < 1)
    absZ = np.abs(Z)
    mask = absZ > 1.0 - _EPS
    Z = np.where(mask, Z / absZ * (1.0 - _EPS), Z)

    # Mean-field synaptic input S_m = Σ_n A_mn <s>_{Z_n}
    S_vec = oa_synaptic_mean_complex(n_pulse, Z, s_hat, cn)  # (M,) real
    E = eta_pop + J * (A @ S_vec)  # (M,) real effective η

    # Complex OA equation:  Ż = ½[(iE - Δ)(1+Z)² + i(1-Z)²]
    one_plus_Z = 1.0 + Z
    one_minus_Z = 1.0 - Z
    coeff = delta_pop - 1j * E # (M,) complex
    Zdot = -0.5 * (coeff * one_plus_Z ** 2 + 1j * one_minus_Z ** 2)

    # Plasticity in terms of R = |Z|, Ψ = arg(Z)
    R = np.abs(Z)
    Psi = np.angle(Z)
    dPsi_mat = Psi[np.newaxis, :] - Psi[:, np.newaxis]  # Ψ_n - Ψ_m
    rr = R[:, np.newaxis] * R[np.newaxis, :]
    dA = mu * rr * f_oa(dPsi_mat) - gamma * A

    return np.concatenate([Zdot.real, Zdot.imag, dA.ravel()])


def oa_order_parameter_complex(Z):
    """Global order parameter |mean_m Z_m| from complex Z array (M, steps)."""
    return np.abs(np.mean(Z, axis=0))


# ═══════════════════════════════════════════════════════════════════════════════
# Combined simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(
        N=500,
        d=500,
        T=200.0,
        J=1.0,
        mu=0.0,
        gamma=0.0,
        eta0=-0.5,
        Delta0=0.5,
        n_pulse=2,
        plasticity="hebbian",
        dist="lorentzian",
        seed=42,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
):
    assert N % d == 0, "N must be divisible by d"
    M = N // d
    cn = coupling_norm(n_pulse)
    s_hat = fourier_coeffs_s(n_pulse)

    print(f"Theta neuron network (complex-Z OA): N={N}, d={d}, M={M}")
    print(f"J={J}, μ={mu}, γ={gamma}, η₀={eta0}, Δ₀={Delta0}, n_pulse={n_pulse}")
    print(f"c_n = {cn:.6f}")
    verify_fourier_coeffs(n_pulse, cn, s_hat)

    f = hebbian if plasticity == "hebbian" else antihebbian
    f_oa = hebbian if plasticity == "hebbian" else antihebbian

    eta_micro, theta0, eta_pop, delta_pop, Z0 = \
        make_initial_conditions(N, d, eta0, Delta0, dist, seed)

    A0_micro = np.ones((N, N))
    A0_oa = np.ones((M, M))

    # ── Microscopic ──────────────────────────────────────────────────────────
    y0_tn = np.concatenate([theta0, A0_micro.ravel()])
    print("Running TN (microscopic) simulation …")
    sol_tn = solve_ivp(
        tn_ode, (0, T), y0_tn, method=method,
        args=(eta_micro, J, mu, gamma, n_pulse, cn, f),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_tn.success:
        raise RuntimeError(f"TN failed: {sol_tn.message}")
    print(f"  Done — {sol_tn.t.size} steps, {sol_tn.nfev} evaluations")

    t_tn = sol_tn.t
    theta = sol_tn.y[:N]
    A_tn = sol_tn.y[N:].reshape(N, N, -1)
    R_tn = tn_order_parameter(theta)

    # ── Macroscopic OA (complex Z) ───────────────────────────────────────────
    # State: [Re(Z_0),...,Re(Z_{M-1}), Im(Z_0),...,Im(Z_{M-1}), A.ravel()]
    y0_oa = np.concatenate([Z0.real, Z0.imag, A0_oa.ravel()])
    print("Running OA (complex-Z) simulation …")
    sol_oa = solve_ivp(
        oa_ode_complex, (0, T), y0_oa, method=method,
        args=(eta_pop, delta_pop, J, mu, gamma, n_pulse, s_hat, cn, f_oa),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_oa.success:
        raise RuntimeError(f"OA failed: {sol_oa.message}")
    print(f"  Done — {sol_oa.t.size} steps, {sol_oa.nfev} evaluations")

    t_oa = sol_oa.t
    Z_oa = sol_oa.y[:M] + 1j * sol_oa.y[M:2 * M]  # (M, steps) complex
    A_oa = sol_oa.y[2 * M:].reshape(M, M, -1)
    R_oa = oa_order_parameter_complex(Z_oa)  # (steps,) global |<Z_m>|

    return dict(
        t_tn=t_tn, theta=theta, A_tn=A_tn, R_tn=R_tn,
        eta_micro=eta_micro,
        t_oa=t_oa, Z_oa=Z_oa, A_oa=A_oa, R_oa=R_oa,
        eta_pop=eta_pop, delta_pop=delta_pop,
        N=N, M=M, d=d, mu=mu, gamma=gamma, T=T,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(res):
    N, M, d = res["N"], res["M"], res["d"]
    mu, gamma = res["mu"], res["gamma"]

    t_tn, R_tn = res["t_tn"], res["R_tn"]
    t_oa, R_oa = res["t_oa"], res["R_oa"]

    A_tn_final = res["A_tn"][:, :, -1]
    A_oa_final = res["A_oa"][:, :, -1]
    A_tn_cg = tn_coarse_grain(A_tn_final, d)

    # Figure 1: order parameter
    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_tn, R_tn, color="steelblue", lw=1.8, label=f"TN  (N={N})")
    ax.plot(t_oa, R_oa, color="crimson", lw=1.8, ls="--",
            label=f"OA complex-Z  (M={M}, d={d})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Global order parameter $R(t)$")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_title(f"TN vs OA (complex Z)  |  N={N}, d={d}, M={M},  "
                 f"μ={mu}, γ={gamma}")
    ax.legend(fontsize=11)
    fig1.tight_layout()

    # Figure 2: coupling matrices
    vmax_full = np.abs(A_tn_final).max() or 1.0
    vmax_cg = max(np.abs(A_tn_cg).max(), np.abs(A_oa_final).max()) or 1.0

    fig2 = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)

    ax0 = fig2.add_subplot(gs[0])
    im0 = ax0.imshow(A_tn_final, cmap="RdBu_r", vmin=-vmax_full, vmax=vmax_full,
                     interpolation="nearest", aspect="auto")
    plt.colorbar(im0, ax=ax0, label="$A_{kl}$", shrink=0.85)
    for tick in range(0, N + 1, d):
        ax0.axhline(tick - 0.5, color="k", lw=0.4)
        ax0.axvline(tick - 0.5, color="k", lw=0.4)
    ax0.set_title(f"TN $A^{{TN}}$  ({N}×{N})")
    ax0.set_xlabel("Neuron $l$");
    ax0.set_ylabel("Neuron $k$")

    ax1 = fig2.add_subplot(gs[1])
    im1 = ax1.imshow(A_tn_cg, cmap="RdBu_r", vmin=-vmax_cg, vmax=vmax_cg,
                     interpolation="nearest", aspect="equal")
    plt.colorbar(im1, ax=ax1, label="$\\bar{A}_{mn}$", shrink=0.85)
    ax1.set_xticks(range(M));
    ax1.set_yticks(range(M))
    ax1.set_title(f"TN block-avg $\\bar{{A}}^{{TN}}$  ({M}×{M})")
    ax1.set_xlabel("Pop $n$");
    ax1.set_ylabel("Pop $m$")

    ax2 = fig2.add_subplot(gs[2])
    im2 = ax2.imshow(A_oa_final, cmap="RdBu_r", vmin=-vmax_cg, vmax=vmax_cg,
                     interpolation="nearest", aspect="equal")
    plt.colorbar(im2, ax=ax2, label="$A_{mn}$", shrink=0.85)
    ax2.set_xticks(range(M));
    ax2.set_yticks(range(M))
    ax2.set_title(f"OA $A^{{OA}}$  ({M}×{M})")
    ax2.set_xlabel("Pop $n$");
    ax2.set_ylabel("Pop $m$")

    fig2.suptitle(
        f"Final coupling matrices T={res['T']:.0f}  (μ={mu}, γ={gamma}, d={d})",
        fontsize=12, fontweight="bold")
    fig2.tight_layout()
    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    d = 400
    CONFIG = dict(
        N=1 * d,
        d=d,
        T=200.0,
        J=2.0,
        mu=0.0,
        gamma=0.0,
        eta0=-0.5,
        Delta0=1.0,
        n_pulse=1,
        plasticity="hebbian",
        dist="lorentzian",
        seed=42,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    res = simulate(**CONFIG)
    fig1, fig2 = plot_comparison(res)

    fig1.savefig("tn_complexZ_order_parameter.png", dpi=150, bbox_inches="tight")
    fig2.savefig("tn_complexZ_coupling_matrices.png", dpi=150, bbox_inches="tight")
    print("Figures saved.")
    plt.show()

