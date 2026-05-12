"""
Theta Neuron Network: Microscopic vs. Ott-Antonsen Macroscopic Comparison
Pulse-based plasticity rule
==========================================================================
Microscopic model (N theta neurons):
    dθ_k/dt  = (1 - cos θ_k) + (1 + cos θ_k) * [η_k + J/N Σ_l A_kl s(n, θ_l)]
    dA_kl/dt = μ s(n2, θ_k) s(n3, θ_l) - γ A_kl
    s(n, θ)  = c_n (1 - cos θ)^n,   c_n = 2^n (n!)^2 / (2n)!

    Three separate pulse parameters:
        n      : synaptic coupling kernel
        n2     : pre-synaptic plasticity kernel  (multiplies θ_k, the POST-synaptic neuron)
        n3     : post-synaptic plasticity kernel (multiplies θ_l, the PRE-synaptic neuron)

OA mean-field (M populations, state [R_m, Ψ_m, A_mn]):
    ṙ_m  = -Δ_m R_m - Δ_m(1+R_m²)/2 cos Ψ_m + (E_m-1)(1-R_m²)/2 sin Ψ_m
    Ψ̇_m  = (E_m+1) + (E_m-1)(1+R_m²)/(2R_m) cos Ψ_m + Δ_m(1-R_m²)/(2R_m) sin Ψ_m
    Ȧ_mn = μ S2_m S3_n - γ A_mn

    where  E_m  = η̄_m + J/M Σ_n A_mn S_n      (synaptic drive)
           S_n  = <s(n,  θ)>_{Z_n}              (mean coupling kernel,   pulse n)
           S2_m = <s(n2, θ)>_{Z_m}              (mean pre-syn  kernel,   pulse n2)
           S3_n = <s(n3, θ)>_{Z_n}              (mean post-syn kernel,   pulse n3)

Derivation of the OA plasticity rule
--------------------------------------
    Ȧ_mn = μ/d² Σ_{k∈m} Σ_{l∈n} s(n2,θ_k) s(n3,θ_l) - γ A_mn
          = μ [1/d Σ_{k∈m} s(n2,θ_k)] [1/d Σ_{l∈n} s(n3,θ_l)] - γ A_mn
          →  μ <s(n2,θ)>_{Z_m} <s(n3,θ)>_{Z_n} - γ A_mn     (OA manifold)

    The factorisation is exact because s(n2,θ_k) and s(n3,θ_l) depend on
    different neurons (k∈m vs l∈n), so the double sum always separates.
    Unlike cos/sin phase-difference rules, this rule is NOT antisymmetric
    in general (unless n2 == n3).
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
# Pulse kernels and Fourier coefficients
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_norm(n):
    """c_n = 2^n (n!)^2 / (2n)!"""
    return (2 ** n * factorial(n) ** 2) / factorial(2 * n)


def fourier_coeffs_s(n):
    """
    Real Fourier cosine coefficients of (1-cosθ)^n.
    Returns s_hat of shape (n+1,):
        s_hat[0] = DC,  s_hat[p] = amplitude of cos(pθ) for p>=1.
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
    theta = np.linspace(-np.pi, np.pi, n_quad, endpoint=False)
    s_num = cn * (1.0 - np.cos(theta)) ** n
    print(f"\n--- Fourier verification (n={n}, c_n={cn:.6f}) ---")
    print(f"  p=0: numerical={s_num.mean():.8f},  analytical={cn * s_hat[0]:.8f}")
    for p in range(1, min(n + 1, 5)):
        amp_num = 2.0 * (s_num * np.cos(p * theta)).mean()
        print(f"  p={p}: numerical={amp_num:.8f},  analytical={cn * s_hat[p]:.8f}")


def oa_synaptic_mean(n_pulse, R, Psi, s_hat, cn):
    """
    <s(n_pulse, θ)>_Z on the OA manifold for Z = R exp(iΨ).
    Returns real array of shape matching R.
    """
    S = np.full_like(R, s_hat[0], dtype=float)
    for p in range(1, n_pulse + 1):
        S += s_hat[p] * R ** p * np.cos(p * Psi)
    return cn * S


def s_micro(n_pulse, theta, cn):
    """Microscopic kernel s(n, θ) = c_n (1-cosθ)^n."""
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
    r0 = np.empty(M)
    psi0 = np.empty(M)

    for I in range(M):
        th = rng.uniform(-np.pi, np.pi, d)
        idx = slice(I * d, (I + 1) * d)
        eta_micro[idx] = lorentzian2(d, eta_pop[I], delta_pop[I])
        theta0[idx] = th
        z_mean = np.mean(np.exp(1j * th))
        psi0[I] = np.angle(z_mean)
        r0[I] = np.clip(np.abs(z_mean), _EPS, 1.0 - _EPS)

    return eta_micro, theta0, eta_pop, delta_pop, r0, psi0


# ═══════════════════════════════════════════════════════════════════════════════
# Microscopic theta neuron ODE — pulse plasticity
# ═══════════════════════════════════════════════════════════════════════════════

def tn_ode(t, y, eta, J, mu, gamma, n_pulse, cn, n2, cn2, n3, cn3):
    """
    Theta neuron ODE with pulse-based plasticity.

    dθ_k/dt  = (1-cosθ_k) + (1+cosθ_k)[η_k + J/N Σ_l A_kl s(n,θ_l)]
    dA_kl/dt = μ s(n2,θ_k) s(n3,θ_l) - γ A_kl   (diagonal zeroed)

    Note: k is the post-synaptic (row) index, l is the pre-synaptic (col) index.
    s(n2,θ_k) — post-synaptic factor — broadcasts over columns.
    s(n3,θ_l) — pre-synaptic factor  — broadcasts over rows.
    """
    N = len(eta)
    theta = y[:N]
    A = y[N:].reshape(N, N)

    # Synaptic input
    s_syn = s_micro(n_pulse, theta, cn)  # (N,)
    I_syn = (J / N) * (A @ s_syn)  # (N,)

    cm = np.cos(theta)
    dtheta = (1.0 - cm) + (1.0 + cm) * (eta + I_syn)

    # Pulse-based plasticity: outer product of pre- and post-synaptic kernels
    s_post = s_micro(n2, theta, cn2)  # (N,) post-synaptic (row)
    s_pre = s_micro(n3, theta, cn3)  # (N,) pre-synaptic  (col)
    dA = mu * np.outer(s_post, s_pre) - gamma * A
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
# Macroscopic OA ODE — pulse plasticity
# ═══════════════════════════════════════════════════════════════════════════════

def oa_ode(t, y, eta_pop, delta_pop, J, mu, gamma,
           n_pulse, s_hat, cn,
           n2, s_hat2, cn2,
           n3, s_hat3, cn3):
    """
    OA mean-field ODE with pulse-based plasticity.

    State: y = [R_0,...,R_{M-1}, Ψ_0,...,Ψ_{M-1}, A_00,...,A_{M-1,M-1}]

    ṙ_m  = -Δ_m R_m - Δ_m(1+R_m²)/2 cosΨ_m + (E_m-1)(1-R_m²)/2 sinΨ_m
    Ψ̇_m  = (E_m+1) + (E_m-1)(1+R_m²)/(2R_m) cosΨ_m + Δ_m(1-R_m²)/(2R_m) sinΨ_m
    Ȧ_mn  = μ S2_m S3_n - γ A_mn

    where:
        E_m  = η̄_m + J/M Σ_n A_mn S_n     S_n  = <s(n,  θ)>_{Z_n}
        S2_m = <s(n2, θ)>_{Z_m}             (post-synaptic plasticity kernel)
        S3_n = <s(n3, θ)>_{Z_n}             (pre-synaptic  plasticity kernel)
    """
    M = len(eta_pop)
    R = np.clip(y[:M], _EPS, 1.0 - _EPS)
    Psi = y[M:2 * M]
    A = y[2 * M:].reshape(M, M)

    # Mean-field synaptic input
    S = oa_synaptic_mean(n_pulse, R, Psi, s_hat, cn)  # (M,) coupling kernel
    E = eta_pop + (J / M) * (A @ S)  # (M,) effective drive

    cos_P = np.cos(Psi)
    sin_P = np.sin(Psi)

    # OA equations (corrected signs from theta-neuron derivation)
    dR = (-delta_pop * R
          - delta_pop * (1.0 + R ** 2) / 2.0 * cos_P
          + (E - 1.0) * (1.0 - R ** 2) / 2.0 * sin_P)

    dPsi = ((E + 1.0)
            + (E - 1.0) * (1.0 + R ** 2) / (2.0 * R) * cos_P
            + delta_pop * (1.0 - R ** 2) / (2.0 * R) * sin_P)

    # Pulse-based plasticity: outer product of population-averaged kernels
    S2 = oa_synaptic_mean(n2, R, Psi, s_hat2, cn2)  # (M,) post-syn kernel
    S3 = oa_synaptic_mean(n3, R, Psi, s_hat3, cn3)  # (M,) pre-syn  kernel
    dA = mu * np.outer(S2, S3) - gamma * A  # (M, M)

    return np.concatenate([dR, dPsi, dA.ravel()])


def oa_order_parameter(R, Psi):
    return np.abs(np.mean(R * np.exp(1j * Psi), axis=0))


# ═══════════════════════════════════════════════════════════════════════════════
# Combined simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(
        N=500,
        d=50,
        T=200.0,
        J=1.0,
        mu=0.1,
        gamma=0.001,
        eta0=-0.5,
        Delta0=1.0,
        n_pulse=1,  # synaptic coupling pulse shape
        n2=2,  # post-synaptic plasticity pulse shape
        n3=3,  # pre-synaptic  plasticity pulse shape
        dist="lorentzian",
        seed=42,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
):
    assert N % d == 0, "N must be divisible by d"
    M = N // d

    # Precompute all three sets of Fourier coefficients and norms
    cn, s_hat = coupling_norm(n_pulse), fourier_coeffs_s(n_pulse)
    cn2, s_hat2 = coupling_norm(n2), fourier_coeffs_s(n2)
    cn3, s_hat3 = coupling_norm(n3), fourier_coeffs_s(n3)

    print(f"Theta neuron (pulse plasticity): N={N}, d={d}, M={M}")
    print(f"J={J}, μ={mu}, γ={gamma}, η₀={eta0}, Δ₀={Delta0}")
    print(f"n (synaptic)={n_pulse}, n2 (post-syn)={n2}, n3 (pre-syn)={n3}")
    verify_fourier_coeffs(n_pulse, cn, s_hat)
    verify_fourier_coeffs(n2, cn2, s_hat2)
    verify_fourier_coeffs(n3, cn3, s_hat3)

    eta_micro, theta0, eta_pop, delta_pop, r0, psi0 = \
        make_initial_conditions(N, d, eta0, Delta0, dist, seed)

    A0_micro = np.ones((N, N))
    A0_oa = np.ones((M, M))

    # ── Microscopic ──────────────────────────────────────────────────────────
    y0_tn = np.concatenate([theta0, A0_micro.ravel()])
    print("\nRunning TN (microscopic) simulation …")
    sol_tn = solve_ivp(
        tn_ode, (0, T), y0_tn, method=method,
        args=(eta_micro, J, mu, gamma, n_pulse, cn, n2, cn2, n3, cn3),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_tn.success:
        raise RuntimeError(f"TN failed: {sol_tn.message}")
    print(f"  Done — {sol_tn.t.size} steps, {sol_tn.nfev} evaluations")

    t_tn = sol_tn.t
    theta = sol_tn.y[:N]
    A_tn = sol_tn.y[N:].reshape(N, N, -1)
    R_tn = tn_order_parameter(theta)

    # ── Macroscopic OA ───────────────────────────────────────────────────────
    y0_oa = np.concatenate([r0, psi0, A0_oa.ravel()])
    print("Running OA (macroscopic) simulation …")
    sol_oa = solve_ivp(
        oa_ode, (0, T), y0_oa, method=method,
        args=(eta_pop, delta_pop, J, mu, gamma,
              n_pulse, s_hat, cn,
              n2, s_hat2, cn2,
              n3, s_hat3, cn3),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_oa.success:
        raise RuntimeError(f"OA failed: {sol_oa.message}")
    print(f"  Done — {sol_oa.t.size} steps, {sol_oa.nfev} evaluations")

    t_oa = sol_oa.t
    R_oa_m = sol_oa.y[:M]
    Psi_oa = sol_oa.y[M:2 * M]
    A_oa = sol_oa.y[2 * M:].reshape(M, M, -1)
    R_oa = oa_order_parameter(R_oa_m, Psi_oa)

    return dict(
        t_tn=t_tn, theta=theta, A_tn=A_tn, R_tn=R_tn,
        eta_micro=eta_micro,
        t_oa=t_oa, R_oa_m=R_oa_m, Psi_oa=Psi_oa, A_oa=A_oa, R_oa=R_oa,
        eta_pop=eta_pop, delta_pop=delta_pop,
        N=N, M=M, d=d, mu=mu, gamma=gamma, T=T,
        n_pulse=n_pulse, n2=n2, n3=n3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(res):
    N, M, d = res["N"], res["M"], res["d"]
    mu, gamma = res["mu"], res["gamma"]
    n_pulse, n2, n3 = res["n_pulse"], res["n2"], res["n3"]

    t_tn, R_tn = res["t_tn"], res["R_tn"]
    t_oa, R_oa = res["t_oa"], res["R_oa"]

    A_tn_final = res["A_tn"][:, :, -1]
    A_oa_final = res["A_oa"][:, :, -1]
    A_tn_cg = tn_coarse_grain(A_tn_final, d)

    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_tn, R_tn, color="steelblue", lw=1.8, label=f"TN  (N={N})")
    ax.plot(t_oa, R_oa, color="crimson", lw=1.8, ls="--",
            label=f"OA  (M={M}, d={d})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Global order parameter $R(t)$")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_title(
        f"TN vs OA  |  N={N}, d={d}, M={M},  μ={mu}, γ={gamma}\n"
        f"n (syn)={n_pulse}, n2 (post)={n2}, n3 (pre)={n3}"
    )
    ax.legend(fontsize=11)
    fig1.tight_layout()

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
    ax0.set_xlabel("Neuron $l$ (pre)");
    ax0.set_ylabel("Neuron $k$ (post)")

    ax1 = fig2.add_subplot(gs[1])
    im1 = ax1.imshow(A_tn_cg, cmap="RdBu_r", vmin=-vmax_cg, vmax=vmax_cg,
                     interpolation="nearest", aspect="equal")
    plt.colorbar(im1, ax=ax1, label="$\\bar{A}_{mn}$", shrink=0.85)
    ax1.set_xticks(range(M));
    ax1.set_yticks(range(M))
    ax1.set_title(f"TN block-avg $\\bar{{A}}^{{TN}}$  ({M}×{M})")
    ax1.set_xlabel("Pop $n$ (pre)");
    ax1.set_ylabel("Pop $m$ (post)")

    ax2 = fig2.add_subplot(gs[2])
    im2 = ax2.imshow(A_oa_final, cmap="RdBu_r", vmin=-vmax_cg, vmax=vmax_cg,
                     interpolation="nearest", aspect="equal")
    plt.colorbar(im2, ax=ax2, label="$A_{mn}$", shrink=0.85)
    ax2.set_xticks(range(M));
    ax2.set_yticks(range(M))
    ax2.set_title(f"OA $A^{{OA}}$  ({M}×{M})")
    ax2.set_xlabel("Pop $n$ (pre)");
    ax2.set_ylabel("Pop $m$ (post)")

    fig2.suptitle(
        f"Final coupling matrices T={res['T']:.0f}  "
        f"(μ={mu}, γ={gamma}, d={d}, n={n_pulse}, n2={n2}, n3={n3})",
        fontsize=12, fontweight="bold",
    )
    fig2.tight_layout()
    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    d = 20
    CONFIG = dict(
        N=10*d,
        d=d,
        T=200.0,
        J=1.0,
        mu=0.03,
        gamma=0.001,
        eta0=-1.0,
        Delta0=1.0,
        n_pulse=10,  # synaptic coupling pulse shape
        n2=2,  # post-synaptic plasticity kernel
        n3=2,  # pre-synaptic  plasticity kernel
        dist="lorentzian",
        seed=42,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    res = simulate(**CONFIG)
    fig1, fig2 = plot_comparison(res)

    fig1.savefig("tn_pulse_plasticity_order_parameter.png", dpi=150, bbox_inches="tight")
    fig2.savefig("tn_pulse_plasticity_coupling_matrices.png", dpi=150, bbox_inches="tight")
    print("Figures saved.")
    plt.show()

