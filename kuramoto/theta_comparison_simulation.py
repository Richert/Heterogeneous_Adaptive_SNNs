"""
Theta Neuron Network: Microscopic vs. Ott-Antonsen Macroscopic Comparison
==========================================================================
[Same model as before — see inline docstrings for equations]

Bugs fixed vs. previous version
---------------------------------
1. fourier_coeffs_s: s_hat[0] (DC term) was never computed (stayed zero),
   causing the OA model to miss the constant background synaptic drive.
   Also the frequency condition was inverted. Rewritten from scratch using
   explicit complex accumulation, then verified against numerical quadrature.

2. oa_order_parameter was not called in simulate(): R_oa was stored as the
   raw (M, steps) array of per-population magnitudes instead of the scalar
   global order parameter |<z_m>|, causing a shape mismatch in plotting.

3. Minor: oa_order_parameter added to the result dict for clean access.
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
# Synaptic coupling kernel and its OA mean-field average
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_norm(n):
    """Normalisation constant c_n = 2^n (n!)^2 / (2n)!"""
    return (2**n * factorial(n)**2) / factorial(2 * n)


def fourier_coeffs_s(n):
    """
    Compute the real Fourier cosine coefficients of (1 - cos θ)^n.

    Expansion:
        (1 - cos θ)^n = Σ_{k=0}^{n} C(n,k)(-1)^k cos^k θ
        cos^k θ = 1/2^k Σ_{j=0}^{k} C(k,j) e^{i(2j-k)θ}

    Accumulate complex coefficients indexed by frequency p ∈ [-n, n],
    then extract the cosine (real) series for p = 0, 1, ..., n.

    Returns s_hat of shape (n+1,):
        s_hat[0]  = DC coefficient  (NOT halved — used as-is in the sum)
        s_hat[p]  = coefficient of cos(pθ) for p ≥ 1

    Verified against numerical quadrature in verify_fourier_coeffs().
    """
    # Complex Fourier coefficients; index p is stored at position p + n
    c = np.zeros(2 * n + 1, dtype=complex)
    for k in range(n + 1):
        binom_nk = comb(n, k) * (-1)**k / 2**k
        for j in range(k + 1):
            p = 2 * j - k          # frequency of this term
            c[p + n] += binom_nk * comb(k, j)

    # Real cosine series: for p>0,  cos(pθ) = (e^{ipθ} + e^{-ipθ})/2
    # so the coefficient of cos(pθ) is 2 * Re(c[p])
    s_hat = np.zeros(n + 1)
    s_hat[0] = c[n].real                          # DC: just Re(c[0])
    for p in range(1, n + 1):
        s_hat[p] = 2.0 * c[p + n].real            # cosine amplitude

    return s_hat


def verify_fourier_coeffs(n, cn, s_hat, n_quad=10000):
    """
    Verify s_hat against numerical quadrature of s(n, θ) = cn*(1-cosθ)^n.
    Prints the DC term and first few cosine amplitudes both ways.
    """
    theta = np.linspace(-np.pi, np.pi, n_quad, endpoint=False)
    s_num = cn * (1.0 - np.cos(theta))**n
    print(f"\n--- Fourier coefficient verification (n={n}) ---")
    # DC
    dc_num = s_num.mean()
    dc_ana = cn * s_hat[0]
    print(f"  p=0: numerical={dc_num:.8f},  analytical={dc_ana:.8f}")
    for p in range(1, min(n + 1, 4)):
        cos_p = np.cos(p * theta)
        amp_num = 2.0 * (s_num * cos_p).mean()
        amp_ana = cn * s_hat[p]
        print(f"  p={p}: numerical={amp_num:.8f},  analytical={amp_ana:.8f}")
    print()


def oa_synaptic_mean(n_pulse, R, Psi, s_hat, cn):
    """
    Mean-field average <s(n_pulse, θ)> over the OA (Poisson kernel) distribution
    with order parameter Z = R exp(iΨ).

    On the OA manifold: <e^{ipθ}> = Z^p = R^p e^{ipΨ}  for p ≥ 0.
    Since s is real and even, only cosine modes contribute:

        S(R, Ψ) = cn * [s_hat[0] + Σ_{p=1}^{n} s_hat[p] * R^p * cos(p*Ψ)]
    """
    S = np.full_like(R, s_hat[0], dtype=float)
    for p in range(1, n_pulse + 1):
        S += s_hat[p] * R**p * np.cos(p * Psi)
    return cn * S


def s_micro(n_pulse, theta, cn):
    """Microscopic synaptic kernel s(n, θ)."""
    return cn * (1.0 - np.cos(theta))**n_pulse


# ═══════════════════════════════════════════════════════════════════════════════
# Shared initial-condition factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_initial_conditions(N, d, eta0, Delta0, dist, seed):
    assert N % d == 0, "N must be divisible by d"
    M = N // d
    rng = np.random.default_rng(seed)

    if dist == "uniform":
        eta_pop   = uniform(M, eta0, Delta0)
        delta_pop = np.full(M, Delta0 / M)
    elif dist == "lorentzian":
        n_idx     = np.arange(1, M + 1)
        eta_pop   = eta0 + Delta0 * np.tan(0.5 * np.pi * (2*n_idx - M - 1) / (M + 1))
        delta_pop = Delta0 * (np.tan(0.5*np.pi*(2*n_idx - M - 0.5) / (M + 1))
                            - np.tan(0.5*np.pi*(2*n_idx - M - 1.5) / (M + 1)))
    else:
        raise ValueError(f"Invalid dist='{dist}'")

    eta_micro = np.empty(N)
    theta0    = np.empty(N)
    r0        = np.empty(M)
    psi0      = np.empty(M)

    for I in range(M):
        th  = rng.uniform(-np.pi, np.pi, d)
        idx = slice(I * d, (I + 1) * d)
        eta_micro[idx] = lorentzian2(d, eta_pop[I], delta_pop[I])
        theta0[idx]    = th
        z_mean  = np.mean(np.exp(1j * th))
        psi0[I] = np.angle(z_mean)
        r0[I]   = np.clip(np.abs(z_mean), _EPS, 1.0 - _EPS)

    return eta_micro, theta0, eta_pop, delta_pop, r0, psi0


# ═══════════════════════════════════════════════════════════════════════════════
# Plasticity rules
# ═══════════════════════════════════════════════════════════════════════════════

def hebbian(x):
    return np.cos(x)

def antihebbian(x):
    return np.sin(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Microscopic theta neuron ODE
# ═══════════════════════════════════════════════════════════════════════════════

def tn_ode(t, y, eta, J, mu, gamma, n_pulse, cn, f):
    N     = len(eta)
    theta = y[:N]
    A     = y[N:].reshape(N, N)

    s_vec  = s_micro(n_pulse, theta, cn)       # (N,)
    I_syn  = (J / N) * (A @ s_vec)             # (N,)

    cm     = np.cos(theta)
    dtheta = (1.0 - cm) + (1.0 + cm) * (eta + I_syn)

    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    dA   = mu * f(diff) - gamma * A
    np.fill_diagonal(dA, 0.0)

    return np.concatenate([dtheta, dA.ravel()])


def tn_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


def tn_coarse_grain(A_fine, d):
    N  = A_fine.shape[0]
    M  = N // d
    Ac = np.zeros((M, M))
    for I in range(M):
        for J in range(M):
            Ac[I, J] = A_fine[I*d:(I+1)*d, J*d:(J+1)*d].mean()
    return Ac


# ═══════════════════════════════════════════════════════════════════════════════
# Macroscopic OA ODE
# ═══════════════════════════════════════════════════════════════════════════════

def oa_ode(t, y, eta_pop, delta_pop, J, mu, gamma, n_pulse, s_hat, cn, f_oa):
    M   = len(eta_pop)
    R   = np.clip(y[:M],   _EPS, 1.0 - _EPS)
    Psi = y[M:2*M]
    A   = y[2*M:].reshape(M, M)

    S   = oa_synaptic_mean(n_pulse, R, Psi, s_hat, cn)   # (M,)
    E   = eta_pop + (J/M) * (A @ S)                           # (M,)

    cos_P = np.cos(Psi)
    sin_P = np.sin(Psi)

    dR   = (-delta_pop * R
            - delta_pop * (1.0 + R**2) / 2.0 * cos_P
            + (E - 1.0) * (1.0 - R**2) / 2.0 * sin_P)

    dPsi = ((E + 1.0)
            + (E - 1.0) * (1.0 + R**2) / (2.0 * R) * cos_P
            + delta_pop * (1.0 - R**2) / (2.0 * R) * sin_P)

    dPsi_mat = Psi[np.newaxis, :] - Psi[:, np.newaxis]
    rr        = R[:, np.newaxis] * R[np.newaxis, :]
    dA        = mu * rr * f_oa(dPsi_mat) - gamma * A

    return np.concatenate([dR, dPsi, dA.ravel()])


def oa_order_parameter(R, Psi):
    """Global OA order parameter |<R_m exp(iΨ_m)>| over populations."""
    return np.abs(np.mean(R * np.exp(1j * Psi), axis=0))


# ═══════════════════════════════════════════════════════════════════════════════
# Combined simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(
    N          = 500,
    d          = 500,
    T          = 200.0,
    J          = 1.0,
    mu         = 0.0,
    gamma      = 0.0,
    eta0       = -0.5,
    Delta0     = 0.5,
    n_pulse    = 2,
    plasticity = "hebbian",
    dist       = "lorentzian",
    seed       = 42,
    method     = "RK45",
    rtol       = 1e-6,
    atol       = 1e-8,
):
    assert N % d == 0, "N must be divisible by d"
    M     = N // d
    cn    = coupling_norm(n_pulse)
    s_hat = fourier_coeffs_s(n_pulse)

    print(f"Theta neuron network: N={N}, d={d}, M={M}")
    print(f"J={J}, μ={mu}, γ={gamma}, η₀={eta0}, Δ₀={Delta0}, n_pulse={n_pulse}")
    print(f"c_n={cn:.6f}")
    verify_fourier_coeffs(n_pulse, cn, s_hat)

    f    = hebbian if plasticity == "hebbian" else antihebbian
    f_oa = hebbian if plasticity == "hebbian" else antihebbian

    eta_micro, theta0, eta_pop, delta_pop, r0, psi0 = \
        make_initial_conditions(N, d, eta0, Delta0, dist, seed)

    A0_micro = np.ones((N, N))
    A0_oa    = np.ones((M, M))

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

    t_tn  = sol_tn.t
    theta = sol_tn.y[:N]
    A_tn  = sol_tn.y[N:].reshape(N, N, -1)
    R_tn  = tn_order_parameter(theta)

    # ── Macroscopic OA ───────────────────────────────────────────────────────
    y0_oa = np.concatenate([r0, psi0, A0_oa.ravel()])
    print("Running OA (macroscopic) simulation …")
    sol_oa = solve_ivp(
        oa_ode, (0, T), y0_oa, method=method,
        args=(eta_pop, delta_pop, J, mu, gamma, n_pulse, s_hat, cn, f_oa),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_oa.success:
        raise RuntimeError(f"OA failed: {sol_oa.message}")
    print(f"  Done — {sol_oa.t.size} steps, {sol_oa.nfev} evaluations")

    t_oa   = sol_oa.t
    R_oa_m = sol_oa.y[:M]                              # (M, steps) per-population
    Psi_oa = sol_oa.y[M:2*M]
    A_oa   = sol_oa.y[2*M:].reshape(M, M, -1)
    # FIX: compute scalar global order parameter correctly
    R_oa   = oa_order_parameter(R_oa_m, Psi_oa)        # (steps,)

    return dict(
        t_tn=t_tn, theta=theta, A_tn=A_tn, R_tn=R_tn,
        eta_micro=eta_micro,
        t_oa=t_oa, R_oa_m=R_oa_m, Psi_oa=Psi_oa, A_oa=A_oa, R_oa=R_oa,
        eta_pop=eta_pop, delta_pop=delta_pop,
        N=N, M=M, d=d, mu=mu, gamma=gamma, T=T,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(res):
    N, M, d   = res["N"], res["M"], res["d"]
    mu, gamma = res["mu"], res["gamma"]

    t_tn, R_tn = res["t_tn"], res["R_tn"]
    t_oa, R_oa = res["t_oa"], res["R_oa"]

    A_tn_final = res["A_tn"][:, :, -1]
    A_oa_final = res["A_oa"][:, :, -1]
    A_tn_cg    = tn_coarse_grain(A_tn_final, d)

    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_tn, R_tn, color="steelblue", lw=1.8, label=f"TN  (N={N})")
    ax.plot(t_oa, R_oa, color="crimson",   lw=1.8, ls="--",
            label=f"OA  (M={M}, d={d})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Global order parameter $R(t)$")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_title(f"TN vs OA  |  N={N}, d={d}, M={M},  μ={mu}, γ={gamma}")
    ax.legend(fontsize=11)
    fig1.tight_layout()

    vmax_full = np.abs(A_tn_final).max() or 1.0
    vmax_cg   = max(np.abs(A_tn_cg).max(), np.abs(A_oa_final).max()) or 1.0

    fig2 = plt.figure(figsize=(20, 8))
    gs   = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)

    ax0 = fig2.add_subplot(gs[0])
    im0 = ax0.imshow(A_tn_final, cmap="RdBu_r", vmin=-vmax_full, vmax=vmax_full,
                     interpolation="nearest", aspect="auto")
    plt.colorbar(im0, ax=ax0, label="$A_{kl}$", shrink=0.85)
    for tick in range(0, N + 1, d):
        ax0.axhline(tick - 0.5, color="k", lw=0.4)
        ax0.axvline(tick - 0.5, color="k", lw=0.4)
    ax0.set_title(f"TN $A^{{TN}}$  ({N}×{N})")
    ax0.set_xlabel("Neuron $l$"); ax0.set_ylabel("Neuron $k$")

    ax1 = fig2.add_subplot(gs[1])
    im1 = ax1.imshow(A_tn_cg, cmap="RdBu_r", vmin=-vmax_cg, vmax=vmax_cg,
                     interpolation="nearest", aspect="equal")
    plt.colorbar(im1, ax=ax1, label="$\\bar{A}_{mn}$", shrink=0.85)
    ax1.set_xticks(range(M)); ax1.set_yticks(range(M))
    ax1.set_title(f"TN block-avg $\\bar{{A}}^{{TN}}$  ({M}×{M})")
    ax1.set_xlabel("Pop $n$"); ax1.set_ylabel("Pop $m$")

    ax2 = fig2.add_subplot(gs[2])
    im2 = ax2.imshow(A_oa_final, cmap="RdBu_r", vmin=-vmax_cg, vmax=vmax_cg,
                     interpolation="nearest", aspect="equal")
    plt.colorbar(im2, ax=ax2, label="$A_{mn}$", shrink=0.85)
    ax2.set_xticks(range(M)); ax2.set_yticks(range(M))
    ax2.set_title(f"OA $A^{{OA}}$  ({M}×{M})")
    ax2.set_xlabel("Pop $n$"); ax2.set_ylabel("Pop $m$")

    fig2.suptitle(f"Final coupling matrices T={res['T']:.0f}  (μ={mu}, γ={gamma}, d={d})",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout()
    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    d = 50
    CONFIG = dict(
        N          = 10 * d,
        d          = d,
        T          = 200.0,
        J          = 1.0,
        mu         = 0.1,
        gamma      = 0.001,
        eta0       = -0.5,
        Delta0     = 1.0,
        n_pulse    = 1,
        plasticity = "antihebbian",
        dist       = "lorentzian",
        seed       = 42,
        method     = "RK45",
        rtol       = 1e-6,
        atol       = 1e-8,
    )

    res = simulate(**CONFIG)
    fig1, fig2 = plot_comparison(res)

    fig1.savefig("tn_comparison_order_parameter.png",  dpi=150, bbox_inches="tight")
    fig2.savefig("tn_comparison_coupling_matrices.png", dpi=150, bbox_inches="tight")
    print("Figures saved.")
    plt.show()

