"""
Theta Neuron Network: Microscopic vs. Ott-Antonsen Macroscopic Comparison
Pulse-based plasticity rules
==========================================================================
Microscopic model (N theta neurons):
    dθ_k/dt  = (1 - cos θ_k) + (1 + cos θ_k) * [η_k + J/N Σ_l A_kl s(n, θ_l)]
    s(n, θ)  = c_n (1 - cos θ)^n,   c_n = 2^n (n!)^2 / (2n)!

Three selectable plasticity rules (parameter `plasticity`):

    "hebbian"
        Ȧ_ij = μ s(n2, θ_i) s(n3, θ_j) - γ A_ij

    "antihebbian"
        Ȧ_ij = μ [s(n1, θ_j) - A_ij s(n2, θ_i) s(n3, θ_j)] - γ A_ij

    "oja"
        Ȧ_ij = μ [s(n2, θ_i) s(n3, θ_j) - A_ij s(n1, θ_i)^2] - γ A_ij

Microscopic excitabilities η_i are sampled from a uniform OR a Lorentzian
distribution with parameters (eta0, Delta0).  The OA reduction partitions
the η axis into M ensembles whose centres η̄_m follow the same distribution
and whose widths Δ_m are chosen to match the spacing between consecutive
centres.

OA mean-field reduction
-----------------------
For each population m we carry the OA order parameter R_m e^{iΨ_m} and the
coarse-grained coupling A_mn.  The dynamics read

    ṙ_m  = -Δ_m R_m - Δ_m(1+R_m²)/2 cosΨ_m
                                 + (E_m-1)(1-R_m²)/2 sinΨ_m
    Ψ̇_m  = (E_m+1) + (E_m-1)(1+R_m²)/(2R_m) cosΨ_m
                  + Δ_m(1-R_m²)/(2R_m) sinΨ_m
    E_m  = η̄_m + (J/M) Σ_n A_mn S_n

with the pulse averages S_q,m := <s(q, θ)>_{Z_m} = c_q Σ_p ŝ_q[p] R_m^p
cos(p Ψ_m).  Plasticity:

    hebbian      :  Ȧ_mn = μ S2_m S3_n - γ A_mn
    antihebbian  :  Ȧ_mn = μ (S1_n - A_mn S2_m S3_n) - γ A_mn
    oja          :  Ȧ_mn = μ (S2_m S3_n - A_mn Ŝsq_m) - γ A_mn

where Ŝsq_m := <s(n1, θ)^2>_{Z_m} is the OA mean of the squared pulse and
is computed analytically from the Fourier coefficients of (1-cosθ)^{2n1}.

Derivation notes (block-averaging on the OA manifold)
- Hebbian: <s(n2, θ_i) s(n3, θ_j)> factorises exactly because i ∈ m and
  j ∈ n are different neurons.
- Antihebbian decay term -A_ij s(n2, θ_i) s(n3, θ_j): using the standard
  closure A_ij ≈ A_mn for i∈m, j∈n, the block average is -A_mn S2_m S3_n.
- Oja decay term -A_ij s(n1, θ_i)^2: identical closure gives -A_mn Ŝsq_m.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from math import factorial, comb
from config.utility_functions import uniform, lorentzian2

_EPS = 1e-12
plt.rcParams["font.size"] = 16.0

PLASTICITY_RULES = ("hebbian", "antihebbian", "oja")


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
    """
    S = np.full_like(R, s_hat[0], dtype=float)
    for p in range(1, n_pulse + 1):
        S += s_hat[p] * R ** p * np.cos(p * Psi)
    return cn * S


def oa_pulse_squared_mean(n_pulse, R, Psi, s_hat_2n, cn):
    """
    <s(n_pulse, θ)^2>_Z = c_n² <(1 - cosθ)^{2 n_pulse}>_Z.

    Implemented by feeding the precomputed Fourier coefficients of
    (1 - cosθ)^{2 n_pulse} (length 2 n_pulse + 1) into the standard OA
    moment expansion.  The factor cn = c_{n_pulse} (NOT c_{2 n_pulse})
    is applied squared, because we are squaring the original pulse, not
    re-normalising as a higher-order pulse.
    """
    two_n = 2 * n_pulse
    S = np.full_like(R, s_hat_2n[0], dtype=float)
    for p in range(1, two_n + 1):
        S += s_hat_2n[p] * R ** p * np.cos(p * Psi)
    return (cn ** 2) * S


def s_micro(n_pulse, theta, cn):
    """Microscopic kernel s(n, θ) = c_n (1-cosθ)^n."""
    return cn * (1.0 - np.cos(theta)) ** n_pulse


# ═══════════════════════════════════════════════════════════════════════════════
# Shared initial conditions
# ═══════════════════════════════════════════════════════════════════════════════

def make_initial_conditions(N, d, eta0, Delta0, dist, seed):
    """
    Draws microscopic excitabilities η_i and initial phases θ_i(0) for N
    neurons partitioned into M = N/d ensembles.  Both the per-ensemble
    centres η̄_m and the per-ensemble microscopic samples η_{i∈m} follow
    the chosen distribution.

    "uniform"      η_{i∈m} ∼ Uniform(η̄_m - Δ_m, η̄_m + Δ_m), Δ_m = Δ₀/M.
    "lorentzian"   η_{i∈m} ∼ Cauchy(η̄_m, Δ_m), where η̄_m and Δ_m are the
                   tangent-warped centres and half-widths so that adjacent
                   centres are spaced by Δ_m + Δ_{m+1}.
    """
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
        if dist == "uniform":
            # Sample η_{i∈m} uniformly across the ensemble's half-width Δ_m,
            # so that the union over m exactly tiles [η₀-Δ₀, η₀+Δ₀].
            eta_micro[idx] = rng.uniform(eta_pop[I] - delta_pop[I],
                                         eta_pop[I] + delta_pop[I], d)
        else:
            eta_micro[idx] = lorentzian2(d, eta_pop[I], delta_pop[I])
        theta0[idx] = th
        z_mean = np.mean(np.exp(1j * th))
        psi0[I] = np.angle(z_mean)
        r0[I] = np.clip(np.abs(z_mean), _EPS, 1.0 - _EPS)

    return eta_micro, theta0, eta_pop, delta_pop, r0, psi0


# ═══════════════════════════════════════════════════════════════════════════════
# Microscopic theta neuron ODE — selectable plasticity rule
# ═══════════════════════════════════════════════════════════════════════════════

def tn_ode(t, y, eta, J, mu, gamma,
           n_pulse, cn,
           n1, cn1,
           n2, cn2,
           n3, cn3,
           plasticity):
    """
    Theta neuron ODE with one of three pulse-based plasticity rules.

    Phase dynamics (identical across rules):
        dθ_k/dt = (1-cosθ_k) + (1+cosθ_k)[η_k + J/N Σ_l A_kl s(n, θ_l)]

    Plasticity (selected by `plasticity`):
        hebbian      Ȧ_kl = μ s(n2, θ_k) s(n3, θ_l)              - γ A_kl
        antihebbian  Ȧ_kl = μ [s(n1, θ_l) - A_kl s(n2,θ_k) s(n3,θ_l)] - γ A_kl
        oja          Ȧ_kl = μ [s(n2,θ_k) s(n3,θ_l) - A_kl s(n1,θ_k)²] - γ A_kl

    Convention: k = post-synaptic (row), l = pre-synaptic (column).  In the
    antihebbian rule s(n1, θ_l) acts as a presynaptic baseline term, so it
    broadcasts across rows.  In the oja rule s(n1, θ_k)² acts as a
    postsynaptic normalisation, so it broadcasts across columns.
    """
    N = len(eta)
    theta = y[:N]
    A = y[N:].reshape(N, N)

    # Synaptic input
    s_syn = s_micro(n_pulse, theta, cn)
    I_syn = (J / N) * (A @ s_syn)

    cm = np.cos(theta)
    dtheta = (1.0 - cm) + (1.0 + cm) * (eta + I_syn)

    # Pre/post-synaptic plasticity kernels (always needed)
    s_post = s_micro(n2, theta, cn2)        # s(n2, θ_k) — row-broadcast
    s_pre  = s_micro(n3, theta, cn3)        # s(n3, θ_l) — col-broadcast
    hebb = np.outer(s_post, s_pre)

    if plasticity == "hebbian":
        dA = mu * hebb - gamma * A
    elif plasticity == "antihebbian":
        s1 = s_micro(n1, theta, cn1)        # s(n1, θ_l) — col-broadcast
        dA = mu * (s1[np.newaxis, :] - A * hebb) - gamma * A
    elif plasticity == "oja":
        s1_sq = s_micro(n1, theta, cn1) ** 2   # s(n1, θ_k)² — row-broadcast
        dA = mu * (hebb - A * s1_sq[:, np.newaxis]) - gamma * A
    else:
        raise ValueError(f"Invalid plasticity='{plasticity}'. "
                         f"Choose from {PLASTICITY_RULES}.")

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
# Macroscopic OA ODE — selectable plasticity rule
# ═══════════════════════════════════════════════════════════════════════════════

def oa_ode(t, y, eta_pop, delta_pop, J, mu, gamma,
           n_pulse, s_hat, cn,
           n1, s_hat1, cn1, s_hat1_sq,
           n2, s_hat2, cn2,
           n3, s_hat3, cn3,
           plasticity):
    """
    OA mean-field ODE with one of three pulse-based plasticity rules.
    State: y = [R_0,...,R_{M-1}, Ψ_0,...,Ψ_{M-1}, A_{0,0},...,A_{M-1,M-1}].

    Plasticity (selected by `plasticity`):
        hebbian      Ȧ_mn = μ S2_m S3_n              - γ A_mn
        antihebbian  Ȧ_mn = μ (S1_n - A_mn S2_m S3_n) - γ A_mn
        oja          Ȧ_mn = μ (S2_m S3_n - A_mn Ŝsq_m) - γ A_mn

    where  S_q,m := <s(q, θ)>_{Z_m}  and  Ŝsq_m := <s(n1, θ)²>_{Z_m}.
    `s_hat1_sq` carries the Fourier coefficients of (1-cosθ)^{2 n1}.
    """
    M = len(eta_pop)
    R = np.clip(y[:M], _EPS, 1.0 - _EPS)
    Psi = y[M:2 * M]
    A = y[2 * M:].reshape(M, M)

    # Mean-field synaptic input
    S  = oa_synaptic_mean(n_pulse, R, Psi, s_hat, cn)
    E  = eta_pop + (J / M) * (A @ S)

    cos_P = np.cos(Psi)
    sin_P = np.sin(Psi)

    dR = (-delta_pop * R
          - delta_pop * (1.0 + R ** 2) / 2.0 * cos_P
          + (E - 1.0) * (1.0 - R ** 2) / 2.0 * sin_P)

    dPsi = ((E + 1.0)
            + (E - 1.0) * (1.0 + R ** 2) / (2.0 * R) * cos_P
            + delta_pop * (1.0 - R ** 2) / (2.0 * R) * sin_P)

    # Pre/post-synaptic plasticity kernel means
    S2 = oa_synaptic_mean(n2, R, Psi, s_hat2, cn2)
    S3 = oa_synaptic_mean(n3, R, Psi, s_hat3, cn3)
    hebb = np.outer(S2, S3)

    if plasticity == "hebbian":
        dA = mu * hebb - gamma * A
    elif plasticity == "antihebbian":
        S1 = oa_synaptic_mean(n1, R, Psi, s_hat1, cn1)        # (M,)
        # S1_n broadcasts across rows m → np.tile(S1, (M,1)) equivalently:
        dA = mu * (S1[np.newaxis, :] - A * hebb) - gamma * A
    elif plasticity == "oja":
        Ssq = oa_pulse_squared_mean(n1, R, Psi, s_hat1_sq, cn1)   # (M,)
        # <s(n1,θ_k)²>_{Z_m} broadcasts across columns n
        dA = mu * (hebb - A * Ssq[:, np.newaxis]) - gamma * A
    else:
        raise ValueError(f"Invalid plasticity='{plasticity}'. "
                         f"Choose from {PLASTICITY_RULES}.")

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
        n_pulse=1,   # synaptic coupling pulse shape
        n1=2,        # plasticity baseline (anti-hebbian) or normalisation (oja)
        n2=2,        # post-synaptic plasticity pulse shape
        n3=3,        # pre-synaptic  plasticity pulse shape
        plasticity="hebbian",
        dist="lorentzian",
        seed=42,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
):
    assert N % d == 0, "N must be divisible by d"
    if plasticity not in PLASTICITY_RULES:
        raise ValueError(f"Invalid plasticity='{plasticity}'. "
                         f"Choose from {PLASTICITY_RULES}.")
    M = N // d

    cn,  s_hat   = coupling_norm(n_pulse), fourier_coeffs_s(n_pulse)
    cn1, s_hat1  = coupling_norm(n1),      fourier_coeffs_s(n1)
    cn2, s_hat2  = coupling_norm(n2),      fourier_coeffs_s(n2)
    cn3, s_hat3  = coupling_norm(n3),      fourier_coeffs_s(n3)
    # Coefficients of (1-cosθ)^{2 n1} for the Oja normalisation term
    s_hat1_sq    = fourier_coeffs_s(2 * n1)

    print(f"Theta neuron (pulse plasticity, rule='{plasticity}'):  "
          f"N={N}, d={d}, M={M}")
    print(f"J={J}, μ={mu}, γ={gamma}, η₀={eta0}, Δ₀={Delta0}, dist={dist}")
    print(f"n (synaptic)={n_pulse}, n1={n1}, n2 (post)={n2}, n3 (pre)={n3}")
    verify_fourier_coeffs(n_pulse, cn, s_hat)
    verify_fourier_coeffs(n2, cn2, s_hat2)
    verify_fourier_coeffs(n3, cn3, s_hat3)
    if plasticity in ("antihebbian", "oja"):
        verify_fourier_coeffs(n1, cn1, s_hat1)

    eta_micro, theta0, eta_pop, delta_pop, r0, psi0 = \
        make_initial_conditions(N, d, eta0, Delta0, dist, seed)

    A0_micro = np.ones((N, N))
    A0_oa = np.ones((M, M))

    # ── Microscopic ──────────────────────────────────────────────────────────
    y0_tn = np.concatenate([theta0, A0_micro.ravel()])
    print("\nRunning TN (microscopic) simulation …")
    sol_tn = solve_ivp(
        tn_ode, (0, T), y0_tn, method=method,
        args=(eta_micro, J, mu, gamma,
              n_pulse, cn,
              n1, cn1,
              n2, cn2,
              n3, cn3,
              plasticity),
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
              n1, s_hat1, cn1, s_hat1_sq,
              n2, s_hat2, cn2,
              n3, s_hat3, cn3,
              plasticity),
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
        n_pulse=n_pulse, n1=n1, n2=n2, n3=n3,
        plasticity=plasticity, dist=dist,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(res):
    N, M, d = res["N"], res["M"], res["d"]
    mu, gamma = res["mu"], res["gamma"]
    n_pulse, n1, n2, n3 = res["n_pulse"], res["n1"], res["n2"], res["n3"]
    rule, dist = res["plasticity"], res["dist"]

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
        f"TN vs OA  |  rule={rule},  dist={dist},  N={N}, d={d}, M={M},  "
        f"μ={mu}, γ={gamma}\n"
        f"n (syn)={n_pulse}, n1={n1}, n2 (post)={n2}, n3 (pre)={n3}"
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
        f"(rule={rule}, dist={dist}, μ={mu}, γ={gamma}, d={d}, "
        f"n={n_pulse}, n1={n1}, n2={n2}, n3={n3})",
        fontsize=12, fontweight="bold",
    )
    fig2.tight_layout()
    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CONFIG = dict(
        N=500,
        d=10,
        T=150.0,
        J=-2.0,
        mu=0.01,
        gamma=0.001,
        eta0=1.0,
        Delta0=1.0,
        n_pulse=10,           # synaptic coupling pulse shape
        n1=10,                 # antihebbian / oja plasticity pulse shape
        n2=2,                 # post-synaptic plasticity kernel
        n3=3,                 # pre-synaptic  plasticity kernel
        plasticity="antihebbian", # {"hebbian", "antihebbian", "oja"}
        dist="uniform",       # {"uniform", "lorentzian"}
        seed=42,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    res = simulate(**CONFIG)
    fig1, fig2 = plot_comparison(res)

    rule = CONFIG["plasticity"]
    fig1.savefig(f"tn_pulse_plasticity_{rule}_order_parameter.png",
                 dpi=150, bbox_inches="tight")
    fig2.savefig(f"tn_pulse_plasticity_{rule}_coupling_matrices.png",
                 dpi=150, bbox_inches="tight")
    print("Figures saved.")
    plt.show()