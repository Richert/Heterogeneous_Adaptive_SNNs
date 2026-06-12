r"""
Microscopic (finite-N) realization of the adaptive Kuramoto–Ott-Antonsen model
==============================================================================

This is the *spiking-level* counterpart of ``kmo_macro_simulation.py``: the
finite-N Kuramoto network whose Ott–Antonsen (OA) mean-field reduction is
exactly that macroscopic model. As N → ∞ (oscillators per population) the
population order parameters of this model converge to the macro ODEs.

Correspondence
--------------
The macro model has M coupled POPULATIONS; population i is an ensemble of phase
oscillators with a Lorentzian natural-frequency distribution (centre ω_i, HWHM
Δ_i), and the inter-population weights A_ij (an M×M matrix) co-evolve. The
microscopic model is therefore:

    θ̇_{i,k} = ω_{i,k} + (K/M) Σ_j A_ij · Im[ z_j e^{-iθ_{i,k}} ]
             = ω_{i,k} + (K/M) Σ_j A_ij r_j sin(ψ_j − θ_{i,k})

    Ȧ_ij    = μ · Re[ z_i* z_j ] − γ A_ij            (Hebbian)
            = μ r_i r_j cos(ψ_j − ψ_i) − γ A_ij

where z_i = r_i e^{iψ_i} = (1/N) Σ_k e^{iθ_{i,k}} is population i's order
parameter and ω_{i,k} ~ Lorentzian(ω_i, Δ_i). Inserting these into the OA
ansatz yields, term for term, the macro equations

    ṙ_i = −Δ_i r_i + ½(1−r_i²)(K/M) Σ_j A_ij r_j cos(ψ_j−ψ_i)
    ψ̇_i = ω_i + ½(1+r_i²)/r_i (K/M) Σ_j A_ij r_j sin(ψ_j−ψ_i)
    Ȧ_ij = μ r_i r_j cos(ψ_j−ψ_i) − γ A_ij

The diagonal weight A_ii (self-coupling of a population to its own mean field)
is kept, exactly as in the macro model. The anti-Hebbian variant uses
``μ |Im[z_i* z_j]| = μ r_i r_j |sin(ψ_j−ψ_i)|``, matching the macro's
``oa_antihebbian_ode``.

How "close" is the match (read this!)
-------------------------------------
The mean-field system is SENSITIVE to the random draw: different ``seed`` values
give qualitatively different attractors (locked / breathing / drifting / torus).
So the micro–macro correspondence must be judged **qualitatively and over
several seeds** — do both show the same *kind* of state (synchronised
equilibrium vs. oscillation vs. multi-frequency drift), comparable order-
parameter levels, and similar weight structure? — NOT by expecting a single
finite-N trajectory to track the macro trajectory point-for-point. (Even at
large N they decorrelate whenever the dynamics is chaotic/quasiperiodic; and at
finite N there is always O(1/√N) sampling noise.) ``compare_macro_micro`` runs a
list of seeds and overlays them for exactly this qualitative read-off.

Performance
-----------
The coupling is MEAN-FIELD: each oscillator only sees the M complex population
fields, so the right-hand side is O(M·N) per step, NOT the O(N²) all-to-all sum
of a naive Kuramoto network. That reformulation — collapsing
Σ_j A_ij sin(θ_j−θ_i) into Im[ e^{-iθ} · (A z) ] — is the real speed-up here.
The remaining linear-algebra (the M×M field H = (K/M)·A z and the M×M Hebbian
drive Re[z_i* z_j]) is a tiny matrix–vector / outer product that NumPy already
dispatches to BLAS (``zgemv``/``zgeru``); at M = O(10) it is utterly negligible
next to the elementwise ``exp``/``sin``/``cos`` over the N oscillators, so a
hand-rolled ``scipy.linalg.blas`` call buys nothing measurable (see
``benchmark_rhs``). If you ever need more speed, the lever is the elementwise
trig (e.g. a numba-jitted RHS), not BLAS.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import kmo_macro_simulation as macro

_EPS = 1e-12


# ── Population-level parameters (mirrors kmo_macro_simulation.simulate's RNG) ──
def draw_population_params(M, omega_mean, omega_std, delta_mean, delta_std,
                           r0_mean, r0_std, psi_mean, psi_std,
                           A0_center, A0_scale, seed):
    """Draw the M-population parameters in the EXACT same RNG order as
    ``kmo_macro_simulation.simulate`` so that, for a given ``seed``, the micro
    and macro models start from the same (ω_i, Δ_i, r_i(0), ψ_i(0), A_ij(0))."""
    rng = np.random.default_rng(seed)
    omega = rng.uniform(-omega_std, omega_std, M) + omega_mean
    delta = np.maximum(0.0, rng.uniform(-delta_std, delta_std, M) + delta_mean)
    r0 = np.clip(rng.uniform(-r0_std, r0_std, M) + r0_mean, _EPS, 1.0 - _EPS)
    psi0 = rng.uniform(-psi_std, psi_std, M) + psi_mean
    if A0_scale > 0:
        A0 = rng.normal(A0_center, A0_scale, (M, M))
        A0 = (A0 + A0.T) / 2.0
    else:
        A0 = np.zeros((M, M)) + A0_center
    return omega, delta, r0, psi0, A0, rng


# ── Within-population sampling: Lorentzian frequencies + wrapped-Cauchy phases ─
def lorentzian_frequencies(omega, delta, N, rng, deterministic=True, trunc=50.0):
    """(M, N) natural frequencies, population i ~ Lorentzian(ω_i, Δ_i), truncated
    at ±``trunc``·Δ_i.

    The Lorentzian (Cauchy) has heavy tails: its extreme quantiles diverge
    (≈ Δ·2N/π for the deterministic grid), which would spawn a handful of
    absurdly fast oscillators and make the integrator stiff for no dynamical
    reason — far-tail oscillators never entrain and contribute ≈0 to the order
    parameter. We therefore restrict the sampled quantiles to the central band
    that keeps |ω_{i,k} − ω_i| ≤ trunc·Δ_i (trunc=50 keeps ~99% of the mass).

    ``deterministic=True`` uses evenly spaced quantiles (low finite-size noise);
    ``False`` draws random Cauchy variates (a genuine finite ensemble, more
    sampling noise). Frequencies are randomly permuted within each population so
    they carry no spurious correlation with the initial phases.
    """
    M = len(omega)
    p0 = 0.5 - np.arctan(trunc) / np.pi          # quantile of −trunc·Δ
    freqs = np.empty((M, N))
    for i in range(M):
        if deterministic:
            p = p0 + (1.0 - 2.0 * p0) * (np.arange(N) + 0.5) / N
            f = omega[i] + delta[i] * np.tan(np.pi * (p - 0.5))
            f = rng.permutation(f)
        else:
            f = omega[i] + delta[i] * rng.standard_cauchy(N)
            cap = trunc * delta[i]
            f = np.clip(f, omega[i] - cap, omega[i] + cap)
        freqs[i] = f
    return freqs


def wrapped_cauchy_phases(psi0, r0, N, rng):
    """(M, N) initial phases; population i drawn from a wrapped-Cauchy (the OA /
    Poisson-kernel density) with mean phase ψ_i(0) and mean resultant length
    r_i(0), so each population starts on the OA manifold with order parameter
    r_i(0) e^{iψ_i(0)} (up to O(1/√N) sampling error)."""
    M = len(psi0)
    th = np.empty((M, N))
    for i in range(M):
        u = rng.uniform(0.0, 1.0, N)
        rho = float(np.clip(r0[i], 0.0, 1.0 - 1e-9))
        th[i] = psi0[i] + 2.0 * np.arctan(((1.0 - rho) / (1.0 + rho))
                                          * np.tan(np.pi * (u - 0.5)))
    return th


# ── State packing ─────────────────────────────────────────────────────────────
def pack(theta_flat, A):
    return np.concatenate([theta_flat, A.ravel()])


def unpack(y, M, N):
    theta = y[:M * N].reshape(M, N)
    A = y[M * N:].reshape(M, M)
    return theta, A


# ── Right-hand side (mean-field, O(M·N)) ──────────────────────────────────────
def micro_rhs(t, y, M, N, omega_flat, K_over_M, mu, gamma, hebbian):
    """RHS of the M-population finite-N adaptive Kuramoto network.

    ``omega_flat`` is the (M·N,) per-oscillator frequency vector in
    population-major order (row i = population i), matching the reshape below.
    """
    theta = y[:M * N].reshape(M, N)
    A = y[M * N:].reshape(M, M)

    e = np.exp(1j * theta)                       # (M, N) e^{iθ}
    z = e.mean(axis=1)                            # (M,) population order parameters
    H = K_over_M * (A @ z)                        # (M,) complex field per pop (BLAS zgemv)

    # θ̇_{i,k} = ω_{i,k} + Im[ H_i e^{-iθ_{i,k}} ]   (every osc in pop i sees H_i)
    dtheta = omega_flat.reshape(M, N) + (H[:, None] * e.conj()).imag

    # Hebbian:  Ȧ_ij = μ Re[z_i* z_j] − γ A_ij ;  anti-Hebbian uses |Im[z_i* z_j]|
    G = np.outer(z.conj(), z)                     # G_ij = z_i* z_j  (BLAS zgeru)
    drive = G.real if hebbian else np.abs(G.imag)
    dA = mu * drive - gamma * A

    out = np.empty_like(y)
    out[:M * N] = dtheta.ravel()
    out[M * N:] = dA.ravel()
    return out


def population_order_parameters(theta_t, M, N):
    """Map a phase trajectory (M·N, steps) to population order parameters
    r (M, steps) and ψ (M, steps)."""
    th = theta_t.reshape(M, N, -1)
    z = np.exp(1j * th).mean(axis=1)              # (M, steps)
    return np.abs(z), np.angle(z)


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate(
    M=5, N=1000, T=500.0,
    K=1.0, mu=0.1, gamma=0.2,
    omega_mean=0.4, omega_std=0.2,
    delta_mean=0.02, delta_std=0.0,
    r0_mean=0.9, r0_std=0.1,
    psi_mean=0.0, psi_std=0.1,
    A0_center=0.2, A0_scale=0.1,
    plasticity="hebbian",
    deterministic_freqs=True, trunc=50.0,
    seed=42,
    method="RK45", rtol=1e-7, atol=1e-9,
    n_save=1000,
):
    """Integrate the microscopic M-population network and return POPULATION-level
    observables, directly comparable to ``kmo_macro_simulation.simulate``.

    Parameters mirror the macro ``simulate`` (same names / RNG order), plus:
      N                   oscillators per population (N → ∞ ⇒ macro)
      deterministic_freqs evenly-spaced vs random Lorentzian sampling
      trunc               Lorentzian-tail truncation in units of Δ
      n_save              number of stored time points (t_eval grid)

    Returns
    -------
    t      : (steps,)
    r      : (M, steps)      per-population order-parameter magnitude
    psi    : (M, steps)      per-population mean phase
    A_traj : (M, M, steps)   coupling weights over time
    omega  : (M,)            population centre frequencies
    delta  : (M,)            population half-widths
    """
    omega, delta, r0, psi0, A0, rng = draw_population_params(
        M, omega_mean, omega_std, delta_mean, delta_std,
        r0_mean, r0_std, psi_mean, psi_std, A0_center, A0_scale, seed)

    omega_micro = lorentzian_frequencies(omega, delta, N, rng,
                                         deterministic=deterministic_freqs, trunc=trunc)
    theta0 = wrapped_cauchy_phases(psi0, r0, N, rng)
    y0 = pack(theta0.ravel(), A0)

    print(f"Micro adaptive Kuramoto — M={M} populations × N={N} = {M*N} oscillators, T={T}")
    print(f"  K={K}, μ={mu}, γ={gamma}   ω∈[{omega.min():.3f},{omega.max():.3f}]  Δ̄={delta.mean():.3f}")
    z0 = np.exp(1j * theta0).mean(axis=1)
    print(f"  initial per-pop r: target {np.array2string(r0, precision=2)} "
          f"→ realized {np.array2string(np.abs(z0), precision=2)}")

    t_eval = np.linspace(0.0, T, n_save)
    sol = solve_ivp(
        micro_rhs, (0.0, T), y0, method=method, t_eval=t_eval,
        args=(M, N, omega_micro.ravel(), K / M, mu, gamma, plasticity == "hebbian"),
        rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    print(f"  Done — {sol.t.size} stored steps, {sol.nfev} RHS evaluations")

    r, psi = population_order_parameters(sol.y[:M * N], M, N)
    A_traj = sol.y[M * N:].reshape(M, M, -1)
    return sol.t, r, psi, A_traj, omega, delta


# ── Macro run on a shared time grid (uses the imported macro RHS) ─────────────
def _simulate_macro_on_grid(t_eval, omega, delta, r0, psi0, A0, K, mu, gamma,
                            plasticity, method, rtol, atol):
    """Integrate the macro OA model from the SAME population params, on the SAME
    time grid, so micro and macro can be overlaid directly."""
    rhs = macro.oa_hebbian_ode if plasticity == "hebbian" else macro.oa_antihebbian_ode
    y0 = macro.pack(r0, psi0, A0)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, method=method, t_eval=t_eval,
                    args=(K, omega, delta, mu, gamma), rtol=rtol, atol=atol)
    M = len(omega)
    r = sol.y[:M, :]
    psi = sol.y[M:2 * M, :]
    A_traj = sol.y[2 * M:, :].reshape(M, M, -1)
    return r, psi, A_traj


# ── Qualitative micro–macro comparison across several seeds ───────────────────
def compare_macro_micro(seeds=(1, 2, 3, 4), M=5, N=1000, T=500.0,
                        K=1.0, mu=0.1, gamma=0.2,
                        omega_mean=0.4, omega_std=0.2,
                        delta_mean=0.02, delta_std=0.0,
                        r0_mean=0.9, r0_std=0.1, psi_mean=0.0, psi_std=0.1,
                        A0_center=0.2, A0_scale=0.1, plasticity="hebbian",
                        deterministic_freqs=True, trunc=50.0,
                        method="RK45", rtol=1e-7, atol=1e-9, n_save=1000,
                        savefig="micro_vs_macro.png"):
    """Run macro and micro from identical population parameters for each seed and
    overlay them. Because the mean-field system is seed-sensitive, this is a
    QUALITATIVE check across seeds — look for the same attractor type and
    comparable order-parameter / weight levels, not point-wise agreement.

    Each seed gets a row: (left) global order parameter R(t) = |⟨z_i⟩| and the
    per-population r_i(t), macro solid vs micro dashed; (right) final weight
    matrices A(T), macro vs micro.
    """
    t_eval = np.linspace(0.0, T, n_save)
    nseed = len(seeds)
    fig, axes = plt.subplots(nseed, 2, figsize=(13, 3.0 * nseed),
                             gridspec_kw={"width_ratios": [2.4, 1.0]}, squeeze=False)

    for row, seed in enumerate(seeds):
        omega, delta, r0, psi0, A0, rng = draw_population_params(
            M, omega_mean, omega_std, delta_mean, delta_std,
            r0_mean, r0_std, psi_mean, psi_std, A0_center, A0_scale, seed)

        # macro on the shared grid
        r_mac, psi_mac, A_mac = _simulate_macro_on_grid(
            t_eval, omega, delta, r0, psi0, A0, K, mu, gamma, plasticity,
            method, rtol, atol)

        # micro from the same population params
        omega_micro = lorentzian_frequencies(omega, delta, N, rng,
                                              deterministic=deterministic_freqs, trunc=trunc)
        theta0 = wrapped_cauchy_phases(psi0, r0, N, rng)
        y0 = pack(theta0.ravel(), A0)
        sol = solve_ivp(micro_rhs, (0.0, T), y0, method=method, t_eval=t_eval,
                        args=(M, N, omega_micro.ravel(), K / M, mu, gamma,
                              plasticity == "hebbian"),
                        rtol=rtol, atol=atol)
        r_mic, psi_mic = population_order_parameters(sol.y[:M * N], M, N)
        A_mic = sol.y[M * N:].reshape(M, M, -1)

        # global (inter-population) order parameter R = |mean_i r_i e^{iψ_i}|
        R_mac = np.abs(np.mean(r_mac * np.exp(1j * psi_mac), axis=0))
        R_mic = np.abs(np.mean(r_mic * np.exp(1j * psi_mic), axis=0))

        axL = axes[row][0]
        for i in range(M):
            axL.plot(t_eval, r_mac[i], color="steelblue", lw=0.8, alpha=0.5)
            axL.plot(t_eval, r_mic[i], color="darkorange", lw=0.8, alpha=0.5, ls="--")
        axL.plot(t_eval, R_mac, color="navy", lw=2.0, label="macro (OA)")
        axL.plot(t_eval, R_mic, color="crimson", lw=2.0, ls="--", label=f"micro (N={N})")
        axL.set_ylim(0, 1.05)
        axL.set_ylabel(f"seed {seed}\norder param")
        if row == 0:
            axL.set_title(r"global $R$ (bold) & per-population $r_i$ (thin): "
                          "macro solid vs micro dashed")
            axL.legend(loc="lower left", fontsize=8)
        if row == nseed - 1:
            axL.set_xlabel("time")

        # final weight matrices, shared colour scale
        vmax = max(np.abs(A_mac[:, :, -1]).max(), np.abs(A_mic[:, :, -1]).max()) or 1.0
        axR = axes[row][1]
        # pack the two MxM matrices side by side
        gap = np.full((M, 1), np.nan)
        combo = np.hstack([A_mac[:, :, -1], gap, A_mic[:, :, -1]])
        im = axR.imshow(combo, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        interpolation="nearest", aspect="auto")
        axR.set_xticks([(M - 1) / 2, M + 1 + (M - 1) / 2])
        axR.set_xticklabels(["macro", "micro"])
        axR.set_yticks([])
        if row == 0:
            axR.set_title(r"final $A_{ij}$")
        fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)

    fig.suptitle(f"Micro (finite-N) vs macro (OA) — qualitative comparison across seeds\n"
                 f"M={M}, K={K}, μ={mu}, γ={gamma}, Δ={delta_mean}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if savefig:
        fig.savefig(savefig, dpi=130, bbox_inches="tight")
        print(f"saved {savefig}")
    return fig


# ── Quick RHS benchmark (substantiates the BLAS note in the docstring) ────────
def benchmark_rhs(M=5, N=2000, repeats=200, seed=0):
    """Time the RHS and its sub-parts to show the elementwise trig dominates and
    the A·z field (BLAS) is negligible — i.e. scipy.linalg.blas would not help."""
    import time
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, (M, N))
    A = rng.normal(0, 0.5, (M, M))
    omega = rng.uniform(-1, 1, M * N)
    y = pack(theta.ravel(), A)

    def timeit(fn):
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return 1e3 * (time.perf_counter() - t0) / repeats

    full = timeit(lambda: micro_rhs(0.0, y, M, N, omega, 1.0 / M, 0.1, 0.2, True))
    expo = timeit(lambda: np.exp(1j * theta))
    z = np.exp(1j * theta).mean(axis=1)
    field = timeit(lambda: A @ z)
    print(f"benchmark_rhs  M={M} N={N} ({M*N} oscillators):")
    print(f"  full RHS         : {full:.3f} ms")
    print(f"  exp(iθ) only     : {expo:.3f} ms  ({100*expo/full:.0f}% of RHS)")
    print(f"  A·z field (BLAS) : {field*1e3:.3f} µs  ({100*field/full:.2f}% of RHS)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    benchmark_rhs(M=5, N=1000)
    # qualitative micro-vs-macro comparison across several seeds
    compare_macro_micro(
        seeds=(1, 2, 4, 8),
        M=1, N=500, T=100.0,
        K=0.5, mu=0.01, gamma=0.0,
        omega_mean=0.0, omega_std=0.2, delta_mean=0.02, delta_std=0.0,
        r0_mean=0.9, r0_std=0.1, psi_mean=0.0, psi_std=0.1,
        A0_center=0.2, A0_scale=0.1, plasticity="hebbian",
    )
    plt.show()
