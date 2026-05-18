"""
Synaptic Pattern Formation in a Ring of Theta-Neuron E/I Ensembles
===================================================================
A ring of M ensembles, each containing N_E excitatory + N_I inhibitory
theta neurons (default 80%/20%).  Within an ensemble, the four
sub-populations (E→E, E→I, I→E, I→I) couple via STATIC local synaptic
strengths.  Between ensembles, only the excitatory neurons project
out, and these projections are ADAPTIVE — there are two M×M coupling
matrices,

    A^E_{mn}   E→E across ensembles (post=E in ensemble m, pre=E in n)
    A^I_{mn}   E→I across ensembles (post=I in ensemble m, pre=E in n)

both evolving under one of the pulse-based plasticity rules from
`theta_ensemble_fitting.py` (hebbian / antihebbian / oja).

The excitatory excitability statistics carry a LINEAR gradient along
the ring:

    η̄^E_m = η̄^E_lo + (η̄^E_hi − η̄^E_lo) · m/(M-1)
    Δ^E_m = Δ^E_lo  + (Δ^E_hi  − Δ^E_lo ) · m/(M-1)

so one side of the ring is mean-suprathreshold and broad while the
other is sub-threshold and sharp.  The inhibitory population is
identical across ensembles (no I gradient was requested).

The adaptive matrices are initialised with a Gaussian distance kernel
on the ring:

    A^X_{mn}(0) = exp(-d(m,n)² / (2σ²))   (X∈{E,I}, m≠n)

where d(m,n) = min(|m−n|, M−|m−n|) is the periodic ring distance.
Diagonals are zero — within-ensemble coupling is purely local.

Workflow
--------
    1.  Build the ring with E/I substructure and gradient.
    2.  Simulate the full theta-neuron network (microscopic).
    3.  Simulate the matched 2M-population OA mean-field.
    4.  Plot 3 rows × 1 figure:
          row 0  firing-rate traces per ensemble (E-population), TN
                  dashed vs OA solid, with their averages in black.
          row 1  final A^E matrices: TN (left) and OA (right), shared
                  colourbar.
          row 2  final A^I matrices: TN (left) and OA (right), shared
                  colourbar.

Usage
-----
    python theta_ring_ei_adaptation.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

# Reuse the PRX style + colours from kuramoto_ensemble_fitting so this
# figure matches the rest of the manuscript.
from kuramoto.kuramoto_ensemble_fitting import (
    set_prx_style, make_panel_label, C_KM, C_OA,
)

_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Pulse machinery: s(n, θ) = c_n (1 − cosθ)^n  and its OA expectation
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_norm(n):
    """c_n such that ∫_{-π}^{π} c_n (1-cosθ)^n dθ / (2π) = 1.

    c_n = (2n)! / (2^n (n!)^2)⁻¹  →  the reciprocal of the binomial
    coefficient C(2n,n)/2^{2n}.  We compute it via log-factorials to
    stay numerically clean for moderate n.
    """
    from math import lgamma, exp
    # 2π * c_n * ∫(1-cosθ)^n dθ / (2π) = 2π
    # ∫_{-π}^{π} (1-cosθ)^n dθ = 2π · C(2n, n) / 2^n
    # so c_n = 2^n / C(2n,n) = 2^n · n!·n! / (2n)!
    log_cn = n * np.log(2.0) + 2 * lgamma(n + 1) - lgamma(2 * n + 1)
    return float(exp(log_cn))


def fourier_coeffs_s(n):
    """Fourier coefficients ŝ_k of (1-cosθ)^n for k = -n..n.

    (1 − cosθ)^n = Σ_{k=-n}^{n} ŝ_k e^{ikθ}.  Computed by expanding
    (1 − ½e^{iθ} − ½e^{-iθ})^n via the multinomial theorem; here we
    just use FFT for simplicity and robustness.
    """
    M_grid = 4 * n + 8
    theta = 2 * np.pi * np.arange(M_grid) / M_grid
    f = (1.0 - np.cos(theta)) ** n
    F = np.fft.fft(f) / M_grid
    s_hat = np.zeros(2 * n + 1, dtype=complex)
    # k = -n .. n maps to FFT bins via k mod M_grid
    for idx, k in enumerate(range(-n, n + 1)):
        s_hat[idx] = F[k % M_grid]
    return s_hat


def oa_synaptic_mean(n, R, Psi, s_hat, cn):
    """⟨s(n,θ)⟩ under a Lorentzian distribution with order param Re^{iΨ}.

    For each component k, ⟨e^{ikθ}⟩ = (R e^{iΨ})^k for k≥0 and the
    conjugate for k<0.  Returns a real array shaped like R.
    """
    R = np.asarray(R)
    Psi = np.asarray(Psi)
    z = R * np.exp(1j * Psi)
    n = (len(s_hat) - 1) // 2
    out = np.zeros_like(R, dtype=complex)
    for idx, k in enumerate(range(-n, n + 1)):
        if k >= 0:
            out = out + s_hat[idx] * z ** k
        else:
            out = out + s_hat[idx] * np.conj(z) ** (-k)
    return cn * out.real


def oa_pulse_squared_mean(n, R, Psi, s_hat_sq, cn):
    """⟨s(n,θ)²⟩ = c_n² · ⟨(1-cosθ)^{2n}⟩.  s_hat_sq must be the 2n-Fourier
    coefficients."""
    # Use the same expansion machinery with k=-2n..2n
    R = np.asarray(R)
    Psi = np.asarray(Psi)
    z = R * np.exp(1j * Psi)
    two_n = (len(s_hat_sq) - 1) // 2
    out = np.zeros_like(R, dtype=complex)
    for idx, k in enumerate(range(-two_n, two_n + 1)):
        if k >= 0:
            out = out + s_hat_sq[idx] * z ** k
        else:
            out = out + s_hat_sq[idx] * np.conj(z) ** (-k)
    return (cn ** 2) * out.real


def s_micro(n, theta, cn):
    """Microscopic pulse value c_n (1−cosθ)^n."""
    return cn * (1.0 - np.cos(theta)) ** n


PLASTICITY_RULES = ("hebbian", "antihebbian", "oja")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Ring construction: excitabilities + initial adaptive couplings
# ═══════════════════════════════════════════════════════════════════════════════

def ring_distance(M):
    """M×M matrix of periodic ring distances d(m,n) = min(|m−n|, M−|m−n|)."""
    idx = np.arange(M)
    d = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(d, M - d).astype(float)


def gaussian_ring_kernel(M, sigma):
    """K_{mn} = exp(-d²/2σ²) with K_{mm}=0 (no self-coupling between ensembles)."""
    d = ring_distance(M)
    K = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(K, 0.0)
    return K


def sample_cauchy_excitabilities(N, eta_bar, delta, rng):
    """Draw N samples from Cauchy(eta_bar, delta).  Uses inverse-CDF for
    reproducibility regardless of numpy version."""
    u = rng.uniform(_EPS, 1.0 - _EPS, N)
    return eta_bar + delta * np.tan(np.pi * (u - 0.5))


def build_ring(M, N_E, N_I, eta_bar_E_range, delta_E_range,
               eta_bar_I, delta_I, seed):
    """Build per-ensemble excitabilities for E and I populations.

    Returns a dict with:
        eta_E   shape (M, N_E)   per-ensemble E excitabilities
        eta_I   shape (M, N_I)   per-ensemble I excitabilities
        eta_bar_E, delta_E  shape (M,)  the OA parameters
        eta_bar_I, delta_I  scalars
    """
    rng = np.random.default_rng(seed)

    # Linear gradients along the ring for the E population.
    if M == 1:
        eta_bar_E = np.array([0.5 * sum(eta_bar_E_range)])
        delta_E   = np.array([0.5 * sum(delta_E_range)])
    else:
        frac = np.linspace(0.0, 1.0, M)
        eta_bar_E = eta_bar_E_range[0] + frac * (eta_bar_E_range[1]
                                                 - eta_bar_E_range[0])
        delta_E   = delta_E_range[0]  + frac * (delta_E_range[1]
                                                 - delta_E_range[0])

    eta_E = np.empty((M, N_E))
    eta_I = np.empty((M, N_I))
    for m in range(M):
        eta_E[m] = sample_cauchy_excitabilities(N_E, eta_bar_E[m],
                                                delta_E[m], rng)
        # Inhibitory population: same statistics on every ensemble
        eta_I[m] = sample_cauchy_excitabilities(N_I, eta_bar_I,
                                                delta_I, rng)

    return dict(
        eta_E=eta_E, eta_I=eta_I,
        eta_bar_E=eta_bar_E, delta_E=delta_E,
        eta_bar_I=float(eta_bar_I), delta_I=float(delta_I),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Microscopic theta-neuron network with E/I substructure on a ring
# ═══════════════════════════════════════════════════════════════════════════════
#
# State layout (one flat vector of length 2*M*(N_E+N_I) + 2*M*M ... wait, no:
# we have NE+NI neurons per ensemble.  Total neurons N_tot = M*(N_E+N_I).
# State vector:
#   y[0           : M*N_E         ]  θ for E neurons,  block layout (m, i)
#   y[M*N_E       : M*(N_E+N_I)   ]  θ for I neurons,  block layout (m, j)
#   y[N_tot       : N_tot + M*M   ]  A^E flattened  (post=row=m, pre=col=n)
#   y[N_tot+M*M   : N_tot + 2*M*M ]  A^I flattened
# ─────────────────────────────────────────────────────────────────────────────

def tn_rhs(t, y, ring,
           N_E, N_I, M, J_loc, J_glob_E, J_glob_I,
           n_pulse, n1, n2, n3,
           cn, cn1, cn2, cn3,
           mu, gamma, plasticity,
           tau_E=1.0, tau_I=1.0):
    """RHS for the full E/I theta network on a ring with adaptive
    between-ensemble couplings A^E, A^I.  Static within-ensemble J_loc.

    Membrane timescales τ_E, τ_I rescale the θ dynamics only:
        τ_X · θ̇_X = (1 − cosθ_X) + (1 + cosθ_X) η_eff_X
    The plasticity dynamics of A^E, A^I are not affected by τ."""
    Ntot_E = M * N_E
    Ntot_I = M * N_I
    Ntot   = Ntot_E + Ntot_I
    MM     = M * M

    theta_E = y[0 : Ntot_E].reshape(M, N_E)
    theta_I = y[Ntot_E : Ntot_E + Ntot_I].reshape(M, N_I)
    A_E     = y[Ntot : Ntot + MM].reshape(M, M)
    A_I     = y[Ntot + MM : Ntot + 2 * MM].reshape(M, M)

    # ── Per-ensemble per-population pulse means s^X_m(t) ──────────────────────
    pulse_E_full = cn * (1.0 - np.cos(theta_E)) ** n_pulse   # (M, N_E)
    pulse_I_full = cn * (1.0 - np.cos(theta_I)) ** n_pulse   # (M, N_I)
    s_E = pulse_E_full.mean(axis=1)                          # (M,)
    s_I = pulse_I_full.mean(axis=1)                          # (M,)

    # ── Synaptic drive to each population in each ensemble ────────────────────
    # Local drive (static):   J_loc[X,Y] · s^Y_m
    # Global drive (adaptive between ensembles): J_glob_X · Σ_{n≠m} A^X_{mn} s^E_n
    # The diagonals A^X_{mm} are kept at 0 (no self-coupling between ensembles).
    drive_E_loc = J_loc[0, 0] * s_E + J_loc[0, 1] * s_I       # (M,)
    drive_I_loc = J_loc[1, 0] * s_E + J_loc[1, 1] * s_I       # (M,)

    drive_E_glob = J_glob_E * (A_E @ s_E)                     # (M,)
    drive_I_glob = J_glob_I * (A_I @ s_E)                     # (M,)

    # Effective synaptic input per ensemble — broadcast to neuron level.
    syn_E = (drive_E_loc + drive_E_glob)[:, None]             # (M, 1)
    syn_I = (drive_I_loc + drive_I_glob)[:, None]             # (M, 1)

    # ── Theta-neuron ODE: τ·θ̇ = (1−cosθ) + (1+cosθ)·η_eff ────────────────────
    cos_E = np.cos(theta_E); cos_I = np.cos(theta_I)
    eta_eff_E = ring["eta_E"] + syn_E                          # (M, N_E)
    eta_eff_I = ring["eta_I"] + syn_I                          # (M, N_I)
    dtheta_E = ((1.0 - cos_E) + (1.0 + cos_E) * eta_eff_E) / tau_E
    dtheta_I = ((1.0 - cos_I) + (1.0 + cos_I) * eta_eff_I) / tau_I

    # ── Adaptive couplings A^E, A^I — between ensembles only ──────────────────
    # Pulse "channels" for plasticity (different n than the synaptic n_pulse):
    #   pre channel: n3 (excitatory pre-syn only — both A^E and A^I are E-pre)
    #   post channel A^E: n2 on s^E    →   "post = E in ensemble m"
    #   post channel A^I: n2 on s^I    →   "post = I in ensemble m"
    #   antihebbian S1 term: n1 on s^E (pre)
    #   oja Ssq term: n1² on s^E (pre)
    s_E_n2 = (cn2 * (1.0 - np.cos(theta_E)) ** n2).mean(axis=1)   # (M,)
    s_I_n2 = (cn2 * (1.0 - np.cos(theta_I)) ** n2).mean(axis=1)   # (M,)
    s_E_n3 = (cn3 * (1.0 - np.cos(theta_E)) ** n3).mean(axis=1)   # (M,)

    hebb_E = np.outer(s_E_n2, s_E_n3)        # (M,M)   row=post(E), col=pre(E)
    hebb_I = np.outer(s_I_n2, s_E_n3)        # (M,M)   row=post(I), col=pre(E)

    if plasticity == "hebbian":
        dA_E = mu * hebb_E - gamma * A_E
        dA_I = mu * hebb_I - gamma * A_I
    elif plasticity == "antihebbian":
        s_E_n1 = (cn1 * (1.0 - np.cos(theta_E)) ** n1).mean(axis=1)
        # Antihebbian: μ·(S1_pre − A·hebb) − γ·A.
        dA_E = mu * (s_E_n1[None, :] - A_E * hebb_E) - gamma * A_E
        dA_I = mu * (s_E_n1[None, :] - A_I * hebb_I) - gamma * A_I
    elif plasticity == "oja":
        # Oja: μ·(hebb − A · S1_post²) − γ·A,  with post-population pulse².
        s_E_n1_sq_per_neuron = (cn1 * (1.0 - np.cos(theta_E)) ** n1) ** 2
        s_I_n1_sq_per_neuron = (cn1 * (1.0 - np.cos(theta_I)) ** n1) ** 2
        Ssq_E = s_E_n1_sq_per_neuron.mean(axis=1)              # (M,)
        Ssq_I = s_I_n1_sq_per_neuron.mean(axis=1)              # (M,)
        dA_E = mu * (hebb_E - A_E * Ssq_E[:, None]) - gamma * A_E
        dA_I = mu * (hebb_I - A_I * Ssq_I[:, None]) - gamma * A_I
    else:
        raise ValueError(f"Unknown plasticity rule {plasticity!r}")

    # Zero out the diagonals (between-ensemble couplings only)
    np.fill_diagonal(dA_E, 0.0)
    np.fill_diagonal(dA_I, 0.0)

    return np.concatenate([
        dtheta_E.ravel(),
        dtheta_I.ravel(),
        dA_E.ravel(),
        dA_I.ravel(),
    ])


def tn_firing_rate_per_ensemble(theta_E_block, n_pulse, cn):
    """Per-ensemble E-population mean pulse rate s^E_m(t).

    theta_E_block has shape (M, N_E, T).  Returns shape (M, T).
    """
    return cn * np.mean((1.0 - np.cos(theta_E_block)) ** n_pulse, axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  OA mean-field for the ring of E/I ensembles
# ═══════════════════════════════════════════════════════════════════════════════
#
# Per-ensemble OA state: (R^E_m, Ψ^E_m, R^I_m, Ψ^I_m) for m=1..M.
# Plus the two M×M adaptive matrices A^E, A^I.
# Flat layout:
#     y[0   : M  ]   R^E
#     y[M   : 2M ]   Ψ^E
#     y[2M  : 3M ]   R^I
#     y[3M  : 4M ]   Ψ^I
#     y[4M  : 4M+M*M ]   A^E (post×pre)
#     y[4M+M*M : 4M+2*M*M]   A^I (post×pre)
# ─────────────────────────────────────────────────────────────────────────────

def _theta_oa_pop(R, Psi, delta, E, tau=1.0):
    """Standard theta-neuron OA equations for one population, with an
    optional membrane timescale τ.  Setting τ≠1 rescales the entire RHS
    by 1/τ, mirroring the microscopic equation  τ·θ̇ = (1−cosθ) + (1+cosθ)η.

        τ·Ṙ   = -ΔR - Δ(1+R²)/2·cosΨ + (E−1)(1−R²)/2·sinΨ
        τ·Ψ̇   = (E+1) + (E−1)(1+R²)/(2R)·cosΨ + Δ(1−R²)/(2R)·sinΨ
    """
    R   = np.clip(R, _EPS, 1.0 - _EPS)
    cP  = np.cos(Psi); sP = np.sin(Psi)
    dR  = (-delta * R
           - delta * (1.0 + R ** 2) / 2.0 * cP
           + (E - 1.0) * (1.0 - R ** 2) / 2.0 * sP)
    dPsi = ((E + 1.0)
            + (E - 1.0) * (1.0 + R ** 2) / (2.0 * R) * cP
            + delta * (1.0 - R ** 2) / (2.0 * R) * sP)
    return dR / tau, dPsi / tau


def oa_rhs(t, y, ring,
           M, J_loc, J_glob_E, J_glob_I,
           n_pulse, n1, n2, n3,
           cn, cn1, cn2, cn3,
           s_hat, s_hat1, s_hat2, s_hat3, s_hat1_sq,
           mu, gamma, plasticity,
           tau_E=1.0, tau_I=1.0):
    R_E = y[0 * M : 1 * M]
    P_E = y[1 * M : 2 * M]
    R_I = y[2 * M : 3 * M]
    P_I = y[3 * M : 4 * M]
    A_E = y[4 * M : 4 * M + M * M].reshape(M, M)
    A_I = y[4 * M + M * M : 4 * M + 2 * M * M].reshape(M, M)

    # OA pulse means S^X_m(R^X_m, Ψ^X_m)
    S_E = oa_synaptic_mean(n_pulse, R_E, P_E, s_hat, cn)        # (M,)
    S_I = oa_synaptic_mean(n_pulse, R_I, P_I, s_hat, cn)        # (M,)

    # Per-ensemble excitatory drives (entering both E and I sub-populations)
    drive_E_loc = J_loc[0, 0] * S_E + J_loc[0, 1] * S_I          # (M,)
    drive_I_loc = J_loc[1, 0] * S_E + J_loc[1, 1] * S_I          # (M,)
    drive_E_glob = J_glob_E * (A_E @ S_E)                        # (M,)
    drive_I_glob = J_glob_I * (A_I @ S_E)                        # (M,)

    Eeff_E = ring["eta_bar_E"] + drive_E_loc + drive_E_glob      # (M,)
    Eeff_I = ring["eta_bar_I"] + drive_I_loc + drive_I_glob      # (M,)

    dR_E, dP_E = _theta_oa_pop(R_E, P_E, ring["delta_E"], Eeff_E, tau=tau_E)
    dR_I, dP_I = _theta_oa_pop(R_I, P_I, ring["delta_I"], Eeff_I, tau=tau_I)

    # ── Plasticity kernel expectations ────────────────────────────────────────
    S_E_n2 = oa_synaptic_mean(n2, R_E, P_E, s_hat2, cn2)         # post=E channel
    S_I_n2 = oa_synaptic_mean(n2, R_I, P_I, s_hat2, cn2)         # post=I channel
    S_E_n3 = oa_synaptic_mean(n3, R_E, P_E, s_hat3, cn3)         # pre =E channel
    hebb_E = np.outer(S_E_n2, S_E_n3)
    hebb_I = np.outer(S_I_n2, S_E_n3)

    if plasticity == "hebbian":
        dA_E = mu * hebb_E - gamma * A_E
        dA_I = mu * hebb_I - gamma * A_I
    elif plasticity == "antihebbian":
        S_E_n1 = oa_synaptic_mean(n1, R_E, P_E, s_hat1, cn1)
        dA_E = mu * (S_E_n1[None, :] - A_E * hebb_E) - gamma * A_E
        dA_I = mu * (S_E_n1[None, :] - A_I * hebb_I) - gamma * A_I
    elif plasticity == "oja":
        Ssq_E = oa_pulse_squared_mean(n1, R_E, P_E, s_hat1_sq, cn1)
        Ssq_I = oa_pulse_squared_mean(n1, R_I, P_I, s_hat1_sq, cn1)
        dA_E = mu * (hebb_E - A_E * Ssq_E[:, None]) - gamma * A_E
        dA_I = mu * (hebb_I - A_I * Ssq_I[:, None]) - gamma * A_I
    else:
        raise ValueError(f"Unknown plasticity rule {plasticity!r}")

    np.fill_diagonal(dA_E, 0.0)
    np.fill_diagonal(dA_I, 0.0)

    return np.concatenate([dR_E, dP_E, dR_I, dP_I,
                           dA_E.ravel(), dA_I.ravel()])


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Top-level simulation: build, run TN, run OA, return everything
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(*,
             M=10, N_E=80, N_I=20, T=120.0,
             # Local (within-ensemble) static synaptic strengths J_loc[X,Y]:
             #   X = post (row, 0=E, 1=I), Y = pre (col, 0=E, 1=I)
             J_EE_loc=1.5, J_EI_loc=-1.0, J_IE_loc=1.0, J_II_loc=-0.5,
             # Global (between-ensemble) scales — both project from E
             J_glob_E=1.5, J_glob_I=1.5,
             # Membrane timescales (act only on θ-dynamics in both
             # microscopic and OA equations; plasticity is unaffected).
             tau_E=1.0, tau_I=1.0,
             # Plasticity
             plasticity="hebbian", mu=0.02, gamma=0.001,
             n_pulse=10, n1=2, n2=2, n3=3,
             # Excitability gradient on the ring (E side: low→high m)
             eta_bar_E_range=(-0.3, 0.8),
             delta_E_range=(0.10, 0.30),
             # Inhibitory population (uniform across ring)
             eta_bar_I=0.2, delta_I=0.15,
             # Initial adaptive coupling: Gaussian distance kernel
             A0_sigma=2.0, A0_scale=1.0,
             seed=42, method="RK45", rtol=1e-6, atol=1e-8,
             ):
    print(f"Building ring (M={M}, N_E={N_E}, N_I={N_I}) …")
    ring = build_ring(M, N_E, N_I,
                      eta_bar_E_range, delta_E_range,
                      eta_bar_I, delta_I, seed=seed)

    J_loc = np.array([[J_EE_loc, J_EI_loc],
                      [J_IE_loc, J_II_loc]], float)

    # Precompute pulse normalisations and Fourier coefficients
    cn  = coupling_norm(n_pulse); s_hat  = fourier_coeffs_s(n_pulse)
    cn1 = coupling_norm(n1);      s_hat1 = fourier_coeffs_s(n1)
    cn2 = coupling_norm(n2);      s_hat2 = fourier_coeffs_s(n2)
    cn3 = coupling_norm(n3);      s_hat3 = fourier_coeffs_s(n3)
    s_hat1_sq = fourier_coeffs_s(2 * n1)

    # Initial conditions — matched between TN and OA
    rng = np.random.default_rng(seed + 1)
    theta0_E = rng.uniform(-np.pi, np.pi, (M, N_E))
    theta0_I = rng.uniform(-np.pi, np.pi, (M, N_I))

    # OA: derive R, Ψ from the empirical complex order parameter
    z0_E = np.mean(np.exp(1j * theta0_E), axis=1)
    z0_I = np.mean(np.exp(1j * theta0_I), axis=1)
    R0_E = np.clip(np.abs(z0_E), _EPS, 1.0 - _EPS)
    R0_I = np.clip(np.abs(z0_I), _EPS, 1.0 - _EPS)
    P0_E = np.angle(z0_E)
    P0_I = np.angle(z0_I)

    # Initial adaptive matrices: Gaussian distance kernel, zero diagonal
    A0_kernel = A0_scale * gaussian_ring_kernel(M, A0_sigma)
    A0_E = A0_kernel.copy()
    A0_I = A0_kernel.copy()

    # ── Microscopic run ───────────────────────────────────────────────────────
    print(f"Running TN (microscopic, {plasticity}) …")
    y0_tn = np.concatenate([
        theta0_E.ravel(), theta0_I.ravel(),
        A0_E.ravel(), A0_I.ravel(),
    ])
    sol_tn = solve_ivp(
        tn_rhs, (0, T), y0_tn, method=method,
        args=(ring, N_E, N_I, M, J_loc, J_glob_E, J_glob_I,
              n_pulse, n1, n2, n3, cn, cn1, cn2, cn3,
              mu, gamma, plasticity,
              tau_E, tau_I),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_tn.success:
        raise RuntimeError(f"TN failed: {sol_tn.message}")
    print(f"  done — {sol_tn.t.size} steps, {sol_tn.nfev} evals")

    Ntot_E = M * N_E; Ntot_I = M * N_I; Ntot = Ntot_E + Ntot_I; MM = M * M
    theta_E_t = sol_tn.y[0:Ntot_E].reshape(M, N_E, -1)
    theta_I_t = sol_tn.y[Ntot_E:Ntot].reshape(M, N_I, -1)
    A_E_tn_final = sol_tn.y[Ntot:Ntot + MM, -1].reshape(M, M)
    A_I_tn_final = sol_tn.y[Ntot + MM:Ntot + 2 * MM, -1].reshape(M, M)

    # Per-ensemble firing rate of E populations (the canonical "ring trace")
    s_E_tn = tn_firing_rate_per_ensemble(theta_E_t, n_pulse, cn)   # (M, T_tn)
    s_I_tn = tn_firing_rate_per_ensemble(theta_I_t, n_pulse, cn)  # (M, T_tn)

    # ── OA run ────────────────────────────────────────────────────────────────
    print(f"Running OA (mean-field, {plasticity}) …")
    y0_oa = np.concatenate([R0_E, P0_E, R0_I, P0_I,
                            A0_E.ravel(), A0_I.ravel()])
    sol_oa = solve_ivp(
        oa_rhs, (0, T), y0_oa, method=method,
        args=(ring, M, J_loc, J_glob_E, J_glob_I,
              n_pulse, n1, n2, n3, cn, cn1, cn2, cn3,
              s_hat, s_hat1, s_hat2, s_hat3, s_hat1_sq,
              mu, gamma, plasticity,
              tau_E, tau_I),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_oa.success:
        raise RuntimeError(f"OA failed: {sol_oa.message}")
    print(f"  done — {sol_oa.t.size} steps, {sol_oa.nfev} evals")

    R_E_oa = sol_oa.y[0 * M : 1 * M]
    P_E_oa = sol_oa.y[1 * M : 2 * M]
    R_I_oa = sol_oa.y[2 * M: 3 * M]
    P_I_oa = sol_oa.y[3 * M: 4 * M]
    A_E_oa_final = sol_oa.y[4 * M : 4 * M + MM, -1].reshape(M, M)
    A_I_oa_final = sol_oa.y[4 * M + MM : 4 * M + 2 * MM, -1].reshape(M, M)

    # Per-ensemble OA E-firing rate ⟨s⟩_m
    s_E_oa, s_I_oa = np.empty_like(R_E_oa), np.empty_like(R_E_oa)
    for k in range(R_E_oa.shape[1]):
        s_E_oa[:, k] = oa_synaptic_mean(n_pulse, R_E_oa[:, k], P_E_oa[:, k],
                                        s_hat, cn)
        s_I_oa[:, k] = oa_synaptic_mean(n_pulse, R_I_oa[:, k], P_I_oa[:, k],
                                        s_hat, cn)

    return dict(
        # config
        M=M, N_E=N_E, N_I=N_I, T=T, plasticity=plasticity,
        ring=ring,
        # trajectories
        t_tn=sol_tn.t, s_E_tn=s_E_tn.mean(axis=0), s_I_tn=s_I_tn.mean(axis=0),
        t_oa=sol_oa.t, s_E_oa=s_E_oa.mean(axis=0), s_I_oa=s_I_oa.mean(axis=0),
        # final matrices
        A_E_tn_final=A_E_tn_final, A_I_tn_final=A_I_tn_final,
        A_E_oa_final=A_E_oa_final, A_I_oa_final=A_I_oa_final,
        # initial matrix for reference
        A0_kernel=A0_kernel,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Figure: firing-rate traces + final A^E + A^I matrices, TN vs OA
# ═══════════════════════════════════════════════════════════════════════════════

def plot_figure(res, savepath="theta_ring_ei.pdf"):
    set_prx_style()
    M = res["M"]
    T = res["T"]

    # Row 0: one wide firing-rate panel.  Rows 1 and 2: each has TN+OA+cbar.
    # We use a 3×3 gridspec where row 0 spans all three columns.
    fig = plt.figure(figsize=(7.0, 7.4))
    gs = gridspec.GridSpec(
        nrows=3, ncols=3, figure=fig,
        height_ratios=[1.0, 1.0, 1.0],
        width_ratios =[1.0, 1.0, 0.06],   # last column is a colorbar slot
        hspace=0.55, wspace=0.20,
        left=0.095, right=0.93, top=0.955, bottom=0.075,
    )

    letters = iter("abcdefghij")

    # ── Row 0: firing-rate traces ──────────────────────────────────────────
    ax_R = fig.add_subplot(gs[0, :])

    # Per-ensemble traces: TN dashed, OA solid, coloured by ring position
    # cmap_lines = plt.get_cmap("viridis")

    # Population-averaged traces in black, on top
    ax_R.plot(res["t_oa"], res["s_I_oa"],
              color="darkorange", lw=1.8, ls="-", label="OA ⟨s^I⟩")
    ax_R.plot(res["t_tn"], res["s_I_tn"],
              color="darkorange", lw=1.8, ls="--", label="TN ⟨s^I⟩")
    ax_R.plot(res["t_oa"], res["s_E_oa"],
              color="royalblue", lw=1.8, ls="-",  label="OA ⟨s^E⟩")
    ax_R.plot(res["t_tn"], res["s_E_tn"],
              color="royalblue", lw=1.8, ls="--", label="TN ⟨s^E⟩")

    ax_R.set_xlim(0, T)
    ax_R.set_xlabel(r"time $t$")
    ax_R.set_ylabel(r"firing rate $\langle s_m(t)\rangle$")
    ax_R.legend(loc="lower right", frameon=True, framealpha=0.9,
                handlelength=2.4, borderaxespad=0.4, fontsize=9,
                edgecolor="none")
    make_panel_label(ax_R, f"({next(letters)})", x=-0.05, y=1.04)

    # ── Rows 1 & 2: final adaptive matrices, TN vs OA ──────────────────────
    matrix_rows = [
        ("row1", 1, res["A_E_tn_final"], res["A_E_oa_final"], r"$A^{E}$"),
        ("row2", 2, res["A_I_tn_final"], res["A_I_oa_final"], r"$A^{I}$"),
    ]

    tick_step = max(1, M // 6)
    ticks = np.arange(0, M, tick_step)

    for _, row_idx, A_tn, A_oa, ylabel in matrix_rows:
        # Shared colour scale within a row.  Pulse rates are non-negative,
        # so a sequential map is appropriate.
        vmax = max(A_tn.max(), A_oa.max()) or 1.0
        vmin = 0.0
        kw = dict(cmap="viridis", vmin=vmin, vmax=vmax,
                  interpolation="nearest", aspect="equal")

        ax_tn = fig.add_subplot(gs[row_idx, 0])
        ax_oa = fig.add_subplot(gs[row_idx, 1])
        ax_cb = fig.add_subplot(gs[row_idx, 2])

        ax_tn.imshow(A_tn, **kw)
        im_oa = ax_oa.imshow(A_oa, **kw)

        ax_tn.set_title("theta network", fontsize=10)
        ax_oa.set_title("mean field", fontsize=10)

        for ax in (ax_tn, ax_oa):
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels(ticks); ax.set_yticklabels(ticks)
            ax.tick_params(axis="both", labelsize=9)
            ax.set_xlabel(r"pre  ensemble $n$")
        ax_tn.set_ylabel(ylabel + r"   (post ensemble $m$)", fontsize=10)
        ax_oa.set_yticklabels([])
        ax_oa.set_ylabel("")

        cb = fig.colorbar(im_oa, cax=ax_cb)
        cb.set_label(ylabel + r"$_{mn}$", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        cb.locator = plt.MaxNLocator(nbins=4)
        cb.update_ticks()

        make_panel_label(ax_tn, f"({next(letters)})", x=-0.10, y=1.04)

    fig.savefig(savepath, bbox_inches="tight")
    print(f"\nFigure saved → {savepath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CONFIG = dict(
        M=10, N_E=400, N_I=100, T=150.0,
        # Local (within-ensemble) synaptic strengths
        J_EE_loc= 1.5, J_EI_loc=-1.2,
        J_IE_loc= 1.0, J_II_loc=-0.4,
        # Global (between-ensemble) scales — kept weak so the excitability
        # gradient and ring-distance bias remain visible in the final A.
        J_glob_E=0.20, J_glob_I=0.20,
        # Membrane timescales (τ acts on the θ equation only;
        # τ_E = τ_I = 1.0 recovers the original behaviour).
        tau_E=1.0, tau_I=2.0,
        # Plasticity: choose from {"hebbian", "antihebbian", "oja"}
        plasticity="hebbian",
        # μ and γ chosen so the steady-state Hebbian weight μ·⟨s⟩²/γ is
        # O(1) and the gradient + ring-distance bias remain visible
        # in the final matrices.
        mu=0.01, gamma=0.0,
        n_pulse=10, n1=3, n2=3, n3=3,
        # Linear gradients along the ring (E side)
        eta_bar_E_range=(-0.5, 1.0),
        delta_E_range=(0.10, 0.30),
        # Inhibitory population (uniform across ring)
        eta_bar_I=0.2, delta_I=0.15,
        # Initial connectivity: Gaussian on ring distance
        A0_sigma=2.0, A0_scale=1.0,
        seed=42, method="RK45", rtol=1e-6, atol=1e-8,
    )

    res = simulate(**CONFIG)
    plot_figure(res, savepath="theta_ring_ei.pdf")
    plt.show()
