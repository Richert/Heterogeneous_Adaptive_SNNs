"""
Bump Formation in a Homogeneous Ring of Theta-Neuron E/I Ensembles
===================================================================
Companion script to `theta_ring_comparison.py`, repurposed to study
the formation, stabilisation and drift of activity bumps on a ring of
M ensembles, each containing N_E excitatory + N_I inhibitory theta
neurons.  Differences from the comparison script:

  • Excitability statistics are HOMOGENEOUS along the ring.  η̄^E and
    Δ^E are scalars (no gradient), so every ensemble has the same
    excitability distribution and the ring is translation-invariant
    apart from the initial connectivity kernel.

  • Two SEPARATE Gaussian kernel widths σ_EE and σ_EI control the
    initial E→E and E→I connectivity respectively:

        A^E_{mn}(0) = scale_EE · exp(-d(m,n)² / (2σ_EE²))
        A^I_{mn}(0) = scale_EI · exp(-d(m,n)² / (2σ_EI²))

    (diagonals are zero, d(m,n) = min(|m−n|, M−|m−n|)).  The classic
    bump-attractor regime needs σ_EI > σ_EE so inhibition surrounds
    excitation — varying these widths is the main study knob.

  • The simulation is seeded with a LOCALISED FIRING-RATE BUMP centred
    at ensemble m₀ (default M/2): the E-population there is initialised
    at a peak firing rate r_bump while the rest of the ring sits at a
    baseline rate r_base.  The Montbrió–Pazó–Roxin conformal map
    W = (1−Z)/(1+Z), W = π·r + i·v, converts the (r, v) profile to the
    matching Kuramoto order parameter Z = R·exp(iΨ), which is then used
    as the OA initial condition and as the target distribution for the
    wrapped-Cauchy θ samples on the microscopic side.  Specifying the
    IC in firing-rate units is more physically meaningful than the
    coherence R, especially since R alone is ambiguous (the same R can
    correspond to a quiescent or a highly-active state, depending on Ψ).

  • The default diagnostic figure has 2 rows:
        row 0   kymographs of E and I firing rate, m × t
        row 1   final activity profile (E + I), final A^E, final A^I

  • The TN simulation is optional (run_tn=False by default) since
    OA already provides clean bump diagnostics; flip the flag on to
    cross-check.

Usage
-----
    python theta_ring_bump.py
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


def sample_wrapped_cauchy(N, R, Psi, rng):
    """Draw N phases from the wrapped-Cauchy distribution on [-π, π].

    The wrapped-Cauchy density with mean direction Ψ and concentration
    R∈[0,1) has Fourier coefficients ⟨e^{ikθ}⟩ = R^k e^{ikΨ}, so its
    empirical complex order parameter is consistent with the OA
    ansatz z = R e^{iΨ}.  The standard inverse-CDF sampler is

        θ = Ψ + 2·arctan( (1-R)/(1+R) · tan(π(u - 1/2)) )

    for u ~ U(0,1).  At R=0 this reduces to a uniform distribution.
    """
    R = float(np.clip(R, 0.0, 1.0 - _EPS))
    u = rng.uniform(_EPS, 1.0 - _EPS, N)
    if R < _EPS:
        return rng.uniform(-np.pi, np.pi, N)
    return Psi + 2.0 * np.arctan(((1.0 - R) / (1.0 + R))
                                 * np.tan(np.pi * (u - 0.5)))


def montbrio_rate_to_kuramoto(r, v=0.0):
    """Montbrió–Pazó–Roxin conformal map: (r, v) → (R, Ψ).

    For QIF / theta-neuron networks under the Lorentzian/OA ansatz, the
    QIF complex variable W = π·r + i·v (with r the population firing
    rate and v the mean membrane potential) is related to the Kuramoto
    order parameter Z = R·exp(iΨ) by the Cayley transform

        W = (1 − Z) / (1 + Z),     Z = (1 − W) / (1 + W).

    (See Montbrió, Pazó & Roxin, *PRX* **5**, 021028 (2015),
    appendix and Eq. (6).)  This routine inverts that mapping.

    For the special case v = 0 the answer is real:
        r < 1/π:   Z > 0  ⇒  R = (1 − π r)/(1 + π r),  Ψ = 0
        r > 1/π:   Z < 0  ⇒  R = (π r − 1)/(π r + 1),  Ψ = π
    so high firing rate corresponds to high coherence with Ψ → π
    (population clustered near the θ = π spike).

    Accepts scalars or numpy arrays of equal shape; returns (R, Ψ).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    W = np.pi * r + 1j * v
    Z = (1.0 - W) / (1.0 + W)
    R = np.clip(np.abs(Z), _EPS, 1.0 - _EPS)
    Psi = np.angle(Z)
    return R, Psi


def build_ring(M, N_E, N_I, eta_bar_E, delta_E,
               eta_bar_I, delta_I, seed):
    """Build per-ensemble excitabilities for E and I populations.

    The ring is HOMOGENEOUS: η̄^E, Δ^E (E side) and η̄^I, Δ^I (I side)
    are scalars shared by every ensemble — the only source of spatial
    structure is the connectivity kernel and the bump initial condition.

    Returns a dict with:
        eta_E   shape (M, N_E)   per-ensemble E excitabilities (sampled)
        eta_I   shape (M, N_I)   per-ensemble I excitabilities (sampled)
        eta_bar_E, delta_E  shape (M,)  the OA parameters (constant)
        eta_bar_I, delta_I  scalars
    """
    rng = np.random.default_rng(seed)

    # Homogeneous excitability statistics on both sides.
    eta_bar_E_arr = np.full(M, float(eta_bar_E))
    delta_E_arr   = np.full(M, float(delta_E))

    eta_E = np.empty((M, N_E))
    eta_I = np.empty((M, N_I))
    for m in range(M):
        eta_E[m] = sample_cauchy_excitabilities(N_E, eta_bar_E_arr[m],
                                                delta_E_arr[m], rng)
        eta_I[m] = sample_cauchy_excitabilities(N_I, eta_bar_I,
                                                delta_I, rng)

    return dict(
        eta_E=eta_E, eta_I=eta_I,
        eta_bar_E=eta_bar_E_arr, delta_E=delta_E_arr,
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
             M=64, N_E=80, N_I=20, T=120.0,
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
             # Homogeneous excitabilities (no gradient on this ring)
             eta_bar_E=0.0,  delta_E=0.10,
             eta_bar_I=-0.2, delta_I=0.15,
             # Initial adaptive coupling: independent Gaussian kernels
             sigma_EE=2.0, sigma_EI=6.0,
             scale_EE=1.0, scale_EI=1.0,
             # Bump initial condition — specified by firing-rate profile
             # (Montbrió–Pazó–Roxin conformal map gives the matching R, Ψ
             # for both the OA state and the wrapped-Cauchy θ samples).
             bump_centre=None,         # int in [0, M); default M//2
             r_bump=20.0,               # peak E firing rate at the bump
             r_base=0.0,              # baseline E firing rate elsewhere
             r_I_base=0.0,            # baseline I firing rate (uniform)
             v_E_init=-2.0,             # mean E membrane potential (usually 0)
             v_I_init=-1.0,             # mean I membrane potential
             bump_width=2.0,           # σ of Gaussian-in-ring-distance for IC
             # Run controls
             run_tn=False,
             seed=42, method="RK45", rtol=1e-6, atol=1e-8,
             ):
    """Single-run bump simulation.

    Initial condition is specified in terms of QIF/Montbrió firing-rate
    statistics: a Gaussian bump of mean rate r_bump on a baseline r_base
    for the excitatory population, and uniform r_I_base for inhibition.
    The conformal map W = (1−Z)/(1+Z) with W = π·r + i·v is used to
    convert (r, v) → (R, Ψ), which then serves as both the OA initial
    condition and the target for the wrapped-Cauchy θ samples on the
    microscopic side.

    Note that high r corresponds to Ψ ≈ π (population clustered near
    the θ = π spike) and high R, while r ≈ 0 corresponds to Ψ ≈ 0 with
    moderately high R (population clustered near θ = 0, the silent
    fixed point).

    Returns a dict with full per-ensemble trajectories for both the OA
    mean-field and (optionally) the matched microscopic theta network.
    Activity arrays are shaped (M, T_eval); each row is one ring site.
    """
    print(f"Building ring (M={M}, N_E={N_E}, N_I={N_I}) …")
    ring = build_ring(M, N_E, N_I,
                      eta_bar_E, delta_E,
                      eta_bar_I, delta_I, seed=seed)

    J_loc = np.array([[J_EE_loc, J_EI_loc],
                      [J_IE_loc, J_II_loc]], float)

    # Precompute pulse normalisations and Fourier coefficients
    cn  = coupling_norm(n_pulse); s_hat  = fourier_coeffs_s(n_pulse)
    cn1 = coupling_norm(n1);      s_hat1 = fourier_coeffs_s(n1)
    cn2 = coupling_norm(n2);      s_hat2 = fourier_coeffs_s(n2)
    cn3 = coupling_norm(n3);      s_hat3 = fourier_coeffs_s(n3)
    s_hat1_sq = fourier_coeffs_s(2 * n1)

    # ── Bump initial condition (matched OA ↔ TN) ────────────────────────────
    if bump_centre is None:
        bump_centre = M // 2
    bump_centre = int(bump_centre) % M

    d_centre = ring_distance(M)[bump_centre]                  # (M,)
    bump_profile = np.exp(-(d_centre ** 2) / (2.0 * bump_width ** 2))

    # Per-ensemble firing-rate IC: Gaussian-in-ring-distance bump on a
    # uniform baseline (excitatory side).  Inhibitory side is uniform.
    r0_E = r_base + (r_bump - r_base) * bump_profile          # (M,)
    r0_I = np.full(M, r_I_base)
    v0_E = np.full(M, v_E_init)
    v0_I = np.full(M, v_I_init)

    # Conformal map → matching (R, Ψ) for both OA and the wrapped-Cauchy θ IC.
    R0_E, P0_E = montbrio_rate_to_kuramoto(r0_E, v0_E)
    R0_I, P0_I = montbrio_rate_to_kuramoto(r0_I, v0_I)

    rng = np.random.default_rng(seed + 1)

    # Microscopic θ ICs: draw from a wrapped-Cauchy that matches each (R, Ψ)
    theta0_E = np.empty((M, N_E))
    theta0_I = np.empty((M, N_I))
    for m in range(M):
        theta0_E[m] = sample_wrapped_cauchy(N_E, R0_E[m], P0_E[m], rng)
        theta0_I[m] = sample_wrapped_cauchy(N_I, R0_I[m], P0_I[m], rng)

    # ── Initial adaptive matrices: two independent Gaussian ring kernels ─────
    A0_E = scale_EE * gaussian_ring_kernel(M, sigma_EE)
    A0_I = scale_EI * gaussian_ring_kernel(M, sigma_EI)

    Ntot_E = M * N_E; Ntot_I = M * N_I; Ntot = Ntot_E + Ntot_I; MM = M * M

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
    R_I_oa = sol_oa.y[2 * M : 3 * M]
    P_I_oa = sol_oa.y[3 * M : 4 * M]
    A_E_oa_final = sol_oa.y[4 * M : 4 * M + MM, -1].reshape(M, M)
    A_I_oa_final = sol_oa.y[4 * M + MM : 4 * M + 2 * MM, -1].reshape(M, M)

    # Per-ensemble OA firing rates (full M × T arrays — needed for kymographs)
    T_oa = R_E_oa.shape[1]
    s_E_oa = np.empty((M, T_oa))
    s_I_oa = np.empty((M, T_oa))
    for k in range(T_oa):
        s_E_oa[:, k] = oa_synaptic_mean(n_pulse, R_E_oa[:, k], P_E_oa[:, k],
                                        s_hat, cn)
        s_I_oa[:, k] = oa_synaptic_mean(n_pulse, R_I_oa[:, k], P_I_oa[:, k],
                                        s_hat, cn)

    out = dict(
        # config
        M=M, N_E=N_E, N_I=N_I, T=T, plasticity=plasticity,
        sigma_EE=sigma_EE, sigma_EI=sigma_EI,
        bump_centre=bump_centre,
        ring=ring,
        # OA trajectories — FULL per-ensemble (M, T_oa)
        t_oa=sol_oa.t, s_E_oa=s_E_oa, s_I_oa=s_I_oa,
        R_E_oa=R_E_oa, P_E_oa=P_E_oa,
        R_I_oa=R_I_oa, P_I_oa=P_I_oa,
        # Final OA matrices
        A_E_oa_final=A_E_oa_final, A_I_oa_final=A_I_oa_final,
        # Initial kernels for reference
        A0_E=A0_E, A0_I=A0_I,
        # Bump IC
        # Initial rate-profile reference for the figure overlay
        r0_E_ic=r0_E.copy(), r0_I_ic=r0_I.copy(),
    )

    # ── Optional microscopic run ─────────────────────────────────────────────
    if run_tn:
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

        theta_E_t = sol_tn.y[0:Ntot_E].reshape(M, N_E, -1)
        theta_I_t = sol_tn.y[Ntot_E:Ntot].reshape(M, N_I, -1)
        A_E_tn_final = sol_tn.y[Ntot:Ntot + MM, -1].reshape(M, M)
        A_I_tn_final = sol_tn.y[Ntot + MM:Ntot + 2 * MM, -1].reshape(M, M)

        s_E_tn = tn_firing_rate_per_ensemble(theta_E_t, n_pulse, cn)
        s_I_tn = tn_firing_rate_per_ensemble(theta_I_t, n_pulse, cn)

        out.update(dict(
            t_tn=sol_tn.t, s_E_tn=s_E_tn, s_I_tn=s_I_tn,
            A_E_tn_final=A_E_tn_final, A_I_tn_final=A_I_tn_final,
        ))

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Figure: kymographs of bump activity + final profiles + final matrices
# ═══════════════════════════════════════════════════════════════════════════════

def plot_figure(res, savepath="theta_ring_bump.pdf",
                avg_window_frac=0.10):
    """2-row diagnostic figure for bump dynamics.

        row 0: kymographs s^E_m(t) and s^I_m(t) (M × t imshow)
        row 1: final activity profile (E + I) | final A^E | final A^I

    The "final" profile is averaged over the last `avg_window_frac`
    fraction of the simulation to smooth fast oscillations and reveal
    the underlying bump envelope.
    """
    set_prx_style()
    M = res["M"]
    T = res["T"]

    fig = plt.figure(figsize=(7.5, 9.0))
    gs = gridspec.GridSpec(
        nrows=3, ncols=3, figure=fig,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.45, wspace=0.55,
        left=0.075, right=0.97, top=0.905, bottom=0.085,
    )

    letters = iter("abcdefghij")

    # ── Row 0: kymographs of E and I firing rate ──────────────────────────
    s_E = res["s_E_oa"]          # (M, T_oa)
    s_I = res["s_I_oa"]
    t   = res["t_oa"]

    # Separate colour scales for E and I (different magnitudes typically).
    v_E = max(s_E.max(), 1e-6)
    v_I = max(s_I.max(), 1e-6)
    kym_kw = dict(aspect="auto", origin="lower", interpolation="nearest",
                  extent=(t[0], t[-1], -0.5, M - 0.5))

    # Sub-gridspec for row 0: two equal-width kymograph panels.
    ax_kE = fig.add_subplot(gs[0, :])
    im_kE = ax_kE.imshow(s_E, cmap="magma", vmin=0.0, vmax=v_E, **kym_kw)
    ax_kE.set_xlabel(r"time $t$")
    ax_kE.set_ylabel(r"ensemble $m$")
    ax_kE.set_title("E firing rate $s^E_m(t)$", fontsize=10)
    ax_kE.axhline(res["bump_centre"], color="white", lw=0.5,
                  ls="--", alpha=0.6)
    make_panel_label(ax_kE, "(a)", x=-0.12, y=1.06)
    cbE = fig.colorbar(im_kE, ax=ax_kE, pad=0.02, fraction=0.046, aspect=18)
    cbE.set_label(r"$s^E_m$", fontsize=10)
    cbE.ax.tick_params(labelsize=9)

    ax_kI = fig.add_subplot(gs[1, :])
    im_kI = ax_kI.imshow(s_I, cmap="magma", vmin=0.0, vmax=v_I, **kym_kw)
    ax_kI.set_xlabel(r"time $t$")
    ax_kI.set_ylabel(r"ensemble $m$")
    ax_kI.set_title("I firing rate $s^I_m(t)$", fontsize=10)
    ax_kI.axhline(res["bump_centre"], color="white", lw=0.5,
                  ls="--", alpha=0.6)
    make_panel_label(ax_kI, "(b)", x=-0.12, y=1.06)
    cbI = fig.colorbar(im_kI, ax=ax_kI, pad=0.02, fraction=0.046, aspect=18)
    cbI.set_label(r"$s^I_m$", fontsize=10)
    cbI.ax.tick_params(labelsize=9)

    # Subsequent panel labels start at (c).
    letters = iter("cdefghij")

    # ── Row 1: final E + I profile, final A^E, final A^I ───────────────────
    # Profile averaged over the last avg_window_frac fraction of time.
    n_avg = max(1, int(avg_window_frac * t.size))
    profile_E = s_E[:, -n_avg:].mean(axis=1)
    profile_I = s_I[:, -n_avg:].mean(axis=1)
    ic_profile_E = res["r0_E_ic"]   # initial rate-bump for visual reference

    ax_prof = fig.add_subplot(gs[2, 0])
    m_axis = np.arange(M)
    ax_prof.plot(m_axis, profile_E, "-",  color="royalblue",
                 lw=1.6, label=r"$\langle s^E_m\rangle_t$")
    ax_prof.plot(m_axis, profile_I, "-",  color="darkorange",
                 lw=1.6, label=r"$\langle s^I_m\rangle_t$")
    # Thin grey curve: initial E bump amplitude (rescaled for comparison)
    if profile_E.max() > 0:
        ic_scaled = ic_profile_E * profile_E.max() / max(ic_profile_E.max(), _EPS)
        ax_prof.plot(m_axis, ic_scaled, ":", color="0.4", lw=1.0,
                     label="IC bump (rescaled)")
    ax_prof.set_xlabel(r"ensemble $m$")
    ax_prof.set_ylabel(r"firing rate (final $t$)")
    ax_prof.legend(loc="upper right", frameon=False, fontsize=9,
                   handlelength=2.0)
    ax_prof.set_xlim(-0.5, M - 0.5)
    make_panel_label(ax_prof, f"({next(letters)})", x=-0.20, y=1.06)

    # Final coupling matrices A^E, A^I.
    for col_idx, (A, sym) in enumerate([
        (res["A_E_oa_final"], r"$A^E$"),
        (res["A_I_oa_final"], r"$A^I$"),
    ], start=1):
        ax = fig.add_subplot(gs[2, col_idx])
        vmax = max(A.max(), 1e-12)
        im = ax.imshow(A, cmap="viridis", vmin=0.0, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_xlabel(r"pre $n$")
        ax.set_ylabel(r"post $m$")
        ax.set_title(sym + " (final)", fontsize=10)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046, aspect=18)
        cb.set_label(sym + r"$_{mn}$", fontsize=10)
        cb.locator = plt.MaxNLocator(nbins=4)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=9)
        tick_step = max(1, M // 6)
        ticks = np.arange(0, M, tick_step)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        make_panel_label(ax, f"({next(letters)})", x=-0.20, y=1.06)

    fig.suptitle(rf"Bump dynamics — $\sigma_{{EE}}={res['sigma_EE']:g}$, "
                 rf"$\sigma_{{EI}}={res['sigma_EI']:g}$, "
                 rf"bump centre $m_0={res['bump_centre']}$",
                 fontsize=11, y=0.985)

    fig.savefig(savepath, bbox_inches="tight")
    print(f"\nFigure saved → {savepath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CONFIG = dict(
        # Homogeneous ring: many ensembles, finer spatial resolution
        M=64, N_E=200, N_I=50, T=20.0,
        # Local (within-ensemble) synaptic strengths
        J_EE_loc= 15.0, J_EI_loc=-10.0,
        J_IE_loc= 0.0, J_II_loc=-0.0,
        # Global (between-ensemble) scales — strong enough to support
        # a self-sustained bump given the kernel widths below.
        J_glob_E=0.0, J_glob_I=0.0,
        # Membrane timescales
        tau_E=1.0, tau_I=2.0,
        # Plasticity: pick a rule.  γ=0.0 + Oja keeps weights bounded
        # via the normalisation term rather than passive decay.
        plasticity="hebbian",
        mu=0.0, gamma=0.0,
        n_pulse=10, n1=3, n2=3, n3=3,
        # HOMOGENEOUS excitabilities (no gradient on this ring)
        eta_bar_E=-10.0, delta_E=1.0,
        eta_bar_I=-10.0, delta_I=1.0,
        # Independent connectivity kernel widths.  σ_EI > σ_EE
        # implements the classic "Mexican-hat" lateral-inhibition
        # regime that supports stable activity bumps.
        sigma_EE=5.0,  scale_EE=1.0,
        sigma_EI=20.0,  scale_EI=1.0,
        # Bump initial condition — specified in firing-rate units.
        # The conformal map (Montbrió–Pazó–Roxin) translates these to
        # the matching Kuramoto order-parameter values automatically.
        bump_centre=None,             # → M//2
        r_bump=20.0,                   # peak E rate at the bump
        r_base=0.0,                  # baseline E rate elsewhere
        r_I_base=0.0,                # uniform I baseline rate
        v_E_init=-2.0, v_I_init=-3.0,
        bump_width=3.0,
        # Run controls
        run_tn=False,                 # OA-only by default for speed
        seed=42, method="RK45", rtol=1e-6, atol=1e-8,
    )

    res = simulate(**CONFIG)
    plot_figure(res, savepath="theta_ring_bump.pdf")
    plt.show()
