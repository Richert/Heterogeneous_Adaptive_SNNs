"""
Bump Formation in a Homogeneous Ring of Theta-Neuron Ensembles
==============================================================
Mean-field (Ott–Antonsen) ring model of M single-population theta-neuron
ensembles with non-local synaptic coupling.  The within-ensemble structure
is reduced to a single Lorentzian distribution of excitabilities; the
spatial structure comes from a translation-invariant coupling kernel
w(x) along the ring.

This version follows the formulation of Schmidt & Avitabile, "Bumps and
oscillons in networks of spiking neurons", Chaos 30 (2020) 033133, but
expresses the model as a network of theta neurons rather than QIF
neurons (the two are conformally equivalent — see notes below).

  • Ring geometry.  M ensembles sit at evenly spaced positions
        x_m = (m/M − 1/2) · L,    m = 0, 1, …, M-1,
    on a periodic 1-D domain of length L (so x ∈ (−L/2, L/2]).

  • Two coupling kernels are available, selected via `coupling_kernel`:

      "schmidt_avitabile":  Mexican-hat bi-exponential (Eq. 5 of the
        paper) with long-range inhibition and short-range excitation:
              w(x) = exp(-|x|) − ¼ · exp(-|x|/2)
        Normalisation: ∫ w(y) dy = 1 on ℝ.

      "cosine_wizard":  cosine "wizard hat" used in Laing & Omel'chenko
        and many earlier studies, parameterised on the canonical ring
        of circumference 2π and rescaled to L:
              K(x) = k_0 + k_1 · cos(2π x / L)
        For the classical local-excitation / lateral-inhibition regime
        take k_0 ≥ 0 small and k_1 > 0.

    In both cases the synaptic drive at ring site m is
              drive_m = J · δx · Σ_n W_{mn} · s_n,
    where W_{mn} = w(d(x_m, x_n)), d is the periodic ring distance,
    δx = L/M is the lattice spacing, and s_n is the OA pulse-mean of
    population n.  The δx factor turns the discrete sum into a midpoint
    approximation of the continuous convolution.

  • Excitability statistics are HOMOGENEOUS along the ring: η̄ and Δ
    are scalars shared by every ensemble.  Spatial structure comes from
    the coupling kernel and the bump initial condition only.

  • The IC is a localised bump.  Off-bump ring sites are placed at the
    UNCOUPLED OA fixed point (R*, Ψ*) of `_theta_oa_pop`, found by
    short relaxation; the bump centre is set by a target (r_bump, v_bump)
    via the Montbrió–Pazó–Roxin conformal map W = (1−Z)/(1+Z) with
    W = π·r + i·v.  Intermediate sites linearly interpolate in the
    complex Z = R·exp(iΨ) (NOT in R, Ψ separately — Ψ wraps).

  • Synaptic plasticity (Hebbian, anti-Hebbian, Oja) acts on a single
    M×M coupling matrix A on top of the static kernel W.  When μ = γ = 0
    the matrix is frozen at its initial value W and the coupling reduces
    to the static Schmidt-Avitabile / cosine setup.  When μ, γ ≠ 0 the
    matrix evolves under the chosen rule, using pulse-mean factors at
    pulse exponents n1, n2, n3 (same machinery as the original E/I
    version of this script, collapsed to one population).

  • Pulse-mean vs firing rate.  The synaptic coupling is driven by the
    PULSE-MEAN  s_m = c_n · ⟨(1−cosθ)^n⟩  (see `oa_synaptic_mean`),
    which is what one neuron actually receives from its presynaptic
    partners.  The PHYSICAL firing rate of a QIF/theta-neuron population
    under the OA ansatz is  r = Re(W)/π, where W = (1−Z)/(1+Z) is the
    Montbrió–Pazó–Roxin conformal image of Z = R·exp(iΨ).  The two
    quantities are NOT proportional: for very negative η̄ the silent
    fixed point migrates to θ* near ±π, where (1−cosθ)^n is large
    even though no spikes occur, so the pulse-mean spuriously inflates.
    Kymographs and profiles in this script always report the physical
    firing rate; the pulse-mean is used internally for the synaptic drive.

QIF ↔ theta-neuron parameter translation
----------------------------------------
The QIF model of Schmidt & Avitabile,
        V̇_i = V_i² + η_i + J · s_i ,
maps to the theta-neuron model used here,
        θ̇_j = (1 − cosθ_j) + (1 + cosθ_j) · (η_j + κ · I_j) ,
via the standard conformal substitution V = tan(θ/2) (Ermentrout-Kopell).
Under this map:

    QIF η̄                ↔  theta η̄ (same scalar)
    QIF Δ  (Lorentzian)   ↔  theta Δ (same scalar)
    QIF J · (w ⊗ r)       ↔  theta κ · (W · s),   κ = J, s = pulse-mean

Two subtleties:

  (i) Schmidt & Avitabile use δ-pulse synapses (a(t) = δ(t)) so the
      "rate-like quantity" driving the kernel convolution is the
      instantaneous QIF firing rate r = Re(W)/π directly.  In the
      theta-neuron formulation we instead use the pulse-mean
      s = c_n ⟨(1−cosθ)^n⟩ as the rate-like quantity (the natural
      coupling channel for theta neurons), with n_pulse controlling
      pulse width.  Both reduce to the same OA closure as n_pulse → ∞;
      for finite n_pulse the coupling is smoothed in θ but the
      bifurcation structure is qualitatively the same.

  (ii) The QIF Lorentzian half-width is sometimes called Δ and
       sometimes called γ in the literature.  The script always uses Δ.

For Schmidt & Avitabile's Fig. 1 parameter set:
        η̄ = −10,  Δ = 2,  J = 15·√2 ≈ 21.213,  L = 50,
        coupling_kernel = "schmidt_avitabile"
the theta-neuron analogue uses the SAME numerical values for η̄, Δ, κ=J
and the SAME L and kernel.  See the entry-point CONFIG below.

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

    c_n = 2^n · n!·n! / (2n)! — the reciprocal of the binomial coefficient
    C(2n,n)/2^{2n}.  Computed via log-factorials to stay clean for moderate n.
    """
    from math import lgamma, exp
    log_cn = n * np.log(2.0) + 2 * lgamma(n + 1) - lgamma(2 * n + 1)
    return float(exp(log_cn))


def fourier_coeffs_s(n):
    """Fourier coefficients ŝ_k of (1-cosθ)^n for k = -n..n.

    (1 − cosθ)^n = Σ_{k=-n}^{n} ŝ_k e^{ikθ}.  Computed by FFT for
    numerical robustness.
    """
    M_grid = 4 * n + 8
    theta = 2 * np.pi * np.arange(M_grid) / M_grid
    f = (1.0 - np.cos(theta)) ** n
    F = np.fft.fft(f) / M_grid
    s_hat = np.zeros(2 * n + 1, dtype=complex)
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


PLASTICITY_RULES = ("hebbian", "antihebbian", "oja")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Ring geometry and coupling kernels
# ═══════════════════════════════════════════════════════════════════════════════

def ring_positions(M, L):
    """Evenly spaced ring positions x_m ∈ (−L/2, L/2] for m = 0..M-1.

    Schmidt & Avitabile use x_m = (m/M − ½)·L (their domain is open at
    −L/2 and closed at +L/2); we follow that convention.  Equivalent up
    to a global shift to placing x_m at the centre of the m-th cell.
    """
    return (np.arange(M) / M - 0.5) * L


def periodic_ring_distance(M, L):
    """M×M matrix of periodic ring distances d(x_m, x_n) on length-L circle.

    d(x, y) = min(|x − y|, L − |x − y|),  so 0 ≤ d ≤ L/2.
    """
    x = ring_positions(M, L)
    d = np.abs(x[:, None] - x[None, :])
    return np.minimum(d, L - d)


def schmidt_avitabile_kernel(x, k0: float = 0.25, k1: float = 2.0):
    """Bi-exponential Mexican-hat kernel from Schmidt & Avitabile (Eq. 5):

            w(x) = exp(−|x|) − ¼ · exp(−|x|/2)

    Long-range inhibition (the second term decays more slowly than the
    first) on a base of short-range excitation.  Normalised so that
    ∫_{−∞}^{∞} w(y) dy = 1.
    """
    ax = np.abs(x)
    return np.exp(-ax) - k0 * np.exp(-ax / k1)


def cosine_wizard_kernel(x, L, k0=0.1, k1=0.3):
    """Cosine "wizard hat" kernel (Laing-Omel'chenko form, rescaled to L):

            K(x) = k_0 + k_1 · cos(2π x / L)

    On the canonical 2π-circle (L = 2π) this reduces to k_0 + k_1·cos(x);
    the rescaling keeps the shape comparable across different domain
    lengths.  Defaults k_0 = 0.1, k_1 = 0.3 come from Laing-Omel'chenko
    (their Eq. 4 with the asymmetry parameter B = 0).
    """
    return k0 + k1 * np.cos(2.0 * np.pi * x / L)


def build_coupling_kernel(M, L, kind="schmidt_avitabile", **kwargs):
    """Build the static M×M coupling matrix W_{mn} = w(d(x_m, x_n)).

    Returns a tuple (W, dx, info_dict).  `dx = L/M` is the lattice
    spacing that converts the discrete sum  Σ_n W_{mn} s_n  to the
    midpoint approximation of the continuous integral ∫ w(x−y) s(y) dy.

    Diagonal entries W_{mm} are kept (they correspond to the
    within-site contribution of the kernel at zero distance) — this
    matches the standard convolution discretisation.  If you want to
    explicitly suppress self-coupling, set `zero_diagonal=True` in
    kwargs.

    Available kernels (selected by `kind`):
        "schmidt_avitabile":  w(x) = exp(-|x|) - ¼·exp(-|x|/2)
        "cosine_wizard":      w(x) = k0 + k1·cos(2π x / L)
                              (kwargs: k0=0.1, k1=0.3)

    Extra kwargs:
        zero_diagonal: bool (default False) — zero out W_{mm}
    """
    d = periodic_ring_distance(M, L)
    zero_diagonal = bool(kwargs.pop("zero_diagonal", False))
    if kind == "schmidt_avitabile":
        k0 = float(kwargs.pop("k0", 0.25))
        k1 = float(kwargs.pop("k1", 2.0))
        if kwargs:
            raise TypeError(f"schmidt_avitabile takes no extra kwargs, "
                            f"got {list(kwargs)}")
        W = schmidt_avitabile_kernel(d, k0, k1)
        info = dict(kind=kind, L=L, M=M)
    elif kind == "cosine_wizard":
        k0 = float(kwargs.pop("k0", 0.1))
        k1 = float(kwargs.pop("k1", 0.3))
        if kwargs:
            raise TypeError(f"cosine_wizard got unexpected kwargs "
                            f"{list(kwargs)}")
        # For the cosine kernel, the *signed* distance matters because
        # cos is even — but periodic_ring_distance is already |x|.
        # cos(2π d/L) gives the symmetric profile on the ring.
        W = cosine_wizard_kernel(d, L, k0=k0, k1=k1)
        info = dict(kind=kind, L=L, M=M, k0=k0, k1=k1)
    else:
        raise ValueError(f"Unknown coupling_kernel {kind!r}; expected one "
                         f"of 'schmidt_avitabile', 'cosine_wizard'")

    if zero_diagonal:
        np.fill_diagonal(W, 0.0)

    dx = L / M
    return W, dx, info


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Conformal maps Z ↔ W ↔ (r, v) and the uncoupled OA fixed point
# ═══════════════════════════════════════════════════════════════════════════════

def montbrio_rate_to_kuramoto(r, v=0.0):
    """Montbrió–Pazó–Roxin conformal map: (r, v) → (R, Ψ).

    QIF mean variable W = π·r − i·v relates to Kuramoto Z = ⟨e^{iθ}⟩
    via the Cayley transform W = (1 − Z)/(1 + Z), so

        Z = (1 − W) / (1 + W),   W = π·r − i·v.

    The −i·v sign is the Montbrió-Pazó-Roxin convention (PRX 5, 021028
    (2015), Eq. 6) — it is consistent with the QIF mean-field equations
        ṙ = Δ/π + 2 r v,
        v̇ = v² + η̄ + J(w⊗r) − π² r²
    used by Schmidt & Avitabile.  See `kuramoto_to_montbrio_rate` for
    the inverse map and the firing-rate identity r = Re(W)/π.

    For v = 0:
        r < 1/π:   Ψ = 0,    R = (1 − π r)/(1 + π r)
        r > 1/π:   Ψ = π,    R = (π r − 1)/(π r + 1)
    so high firing rate corresponds to high coherence with Ψ → π.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    W = np.pi * r - 1j * v
    Z = (1.0 - W) / (1.0 + W)
    R = np.clip(np.abs(Z), _EPS, 1.0 - _EPS)
    Psi = np.angle(Z)
    return R, Psi


def kuramoto_to_montbrio_rate(R, Psi):
    """Inverse conformal map: (R, Ψ) → (r, v) with r = Re(W)/π, v = −Im(W).

    With the MPR convention W = π·r − i·v (see `montbrio_rate_to_kuramoto`),
    we recover r = Re(W)/π and v = −Im(W).  The firing-rate identity
    r = Re(W)/π gives the PHYSICAL QIF firing rate of the population
    under the OA ansatz, which is monotone in η̄ as required.  It is
    NOT the same as the pulse-mean c_n⟨(1−cosθ)^n⟩ that enters the
    synaptic coupling (see module docstring).
    """
    R = np.clip(np.asarray(R, dtype=float), _EPS, 1.0 - _EPS)
    Psi = np.asarray(Psi, dtype=float)
    Z = R * np.exp(1j * Psi)
    W = (1.0 - Z) / (1.0 + Z)
    r = W.real / np.pi
    v = -W.imag                # MPR convention: W = πr − iv
    return r, v


def build_ring(M, eta_bar, delta):
    """Bundle the OA excitability parameters into a dict.

    The ring is HOMOGENEOUS: η̄ and Δ are scalars shared by every site.
    This is a stub left in place so future heterogeneous variants
    (η̄_m, Δ_m varying along the ring) have an obvious entry point.
    """
    return dict(eta_bar=float(eta_bar), delta=float(delta))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  OA mean-field for the ring of single-population ensembles
# ═══════════════════════════════════════════════════════════════════════════════
#
# Per-ensemble OA state: (R_m, Ψ_m) for m = 1..M, plus one M×M adaptive
# coupling matrix A.  Flat layout:
#     y[0      : M     ]   R
#     y[M      : 2M    ]   Ψ
#     y[2M     : 2M+M² ]   A (post m × pre n)
# ─────────────────────────────────────────────────────────────────────────────

def _theta_oa_pop(R, Psi, delta, E, tau=1.0):
    """Standard theta-neuron OA equations for one population, with an
    optional membrane timescale τ.  Setting τ≠1 rescales the entire RHS
    by 1/τ, mirroring the microscopic τ·θ̇ = (1−cosθ) + (1+cosθ)η.

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


def uncoupled_fixed_point(eta_bar, delta, tau=1.0,
                          T_relax=200.0, rtol=1e-9, atol=1e-12):
    """(R*, Ψ*) for the uncoupled OA population by direct relaxation.

    Integrates `_theta_oa_pop` from a generic IC for time `T_relax` and
    returns the final (R, Ψ) as floats.  For a single Lorentzian
    population with no external drive there is a unique stable fixed
    point at every (η̄, Δ) with Δ > 0, so the initial guess does not
    matter as long as it is not exactly on an unstable saddle.
    """
    def rhs(t, y):
        R, Psi = y
        dR, dPsi = _theta_oa_pop(np.array([R]), np.array([Psi]),
                                 float(delta), np.array([eta_bar]),
                                 tau=tau)
        return [float(dR[0]), float(dPsi[0])]

    sol = solve_ivp(rhs, (0.0, T_relax), [0.5, 0.0],
                    method="RK45", rtol=rtol, atol=atol,
                    dense_output=False)
    if not sol.success:
        raise RuntimeError(
            f"uncoupled_fixed_point relaxation failed: {sol.message}")
    R_star = float(np.clip(sol.y[0, -1], _EPS, 1.0 - _EPS))
    Psi_star = float(sol.y[1, -1])
    Psi_star = (Psi_star + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
    return R_star, Psi_star


def oa_rhs(t, y, ring,
           M, kappa, dx,
           n_pulse, n1, n2, n3,
           cn, cn1, cn2, cn3,
           s_hat, s_hat1, s_hat2, s_hat3, s_hat1_sq,
           mu, gamma, plasticity,
           tau=1.0, coupling_channel="firing_rate"):
    """Mean-field RHS for the single-population ring with non-local kernel.

    State layout (length 2M + M²):  y = [R, Ψ, vec(A)].

    The synaptic drive at ring site m is
            drive_m = κ · dx · Σ_n A_{mn} · u_n,
    where u_n is the "rate-like" quantity at site n.  Two choices for u:

      coupling_channel = "firing_rate"  (default):
          u = r = Re(W)/π — the physical QIF firing rate.  This matches
          Schmidt & Avitabile's QIF formulation exactly: with κ = J they
          use, the QIF mean-field convolution J·(w ⊗ r) and the theta
          analogue with this channel produce numerically identical drives.

      coupling_channel = "pulse":
          u = c_n ⟨(1−cosθ)^n⟩ — the OA pulse-mean.  This is the natural
          coupling channel for an explicit theta-neuron network with
          finite-width pulses (cf. Laing-Omel'chenko Eq. 2).  As n_pulse
          increases it approaches the firing-rate channel up to a
          constant; at moderate n the two differ substantially, so
          mixing channels would mean re-tuning κ.

    dx · A is the discretisation of the convolution operator w ⊗ u.

    Plasticity acts on A with the same machinery as the original E/I
    script, collapsed to a single population.  When μ = γ = 0 the
    matrix A stays frozen at its initial (kernel) value.  Note that
    plasticity factors use the pulse-mean at exponents n1, n2, n3
    regardless of the synaptic coupling channel — the Hebbian rule is
    defined in terms of pre-/postsynaptic pulse trains, not firing
    rates.
    """
    R   = y[0 * M : 1 * M]
    Psi = y[1 * M : 2 * M]
    A   = y[2 * M : 2 * M + M * M].reshape(M, M)

    # Rate-like quantity that drives synapses.
    if coupling_channel == "firing_rate":
        # QIF firing rate r = Re((1−Z)/(1+Z))/π — matches Schmidt-Avitabile.
        u, _ = kuramoto_to_montbrio_rate(R, Psi)
    elif coupling_channel == "pulse":
        # OA pulse-mean — the natural channel for finite-width pulse coupling.
        u = oa_synaptic_mean(n_pulse, R, Psi, s_hat, cn)
    else:
        raise ValueError(f"Unknown coupling_channel {coupling_channel!r}; "
                         f"expected 'firing_rate' or 'pulse'")

    # Non-local synaptic drive:  drive_m = κ · dx · Σ_n A_{mn} · u_n.
    drive = kappa * dx * (A @ u)                            # (M,)

    Eeff = ring["eta_bar"] + drive
    dR, dPsi = _theta_oa_pop(R, Psi, ring["delta"], Eeff, tau=tau)

    # ── Plasticity kernel expectations ────────────────────────────────────────
    S_n2 = oa_synaptic_mean(n2, R, Psi, s_hat2, cn2)         # post channel
    S_n3 = oa_synaptic_mean(n3, R, Psi, s_hat3, cn3)         # pre  channel
    hebb = np.outer(S_n2, S_n3)                              # (M,M)

    if plasticity == "hebbian":
        dA = mu * hebb - gamma * A
    elif plasticity == "antihebbian":
        S_n1 = oa_synaptic_mean(n1, R, Psi, s_hat1, cn1)
        dA = mu * (S_n1[None, :] - A * hebb) - gamma * A
    elif plasticity == "oja":
        Ssq = oa_pulse_squared_mean(n1, R, Psi, s_hat1_sq, cn1)
        dA = mu * (hebb - A * Ssq[:, None]) - gamma * A
    else:
        raise ValueError(f"Unknown plasticity rule {plasticity!r}")

    return np.concatenate([dR, dPsi, dA.ravel()])


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Top-level simulation: build the ring, run OA, return everything
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(*,
             # Spatial discretisation
             M=128, L=50.0, T=40.0,
             # Coupling: kernel choice, strength κ, optional zero-diagonal flag
             coupling_kernel="schmidt_avitabile",
             kernel_kwargs=None,
             kappa=15.0 * np.sqrt(2.0),
             # Coupling channel: "firing_rate" matches Schmidt-Avitabile's QIF
             # formulation (the kernel acts on r = Re(W)/π); "pulse" uses the
             # OA pulse-mean c_n·⟨(1−cosθ)^n⟩, the natural channel for an
             # explicit theta-neuron network with finite-width pulses.
             coupling_channel="firing_rate",
             # Plasticity
             plasticity="hebbian", mu=0.0, gamma=0.0,
             n_pulse=2, n1=2, n2=2, n3=3,
             # Membrane timescale (acts only on θ-dynamics; plasticity unaffected)
             tau=1.0,
             # Homogeneous Lorentzian excitabilities
             eta_bar=-10.0, delta=2.0,
             # Bump initial condition — bump CENTRE in (r, v); off-bump sites
             # sit at the uncoupled OA fixed point.  Intermediate sites linearly
             # interpolate in the complex order parameter Z = R·exp(iΨ).
             bump_centre=None,         # x position; default 0 (centre of ring)
             r_bump=2.0,               # peak firing rate at the bump
             v_bump=0.0,               # mean membrane potential at the bump
             bump_width=2.5,           # σ of Gaussian-in-ring-distance for IC
             # External transient stimulus (mimics S&A's protocol of inducing
             # the bump with a brief localised current; set amplitude > 0 to
             # use, otherwise the IC alone seeds the bump).
             stim_amplitude=0.0,
             stim_centre=None,
             stim_halfwidth=2.5,
             stim_t_on=0.0, stim_t_off=5.0,
             # Run controls
             method="RK45", rtol=1e-6, atol=1e-8,
             ):
    """Single-run bump simulation on a single-population theta-neuron ring.

    The IC is specified in firing-rate units only at the bump centre, via
    a target (r_bump, v_bump).  The off-bump baseline is set to the
    uncoupled OA fixed point so that, with κ = 0, the ring stays flat at
    the physical low-rate equilibrium and the kymograph shows no spurious
    transient.  Intermediate sites are obtained by linear interpolation
    of the complex order parameter Z = R·exp(iΨ), which is well-defined
    even when Ψ_bump and Ψ_* sit on opposite sides of the branch cut.

    Returns a dict with full per-ensemble trajectories for both the
    synaptic pulse-mean (s_oa) AND the physical firing rate (r_oa).
    Each activity array is shaped (M, T_eval); each row is one ring site.
    Kymographs and profiles in `plot_figure` use the physical firing
    rate; the pulse-mean is kept for diagnostics.
    """
    if kernel_kwargs is None:
        kernel_kwargs = {}

    print(f"Building ring (M={M}, L={L}, kernel={coupling_kernel}) …")
    ring = build_ring(M, eta_bar, delta)
    W_kernel, dx, kernel_info = build_coupling_kernel(
        M, L, kind=coupling_kernel, **kernel_kwargs)

    # Precompute pulse normalisations and Fourier coefficients
    cn  = coupling_norm(n_pulse); s_hat  = fourier_coeffs_s(n_pulse)
    cn1 = coupling_norm(n1);      s_hat1 = fourier_coeffs_s(n1)
    cn2 = coupling_norm(n2);      s_hat2 = fourier_coeffs_s(n2)
    cn3 = coupling_norm(n3);      s_hat3 = fourier_coeffs_s(n3)
    s_hat1_sq = fourier_coeffs_s(2 * n1)

    # ── Bump initial condition ──────────────────────────────────────────────
    x = ring_positions(M, L)
    if bump_centre is None:
        bump_centre_x = 0.0          # centre of ring at x = 0
    else:
        bump_centre_x = float(bump_centre)
    # Ring-periodic distance from each site to bump centre
    d_from_centre = np.abs(x - bump_centre_x)
    d_from_centre = np.minimum(d_from_centre, L - d_from_centre)
    bump_profile = np.exp(-(d_from_centre ** 2) / (2.0 * bump_width ** 2))

    # Off-bump baseline: uncoupled OA fixed point.
    R_star, Psi_star = uncoupled_fixed_point(eta_bar, delta, tau=tau)
    Z_star = R_star * np.exp(1j * Psi_star)

    # Bump-centre target from the conformal map.
    R_peak, Psi_peak = montbrio_rate_to_kuramoto(np.array(r_bump),
                                                  np.array(v_bump))
    Z_peak = float(R_peak) * np.exp(1j * float(Psi_peak))

    # Linear interpolation in the COMPLEX plane along the Gaussian envelope.
    Z0 = Z_star + bump_profile * (Z_peak - Z_star)
    R0 = np.clip(np.abs(Z0), _EPS, 1.0 - _EPS)
    P0 = np.angle(Z0)

    # Initial firing-rate profile (for figure overlay).
    r0_ic, _ = kuramoto_to_montbrio_rate(R0, P0)

    # ── External stimulus profile (S&A-style transient bump induction) ──────
    use_stim = stim_amplitude != 0.0
    if use_stim:
        if stim_centre is None:
            stim_centre_x = bump_centre_x
        else:
            stim_centre_x = float(stim_centre)
        d_stim = np.abs(x - stim_centre_x)
        d_stim = np.minimum(d_stim, L - d_stim)
        stim_spatial = (d_stim <= stim_halfwidth).astype(float)
    else:
        stim_spatial = None

    # ── Initial coupling matrix: the static kernel ──────────────────────────
    A0 = W_kernel.copy()
    MM = M * M

    # ── OA run ──────────────────────────────────────────────────────────────
    print(f"Running OA (plasticity={plasticity}, mu={mu}, gamma={gamma}) …")
    print(f"  uncoupled FP: (R*, Ψ*) = ({R_star:.4f}, {Psi_star:+.4f})")
    y0 = np.concatenate([R0, P0, A0.ravel()])

    if use_stim:
        def rhs_with_stim(t, y):
            base = oa_rhs(t, y, ring, M, kappa, dx,
                          n_pulse, n1, n2, n3, cn, cn1, cn2, cn3,
                          s_hat, s_hat1, s_hat2, s_hat3, s_hat1_sq,
                          mu, gamma, plasticity, tau, coupling_channel)
            if stim_t_on <= t <= stim_t_off:
                # Add a transient external current to η_eff at the stim sites
                # by perturbing dR, dPsi directly.  Re-run _theta_oa_pop with
                # an η offset; use the same coupling_channel as the base RHS.
                R   = y[0 * M : 1 * M]
                Psi = y[1 * M : 2 * M]
                A   = y[2 * M : 2 * M + MM].reshape(M, M)
                if coupling_channel == "firing_rate":
                    u, _ = kuramoto_to_montbrio_rate(R, Psi)
                else:  # "pulse"
                    u = oa_synaptic_mean(n_pulse, R, Psi, s_hat, cn)
                drive_with_stim = (kappa * dx * (A @ u)
                                   + stim_amplitude * stim_spatial)
                Eeff = ring["eta_bar"] + drive_with_stim
                dR_s, dP_s = _theta_oa_pop(R, Psi, ring["delta"], Eeff,
                                            tau=tau)
                # Overwrite only the R, Ψ slots of `base`
                base[0 * M : 1 * M] = dR_s
                base[1 * M : 2 * M] = dP_s
            return base
        rhs_fn = rhs_with_stim
        args = ()
    else:
        rhs_fn = oa_rhs
        args = (ring, M, kappa, dx,
                n_pulse, n1, n2, n3, cn, cn1, cn2, cn3,
                s_hat, s_hat1, s_hat2, s_hat3, s_hat1_sq,
                mu, gamma, plasticity, tau, coupling_channel)

    sol = solve_ivp(rhs_fn, (0, T), y0, method=method,
                    args=args, rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"OA failed: {sol.message}")
    print(f"  done — {sol.t.size} steps, {sol.nfev} evals")

    R_oa = sol.y[0 * M : 1 * M]
    P_oa = sol.y[1 * M : 2 * M]
    A_oa_final = sol.y[2 * M : 2 * M + MM, -1].reshape(M, M)

    # Both the synaptic pulse-mean and the physical firing rate.
    T_oa = R_oa.shape[1]
    s_oa = np.empty((M, T_oa))
    r_oa = np.empty((M, T_oa))
    for k in range(T_oa):
        s_oa[:, k] = oa_synaptic_mean(n_pulse, R_oa[:, k], P_oa[:, k],
                                       s_hat, cn)
        r_oa[:, k], _ = kuramoto_to_montbrio_rate(R_oa[:, k], P_oa[:, k])

    out = dict(
        # config
        M=M, L=L, T=T, plasticity=plasticity,
        coupling_kernel=coupling_kernel, kernel_info=kernel_info,
        coupling_channel=coupling_channel,
        kappa=kappa, dx=dx,
        bump_centre_x=bump_centre_x,
        ring=ring,
        # Spatial grid
        x=x,
        # Uncoupled fixed point (off-bump baseline) for reference
        R_star=R_star, Psi_star=Psi_star,
        # OA trajectories — FULL per-ensemble (M, T_oa)
        t_oa=sol.t,
        s_oa=s_oa,                  # pulse-mean (synaptic drive)
        r_oa=r_oa,                  # physical firing rate
        R_oa=R_oa, P_oa=P_oa,
        # Final coupling matrix (= W_kernel when plasticity is off)
        A_oa_final=A_oa_final,
        # Initial coupling kernel for reference
        A0=A0, W_kernel=W_kernel,
        # Initial firing-rate profile for the figure overlay
        r0_ic=r0_ic,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Figure: kymograph + final profile + coupling matrix
# ═══════════════════════════════════════════════════════════════════════════════

def plot_figure(res, savepath="theta_ring_bump.pdf",
                avg_window_frac=0.10):
    """3-row diagnostic figure for bump dynamics (single-population ring).

        row 0: kymograph r_m(t) (physical QIF firing rate)
        row 1: final activity profile r_m vs x_m
        row 2: coupling matrix A and kernel cross-section w(x)

    All firing-rate quantities are the physical QIF rate r = Re(W)/π
    (see module docstring), NOT the pulse-mean that drives the synapses.
    """
    set_prx_style()
    M = res["M"]
    x = res["x"]
    L = res["L"]

    fig = plt.figure(figsize=(7.5, 9.0))
    gs = gridspec.GridSpec(
        nrows=3, ncols=3, figure=fig,
        height_ratios=[1.2, 1.0, 1.0],
        hspace=0.55, wspace=0.55,
        left=0.10, right=0.97, top=0.90, bottom=0.085,
    )

    # ── Row 0: kymograph of PHYSICAL firing rate ──────────────────────────────
    r_t = res["r_oa"]              # (M, T_oa)
    t   = res["t_oa"]

    v_max = max(r_t.max(), 1e-6)
    kym_kw = dict(aspect="auto", origin="lower", interpolation="nearest",
                  extent=(t[0], t[-1], x[0], x[-1]))

    ax_k = fig.add_subplot(gs[0, :])
    im_k = ax_k.imshow(r_t, cmap="magma", vmin=0.0, vmax=v_max, **kym_kw)
    ax_k.set_xlabel(r"time $t$")
    ax_k.set_ylabel(r"position $x$")
    ax_k.set_title(r"firing rate $r(x, t)$", fontsize=10)
    ax_k.axhline(res["bump_centre_x"], color="white", lw=0.5,
                 ls="--", alpha=0.6)
    make_panel_label(ax_k, "(a)", x=-0.10, y=1.06)
    cb = fig.colorbar(im_k, ax=ax_k, pad=0.02, fraction=0.046, aspect=18)
    cb.set_label(r"$r$", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    letters = iter("bcdefghij")

    # ── Row 1 (left two panels merged): final profile vs x ────────────────────
    n_avg = max(1, int(avg_window_frac * t.size))
    profile = r_t[:, -n_avg:].mean(axis=1)
    ic_profile = res["r0_ic"]

    ax_prof = fig.add_subplot(gs[1, :2])
    ax_prof.plot(x, profile, "-", color="royalblue", lw=1.6,
                 label=r"$\langle r(x)\rangle_t$ (final)")
    if profile.max() > 0:
        ic_scaled = ic_profile * profile.max() / max(ic_profile.max(), _EPS)
        ax_prof.plot(x, ic_scaled, ":", color="0.4", lw=1.0,
                     label="IC bump (rescaled)")
    ax_prof.axvline(res["bump_centre_x"], color="0.7", lw=0.5, ls="--")
    ax_prof.set_xlabel(r"position $x$")
    ax_prof.set_ylabel(r"firing rate (final $t$)")
    ax_prof.legend(loc="best", frameon=False, fontsize=9, handlelength=2.0)
    ax_prof.set_xlim(x[0], x[-1])
    make_panel_label(ax_prof, f"({next(letters)})", x=-0.13, y=1.06)

    # ── Row 1 (right panel): kernel cross-section w(x) ────────────────────────
    ax_w = fig.add_subplot(gs[1, 2])
    # Take the kernel row passing through the bump centre — that's the
    # actual one-dimensional kernel shape on the ring.
    centre_idx = int(np.argmin(np.abs(x - res["bump_centre_x"])))
    w_row = res["W_kernel"][centre_idx]
    # Re-order so that the centre site is in the middle of the plot.
    order = np.argsort(((x - x[centre_idx] + L / 2) % L) - L / 2)
    x_sorted = x[order] - x[centre_idx]
    x_sorted = ((x_sorted + L / 2) % L) - L / 2
    sort2 = np.argsort(x_sorted)
    ax_w.plot(x_sorted[sort2], w_row[order][sort2], "-",
              color="darkred", lw=1.5)
    ax_w.axhline(0.0, color="0.7", lw=0.5)
    ax_w.set_xlabel(r"$x - x_0$")
    ax_w.set_ylabel(r"$w(x)$")
    ax_w.set_title(f"{res['coupling_kernel']}", fontsize=9)
    make_panel_label(ax_w, f"({next(letters)})", x=-0.27, y=1.06)

    # ── Row 2: final coupling matrix A (centred so the bump is in middle) ─────
    ax_A = fig.add_subplot(gs[2, :2])
    A = res["A_oa_final"]
    vmax = max(abs(A).max(), 1e-12)
    im_A = ax_A.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest", aspect="auto",
                       extent=(x[0], x[-1], x[-1], x[0]))
    ax_A.set_xlabel(r"pre $x_n$")
    ax_A.set_ylabel(r"post $x_m$")
    ax_A.set_title(r"$A_{mn}$ (final)", fontsize=10)
    cb_A = fig.colorbar(im_A, ax=ax_A, pad=0.02, fraction=0.046, aspect=18)
    cb_A.set_label(r"$A_{mn}$", fontsize=10)
    cb_A.locator = plt.MaxNLocator(nbins=4)
    cb_A.update_ticks()
    cb_A.ax.tick_params(labelsize=9)
    make_panel_label(ax_A, f"({next(letters)})", x=-0.10, y=1.06)

    # ── Row 2 right: small annotation panel with parameter summary ───────────
    ax_info = fig.add_subplot(gs[2, 2])
    ax_info.axis("off")
    info_lines = [
        rf"$\bar\eta = {res['ring']['eta_bar']:g}$",
        rf"$\Delta = {res['ring']['delta']:g}$",
        rf"$\kappa = {res['kappa']:.4g}$",
        rf"$L = {L:g}$,  $M = {M}$",
        rf"$\delta x = {res['dx']:.3g}$",
        rf"kernel: {res['coupling_kernel']}",
        rf"channel: {res['coupling_channel']}",
        rf"plasticity: {res['plasticity']}",
        rf"$R^* = {res['R_star']:.4f}$",
        rf"$\Psi^* = {res['Psi_star']:+.4f}$",
    ]
    ax_info.text(0.0, 1.0, "\n".join(info_lines),
                 transform=ax_info.transAxes, ha="left", va="top",
                 fontsize=9, family="monospace")

    fig.suptitle(rf"Bump dynamics, single-population ring "
                 rf"($\bar\eta={res['ring']['eta_bar']:g}$, "
                 rf"$\Delta={res['ring']['delta']:g}$, "
                 rf"$\kappa={res['kappa']:.3g}$, kernel: {res['coupling_kernel']})",
                 fontsize=10, y=0.985)

    fig.savefig(savepath, bbox_inches="tight")
    print(f"\nFigure saved → {savepath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Schmidt & Avitabile Fig. 1 parameter set, translated to the theta-neuron
    # formulation (the conformal map V = tan(θ/2) leaves η̄, Δ, κ unchanged;
    # see the QIF ↔ theta translation section in the module docstring).
    #
    # Plasticity is OFF here (μ = γ = 0) so the coupling matrix A stays
    # frozen at the static Mexican-hat kernel — this is exactly the static
    # neural-field setup S&A use to demonstrate the bump in their Fig. 1.
    # Turn μ on to let the kernel evolve under one of the plasticity rules.
    CONFIG = dict(
        # Spatial discretisation
        M=128, L=50.0, T=40.0,
        # Coupling: bi-exponential Mexican hat from Eq. 5 of S&A
        coupling_kernel="schmidt_avitabile",
        kappa=15.0 * np.sqrt(2.0),         # J = 15·√2 ≈ 21.213 in S&A Fig. 1
        # Plasticity: OFF (matrix stays = kernel)
        plasticity="antihebbian", mu=0.001, gamma=0.0,
        n_pulse=2, n1=2, n2=2, n3=3,
        # Membrane timescale (set to 1 in S&A)
        tau=1.0,
        # Homogeneous Lorentzian excitabilities (S&A Fig. 1)
        eta_bar=-10.0, delta=2.0,
        # Bump IC — centre at x = 0 (middle of the L = 50 ring)
        bump_centre=0.0,
        r_bump=2.0, v_bump=0.0, bump_width=2.5,
        # Optional S&A-style transient external current (set amplitude > 0
        # to mimic Fig. 1: I(x,t) ≡ 5 on x ∈ [−2.5, 2.5], t ∈ [0, 5]).
        # Leave at 0 to use only the IC bump.
        stim_amplitude=5.0,
        stim_halfwidth=2.5, stim_t_on=0.0, stim_t_off=5.0,
        # Run controls
        method="RK45", rtol=1e-6, atol=1e-8,
        kernel_kwargs={"k0": 0.2, "k1": 3.0}
    )

    res = simulate(**CONFIG)
    plot_figure(res, savepath="theta_ring_bump.pdf")
    plt.show()