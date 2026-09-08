r"""
Pulse kernels for pulse-driven weight adaptation, and their Daido-moment algebra
================================================================================

Pulse function (the repo's convention, cf. ``kuramoto/theta_pulsed_adaptation.py``):

    s(n, θ) = c_n (1 − cos θ)^n,        c_n = 2^n (n!)² / (2n)!

which is NON-NEGATIVE — the property that confines the adaptation rule

    dA_ij/dt = μ_p G_p(θ_i,θ_j) (A_m − A_ij) − μ_d G_d(θ_i,θ_j) A_ij

to A_ij ∈ [0, A_m].  Kernels are taken separable,

    G(θ_i, θ_j) = s(n_post, θ_i) · s(n_pre, θ_j),

specified as the pair ``(n_post, n_pre)``; ``n = 0`` drops that factor since
s(0,·) ≡ 1.  The default ``duchet`` choice is coincidence-driven potentiation with
purely PRE-synaptic depression (Duchet et al., Neural Computation 2023).

Why the moments close
---------------------
Two exact identities make every drive moment a finite polynomial in the Daido
order parameters z_q = ⟨exp(i q θ)⟩:

1. products stay in the family:   s(m,θ) s(m',θ) = κ(m,m') s(m+m', θ),
                                  κ(m,m') = c_m c_m' / c_{m+m'}
2. each pulse average is a finite harmonic sum:
       ⟨s(m,θ)⟩ = c_m Σ_{q=-m}^{m} C_q^{(m)} z_q
                = c_m ( C_0^{(m)} + 2 Σ_{q≥1} C_q^{(m)} Re z_q )
   with C_q^{(m)} the harmonic coefficients of (1 − cos θ)^m.

Hence, for two separable kernels and independent θ_i, θ_j (exact in the
thermodynamic limit, and exact even with heterogeneity — no OA ansatz needed),

    ⟨G_1 G_2⟩ = [κ(a₁,a₂) M(a₁+a₂)] · [κ(b₁,b₂) M(b₁+b₂)],    M(m) ≡ ⟨s(m,θ)⟩

so Ḡ_p, Ḡ_d, Var(G_p), Var(G_d), Cov(G_p,G_d) all follow from M(m) alone.  Only if
one additionally reduces z_q → Z^q (OA) or Σ_m w_m Z_m^q (LMMF) does an ansatz enter.
"""

import numpy as np
from math import factorial as fac

# ════════════════════════════════════════════════════════════════════════════
#  pulse function and its harmonic coefficients
# ════════════════════════════════════════════════════════════════════════════
_C_CACHE, _CQ_CACHE = {}, {}


def c_n(n):
    """Normalisation making ⟨s(n,θ)⟩ = 1 for a uniform phase distribution."""
    n = int(n)
    if n not in _C_CACHE:
        _C_CACHE[n] = 2.0 ** n * fac(n) ** 2 / fac(2 * n)
    return _C_CACHE[n]


def s_pulse(n, theta):
    """s(n, θ) = c_n (1 − cos θ)^n; s(0,·) ≡ 1."""
    if n == 0:
        return np.ones_like(np.asarray(theta, dtype=float))
    return c_n(n) * (1.0 - np.cos(theta)) ** n


def C_q(q, n):
    """Harmonic coefficient of exp(i q θ) in (1 − cos θ)^n (real, symmetric in q)."""
    key = (int(abs(q)), int(n))
    if key in _CQ_CACHE:
        return _CQ_CACHE[key]
    q, tot = abs(int(q)), 0.0
    for i in range(n + 1):
        for j in range(i + 1):
            if i - 2 * j == q:
                tot += fac(n) * (-1) ** i / (2 ** i * fac(n - i) * fac(j) * fac(i - j))
    _CQ_CACHE[key] = tot
    return tot


def kappa(m, mp):
    """s(m,·) s(m',·) = kappa(m,m') s(m+m',·)."""
    return c_n(m) * c_n(mp) / c_n(m + mp)


# ════════════════════════════════════════════════════════════════════════════
#  kernel registry:  (n_post, n_pre) for the potentiating and depressing terms
# ════════════════════════════════════════════════════════════════════════════
KERNELS = dict(
    #: coincidence potentiation, purely PRE-synaptic depression (Duchet-style)
    duchet=dict(p=(2, 2), d=(0, 2)),
    #: same, sharper pulses (larger n = stronger per-pair modulation)
    duchet_sharp=dict(p=(4, 4), d=(0, 4)),
    #: coincidence potentiation AND coincidence depression
    symmetric=dict(p=(2, 2), d=(2, 2)),
    #: coincidence potentiation, purely POST-synaptic depression
    postsyn=dict(p=(2, 2), d=(2, 0)),
)


def max_harmonic(kernel):
    """Highest Daido order q needed by the second-order drive moments."""
    (ap, bp), (ad, bd) = kernel["p"], kernel["d"]
    return max(2 * ap, 2 * bp, 2 * ad, 2 * bd, ap + ad, bp + bd)


# ════════════════════════════════════════════════════════════════════════════
#  Daido-moment reduction of the drive statistics
# ════════════════════════════════════════════════════════════════════════════
def M_pulse(m, z):
    """⟨s(m,θ)⟩ from the Daido order parameters. ``z[q]`` = z_q for q ≥ 0 (z[0]=1)."""
    m = int(m)
    if m == 0:
        return 1.0
    tot = C_q(0, m)
    for q in range(1, m + 1):
        tot += 2.0 * C_q(q, m) * np.real(z[q])
    return c_n(m) * tot


def _pair(spec1, spec2, z):
    """⟨G_1 G_2⟩ for separable kernels with independent θ_i, θ_j."""
    a1, b1 = spec1
    a2, b2 = spec2
    return (kappa(a1, a2) * M_pulse(a1 + a2, z)) * (kappa(b1, b2) * M_pulse(b1 + b2, z))


def drive_moments(kernel, z):
    """(Ḡ_p, Ḡ_d, Var G_p, Var G_d, Cov(G_p,G_d)) from the Daido order parameters.

    ``z`` is indexable by q = 0 .. max_harmonic(kernel) with z[0] = 1.
    """
    sp_, sd = kernel["p"], kernel["d"]
    zero = (0, 0)
    Gp = _pair(sp_, zero, z)
    Gd = _pair(sd, zero, z)
    return (Gp, Gd,
            _pair(sp_, sp_, z) - Gp ** 2,
            _pair(sd, sd, z) - Gd ** 2,
            _pair(sp_, sd, z) - Gp * Gd)


def daido_from_phases(theta, qmax):
    """Empirical Daido order parameters z_q, q = 0..qmax, from a phase sample."""
    z = np.empty(qmax + 1, dtype=complex)
    z[0] = 1.0
    for q in range(1, qmax + 1):
        z[q] = np.mean(np.exp(1j * q * np.asarray(theta)))
    return z


def daido_oa(Z, qmax):
    """Daido parameters under the OA ansatz, z_q = Z^q (LMMF: pass Σ_m w_m Z_m^q)."""
    return np.array([Z ** q for q in range(qmax + 1)], dtype=complex)
