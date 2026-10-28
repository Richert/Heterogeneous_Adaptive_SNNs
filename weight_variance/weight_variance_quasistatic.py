r"""
Quasi-static (structure-aware) self-consistency for the symmetric adaptation rule G = cos
==========================================================================================

The mean-field equations replace every coupling weight by its average Ā = 1 + κR² (κ = μ/γ), so that
each oscillator feels the field K Ā R. With slaved weights A_ij = 1 + κ Γ_ij, Γ_ij = ⟨cos(θ_j − θ_i)⟩_t
= Re(c_j c_i*), c_j = ⟨e^{iθ_j}⟩_t (Ψ = 0, Ω = 0 by symmetry), oscillator i instead feels

    F_i = ⟨A_ij c_j⟩_j = R + (κ/2)(W c_i* + S c_i) = R + κ (P x_i + i Q y_i),   c_i = x_i + i y_i,

with S = ⟨|c|²⟩, W = ⟨c²⟩ (= R² on the OA manifold), P = (S + W)/2, Q = (S − W)/2. Hence
  * locked oscillators (c = e^{iφ}):  ω = K R sin φ + (K κ W / 2) sin 2φ   (MF: ω = K Ā R sin φ),
    locked for |ω| < b_eff = max_φ of the right-hand side on the stable branch;
  * drifting oscillators feel the complex field K(R + iκQ y_i) and have c = e^{iα} i q(ω; ρ).
The weight structure enters only through Cov_j(Γ_ij, c_j) = F_i − Ā R, i.e. it is the feedback of the
weight (co)variance onto the phase dynamics that the mean field omits. Self-consistency ⟨c⟩ = R.

The static weight variance follows from the same state, σ_Γ² = ½(S² + W²) − R⁴ (closure: W = R²),
and V_A = κ² σ_Γ² + μ² σ_g² / (γ(γ + 2Δ)), σ_g² = ½(1 − S²) as in the closure.
"""
import numpy as np
from scipy.optimize import brentq


def _drift_q(w, rho):
    return (w - np.sign(w) * np.sqrt(w ** 2 - rho ** 2)) / rho


def locked_phases(omega, R, kW, K, n_grid=4000):
    """Stable locked phases of ω = K R sin φ + (K κW/2) sin 2φ (kW = κW). Returns (φ, b_eff); φ = nan
    outside the locking band. The stable branch is the monotonic rise from φ = 0 to the first maximum."""
    ph = np.linspace(0.0, np.pi, n_grid)
    rhs = K * R * np.sin(ph) + 0.5 * K * kW * np.sin(2.0 * ph)
    imax = np.argmax(np.diff(rhs) <= 0) if np.any(np.diff(rhs) <= 0) else n_grid - 1
    ph, rhs = ph[:imax + 1], rhs[:imax + 1]
    b_eff = rhs[-1]
    phi = np.full(omega.size, np.nan)
    lk = np.abs(omega) < b_eff
    phi[lk] = np.sign(omega[lk]) * np.interp(np.abs(omega[lk]), rhs, ph)
    return phi, b_eff


def state_RW(omega, R, W, kap, K, n_outer=100, n_drift=40, tol=1e-12):
    """Moments c_i for given (R, W), with S iterated to self-consistency. Returns the state and the
    two residuals (⟨Re c⟩ − R, Re⟨c²⟩ − W) of the full self-consistency problem."""
    phi, b_eff = locked_phases(omega, R, kap * W, K)
    lk = np.isfinite(phi)
    wd = omega[~lk]
    S, x, y = 1.0, np.zeros(wd.size), np.zeros(wd.size)
    for _ in range(n_outer):
        P, Q = 0.5 * (S + W), 0.5 * (S - W)
        for _ in range(n_drift):
            Fd = K * (R + kap * (P * x + 1j * Q * y))
            rho = np.minimum(np.abs(Fd), np.abs(wd) * (1 - 1e-12))
            cd = np.exp(1j * np.angle(Fd)) * 1j * _drift_q(wd, rho)
            x, y = 0.5 * x + 0.5 * cd.real, 0.5 * y + 0.5 * cd.imag
        S_new = (lk.sum() + np.sum(x ** 2 + y ** 2)) / omega.size
        if abs(S_new - S) < tol:
            S = S_new
            break
        S = S_new
    c = np.empty(omega.size, complex)
    c[lk] = np.exp(1j * phi[lk])
    c[~lk] = x + 1j * y
    return dict(R=R, S=S, W=W, c=c, b_eff=b_eff, locked=lk,
                res=np.array([c.real.mean() - R, np.mean(c ** 2).real - W]))


def solve_RW(omega, kap, K, R0, W0=None):
    """Full self-consistent state (R, W, S) by a root find from (R0, W0); None if not converged or R ≤ 0."""
    from scipy.optimize import root
    W0 = R0 ** 2 if W0 is None else W0
    sol = root(lambda v: state_RW(omega, v[0], v[1], kap, K)["res"], [R0, W0], method="hybr",
               options=dict(xtol=1e-10))
    if not sol.success or sol.x[0] <= 1e-4 or np.max(np.abs(sol.fun)) > 1e-7:
        return None
    return state_RW(omega, sol.x[0], sol.x[1], kap, K)


def iterate_RW(omega, kap, K, R0, W0, n_it=300, n_avg=50, damp=0.5, R_min=0.02, R_tol=2e-3):
    """Damped fixed-point iteration (R, W) ← (⟨Re c⟩, Re⟨c²⟩) from (R0, W0); follows STABLE states.
    Edge oscillators may keep switching on a finite frequency set, so (R, W) are averaged over the last
    `n_avg` iterations. Returns None if R collapses below `R_min` or does not settle (spread > R_tol)."""
    R, W, hist = R0, W0, []
    for it in range(n_it):
        st = state_RW(omega, R, W, kap, K)
        R = (1 - damp) * R + damp * (st["res"][0] + R)
        W = (1 - damp) * W + damp * (st["res"][1] + W)
        if R < R_min:
            return None
        if it >= n_it - n_avg:
            hist.append((R, W))
    hist = np.array(hist)
    if hist[:, 0].std() > R_tol:
        return None
    return state_RW(omega, *hist.mean(axis=0), kap, K)


def sync_branch(omega_of, deltas, kap, K):
    """Continuation of the synchronized quasi-static branch along increasing Δ (root find, fixed-point
    iteration as fallback); entries after the branch is lost (fold) are None."""
    out, R0, W0, lost = [], 0.99, 0.98, False
    for D in deltas:
        s = None
        if not lost:
            om = omega_of(D)
            s = solve_RW(om, kap, K, R0, W0) or iterate_RW(om, kap, K, R0, W0)
            if s is None:
                lost = True
            else:
                R0, W0 = s["R"], s["W"]
        out.append(s)
    return out


def state_at_R(omega, R, kap, K, n_outer=200, n_drift=40, tol=1e-11):
    """Self-consistent (S, W) and moments c_i at FIXED R. Returns dict incl. residual ⟨Re c⟩ − R."""
    S, W = 1.0, R ** 2
    c = np.zeros(omega.size, complex)
    for _ in range(n_outer):
        P, Q = 0.5 * (S + W), 0.5 * (S - W)
        phi, b_eff = locked_phases(omega, R, kap * W, K)
        lk = np.isfinite(phi)
        c_new = np.empty_like(c)
        c_new[lk] = np.exp(1j * phi[lk])
        wd = omega[~lk]
        y = c[~lk].imag if np.any(c[~lk] != 0) else 0.0 * wd
        x = c[~lk].real
        for _ in range(n_drift):                               # per-drifter field fixed point
            Fd = K * (R + kap * (P * x + 1j * Q * y))
            rho, al = np.abs(Fd), np.angle(Fd)
            rho = np.minimum(rho, np.abs(wd) * (1 - 1e-12))     # keep the drifter drifting
            cd = np.exp(1j * al) * 1j * _drift_q(wd, rho)
            x, y = 0.5 * x + 0.5 * cd.real, 0.5 * y + 0.5 * cd.imag
        c_new[~lk] = x + 1j * y
        S_new, W_new = np.mean(np.abs(c_new) ** 2), np.mean(c_new ** 2).real
        dS = abs(S_new - S) + abs(W_new - W)
        S, W, c = 0.5 * S + 0.5 * S_new, 0.5 * W + 0.5 * W_new, c_new
        if dS < tol:
            break
    return dict(R=R, S=S, W=W, c=c, b_eff=b_eff, locked=np.isfinite(phi), res=c.real.mean() - R)


def branches_R(omega, kap, K, R_grid=None):
    """All roots R* of ⟨Re c⟩(R) = R on R ∈ (0,1) (sync and saddle branches), via a scan + brentq.
    Each root is returned with its state and the sign of d(res)/dR (< 0: stable in R)."""
    if R_grid is None:
        R_grid = np.linspace(0.01, 0.995, 100)
    res = np.array([state_at_R(omega, R, kap, K)["res"] for R in R_grid])
    roots = []
    for k in np.nonzero(np.sign(res[:-1]) != np.sign(res[1:]))[0]:
        Rr = brentq(lambda R: state_at_R(omega, R, kap, K)["res"], R_grid[k], R_grid[k + 1], xtol=1e-7)
        st = state_at_R(omega, Rr, kap, K)
        st["stable"] = res[k + 1] < res[k]
        roots.append(st)
    return roots


def weight_stats(st, kap, mu, gamma, Delta):
    """(Ā, V_A, V_A/Ā², σ_Γ², σ_g²) of a quasi-static state."""
    R, S, W = st["R"], st["S"], st["W"]
    A = 1.0 + kap * R ** 2
    sG = 0.5 * (S ** 2 + W ** 2) - R ** 4
    sg = 0.5 * (1.0 - S ** 2)
    VA = kap ** 2 * sG + mu ** 2 * sg / (gamma * (gamma + 2.0 * Delta))
    return dict(Abar=A, VA=VA, VArel=VA / A ** 2, sG=sG, sg=sg)
