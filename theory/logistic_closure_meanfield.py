r"""
Saturating (logistic) adaptive coupling: moment closures for the weight statistics
==================================================================================

Companion to ``logistic_closure_micro.py``.  The exact two-moment hierarchy of the
logistic rule dA_ij/dt = μ G_ij A_ij (A_m − A_ij) is, with ' ≡ (1/μ) d/dt,
a = A − Ā, C_A = ⟨G a⟩, P = ⟨G a²⟩, Q = ⟨G a³⟩:

    Ā'   =   Ā(A_m−Ā) Ḡ + (A_m−2Ā) C_A −  P
    V_A' = 2 Ā(A_m−Ā) C_A + 2(A_m−2Ā) P  − 2Q          (Ḡ cancels identically)

Two facts constrain any closure for (P, Q):

1. Bhatia–Davis: since A ∈ [0, A_m],  V_A ≤ Ā(A_m−Ā), with equality exactly on the
   two-atom law.  Define the saturation  s = V_A / [Ā(A_m−Ā)] ∈ [0, 1].
2. s = 1 is a manifold of EXACT fixed points (A(A_m−A) = 0 pairwise).  Tangency of
   the exact flow there requires, on s = 1,

       2Q = 3(A_m−2Ā) P − (A_m² − 6A_mĀ + 6Ā²) C_A − Ā(A_m−Ā)(A_m−2Ā) Ḡ

   which the two-atom moments satisfy identically.  Dropping Q destroys this, and
   then V_A' = 2Ā(A_m−Ā) C_A > 0 with nothing able to bound V_A.

Closures implemented
--------------------
``zero``   P = Q = 0.  The naive closure; shown to be inadmissible (V_A diverges).
``gauss``  Latent-drive closure with a GAUSSIAN latent.  For a frozen per-pair drive
           u the rule integrates exactly to A = A_m σ(λu + b), so the weight law is a
           sigmoid transform of the latent.  Matching (Ā, V_A) fixes (λσ_u, λū+b);
           because ⟨G a^k⟩ = ⟨u a^k⟩ and C_A = σ_u⟨w a⟩, the scale σ_u cancels:
               P̂ = Ḡ V_A   + C_A ⟨w a²⟩/⟨w a⟩,
               Q̂ = Ḡ ⟨a³⟩  + C_A ⟨w a³⟩/⟨w a⟩.
``mix``    Latent-drive closure with a LOCKED + DRIFTING latent, i.e. an atom at
           u = u_L > 0 (phase-locked pairs, weight n_L) plus a zero-mean Gaussian
           (drifting pairs, whose pair drive u = Re c_i* c_j is positive for
           same-sign and negative for opposite-sign detunings).  This is the weight-
           level analogue of the S = S_L + S_D locked/drifting split.  In scaled
           variables x = λu + b the four unknowns (α, δ, n_L, ν=1/λ) are fixed by
           the four state variables (Ā, V_A, C_A, Ḡ):
               locked   : x = b + α,        weight n_L
               drifting : x = b + δ w,      weight 1 − n_L,  w ~ N(0,1)
               ⟨u a^k⟩ = ν [ n_L α a_L^k + (1−n_L) δ ⟨w a^k⟩_D ]

Usage (``pycobi`` env)
----------------------
    python logistic_closure_meanfield.py               # all regimes
    python logistic_closure_meanfield.py partial
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq, least_squares
from scipy.spatial import cKDTree
from scipy.special import expit, logit

CONFIG = dict(
    in_dir=dp.KMO_ADAPTIVE,
    out=dp.kmo_adaptive("logistic_closure_scored"),
    closures=("zero", "gauss", "mix", "mix_bd"),
    skip_frac=0.05,                 # ignore the first 5% of the trace when scoring
)

#: Gauss–Hermite nodes/weights for E[f(w)], w ~ N(0,1)
_NODES, _W = np.polynomial.hermite_e.hermegauss(161)
_W = _W / _W.sum()


# ════════════════════════════════════════════════════════════════════════════
#  exact hierarchy
# ════════════════════════════════════════════════════════════════════════════
def hierarchy_rhs(Abar, VA, CA, Gbar, P, Q, Am, mu):
    """(dĀ/dt, dV_A/dt) from the exact two-moment equations."""
    D = Abar * (Am - Abar)
    return (mu * (D * Gbar + (Am - 2 * Abar) * CA - P),
            mu * (2 * D * CA + 2 * (Am - 2 * Abar) * P - 2 * Q))


def boundary_Q(Abar, CA, Gbar, P, Am):
    """Q required for tangency of the exact flow to the Bhatia–Davis boundary s=1."""
    return 0.5 * (3 * (Am - 2 * Abar) * P
                  - (Am ** 2 - 6 * Am * Abar + 6 * Abar ** 2) * CA
                  - Abar * (Am - Abar) * (Am - 2 * Abar) * Gbar)


# ════════════════════════════════════════════════════════════════════════════
#  closure "gauss": sigmoid of a Gaussian latent, 2 parameters
# ════════════════════════════════════════════════════════════════════════════
def _sig_moments(c, d):
    """(m, v, m3, ⟨wa⟩, ⟨wa²⟩, ⟨wa³⟩) for y = σ(c w + d), w ~ N(0,1)."""
    y = expit(c * _NODES + d)
    m = _W @ y
    a = y - m
    return (m, _W @ a ** 2, _W @ a ** 3,
            _W @ (_NODES * a), _W @ (_NODES * a ** 2), _W @ (_NODES * a ** 3))


def _fit_gauss(m_target, s_target):
    """(c, d) reproducing the mean m and the saturation s = v/[m(1-m)]."""
    d0 = logit(m_target)                 # exact at c=0; centre the bracket on it

    def s_of_c(c):
        lo, hi_ = d0 - 4 - 8 * c, d0 + 4 + 8 * c
        d = brentq(lambda dd: _sig_moments(c, dd)[0] - m_target, lo, hi_, xtol=1e-13)
        m, v = _sig_moments(c, d)[:2]
        return v / (m * (1 - m)), d

    hi = 25.0                                        # beyond this σ() is numerically binary
    s_hi, d_hi = s_of_c(hi)
    if s_hi < s_target:                              # target beyond the family's reach
        return hi, d_hi
    s_lo, d_lo = s_of_c(1e-6)
    if s_lo > s_target:
        return 1e-6, d_lo
    c = brentq(lambda cc: s_of_c(cc)[0] - s_target, 1e-6, hi, xtol=1e-10)
    return c, s_of_c(c)[1]


def closure_gauss(Abar, VA, CA, Gbar, Am, **_):
    m = np.clip(Abar / Am, 1e-9, 1 - 1e-9)
    s = np.clip(VA / (Am ** 2 * m * (1 - m)), 1e-9, 1 - 1e-9)
    c, d = _fit_gauss(m, s)
    _, _, m3, wa1, wa2, wa3 = _sig_moments(c, d)
    m3A = m3 * Am ** 3
    if abs(wa1) < 1e-14:
        return Gbar * VA, Gbar * m3A
    return (Gbar * VA + CA * (wa2 / wa1) * Am,
            Gbar * m3A + CA * (wa3 / wa1) * Am ** 2)


# ════════════════════════════════════════════════════════════════════════════
#  closure "mix": locked atom + drifting Gaussian latent, 4 parameters
# ════════════════════════════════════════════════════════════════════════════
def _mix_moments(alpha, delta, nL, b):
    """Scaled weight/latent moments for the locked+drifting latent.

    Returns (m, v, aL, ED_w_ak) with ED_w_ak = [⟨w a⁰⟩, ⟨w a⟩, ⟨w a²⟩, ⟨w a³⟩]_D
    over the drifting component and a = y - m in units of A_m.
    """
    yL = expit(b + alpha)
    yD = expit(b + delta * _NODES)
    m = nL * yL + (1 - nL) * (_W @ yD)
    aL = yL - m
    aD = yD - m
    ED = np.array([_W @ (_NODES * aD ** k) for k in range(4)])
    v = nL * aL ** 2 + (1 - nL) * (_W @ aD ** 2)
    return m, v, aL, ED


#: grid over the latent SHAPE parameters (alpha, delta, n_L); the table is built once
MIX_GRID = dict(n_alpha=44, n_delta=44, n_nL=40, alpha=(0.02, 25.0), delta=(0.01, 25.0),
                nL=(0.002, 0.998), n_neigh=8)


def _mix_table(b, grid=MIX_GRID):
    r"""Forward table over the latent shape, exploiting an exact scaling.

    With nu = 1/lambda and rho = (1-n_L) delta / (n_L alpha),

        Gbar = nu n_L alpha,       <u a^k>/A_m^k = Gbar (a_L^k + rho E_k)

    so the two RATIOS below are independent of Gbar (and of nu):

        f_P = P/(A_m   C_A) = (a_L^2 + rho E_2)/(a_L + rho E_1)
        f_Q = Q/(A_m^2 C_A) = (a_L^3 + rho E_3)/(a_L + rho E_1)

    and the remaining scale-free coordinate is  phi = A_m Gbar / C_A = 1/(a_L + rho E_1).
    The closure is therefore a map (m, s, phi) -> (f_P, f_Q), tabulated here.
    """
    al = np.geomspace(*grid["alpha"], grid["n_alpha"])
    de = np.geomspace(*grid["delta"], grid["n_delta"])
    nl = np.linspace(*grid["nL"], grid["n_nL"])
    A_, D_, L_ = np.meshgrid(al, de, nl, indexing="ij")
    A_, D_, L_ = A_.ravel(), D_.ravel(), L_.ravel()

    yL = expit(b + A_)                                     # (npts,)
    yD = expit(b + D_[:, None] * _NODES[None, :])          # (npts, nq)
    mD = yD @ _W
    m = L_ * yL + (1 - L_) * mD
    aL = yL - m
    aD = yD - m[:, None]
    E = np.stack([(_NODES * aD ** k) @ _W for k in range(4)], axis=1)   # (npts, 4)
    v = L_ * aL ** 2 + (1 - L_) * (aD ** 2 @ _W)
    rho = (1 - L_) * D_ / (L_ * A_)

    cov = aL + rho * E[:, 1]                               # C_A/(A_m Gbar)
    s = v / (m * (1 - m))
    ok = (cov > 1e-6) & (s > 1e-6) & (s < 1 - 1e-6) & (m > 1e-4) & (m < 1 - 1e-4)
    phi = 1.0 / cov[ok]
    pts = np.column_stack([m[ok], s[ok], np.tanh(phi)])     # tanh squashes phi to (0,1)
    vals = np.column_stack([(aL[ok] ** 2 + rho[ok] * E[ok, 2]) / cov[ok],
                            (aL[ok] ** 3 + rho[ok] * E[ok, 3]) / cov[ok]])
    return cKDTree(pts), vals


_MIX_CACHE = {}


def closure_mix(Abar, VA, CA, Gbar, Am, b=0.0, **_):
    """Locked+drifting latent closure via the tabulated (m, s, phi) -> (f_P, f_Q) map."""
    if b not in _MIX_CACHE:
        _MIX_CACHE[b] = _mix_table(b)
    tree, vals = _MIX_CACHE[b]
    if abs(CA) < 1e-12:                     # no weight/drive covariance => no source
        return 0.0, 0.0
    m = np.clip(Abar / Am, 1e-4, 1 - 1e-4)
    s = np.clip(VA / (Am ** 2 * m * (1 - m)), 1e-6, 1 - 1e-6)
    phi = Am * Gbar / CA
    q = np.array([m, s, np.tanh(phi)])
    dist, idx = tree.query(q, k=MIX_GRID["n_neigh"])
    w = 1.0 / np.maximum(dist, 1e-9)
    w /= w.sum()
    fP, fQ = w @ vals[idx]
    return Am * CA * fP, Am ** 2 * CA * fQ


#: s above which Q is blended towards its boundary-tangency value
BLEND_S0 = 0.90


def admissible(fn, s0=BLEND_S0):
    r"""Wrap a closure so the Bhatia–Davis boundary s=1 is invariant by construction.

    Near s = 1 the exact flow is tangential to V_A = Ā(A_m−Ā) only if Q takes the
    value ``boundary_Q``.  Blending Q towards it over s ∈ [s0, 1] therefore makes
    s = 1 an invariant manifold of the CLOSED system too, so V_A can never leave
    the physically attainable region.
    """
    def wrapped(Abar, VA, CA, Gbar, Am, **kw):
        P, Q = fn(Abar, VA, CA, Gbar, Am, **kw)
        s = VA / (Abar * (Am - Abar))
        w = np.clip((s - s0) / (1.0 - s0), 0.0, 1.0)
        if w > 0:
            Q = (1 - w) * Q + w * boundary_Q(Abar, CA, Gbar, P, Am)
        return P, Q
    return wrapped


CLOSURES = dict(
    zero=lambda *a, **k: (0.0, 0.0),
    gauss=closure_gauss,
    mix=closure_mix,
    mix_bd=admissible(closure_mix),
)


# ════════════════════════════════════════════════════════════════════════════
#  scoring: integrate (Ā, V_A) driven by the microscopic Ḡ(t) and C_A(t)
# ════════════════════════════════════════════════════════════════════════════
def score_regime(d, closures=CONFIG["closures"], skip_frac=CONFIG["skip_frac"]):
    t, R, Ab, VA, CA, P, Q = (d[k] for k in "t R Abar VA CA P Q".split())
    Gb = d["g1"]
    Am, mu, b = float(d["Am"]), float(d["mu"]), float(logit(float(d["A0"]) / float(d["Am"])))
    Gb_f = interp1d(t, Gb, bounds_error=False, fill_value=(Gb[0], Gb[-1]))
    CA_f = interp1d(t, CA, bounds_error=False, fill_value=(CA[0], CA[-1]))
    m = t > t[-1] * skip_frac

    out = {"t": t, "R": R, "Abar": Ab, "VA": VA, "CA": CA, "P": P, "Q": Q, "Gbar": Gb,
           "bound": Ab * (Am - Ab), "sat": VA / (Ab * (Am - Ab))}

    # --- pointwise closure accuracy for P and Q
    for name in closures:
        fn, Ph, Qh = CLOSURES[name], np.zeros_like(P), np.zeros_like(Q)
        for i in range(len(t)):
            if VA[i] <= 1e-12 or not (0 < Ab[i] < Am):
                continue
            Ph[i], Qh[i] = fn(Ab[i], VA[i], CA[i], Gb[i], Am, b=b, key=f"pt{name}")
        out[f"P_{name}"], out[f"Q_{name}"] = Ph, Qh
        rel = lambda h, x: np.sqrt(np.mean((h - x)[m] ** 2)) / np.sqrt(np.mean(x[m] ** 2))
        out[f"errP_{name}"], out[f"errQ_{name}"] = rel(Ph, P), rel(Qh, Q)

    # --- integrate the hierarchy with each closure
    for name in closures:
        fn = CLOSURES[name]

        def rhs(tt, y):
            A_, V_ = y
            A_ = min(max(A_, 1e-6), Am - 1e-6)
            V_ = min(max(V_, 1e-12), 0.999 * A_ * (Am - A_))
            g, c = float(Gb_f(tt)), float(CA_f(tt))
            Ph, Qh = fn(A_, V_, c, g, Am, b=b, key=f"ode{name}")
            return hierarchy_rhs(A_, V_, c, g, Ph, Qh, Am, mu)

        sol = solve_ivp(rhs, (t[0], t[-1]), [Ab[0], max(VA[0], 1e-10)], t_eval=t,
                        method="LSODA", rtol=1e-7, atol=1e-11)
        A_, V_ = (sol.y if sol.y.shape[1] == len(t) else
                  np.full((2, len(t)), np.nan))
        out[f"Abar_{name}"], out[f"VA_{name}"] = A_, V_
        out[f"errAbar_{name}"] = np.sqrt(np.nanmean((A_ - Ab)[m] ** 2)) / np.sqrt(np.mean(Ab[m] ** 2))
        out[f"errVA_{name}"] = np.sqrt(np.nanmean((V_ - VA)[m] ** 2)) / np.sqrt(np.mean(VA[m] ** 2))
    return out


# ════════════════════════════════════════════════════════════════════════════
#  latent diagnostics: is A really a sigmoid of the latent drive?
# ════════════════════════════════════════════════════════════════════════════
def latent_fit(u, A, Am, b):
    """Least-squares (lambda, b_eff) for A = A_m σ(λ u + b_eff); returns also R²."""
    def resid(th):
        return Am * expit(th[0] * u + th[1]) - A
    r = least_squares(resid, [10.0, b], xtol=1e-12, ftol=1e-12)
    ss = np.sum((A - A.mean()) ** 2)
    return r.x[0], r.x[1], 1.0 - np.sum(r.fun ** 2) / ss


def latent_mixture_fit(u):
    """Fit u ~ n_L δ(u−u_L) + (1−n_L) N(0, σ_D²) by moment matching on (mean, var, skew)."""
    m1, m2, m3 = u.mean(), np.mean(u ** 2), np.mean(u ** 3)
    def resid(th):
        nL, uL, sD = th
        return [nL * uL - m1,
                nL * uL ** 2 + (1 - nL) * sD ** 2 - m2,
                nL * uL ** 3 - m3]
    lo = np.array([1e-4, -2.0, 1e-4])
    hi = np.array([1 - 1e-4, 2.0, 5.0])
    x0 = np.clip([0.3, m1 / 0.3, np.sqrt(max(m2, 1e-6))], lo + 1e-6, hi - 1e-6)
    r = least_squares(resid, x0, bounds=(lo, hi))
    return dict(zip(("nL", "uL", "sD"), r.x))


# ════════════════════════════════════════════════════════════════════════════
#  entry point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import glob
    # discover regimes from the micro output (importing the micro script would pull in numba)
    found = sorted(os.path.basename(f)[len("logistic_closure_"):-len(".npz")]
                   for f in glob.glob(os.path.join(CONFIG["in_dir"], "logistic_closure_*.npz"))
                   if "scored" not in f)
    which = sys.argv[1:] or found
    if not which:
        raise SystemExit(f"no micro output in {CONFIG['in_dir']} — "
                         "run logistic_closure_micro.py first (allen env)")
    res, lat = {}, {}
    for name in which:
        f = os.path.join(CONFIG["in_dir"], f"logistic_closure_{name}.npz")
        if not os.path.exists(f):
            print(f"[{name}] missing {f} — run logistic_closure_micro.py first"); continue
        d = np.load(f)
        Am, b = float(d["Am"]), logit(float(d["A0"]) / float(d["Am"]))
        r = score_regime(d)
        res[name] = r

        # latent diagnostics on the last snapshot (off-diagonal entries only)
        u, A = d["u_snap"][-1], d["A_snap"][-1]
        n = u.shape[0]
        off = ~np.eye(n, dtype=bool)
        uu, AA = u[off], A[off]
        lam, beff, r2 = latent_fit(uu, AA, Am, b)
        mixp = latent_mixture_fit(uu)
        lat[name] = dict(u=uu, A=AA, lam=lam, b_eff=beff, r2=r2, **mixp)

        print(f"\n=== {name}: R={r['R'][-1]:.3f}  Ā={r['Abar'][-1]:.3f}  "
              f"V_A={r['VA'][-1]:.4f}  s={r['sat'][-1]:.3f}")
        print(f"    latent: A = A_m σ({lam:.1f} u + {beff:+.2f})  ->  R² = {r2:.3f}"
              f"   |  mixture n_L={mixp['nL']:.3f} u_L={mixp['uL']:.3f} σ_D={mixp['sD']:.3f}")
        print(f"    {'closure':<8} {'relRMSE P':>10} {'relRMSE Q':>10} "
              f"{'err Ābar':>10} {'err V_A':>10} {'V_A(T)':>9}")
        for c in CONFIG["closures"]:
            print(f"    {c:<8} {r[f'errP_{c}']:>10.3f} {r[f'errQ_{c}']:>10.3f} "
                  f"{r[f'errAbar_{c}']:>10.3f} {r[f'errVA_{c}']:>10.3f} "
                  f"{r[f'VA_{c}'][-1]:>9.4f}")
        print(f"    {'(truth)':<8} {'-':>10} {'-':>10} {'-':>10} {'-':>10} "
              f"{r['VA'][-1]:>9.4f}")

    flat = {}
    for k, v in res.items():
        flat.update({f"{k}/{kk}": vv for kk, vv in v.items()})
    for k, v in lat.items():
        flat.update({f"lat_{k}/{kk}": vv for kk, vv in v.items()})
    np.savez_compressed(CONFIG["out"] + ".npz", regimes=np.array(list(res)), **flat)
    print(f"\n[saved] {CONFIG['out']}.npz")
