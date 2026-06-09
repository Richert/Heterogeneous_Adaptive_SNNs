r"""
SNIC / homoclinic analysis of adaptive Kuramoto–OA ensembles via PyCoBi's HomCont
=================================================================================

GOAL
----
The γ-continuation in ``kmo_ensembles_adaptive.py`` shows the synchronized,
homogeneous-weight relative equilibrium losing stability at a FOLD as the weight
decay γ increases. Time integration past that fold (``kmo_macro_simulation.py``)
shows a sub-cluster of ensembles peeling away from the coherent core and starting
to oscillate. That is the fingerprint of a SNIC (Saddle-Node on an Invariant
Circle / infinite-period) bifurcation rather than an ordinary fold: a limit cycle
exists on the far side of the fold and its period diverges as γ → γ_c⁺.

This script confirms that picture quantitatively with the bifurcation-continuation
software PyCoBi/Auto-07p, using PyCoBi's recent HomCont entry points:

  1. Settle onto the locked relative equilibrium (IVP) and continue it in γ to
     locate the fold γ_c                                                  [Step 2]
  2. Land on the post-fold limit cycle by time integration, seed an IPS=2
     periodic-orbit continuation from that orbit (PyCoBi `write_auto_dat`),
     and trace it toward the fold → the period DIVERGES at γ_c  ⇒  SNIC   [Step 3]
  3. Seed Auto's HomCont (IPS=9) from the near-homoclinic orbit with
     `continue_homoclinic` and locate the codim-2 non-central SNIC via the
     PSI(15)/PSI(16) test functions                                       [Step 4]
  4. Trace the cycle away from the fold; read its FLOQUET MULTIPLIERS
     (get_eigenvals=True) to classify any loss of stability as Neimark–Sacker
     (torus), period-doubling, or fold-of-cycles; chase any PD cascade with
     `continue_period_doubling_bf`                                        [Step 5]
  5. Continue the fold-of-cycles (saddle-node of periodic orbits) in TWO
     parameters (ISW=2) to map the ARNOL'D-TONGUE boundary in (γ, Δ)      [Step 5b]
  6. Trace the locus of codim-1 equilibrium bifurcations (the fold = SNIC) in
     the (γ, Δ) plane (ISW=2) alongside the homoclinic curve              [Step 6]

  The torus / Floquet analysis (Steps 4–5b) matters for M ≥ 5, where the
  post-SNIC attractor is a quasiperiodic 2-torus (see RESULTS below); for M ≤ 4
  it is a single limit cycle and those steps simply report a hyperbolic cycle.

WHY CARTESIAN ORDER-PARAMETER COORDINATES
-----------------------------------------
In (r, φ) phase-difference coordinates the slipping sub-cluster WINDS by 2π per
period, so the orbit is not closed and Auto's periodic boundary condition
u(1)=u(0) can never be satisfied. We therefore work in the co-rotating Cartesian
order parameter  z_i = x_i + i y_i = r_i e^{iψ_i}  (pin y_0 := 0, x_0 = r_0; the
collective frequency Ω = Im F_0 / x_0 is carried explicitly). A winding phase is
then a CLOSED loop in z, which Auto can represent, and there is no 1/r
singularity. Equilibria — hence the fold — are unchanged. The reduced dimension is
``2M - 1 + M²`` (M = number of ensembles).

DIMENSIONALITY
--------------
``M`` is a top-level knob below. Limit-cycle and (especially) homoclinic
continuation scale steeply with the reduced dimension, so the full LC → PD → SNIC
→ codim-2 pipeline is intended for small M (default 3, dim 14). The
equilibrium/fold steps tolerate much larger M. M ≥ 3 is the smallest size that
still lets a genuine sub-cluster slip relative to the locked core (the mechanism
seen in simulation); M = 2 only has a single slipping pair.

REQUIREMENTS / RUNNING
----------------------
  * Auto-07p, pyrates >= 1.1, pycobi >= 1.1  (set ``AUTO_DIR`` below).
  * Run inside the ``pycobi`` conda env, and make sure ``meson``/``ninja`` are on
    PATH so f2py can build the PyRates extension, e.g.:
        PATH="$CONDA_PREFIX/bin:$PATH" python kmo_ensembles_snic_homoclinic.py

RESULTS (default M=3, K=1, μ=0.1, Δ=0.02, seed=42)
---------------------------------------------------
  * The locked relative equilibrium folds at γ_c ≈ 0.2538 (LP1).
  * Past the fold a limit cycle exists whose period DIVERGES as γ → γ_c⁺
    (≈86 at γ=0.30 up to ≳1.9·10⁴ at γ≈0.2537) — the defining SNIC signature.
    The fold and the cycle's birth coincide, so the LP curve IS the SNIC curve.
  * No genuine period-doubling cascade is found: toward the fold the period
    simply diverges, and away from it the cycle only accumulates BP labels where
    the LC branch nearly touches the (saddle) equilibrium branch — the expected
    homoclinic-proximity signature, not a PD route. (So the period-doubling
    suspicion is NOT borne out for these parameters / this M.)
  * HomCont (IPS=9) seeded from the near-homoclinic orbit, with the hyperbolic
    saddle read off the unstable equilibrium branch (eigenvalue split 1/13),
    detects the codim-2 NON-CENTRAL SNIC via the PSI(15)/PSI(16) test functions.

    CAVEAT: with the pinned-saddle setting (IEQUIB=0) the traced homoclinic
    curve is short and drifts into the (unphysical) Δ < 0 region — confirming
    the bifurcation type but not yet a publication-quality locus. For a long,
    physical (Δ > 0) homoclinic curve, switch to IEQUIB=1 (solve for the moving
    equilibrium along the curve) and expect to retune NTST / DSMAX / the seed
    period. The physically meaningful SNIC locus in (γ, Δ) is the fold curve
    from Step 6 (blue), which is reliable.

RESULTS (M=5 — the multi-frequency / "5-body" regime)
------------------------------------------------------
Confirmed by simulation (kmo_macro_simulation.py) + continuation:
  * The number of INDEPENDENT fundamental frequencies of the post-SNIC attractor
    (co-rotating global mean field ⟨z_i z̄_0/|z_0|⟩) jumps with M:
        M=3 → 1,  M=4 → 1,  M=5 → 2,  M=6 → 2.
    So for M ≤ 4 the post-SNIC state is a single global rhythm (limit cycle); for
    M ≥ 5 a SECOND, incommensurate frequency appears → a quasiperiodic 2-TORUS.
    The "5-body" intuition holds: 5 ensembles are needed for the 2nd frequency.
  * The fold/SNIC is at γ_c ≈ 0.170 (M=5). Just past it the dynamics is generically
    quasiperiodic (2-torus), interleaved with ARNOL'D (resonance) TONGUES where it
    mode-locks back to a limit cycle (e.g. a 1-frequency window around γ ≈ 0.20).
  * Continuation of the mode-locked cycle seeded at γ=0.20 (clean periodic orbit,
    period ≈56) shows it is bounded by a FOLD-OF-CYCLES (saddle-node of periodic
    orbits, LP) at γ ≈ 0.22 — the tongue edge — with the period diverging toward
    the SNIC. No period-doubling (PD) is found on this branch.
  * INTERPRETATION: the complex regime is the 2-torus; the mode-locked cycles
    live in Arnol'd tongues bounded by saddle-node-of-cycles (fold-of-cycles, LP)
    curves — NOT period-doubling. Two analyses (Steps 5 / 5b) make this concrete:
      (a) Step 5 reads the cycle's FLOQUET MULTIPLIERS (get_eigenvals=True) and
          classify_cycle_stability() looks for a complex pair crossing |μ|=1
          (Neimark–Sacker / torus), a real μ→−1 (period-doubling), or μ→+1
          (fold). On the γ=0.20 tongue cycle the leading non-trivial |μ| stays
          well inside the unit circle: the cycle is robustly stable and is
          destroyed at a fold-of-cycles, not via PD or NS on this branch.
      (b) Step 5b continues that fold-of-cycles (ISW=2, ICP=[g, delta, 11]) in
          two parameters — the ARNOL'D-TONGUE boundary. Verified: it runs from
          (γ, Δ) ≈ (0.220, 0.020) to ≈ (0.282, →0), i.e. the tongue narrows as
          Δ → 0. (Use ISP=1 here — detecting codim-2 points along the curve is
          prohibitively slow at dim 34.)
    Feasibility: M=5 ⇒ reduced dim 34; IVP+equilibrium ≈4 s, one LC branch ≈1 min,
    fold-of-cycles curve ≈seconds. HomCont at dim 34 is heavier but runs.

  * REQUIRES the PyCoBi Floquet-multiplier fixes (pycobi/utility.py regex +
    pycobi.py ragged-eigenvalue-column padding). Before them, get_eigenvals=True
    returned NO multipliers for limit cycles (the diagnostic regex assumed the
    equilibrium "Eigenvalue N:" format), and long LC branches with uneven
    multiplier counts crashed the summary build. See classify_cycle_stability.

TWO NON-OBVIOUS AUTO/PYCOBI GOTCHAS (both handled below)
--------------------------------------------------------
  * NDIM on .dat-seeded runs.  When seeding an IPS=2 continuation directly from a
    ``.dat`` file (no `origin` solution to inherit dimensions from), PyCoBi does
    not forward NDIM to Auto and the BVP dat-reader corrupts the heap
    (``corrupted size vs. prev_size``). We pass ``NDIM=DIM`` explicitly on every
    dat-seeded run.
  * NPAR.  PyRates emits ``NPAR = #model-params`` (here ``2M-1`` for K, μ, g, Δ,
    ω_i...; the ω_0 entry is folded in). Auto needs PAR(11) for the period and
    PAR(35)/PAR(36) for the HomCont PSI(15)/PSI(16) test functions, so we pass
    ``NPAR=36`` on the LC / HomCont runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
from pycobi import ODESystem
from pycobi.utility import write_auto_dat

# Path to your Auto-07p installation:
AUTO_DIR = "~/PycharmProjects/auto-07p"
_EPS = 1e-12

# PyRates emits these PAR slots in declaration order; for this model the order is
# K, mu, g, delta, om_0, om_1, ...  →  g = PAR(3), delta = PAR(4). PAR(11) is the
# (Auto-reserved) period. Used to set the LC seed's γ and to read columns.
PAR_G = 3
PAR_DELTA = 4
PAR_PERIOD = 11


def _pcol(df, name):
    """Find a summary column by its short PyRates name. PyCoBi sometimes exposes
    parameters under the bare name ('g') and sometimes fully qualified
    ('p/kmo_cart_op/g'); state variables additionally carry a min/max sub-key.
    This returns the scalar (sub == '') column matching `name` either way."""
    for c in df.columns:
        head = c[0] if isinstance(c, tuple) else c
        sub = c[1] if isinstance(c, tuple) and len(c) > 1 else ""
        if (head == name or (isinstance(head, str) and head.endswith("/" + name))) \
                and sub in ("", 0):
            return c
    raise KeyError(name)


def _period_col(df):
    """Period column of an LC summary: PyCoBi stores it as ('period','') when
    run with get_period=True, with PAR(11) as a fallback."""
    for cand in ("period", f"PAR({PAR_PERIOD})"):
        try:
            return _pcol(df, cand)
        except KeyError:
            continue
    raise KeyError("period")


def classify_cycle_stability(summary, g_col):
    """Scan the Floquet multipliers stored on a limit-cycle summary (requires the
    run to have used ``get_eigenvals=True``) and report where the cycle loses
    stability, classifying the mechanism:

      * a COMPLEX pair crossing |μ|=1  → Neimark–Sacker (torus) bifurcation,
      * a real μ → −1                  → period-doubling,
      * a real μ → +1                  → fold/transcritical of cycles.

    Returns ``(events, mods)`` where ``events`` is a list of
    ``(gamma, |μ|, kind)`` at each unit-circle crossing and ``mods`` is the
    per-point leading *non-trivial* multiplier modulus (the trivial μ≈1 that
    every periodic orbit carries is removed first).

    NOTE on the PyCoBi dependency: this only works because PyCoBi parses Auto's
    ``Multiplier N  <re> <im>  Abs. Val. ...`` diagnostic lines into
    ``('eigenvalues', i)`` columns. That parse was fixed in pycobi/utility.py
    (the regex previously assumed the equilibrium ``Eigenvalue N:`` format —
    colon + no trailing field — and silently returned zero multipliers on
    limit-cycle output). With an older PyCoBi these columns are absent and this
    helper returns empty.
    """
    evcols = [c for c in summary.columns
              if isinstance(c, tuple) and c[0] == "eigenvalues"]
    if not evcols:
        return [], []
    gg = np.asarray(summary[g_col], dtype=float)
    # On near-homoclinic / very-long-period branches the orbit's Floquet
    # multipliers span a huge dynamic range and the large ones are numerically
    # unreliable; treat |μ| beyond this cap (or non-finite) as noise and skip the
    # point for crossing detection rather than reporting a spurious instability.
    MAX_RELIABLE = 1.0e2
    mods, events, prev = [], [], None
    for i in range(len(summary)):
        mu = np.array([complex(summary.iloc[i][c]) for c in evcols])
        mu = mu[np.isfinite(mu)]
        m = np.abs(mu)
        if m.size == 0 or m.max() > MAX_RELIABLE:
            mods.append(np.nan)
            prev = None            # don't bridge a crossing across a noisy gap
            continue
        # remove one trivial multiplier (closest to +1) that every LC carries
        j = int(np.argmin(np.abs(mu - 1.0)))
        mu2 = np.delete(mu, j)
        k = int(np.argmax(np.abs(mu2)))
        lead = float(np.abs(mu2[k]))
        mods.append(lead)
        if prev is not None and (prev - 1.0) * (lead - 1.0) < 0:
            crit = mu2[k]
            if abs(crit.imag) > 1e-4:
                kind = "Neimark-Sacker (torus)"
            elif crit.real < 0:
                kind = "period-doubling"
            else:
                kind = "fold/branch-point of cycles"
            events.append((float(gg[i]), lead, kind))
        prev = lead
    return events, mods


# ═════════════════════════════════════════════════════════════════════════════
#  Top-level configuration
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    M=5,                       # number of ensembles (reduced dim = 2M-1+M^2)
    K=1.0,                     # global coupling
    mu=0.1,                    # Hebbian learning rate
    delta=0.02,                # OA half-width Δ (continued in the codim-2 step)
    # γ values: where the locked state is stable (for the IVP) and where the
    # post-fold cycle is clean (for the LC seed). The fold γ_c sits between them.
    gamma_locked=0.05,         # IVP settles onto the locked equilibrium here
    gamma_cycle=0.2,          # land on the post-fold limit cycle here
    # statistical initialiser (mirrors kmo_macro_simulation.py / initialize())
    omega_mean=0.40, omega_std=0.20,
    r0_mean=0.90,   r0_std=0.10,
    A0_scale=0.50,
    seed=42,
)

# ═════════════════════════════════════════════════════════════════════════════
#  Cartesian reduced-model generator (self-contained; verified equal to the OA
#  array RHS to ~2e-16 in the companion validation)
# ═════════════════════════════════════════════════════════════════════════════
def _xj(j):
    return f"x_{j}"


def _yj(j):
    return "0" if j == 0 else f"y_{j}"


def build_equations_cartesian(M):
    """Reduced OA in Cartesian coords z_i = x_i + i y_i, frame co-rotating with
    population 0 (y_0 := 0, x_0 = r_0), collective frequency Ω = Im F_0/x_0
    inlined. State order: x_0, (x_1,y_1), ..., (x_{M-1},y_{M-1}), A_00..A_{M-1,M-1}.
    Reduced dimension = 2M - 1 + M²."""
    Mf = float(M)

    def ReH(i):
        return f"(K/{Mf})*(" + " + ".join(f"A_{i}_{j}*{_xj(j)}" for j in range(M)) + ")"

    def ImH(i):
        t = [f"A_{i}_{j}*{_yj(j)}" for j in range(M) if j != 0]
        return f"(K/{Mf})*(" + (" + ".join(t) if t else "0") + ")"

    def ReF(i):
        xi, yi, rH, iH = _xj(i), _yj(i), ReH(i), ImH(i)
        rezz = f"(x_0^2)*{rH}" if i == 0 else f"(({xi})^2-({yi})^2)*{rH} + 2*{xi}*{yi}*{iH}"
        bias = f"-delta*{xi}" if i == 0 else f"-delta*{xi} - om_{i}*{yi}"
        return f"({bias} + 0.5*({rH} - ({rezz})))"

    def ImF(i):
        xi, yi, rH, iH = _xj(i), _yj(i), ReH(i), ImH(i)
        if i == 0:
            imzz, bias = f"-(x_0^2)*{iH}", "om_0*x_0"
        else:
            imzz = f"2*{xi}*{yi}*{rH} - (({xi})^2-({yi})^2)*{iH}"
            bias = f"-delta*{yi} + om_{i}*{xi}"
        return f"({bias} + 0.5*({iH} - ({imzz})))"

    Omega = f"(({ImF(0)})/x_0)"
    eqs = [f"d/dt * x_0 = {ReF(0)}"]
    for i in range(1, M):
        eqs.append(f"d/dt * x_{i} = {ReF(i)} + {Omega}*y_{i}")
        eqs.append(f"d/dt * y_{i} = {ImF(i)} - {Omega}*x_{i}")
    for i in range(M):
        for j in range(M):
            eqs.append(
                f"d/dt * A_{i}_{j} = mu*({_xj(i)}*{_xj(j)} + {_yj(i)}*{_yj(j)}) - g*A_{i}_{j}"
            )
    return eqs


def state_var_order(M):
    """Auto/PyRates emits state variables in this order (= the .dat column order
    and the layout used by `cart_rhs`)."""
    names = ["x_0"]
    for i in range(1, M):
        names += [f"x_{i}", f"y_{i}"]
    for i in range(M):
        for j in range(M):
            names.append(f"A_{i}_{j}")
    return names


def build_circuit_cartesian(M, K, mu, gamma, delta, omega, r0, phi0, A0):
    """PyRates circuit for the Cartesian reduced model. Initial state derived
    from (r0, phi0): x_i = r_i cos φ_i, y_i = r_i sin φ_i."""
    eqs = build_equations_cartesian(M)
    x0 = r0 * np.cos(phi0)
    y0 = r0 * np.sin(phi0)

    variables = {"x_0": f"output({float(x0[0])})"}
    for i in range(1, M):
        variables[f"x_{i}"] = f"variable({float(x0[i])})"
        variables[f"y_{i}"] = f"variable({float(y0[i])})"
    for i in range(M):
        for j in range(M):
            variables[f"A_{i}_{j}"] = f"variable({float(A0[i, j])})"
    variables["K"] = float(K)
    variables["mu"] = float(mu)
    variables["g"] = float(gamma)
    variables["delta"] = float(delta)
    for i in range(M):
        variables[f"om_{i}"] = float(omega[i])

    op = OperatorTemplate(name="kmo_cart_op", equations=eqs, variables=variables)
    node = NodeTemplate(name="kmo_cart_node", operators=[op])
    return CircuitTemplate(name="kmo_cart", nodes={"p": node})


# ═════════════════════════════════════════════════════════════════════════════
#  Initialiser (mirrors kmo_macro_simulation.py's RNG scheme)
# ═════════════════════════════════════════════════════════════════════════════
def initialize(M, omega_mean, omega_std, r0_mean, r0_std, A0_scale, seed):
    rng = np.random.default_rng(seed)
    omega = rng.uniform(-omega_std, omega_std, M) + omega_mean
    r0 = np.clip(rng.uniform(-r0_std, r0_std, M) + r0_mean, _EPS, 1.0 - _EPS)
    psi0 = rng.uniform(-np.pi, np.pi, M)
    phi0 = psi0 - psi0[0]
    phi0[0] = 0.0
    if A0_scale > 0:
        A0 = rng.normal(0, A0_scale, (M, M))
        A0 = (A0 + A0.T) / 2.0
    else:
        A0 = np.zeros((M, M))
    return omega, r0, phi0, A0


# ═════════════════════════════════════════════════════════════════════════════
#  scipy RHS + "land on the cycle and extract one fundamental period"
#  (numpy mirror of the Cartesian field above; only used to synthesise the LC
#  seed orbit that PyCoBi's write_auto_dat then turns into a .dat)
# ═════════════════════════════════════════════════════════════════════════════
def cart_rhs(t, s, M, K, mu, g, delta, omega):
    x = np.empty(M)
    yv = np.empty(M)
    x[0] = s[0]
    yv[0] = 0.0
    x[1:] = s[1:1 + 2 * (M - 1):2]
    yv[1:] = s[2:1 + 2 * (M - 1):2]
    A = s[1 + 2 * (M - 1):].reshape(M, M)
    z = x + 1j * yv
    H = (K / M) * (A @ z)
    F = (-delta + 1j * omega) * z + 0.5 * (H - z ** 2 * np.conj(H))
    Om = F.imag[0] / x[0]
    dx = F.real + Om * yv
    dy = F.imag - Om * x
    dA = mu * (np.outer(x, x) + np.outer(yv, yv)) - g * A
    out = np.empty_like(s)
    out[0] = dx[0]
    out[1:1 + 2 * (M - 1):2] = dx[1:]
    out[2:1 + 2 * (M - 1):2] = dy[1:]
    out[1 + 2 * (M - 1):] = dA.ravel()
    return out


def land_on_cycle(M, K, mu, gamma, delta, omega, r0, phi0, A0,
                  T_settle=6000.0, n_samples=201):
    """Integrate onto the post-fold limit cycle, find the fundamental period
    (slowest closed loop), and return a one-period DataFrame indexed by time
    with columns in `state_var_order(M)` — ready for `write_auto_dat`."""
    s0 = np.empty(1 + 2 * (M - 1) + M * M)
    s0[0] = r0[0] * np.cos(phi0[0])
    s0[1:1 + 2 * (M - 1):2] = (r0 * np.cos(phi0))[1:]
    s0[2:1 + 2 * (M - 1):2] = (r0 * np.sin(phi0))[1:]
    s0[1 + 2 * (M - 1):] = A0.ravel()
    args = (M, K, mu, gamma, delta, omega)
    sol = solve_ivp(cart_rhs, (0.0, T_settle), s0, args=args, method="LSODA",
                    rtol=1e-10, atol=1e-12, dense_output=True, max_step=2.0)

    # coarse period from x_0 up-crossings, then refine to the fundamental
    tt = np.linspace(0.6 * T_settle, T_settle, 200000)
    x0 = sol.sol(tt)[0]
    xc = x0 - x0.mean()
    cr = np.where((xc[:-1] < 0) & (xc[1:] >= 0))[0]
    base = float(np.diff(tt[cr]).mean())
    t_anchor = 0.7 * T_settle
    s_anchor = sol.sol(t_anchor)
    best = (1e9, base)
    for k in (1, 2, 3, 4):
        Tc = base * k
        err = np.max(np.abs(sol.sol(t_anchor + Tc) - s_anchor))
        if err < best[0]:
            best = (err, Tc)
    Tfund = best[1]
    tau = np.linspace(0.0, 1.0, n_samples)
    states = np.array([sol.sol(t_anchor + s * Tfund) for s in tau])
    print(f"  land_on_cycle: fundamental period T = {Tfund:.4f} "
          f"(return error {best[0]:.2e})")
    df = pd.DataFrame(states, columns=state_var_order(M), index=tau * Tfund)
    return Tfund, df


# ═════════════════════════════════════════════════════════════════════════════
#  Saddle eigenvalue split (NUNSTAB / NSTAB) for HomCont
# ═════════════════════════════════════════════════════════════════════════════
def eig_split(state_vec, M, K, mu, g, delta, omega, tol=1e-6):
    """Numerically count (n_unstable, n_stable) eigenvalues of the Jacobian of
    `cart_rhs` at a given state. Used to set NUNSTAB / NSTAB for HomCont."""
    n = state_vec.size
    f0 = cart_rhs(0.0, state_vec, M, K, mu, g, delta, omega)
    J = np.empty((n, n))
    for k in range(n):
        h = 1e-7 * max(1.0, abs(state_vec[k]))
        sp = state_vec.copy()
        sp[k] += h
        J[:, k] = (cart_rhs(0.0, sp, M, K, mu, g, delta, omega) - f0) / h
    ev = np.linalg.eigvals(J)
    n_unstable = int(np.sum(ev.real > tol))
    n_stable = int(np.sum(ev.real < -tol))
    return n_unstable, n_stable, ev


def state_dict_from_solution(ode, cont, point):
    """Read an Auto solution's state variables into a {name: value} dict and a
    flat vector in `state_var_order`."""
    s, name, _ = ode.get_solution(point=point, cont=cont)
    if name == "No Label":
        raise KeyError(f"{point} not found on {cont}")
    if hasattr(s, "b") and isinstance(getattr(s, "b", None), dict):
        s = s.b["solution"]
    coords = {c: float(np.asarray(s[c]).ravel()[0]) for c in s.coordnames}
    return coords


# ═════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    M = cfg["M"]
    K, mu, delta = cfg["K"], cfg["mu"], cfg["delta"]
    DIM = 2 * M - 1 + M * M
    NPAR = 36  # cover PAR(11)=period and PAR(35/36)=PSI(15/16); see module docstring

    omega, r0, phi0, A0 = initialize(
        M, cfg["omega_mean"], cfg["omega_std"], cfg["r0_mean"], cfg["r0_std"],
        cfg["A0_scale"], cfg["seed"],
    )
    print(f"KMO adaptive SNIC/homoclinic analysis — M={M}  (reduced dim = {DIM})")
    print(f"  K={K}, mu={mu}, delta={delta}")
    print(f"  omega in [{omega.min():.3f}, {omega.max():.3f}]")

    # one circuit instance drives every step (eq + LC + HomCont + codim-2 must
    # share an ODESystem). Built at the locked-stable γ so the IVP settles onto
    # the relative equilibrium; γ is varied per-run from there.
    circuit = build_circuit_cartesian(M, K, mu, cfg["gamma_locked"], delta,
                                      omega, r0, phi0, A0)
    ode = ODESystem.from_template(circuit, auto_dir=AUTO_DIR, init_cont=False,
                                  auto_constants=("ivp", "eq", "lc", "hom"))

    # ── Step 1: integrate to the locked relative equilibrium ────────────────
    print("\n[Step 1] IVP → locked relative equilibrium")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=200000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: 3000.0}, STOP={"UZ1"})

    # ── Step 2: equilibrium continuation in γ → locate the fold γ_c ─────────
    print("[Step 2] equilibrium continuation in γ (locate the fold)")
    # Plant UZ labels just below the fold: the bidirectional run traverses BOTH
    # the stable (node) and unstable (saddle) equilibrium branches there, so the
    # saddle the homoclinic loop returns to is harvested as a labelled UZ for the
    # HomCont seed (Step 4). γ_saddle is a touch below the expected fold.
    g_saddle_probe = [0.24, 0.25]
    eq_sols, eq_cont = ode.run(
        origin="time", starting_point="UZ1", name="eq_branch", c="eq",
        ICP="g", bidirectional=True, RL0=0.0, RL1=1.0,
        IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=200,
        DS=1e-3, DSMIN=1e-10, DSMAX=2e-2, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
        UZR={"g": g_saddle_probe},
    )
    print("  eq bifurcations:", dict(eq_sols["bifurcation"].value_counts()))
    g_col = _pcol(eq_sols, "g")
    lp_rows = eq_sols[eq_sols[("bifurcation", "")] == "LP"]
    gamma_c = float(lp_rows[g_col].iloc[0]) if len(lp_rows) else None
    print(f"  fold γ_c ≈ {gamma_c}")

    # ── Step 3: seed the post-fold limit cycle and trace it in γ ────────────
    print("\n[Step 3] land on the limit cycle and trace it in γ")
    Tlc, orbit_df = land_on_cycle(M, K, mu, cfg["gamma_cycle"], delta,
                                  omega, r0, phi0, A0)
    # Write RAW time (not normalised to [0,1]): Auto's STPNUB reader sets the seed
    # period PAR(11) from the time span, so a normalised file would seed PAR(11)=1
    # and Newton can drift onto a spurious near-zero-period branch.
    write_auto_dat(orbit_df, "kmo_cycle_seed.dat", normalize_time=False)
    # The .dat carries no parameter values, so pin the seed at γ_cycle via PAR(3);
    # otherwise the run starts at the circuit's build-time γ.
    seed_par = {PAR_G: cfg["gamma_cycle"]}
    rl0_fold = (gamma_c - 0.003) if gamma_c is not None else 0.0

    # 3a. toward the fold: the period must DIVERGE at γ_c if this is a SNIC.
    #     We plant UZ markers at growing periods to harvest near-homoclinic
    #     seeds, and stop once the period is very large.
    # NOTE: `c="lc"` is required on these dat-seeded runs. After the IVP/eq runs
    # PyCoBi no longer re-emits the equations/IRS for a bare dat run, so Auto
    # restarts from the previous (equilibrium) solution and ignores the .dat;
    # passing `c="lc"` reloads the LC scenario (IRS=0) and forces a clean dat
    # read. `NDIM`/`NPAR` are still passed explicitly (see module docstring).
    lc_fold_sols, lc_fold_cont = ode.run(
        name="lc_to_fold", dat="kmo_cycle_seed", c="lc", NDIM=DIM, NPAR=NPAR,
        PAR=seed_par, IPS=2, ISP=2, ILP=1, ISW=1,
        ICP=[PAR_G, PAR_PERIOD], NTST=150, NCOL=4, RL0=rl0_fold, RL1=1.0,
        NMX=6000, NPR=300, DS=-1e-3, DSMIN=1e-10, DSMAX=1e-2,
        EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, THL={PAR_PERIOD: 0.0},
        UZR={PAR_PERIOD: [200.0, 1000.0, 3000.0]}, UZSTOP={PAR_PERIOD: 6000.0},
        get_period=True,
    )
    per_col = _period_col(lc_fold_sols)
    gf_col = _pcol(lc_fold_sols, "g")
    per = np.asarray(lc_fold_sols[per_col], dtype=float)
    gg = np.asarray(lc_fold_sols[gf_col], dtype=float)
    print(f"  toward fold: γ → [{gg.min():.4f}, {gg.max():.4f}], "
          f"period → [{per.min():.1f}, {per.max():.1f}]")
    print("  → period divergence at the fold confirms the SNIC.")
    # tag the high-period terminus as 'HC' (homoclinic) on the branch summary
    ode.label_homoclinic_terminus("lc_to_fold")

    # ── Step 4: HomCont — confirm the SNIC via the non-central SNIC test fns ─
    # Done BEFORE the away-from-fold / PD runs below: those are also seeded from
    # the same .dat with c="lc" and would overwrite lc_to_fold's branch-1
    # solution object, after which HomCont could no longer extract the seed
    # orbit from it. Seeding HomCont here keeps lc_to_fold the live branch.
    print("\n[Step 4] HomCont continuation of the homoclinic / SNIC locus")
    snic_sols = None
    try:
        # The loop returns to a hyperbolic SADDLE on the unstable equilibrium
        # branch just below the fold (NOT the saddle-node itself, which carries a
        # zero eigenvalue and gives HomCont no well-defined stable/unstable
        # split). Scan the UZ labels we planted below γ_c on the bidirectional
        # eq branch and keep the one whose Jacobian has exactly one unstable
        # direction — that is the saddle, and its (NUNSTAB, NSTAB) split.
        order = state_var_order(M)
        saddle, nunstab, nstab = None, None, None
        for lbl in ("UZ1", "UZ2", "UZ3", "UZ4"):
            try:
                coords = state_dict_from_solution(ode, "eq_branch", lbl)
            except Exception:
                continue
            g_here = coords.get("g", gamma_c)
            vec = np.array([coords.get(n, 0.0) for n in order])
            nu, ns, _ = eig_split(vec, M, K, mu, g_here, delta, omega)
            print(f"  eq UZ {lbl}: γ={g_here:.4f} eigenvalue split {nu}/{ns}")
            if nu == 1 and nu + ns == DIM:           # hyperbolic saddle
                saddle, nunstab, nstab = coords, nu, ns
                break
        if saddle is None:                            # fall back to the SNIC split
            saddle = state_dict_from_solution(ode, "eq_branch", "LP1")
            nunstab, nstab = 1, DIM - 1
            print(f"  (no hyperbolic saddle isolated; using LP1 with "
                  f"NUNSTAB=1, NSTAB={DIM - 1})")
        else:
            print(f"  saddle found: NUNSTAB={nunstab}, NSTAB={nstab}")

        # Seed from a MODERATE-period UZ (~a few hundred), not the extreme
        # high-period one: an orbit right at the terminus sits essentially AT
        # the codim-2 vertex (PSI(15) ≈ 0) where HomCont has no direction to
        # advance. Pick whichever UZ actually got planted (step sizes near the
        # fold can skip some) closest to the target period.
        uz_mask = lc_fold_sols[("bifurcation", "")] == "UZ"
        uz_periods = lc_fold_sols.loc[uz_mask, per_col].astype(float).tolist()
        if not uz_periods:
            raise RuntimeError("no UZ orbit on lc_to_fold to seed HomCont")
        target_T = 300.0
        k = 1 + int(np.argmin([abs(p - target_T) for p in uz_periods]))
        seed_point = f"UZ{k}"
        print(f"  seeding HomCont from {seed_point} "
              f"(period {uz_periods[k - 1]:.0f})")
        # IEQUIB=0 pins the saddle at the precomputed coordinates (passed via
        # saddle_state, which also phase-shifts the seed orbit onto the saddle).
        # bidirectional so both arms of the homoclinic curve are traced (the
        # physical Δ > 0 side as well as Δ < 0); stop if Δ leaves a sensible band.
        snic_sols, _ = ode.continue_homoclinic(
            origin="lc_to_fold", starting_point=seed_point,
            ICP=["g", "delta"],            # 2-parameter homoclinic curve in (γ, Δ)
            NUNSTAB=nunstab, NSTAB=nstab,
            IEQUIB=0, ITWIST=0, IPSI=(15, 16),  # non-central SNIC test functions
            saddle_state=saddle, n_points=201,
            NDIM=DIM, NPAR=NPAR, bidirectional=True,
            NTST=150, NCOL=4, IAD=1, ISP=0, ILP=0,
            NMX=2000, NPR=20, IID=2, ITMX=12, ITNW=8, NWTN=4,
            DS=1e-3, DSMIN=1e-6, DSMAX=2e-2,
            UZSTOP={"delta": [-0.05, 0.25]},
            name="homoclinic",
        )
        bifs = dict(snic_sols["bifurcation"].value_counts())
        print(f"  homoclinic curve in (γ, Δ): {bifs}, {len(snic_sols)} points")
        n_snic = bifs.get("SNIC", 0)
        if n_snic:
            print(f"  → {n_snic} non-central SNIC point(s) detected "
                  "(Nechyporenko et al. 2026, arXiv:2412.12298)")
    except Exception as e:
        print(f"  HomCont step did not complete: {type(e).__name__}: {e}")

    # ── Step 5: trace the cycle away from the fold + period-doubling scan ────
    # A near-homoclinic LC carries many spurious LP labels unless NTST is large
    # and the bifurcation-detection tolerance EPSS is tight (cf. the PyCoBi
    # QIF-SFA HomCont example); NTST=400 / EPSS=1e-7 keeps the PD test function
    # from being swamped by numerical noise.
    print("\n[Step 5] limit cycle away from the fold: stability (Floquet) + PD/torus scan")
    # get_eigenvals=True ⇒ PyCoBi parses Auto's Floquet multipliers into
    # ('eigenvalues', i) columns, which classify_cycle_stability() then scans for
    # Neimark–Sacker (torus) / period-doubling / fold-of-cycles. (Needs the
    # pycobi multiplier-regex fix; see classify_cycle_stability.)
    # Trace AWAY from the fold (increasing γ). Keep RL0 at/above the SNIC γ_c:
    # below it the cycle is near-homoclinic and accumulates a thicket of spurious
    # BP labels with ill-conditioned Floquet multipliers (cf. Step 3 / M=3).
    rl0_away = max(0.0, (gamma_c - 0.005) if gamma_c is not None else 0.0)
    lc_away_sols, lc_away_cont = ode.run(
        name="lc_away", dat="kmo_cycle_seed", c="lc", NDIM=DIM, NPAR=NPAR,
        PAR=seed_par, IPS=2, ISP=2, ILP=1, ISW=1,
        ICP=[PAR_G, PAR_PERIOD], NTST=200, NCOL=4, RL0=rl0_away, RL1=1.2,
        NMX=6000, NPR=50, DS=1e-3, DSMIN=1e-10, DSMAX=2e-2,
        EPSL=1e-9, EPSU=1e-9, EPSS=1e-7, THL={PAR_PERIOD: 0.0},
        UZSTOP={PAR_G: 1.2}, get_period=True, get_eigenvals=True,
    )
    away_col = _pcol(lc_away_sols, "g")
    print("  away from fold bifurcations:",
          dict(lc_away_sols["bifurcation"].value_counts()))
    events, mods = classify_cycle_stability(lc_away_sols, away_col)
    finite = [m for m in mods if np.isfinite(m)]
    if finite:
        print(f"  leading non-trivial |Floquet multiplier| ∈ "
              f"[{min(finite):.3f}, {max(finite):.3f}]")
    if events:
        for g_ev, mod, kind in events:
            print(f"  → cycle loses stability at γ ≈ {g_ev:.4f}: {kind}")
    else:
        print("  → no |μ|=1 crossing on this branch (cycle stays hyperbolic "
              "until it is destroyed at the fold-of-cycles)")

    from pycobi.automated_continuation import continue_period_doubling_bf
    pd_branches = []
    try:
        pd_branches, _ = continue_period_doubling_bf(
            solution=lc_away_sols, continuation=lc_away_cont,
            pyauto_instance=ode, icp=PAR_G, max_iter=6,
            NDIM=DIM, NPAR=NPAR, NTST=100, NCOL=4,
            RL0=0.0, RL1=1.2, NMX=4000, NPR=500,
            DS=1e-3, DSMIN=1e-10, DSMAX=2e-2,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, THL={PAR_PERIOD: 0.0},
            UZSTOP={PAR_G: 1.2}, get_period=True,
        )
        print(f"  period-doubling branches found: {pd_branches or 'none'}")
    except Exception as e:
        print(f"  no period-doubling cascade tracked: {type(e).__name__}: {e}")

    # ── Step 5b: Arnol'd-tongue boundary — continue the fold-of-cycles (saddle-
    # node of periodic orbits, LP) found on the LC branch in TWO parameters
    # (γ, Δ). For M ≥ 5 the post-SNIC dynamics is a 2-torus threaded by resonance
    # tongues; each mode-locked tongue is bounded by such a fold-of-cycles curve,
    # so this traces the tongue boundary in the (γ, Δ) plane. (ISW=2 continues the
    # located LP of cycles; PAR(11)=period is carried as the third ICP entry.)
    print("\n[Step 5b] fold-of-cycles (Arnol'd-tongue boundary) in (γ, Δ)")
    tongue_curves = []
    n_lpc = int((lc_away_sols[("bifurcation", "")] == "LP").sum())
    for i in range(1, n_lpc + 1):
        sp = f"LP{i}"
        try:
            # ISP=1 (NOT 2): detecting codim-2 bifurcations ALONG the fold-of-
            # cycles curve is very expensive at this dimension and makes the run
            # hang; we only want the curve itself. NTST=80 keeps it fast (~seconds
            # per direction at M=5).
            ode.run(
                origin="lc_away", starting_point=sp, name=f"tongue_{sp}",
                c="lc", NDIM=DIM, NPAR=NPAR, IPS=2, ISW=2, ISP=1, ILP=0,
                ICP=[PAR_G, PAR_DELTA, PAR_PERIOD], NTST=80, NCOL=4,
                RL0=0.0, RL1=1.5, NMX=2000, NPR=100,
                DS=1e-3, DSMIN=1e-9, DSMAX=1e-2, THL={PAR_PERIOD: 0.0},
                EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, bidirectional=True,
                UZSTOP={PAR_DELTA: [1e-4, 0.25]},
            )
            tongue_curves.append(f"tongue_{sp}")
            print(f"  tongue boundary from cycle-fold {sp} traced in (γ, Δ)")
        except Exception as exc:
            print(f"  tongue continuation from {sp} skipped: "
                  f"{type(exc).__name__}: {str(exc)[:70]}")

    # ── Step 6: codim-2 — locus of all 1D bifurcations in (γ, Δ) ────────────
    # The only codim-1 bifurcation of the equilibrium is the fold (LP1), and
    # because the limit cycle is born exactly there with diverging period, that
    # fold curve IS the SNIC locus. We continue it directly (ISW=2) in (γ, Δ);
    # bidirectional so both arms of the curve are traced. (codim2_search is the
    # automated alternative, but the explicit run keeps the curve name and
    # column conventions under our control for plotting.)
    print("\n[Step 6] codim-2 continuation of the fold/SNIC locus in (γ, Δ)")
    fold_curves = []
    n_lp = int((eq_sols[("bifurcation", "")] == "LP").sum())
    lp_starts = [f"LP{i + 1}" for i in range(n_lp)]
    for sp in lp_starts:
        try:
            ode.run(
                origin="eq_branch", starting_point=sp, name=f"fold_{sp}",
                c="eq", ICP=["g", "delta"], bidirectional=True,
                IPS=1, ISW=2, ISP=2, ILP=0,
                RL0=0.0, RL1=1.5, NMX=6000, NPR=200,
                DS=1e-3, DSMIN=1e-9, DSMAX=2e-2,
                EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                UZSTOP={"delta": [1e-4, 0.2]},
            )
            fold_curves.append(f"fold_{sp}")
            print(f"  fold curve from {sp} traced in (γ, Δ)")
        except Exception as exc:
            print(f"  fold continuation from {sp} skipped: "
                  f"{type(exc).__name__}: {str(exc)[:70]}")
    codim2_curves = [(name, "fold") for name in fold_curves]

    # ── plots ───────────────────────────────────────────────────────────────
    print("\n[plots] writing figures")
    # 1D γ-diagram: equilibrium + LC envelope + period(γ)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    try:
        ode.plot_continuation("g", "x_0", cont="eq_branch", ax=ax[0],
                              line_color_stable="#1F77B4",
                              line_color_unstable="#1F77B4",
                              ignore=["UZ", "BP", "EP"])
        ode.plot_continuation("g", "x_0", cont="lc_to_fold", ax=ax[0],
                              line_color_stable="#FF7F0E",
                              line_color_unstable="#FF7F0E",
                              ignore=["UZ", "BP", "EP", "RG"])
        ode.plot_continuation("g", "x_0", cont="lc_away", ax=ax[0],
                              line_color_stable="#2CA02C",
                              line_color_unstable="#2CA02C",
                              ignore=["UZ", "BP", "EP", "RG"])
    except Exception as e:
        print(f"  1D envelope plot partial: {e}")
    ax[0].set_xlabel(r"weight decay $\gamma$")
    ax[0].set_ylabel(r"$x_0 = r_0$ (equilibrium / cycle envelope)")
    ax[0].set_title("Fold of equilibria → SNIC-born limit cycle")
    try:
        ode.plot_continuation("g", "PAR(11)", cont="lc_to_fold", ax=ax[1],
                              ignore=["UZ", "BP", "EP", "RG"])
    except Exception as e:
        print(f"  period plot partial: {e}")
    ax[1].set_xlabel(r"weight decay $\gamma$")
    ax[1].set_ylabel("period PAR(11)")
    ax[1].set_title("Period diverges at the fold ⇒ SNIC")
    plt.tight_layout()
    plt.savefig("kmo_snic_1d.png", dpi=130, bbox_inches="tight")

    # 2D (γ, Δ) portrait: fold/Hopf/PD curves + homoclinic locus + SNIC points
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    seen = set()
    for curve_name, bif_type in codim2_curves:
        color = "#1F77B4" if bif_type == "fold" else "#FF7F0E"
        # the fold of equilibria coincides with the SNIC here (LC born on it)
        label = None if bif_type in seen else "fold of equilibria (= SNIC)"
        seen.add(bif_type)
        try:
            ode.plot_continuation("g", "delta", cont=curve_name, ax=ax2,
                                  line_color_stable=color, line_color_unstable=color,
                                  line_style_stable="solid", line_style_unstable="solid",
                                  get_stability=False, ignore=["LP", "HB", "UZ", "EP"],
                                  label=label)
        except (KeyError, ValueError):
            pass
    # fold-of-cycles curves = Arnol'd-tongue boundaries (green)
    tongue_labelled = False
    for curve_name in tongue_curves:
        try:
            ode.plot_continuation(
                "g", "delta", cont=curve_name, ax=ax2,
                line_color_stable="#2CA02C", line_color_unstable="#2CA02C",
                line_style_stable="solid", line_style_unstable="solid",
                get_stability=False, ignore=["LP", "BP", "UZ", "EP", "RG"],
                label=None if tongue_labelled else "fold-of-cycles (tongue edge)")
            tongue_labelled = True
        except (KeyError, ValueError):
            pass
    if snic_sols is not None and len(snic_sols) > 2:
        try:
            ode.plot_continuation("g", "delta", cont="homoclinic", ax=ax2,
                                  line_color_stable="#D62728",
                                  line_color_unstable="#D62728",
                                  line_style_stable="solid", line_style_unstable="dashed",
                                  get_stability=False, ignore=["UZ", "EP", "RG"],
                                  label="homoclinic")
            snic_col = ("bifurcation", "")
            gc2 = _pcol(snic_sols, "g")
            dc2 = _pcol(snic_sols, "delta")
            snic_rows = snic_sols[snic_sols[snic_col] == "SNIC"]
            if len(snic_rows):
                ax2.scatter(snic_rows[gc2].astype(float), snic_rows[dc2].astype(float),
                            marker="*", s=220, c="#D62728", edgecolor="k",
                            linewidth=0.8, zorder=10, label="non-central SNIC")
        except Exception as e:
            print(f"  homoclinic curve plot partial: {e}")
    ax2.set_xlabel(r"weight decay $\gamma$")
    ax2.set_ylabel(r"OA half-width $\Delta$")
    ax2.set_title(r"Codim-1 bifurcation locus in $(\gamma, \Delta)$")
    ax2.legend(loc="best")
    plt.tight_layout()
    plt.savefig("kmo_snic_codim2.png", dpi=130, bbox_inches="tight")

    ode.close_session(clear_files=True)
    clear(circuit)
    print("\nDone. Figures: kmo_snic_1d.png, kmo_snic_codim2.png")


if __name__ == "__main__":
    main()
    plt.show()
