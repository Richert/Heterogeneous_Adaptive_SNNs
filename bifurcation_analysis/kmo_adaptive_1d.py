r"""
1D bifurcation diagram of the adaptive Kuramoto–OA ensemble model
=================================================================

Goal
----
The mean-field model is highly MULTISTABLE — different seeds settle onto
different attractors. This script builds a proper 1D bifurcation diagram in the
weight-decay parameter γ that traces the equilibrium branch and the limit-cycle
branches (stable and unstable), so the coexisting states can be read off at a
glance.

Pipeline (only the continuations needed for the 1D diagram)
-----------------------------------------------------------
  1. Settle onto the locked relative equilibrium by time integration at a small,
     monostable γ (default 0.05).
  2. Continue that equilibrium in γ (IPS=1). It loses stability at a FOLD; we
     already know (companion HomCont analysis) that this fold is a SNIC — a limit
     cycle is born there with diverging period.
  3. Land on a stable limit cycle just past the fold (time integration), then run
     a LARGE-NMX limit-cycle continuation (IPS=2, ILP=1) BIDIRECTIONALLY. Arclength
     continuation automatically rounds any fold-of-cycles (saddle-node of periodic
     orbits) and carries on along the unstable branch, so a single run yields the
     stable + unstable portions of one connected branch.
  4. Multistability sweep: if the branch has a fold-of-cycles, time-integrate just
     PAST the right-most one to look for a *different* stable limit cycle, and if
     found, continue it too. Repeat until no new stable cycle is found (or the
     attractor there is a fixed point / quasiperiodic torus, which IPS=2 cannot
     continue), up to ``MAX_LC_BRANCHES``.

Final diagram (kmo_adaptive_1d.png)
-----------------------------------
  * equilibria (non-periodic) ............ one colour (blue; dashed where unstable)
  * stable limit cycles .................. green
  * unstable limit cycles ................ red
  * folds of equilibria .................. circle markers
  * folds of cycles (saddle-node of LCs) . triangle markers

Coordinates
-----------
Cartesian co-rotating order parameters z_i = x_i + i y_i (pin y_0 := 0, x_0 = r_0),
so a winding/​slipping phase is a CLOSED loop that Auto can represent as a periodic
orbit. Equilibria (hence the fold) are unchanged. Reduced dim = 2M − 1 + M².

Run inside the ``pycobi`` conda env with meson/ninja on PATH:
    PATH="$CONDA_PREFIX/bin:$PATH" python kmo_adaptive_1d.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
from pycobi import ODESystem
from pycobi.utility import write_auto_dat

AUTO_DIR = "~/PycharmProjects/auto-07p"
_EPS = 1e-12

# PyRates emits PAR slots in declaration order: K, mu, g, delta, om_0, ...
# → g = PAR(3), delta = PAR(4). PAR(11) is Auto's reserved period slot.
PAR_G = 3
PAR_DELTA = 4
PAR_PERIOD = 11

# colours / markers for the 1D diagram
C_EQ = "#1F77B4"      # equilibria (non-periodic)
C_LC_STABLE = "#2CA02C"
C_LC_UNSTABLE = "#D62728"
M_FOLD_EQ = "o"       # fold of equilibria
M_FOLD_LC = "^"       # fold of cycles


# ═════════════════════════════════════════════════════════════════════════════
#  Top-level configuration
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    M=5,                       # number of ensembles (reduced dim = 2M-1+M^2)
    K=1.0,                     # global coupling
    mu=0.1,                    # Hebbian learning rate
    delta=0.02,                # OA half-width Δ
    gamma_locked=0.05,         # IVP settles onto the locked equilibrium here (monostable)
    gamma_cycle=0.20,          # first seed for a post-fold limit cycle
    # statistical initialiser (mirrors kmo_macro_simulation.py)
    omega_mean=0.40, omega_std=0.20,
    r0_mean=0.90,   r0_std=0.10,
    A0_scale=0.50,
    seed=42,
    # multistability sweep
    MAX_LC_BRANCHES=5,         # max number of distinct LC branches to trace
    gamma_max=0.60,            # stop seeding new cycles beyond this γ
    seed_offset=0.012,         # how far past a fold-of-cycles to look for the next cycle
)


# ═════════════════════════════════════════════════════════════════════════════
#  Cartesian reduced-model generator
# ═════════════════════════════════════════════════════════════════════════════
def _xj(j):
    return f"x_{j}"


def _yj(j):
    return "0" if j == 0 else f"y_{j}"


def build_equations_cartesian(M):
    """Reduced OA in Cartesian coords z_i = x_i + i y_i, frame co-rotating with
    population 0 (y_0 := 0, x_0 = r_0), Ω = Im F_0/x_0 inlined. State order:
    x_0, (x_1,y_1), …, (x_{M-1},y_{M-1}), A_00…A_{M-1,M-1}.  Dim = 2M − 1 + M²."""
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
    """Auto/PyRates emits state variables in this order (= .dat column order)."""
    names = ["x_0"]
    for i in range(1, M):
        names += [f"x_{i}", f"y_{i}"]
    for i in range(M):
        for j in range(M):
            names.append(f"A_{i}_{j}")
    return names


def build_circuit_cartesian(M, K, mu, gamma, delta, omega, r0, phi0, A0, name="kmo_cart"):
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
    op = OperatorTemplate(name=f"{name}_op", equations=eqs, variables=variables)
    node = NodeTemplate(name=f"{name}_node", operators=[op])
    return CircuitTemplate(name=name, nodes={"p": node})


def make_jacobian_eval(M, K, mu, delta, omega, r0, phi0, A0):
    """Build PyRates' ANALYTICAL Jacobian of the reduced field (symbolic, via
    ``CircuitTemplate.get_jacobian_func``) and return ``jac(y, g) -> (n×n)`` — the
    same exact Jacobian Auto uses internally (JAC=1), evaluated in Python. Used to
    assign equilibrium stability robustly (see recompute_equilibrium_stability).

    Generated from a throwaway circuit (its own name) with the numpy backend, so
    it does not disturb the Fortran ODESystem circuit."""
    jcirc = build_circuit_cartesian(M, K, mu, 0.05, delta, omega, r0, phi0, A0,
                                    name="kmo_cart_jac")
    Jf, _, jarg_names, jsv = jcirc.get_jacobian_func(
        "kmo_jac", 1e-3, backend="default", vectorize=False, verbose=False, clear=True)
    # state ordering returned by PyRates matches state_var_order(M)
    assert [k.split("/")[-1] for k in jsv] == state_var_order(M)
    pnames = [n.split("/")[-1] for n in jarg_names[2:]]   # params in call order
    base = {"K": float(K), "mu": float(mu), "delta": float(delta)}
    for i in range(M):
        base[f"om_{i}"] = float(omega[i])

    def jac(y, g):
        base["g"] = float(g)
        return np.asarray(Jf(0.0, np.asarray(y, dtype=float),
                             *[base[p] for p in pnames]))

    clear(jcirc)
    return jac


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
#  numpy mirror of the Cartesian field — only to synthesise LC seed orbits
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


def _initial_state(M, r0, phi0, A0):
    s0 = np.empty(1 + 2 * (M - 1) + M * M)
    s0[0] = r0[0] * np.cos(phi0[0])
    s0[1:1 + 2 * (M - 1):2] = (r0 * np.cos(phi0))[1:]
    s0[2:1 + 2 * (M - 1):2] = (r0 * np.sin(phi0))[1:]
    s0[1 + 2 * (M - 1):] = A0.ravel()
    return s0


def settle_and_classify(M, K, mu, gamma, delta, omega, r0, phi0, A0,
                        T_settle=6000.0, n_samples=201,
                        amp_tol=2e-3, closure_tol=3e-2):
    """Time-integrate from the synchronized initial condition at this γ and
    classify the attractor reached:

      * 'fixedpoint' — x_0 stops oscillating (amplitude < amp_tol),
      * 'cycle'      — a periodic orbit whose full state closes to < closure_tol
                       at the fundamental period (⇒ continuable with IPS=2),
      * 'torus'      — oscillatory but does not close (quasiperiodic; IPS=2 N/A).

    For a 'cycle' it returns a one-period DataFrame (columns = state_var_order)
    ready for ``write_auto_dat``. Returns ``(status, period, df, amplitude, err)``.
    """
    s0 = _initial_state(M, r0, phi0, A0)
    args = (M, K, mu, gamma, delta, omega)
    sol = solve_ivp(cart_rhs, (0.0, T_settle), s0, args=args, method="LSODA",
                    rtol=1e-10, atol=1e-12, dense_output=True, max_step=2.0)

    tt = np.linspace(0.7 * T_settle, T_settle, 200000)
    x0 = sol.sol(tt)[0]
    amp = float(x0.max() - x0.min())
    if amp < amp_tol:
        return "fixedpoint", None, None, amp, None

    xc = x0 - x0.mean()
    cr = np.where((xc[:-1] < 0) & (xc[1:] >= 0))[0]
    if cr.size < 2:
        return "torus", None, None, amp, None
    base = float(np.diff(tt[cr]).mean())
    t_anchor = 0.7 * T_settle
    s_anchor = sol.sol(t_anchor)
    best = (1e9, base)
    for k in (1, 2, 3, 4):
        Tc = base * k
        err = float(np.max(np.abs(sol.sol(t_anchor + Tc) - s_anchor)))
        if err < best[0]:
            best = (err, Tc)
    err, Tfund = best
    if err > closure_tol:
        return "torus", None, None, amp, err
    tau = np.linspace(0.0, 1.0, n_samples)
    states = np.array([sol.sol(t_anchor + s * Tfund) for s in tau])
    df = pd.DataFrame(states, columns=state_var_order(M), index=tau * Tfund)
    return "cycle", Tfund, df, amp, err


# ═════════════════════════════════════════════════════════════════════════════
#  small summary helpers
# ═════════════════════════════════════════════════════════════════════════════
def _pcol(df, name):
    """Scalar (sub=='' or 0) summary column matching short PyRates name `name`."""
    for c in df.columns:
        head = c[0] if isinstance(c, tuple) else c
        sub = c[1] if isinstance(c, tuple) and len(c) > 1 else ""
        if (head == name or (isinstance(head, str) and head.endswith("/" + name))) \
                and sub in ("", 0):
            return c
    raise KeyError(name)


def _x0_envelope(df, row_idx):
    """Max over the x_0 column(s) of a summary row — gives the cycle-envelope top
    (or the scalar equilibrium value)."""
    cols = [c for c in df.columns if (c[0] if isinstance(c, tuple) else c) == "x_0"]
    vals = [float(df[c].iloc[row_idx]) for c in cols]
    return max(vals)


def _fold_gammas(sols):
    """γ values of the LP (fold) points on a summary, sorted ascending."""
    g_col = _pcol(sols, "g")
    lp = sols[sols[("bifurcation", "")] == "LP"]
    return sorted(float(v) for v in lp[g_col]) if len(lp) else []


def recompute_equilibrium_stability(sols, M, jac, tol=1e-6):
    """Overwrite the summary's ``stability`` column using the analytical Jacobian.

    Auto only writes its eigenvalue spectrum to the diagnostics at *some* points;
    where it is missing, PyCoBi falls back to the (unreliable) point-index-sign
    convention, so the parsed ``stability`` flickers True/False along a branch
    that is actually uniformly stable (e.g. the locked node here renders dashed).
    We instead evaluate PyRates' analytical Jacobian at each stored equilibrium
    state and flag it stable iff no eigenvalue has positive real part. Mutates
    ``sols`` in place (the object plot_continuation reads)."""
    order = state_var_order(M)
    g_col = _pcol(sols, "g")
    var_cols = [_pcol(sols, n) for n in order]
    stab = np.empty(len(sols), dtype=bool)
    for r in range(len(sols)):
        g_here = float(sols[g_col].iloc[r])
        y = np.array([float(sols[c].iloc[r]) for c in var_cols])
        ev = np.linalg.eigvals(jac(y, g_here))
        stab[r] = bool(np.sum(ev.real > tol) == 0)
    sols[("stability", "")] = stab
    return sols


def add_fold_markers(ax, sols, marker, color="k", label=None, size=70):
    """Scatter fold (LP) markers at (γ, x_0-envelope) for a continuation."""
    g_col = _pcol(sols, "g")
    bif = sols[("bifurcation", "")].to_numpy()
    idx = np.where(bif == "LP")[0]
    if idx.size == 0:
        return
    gs = [float(sols[g_col].iloc[i]) for i in idx]
    ys = [_x0_envelope(sols, i) for i in idx]
    ax.scatter(gs, ys, marker=marker, s=size, facecolors="none",
               edgecolors=color, linewidths=1.6, zorder=12, label=label)


# ═════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    M = cfg["M"]
    K, mu, delta = cfg["K"], cfg["mu"], cfg["delta"]
    DIM = 2 * M - 1 + M * M
    NPAR = max(11, 2 * M)        # cover PAR(11)=period; model params are PAR(1..2M-1)

    omega, r0, phi0, A0 = initialize(
        M, cfg["omega_mean"], cfg["omega_std"], cfg["r0_mean"], cfg["r0_std"],
        cfg["A0_scale"], cfg["seed"],
    )
    print(f"KMO adaptive — 1D bifurcation diagram in γ — M={M}  (reduced dim = {DIM})")
    print(f"  K={K}, mu={mu}, delta={delta},  omega∈[{omega.min():.3f},{omega.max():.3f}]")

    # PyRates' ANALYTICAL Jacobian (symbolic), evaluated in Python — used to
    # assign equilibrium stability robustly (see recompute_equilibrium_stability).
    jac = make_jacobian_eval(M, K, mu, delta, omega, r0, phi0, A0)

    circuit = build_circuit_cartesian(M, K, mu, cfg["gamma_locked"], delta,
                                      omega, r0, phi0, A0)
    # analytical_jacobian=True (the default, set explicitly here) makes PyRates
    # emit DFDU/DFDP into the generated Fortran and write JAC=1 into the c.* files,
    # so AUTO-07P uses the EXACT analytical Jacobian for every continuation rather
    # than finite differences — more accurate and more robust, especially for the
    # limit-cycle (IPS=2) steps.
    ode = ODESystem.from_template(circuit, auto_dir=AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True,
                                  auto_constants=("ivp", "eq", "lc"))

    # ── Step 1: IVP → locked relative equilibrium ───────────────────────────
    print("\n[1] IVP → locked relative equilibrium (γ = {:.3f})".format(cfg["gamma_locked"]))
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=200000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: 3000.0}, STOP={"UZ1"})

    # ── Step 2: equilibrium continuation in γ → fold (= SNIC) ───────────────
    print("[2] equilibrium continuation in γ")
    eq_sols, _ = ode.run(
        origin="time", starting_point="UZ1", name="eq_branch", c="eq",
        ICP="g", bidirectional=True, RL0=0.0, RL1=1.0,
        IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=500,
        DS=1e-3, DSMIN=1e-10, DSMAX=2e-2, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
        get_stability=True,
    )
    print("  eq bifurcations:", dict(eq_sols["bifurcation"].value_counts()))
    # Robust stability for the equilibrium branch from the analytical Jacobian
    # (Auto's parsed stability flickers where it omits the eigenvalue line).
    recompute_equilibrium_stability(eq_sols, M, jac)
    eq_folds = _fold_gammas(eq_sols)
    gamma_c = eq_folds[0] if eq_folds else None
    print(f"  fold(s) of equilibria at γ = {eq_folds}  (SNIC at γ_c ≈ {gamma_c})")

    # ── Steps 3–4: iteratively trace limit-cycle branches ───────────────────
    print("\n[3] trace limit-cycle branches (large NMX, bidirectional)")
    lc_branches = []                  # list of (name, summary)
    seed_gamma = cfg["gamma_cycle"]
    # Period cutoff for the toward-SNIC arm. The period DIVERGES at the SNIC, and
    # very close to it the branch develops near-homoclinic "snaking" (a cascade of
    # tiny folds as period → ∞). Stopping at a moderate period keeps the diagram
    # readable while still showing the period blow-up that identifies the SNIC.
    per_max = 2100.0

    for b in range(cfg["MAX_LC_BRANCHES"]):
        print(f"\n  branch {b}: probing for a stable cycle at γ = {seed_gamma:.4f}")
        status, T, orbit_df, amp, err = settle_and_classify(
            M, K, mu, seed_gamma, delta, omega, r0, phi0, A0)
        if status != "cycle":
            print(f"    attractor is '{status}' (amp={amp:.3g}, closure err="
                  f"{None if err is None else f'{err:.2g}'}); stop seeding.")
            break
        print(f"    landed on a stable cycle: period T = {T:.2f} (closure {err:.1e})")

        dat = f"seed_b{b}"
        write_auto_dat(orbit_df, f"{dat}.dat", normalize_time=False)
        # Large-NMX, bidirectional LC continuation. ILP=1 ⇒ arclength rounds any
        # fold-of-cycles and carries on along the unstable branch; get_stability
        # ⇒ stable/unstable colouring. c="lc" forces a clean dat reload; NDIM/NPAR
        # are passed explicitly on dat-seeded runs (PyCoBi won't infer them).
        name = f"lc{b}"
        sols, _ = ode.run(
            name=name, dat=dat, c="lc", NDIM=DIM, NPAR=NPAR,
            PAR={PAR_G: seed_gamma}, IPS=2, ISP=2, ILP=1, ISW=1,
            ICP=[PAR_G, PAR_PERIOD], NTST=150, NCOL=4,
            RL0=0.0, RL1=cfg["gamma_max"] + 0.2, NMX=8000, NPR=100,
            DS=1e-3, DSMIN=1e-11, DSMAX=1e-2,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-7, THL={PAR_PERIOD: 0.0},
            UZSTOP={PAR_PERIOD: per_max}, bidirectional=True,
            get_period=True, get_stability=True,
        )
        lc_branches.append((name, sols))
        g_col = _pcol(sols, "g")
        gg = np.asarray(sols[g_col], dtype=float)
        folds = _fold_gammas(sols)
        print(f"    branch γ ∈ [{gg.min():.4f}, {gg.max():.4f}]; "
              f"fold(s)-of-cycles at γ = {[round(x, 4) for x in folds]}")

        # next branch: just past the right-most fold-of-cycles on this branch.
        if not folds:
            print("    no fold-of-cycles on this branch → branch spans the range; done.")
            break
        next_gamma = max(folds) + cfg["seed_offset"]
        if next_gamma > cfg["gamma_max"]:
            print(f"    next seed γ={next_gamma:.4f} exceeds γ_max; done.")
            break
        # avoid re-seeding inside a γ-range this branch already covers
        if gg.min() - 1e-6 <= next_gamma <= gg.max() + 1e-6:
            print(f"    next seed γ={next_gamma:.4f} already covered by this branch; done.")
            break
        seed_gamma = next_gamma

    # ── Final 1D bifurcation diagram ────────────────────────────────────────
    print("\n[plot] assembling 1D bifurcation diagram")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # equilibria (non-periodic) — one colour, dashed where unstable (the stable
    # node and unstable saddle below the fold nearly coincide in x_0, as expected
    # at a saddle-node/SNIC).
    ode.plot_continuation("g", "x_0", cont="eq_branch", ax=ax,
                          line_color_stable=C_EQ, line_color_unstable=C_EQ,
                          line_style_stable="solid", line_style_unstable="dashed",
                          ignore=["UZ", "BP", "EP"], bifurcation_legend=False,
                          linewidths=2.4)
    add_fold_markers(ax, eq_sols, M_FOLD_EQ, color="k", size=90)

    # limit cycles — stable green, unstable red (envelope = min/max of x_0)
    for name, sols in lc_branches:
        try:
            ode.plot_continuation("g", "x_0", cont=name, ax=ax,
                                  line_color_stable=C_LC_STABLE,
                                  line_color_unstable=C_LC_UNSTABLE,
                                  line_style_stable="solid", line_style_unstable="solid",
                                  ignore=["UZ", "BP", "EP", "RG"], bifurcation_legend=False,
                                  linewidths=1.8)
        except Exception as e:
            print(f"  LC plot for {name} partial: {e}")
        add_fold_markers(ax, sols, M_FOLD_LC, color="k", size=90)

    if gamma_c is not None:
        ax.axvline(gamma_c, color="0.6", lw=0.8, ls=":", zorder=0)

    ax.set_xlabel(r"weight decay $\gamma$")
    ax.set_ylabel(r"$x_0 = r_0$  (equilibrium value / cycle envelope)")
    ax.set_title(f"1D bifurcation diagram in $\\gamma$  (M={M})")
    legend = [
        Line2D([0], [0], color=C_EQ, lw=2, label="equilibrium (solid=stable, dashed=unstable)"),
        Line2D([0], [0], color=C_LC_STABLE, lw=2, label="stable limit cycle"),
        Line2D([0], [0], color=C_LC_UNSTABLE, lw=2, label="unstable limit cycle"),
        Line2D([0], [0], marker=M_FOLD_EQ, color="k", lw=0, markerfacecolor="none",
               label="fold of equilibria (SNIC)"),
        Line2D([0], [0], marker=M_FOLD_LC, color="k", lw=0, markerfacecolor="none",
               label="fold of cycles"),
    ]
    ax.legend(handles=legend, loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig("kmo_adaptive_1d.png", dpi=140, bbox_inches="tight")
    print("  saved kmo_adaptive_1d.png")

    ode.close_session(clear_files=True)
    clear(circuit)
    print("Done.")


if __name__ == "__main__":
    main()
    plt.show()
