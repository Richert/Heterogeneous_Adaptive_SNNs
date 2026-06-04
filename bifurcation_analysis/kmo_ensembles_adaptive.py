"""
Bifurcation analysis & parameter continuation of coupled Ott–Antonsen Kuramoto
populations with adaptive (Hebbian) inter-population coupling — via PyCoBi/Auto-07p.
===============================================================================

This script takes the ensemble equations from `kmo_macro_simulation.py` and sets
them up for numerical continuation in PyCoBi. Two things have to be handled before
Auto-07p can do anything useful; both are dealt with here and explained inline:

  (1) PHASE-SHIFT SYMMETRY.  Every RHS term in the model depends only on phase
      DIFFERENCES psi_j - psi_i, so the system is invariant under the global shift
      psi_i -> psi_i + c.  Consequently the phase-locked states are *relative
      equilibria*: r_i and A_ij are constant while all psi_i advance at a common
      frequency Omega.  These are NOT fixed points of the raw equations, so Auto
      (which finds f(y)=0) would never converge.  We quotient out the symmetry by
      moving to phase differences phi_i = psi_i - psi_0  (phi_0 := 0).  The reduced
      state (r_0..r_{M-1}, phi_1..phi_{M-1}, A_00..A_{M-1,M-1}) is autonomous and
      closed, and its equilibria ARE the rotating-wave / phase-locked states.
      Reduced dimension = 2M - 1 + M^2.  (See validate_reduction.py for a numerical
      proof that this reduction reproduces the full model exactly.)

  (2) DIMENSIONALITY.  Equilibrium continuation in Auto is comfortable into the
      hundreds of variables, but periodic-orbit (limit-cycle) continuation scales
      with NTST*NCOL*dim and gets expensive fast.  Keep M small (M=2 or 3) for the
      full pipeline; equilibrium-only sweeps tolerate larger M.

  (3) SMOOTHNESS.  Auto needs a C^1 (ideally C^2) vector field. The anti-Hebbian
      rule in the original script uses |sin(dpsi)|, which is non-differentiable at
      dpsi=0. For continuation we substitute the smooth surrogate sin^2(dpsi)
      (same "quadrature-rewarded" character, infinitely differentiable). The pure
      Hebbian (cos) rule is already smooth and is the default below.

REQUIREMENTS
------------
  * Auto-07p installed (https://github.com/auto-07p/auto-07p)
  * pyrates  >= 1.1   (pip install pyrates)
  * pycobi   >= 0.10  (pip install pycobi)
  Set AUTO_DIR below to your Auto-07p installation directory.

NOTE: this file was written and the *model equations + initialiser were numerically
verified* (see companion validation), but it has not been executed end-to-end here
because Auto-07p/PyCoBi were not available in the authoring environment. Treat the
Auto constants (NMX, DSMAX, RL0/RL1, ...) as sensible starting values you will tune.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
from pycobi import ODESystem

# Path to your Auto-07p installation:
AUTO_DIR = "~/PycharmProjects/auto-07p"

# Guard against the 1/r singularity in the phi-equation and the OA boundary r=1.
_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
#  Reduced-model equation generator  (verified equal to the array RHS to 1e-16)
# ─────────────────────────────────────────────────────────────────────────────
def _ph(i):
    return "0" if i == 0 else f"phi_{i}"

def _diff(j, i):
    """String for (phi_j - phi_i); None when identically zero (cos->1, sin->0)."""
    if i == j:
        return None
    a, b = _ph(j), _ph(i)
    if b == "0":
        return a
    if a == "0":
        return f"(-{b})"
    return f"({a} - {b})"

def _cos(j, i):
    d = _diff(j, i); return "1" if d is None else f"cos({d})"

def _sin(j, i):
    d = _diff(j, i); return "0" if d is None else f"sin({d})"

def build_equations(M, plasticity="hebbian"):
    """
    Generate the reduced ODE system as a list of PyRates equation strings.

    State: r_0..r_{M-1}, phi_1..phi_{M-1}, A_i_j (i,j in 0..M-1).  phi_0 := 0.
    Coupling uses the K/M normalisation from the original script.
    """
    Mf = float(M)
    eqs = []

    def Scos(i):
        return " + ".join(f"A_{i}_{j}*r_{j}*{_cos(j, i)}" for j in range(M))

    def Ssin(i):
        terms = [f"A_{i}_{j}*r_{j}*{_sin(j, i)}" for j in range(M) if _sin(j, i) != "0"]
        return " + ".join(terms) if terms else "0"

    # dr_i/dt
    for i in range(M):
        eqs.append(
            f"d/dt * r_{i} = -delta*r_{i} "
            f"+ 0.5*(1 - r_{i}^2)*(K/{Mf})*({Scos(i)})"
        )
    # dphi_i/dt = (om_i - om_0) + vel_i - vel_0
    for i in range(1, M):
        vel_i = f"0.5*(1 + r_{i}^2)/r_{i}*(K/{Mf})*({Ssin(i)})"
        vel_0 = f"0.5*(1 + r_0^2)/r_0*(K/{Mf})*({Ssin(0)})"
        eqs.append(f"d/dt * phi_{i} = (om_{i} - om_0) + {vel_i} - {vel_0}")
    # dA_ij/dt
    for i in range(M):
        for j in range(M):
            if plasticity == "hebbian":
                drive = _cos(j, i)
            else:  # smooth anti-Hebbian surrogate
                s = _sin(j, i)
                drive = "0" if s == "0" else f"({s})^2"
            eqs.append(f"d/dt * A_{i}_{j} = mu*r_{i}*r_{j}*{drive} - g*A_{i}_{j}")
    return eqs


# ─────────────────────────────────────────────────────────────────────────────
#  Flexible M-population initialiser
#  (mirrors the RNG scheme of simulate() in kmo_macro_simulation.py)
# ─────────────────────────────────────────────────────────────────────────────
def initialize(
    M,
    omega_mean=0.40, omega_std=0.20,    # centre frequencies  ω_i ~ U(mean ± std)
    r0_mean=0.55,    r0_std=0.10,        # initial order-parameter magnitude r_i
    A0_scale=0.50,                       # std of initial (symmetric) weights; 0 -> zeros
    seed=42,
):
    """
    Draw per-population parameters and an initial state for an arbitrary number
    of ensembles M, exactly as in the original `simulate()` initialisation, but
    returning the *reduced* state ingredients used by build_circuit().

    Returns
    -------
    omega : (M,)    centre frequencies
    r0    : (M,)    initial order-parameter magnitudes in (0, 1)
    phi0  : (M,)    initial phase DIFFERENCES phi_i = psi_i - psi_0 (phi0[0] = 0)
    A0    : (M, M)  initial coupling-weight matrix (symmetric)

    Notes
    -----
    * r0_mean is kept comfortably away from 0: the reduced phi-equation contains a
      1/r_i term, so the IVP must start (and stay) at r_i > 0. Starting near r≈0.5
      reliably converges onto the phase-locked branch for the default parameters.
    * The continuation start needs a *stable locked equilibrium*; if you crank up
      omega_std (large detuning) or drop the effective coupling, the system may
      instead drift and the IVP will not settle. Increase K0 / A0_scale or reduce
      omega_std in that case.
    """
    rng = np.random.default_rng(seed)

    omega = rng.uniform(-omega_std, omega_std, M) + omega_mean
    r0    = np.clip(rng.uniform(-r0_std, r0_std, M) + r0_mean, _EPS, 1.0 - _EPS)

    # absolute phases (as in the original), then reduce to phase differences
    psi0 = rng.uniform(-np.pi, np.pi, M)
    phi0 = psi0 - psi0[0]
    phi0[0] = 0.0                         # phi_0 := 0 (removed by the reduction)

    if A0_scale > 0:
        A0 = rng.normal(0, A0_scale, (M, M))
        A0 = (A0 + A0.T) / 2              # start symmetric
    else:
        A0 = np.zeros((M, M))

    return omega, r0, phi0, A0


# ─────────────────────────────────────────────────────────────────────────────
#  Build the PyRates circuit
# ─────────────────────────────────────────────────────────────────────────────
def build_circuit(M, K, mu, gamma, omega, delta, r0, phi0, A0,
                  plasticity="hebbian"):
    """Assemble a single-node PyRates CircuitTemplate for the reduced system."""
    eqs = build_equations(M, plasticity)

    variables = {}
    # state variables (initial values go inside the type spec, e.g. 'variable(0.5)');
    # exactly one variable must be declared 'output'.
    variables["r_0"] = f"output({float(r0[0])})"
    for i in range(1, M):
        variables[f"r_{i}"] = f"variable({float(r0[i])})"
    for i in range(1, M):
        variables[f"phi_{i}"] = f"variable({float(phi0[i])})"
    for i in range(M):
        for j in range(M):
            variables[f"A_{i}_{j}"] = f"variable({float(A0[i, j])})"

    # parameters (plain numbers -> emitted as Auto PAR(), continuable by name)
    variables["K"] = float(K)
    variables["mu"] = float(mu)
    variables["g"] = float(gamma)
    variables[f"delta"] = float(delta)
    for i in range(M):
        variables[f"om_{i}"] = float(omega[i])

    op = OperatorTemplate(name="kmo_op", equations=eqs, variables=variables)
    node = NodeTemplate(name="kmo_node", operators=[op])
    circuit = CircuitTemplate(name="kmo", nodes={"p": node})
    return circuit


# ─────────────────────────────────────────────────────────────────────────────
#  Continuation pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ───── choose the model here ─────────────────────────────────────────────
    # Pick ANY number of ensembles. M=2 or 3 is recommended for the full pipeline
    # (limit cycles + codim-2); larger M works for the equilibrium sweeps but the
    # periodic-orbit steps get expensive (reduced dim = 2M - 1 + M^2).
    M  = 10
    K0 = 1.0
    mu, gamma = 0.1, 0.05          # weight bound |A*| <= mu/gamma = 3.0
    delta = 0.02
    plasticity = "hebbian"
    p1 = "g"
    p1_title = r"weight decay $\gamma$"
    p2 = "K"
    p2_title = r"global coupling $K$"
    p2_min, p2_max = 0.01, 2.0

    # statistical initialisation (same knobs as the original simulate())
    omega, r0, phi0, A0 = initialize(
        M,
        omega_mean=0.40, omega_std=0.20,
        r0_mean=0.9,    r0_std=0.10,
        A0_scale=0.5,
        seed=42,
    )

    print(f"OA adaptive continuation  —  M={M} populations  (reduced dim "
          f"= {2*M - 1 + M*M})")
    print(f"  mu={mu}, gamma={gamma}")
    print(f"  omega in [{omega.min():.3f}, {omega.max():.3f}]   "
          f"delta={delta}")
    print(f"  r0 in [{r0.min():.3f}, {r0.max():.3f}]")

    circuit = build_circuit(M, K0, mu, gamma, omega, delta, r0, phi0, A0,
                            plasticity=plasticity)

    # from_template generates the Fortran for Auto; analytical Jacobian by default.
    ode = ODESystem.from_template(circuit, auto_dir=AUTO_DIR, init_cont=False)

    # ── Step 1: integrate in time to the steady (relative-equilibrium) state ──
    # PAR(14) holds time in Auto's 'ivp' constants set; we stop at t = 2000.
    t_sols, t_cont = ode.run(
        c="ivp", name="time",
        DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=200000,
        EPSL=1e-8, EPSU=1e-8, EPSS=1e-6,
        UZR={14: 5000.0}, STOP={"UZ1"},
    )
    # sanity check: plot the approach to equilibrium
    ode.plot_continuation("t", "r_0", cont="time")
    plt.title("IVP: convergence to the phase-locked relative equilibrium")
    plt.savefig("kmo_ivp.png", dpi=130, bbox_inches="tight")
    # plt.close()

    # ── Step 2: 1-D equilibrium continuation in the global coupling K ─────────
    # IPS=1 equilibrium, ILP=1 fold detection, ISP=2 full bifurcation detection.
    k_sols, k_cont = ode.run(
        origin=t_cont, starting_point="UZ1", name=f"{p1}_cont", bidirectional=True,
        ICP=p1, RL0=0.01, RL1=1.0,
        IPS=1, ILP=1, ISP=2, ISW=1,
        NMX=6000, NPR=10,
        DS=1e-3, DSMIN=1e-10, DSMAX=5e-2,
        EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, UZR={p1: [0.16]}, STOP={}
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ode.plot_continuation(p1, "r_0", cont=f"{p1}_cont", ax=ax)
    ax.set_xlabel(p1_title); ax.set_ylabel(r"$r_0$ at equilibrium")
    ax.set_title(f"1-D continuation in {p1}  (LP = fold, HB = Hopf)")
    plt.savefig(f"kmo_cont_{p1}.png", dpi=130, bbox_inches="tight")
    # plt.close()

    # ── Step 3: 1-D continuation in the frequency detuning (om_1) ────────────
    # Loss of phase locking typically appears here as a fold / SNIC: the locked
    # equilibrium collides and disappears, beyond which populations drift.
    # (om_1 exists for any M >= 2; continue any om_i you like.)
    d_sols, d_cont = ode.run(
        origin=k_cont, starting_point="UZ1", name=f"{p2}_cont", bidirectional=True,
        ICP=p2, RL0=p2_min, RL1=p2_max,
        IPS=1, ILP=1, ISP=2, ISW=1,
        NMX=6000, NPR=10,
        DS=1e-3, DSMIN=1e-10, DSMAX=5e-2,
        EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ode.plot_continuation(p2, "r_0", cont=f"{p2}_cont", ax=ax)
    ax.set_xlabel(p2_title); ax.set_ylabel(r"$r_0$")
    ax.set_title(f"1-D continuation in {p2_title}")
    plt.savefig(f"kmo_cont_{p2}.png", dpi=130, bbox_inches="tight")

    # ── Step 4: branch-switch to limit cycles at the first Hopf (if present) ──
    # "Breathing" / modulated-locking states are periodic orbits of the reduced
    # system. IPS=2 = periodic-orbit continuation, ISW=-1 = switch branch at HB1.
    try:
        lc_sols, lc_cont = ode.run(
            origin=k_cont, starting_point="HB1", name=f"{p1}_lc_cont",
            IPS=2, ISP=2, ISW=-1, ILP=0,
            NTST=200, NCOL=4,
            NMX=4000, NPR=10,
            DS=1e-3, DSMIN=1e-9, DSMAX=2e-2,
            STOP={"BP2", "LP5"},
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ode.plot_continuation("K", "r_0", cont=f"{p1}_cont", ax=ax)
        ode.plot_continuation("K", "r_0", cont=f"{p1}_lc_cont", ax=ax,
                              ignore=["UZ", "BP"], line_color_stable="green")
        ax.set_title(f"Equilibria + limit-cycle envelope (min/max of $r_0$) vs {p1}")
        plt.savefig(f"kmo_limitcycle_{p1}.png", dpi=130, bbox_inches="tight")
        # plt.close()
    except Exception as e:
        print(f"[Step 4] No Hopf branch switched (HB1 may not exist here): {e}")

    # ── Step 5: 2-parameter (codim-2) continuation of a fold in (K, om_1) ────
    # ISW=2 continues the located fold (LP1) as a curve in two parameters,
    # tracing the boundary of the phase-locking region in parameter space.
    try:
        lp2_sols, lp2_cont = ode.run(
            origin=k_cont, starting_point="LP1", name="fold_2p", bidirectional=True,
            ICP=[p2, p1], RL0=p2_min, RL1=p2_max,
            IPS=1, ISW=2, ISP=2, ILP=0,
            NMX=8000, NPR=50,
            DS=1e-3, DSMIN=1e-8, DSMAX=5e-2,
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        ode.plot_continuation(p1, p2, cont="fold_2p", ax=ax)
        ax.set_xlabel(p1_title); ax.set_ylabel(p2_title)
        ax.set_title("Codim-2: fold curve = boundary of phase-locked region")
        plt.savefig("kmo_codim2.png", dpi=130, bbox_inches="tight")
        # plt.close()
    except Exception as e:
        print(f"[Step 5] Fold continuation skipped (LP1 may not exist here): {e}")

    # ---- tidy up Auto/PyRates temp files ----
    ode.close_session(clear_files=True)
    clear(circuit)
    print("Done. Figures: kmo_ivp.png, kmo_cont_K.png, kmo_cont_detune.png, "
          "kmo_limitcycle_K.png, kmo_codim2.png")
    plt.show()

if __name__ == "__main__":
    main()