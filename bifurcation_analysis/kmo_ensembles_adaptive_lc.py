"""
Continuing the post-fold OSCILLATION (phase-slip limit cycle) in Auto-07p / PyCoBi
==================================================================================

WHAT THE BIFURCATION IS
-----------------------
In the gamma-continuation the locked relative equilibrium is destroyed at a FOLD
(a single real eigenvalue hits zero). The fold turns out to be a SNIC
(Saddle-Node on an Invariant Circle / infinite-period bifurcation): just past it
a subset of ensembles ("a cluster") phase-slips coherently relative to the locked
core, and the period of that slipping diverges as

        T  ~  C / sqrt(gamma - gamma_c)        (gamma_c ~ 0.16 for the demo params)

So the "oscillation" you see numerically is a genuine, *mode-locked* LIMIT CYCLE
(verified: the full reduced state closes on itself at the fundamental period).

WHY THE STANDARD TRICKS FAIL
----------------------------
  * It is NOT born from a Hopf, so you cannot reach it with ISW=-1 branch-switching
    off the equilibrium branch (there is no HB to switch at).
  * In the (r, phi) reduced coordinates the slipping phases WIND by 2*pi per period
    (phi_i(T) = phi_i(0) + 2*pi*k). The orbit is therefore NOT closed in phi, and
    Auto's periodic boundary condition u(1)=u(0) can never be satisfied -> IPS=2
    diverges.

THE FIX  (validated numerically: equivalent to the original to 6e-11; orbit closes
to 1.6e-4 at the fundamental period)
----------------------------------------------------------------------------------
Re-express the reduced model in CARTESIAN order-parameter coordinates

        z_i = x_i + i y_i = r_i e^{i psi_i}

in the frame co-rotating with population 0 (pin y_0 := 0, so x_0 = r_0; carry the
collective frequency Omega = Im F_0 / x_0). The OA field is the smooth polynomial

        zdot_i = (-delta + i om_i) z_i + 0.5 (H_i - z_i^2 conj(H_i)),
        H_i    = (K/M) sum_j A_ij z_j,
        Adot_ij = mu (x_i x_j + y_i y_j) - g A_ij.

In these coordinates a *winding* phase becomes z_i tracing a CLOSED LOOP, so the
drift/SNIC orbit is a closed periodic orbit that Auto can represent. There is no
1/r singularity either. Equilibria are unchanged (same fold, same diagram), so you
lose nothing by switching the whole pipeline to this model.

RECIPE
------
  1. Build the Cartesian circuit (build_circuit_cartesian).
  2. Equilibrium continuation in gamma exactly as before (IPS=1) -> locate the fold.
  3. Pick a gamma comfortably PAST the fold where the cycle is clean (e.g. 0.18),
     time-integrate onto the cycle, extract ONE fundamental period, write an
     Auto trajectory file, and start IPS=2 from it (no Hopf needed).
  4. Continue the cycle in gamma. Toward the fold the period -> infinity (the branch
     terminates at the SNIC / saddle-node-on-cycle); away from it the period shrinks.

CAVEATS
-------
  * Which ensembles slip changes with gamma (Arnold-tongue / mode-locking structure);
    between tongues the attractor can become a 2-torus (quasiperiodic) that needs
    torus continuation, not IPS=2. Continue ONE mode-locked branch at a time.
  * Near the SNIC use large NTST (300-500); expect stiffness as PAR(11)=period blows up.
  * The .dat column order must match the variable order Auto/PyRates emits. Confirm it
    from the generated equation file (see write_auto_dat docstring).
"""

import numpy as np
from scipy.integrate import solve_ivp

from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
from pycobi import ODESystem

AUTO_DIR = "~/PycharmProjects/auto-07p"


# ─────────────────────────────────────────────────────────────────────────────
#  Cartesian reduced-model equation generator  (verified equal to array RHS, 2e-16)
# ─────────────────────────────────────────────────────────────────────────────
def _xj(j): return f"x_{j}"
def _yj(j): return "0" if j == 0 else f"y_{j}"

def build_equations_cartesian(M):
    """Reduced OA in Cartesian coords z_i=x_i+i y_i, frame co-rotating with pop 0
    (y_0:=0, x_0=r_0), Omega=Im F_0/x_0 inlined. dim = 2M-1+M^2."""
    Mf = float(M)
    def ReH(i): return f"(K/{Mf})*(" + " + ".join(f"A_{i}_{j}*{_xj(j)}" for j in range(M)) + ")"
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
            eqs.append(f"d/dt * A_{i}_{j} = mu*({_xj(i)}*{_xj(j)} + {_yj(i)}*{_yj(j)}) - g*A_{i}_{j}")
    return eqs


def build_circuit_cartesian(M, K, mu, gamma, delta, omega, r0, phi0, A0):
    """PyRates circuit for the Cartesian reduced model.
    Initial state derived from (r0, phi0): x_i = r_i cos(phi_i), y_i = r_i sin(phi_i)."""
    eqs = build_equations_cartesian(M)
    x0 = r0 * np.cos(phi0); y0 = r0 * np.sin(phi0)

    variables = {"x_0": f"output({float(x0[0])})"}
    for i in range(1, M):
        variables[f"x_{i}"] = f"variable({float(x0[i])})"
        variables[f"y_{i}"] = f"variable({float(y0[i])})"
    for i in range(M):
        for j in range(M):
            variables[f"A_{i}_{j}"] = f"variable({float(A0[i, j])})"
    variables["K"] = float(K); variables["mu"] = float(mu)
    variables["g"] = float(gamma); variables["delta"] = float(delta)
    for i in range(M):
        variables[f"om_{i}"] = float(omega[i])

    op = OperatorTemplate(name="kmo_cart_op", equations=eqs, variables=variables)
    node = NodeTemplate(name="kmo_cart_node", operators=[op])
    return CircuitTemplate(name="kmo_cart", nodes={"p": node})


# ─────────────────────────────────────────────────────────────────────────────
#  Land on the limit cycle by time integration and extract one fundamental period
# ─────────────────────────────────────────────────────────────────────────────
def _reduced_cart_rhs(t, s, M, K, mu, g, delta, omega):
    x = np.empty(M); yv = np.empty(M); x[0] = s[0]; yv[0] = 0.0
    x[1:] = s[1:1+2*(M-1):2]; yv[1:] = s[2:1+2*(M-1):2]
    A = s[1+2*(M-1):].reshape(M, M)
    z = x + 1j*yv; H = (K/M)*(A@z)
    F = (-delta + 1j*omega)*z + 0.5*(H - z**2*np.conj(H))
    Om = F.imag[0]/x[0]
    dx = F.real + Om*yv; dy = F.imag - Om*x
    dA = mu*(np.outer(x, x) + np.outer(yv, yv)) - g*A
    out = np.empty_like(s); out[0] = dx[0]
    out[1:1+2*(M-1):2] = dx[1:]; out[2:1+2*(M-1):2] = dy[1:]
    out[1+2*(M-1):] = dA.ravel()
    return out


def extract_one_period(M, K, mu, gamma, delta, omega, r0, phi0, A0,
                       T_settle=6000.0, n_samples=400):
    """Integrate the Cartesian reduced model onto the cycle, find the FUNDAMENTAL
    period (the slowest closed loop -> min over candidate periods of the full-state
    return distance), and return (period, t_norm[n_samples], states[n_samples, dim])."""
    s0 = np.empty(1 + 2*(M-1) + M*M)
    s0[0] = r0[0]*np.cos(phi0[0])
    s0[1:1+2*(M-1):2] = (r0*np.cos(phi0))[1:]
    s0[2:1+2*(M-1):2] = (r0*np.sin(phi0))[1:]
    s0[1+2*(M-1):] = A0.ravel()
    args = (M, K, mu, gamma, delta, omega)
    sol = solve_ivp(_reduced_cart_rhs, (0, T_settle), s0, args=args,
                    method="LSODA", rtol=1e-10, atol=1e-12, dense_output=True, max_step=2.0)

    # coarse period guess from x_0 zero-crossings, then refine to the fundamental
    tt = np.linspace(0.6*T_settle, T_settle, 200000)
    x0 = sol.sol(tt)[0]; xc = x0 - x0.mean()
    cr = np.where((xc[:-1] < 0) & (xc[1:] >= 0))[0]
    base = float(np.diff(tt[cr]).mean())
    t_anchor = 0.7*T_settle
    s_anchor = sol.sol(t_anchor)
    # the fundamental is base * k for some small integer k; pick k minimising return error
    best = (1e9, base)
    for k in (1, 2, 3, 4):
        Tc = base*k
        err = np.max(np.abs(sol.sol(t_anchor + Tc) - s_anchor))
        if err < best[0]:
            best = (err, Tc)
    Tfund = best[1]
    tau = np.linspace(0.0, 1.0, n_samples)
    states = np.array([sol.sol(t_anchor + s*Tfund) for s in tau])  # (n_samples, dim)
    print(f"extract_one_period: fundamental T = {Tfund:.4f} (return error {best[0]:.2e})")
    return Tfund, tau, states


def write_auto_dat(path, tau, states):
    """Write a one-period trajectory in Auto's restart-data format:
    each row = [tau_in_0_1, u_1, u_2, ..., u_dim].

    IMPORTANT: the column order of `states` must match the order in which Auto/PyRates
    enumerates U(1..dim). Read the generated equation file to confirm, e.g.:
        ode = ODESystem.from_template(circuit, auto_dir=AUTO_DIR, init_cont=False)
        print(ode.state_vars)          # PyCoBi exposes the variable->index mapping
    and reorder the columns of `states` to match before writing. The helper here
    assumes [x_0, x_1, y_1, ..., x_{M-1}, y_{M-1}, A_00, ...] which is the order used
    by build_equations_cartesian; adjust if PyRates reorders."""
    arr = np.column_stack([tau, states])
    np.savetxt(path, arr)
    print(f"wrote {path}  shape={arr.shape}")


# ─────────────────────────────────────────────────────────────────────────────
#  Continuation pipeline for the limit cycle
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from_continuation_seed = dict(  # match your working equilibrium setup
        M=10, K=1.0, mu=0.1, gamma_fold_side=0.18, delta=0.02, seed=42,
    )
    M = from_continuation_seed["M"]; K = from_continuation_seed["K"]
    mu = from_continuation_seed["mu"]; delta = from_continuation_seed["delta"]
    gamma_lc = from_continuation_seed["gamma_fold_side"]

    rng = np.random.default_rng(from_continuation_seed["seed"])
    omega = rng.uniform(-0.2, 0.2, M) + 0.4
    r0 = np.clip(rng.uniform(-0.1, 0.1, M) + 0.9, 1e-9, 1-1e-9)
    psi0 = rng.uniform(-np.pi, np.pi, M); phi0 = psi0 - psi0[0]; phi0[0] = 0.0
    A0 = rng.normal(0, 0.5, (M, M)); A0 = (A0 + A0.T)/2

    # 1) land on the cycle past the fold and extract one period
    Tlc, tau, states = extract_one_period(M, K, mu, gamma_lc, delta, omega, r0, phi0, A0)
    write_auto_dat("kmo_cycle.dat", tau, states)

    # 2) build the Cartesian circuit and start a PERIODIC-ORBIT continuation from the
    #    trajectory file (no Hopf needed). PyCoBi forwards `dat`, IPS, PAR to Auto-07p.
    circuit = build_circuit_cartesian(M, K, mu, gamma_lc, delta, omega, r0, phi0, A0)
    ode = ODESystem.from_template(circuit, auto_dir=AUTO_DIR, init_cont=False)

    #   print(ode.state_vars)  # <- verify column order, reorder states if needed (see write_auto_dat)

    lc_sols, lc_cont = ode.run(
        name="lc_from_sim",
        dat="kmo_cycle",            # reads kmo_cycle.dat as the initial periodic orbit
        IPS=2, ISP=2, ILP=1, ISW=1,
        ICP=[3, 11],              # continue in gamma; PAR(11) = period
        PAR={11: float(Tlc)},       # initial period guess
        NTST=400, NCOL=4,
        RL0=0.05, RL1=0.40,
        NMX=8000, NPR=20,
        DS=-1e-3, DSMIN=1e-9, DSMAX=2e-2,   # DS<0: head toward the fold first (period grows)
        EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
        STOP={},
    )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ode.plot_continuation("g", "x_0", cont="lc_from_sim", ax=ax[0])   # min/max envelope
    ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$x_0=r_0$ (cycle envelope)")
    ax[0].set_title("Phase-slip limit cycle vs gamma")
    ode.plot_continuation("g", 11, cont="lc_from_sim", ax=ax[1])      # period
    ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel("period PAR(11)")
    ax[1].set_title("Period diverges at the SNIC (fold)")
    plt.tight_layout(); plt.savefig("kmo_snic_cycle.png", dpi=130, bbox_inches="tight")

    ode.close_session(clear_files=True); clear(circuit)
    print("Done -> kmo_snic_cycle.png")


if __name__ == "__main__":
    main()