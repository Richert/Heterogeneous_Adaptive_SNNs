r"""
Bifurcation analysis of the Allen-fitted QIF threshold-heterogeneity mean field
================================================================================

Loads the Lorentzian-mixture fit of the Allen excitability gap v_θ − v_r
(``data_fitting/allen_lorentzian_fit.py``) and runs a PyCoBi/Auto-07p bifurcation
analysis of the Gast-Solla-Kennedy 2023 mean field (b=κ=0 ⇒ QIF, C=k=1, additive
current Js; see ``qif_simulations/allen_qif_meanfield.py``). For each Lorentzian
component m (weight w_m, centre Ω_m, width Δ_m → mean threshold v̄_{θ,m}=v_r+Ω_m):

    ṙ_m = (Δ_m/π)(v_m − v_r) + r_m (2 v_m − v_r − v̄_{θ,m})
    v̇_m = (v_m − v_r)(v_m − v̄_{θ,m}) − (π r_m)² − π Δ_m r_m + I + J s
    ȧ   = (Σ_m w_m r_m − a)/τ_s ,   ṡ = (a − s)/τ_s     (alpha synapse, shared)

State = [r_0..r_{M−1}, v_0..v_{M−1}, a, s] (dim 2M+2); v_r, τ_s, Δ_m, v̄_{θ,m}, w_m
are inlined constants, the external input I (``Iext``) and the global coupling J are
the continuable Auto parameters.

Pipeline:
  1. settle the IVP onto a stable equilibrium (low-activity state),
  2. continue it in I → locate fold (LP) / Hopf (HB) bifurcations (J chosen large
     enough that recurrent excitation makes them appear),
  3. continue every codim-1 LP / HB in the J–I plane (codim-2).

Run in the ``pycobi`` (or ``allen``) conda env with Auto-07p / meson on PATH:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python allen_qif_bifurcation.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
from pycobi import ODESystem

_HERE = os.path.dirname(os.path.abspath(__file__))
AUTO_DIR = "~/PycharmProjects/auto-07p"

C_EQ = "#1F77B4"
C_FOLD = "k"
C_HOPF = "#D62728"
M_FOLD, M_HOPF = "o", "s"


# ═════════════════════════════════════════════════════════════════════════════
#  configuration
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    fit_npz=os.path.join(_HERE, "..", "data_fitting", "allen_lorentzian_pyramidal_L23.npz"),
    v_r=-70.0,
    tau_s=10.0,
    J0=100.0,                # coupling for the 1-parameter I-continuation (large ⇒ folds: a
                             # bistable region with two LPs / a cusp; none survive below J≈50)
    I0=300.0,                # input at which to settle the IVP onto a stable equilibrium
    I_min=-200.0, I_max=700.0,   # I-continuation bounds
    J_min=0.0, J_max=200.0,      # J bounds for the codim-2 (J–I) continuation
    r0=0.05,                 # IVP initial firing rate per ensemble (low-activity guess)
    T_settle=2000.0,
)


# ═════════════════════════════════════════════════════════════════════════════
#  model generator
# ═════════════════════════════════════════════════════════════════════════════
def state_var_order(M):
    return [f"r_{i}" for i in range(M)] + [f"v_{i}" for i in range(M)] + ["a", "s"]


def build_equations(M, vthbar, delta, weights, v_r, tau_s):
    PI = repr(float(np.pi))
    vr = repr(float(v_r))
    ts = repr(float(tau_s))
    rtot = "(" + " + ".join(f"{float(weights[j])}*r_{j}" for j in range(M)) + ")"
    eqs = []
    for i in range(M):
        d, vt = repr(float(delta[i])), repr(float(vthbar[i]))
        eqs.append(f"d/dt * r_{i} = {d}/{PI}*(v_{i} - {vr}) + r_{i}*(2*v_{i} - {vr} - {vt})")
        eqs.append(f"d/dt * v_{i} = (v_{i} - {vr})*(v_{i} - {vt}) - ({PI}*r_{i})^2 "
                   f"- {PI}*{d}*r_{i} + Iext + J*s")
    eqs.append(f"d/dt * a = ({rtot} - a)/{ts}")
    eqs.append(f"d/dt * s = (a - s)/{ts}")
    return eqs


def build_circuit(M, vthbar, delta, weights, v_r, tau_s, J0, I0, r0, v0, name="qif_ens"):
    eqs = build_equations(M, vthbar, delta, weights, v_r, tau_s)
    variables = {"r_0": f"output({float(r0[0])})"}
    for i in range(1, M):
        variables[f"r_{i}"] = f"variable({float(r0[i])})"
    for i in range(M):
        variables[f"v_{i}"] = f"variable({float(v0[i])})"
    variables["a"] = f"variable({float(r0.mean())})"
    variables["s"] = f"variable({float(r0.mean())})"
    variables["Iext"] = float(I0)
    variables["J"] = float(J0)
    op = OperatorTemplate(name=f"{name}_op", equations=eqs, variables=variables)
    node = NodeTemplate(name=f"{name}_node", operators=[op])
    return CircuitTemplate(name=name, nodes={"p": node})


# ═════════════════════════════════════════════════════════════════════════════
#  helpers
# ═════════════════════════════════════════════════════════════════════════════
def load_fit(path):
    d = np.load(path, allow_pickle=False)
    w = np.asarray(d["weights"], float)
    return w / w.sum(), np.asarray(d["omega"], float), np.asarray(d["delta"], float), int(d["M"])


def _pcol(df, name):
    for c in df.columns:
        head = c[0] if isinstance(c, tuple) else c
        sub = c[1] if isinstance(c, tuple) and len(c) > 1 else ""
        if (head == name or (isinstance(head, str) and head.endswith("/" + name))) and sub in ("", 0):
            return c
    raise KeyError(name)


def _bif_vals(sols, label, pname):
    rows = sols[sols[("bifurcation", "")] == label]
    return sorted(float(v) for v in rows[_pcol(sols, pname)]) if len(rows) else []


def add_markers(ax, sols, label, marker, xname, yname, color):
    bif = sols[("bifurcation", "")].to_numpy()
    idx = np.where(bif == label)[0]
    if idx.size == 0:
        return
    xc, yc = _pcol(sols, xname), _pcol(sols, yname)
    ax.scatter([float(sols[xc].iloc[i]) for i in idx],
               [float(sols[yc].iloc[i]) for i in idx],
               marker=marker, s=70, facecolors="none", edgecolors=color,
               linewidths=1.6, zorder=12)


# ═════════════════════════════════════════════════════════════════════════════
#  main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    w, Omega, Delta, M = load_fit(cfg["fit_npz"])
    v_r = cfg["v_r"]
    vthbar = v_r + Omega
    DIM = 2 * M + 2
    print(f"Allen-QIF mean-field bifurcation — M={M} (dim {DIM})")
    print(f"  v̄_θ,m = {np.round(vthbar, 2)}   Δ_m = {np.round(Delta, 3)}   w_m = {np.round(w, 3)}")
    print(f"  J0={cfg['J0']}, settle at I={cfg['I0']}")

    r0 = np.full(M, cfg["r0"])
    v0 = np.full(M, v_r)
    circuit = build_circuit(M, vthbar, Delta, w, v_r, cfg["tau_s"], cfg["J0"], cfg["I0"], r0, v0)
    ode = ODESystem.from_template(circuit, auto_dir=AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))

    # ── Step 1: IVP → stable equilibrium ────────────────────────────────────
    print(f"\n[1] IVP → stable equilibrium (settle T={cfg['T_settle']})")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6,
            UZR={14: cfg["T_settle"]}, STOP={"UZ1"})

    # ── Step 2: equilibrium continuation in I → fold / Hopf ─────────────────
    print(f"[2] equilibrium continuation in I  (J={cfg['J0']})")
    eq_sols, _ = ode.run(
        origin="time", starting_point="UZ1", name="eq_I", c="eq",
        ICP="Iext", bidirectional=True, RL0=cfg["I_min"], RL1=cfg["I_max"],
        IPS=1, ILP=1, ISP=2, ISW=1, NMX=50000, NPR=50,
        DS=1e-2, DSMIN=1e-8, DSMAX=0.1, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
        get_stability=True)
    # NB: get_stability relies on the analytical Jacobian (analytical_jacobian=True in
    # from_template) + DENSE output (small NPR / DSMAX) so Auto emits an eigenvalue line
    # at (almost) every point — otherwise the saddle branch between the folds is mislabeled.
    print(f"  branch points: {len(eq_sols)}")
    print("  bifurcations:", dict(eq_sols["bifurcation"].value_counts()))
    folds = _bif_vals(eq_sols, "LP", "Iext")
    hopfs = _bif_vals(eq_sols, "HB", "Iext")
    print(f"  fold(s) LP at I = {[round(x, 2) for x in folds]}")
    print(f"  Hopf(s) HB at I = {[round(x, 2) for x in hopfs]}")

    # ── Step 3: codim-2 continuation of each LP / HB in the J–I plane ───────
    print("[3] codim-2 continuation in (I, J)")
    n_lp = len(eq_sols[eq_sols[("bifurcation", "")] == "LP"])
    n_hb = len(eq_sols[eq_sols[("bifurcation", "")] == "HB"])
    codim2 = []
    for k in range(1, n_lp + 1):
        nm = f"fold_{k}"
        try:
            s, _ = ode.run(origin="eq_I", starting_point=f"LP{k}", name=nm, c="eq",
                           ICP=["Iext", "J"], bidirectional=True, IPS=1, ISW=2, ISP=2, ILP=0,
                           RL0=cfg["I_min"], RL1=cfg["I_max"], NMX=20000, NPR=10,
                           DS=1e-2, DSMIN=1e-8, DSMAX=0.25, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                           UZSTOP={"J": [cfg["J_min"], cfg["J_max"]]})
            codim2.append((nm, "fold", s))
            print(f"  {nm}: fold curve, {len(s)} pts")
        except Exception as e:
            print(f"  {nm}: FAILED ({type(e).__name__}: {e})")
    for k in range(1, n_hb + 1):
        nm = f"hopf_{k}"
        try:
            s, _ = ode.run(origin="eq_I", starting_point=f"HB{k}", name=nm, c="eq",
                           ICP=["Iext", "J"], bidirectional=True, IPS=1, ISW=2, ISP=2, ILP=0,
                           RL0=cfg["I_min"], RL1=cfg["I_max"], NMX=20000, NPR=10,
                           DS=1e-2, DSMIN=1e-8, DSMAX=0.25, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                           UZSTOP={"J": [cfg["J_min"], cfg["J_max"]]})
            codim2.append((nm, "hopf", s))
            print(f"  {nm}: Hopf curve, {len(s)} pts")
        except Exception as e:
            print(f"  {nm}: FAILED ({type(e).__name__}: {e})")

    # ── figure ──────────────────────────────────────────────────────────────
    print("\n[plot] writing allen_qif_bifurcation.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) 1-parameter branch: synaptic activation s (= total rate at equilibrium) vs I
    ode.plot_continuation("Iext", "s", cont="eq_I", ax=ax1,
                          line_color_stable=C_EQ, line_color_unstable=C_EQ,
                          line_style_stable="solid", line_style_unstable="dashed",
                          ignore=["UZ", "BP", "EP"], bifurcation_legend=False, linewidths=2.0)
    add_markers(ax1, eq_sols, "LP", M_FOLD, "Iext", "s", C_FOLD)
    add_markers(ax1, eq_sols, "HB", M_HOPF, "Iext", "s", C_HOPF)
    ax1.set_xlabel(r"external input $I$")
    ax1.set_ylabel(r"synaptic activation $s$  ($=\sum_m w_m r_m$ at equilibrium)")
    ax1.set_title(f"(a)  equilibrium branch  (J={cfg['J0']:g}, M={M})")
    ax1.legend(handles=[
        Line2D([0], [0], color=C_EQ, lw=2, label="equilibrium (solid=stable, dashed=unstable)"),
        Line2D([0], [0], marker=M_FOLD, color=C_FOLD, lw=0, markerfacecolor="none", label="fold (LP)"),
        Line2D([0], [0], marker=M_HOPF, color=C_HOPF, lw=0, markerfacecolor="none", label="Hopf (HB)"),
    ], loc="best", fontsize=8)

    # (b) codim-2: fold/Hopf loci in the J–I plane
    for nm, kind, s in codim2:
        col = C_FOLD if kind == "fold" else C_HOPF
        xc, yc = _pcol(s, "Iext"), _pcol(s, "J")
        ax2.plot(s[xc].to_numpy(float), s[yc].to_numpy(float), color=col, lw=1.8)
    ax2.axhline(cfg["J0"], color="0.6", ls=":", lw=1.0)
    ax2.set_xlabel(r"external input $I$")
    ax2.set_ylabel(r"coupling $J$")
    ax2.set_title("(b)  codim-2 bifurcation curves in the $J$–$I$ plane")
    ax2.set_xlim(cfg["I_min"], cfg["I_max"])
    ax2.set_ylim(cfg["J_min"], cfg["J_max"])
    ax2.legend(handles=[
        Line2D([0], [0], color=C_FOLD, lw=1.8, label="fold (LP)"),
        Line2D([0], [0], color=C_HOPF, lw=1.8, label="Hopf (HB)"),
        Line2D([0], [0], color="0.6", ls=":", lw=1.0, label=f"J={cfg['J0']:g} (panel a)"),
    ], loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(_HERE, "allen_qif_bifurcation.png"), dpi=140, bbox_inches="tight")
    plt.savefig(os.path.join(_HERE, "allen_qif_bifurcation.pdf"), bbox_inches="tight")
    print("  saved allen_qif_bifurcation.{png,pdf}")

    ode.close_session(clear_files=True)
    clear(circuit)
    print("Done.")


if __name__ == "__main__":
    main()
