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
import sys

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
    # which Allen fit to analyse: selects ../data_fitting/allen_lorentzian_<tag>.npz and names
    # the output figure allen_qif_bifurcation_<tag>.{png,pdf} (same tag scheme as the fits)
    cell_class="PV+ Interneuron",      # "Pyramidal" | "PV+ interneuron" | "SOM interneuron"
    layer="L2/3",                # "L2/3" | "L5/6"
    v_r=-70.0,
    tau_s=8.0,
    J0=-100.0,                # coupling for the 1-parameter I-continuation (large ⇒ folds: a
                             # bistable region with two LPs / a cusp; none survive below J≈50)
    I0=0.0,                # input at which to settle the IVP onto a stable equilibrium
    I_min=-100.0, I_max=2000.0,   # I-continuation bounds
    J_min=-200.0, J_max=0.0,      # J bounds for the codim-2 (J–I) continuation
    r0=0.05,                 # IVP initial firing rate per ensemble (low-activity guess)
    T_settle=2000.0,
)
# CLI args override the cell class / layer -> `python allen_qif_bifurcation.py "Pyramidal" "L5/6"`
if len(sys.argv) > 2:
    CONFIG["cell_class"], CONFIG["layer"] = sys.argv[1], sys.argv[2]


# ═════════════════════════════════════════════════════════════════════════════
#  model generator
# ═════════════════════════════════════════════════════════════════════════════
def state_var_order(M):
    return [f"r_{i}" for i in range(M)] + [f"v_{i}" for i in range(M)] + ["a", "s"]


def build_equations(M, omega, delta, weights, v_r, tau_s, combined=False):
    """Mean-field equations with GLOBAL heterogeneity knobs (= 1 at the data fit).
      two-knob (default): hD width scaling Δ_m = hD·Δ_m^0;  hC centre-spread
                          Ω_m = Ω̄ + hC·(Ω_m^0 − Ω̄),  Ω̄ = Σ w_m Ω_m^0  (weighted mean).
      combined=True: a SINGLE knob `h` drives BOTH (hD = hC = h), so h→0 collapses the mixture
                     to one delta at Ω̄ (homogeneous), h=1 = data, h>1 = more heterogeneous.
    v̄_θ,m = v_r + Ω_m. The knob(s) are Auto-continuable parameters (alongside Iext, J)."""
    PI = repr(float(np.pi))
    vr = repr(float(v_r))
    Ombar = repr(float(np.asarray(weights, float) @ np.asarray(omega, float)))
    rtot = "(" + " + ".join(f"{float(weights[j])}*r_{j}" for j in range(M)) + ")"
    hw, hc = ("h", "h") if combined else ("hD", "hC")            # width / centre-spread knob names
    eqs = []
    for i in range(M):
        d0, om0 = repr(float(delta[i])), repr(float(omega[i]))
        de = f"({hw}*{d0})"                                          # effective Δ_m
        vt = f"({vr} + {Ombar} + {hc}*({om0} - {Ombar}))"          # effective v̄_θ,m
        eqs.append(f"d/dt * r_{i} = {de}/{PI}*(v_{i} - {vr}) + r_{i}*(2*v_{i} - {vr} - {vt})")
        eqs.append(f"d/dt * v_{i} = (v_{i} - {vr})*(v_{i} - {vt}) - ({PI}*r_{i})^2 "
                   f"- {PI}*{de}*r_{i} + Iext + J*s")
    eqs.append(f"d/dt * a = ({rtot} - a)/tau_s")               # tau_s is an Auto-continuable parameter
    eqs.append(f"d/dt * s = (a - s)/tau_s")
    return eqs


def build_circuit(M, omega, delta, weights, v_r, tau_s, J0, I0, r0, v0,
                  hD0=1.0, hC0=1.0, combined=False, h0=1.0, name="qif_ens"):
    eqs = build_equations(M, omega, delta, weights, v_r, tau_s, combined=combined)
    variables = {"r_0": f"output({float(r0[0])})"}
    for i in range(1, M):
        variables[f"r_{i}"] = f"variable({float(r0[i])})"
    for i in range(M):
        variables[f"v_{i}"] = f"variable({float(v0[i])})"
    variables["a"] = f"variable({float(r0.mean())})"
    variables["s"] = f"variable({float(r0.mean())})"
    variables["Iext"] = float(I0)
    variables["J"] = float(J0)
    variables["tau_s"] = float(tau_s)
    if combined:
        variables["h"] = float(h0)
    else:
        variables["hD"] = float(hD0)
        variables["hC"] = float(hC0)
    op = OperatorTemplate(name=f"{name}_op", equations=eqs, variables=variables)
    node = NodeTemplate(name=f"{name}_node", operators=[op])
    return CircuitTemplate(name=name, nodes={"p": node})


# ═════════════════════════════════════════════════════════════════════════════
#  helpers
# ═════════════════════════════════════════════════════════════════════════════
def _tag(cell_class, layer):
    """Filename tag matching data_fitting/allen_lorentzian_fit.py (e.g. 'pyramidal_L23')."""
    c = cell_class.split("+")[0].split()[0].lower()       # Pyramidal→pyramidal, PV+ int→pv
    return f"{c}_{layer.replace('/', '').replace(' ', '')}"


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


def _model_rhs(M, omega, delta, weights, v_r, tau_s, J, hD=1.0, hC=1.0):
    """Plain-numpy right-hand side of the mean field (state y=[r_0..,v_0..,a,s]) for a given
    external input I and coupling J — used only to recompute equilibrium stability robustly.
    hD/hC are the heterogeneity knobs (see build_equations)."""
    PI = np.pi
    omega = np.asarray(omega, float); delta = np.asarray(delta, float); weights = np.asarray(weights, float)
    Ombar = weights @ omega
    vthbar = v_r + Ombar + hC * (omega - Ombar)
    De = hD * delta

    def rhs(y, I):
        r, v, a, s = y[:M], y[M:2 * M], y[2 * M], y[2 * M + 1]
        dr = De / PI * (v - v_r) + r * (2 * v - v_r - vthbar)
        dv = (v - v_r) * (v - vthbar) - (PI * r) ** 2 - PI * De * r + I + J * s
        return np.concatenate([dr, dv, [(weights @ r - a) / tau_s, (a - s) / tau_s]])
    return rhs


def recompute_stability(eq_sols, M, omega, delta, weights, v_r, tau_s, J, hD=1.0, hC=1.0, tol=1e-6):
    """Overwrite the branch 'stability' column from the eigenvalues of the analytical-ish
    (finite-difference) Jacobian at each equilibrium: stable iff max Re(eig) < tol. PyCoBi's
    parsed stability is unreliable for this system (eigenvalues with large imaginary parts),
    flagging a settings-dependent spurious instability; the eigenvalue recompute is robust."""
    rhs = _model_rhs(M, omega, delta, weights, v_r, tau_s, J, hD, hC)
    Ic = _pcol(eq_sols, "Iext")
    cols = [_pcol(eq_sols, n) for n in state_var_order(M)]
    stab = np.empty(len(eq_sols), bool)
    for k in range(len(eq_sols)):
        I = float(eq_sols[Ic].iloc[k])
        y = np.array([float(eq_sols[c].iloc[k]) for c in cols])
        f0 = rhs(y, I); n = y.size; Jm = np.empty((n, n))
        for j in range(n):
            h = 1e-6 * (1.0 + abs(y[j])); yp = y.copy(); yp[j] += h
            Jm[:, j] = (rhs(yp, I) - f0) / h
        stab[k] = bool(np.max(np.linalg.eigvals(Jm).real) < tol)
    eq_sols[("stability", "")] = stab
    return eq_sols


def _plot_branch(ax, I, s, stab, color, lw=2.0):
    """Draw branch coloured by `color`, solid where stable / dashed where unstable. Each
    contiguous-stability run is one polyline (so dashes render), bridged at folds, broken at jumps."""
    stab = np.asarray(stab, bool)
    step = np.hypot(np.diff(I), np.diff(s))
    thr = 6 * np.median(step[step > 0]) if np.any(step > 0) else np.inf
    flips = list(np.where(np.diff(stab.astype(int)) != 0)[0] + 1)
    for a, b in zip([0] + flips, flips + [stab.size]):
        e = min(b + 1, stab.size)
        y = s[a:e].astype(float).copy()
        for L in np.where(step[a:e - 1] > thr)[0]:
            y[L + 1] = np.nan
        ax.plot(I[a:e], y, color=color, lw=lw, ls="-" if stab[a] else "--", zorder=2)


def save_bif_data(out_npz, eq_sols, codim2, cfg, M):
    """Extract the arrays the summary figure needs into a self-contained .npz (so the plotting
    script needs neither pycobi nor Auto): the 1-D branch (Iext, s, stability), the LP/HB marker
    coordinates, and each codim-2 curve in the (Iext, J) plane. NB at the equilibrium the synaptic
    activation equals the population mean rate (s = a = Σ_m w_m r_m), so the figure relabels s as r."""
    I = eq_sols[_pcol(eq_sols, "Iext")].to_numpy(float)
    s = eq_sols[_pcol(eq_sols, "s")].to_numpy(float)
    stab = (eq_sols[("stability", "")].to_numpy(bool) if ("stability", "") in eq_sols.columns
            else np.ones(I.size, bool))
    bif = eq_sols[("bifurcation", "")].to_numpy()
    data = dict(cell_class=cfg["cell_class"], layer=cfg["layer"], M=np.int64(M),
                J0=float(cfg["J0"]), I_min=float(cfg["I_min"]), I_max=float(cfg["I_max"]),
                J_min=float(cfg["J_min"]), J_max=float(cfg["J_max"]),
                branch_I=I, branch_s=s, branch_stab=stab,
                lp_I=I[bif == "LP"], lp_s=s[bif == "LP"],
                hb_I=I[bif == "HB"], hb_s=s[bif == "HB"],
                n_codim2=np.int64(len(codim2)))
    kinds = []
    for k, (_nm, kind, df) in enumerate(codim2):
        data[f"c2_{k}_I"] = df[_pcol(df, "Iext")].to_numpy(float)
        data[f"c2_{k}_J"] = df[_pcol(df, "J")].to_numpy(float)
        kinds.append(kind)
    data["c2_kinds"] = np.array(kinds if kinds else [""], dtype="<U8")
    np.savez(out_npz, **data)
    print(f"[saved] {os.path.basename(out_npz)}")


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
    tag = _tag(cfg["cell_class"], cfg["layer"])
    fit_npz = os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{tag}.npz")
    out_stem = os.path.join(_HERE, f"allen_qif_bifurcation_{tag}")
    w, Omega, Delta, M = load_fit(fit_npz)
    v_r = cfg["v_r"]
    vthbar = v_r + Omega
    DIM = 2 * M + 2
    print(f"Allen-QIF mean-field bifurcation — {cfg['cell_class']} {cfg['layer']} "
          f"(M={M}, dim {DIM})")
    print(f"  v̄_θ,m = {np.round(vthbar, 2)}   Δ_m = {np.round(Delta, 3)}   w_m = {np.round(w, 3)}")
    print(f"  J0={cfg['J0']}, settle at I={cfg['I0']}")

    r0 = np.full(M, cfg["r0"])
    v0 = np.full(M, v_r)
    circuit = build_circuit(M, Omega, Delta, w, v_r, cfg["tau_s"], cfg["J0"], cfg["I0"], r0, v0)
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
    # NB: PyCoBi's parsed get_stability is UNRELIABLE for this system (eigenvalues with large
    # imaginary parts → a spurious, settings-dependent stability flip; see recompute_stability).
    # Auto's bifurcation DETECTION (ISP=2 Hopf / ILP=1 fold) is correct; we only recompute the
    # stability colouring directly from the Jacobian eigenvalues at each equilibrium.
    recompute_stability(eq_sols, M, Omega, Delta, w, v_r, cfg["tau_s"], cfg["J0"])
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
    print(f"\n[plot] writing {os.path.basename(out_stem)}.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) 1-parameter branch: synaptic activation s (= total rate at equilibrium) vs I.
    #     Plotted manually from the recomputed (Jacobian-eigenvalue) stability — NOT
    #     plot_continuation, which would re-use PyCoBi's unreliable parsed stability.
    _plot_branch(ax1,
                 eq_sols[_pcol(eq_sols, "Iext")].to_numpy(float),
                 eq_sols[_pcol(eq_sols, "s")].to_numpy(float),
                 eq_sols[("stability", "")].to_numpy(bool), C_EQ)
    add_markers(ax1, eq_sols, "LP", M_FOLD, "Iext", "s", C_FOLD)
    add_markers(ax1, eq_sols, "HB", M_HOPF, "Iext", "s", C_HOPF)
    ax1.set_xlabel(r"external input $I$")
    ax1.set_ylabel(r"synaptic activation $s$  ($=\sum_m w_m r_m$ at equilibrium)")
    ax1.set_title(f"(a)  {cfg['cell_class']}, {cfg['layer']}  (J={cfg['J0']:g}, M={M})")
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
    plt.savefig(out_stem + ".png", dpi=140, bbox_inches="tight")
    plt.savefig(out_stem + ".pdf", bbox_inches="tight")
    print(f"  saved {os.path.basename(out_stem)}.{{png,pdf}}")

    # ── persist results: self-contained .npz for the summary figure + full pycobi session ──
    save_bif_data(out_stem + ".npz", eq_sols, codim2, cfg, M)
    try:
        ode.to_file(out_stem + ".pkl", results_only=True)
        print(f"[saved] {os.path.basename(out_stem)}.pkl (pycobi session)")
    except Exception as e:
        print(f"  (ode.to_file skipped: {type(e).__name__}: {e})")

    ode.close_session(clear_files=True)
    clear(circuit)
    print("Done.")


if __name__ == "__main__":
    main()
