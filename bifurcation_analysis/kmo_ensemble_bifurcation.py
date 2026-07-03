r"""
Bifurcation analysis of the fitted weighted-Lorentzian adaptive-Kuramoto OA model
=================================================================================

Loads the mean-field network parameters + initial conditions exported by
``kuramoto/kuramoto_ensemble_fitting.py`` (``oa_params_M<M>.npz``) and performs a
PyCoBi/Auto-07p bifurcation analysis: settle onto the relative-equilibrium
(phase-locked) state, continue it in the weight-decay γ, and locate where it
loses stability (fold ``LP`` / Hopf ``HB``).

Per the project scope: proper limit-cycle / homoclinic continuation is infeasible
for large M (the reduced dimension is 2M − 1 + M²), so this script deliberately
stops at the equilibrium branch and its codim-1 instabilities. Those are the
robust, M-scalable results.

The model (see ``kuramoto_ensemble_fitting.oa_ode``) — note it DIFFERS from
``kmo_macro_simulation`` in three ways, all handled here:
  * coupling is weighted by the mixture weights w_j with NO 1/M normalisation:
        field_i = K Σ_j w_j A_ij z_j ;
  * the (anti-)Hebbian kernel for the smooth model is cos / sin (not |sin|):
        Ȧ_ij = μ r_i r_j cos(ψ_j−ψ_i) − γ A_ij      (Hebbian),
        Ȧ_ij = μ r_i r_j sin(ψ_j−ψ_i) − γ A_ij      (anti-Hebbian);
  * the centres ω_i, widths δ_i are per-ensemble (heterogeneous).

As before we work in co-rotating Cartesian order parameters z_i = x_i + i y_i
(pin y_0 := 0, x_0 = r_0; carry Ω = Im F_0/x_0) so the relative equilibrium is a
genuine fixed point. ω_i, δ_i, w_i are inlined as constants; K, μ, g are the
continuable Auto parameters.

Run inside the ``pycobi`` conda env with meson/ninja on PATH:
    PATH="$CONDA_PREFIX/bin:$PATH" python kmo_ensemble_bifurcation.py
(The .npz is produced by kuramoto_ensemble_fitting.py, which must run in the
``sbi`` env — it needs sklearn — so the two steps use different environments.)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
from pycobi import ODESystem

AUTO_DIR = "~/PycharmProjects/auto-07p"
_EPS = 1e-12

# colours / markers
C_EQ = "#1F77B4"
M_FOLD = "o"
M_HOPF = "s"


# ═════════════════════════════════════════════════════════════════════════════
#  Top-level configuration
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    params_npz="../kuramoto/oa_params_M5.npz",   # exported by kuramoto_ensemble_fitting.py
    # Continuation parameter and range. Hebbian (symmetric) rule, μ>0 fixed: the
    # synchronized relative equilibrium A*=(μ/γ) r_i r_j cos(ψ_j−ψ_i) exists and is
    # stable at small γ; we continue UP in the weight decay γ to find where it
    # loses stability (fold LP / Hopf HB) — exactly as in the previous adaptive
    # bifurcation analysis. (γ=0 / μ-continuation is degenerate, see git history.)
    cont_param="g",
    cont_min=0.0, cont_max=0.2,
    # γ at which to settle the IVP onto the synchronized relative equilibrium.
    # None ⇒ use the γ from the .npz (μ=0 ⇒ weights stay frozen at A=ones, so the
    # IVP relaxes to the ordinary Kuramoto synchronized state).
    gamma_start=None,
    T_settle=3000.0,
    # IVP start: a SYNCHRONIZED guess (r_0 high, phases aligned) to land on the
    # synchronized branch. Set use_exported_ic=True to use the exported r0/psi0.
    r0_start=0.9,
    use_exported_ic=False,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Cartesian reduced-model generator (weighted, per-ensemble ω_i/δ_i, cos/sin)
# ═════════════════════════════════════════════════════════════════════════════
def _xj(j):
    return f"x_{j}"


def _yj(j):
    return "0" if j == 0 else f"y_{j}"


def _adrive(i, j, plasticity):
    """Cartesian form of the plasticity drive for A_ij (with y_0 := 0):
        Hebbian      μ Re[z_i* z_j] = μ (x_i x_j + y_i y_j),
        anti-Hebbian μ Im[z_i* z_j] = μ (x_i y_j − y_i x_j)."""
    xi, yi, xj, yj = _xj(i), _yj(i), _xj(j), _yj(j)
    if plasticity == "hebbian":
        terms = [f"{xi}*{xj}"]
        if yi != "0" and yj != "0":
            terms.append(f"{yi}*{yj}")
        return " + ".join(terms)
    # anti-Hebbian: x_i y_j − y_i x_j
    terms = []
    if yj != "0":
        terms.append(f"{xi}*{yj}")
    if yi != "0":
        terms.append(f"-{yi}*{xj}")
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def build_equations_cartesian(M, omega, delta, weights, plasticity):
    """Reduced OA in co-rotating Cartesian coords for the WEIGHTED model.
    field_i = K Σ_j w_j A_ij z_j ; ω_i, δ_i, w_i inlined; dim = 2M − 1 + M²."""
    w = [float(x) for x in weights]
    om = [float(x) for x in omega]
    dl = [float(x) for x in delta]

    def ReH(i):
        return "K*(" + " + ".join(f"{w[j]}*A_{i}_{j}*{_xj(j)}" for j in range(M)) + ")"

    def ImH(i):
        t = [f"{w[j]}*A_{i}_{j}*{_yj(j)}" for j in range(M) if j != 0]
        return "K*(" + (" + ".join(t) if t else "0") + ")"

    def ReF(i):
        xi, yi, rH, iH = _xj(i), _yj(i), ReH(i), ImH(i)
        rezz = f"(x_0^2)*{rH}" if i == 0 else f"(({xi})^2-({yi})^2)*{rH} + 2*{xi}*{yi}*{iH}"
        bias = f"-{dl[i]}*{xi}" if i == 0 else f"-{dl[i]}*{xi} - {om[i]}*{yi}"
        return f"({bias} + 0.5*({rH} - ({rezz})))"

    def ImF(i):
        xi, yi, rH, iH = _xj(i), _yj(i), ReH(i), ImH(i)
        if i == 0:
            imzz, bias = f"-(x_0^2)*{iH}", f"{om[0]}*x_0"
        else:
            imzz = f"2*{xi}*{yi}*{rH} - (({xi})^2-({yi})^2)*{iH}"
            bias = f"-{dl[i]}*{yi} + {om[i]}*{xi}"
        return f"({bias} + 0.5*({iH} - ({imzz})))"

    Omega = f"(({ImF(0)})/x_0)"
    eqs = [f"d/dt * x_0 = {ReF(0)}"]
    for i in range(1, M):
        eqs.append(f"d/dt * x_{i} = {ReF(i)} + {Omega}*y_{i}")
        eqs.append(f"d/dt * y_{i} = {ImF(i)} - {Omega}*x_{i}")
    for i in range(M):
        for j in range(M):
            eqs.append(f"d/dt * A_{i}_{j} = mu*({_adrive(i, j, plasticity)}) - g*A_{i}_{j}")
    return eqs


def state_var_order(M):
    names = ["x_0"]
    for i in range(1, M):
        names += [f"x_{i}", f"y_{i}"]
    for i in range(M):
        for j in range(M):
            names.append(f"A_{i}_{j}")
    return names


def build_circuit(M, K, mu, gamma, omega, delta, weights, r0, phi0, A0,
                  plasticity, name="kmo_ens"):
    eqs = build_equations_cartesian(M, omega, delta, weights, plasticity)
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
    op = OperatorTemplate(name=f"{name}_op", equations=eqs, variables=variables)
    node = NodeTemplate(name=f"{name}_node", operators=[op])
    return CircuitTemplate(name=name, nodes={"p": node})


# ═════════════════════════════════════════════════════════════════════════════
#  Parameter loader (standalone — no sklearn dependency, runs in the pycobi env)
# ═════════════════════════════════════════════════════════════════════════════
def load_meanfield_params(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    return dict(
        M=int(d["M"]),
        weights=np.asarray(d["weights"], float),
        omega=np.asarray(d["omega"], float),
        delta=np.asarray(d["delta"], float),
        K=float(d["K"]), mu=float(d["mu"]), gamma=float(d["gamma"]),
        plasticity=str(d["plasticity"]),
        r0=np.asarray(d["r0"], float), psi0=np.asarray(d["psi0"], float),
        A0=np.asarray(d["A0"], float),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Analytical Jacobian (PyRates) → robust equilibrium stability
# ═════════════════════════════════════════════════════════════════════════════
def make_jacobian_eval(M, K, mu, omega, delta, weights, plasticity):
    """PyRates analytical Jacobian J(y) of the reduced field at a given g, used to
    assign equilibrium stability robustly (Auto's parsed stability flickers where
    it omits the eigenvalue line). Returns jac(y, g) → (n×n)."""
    jcirc = build_circuit(M, K, mu, 0.1, omega, delta, weights,
                          np.full(M, 0.5), np.zeros(M), np.ones((M, M)),
                          plasticity, name="kmo_ens_jac")
    Jf, _, jarg_names, jsv = jcirc.get_jacobian_func(
        "kmo_ens_jac", 1e-3, backend="default", vectorize=False, verbose=False, clear=True)
    assert [k.split("/")[-1] for k in jsv] == state_var_order(M)
    pnames = [n.split("/")[-1] for n in jarg_names[2:]]   # subset of {K, mu, g}

    def jac(y, pvals):
        """pvals: dict with the current values of K, mu, g."""
        return np.asarray(Jf(0.0, np.asarray(y, float), *[pvals[p] for p in pnames]))

    clear(jcirc)
    return jac


def _pcol(df, name):
    for c in df.columns:
        head = c[0] if isinstance(c, tuple) else c
        sub = c[1] if isinstance(c, tuple) and len(c) > 1 else ""
        if (head == name or (isinstance(head, str) and head.endswith("/" + name))) \
                and sub in ("", 0):
            return c
    raise KeyError(name)


def recompute_equilibrium_stability(sols, M, jac, cont_param, base_pvals, tol=1e-6):
    """Overwrite the summary 'stability' column from the analytical Jacobian:
    stable iff no eigenvalue has positive real part. ``base_pvals`` holds the
    fixed K/mu/g; the continuation parameter is read per-row from the summary.
    (With γ=0 the M² weight directions are neutral — zero eigenvalues — so
    stability is governed by the order-parameter subspace, which this captures.)"""
    order = state_var_order(M)
    cont_col = _pcol(sols, cont_param)
    var_cols = [_pcol(sols, n) for n in order]
    stab = np.empty(len(sols), dtype=bool)
    for r in range(len(sols)):
        pvals = dict(base_pvals)
        pvals[cont_param] = float(sols[cont_col].iloc[r])
        y = np.array([float(sols[c].iloc[r]) for c in var_cols])
        ev = np.linalg.eigvals(jac(y, pvals))
        stab[r] = bool(np.sum(ev.real > tol) == 0)
    sols[("stability", "")] = stab
    return sols


def _bif_param(sols, label, pname):
    p_col = _pcol(sols, pname)
    rows = sols[sols[("bifurcation", "")] == label]
    return sorted(float(v) for v in rows[p_col]) if len(rows) else []


def add_markers(ax, sols, label, marker, pname, color="k", size=80):
    p_col = _pcol(sols, pname)
    bif = sols[("bifurcation", "")].to_numpy()
    idx = np.where(bif == label)[0]
    if idx.size == 0:
        return
    x0cols = [c for c in sols.columns if (c[0] if isinstance(c, tuple) else c) == "x_0"]
    xs = [float(sols[p_col].iloc[i]) for i in idx]
    ys = [max(float(sols[c].iloc[i]) for c in x0cols) for i in idx]
    ax.scatter(xs, ys, marker=marker, s=size, facecolors="none",
               edgecolors=color, linewidths=1.6, zorder=12)


# ═════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    p = load_meanfield_params(cfg["params_npz"])
    M = p["M"]
    K, mu = p["K"], p["mu"]
    omega, delta, weights = p["omega"], p["delta"], p["weights"]
    plasticity = p["plasticity"]
    DIM = 2 * M - 1 + M * M
    print(f"KMO-ensemble bifurcation — M={M} (dim {DIM}), plasticity={plasticity}")
    print(f"  loaded K={K}, mu={mu}, gamma(file)={p['gamma']:.4g}")
    print(f"  omega∈[{omega.min():.3f},{omega.max():.3f}]  "
          f"delta∈[{delta.min():.3f},{delta.max():.3f}]  weights sum={weights.sum():.3f}")

    # initial state for the IVP. Default: a SYNCHRONIZED guess (r_0 high, phases
    # aligned, A0 = ones) to land on the synchronized relative-equilibrium branch.
    A0 = p["A0"]
    if cfg["use_exported_ic"]:
        r0 = p["r0"]
        phi0 = p["psi0"] - p["psi0"][0]
        phi0[0] = 0.0
    else:
        r0 = np.full(M, cfg["r0_start"])
        phi0 = np.zeros(M)

    jac = make_jacobian_eval(M, K, mu, omega, delta, weights, plasticity)

    gamma_ivp = p["gamma"] if cfg["gamma_start"] is None else cfg["gamma_start"]
    cont_param = cfg["cont_param"]
    circuit = build_circuit(M, K, mu, gamma_ivp, omega, delta, weights,
                            r0, phi0, A0, plasticity)
    ode = ODESystem.from_template(circuit, auto_dir=AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True,
                                  auto_constants=("ivp", "eq"))

    # ── Step 1: IVP → relative equilibrium at gamma_ivp ─────────────────────
    print(f"\n[1] IVP → synchronized relative equilibrium (γ = {gamma_ivp:.4g})")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6,
            UZR={14: cfg["T_settle"]}, STOP={"UZ1"})
    try:
        s, _, _ = ode.get_solution(point="UZ1", cont="time")
        if hasattr(s, "b") and isinstance(getattr(s, "b", None), dict):
            s = s.b["solution"]
        coords = {c.split("/")[-1]: float(np.asarray(s[c]).ravel()[-1])
                  for c in s.coordnames}
        print(f"  settled x_0 = r_0 = {coords.get('x_0', float('nan')):.4f}")
    except Exception as e:
        print(f"  (settled-state read skipped: {type(e).__name__})")

    # ── Step 2: equilibrium continuation in the chosen parameter → fold/Hopf ─
    print(f"[2] equilibrium continuation in {cont_param}")
    eq_sols, _ = ode.run(
        origin="time", starting_point="UZ1", name="eq_branch", c="eq",
        ICP=cont_param, bidirectional=True, RL0=cfg["cont_min"], RL1=cfg["cont_max"],
        IPS=1, ILP=1, ISP=2, ISW=1, NMX=10000, NPR=200,
        DS=1e-3, DSMIN=1e-10, DSMAX=2e-2, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
        get_stability=True)
    base_pvals = {"K": K, "mu": mu, "g": gamma_ivp}
    recompute_equilibrium_stability(eq_sols, M, jac, cont_param, base_pvals)
    bifs = dict(eq_sols["bifurcation"].value_counts())
    print("  eq bifurcations:", bifs)
    folds = _bif_param(eq_sols, "LP", cont_param)
    hopfs = _bif_param(eq_sols, "HB", cont_param)
    print(f"  fold(s) LP at {cont_param} = {[round(x, 4) for x in folds]}")
    print(f"  Hopf(s) HB at {cont_param} = {[round(x, 4) for x in hopfs]}")

    # ── plot: equilibrium branch (x_0 vs cont_param) + stability + bifurcations
    print("\n[plot] writing kmo_ensemble_bifurcation.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ode.plot_continuation(cont_param, "x_0", cont="eq_branch", ax=ax,
                          line_color_stable=C_EQ, line_color_unstable=C_EQ,
                          line_style_stable="solid", line_style_unstable="dashed",
                          ignore=["UZ", "BP", "EP"], bifurcation_legend=False,
                          linewidths=2.2)
    add_markers(ax, eq_sols, "LP", M_FOLD, cont_param, color="k")
    add_markers(ax, eq_sols, "HB", M_HOPF, cont_param, color="#D62728")
    _lab = {"g": r"weight decay $\gamma$", "K": r"coupling $K$",
            "mu": r"learning rate $\mu$"}.get(cont_param, cont_param)
    ax.set_xlabel(_lab)
    ax.set_ylabel(r"$x_0 = r_0$  (relative-equilibrium coherence)")
    ax.set_title(f"Ensemble OA equilibrium branch  (M={M}, {plasticity})")
    legend = [
        Line2D([0], [0], color=C_EQ, lw=2, label="equilibrium (solid=stable, dashed=unstable)"),
        Line2D([0], [0], marker=M_FOLD, color="k", lw=0, markerfacecolor="none", label="fold (LP)"),
        Line2D([0], [0], marker=M_HOPF, color="#D62728", lw=0, markerfacecolor="none", label="Hopf (HB)"),
    ]
    ax.legend(handles=legend, loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig("kmo_ensemble_bifurcation.png", dpi=140, bbox_inches="tight")
    print("  saved kmo_ensemble_bifurcation.png")

    ode.close_session(clear_files=True)
    clear(circuit)
    print("Done.")


if __name__ == "__main__":
    main()
    plt.show()
