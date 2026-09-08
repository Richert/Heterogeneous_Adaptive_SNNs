r"""
Adaptive Kuramoto — Δ-sweep over adaptation rules × adaptation symmetry (PyRates)
=================================================================================

Sweeps the natural-frequency heterogeneity Δ of an all-to-all *adaptively* coupled
Kuramoto network for a set of adaptation-rule FAMILIES, each in a SYMMETRIC and an
ANTISYMMETRIC version, and repeats every sweep point for ``n_trials`` independent random
phase initial conditions. A rule family is a mathematical structure of the weight
dynamics; the two versions differ only in the adaptation function G:

    family `decay`:     Ȧ_ij = μ G(θ_j − θ_i) + γ (1 − A_ij)
    family `logistic`:  Ȧ_ij = μ G(θ_j − θ_i) A_ij (A_max − A_ij)   [bounded: 0 < A < A_max]
                        sym:  G(x) = cos(x)      asym:  G(x) = sin(x)
    family `pulse`:     Ȧ_ij = μ_p G_p(θ_i,θ_j) (A_max − A_ij) − μ_d G_d(θ_i,θ_j) A_ij
                        separate LTP/LTD terms driven by the non-negative pulse function
                        s(n,θ) = c_n(1−cos θ)^n of ``theory/pulse_kernels.py`` (Duchet et
                        al., Neural Comp. 2023; Fennelly et al., Chaos 2025), which keeps
                        A in [0, A_max]. Here the two versions differ in the KERNELS:
                        sym  (Hebbian):   G_p = s(θ_i)s(θ_j),   G_d = ½[s(θ_i)+s(θ_j)]
                                          — coincidence LTP + activity-driven LTD, both
                                          invariant under i↔j, so A stays symmetric;
                        asym (STDP-like): G_p = s(θ_i)s(θ_j−φ), G_d = s(θ_i−φ)s(θ_j)
                                          — LTP if the presynaptic pulse leads the
                                          postsynaptic one by φ, LTD in the reverse order.
                                          G_p^ij = G_d^ji makes the window antisymmetric:
                                          A_ij + A_ji = A_max is conserved (μ_p = μ_d).

Each (family, symmetry) combination is one PyRates ``OperatorTemplate`` in ``config/kuramoto.yaml``,
named ``kmo_adapt_<family>_<sym|asym>``. TO ADD A FAMILY: write its two operators into
that YAML and add one entry to ``RULES`` below (operator names, label, and any
family-specific parameters); nothing else in the script needs to change — the sweep,
the storage layout and the .npz gain a slot along the rule axis automatically.

The oscillators are the shared ``kmo_op`` (θ̇_i = ω_i + Im(s_in,i e^{-iθ_i}), with
s_in,i = (K/N) Σ_j A_ij e_j) from the same file. Model construction is pure PyRates
(``PopulationTemplate`` + an adaptive ``Connectivity`` edge carrying A_ij); the compiled
vector field (``get_run_func``) is integrated with ``scipy.integrate.solve_ivp``.

Recorded per (rule family, symmetry, Δ, trial):
  * R(t)    = |⟨e^{iθ_i(t)}⟩|                  — average phase coherence
  * Ā(t)    = ⟨A_ij(t)⟩                        — average coupling weight
  * V_A(t)  = ⟨A_ij(t)²⟩ − Ā(t)²               — coupling-weight variance
  * A_ij(T) — the final coupling matrix (frequency-sorted, ω ascending)
Ā and V_A are taken over the OFF-diagonal weights only (the self-coupling A_ii has no
mean-field counterpart). Everything is written to a single compressed .npz.

By default the trial ICs are drawn once and re-used across rules, symmetries and Δ, so
that all conditions are compared on identical initial states (``--fresh-ics`` re-draws
the phases for every single simulation instead).

Run in the ``pycobi`` conda env (dev PyRates 1.2.2 + scipy); at the default settings one
simulation takes ≈25 s (N=200, T=1000), i.e. ≈3 h for the full 3×2×13×5 sweep:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_rule_sweep.py
"""

# --- shared library bootstrap (repo-root shared/, theory/) ------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared"), os.path.join(_HERE, "..", "theory")]
import data_paths as dp
import pulse_kernels as pk          # pulse function s(n,θ) = c_n (1−cos θ)^n and its c_n
# ---------------------------------------------------------------------------
import argparse
import time

import numpy as np
from scipy.integrate import solve_ivp

from pyrates import OperatorTemplate, NodeTemplate, EdgeTemplate, CircuitTemplate, clear
from pyrates.frontend.template.population import PopulationTemplate, Connectivity

# shared Kuramoto equation templates (config/kuramoto.yaml)
_KY = os.path.abspath(os.path.join(_HERE, "..", "config", "kuramoto"))


def _op(name):
    return OperatorTemplate.from_yaml(f"{_KY}/{name}")


# ════════════════════════════════════════════════════════════════════════════
#  adaptation-rule registry: family -> {symmetric, antisymmetric} operator
# ════════════════════════════════════════════════════════════════════════════
#  ``ops``        : OperatorTemplate names in config/kuramoto.yaml, one per symmetry.
#  ``label``      : short math label of the family (plots / logs).
#  ``params``     : family-specific edge parameters, merged on top of the shared ones
#                   (mu -> μ, decay -> γ, A -> A(0)); a dict, or a callable(cfg) -> dict
#                   when they derive from the shared settings. Keys must exist in the
#                   operator (typo guard); ``CONFIG["rule_params"][family]`` overrides them.
#  ``sym_params`` : per-symmetry additions, for knobs only one version has (e.g. the STDP lag).
#  Extending the sweep = add the two operators to the YAML + one entry here.
SYMMETRIES = {"sym": "symmetric", "asym": "antisymmetric"}
N_PULSE = 5                          # pulse sharpness n of the `pulse` family (Duchet default)

RULES = {
    "decay": dict(
        label=r"dA/dt = mu*G(theta_j - theta_i) + gamma*(1 - A)",
        ops={"sym": "kmo_adapt_decay_sym", "asym": "kmo_adapt_decay_asym"},
        params={},
    ),
    "logistic": dict(
        label=r"dA/dt = mu*G(theta_j - theta_i)*A*(A_max - A)",
        ops={"sym": "kmo_adapt_logistic_sym", "asym": "kmo_adapt_logistic_asym"},
        params={"A_max": 2.0},           # weights stay in (0, A_max); A(0)=A0 must lie inside
    ),
    "pulse": dict(
        label=r"dA/dt = mu_p*G_p*(A_max - A) - mu_d*G_d*A   [pulse LTP/LTD]",
        ops={"sym": "kmo_adapt_pulse_sym", "asym": "kmo_adapt_pulse_asym"},
        # sym : G_p = s(θ_i)s(θ_j) coincidence LTP, G_d = ½[s(θ_i)+s(θ_j)] activity LTD
        # asym: G_p = s(θ_i)s(θ_j−φ) pre→post LTP, G_d = s(θ_i−φ)s(θ_j) post→pre LTD
        # μ_p = μ_d = μ by default: the pulses are normalised (⟨s⟩=1 at uniform phases), so
        # μ sets the adaptation rate for this family too, and Ā(0)=A_max/2 is the balance
        # point of the asynchronous state.
        params=lambda cfg: dict(mu_p=cfg["mu"], mu_d=cfg["mu"], A_max=2.0,
                                n_pulse=float(N_PULSE), c_pulse=pk.c_n(N_PULSE)),
        sym_params={"asym": dict(phi=0.5 * np.pi)},   # STDP lag φ: G_p^ij = G_d^ji
    ),
    # "<family>": dict(label=..., ops={"sym": "kmo_adapt_<family>_sym",
    #                                  "asym": "kmo_adapt_<family>_asym"},
    #                  params={...}),      # e.g. saturation level, time constants, ...
}


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    # network
    N=200,                          # number of oscillators
    K=1.0,                          # global coupling scale
    omega_bar=0.0,                  # Lorentzian centre (only rotates the frame)
    trunc=10.0,                     # Lorentzian truncated at ±trunc·Δ (tames the fast tails)
    # adaptation: rule families (keys of RULES) × symmetries (keys of SYMMETRIES)
    rules=["decay", "logistic", "pulse"],
    symmetries=["sym", "asym"],
    mu=0.04,                        # adaptation rate μ (shared by all families)
    gamma=0.01,                     # weight decay γ (relaxes A → 1)
    A0=1.0,                         # initial coupling weights A_ij(0)
    rule_params={},                 # per-family parameter overrides: {"decay": {...}}
    # sweep + trials
    deltas=list(np.round(np.linspace(0.02, 1.2, 13), 4)),      # spans sync → async at K=1
    n_trials=5,                     # random-IC repetitions per (rule, symmetry, Δ)
    sigma0=None,                    # phase ICs: None → uniform on [−π,π), else θ_i ~ N(0,σ₀)
    fresh_ics=False,                # True: re-draw the ICs for every simulation
    seed=1,
    # integration (T ≫ 1/γ so the slow weights reach steady state)
    T=1000.0, dts=2.0,
    method="RK45", rtol=1e-6, atol=1e-8,
    # storage
    save_res=200,                   # block-average the final A matrix / ω axis to this size
    out=dp.mpmf("kmo_adaptive_rule_sweep.npz"),
)


# ════════════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════════════
def lorentzian_truncated(N, center, Delta, trunc):
    """Deterministic Lorentzian quantiles (ascending), truncated at ±trunc·Δ so the
    far-tail oscillators (which never entrain) don't make the integrator stiff."""
    p0 = np.arctan(trunc) / np.pi
    p = np.linspace(0.5 - p0, 0.5 + p0, N + 2)[1:-1]
    return center + Delta * np.tan(np.pi * (p - 0.5))


def draw_phases(rng, N, sigma0):
    """Random phase ICs: uniform on the circle (σ₀=None) or wrapped normal N(0,σ₀)."""
    if sigma0 is None:
        return rng.uniform(-np.pi, np.pi, N)
    return rng.normal(0.0, float(sigma0), N)


def block_average(M, res):
    n = M.shape[0]
    if n <= res:
        return M
    b = n // res
    return M[:res * b, :res * b].reshape(res, b, res, b).mean(axis=(1, 3))


def block_average_1d(v, res):
    n = v.size
    if n <= res:
        return v
    b = n // res
    return v[:res * b].reshape(res, b).mean(axis=1)


def _block(vmap, match):
    """Index range of a state-variable block, by key suffix or substring."""
    v = next(val for k, val in vmap.items() if k.endswith(match) or match in k)
    if isinstance(v, (tuple, list)):
        return slice(int(v[0]), int(v[1]))
    return slice(int(v), int(v) + 1)


_RUN_ID = 0


def _run(net, cfg, tag):
    """get_run_func -> solve_ivp. Returns (t_eval, state_trajectory, vmap)."""
    func, args, keys, vmap = net.get_run_func(f"{tag}_vf", step_size=1e-2, backend="numpy",
                                              vectorize=True, clear=False,
                                              float_precision="complex128")
    y0 = np.asarray(args[1])
    extra = args[2:]

    def f(t, y):
        return np.array(func(t, y, *extra))

    t_eval = np.arange(0.0, cfg["T"], cfg["dts"])
    sol = solve_ivp(f, (0.0, cfg["T"]), y0, method=cfg["method"], t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"])
    return t_eval, sol.y, vmap


# ════════════════════════════════════════════════════════════════════════════
#  microscopic model (kmo_op + one adaptation-rule edge, both from the YAML)
# ════════════════════════════════════════════════════════════════════════════
def rule_operator(rule, symmetry):
    """OperatorTemplate name implementing rule family ``rule`` at ``symmetry``."""
    if rule not in RULES:
        raise KeyError(f"unknown rule family '{rule}' (registered: {list(RULES)})")
    ops = RULES[rule]["ops"]
    if symmetry not in ops:
        raise KeyError(f"rule family '{rule}' has no '{symmetry}' version (has: {list(ops)})")
    return ops[symmetry]


def edge_params(rule, symmetry, cfg):
    """Edge parameters for one (rule family, symmetry): shared knobs, then the family's
    own parameters, its per-symmetry additions, and finally the user's overrides."""
    own = RULES[rule].get("params", {})
    own = dict(own(cfg)) if callable(own) else dict(own)
    own.update(RULES[rule].get("sym_params", {}).get(symmetry, {}))
    p = {"mu": cfg["mu"], "decay": cfg["gamma"], "A": cfg["A0"],
         **own, **cfg.get("rule_params", {}).get(rule, {})}
    if "A_max" in p and not 0.0 < p["A"] < p["A_max"]:      # bounded rules: A(0) ∈ (0, A_max)
        raise ValueError(f"rule '{rule}': A(0)={p['A']} outside the invariant interval "
                         f"(0, A_max={p['A_max']})")
    return p


def simulate(rule_op, params, theta0, omega, cfg):
    """All-to-all adaptive Kuramoto network with the adaptation rule ``rule_op``.

    Returns (t, R(t), Ā(t), V_A(t), A(T)); Ā and V_A are off-diagonal statistics."""
    global _RUN_ID
    _RUN_ID += 1
    N, K = omega.size, cfg["K"]
    tag = f"micro{_RUN_ID}"

    node = NodeTemplate(name="osc", operators=[_op("kmo_op")])
    pop = PopulationTemplate(name="osc", node=node, n=N,
                             params={"kmo_op/omega": omega, "kmo_op/theta": theta0})
    edge_op = _op(rule_op)
    edge = EdgeTemplate(name=f"{rule_op}_edge", operators=[edge_op])
    for var, val in params.items():
        if var in edge_op.variables:
            edge.update_var(rule_op, var, val)
        elif var not in ("mu", "decay", "A"):      # shared knobs may be absent; own ones may not
            raise KeyError(f"'{rule_op}' declares no parameter '{var}'")
    W = (K / N) * np.ones((N, N))                          # uniform wiring; A_ij is the edge state
    conn = Connectivity("osc/kmo_op/e", "osc/kmo_op/s_in", weights=W, edge=edge,
                        edge_var_map={"e_pre": "source", "e_post": "osc/kmo_op/e"})
    net = CircuitTemplate(tag, populations={"osc": pop}, connections=[conn])

    t, Y, vmap = _run(net, cfg, tag)
    theta = np.real(Y[_block(vmap, "kmo_op/theta")])       # (N, n_t)
    A_flat = np.real(Y[_block(vmap, "_flat")])             # (N², n_t) adaptive weights
    R = np.abs(np.exp(1j * theta).mean(axis=0))

    diag = np.arange(N) * N + np.arange(N)                 # flat indices of A_ii
    n_off = N * N - N
    s1 = A_flat.sum(axis=0) - A_flat[diag].sum(axis=0)
    s2 = (A_flat ** 2).sum(axis=0) - (A_flat[diag] ** 2).sum(axis=0)
    Abar = s1 / n_off                                      # off-diagonal mean Ā(t)
    VA = s2 / n_off - Abar ** 2                            # off-diagonal variance V_A(t)
    A_final = A_flat[:, -1].reshape(N, N)
    clear(net)
    return t, R, Abar, VA, A_final


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])
    N, res = cfg["N"], cfg["save_res"]
    rules, syms = list(cfg["rules"]), list(cfg["symmetries"])
    deltas = np.asarray(cfg["deltas"], float)
    n_tr, nr, ns, nd = cfg["n_trials"], len(rules), len(syms), deltas.size
    n_t = int(np.arange(0.0, cfg["T"], cfg["dts"]).size)
    n_res = min(N, res)
    ops = np.array([[rule_operator(r, s) for s in syms] for r in rules])   # (nr, ns) names

    print(f"adaptive-Kuramoto rule sweep (PyRates + solve_ivp) — N={N}, K={cfg['K']}, "
          f"μ={cfg['mu']}, γ={cfg['gamma']}, T={cfg['T']:g}")
    for i, r in enumerate(rules):
        print(f"  rule   : {r:<10} {RULES[r]['label']}")
        for j, s in enumerate(syms):
            print(f"           {s:<6} ({SYMMETRIES.get(s, '?')}) -> {ops[i, j]}")
    print(f"  Δ grid : {nd} points ∈ [{deltas.min():g}, {deltas.max():g}]")
    ic_kind = "uniform phases" if cfg["sigma0"] is None else f"θ~N(0,{cfg['sigma0']:g})"
    ic_share = "re-drawn per simulation" if cfg["fresh_ics"] else "shared across conditions"
    print(f"  trials : {n_tr} random ICs ({ic_kind}, {ic_share})")

    # trial ICs (drawn up front unless --fresh-ics: identical states across all conditions)
    theta0s = None if cfg["fresh_ics"] else [draw_phases(rng, N, cfg["sigma0"]) for _ in range(n_tr)]

    t_axis = None
    shape = (nr, ns, nd, n_tr)
    R = np.full(shape + (n_t,), np.nan, dtype=np.float32)
    Abar = np.full(shape + (n_t,), np.nan, dtype=np.float32)
    VA = np.full(shape + (n_t,), np.nan, dtype=np.float32)
    A_final = np.full(shape + (n_res, n_res), np.nan, dtype=np.float32)
    R0 = np.full(shape, np.nan)                             # coherence of the ICs
    omega_axis = np.full((nd, n_res), np.nan)               # frequency-sorted ω per Δ

    for j, D in enumerate(deltas):
        omega_axis[j] = block_average_1d(lorentzian_truncated(N, cfg["omega_bar"], D, cfg["trunc"]),
                                         n_res)

    t0 = time.time()
    for i, rule in enumerate(rules):
        for s, sym in enumerate(syms):
            params = edge_params(rule, sym, cfg)
            for j, D in enumerate(deltas):
                omega = lorentzian_truncated(N, cfg["omega_bar"], float(D), cfg["trunc"])
                for k in range(n_tr):
                    theta0 = draw_phases(rng, N, cfg["sigma0"]) if cfg["fresh_ics"] else theta0s[k]
                    t, Rt, Ab, V, A_fin = simulate(ops[i, s], params, theta0, omega, cfg)
                    t_axis = t if t_axis is None else t_axis
                    R[i, s, j, k], Abar[i, s, j, k], VA[i, s, j, k] = Rt, Ab, V
                    A_final[i, s, j, k] = block_average(A_fin, n_res)
                    R0[i, s, j, k] = float(np.abs(np.exp(1j * theta0).mean()))
                    print(f"  {rule}/{sym:<5} Δ={float(D):6.3f} trial {k + 1}/{n_tr}: "
                          f"R {R0[i, s, j, k]:.3f}→{Rt[-1]:.3f}  Ā(T)={Ab[-1]:.3f}  "
                          f"V_A(T)={V[-1]:.4f}  V_A/Ā²={V[-1] / Ab[-1] ** 2:.4f}  "
                          f"[{time.time() - t0:.0f}s]")

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    np.savez_compressed(cfg["out"], rules=np.array(rules), symmetries=np.array(syms),
                        operators=ops, deltas=deltas, t=t_axis,
                        R=R, Abar=Abar, VA=VA, A_final=A_final, R0=R0, omega_axis=omega_axis,
                        N=N, K=cfg["K"], mu=cfg["mu"], gamma=cfg["gamma"], A0=cfg["A0"],
                        omega_bar=cfg["omega_bar"], trunc=cfg["trunc"], T=cfg["T"], dts=cfg["dts"],
                        n_trials=n_tr, sigma0=np.nan if cfg["sigma0"] is None else cfg["sigma0"],
                        fresh_ics=cfg["fresh_ics"], seed=cfg["seed"], save_res=n_res)
    print(f"[saved] {cfg['out']}  ({time.time() - t0:.0f}s total)")


def parse_args(cfg):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--rules", nargs="+", choices=list(RULES),
                   help=f"rule families to sweep (registered: {list(RULES)})")
    p.add_argument("--symmetries", nargs="+", choices=list(SYMMETRIES),
                   help="adaptation symmetries: sym (G=cos) and/or asym (G=sin)")
    p.add_argument("--N", type=int, help="number of oscillators")
    p.add_argument("--n-trials", type=int, dest="n_trials")
    p.add_argument("--deltas", nargs="+", type=float, help="explicit Δ grid")
    p.add_argument("--mu", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--T", type=float, help="simulation time")
    p.add_argument("--sigma0", type=float, help="phase-IC width (omit for uniform phases)")
    p.add_argument("--fresh-ics", action="store_true", dest="fresh_ics",
                   help="re-draw the ICs for every simulation instead of sharing them")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", help="output .npz")
    args = p.parse_args()
    return {**cfg, **{k: v for k, v in vars(args).items() if v not in (None, False)}}


if __name__ == "__main__":
    main(parse_args(CONFIG))
