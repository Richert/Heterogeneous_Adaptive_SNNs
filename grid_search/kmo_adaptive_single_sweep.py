r"""
Adaptive Kuramoto — single population vs. single Ott–Antonsen ensemble
======================================================================

Studies one population of Kuramoto oscillators with a Lorentzian natural-frequency
distribution (centre ω̄, half-width Δ) and *adaptive* coupling, and compares it to a
single Ott–Antonsen mean-field ensemble. (PRL manuscript Eqs. 1, 2, 9 for the
network; Eqs. 6, 7, 12 for the mean field.)

Both models are built from the shared equation templates in ``config/kuramoto.yaml``
(PyRates) and integrated with scipy.integrate.solve_ivp — only the coupling differs:
  * micro: the phase oscillator ``kmo_op`` + a per-pair adaptive edge ``kmo_adapt_op``
    (dynamic N×N weights A_ij; field s_in_i = (K/N) Σ_j A_ij e_j).
  * mean field: the OA ensemble ``oa_op`` + the mean-coupling ``oa_adapt_op``
    (Ā(t); self-coupling field h = K Ā z).
The adaptation rule G_A ∈ {cos, sin} is selected by (c_cos, c_sin) = (1,0) or (0,1).

This configuration targets the *coupling-weight variance* result of the manuscript
(Fig. 1 / Eqs. 30-37): for the cosine rule it sweeps the heterogeneity Δ over the whole
synchronized branch for two adaptation rates — μ=γ and μ=10γ — to expose the non-linear
(inverted-U) relationship between the relative weight variance V_A/Ā² and Δ, and how the
accuracy of the variance-free ensemble mean field (Eqs. 7-9) degrades where V_A is large.
Per sweep point it writes a tidy CSV (discriminated by `quantity`) holding the micro &
MF R(t) and Ā(t) traces, the microscopic weight variance V_A(t) (off-diagonal), the final
microscopic coupling matrix A (frequency-sorted, block-averaged to `save_res`, trial 0),
and the (block-averaged) frequency axis per (μ, Δ). Both Ā(t) and V_A(t) are computed over
the off-diagonal weights only (self-coupling A_ii is spurious in the mean-field limit).

Run in the ``pycobi`` conda env (dev PyRates 1.2.2 + scipy + pandas):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_single_sweep.py
"""
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from pyrates import OperatorTemplate, NodeTemplate, EdgeTemplate, CircuitTemplate, clear
from pyrates.frontend.template.population import PopulationTemplate, Connectivity

# shared Kuramoto / OA equation templates (config/kuramoto.yaml)
_KY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "kuramoto"))
def _op(name):
    return OperatorTemplate.from_yaml(f"{_KY}/{name}")


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    N=500,                         # number of microscopic oscillators
    K=1.0,                         # global coupling scale
    omega_bar=0.0,                 # Lorentzian centre (only rotates the frame)
    A0=1.0,                        # initial coupling weight (uniform A_ij(0)=A0)
    gamma=0.001,                   # weight decay γ (relaxes A→1); manuscript value
    sigma0=0.5,                    # coherent IC: θ_i(0) ~ N(0, sigma0)
    trunc=20.0,                    # Lorentzian truncated at ±trunc·Δ (tames fast tails)
    # sweep: contrast the clean inverted-U (μ=γ) against the μ≫γ rise-then-collapse case.
    # For each μ, Δ is swept up to `delta_frac_hi`× the synchronized-branch endpoint — the
    # saddle-node Δ_SN=K(γ+μ)²/(8μγ) for μ>γ, or the transcritical Δ=K/2 for μ≤γ — so the
    # full synchronized branch (and a little of the async collapse) is covered.
    mu_sweep=[0.001, 0.005],       # μ=γ (smooth inverted-U) and μ=5γ (rise → collapse)
    n_delta=20,                    # Δ points per μ
    delta_frac_hi=1.1,             # sweep Δ to this fraction of the branch endpoint
    G_A_rules=["cos", "sin", "|sin|"],             # cosine adaptation rule (manuscript main text)
    n_trunc=10,                    # Fourier-truncation order for the |sin| mean field (unused here)
    # integration (scipy solve_ivp); T ≫ 1/γ so the slow weights reach steady state
    # (equilibration confirmed complete by t≈4000 for γ=0.001; 5000 leaves margin near the fold)
    T=5000.0, dts=4.0,
    method="RK45", rtol=1e-6, atol=1e-8,
    # trials (independent random phase ICs; sweep is otherwise deterministic)
    n_trials=1,
    # storage
    save_res=100,                  # block-average the final A matrix / ω axis to this size
    seed=1,
    out_csv="/home/rgast/data/mpmf_simulations/kmo_adaptive_single_sweep.csv",
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


def delta_sweep_for(mu, K, gamma, n, frac_hi):
    """Δ grid (ascending) spanning the synchronized branch for adaptation rate μ.
    The branch ends either at the saddle-node Δ_SN = K(γ+μ)²/(8μγ) (physical only for
    μ>γ) or, for μ≤γ, at the transcritical point Δ = K/2 where R→0; we sweep up to
    `frac_hi`× that endpoint so the sweep also captures the collapse onto the async state."""
    dSN = K * (gamma + mu) ** 2 / (8.0 * mu * gamma)
    d_end = dSN if mu > gamma else min(dSN, K / 2.0)
    return np.linspace(0.02 * K, frac_hi * d_end, n)


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


def _block(vmap, match):
    """Index range of a state-variable block, by key suffix or substring."""
    v = next(val for k, val in vmap.items() if k.endswith(match) or match in k)
    if isinstance(v, (tuple, list)):
        return slice(int(v[0]), int(v[1]))
    return slice(int(v), int(v) + 1)


# ════════════════════════════════════════════════════════════════════════════
#  models (built from config/kuramoto.yaml)
# ════════════════════════════════════════════════════════════════════════════
def rule_coeffs(rule):
    """(c_cos, c_sin, c_abs) selecting the microscopic adaptation rule G_A."""
    return {"cos": (1.0, 0.0, 0.0), "sin": (0.0, 1.0, 0.0),
            "|sin|": (0.0, 0.0, 1.0)}[rule]


def _oa_adapt_op(rule, n_trunc):
    """Mean-field adaptive coupling operator (single ensemble, Eqs. 6/9). The drive
    Ā̇ = μ·F(R) + γ(1−Ā) is rule-specific and relaxes Ā toward 1 (Eq. 9), matching the
    microscopic edge; |sin| uses the Fourier truncation (App. B2):
        cos:  F = R²;  sin:  F = 0;
        |sin|: F = 2/π − (4/π) Σ_{n=1}^{n_trunc} R^{4n}/(4n²−1)   (R⁴ⁿ = (R²)^{2n})."""
    base = {"zc": "output(complex)", "Abar": "variable(1.0)", "z": "input(complex)",
            "mu": 0.1, "decay": 0.0}
    if rule == "cos":
        eqs = ["R2 = real(z)^2 + imag(z)^2", "Abar' = mu*R2 + decay*(1 - Abar)", "zc = Abar*z"]
        base["R2"] = "variable(0.0)"
    elif rule == "sin":
        eqs = ["Abar' = decay*(1 - Abar)", "zc = Abar*z"]
    elif rule == "|sin|":
        c0, c1 = 2.0 / np.pi, 4.0 / np.pi
        terms = " + ".join(f"R2^{2 * n}/{4 * n * n - 1}" for n in range(1, n_trunc + 1))
        eqs = ["R2 = real(z)^2 + imag(z)^2",
               f"Abar' = mu*({c0:.12g} - {c1:.12g}*({terms})) + decay*(1 - Abar)", "zc = Abar*z"]
        base["R2"] = "variable(0.0)"
    else:
        raise ValueError(f"unknown rule {rule}")
    return OperatorTemplate(name="oa_adapt_op", equations=eqs, variables=base)


def simulate_micro(theta0, A0, omega, K, mu, gamma, rule, cfg):
    """Single-population all-to-all adaptive Kuramoto: kmo_op + adaptive matrix edge.
    The |sin| rule is exact at the microscopic level (no Fourier truncation needed)."""
    global _RUN_ID
    _RUN_ID += 1
    N = omega.size
    c_cos, c_sin, c_abs = rule_coeffs(rule)
    node = NodeTemplate(name="osc", operators=[_op("kmo_op")])
    pop = PopulationTemplate(name="osc", node=node, n=N,
                             params={"kmo_op/omega": omega, "kmo_op/theta": theta0})
    edge = EdgeTemplate(name="adapt_edge", operators=[_op("kmo_adapt_op")])
    for var, val in dict(mu=mu, decay=gamma, c_cos=c_cos, c_sin=c_sin, c_abs=c_abs, A=A0).items():
        edge.update_var("kmo_adapt_op", var, val)
    W = (K / N) * np.ones((N, N))                          # uniform connectivity; A_ij is the edge state
    conn = Connectivity("osc/kmo_op/e", "osc/kmo_op/s_in", weights=W, edge=edge,
                        edge_var_map={"e_pre": "source", "e_post": "osc/kmo_op/e"})
    net = CircuitTemplate(f"micro{_RUN_ID}", populations={"osc": pop}, connections=[conn])
    t, Y, vmap = _run(net, cfg, f"micro{_RUN_ID}")
    theta = np.real(Y[_block(vmap, "kmo_op/theta")])       # (N, n_t)
    A_flat = np.real(Y[_block(vmap, "_flat")])             # (N², n_t) adaptive weights
    R = np.abs(np.exp(1j * theta).mean(axis=0))
    # weight statistics over the OFF-diagonal entries only (self-coupling A_ii is spurious
    # in the mean-field limit); Ā(t)=⟨A_ij⟩, V_A(t)=⟨A_ij²⟩−Ā² as in the manuscript.
    diag = np.arange(N) * N + np.arange(N)                  # flat indices of A_ii
    n_off = N * N - N
    s1 = A_flat.sum(axis=0) - A_flat[diag].sum(axis=0)
    s2 = (A_flat ** 2).sum(axis=0) - (A_flat[diag] ** 2).sum(axis=0)
    Abar = s1 / n_off                                      # off-diagonal mean Ā(t)
    VA = s2 / n_off - Abar ** 2                            # off-diagonal variance V_A(t)
    A_final = A_flat[:, -1].reshape(N, N)
    clear(net)
    return t, R, Abar, VA, A_final


def simulate_mf(Delta, mu, gamma, K, omega_bar, rule, n_trunc, R0, A0, cfg):
    """Single Ott–Antonsen ensemble with mean-coupling adaptation: oa_op + oa_adapt_op."""
    global _RUN_ID
    _RUN_ID += 1
    node = NodeTemplate(name="ens", operators=[_op("oa_op"), _oa_adapt_op(rule, n_trunc)])
    pop = PopulationTemplate(name="ens", node=node, n=1,
                             params={"oa_op/Omega": omega_bar, "oa_op/Delta": Delta,
                                     "oa_op/z": np.array([R0 + 0.0j]),
                                     "oa_adapt_op/mu": mu, "oa_adapt_op/decay": gamma,
                                     "oa_adapt_op/Abar": np.array([float(A0)])})
    conn = Connectivity("ens/oa_adapt_op/zc", "ens/oa_op/h", weights=float(K))  # self: h = K Ā z
    net = CircuitTemplate(f"mf{_RUN_ID}", populations={"ens": pop}, connections=[conn])
    t, Y, vmap = _run(net, cfg, f"mf{_RUN_ID}")
    z = Y[_block(vmap, "oa_op/z")]
    R = np.abs(z).reshape(-1)
    Abar = np.real(Y[_block(vmap, "oa_adapt_op/Abar")]).reshape(-1)
    clear(net)
    return t, R, Abar


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])
    N, K, ob, A0 = cfg["N"], cfg["K"], cfg["omega_bar"], cfg["A0"]
    gamma, res = cfg["gamma"], cfg["save_res"]
    n_trials = cfg["n_trials"]

    # independent coherent phase ICs, one per trial (drawn up front for reproducibility)
    theta0s = [rng.normal(0.0, cfg["sigma0"], N) for _ in range(n_trials)]
    R0s = [float(np.abs(np.exp(1j * th).mean())) for th in theta0s]

    print(f"adaptive Kuramoto sweep (PyRates+solve_ivp) — N={N}, K={K}, γ={gamma}, "
          f"n_trials={n_trials}, R(0)∈[{min(R0s):.3f},{max(R0s):.3f}], Ā(0)={A0}")

    rows = []

    def base():
        return dict(K=K, N=N, gamma=gamma, omega_bar=ob, A0=A0)

    # per-μ Δ grids (each spans its own synchronized branch up to the branch endpoint)
    delta_grids = {mu: delta_sweep_for(mu, K, gamma, cfg["n_delta"], cfg["delta_frac_hi"])
                   for mu in cfg["mu_sweep"]}
    for mu, dg in delta_grids.items():
        print(f"  μ={mu:<6} (μ/γ={mu / gamma:g}) Δ∈[{dg[0]:.3f},{dg[-1]:.3f}] "
              f"(Δ_SN={K * (gamma + mu) ** 2 / (8 * mu * gamma):.3f})")

    # frequency axis per (μ, Δ) (block-averaged), shared across trials/rule
    for mu, dg in delta_grids.items():
        for Delta in dg:
            omega = lorentzian_truncated(N, ob, Delta, cfg["trunc"])
            for k, om in enumerate(block_average_1d(omega, res)):
                rows.append({**base(), "quantity": "omega", "mu": mu, "Delta": float(Delta),
                             "idx": k, "value": float(om)})

    for trial in range(n_trials):
        theta0, R0 = theta0s[trial], R0s[trial]
        print(f"--- trial {trial + 1}/{n_trials}  R(0)={R0:.3f} ---")
        for rule in cfg["G_A_rules"]:
            for mu in cfg["mu_sweep"]:
                for Delta in delta_grids[mu]:
                    omega = lorentzian_truncated(N, ob, Delta, cfg["trunc"])
                    t_m, R_m, Ab_m, VA_m, A_fin = simulate_micro(theta0, A0, omega, K, mu, gamma,
                                                                 rule, cfg)
                    t_f, R_f, Ab_f = simulate_mf(Delta, mu, gamma, K, ob, rule, cfg["n_trunc"],
                                                 R0, A0, cfg)
                    rel = VA_m[-1] / Ab_m[-1] ** 2
                    print(f"  [t{trial}] G_A={rule}  μ={mu:<6} Δ={float(Delta):<6.3f} -> "
                          f"R_mic={R_m[-1]:.3f}/R_mf={R_f[-1]:.3f}  "
                          f"Ā_mic={Ab_m[-1]:.3f}/Ā_mf={Ab_f[-1]:.3f}  V_A/Ā²={rel:.4f}")

                    meta = {**base(), "G_A": rule, "Delta": float(Delta), "mu": mu, "trial": trial}
                    for t, v in zip(t_m, R_m):
                        rows.append({**meta, "quantity": "R_micro", "time": float(t), "value": float(v)})
                    for t, v in zip(t_m, Ab_m):
                        rows.append({**meta, "quantity": "Abar_micro", "time": float(t), "value": float(v)})
                    for t, v in zip(t_m, VA_m):
                        rows.append({**meta, "quantity": "VA_micro", "time": float(t), "value": float(v)})
                    for t, v in zip(t_f, R_f):
                        rows.append({**meta, "quantity": "R_mf", "time": float(t), "value": float(v)})
                    for t, v in zip(t_f, Ab_f):
                        rows.append({**meta, "quantity": "Abar_mf", "time": float(t), "value": float(v)})
                    if trial == 0:                          # final coupling matrix (one trial only)
                        Ab = block_average(A_fin, res)
                        for i in range(Ab.shape[0]):
                            for j in range(Ab.shape[1]):
                                rows.append({**meta, "quantity": "A_final", "row": i, "col": j,
                                             "value": float(Ab[i, j])})

    df = pd.DataFrame(rows).reindex(columns=[
        "quantity", "G_A", "Delta", "mu", "trial", "time", "idx", "row", "col", "value",
        "K", "N", "gamma", "omega_bar", "A0"])
    os.makedirs(os.path.dirname(cfg["out_csv"]) or ".", exist_ok=True)
    df.to_csv(cfg["out_csv"], index=False)
    print(f"[saved] {cfg['out_csv']}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
