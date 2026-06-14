r"""
Kuramoto micro vs. Lorentzian-ensemble mean-field — fit-quality sweep
=====================================================================

Compares the average-phase-coherence dynamics R(t) = |(1/N) Σ_j e^{iθ_j}| of a
globally coupled Kuramoto network with the ensemble Ott–Antonsen mean field, where
the ensembles are obtained by fitting the empirical oscillator-frequency
distribution with a weighted mixture of Lorentzians (theory/lorentzian_mixture.py).

Both models are defined as **vector-based PyRates networks** and integrated with
scipy.integrate.solve_ivp. The global (all-to-all) coupling is realized with PyRates
``Connectivity`` objects carrying a *scalar* weight: PyRates then emits a reduction
(``w * vsum(source)``) instead of forming an N×N weight matrix (see the global-
coupling path in pyrates.ir.circuit).

Microscopic network (Eq. 1, G(θ_j,θ_i)=sin(θ_j−θ_i), A_ij=1):
    θ̇_i = ω_i + (K/N) Σ_j sin(θ_j − θ_i)
         = ω_i + K ( S cos θ_i − C sin θ_i ),   C=⟨cos θ⟩, S=⟨sin θ⟩
C, S are scalar Connectivity reductions (weight 1/N) over the population outputs
cos θ / sin θ; the local nonlinearity is applied inside the oscillator operator.
The ω_i are drawn from a user-specified Gaussian mixture.

Ensemble mean field (Eqs. 6–7 with Ā_ml=1), Cartesian z_m = R_m e^{iΨ_m}:
    ż_m = (iΩ_m − Δ_m) z_m + ½ ( h − z_m² h* ),   h = K Σ_l w_l z_l
realized with a scalar Connectivity (weight K) over the weighted source w_m z_m, so
h = K·vsum(w_m z_m). (w_m, Ω_m, Δ_m) come from LorentzianMixture.fit on the ω samples.

2-D sweep over the fit meta-parameters λ (per-ensemble penalty) × M_max at FIXED α,
written to one tidy CSV (discriminated by the `quantity` column):
    omega   : micro oscillator frequencies          (idx, value=ω_i)
    R_micro : micro average phase coherence          (time, value=R)
    R_mf    : ensemble-MF coherence per sweep point  (lambda, M_max, M_star, pvalue, time, value=R)
    mixture : fitted Lorentzian params per sweep pt  (lambda, M_max, M_star, idx, w, Omega, Delta)
plus constant columns K, N, the Δ-bounds and the fixed α. M is chosen by the greedy penalized
CvM search: accept at the smallest M with 1−p < α, else return argmin_M [D(M)+λ·M] (greedy with
patience). Each fit is pruned to its non-degenerate effective order. LARGER λ => fewer ensembles.

Run in the ``pycobi`` conda env (dev PyRates 1.2.2: PopulationTemplate/Connectivity +
the scalar-weight global-coupling reduction; scipy + pandas):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_lorentzian_fit_sweep.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear, PopulationTemplate, Connectivity
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "theory")))
import lorentzian_mixture as LM

# shared Kuramoto / OA equation templates (config/kuramoto.yaml)
_KY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "kuramoto"))
def _op(name):
    return OperatorTemplate.from_yaml(f"{_KY}/{name}")


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    # microscopic oscillator frequencies: a mixture of Gaussians (means, stds, weights)
    gmm_means=[-2.0, 2.0],
    gmm_stds=[0.6, 0.6],
    gmm_weights=[0.5, 0.5],
    N=5000,                       # number of microscopic oscillators
    # coupling
    K=3.0,
    # initial condition (coherent start): θ_i(0) ~ N(0, sigma0); MF z_m(0)=R(0)
    sigma0=0.5,
    # integration (solve_ivp / RK45)
    T=30.0,
    dt=1e-2,                      # initial/max step hint + base for sampling
    dts=0.1,                      # sampling step for R(t)
    rtol=1e-6, atol=1e-8,
    # Lorentzian-mixture fit
    delta_bounds=(1e-4, 1e2),     # hard bounds on every ensemble width
    n_restarts=10,                # restarts per fixed-M fit; too few may leave higher-M
                                  # fits stuck at lower-M local minima (zero relative improvement),
                                  # which spuriously trips the greedy β stop early
    loss="cvm",
    method="slsqp",               # constraint handling: "slsqp" (natural params + Σw=1) or "softmax"
    # fixed GoF acceptance level (kept strict so the λ penalty is what selects M)
    alpha=1e-2,
    # meta-parameter sweep over λ: the per-ensemble penalty in M* = argmin[D(M)+λ·M]
    # (D is the CvM loss). LARGER λ => FEWER ensembles.
    lambda_sweep=[1e-5, 1e-4, 1e-3],
    patience=3,                   # stop after this many non-improving steps in D+λM, return argmin
    M_max_sweep=[1, 2, 4, 8, 16],
    # misc
    seed=1,
    out_csv="/home/rgast/data/mpmf_simulations/kmo_lorentzian_sweep.csv",
)


# ════════════════════════════════════════════════════════════════════════════
#  frequency sampling
# ════════════════════════════════════════════════════════════════════════════
def sample_gaussian_mixture(means, stds, weights, N, rng):
    means, stds = np.asarray(means, float), np.asarray(stds, float)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    comp = rng.choice(len(means), size=N, p=weights)
    return rng.normal(means[comp], stds[comp])


# ════════════════════════════════════════════════════════════════════════════
#  PyRates vector-based models + solve_ivp
# ════════════════════════════════════════════════════════════════════════════
def _var_slice(vmap, suffix):
    """Index range of a state variable in the flat state vector, robust to PyRates
    returning a scalar int for size-1 variables (vs. a (start, stop) tuple)."""
    v = next(val for k, val in vmap.items() if k.endswith(suffix))
    if isinstance(v, (tuple, list)):
        return slice(int(v[0]), int(v[1]))
    return slice(int(v), int(v) + 1)


def _run(net, name, cfg, precision):
    """get_run_func -> solve_ivp. Returns (t_eval, state_trajectory, vmap)."""
    func, args, keys, vmap = net.get_run_func(name, step_size=cfg["dt"], backend="numpy",
                                              vectorize=True, clear=False,
                                              float_precision=precision)
    y0 = np.asarray(args[1])
    extra = args[2:]                                  # (dy_buffer, *params)

    def f(t, y):
        return np.array(func(t, y, *extra))

    t_eval = np.arange(0.0, cfg["T"], cfg["dts"])
    sol = solve_ivp(f, (0.0, cfg["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["dt"])
    return t_eval, sol.y, vmap


def simulate_micro(omega, K, theta0, cfg, tag="micro"):
    """Globally coupled Kuramoto network: shared kmo_op + scalar mean-field reduction."""
    N = omega.size
    node = NodeTemplate(name="osc", operators=[_op("kmo_op")])
    pop = PopulationTemplate(name="osc", node=node, n=N,
                             params={"kmo_op/omega": omega, "kmo_op/theta": theta0})
    # scalar (K/N) Connectivity on e -> s_in = (K/N) vsum(e) = K·Z (no N×N matrix)
    conn = Connectivity("osc/kmo_op/e", "osc/kmo_op/s_in", weights=K / N)
    net = CircuitTemplate("kmo_micro", populations={"osc": pop}, connections=[conn])
    t, Y, vmap = _run(net, f"{tag}_vf", cfg, "complex128")
    theta = np.real(Y[_var_slice(vmap, "kmo_op/theta")])   # (N, T)
    R = np.abs(np.exp(1j * theta).mean(axis=0))
    clear(net)
    return t, R


def simulate_ensemble(w, Omega, Delta, K, R0, cfg, tag="ens"):
    """Ensemble Ott–Antonsen mean field: shared oa_op + ens_coupling_op (zc = w_m z_m)."""
    M = w.size
    node = NodeTemplate(name="ens", operators=[_op("oa_op"), _op("ens_coupling_op")])
    pop = PopulationTemplate(name="ens", node=node, n=M,
                             params={"oa_op/Omega": Omega, "oa_op/Delta": Delta,
                                     "oa_op/z": np.full(M, R0 + 0.0j),
                                     "ens_coupling_op/wm": w})
    # scalar (K) Connectivity over zc = w_m z_m  ->  h = K Σ_l w_l z_l
    conn = Connectivity("ens/ens_coupling_op/zc", "ens/oa_op/h", weights=float(K))
    net = CircuitTemplate("kmo_ens", populations={"ens": pop}, connections=[conn])
    t, Y, vmap = _run(net, f"{tag}_vf", cfg, "complex128")
    z = Y[_var_slice(vmap, "oa_op/z")]                     # (M, T) complex
    R = np.abs(w @ z)
    clear(net)
    return t, R


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])

    # microscopic frequencies + coherent initial condition
    omega = sample_gaussian_mixture(cfg["gmm_means"], cfg["gmm_stds"],
                                    cfg["gmm_weights"], cfg["N"], rng)
    theta0 = rng.normal(0.0, cfg["sigma0"], cfg["N"])
    R0 = float(np.abs(np.exp(1j * theta0).mean()))

    print(f"micro Kuramoto (PyRates+solve_ivp): N={cfg['N']}, K={cfg['K']}, "
          f"ω∈[{omega.min():.2f},{omega.max():.2f}], R(0)={R0:.3f}")
    t_mic, R_mic = simulate_micro(omega, cfg["K"], theta0, cfg)
    print(f"  micro R(end)={R_mic[-1]:.3f}")

    rows = []
    Kc, Nc = cfg["K"], cfg["N"]
    dmin, dmax = cfg["delta_bounds"]

    def base():                                    # constant columns on every row
        return dict(K=Kc, N=Nc, delta_min=dmin, delta_max=dmax, alpha=cfg["alpha"])

    # micro frequencies
    for i, om in enumerate(omega):
        rows.append({**base(), "quantity": "omega", "idx": i, "value": float(om)})
    # micro coherence
    for t, R in zip(t_mic, R_mic):
        rows.append({**base(), "quantity": "R_micro", "time": float(t), "value": float(R)})

    # 2-D meta-parameter sweep over (lambda, M_max)  [alpha fixed = cfg["alpha"]]
    si = 0
    for M_max in cfg["M_max_sweep"]:
        for lam in cfg["lambda_sweep"]:
            res = LM.fit(omega, cfg["delta_bounds"], M_max=M_max, alpha=cfg["alpha"],
                         lambda_M=lam, patience=cfg["patience"], loss=cfg["loss"],
                         n_restarts=cfg["n_restarts"], seed=cfg["seed"], method=cfg["method"])
            m = res["model"]
            M_star = res["M"]
            t_mf, R_mf = simulate_ensemble(m.w, m.Omega, m.Delta, cfg["K"], R0, cfg,
                                           tag=f"ens{si}")
            si += 1
            print(f"  λ={lam:<8g} M_max={M_max:2d} -> M*={M_star}  p={res['pvalue']:.3f}  "
                  f"R_mf(end)={R_mf[-1]:.3f}  (data_loss={res['data_loss']:.2e})")
            # ensemble-MF coherence
            for t, R in zip(t_mf, R_mf):
                rows.append({**base(), "quantity": "R_mf", "lambda": lam, "M_max": M_max,
                             "M_star": M_star, "pvalue": float(res["pvalue"]),
                             "time": float(t), "value": float(R)})
            # fitted Lorentzian mixture parameters
            for k in range(M_star):
                rows.append({**base(), "quantity": "mixture", "lambda": lam, "M_max": M_max,
                             "M_star": M_star, "idx": k, "w": float(m.w[k]),
                             "Omega": float(m.Omega[k]), "Delta": float(m.Delta[k])})

    df = pd.DataFrame(rows).reindex(columns=[
        "quantity", "lambda", "M_max", "M_star", "pvalue", "time", "idx", "value",
        "w", "Omega", "Delta", "K", "N", "delta_min", "delta_max", "alpha"])
    os.makedirs(os.path.dirname(cfg["out_csv"]) or ".", exist_ok=True)
    df.to_csv(cfg["out_csv"], index=False)
    print(f"[saved] {cfg['out_csv']}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
