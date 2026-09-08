r"""
Skardal-model benchmark — extensive parameter sweep (micro + Skardal mean field)
================================================================================

Extended version of theory/skardal_benchmark_simulate.py for the improved Fig. 2 of the
LMMF manuscript. For the family of RATIONAL frequency distributions (Skardal, Phys. Rev.
E 98, 022207 (2018), Eq. 14),

    g_n(ω) = n·sin(π/2n)·Δ^{2n-1} / [ π (ω^{2n} + Δ^{2n}) ] ,   n = 1, 2, 3, ...

(n=1 = Cauchy; n→∞ = uniform on [−Δ,Δ]), this script benchmarks the microscopic Kuramoto
network against the exact Skardal Ott–Antonsen reduction (n complex equations) across a
FOUR-DIMENSIONAL sweep:

    * exponent n          — an extensive geometric sweep (distribution shape / MF dimension)
    * coupling regime     — K set RELATIVE to the n-dependent synchronisation threshold
                            K_c(n) = 2Δ / [ n·sin(π/2n) ]  (= 2/(π g_n(0))); one subcritical,
                            one critical (K=K_c), one supercritical point per n
    * network size N      — three choices, to expose finite-size convergence micro → Skardal
    * trials              — n_trials independent frequency/phase realisations per combination

For each (n, regime, N, trial) it draws N frequencies from g_n, integrates the microscopic
Kuramoto network (KFS.simulate_micro, PyRates+solve_ivp) and the Skardal mean field
(SK.simulate_skardal) from a shared coherent initial condition R(0)=R0, and records R(t).

Output (resumable — existing per-combo files are skipped unless overwrite=True):
    <out_dir>/skardal_sweep_n<n>_<regime>_N<N>.npz    one file per (n, regime, N):
        t (T,), R_micro (n_trials,T), R_skardal (n_trials,T), R0/g0_emp/rmse (n_trials,),
        omega (n_trials,N), analytic g_n grid, and all scalar metadata (n, Δ, K, K_c, ...)
    <out_dir>/skardal_sweep_summary.csv               tidy per-trial metrics for plotting

Run in the ``pycobi`` conda env (dev PyRates 1.2.x):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python skardal_benchmark_sweep.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # grid_search: KFS.simulate_micro
sys.path.insert(0, os.path.join(_HERE, "..", "theory"))     # skardal_benchmark_simulate
import kmo_lorentzian_fit_sweep as KFS                       # reuse simulate_micro (PyRates net)
import skardal_benchmark_simulate as SK                      # reuse gn_density/sample_gn/simulate_skardal


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    # exponents n of Eq. 14 (geometric: n=1 Cauchy ... n=256 ≈ uniform box)
    n_exponents=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    Delta=1.0,                                # spread parameter of g_n
    # coupling regime: K = ratio · K_c(n), with K_c(n) = 2Δ/[n·sin(π/2n)] the n-dependent threshold
    K_ratios={"subcritical": 0.8, "critical": 1.0, "supercritical": 1.2},
    # network sizes (finite-size convergence micro -> Skardal)
    N_list=[200, 1000, 5000],
    n_trials=5,                               # independent realisations per (n, regime, N)
    sigma0=0.5,                               # coherent IC θ_i(0) ~ N(0, σ0); z(0)=R0
    # integration (shared by micro / Skardal; solve_ivp RK45)
    T=100.0, dt=1e-2, dts=0.1, rtol=1e-6, atol=1e-8,
    base_seed=1,                              # per-trial seeds spawned from this
    overwrite=False,                          # if False, skip (n, regime, N) with an existing npz
    out_dir="/home/rgast/data/mpmf_simulations/skardal_sweep",
    out_stem="skardal_sweep",
)


# ════════════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════════════
def K_critical(n, Delta):
    """Onset coupling K_c(n) = 2/(π g_n(0)) = 2Δ / [ n·sin(π/2n) ] (n=1 → 2Δ, n→∞ → 4Δ/π)."""
    return 2.0 * Delta / (n * np.sin(np.pi / (2 * n)))


def _rmse(a, b):
    L = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:L] - b[:L]) ** 2)))


def _trial_rng(cfg, n, N, trial):
    """Reproducible, independent RNG per (base_seed, n, N, trial)."""
    return np.random.default_rng(np.random.SeedSequence([cfg["base_seed"], int(n), int(N), int(trial)]))


def run_combo(n, regime, ratio, N, cfg):
    """Simulate all trials for one (n, regime, N) combination. Returns (result-dict, summary-rows)."""
    Delta = cfg["Delta"]
    Kc = K_critical(n, Delta)
    K = ratio * Kc
    g0 = float(SK.gn_density(np.array([0.0]), n, Delta)[0])
    sim_cfg = {k: cfg[k] for k in ("T", "dt", "dts", "rtol", "atol")}

    nt = cfg["n_trials"]
    t_ref = None
    R_mic_all, R_sk_all = [], []
    R0_all, g0emp_all, rmse_all = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    omega_all = np.zeros((nt, N))
    rows = []
    for trial in range(nt):
        rng = _trial_rng(cfg, n, N, trial)
        # over="ignore": for large n, ω^{2n} overflows to inf outside [−Δ,Δ], correctly making
        # g_n → 0 there (the uniform-box limit) — a benign overflow inside sample_gn, not an error.
        with np.errstate(over="ignore"):
            omega = SK.sample_gn(n, Delta, N, rng)
        theta0 = rng.normal(0.0, cfg["sigma0"], N)
        R0 = float(np.abs(np.exp(1j * theta0).mean()))
        hw = 0.05 * Delta                                    # empirical density at ω=0 (sets effective K_c)
        g0_emp = float(np.mean(np.abs(omega) < hw) / (2 * hw))

        t_m, R_m = KFS.simulate_micro(omega, K, theta0, sim_cfg, tag=f"mic_n{n}_{regime}_N{N}_t{trial}")
        t_s, R_s = SK.simulate_skardal(n, Delta, K, R0, sim_cfg, tag=f"sk_n{n}_{regime}_N{N}_t{trial}")
        t_ref = t_m
        dev = _rmse(R_m, R_s)

        R_mic_all.append(R_m); R_sk_all.append(R_s)
        omega_all[trial] = omega
        R0_all[trial], g0emp_all[trial], rmse_all[trial] = R0, g0_emp, dev
        rows.append(dict(n=n, regime=regime, K_ratio=ratio, K=K, K_c=Kc, N=N, trial=trial,
                         Delta=Delta, g0_exact=g0, g0_emp=g0_emp, R0=R0,
                         R_micro_end=float(R_m[-1]), R_skardal_end=float(R_s[-1]), rmse=dev))

    # pad/truncate to a common length (solve_ivp t_eval is fixed, so lengths match, but be safe)
    L = min(min(len(r) for r in R_mic_all), min(len(r) for r in R_sk_all), len(t_ref))
    R_mic_all = np.array([r[:L] for r in R_mic_all])
    R_sk_all = np.array([r[:L] for r in R_sk_all])
    t_ref = t_ref[:L]
    gx = np.linspace(-5 * Delta, 5 * Delta, 1000)
    with np.errstate(over="ignore"):                         # box limit: g_n → 0 outside [−Δ,Δ]
        g_density = SK.gn_density(gx, n, Delta)

    result = dict(
        n=np.int64(n), regime=regime, K_ratio=float(ratio), K=float(K), K_c=float(Kc),
        N=np.int64(N), n_trials=np.int64(nt), Delta=float(Delta), sigma0=float(cfg["sigma0"]),
        g0_exact=float(g0), n_mf_skardal=np.int64(n),
        T=float(cfg["T"]), dt=float(cfg["dt"]), dts=float(cfg["dts"]),
        rtol=float(cfg["rtol"]), atol=float(cfg["atol"]),
        t=t_ref, R_micro=R_mic_all, R_skardal=R_sk_all,
        R0=R0_all, g0_emp=g0emp_all, rmse=rmse_all, omega=omega_all,
        g_omega=gx, g_density=g_density,
    )
    return result, rows


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def out_path(cfg, n, regime, N):
    return os.path.join(cfg["out_dir"], f"{cfg['out_stem']}_n{n}_{regime}_N{N}.npz")


def main(cfg=CONFIG):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    combos = [(n, regime, ratio, N)
              for n in cfg["n_exponents"]
              for regime, ratio in cfg["K_ratios"].items()
              for N in cfg["N_list"]]
    total = len(combos)
    print(f"Skardal sweep: {len(cfg['n_exponents'])} n × {len(cfg['K_ratios'])} regimes × "
          f"{len(cfg['N_list'])} N × {cfg['n_trials']} trials = {total} combos "
          f"({total * cfg['n_trials']} micro + {total * cfg['n_trials']} Skardal sims)")

    all_rows, t_start = [], time.time()
    for i, (n, regime, ratio, N) in enumerate(combos):
        path = out_path(cfg, n, regime, N)
        Kc = K_critical(n, cfg["Delta"])
        tag = f"[{i+1:>3}/{total}] n={n:<4} {regime:<13} N={N:<6} K={ratio*Kc:.3f} (K_c={Kc:.3f})"
        if os.path.exists(path) and not cfg["overwrite"]:
            d = np.load(path, allow_pickle=False)                       # gather rows for the CSV
            for trial in range(int(d["n_trials"])):
                all_rows.append(dict(n=int(d["n"]), regime=str(d["regime"]), K_ratio=float(d["K_ratio"]),
                                     K=float(d["K"]), K_c=float(d["K_c"]), N=int(d["N"]), trial=trial,
                                     Delta=float(d["Delta"]), g0_exact=float(d["g0_exact"]),
                                     g0_emp=float(d["g0_emp"][trial]), R0=float(d["R0"][trial]),
                                     R_micro_end=float(d["R_micro"][trial, -1]),
                                     R_skardal_end=float(d["R_skardal"][trial, -1]),
                                     rmse=float(d["rmse"][trial])))
            print(f"{tag}  [skip: exists]")
            continue

        t0 = time.time()
        result, rows = run_combo(n, regime, ratio, N, cfg)
        np.savez_compressed(path, **result)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        eta = (time.time() - t_start) / (i + 1) * (total - i - 1)
        print(f"{tag}  <rmse>={result['rmse'].mean():.4f}  micro R_end={result['R_micro'][:,-1].mean():.3f}  "
              f"Skardal R_end={result['R_skardal'][:,-1].mean():.3f}  ({elapsed:.1f}s, ETA {eta/60:.1f}min)")

    # tidy summary CSV (one row per trial) for the figure
    df = pd.DataFrame(all_rows).sort_values(["n", "K_ratio", "N", "trial"]).reset_index(drop=True)
    csv = os.path.join(cfg["out_dir"], f"{cfg['out_stem']}_summary.csv")
    df.to_csv(csv, index=False)
    print(f"[saved] {csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
