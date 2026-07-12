r"""
Skardal-model benchmark — LMMF (Lorentzian-mixture) fit + ensemble simulation
=============================================================================

Second stage of the extended Fig. 2 pipeline. Consumes the microscopic/Skardal sweep
produced by skardal_benchmark_sweep.py and, for EACH TRIAL of EACH parameter combination
(n, coupling regime, N), performs the Lorentzian-mixture-mean-field (LMMF) benchmark:

    1. fit a weighted Lorentzian mixture to that trial's frequency samples ω_i with the
       greedy penalized-CvM algorithm (theory/lorentzian_mixture.py) — same fit settings as
       the original theory/skardal_benchmark_figure.py (FIT below),
    2. simulate the multi-ensemble Ott–Antonsen model built from that mixture
       (KFS.simulate_ensemble, PyRates+solve_ivp) from the SAME coherent initial condition
       R(0)=R0 as the microscopic run for that trial,
    3. store the ensemble dynamics R(t), the fitted mixture, and the fit diagnostics.

This is the fitting logic extracted out of the figure script, so the figure generator only
has to LOAD results. One LMMF file is written per sweep file, aligned trial-by-trial:

    <sweep>_lmmf.npz  (skardal_sweep_n<n>_<regime>_N<N>_lmmf.npz):
        t (T,), R_ensemble (n_trials, T), M (n_trials,), pvalue/data_loss (n_trials,),
        rmse_ens_micro / rmse_ens_skardal (n_trials,), and the fitted mixtures stored FLAT:
        mix_w / mix_Omega / mix_Delta (Σ_trials M,), split per trial via offsets = cumsum(M)
        (helper load_mixture below). All scalar metadata (n, regime, K, K_c, N, ...) + fit
        settings are copied in too. No pickled object arrays — trivially loadable.
    skardal_sweep_lmmf_summary.csv   tidy per-trial metrics for the figure.

Resumable and safe to run WHILE the sweep is still going: it only processes sweep files that
already exist, skips combos whose *_lmmf.npz is present (unless overwrite=True), and can be
re-run to pick up combos finished since the last run.

Run in the ``pycobi`` conda env (dev PyRates 1.2.x):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python skardal_benchmark_lmmf.py
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import kmo_lorentzian_fit_sweep as KFS                       # KFS.LM.fit + KFS.simulate_ensemble
import skardal_benchmark_sweep as SWEEP                      # reuse out_dir/out_stem + _rmse


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    in_dir=SWEEP.CONFIG["out_dir"],          # read the sweep .npz files from here
    in_stem=SWEEP.CONFIG["out_stem"],        # "skardal_sweep"
    overwrite=False,                         # if False, skip combos with an existing *_lmmf.npz
    # Lorentzian-mixture fit settings (mirrors theory/skardal_benchmark_figure.py)
    FIT=dict(delta_bounds=(0.01, 1.5),       # width bounds on the ω-scale (g_n has spread Δ≈1)
             M_max=100, alpha=0.001, lambda_M=1e-6, patience=2,
             n_restarts=10, seed=1, loss="cvm", method="slsqp"),
)


# ════════════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════════════
def sweep_files(cfg):
    """Sweep .npz files (skardal_sweep_n*_*_N*.npz), excluding LMMF outputs."""
    pat = re.compile(rf"^{re.escape(cfg['in_stem'])}_n\d+_[a-z]+_N\d+\.npz$")
    return sorted(f for f in os.listdir(cfg["in_dir"]) if pat.match(f) and not f.endswith("_lmmf.npz"))


def lmmf_path(cfg, sweep_file):
    return os.path.join(cfg["in_dir"], sweep_file[:-4] + "_lmmf.npz")


def load_mixture(d, trial):
    """Reconstruct trial's fitted mixture (w, Omega, Delta) from a flat *_lmmf.npz dict."""
    off = np.concatenate([[0], np.cumsum(d["M"])]).astype(int)
    s = slice(off[trial], off[trial + 1])
    return d["mix_w"][s], d["mix_Omega"][s], d["mix_Delta"][s]


def fit_combo(d, cfg):
    """LMMF fit + ensemble sim for every trial of one loaded sweep file. Returns (result, rows)."""
    fit = cfg["FIT"]
    n, regime = int(d["n"]), str(d["regime"])
    K, Kc, N = float(d["K"]), float(d["K_c"]), int(d["N"])
    K_ratio, Delta = float(d["K_ratio"]), float(d["Delta"])
    nt = int(d["n_trials"])
    t = d["t"]
    sim_cfg = {k: float(d[k]) for k in ("T", "dt", "dts", "rtol", "atol")}

    R_ens = np.zeros((nt, t.size))
    M_all = np.zeros(nt, dtype=np.int64)
    pval, dloss = np.zeros(nt), np.zeros(nt)
    rmse_em, rmse_es = np.zeros(nt), np.zeros(nt)
    w_list, Om_list, De_list = [], [], []
    rows = []
    for trial in range(nt):
        omega = d["omega"][trial]
        R0 = float(d["R0"][trial])
        res = KFS.LM.fit(omega, fit["delta_bounds"], M_max=fit["M_max"], alpha=fit["alpha"],
                         lambda_M=fit["lambda_M"], patience=fit["patience"], loss=fit["loss"],
                         n_restarts=fit["n_restarts"], seed=fit["seed"], method=fit["method"])
        m, M = res["model"], int(res["M"])
        t_e, R_e = KFS.simulate_ensemble(m.w, m.Omega, m.Delta, K, R0, sim_cfg,
                                         tag=f"ens_n{n}_{regime}_N{N}_t{trial}")
        L = min(R_e.size, t.size)
        R_ens[trial, :L] = R_e[:L]
        M_all[trial] = M
        pval[trial], dloss[trial] = float(res["pvalue"]), float(res["data_loss"])
        rmse_em[trial] = SWEEP._rmse(R_e, d["R_micro"][trial])
        rmse_es[trial] = SWEEP._rmse(R_e, d["R_skardal"][trial])
        w_list.append(m.w); Om_list.append(m.Omega); De_list.append(m.Delta)
        rows.append(dict(n=n, regime=regime, K_ratio=K_ratio, K=K, K_c=Kc, N=N, trial=trial,
                         M=M, pvalue=pval[trial], data_loss=dloss[trial], R0=R0,
                         R_micro_end=float(d["R_micro"][trial, -1]),
                         R_skardal_end=float(d["R_skardal"][trial, -1]),
                         R_ens_end=float(R_e[-1]),
                         rmse_ens_micro=rmse_em[trial], rmse_ens_skardal=rmse_es[trial]))

    result = dict(
        n=np.int64(n), regime=regime, K_ratio=float(K_ratio), K=float(K), K_c=float(Kc),
        N=np.int64(N), n_trials=np.int64(nt), Delta=float(Delta),
        t=t, R_ensemble=R_ens, M=M_all, pvalue=pval, data_loss=dloss,
        rmse_ens_micro=rmse_em, rmse_ens_skardal=rmse_es,
        mix_w=np.concatenate(w_list), mix_Omega=np.concatenate(Om_list),
        mix_Delta=np.concatenate(De_list),
        fit_delta_min=float(fit["delta_bounds"][0]), fit_delta_max=float(fit["delta_bounds"][1]),
        fit_M_max=np.int64(fit["M_max"]), fit_alpha=float(fit["alpha"]),
        fit_lambda_M=float(fit["lambda_M"]), fit_patience=np.int64(fit["patience"]),
        fit_n_restarts=np.int64(fit["n_restarts"]), fit_seed=np.int64(fit["seed"]),
        fit_method=fit["method"], fit_loss=fit["loss"],
    )
    return result, rows


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    files = sweep_files(cfg)
    if not files:
        raise SystemExit(f"no {cfg['in_stem']}_n*_N*.npz in {cfg['in_dir']} "
                         f"(run skardal_benchmark_sweep.py first)")
    total = len(files)
    print(f"LMMF fitting over {total} sweep files in {cfg['in_dir']}")

    all_rows, t_start = [], time.time()
    for i, f in enumerate(files):
        out = lmmf_path(cfg, f)
        if os.path.exists(out) and not cfg["overwrite"]:
            d = np.load(out, allow_pickle=False)
            for trial in range(int(d["n_trials"])):
                w, Om, De = load_mixture(d, trial)
                all_rows.append(dict(
                    n=int(d["n"]), regime=str(d["regime"]), K_ratio=float(d["K_ratio"]),
                    K=float(d["K"]), K_c=float(d["K_c"]), N=int(d["N"]), trial=trial,
                    M=int(d["M"][trial]), pvalue=float(d["pvalue"][trial]),
                    data_loss=float(d["data_loss"][trial]), R0=np.nan,
                    R_micro_end=np.nan, R_skardal_end=np.nan,
                    R_ens_end=float(d["R_ensemble"][trial, -1]),
                    rmse_ens_micro=float(d["rmse_ens_micro"][trial]),
                    rmse_ens_skardal=float(d["rmse_ens_skardal"][trial])))
            print(f"[{i+1:>3}/{total}] {f}  [skip: exists]")
            continue

        d = np.load(os.path.join(cfg["in_dir"], f), allow_pickle=False)
        t0 = time.time()
        result, rows = fit_combo(d, cfg)
        np.savez_compressed(out, **result)
        all_rows.extend(rows)
        eta = (time.time() - t_start) / (i + 1) * (total - i - 1)
        print(f"[{i+1:>3}/{total}] {f}  <M>={result['M'].mean():.1f}  "
              f"<rmse ens–micro>={result['rmse_ens_micro'].mean():.4f}  "
              f"({time.time()-t0:.1f}s, ETA {eta/60:.1f}min)")

    df = pd.DataFrame(all_rows).sort_values(["n", "K_ratio", "N", "trial"]).reset_index(drop=True)
    csv = os.path.join(cfg["in_dir"], f"{cfg['in_stem']}_lmmf_summary.csv")
    df.to_csv(csv, index=False)
    print(f"[saved] {csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
