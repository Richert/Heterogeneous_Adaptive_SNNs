r"""
Sampling stability of the selected number of Lorentzians M* (referee point 2) — Fig. 1 example
==============================================================================================

For the Fig. 1 frequency distribution (Gaussian mixture of kmo_lorentzian_fit_sweep.CONFIG) we draw
`n_seeds` independent samples for every sample size N, fit the Lorentzian mixture with the same
selection rule as Fig. 1 (LM.fit: greedy over M with warm start, noise-floor acceptance
noise-floor cap D(M) <= floor_c/(6N), penalty λ, patience) and record M* for every λ.

λ only decides WHERE the greedy loop stops, not the fixed-M fits it visits (these are
λ-independent, including the warm starts). So one trace per (N, seed) with λ = 0 and unlimited
patience — run until the floor is reached or M_max — determines M* for every λ exactly
(`select_from_trace` replays LM.fit's stopping logic; checked against LM.fit in `_selfcheck`).
`python kmo_lorentzian_M_stability.py replay` re-derives the selections from the saved traces.

Output (tidy CSV in dp.mpmf): one row per (N, seed, λ): N, seed, lambda, M_star,
floor_accepted, data_loss, plus the per-M trace rows (quantity="trace").

Run in the pycobi env:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_lorentzian_M_stability.py
"""
# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from multiprocessing import Pool
import lorentzian_mixture as LM
from kmo_lorentzian_fit_sweep import CONFIG as FIG1, sample_gaussian_mixture

CFG = dict(
    N_list=[500, 1000, 2000, 5000, 10000, 20000],
    n_seeds=30,
    seed0=100,                       # sample seeds seed0 .. seed0+n_seeds-1 (Fig. 1 itself uses 1)
    lambda_list=FIG1["lambda_sweep"],  # same rows as Fig. 1(a)
    M_max=16,                        # = largest M_max of Fig. 1(a)
    floor_c=FIG1["floor_c"], patience=FIG1["patience"], n_restarts=FIG1["n_restarts"],
    delta_bounds=FIG1["delta_bounds"], method=FIG1["method"], loss=FIG1["loss"],
    n_proc=12,
    out_csv=dp.mpmf("kmo_lorentzian_M_stability.csv"),
)


def select_from_trace(trace, lam, patience, floor):
    """Replay LM.fit's floor-cap stopping logic on a λ-independent trace.
    Returns (entry, floor_reached)."""
    best, stall = None, 0
    for e in trace:
        total = e["data_loss"] + lam * e["M"]
        if best is None or total < best[1]:
            best, stall = (e, total), 0
        else:
            stall += 1
        if e["data_loss"] <= floor:
            return best[0], True
        if stall >= patience:
            break
    return best[0], False


def _job(args):
    N, seed, cfg = args
    rng = np.random.default_rng(seed)
    om = sample_gaussian_mixture(FIG1["gmm_means"], FIG1["gmm_stds"], FIG1["gmm_weights"], N, rng)
    r = LM.fit(om, cfg["delta_bounds"], M_max=cfg["M_max"], lambda_M=0.0, patience=cfg["M_max"],
               loss=cfg["loss"], n_restarts=cfg["n_restarts"], seed=seed, method=cfg["method"],
               floor_c=cfg["floor_c"])
    trace = [dict(M=t["M"], M_nominal=t["M_nominal"], data_loss=t["data_loss"]) for t in r["trace"]]
    return N, seed, trace


def _selfcheck(cfg, N=2000, seed=7):
    """select_from_trace must reproduce LM.fit for every λ."""
    _, _, trace = _job((N, seed, cfg))
    rng = np.random.default_rng(seed)
    om = sample_gaussian_mixture(FIG1["gmm_means"], FIG1["gmm_stds"], FIG1["gmm_weights"], N, rng)
    floor = cfg["floor_c"] * LM.noise_floor(N)
    for lam in cfg["lambda_list"]:
        r = LM.fit(om, cfg["delta_bounds"], M_max=cfg["M_max"], lambda_M=lam,
                   patience=cfg["patience"], loss=cfg["loss"], n_restarts=cfg["n_restarts"],
                   seed=seed, method=cfg["method"], floor_c=cfg["floor_c"])
        e, _ = select_from_trace(trace, lam, cfg["patience"], floor)
        assert e["M"] == r["M"], (lam, e["M"], r["M"])
    print(f"[selfcheck] replay == LM.fit for all λ (N={N}, seed={seed})", flush=True)


def main(cfg=CFG, replay=False):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _selfcheck(cfg)
    if replay:                                           # reuse the saved λ-independent traces
        old = pd.read_csv(cfg["out_csv"])
        old = old[old.quantity == "trace"]
        results = [(int(N), int(sd), [dict(M=int(r.M_star), M_nominal=int(r.M_nominal),
                                           data_loss=float(r.data_loss)) for r in g.itertuples()])
                   for (N, sd), g in old.groupby(["N", "seed"], sort=False)]
    else:
        jobs = [(N, cfg["seed0"] + s, cfg) for N in cfg["N_list"] for s in range(cfg["n_seeds"])]
        jobs.sort(key=lambda j: -j[0])                   # big N first for load balance
        with Pool(cfg["n_proc"]) as p:
            results = p.map(_job, jobs, chunksize=1)
    rows = []
    for N, seed, trace in results:
        floor = cfg["floor_c"] * LM.noise_floor(N)
        for lam in cfg["lambda_list"]:
            e, acc = select_from_trace(trace, lam, cfg["patience"], floor)
            rows.append(dict(quantity="selection", N=N, seed=seed, **{"lambda": lam},
                             M_star=e["M"], floor_reached=acc, data_loss=e["data_loss"]))
        for t in trace:
            rows.append(dict(quantity="trace", N=N, seed=seed, M_star=t["M"],
                             M_nominal=t["M_nominal"], data_loss=t["data_loss"]))
    df = pd.DataFrame(rows)
    df.to_csv(cfg["out_csv"], index=False)
    print(f"[saved] {cfg['out_csv']}  ({len(df)} rows)")
    sel = df[df.quantity == "selection"]
    summ = sel.groupby(["lambda", "N"])["M_star"].agg(
        mean="mean", sd="std", mode=lambda x: x.mode().iloc[0],
        p_mode=lambda x: (x == x.mode().iloc[0]).mean(), lo="min", hi="max")
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main(replay="replay" in sys.argv[1:])
