"""
Kuramoto vs Ott-Antonsen — Parameter Sweep
============================================
For a single (d, Delta0, mu) parameter set passed via the command line, run
n_trials repetitions of the KM-vs-OA comparison with different RNG seeds and
record agreement metrics in a pandas DataFrame.

Outputs one row per trial with:
    - All input parameters (N, d, M, T, K, mu, gamma, Delta0, omega0,
                            plasticity, dist, n_trials, trial, seed)
    - RMSE_R          : RMSE between R_KM(t) and R_OA(t) on a shared time grid
    - corr_A          : Pearson correlation between block-averaged KM weights
                        and OA weights at final time
    - extras for diagnostics (final-time |ΔR|, mean |A|, etc.)

CLI
---
  python sweep_kmo_oa.py --d 50 --Delta0 0.5 --mu 0.05 [other options...]

The script saves the DataFrame as a Parquet file (or CSV with --csv) under
--out_dir, with a filename encoding the parameter triple so results from many
cluster jobs can be concatenated later:

    sweep_d{d}_Delta{Delta0}_mu{mu}.parquet
"""

import argparse
import time
from pathlib import Path
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr

from config.utility_functions import lorentzian2  # same helper as base script

_EPS = 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# Initial conditions, ODEs, helpers — identical to the reference script
# ═══════════════════════════════════════════════════════════════════════════════

def make_initial_conditions(N, d, omega0, Delta0, dist, seed):
    assert N % d == 0, "N must be divisible by d"
    M = N // d
    rng = np.random.default_rng(seed)

    if dist == "uniform":
        from config.utility_functions import uniform
        omega_pop = uniform(M, omega0, Delta0)
        delta_pop = uniform(M, Delta0 / M, 0.0)
    elif dist == "lorentzian":
        n_idx = np.arange(1, M + 1)
        omega_pop = omega0 + Delta0 * np.tan(0.5 * np.pi * (2 * n_idx - M - 1) / (M + 1))
        delta_pop = Delta0 * (np.tan(0.5 * np.pi * (2 * n_idx - M - 0.5) / (M + 1))
                              - np.tan(0.5 * np.pi * (2 * n_idx - M - 1.5) / (M + 1)))
    else:
        raise ValueError(f"Invalid dist='{dist}'")

    omega_micro = np.empty(N)
    theta0 = np.empty(N)
    r0 = np.empty(M)
    psi0 = np.empty(M)

    for I in range(M):
        th = rng.uniform(-np.pi, np.pi, d)
        idx = slice(I * d, (I + 1) * d)
        omega_micro[idx] = lorentzian2(d, omega_pop[I], delta_pop[I])
        theta0[idx] = th
        z_mean = np.mean(np.exp(1j * th))
        psi0[I] = np.angle(z_mean)
        r0[I] = np.abs(z_mean)

    return omega_micro, theta0, omega_pop, delta_pop, r0, psi0


def hebbian(x):     return np.cos(x)


def antihebbian(x): return np.sin(x)


def km_ode(t, y, K, omega, mu, gamma, f):
    N = len(omega)
    theta = y[:N]
    A = y[N:].reshape(N, N)

    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    interaction = np.sum(A * np.sin(diff), axis=1)
    dtheta = omega + (K / N) * interaction

    dA = mu * f(diff) - gamma * A
    np.fill_diagonal(dA, 0.0)
    return np.concatenate([dtheta, dA.ravel()])


def km_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


def km_coarse_grain(A_fine, d):
    N = A_fine.shape[0]
    M = N // d
    Ac = np.zeros((M, M))
    for I in range(M):
        for J in range(M):
            Ac[I, J] = A_fine[I * d:(I + 1) * d, J * d:(J + 1) * d].mean()
    return Ac


def oa_ode(t, y, K, omega, delta, mu, gamma, f):
    M = len(omega)
    r = np.clip(y[:M], _EPS, 1.0 - _EPS)
    psi = y[M:2 * M]
    A = y[2 * M:].reshape(M, M)

    dpsi = psi[np.newaxis, :] - psi[:, np.newaxis]
    Ar = A * r[np.newaxis, :]
    w_cos = np.sum(Ar * np.cos(dpsi), axis=1)
    w_sin = np.sum(Ar * np.sin(dpsi), axis=1)

    dr = -delta * r + 0.5 * (1.0 - r ** 2) * w_cos * K / M
    dpsi_ = omega + 0.5 * (1.0 + r ** 2) / r * w_sin * K / M

    rr = r[np.newaxis, :] * r[:, np.newaxis]
    dA = mu * rr * f(dpsi) - gamma * A
    return np.concatenate([dr, dpsi_, dA.ravel()])


def oa_order_parameter(r, psi):
    return np.abs(np.mean(r * np.exp(1j * psi), axis=0))


# ═══════════════════════════════════════════════════════════════════════════════
# Single-trial run
# ═══════════════════════════════════════════════════════════════════════════════

def run_trial(N, d, T, K, mu, gamma, omega0, Delta0, plasticity, dist,
              seed, method, rtol, atol, n_eval):
    """
    Run one KM+OA pair with the given seed and return the agreement metrics.
    Uses dense_output so both trajectories can be sampled on a common grid.
    """
    assert N % d == 0
    M = N // d

    omega_micro, theta0, omega_pop, delta_pop, r0, psi0 = \
        make_initial_conditions(N, d, omega0, Delta0, dist, seed)

    f = hebbian if plasticity == "hebbian" else antihebbian

    A_km0 = np.ones((N, N))
    A_oa0 = np.ones((M, M))

    # KM
    y0_km = np.concatenate([theta0, A_km0.ravel()])
    sol_km = solve_ivp(km_ode, (0, T), y0_km, method=method,
                       args=(K, omega_micro, mu, gamma, f),
                       rtol=rtol, atol=atol, dense_output=True)
    if not sol_km.success:
        raise RuntimeError(f"KM failed: {sol_km.message}")

    # OA
    y0_oa = np.concatenate([r0, psi0, A_oa0.ravel()])
    sol_oa = solve_ivp(oa_ode, (0, T), y0_oa, method=method,
                       args=(K, omega_pop, delta_pop, mu, gamma, f),
                       rtol=rtol, atol=atol, dense_output=True)
    if not sol_oa.success:
        raise RuntimeError(f"OA failed: {sol_oa.message}")

    # ── Order parameter on common time grid ──────────────────────────────────
    t_grid = np.linspace(0.0, T, n_eval)
    y_km_g = sol_km.sol(t_grid)
    y_oa_g = sol_oa.sol(t_grid)
    R_km = km_order_parameter(y_km_g[:N])
    R_oa = oa_order_parameter(y_oa_g[:M], y_oa_g[M:2 * M])

    rmse_R = float(np.sqrt(np.mean((R_km - R_oa) ** 2)))
    final_dR = float(np.abs(R_km[-1] - R_oa[-1]))
    mean_R_km = float(R_km.mean())
    mean_R_oa = float(R_oa.mean())

    # ── Coupling matrix correlation at final time ────────────────────────────
    A_km_final = y_km_g[N:, -1].reshape(N, N)
    A_oa_final = y_oa_g[2 * M:, -1].reshape(M, M)
    A_km_cg = km_coarse_grain(A_km_final, d)

    a_km = A_km_cg.ravel()
    a_oa = A_oa_final.ravel()
    if np.std(a_km) < 1e-12 or np.std(a_oa) < 1e-12:
        # Pearson undefined for constant arrays — fall back to NaN
        corr_A = float("nan")
    else:
        corr_A, _ = pearsonr(a_km, a_oa)
        corr_A = float(corr_A)

    frob_A = float(np.sqrt(np.mean((A_km_cg - A_oa_final) ** 2)))

    return dict(
        rmse_R=rmse_R,
        corr_A=corr_A,
        frob_A_blk=frob_A,
        final_dR=final_dR,
        mean_R_km=mean_R_km,
        mean_R_oa=mean_R_oa,
        mean_absA_km=float(np.abs(A_km_cg).mean()),
        mean_absA_oa=float(np.abs(A_oa_final).mean()),
        nfev_km=int(sol_km.nfev),
        nfev_oa=int(sol_oa.nfev),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep over trials for one parameter set
# ═══════════════════════════════════════════════════════════════════════════════

def sweep(args):
    rows = []
    for trial in range(args.n_trials):
        seed = args.seed_base + trial
        t0 = time.time()
        print(f"\n[trial {trial + 1}/{args.n_trials}]  seed={seed}  "
              f"d={args.d}  Δ={args.Delta0}  μ={args.mu}")
        try:
            metrics = run_trial(
                N=args.N, d=args.d, T=args.T, K=args.K,
                mu=args.mu, gamma=args.gamma,
                omega0=args.omega0, Delta0=args.Delta0,
                plasticity=args.plasticity, dist=args.dist,
                seed=seed,
                method=args.method, rtol=args.rtol, atol=args.atol,
                n_eval=args.n_eval,
            )
            status = "ok"
            err = ""
        except Exception as e:
            print(f"  FAILED: {e}")
            metrics = dict(rmse_R=np.nan, corr_A=np.nan, frob_A_blk=np.nan,
                           final_dR=np.nan, mean_R_km=np.nan, mean_R_oa=np.nan,
                           mean_absA_km=np.nan, mean_absA_oa=np.nan,
                           nfev_km=-1, nfev_oa=-1)
            status = "error"
            err = str(e)

        dt = time.time() - t0
        print(f"  RMSE_R={metrics['rmse_R']:.4f}  corr_A={metrics['corr_A']:.4f}  "
              f"({dt:.1f}s)")

        rows.append(dict(
            N=args.N,
            d=args.d,
            M=args.N // args.d,
            T=args.T,
            K=args.K,
            mu=args.mu,
            gamma=args.gamma,
            omega0=args.omega0,
            Delta0=args.Delta0,
            plasticity=args.plasticity,
            dist=args.dist,
            n_trials=args.n_trials,
            trial=trial,
            seed=seed,
            wallclock=dt,
            status=status,
            error=err,
            **metrics,
        ))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="KM vs OA sweep for one parameter set.")
    # Swept parameters
    p.add_argument("--d", type=int, required=False, help="oscillators per population", default=50)
    p.add_argument("--Delta0", type=float, required=False, help="Lorentzian HWHM", default=1.0)
    p.add_argument("--mu", type=float, required=False, help="plasticity learning rate", default=0.1)

    # Fixed model parameters
    p.add_argument("--N", type=int, default=1000, help="total oscillators")
    p.add_argument("--T", type=float, default=100.0)
    p.add_argument("--K", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--plasticity", choices=["hebbian", "antihebbian"], default="antihebbian")
    p.add_argument("--dist", choices=["uniform", "lorentzian"], default="lorentzian")

    # Trial / RNG control
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--seed_base", type=int, default=42)

    # Solver
    p.add_argument("--method", default="RK45")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--n_eval", type=int, default=1000,
                   help="number of time points for shared-grid evaluation")

    # I/O
    p.add_argument("--out_dir", default="sweep_results")
    p.add_argument("--csv", action="store_true", help="save CSV instead of Parquet")
    p.add_argument("--tag", default="", help="optional tag appended to filename")

    return p.parse_args()


def main():
    args = parse_args()

    if args.N % args.d != 0:
        raise SystemExit(f"N={args.N} not divisible by d={args.d}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = sweep(args)

    # Filename encodes the parameter triple for easy concatenation
    tag = f"_{args.tag}" if args.tag else ""
    stem = f"sweep_d{args.d}_Delta{args.Delta0}_mu{args.mu}{tag}"
    out_path = out_dir / (stem + (".csv" if args.csv else ".parquet"))

    if args.csv:
        df.to_csv(out_path, index=False)
    else:
        try:
            df.to_parquet(out_path, index=False)
        except ImportError:
            print("pyarrow/fastparquet not installed; falling back to CSV.")
            out_path = out_path.with_suffix(".csv")
            df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} rows to {out_path}")
    print(df[["trial", "rmse_R", "corr_A", "final_dR",
              "mean_R_km", "mean_R_oa", "wallclock"]].to_string(index=False))


if __name__ == "__main__":
    main()

