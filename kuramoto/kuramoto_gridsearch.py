"""
Systematic Sweep: KMO Microscopic vs. OA Mean-Field Ensemble
==============================================================
For each trial, simulate the microscopic Kuramoto network for each value of
Δ (the width of the frequency distribution), then sweep over (M, μ) in
parallel and simulate the OA mean-field equations.  For each (trial, Δ, M, μ)
combination, compute:

    - rmse_R : RMSE between R_KMO(t) and R_OA(t) sampled on a common time grid
    - corr_A : Pearson correlation between the block-averaged KMO weight
               matrix A^{KMO}_{IJ} and the OA weight matrix A^{OA}_{IJ}
               at the final time step (flattened to vectors)

Frequency distributions
-----------------------
"uniform":
    Microscopic: ω_k drawn from Uniform(ω̄ - Δ, ω̄ + Δ)
    Ensembles:   M centres ω̄_m equidistantly spaced on the same interval
                 Δ_m = Δ / (2M)  for all m
"lorentzian":
    Microscopic: ω_k drawn from Cauchy(ω̄, Δ)
    Ensembles:   centres and HWHM follow Gast et al. PRE 2021, Eq. 22:
                 ω̄_m = ω̄ + Δ tan(π(2m-M-1)/(2(M+1)))
                 Δ_m = Δ (tan(π(2m-M-1/2)/(2(M+1)))
                       - tan(π(2m-M-3/2)/(2(M+1))))
                 Population weights w_m = 1/M  (equal-weight quantile grid).

Initial conditions
------------------
- KMO: θ_k ~ Uniform(-π, π) drawn once per (trial, Δ); A^{KMO} = ones((N, N)).
- OA:  For each (M, μ), the same microscopic phases are used to compute
       r_m^0, ψ_m^0 from the empirical mean of e^{iθ_k} over oscillators k
       assigned to ensemble m by:
         - uniform: k goes to the m for which |ω_k - ω̄_m| is minimal
         - lorentzian: k goes to the m maximising w_m C(ω_k; ω̄_m, Δ_m)

Output
------
A pandas DataFrame with columns
    [trial, dist, Delta, M, mu, rmse_R, corr_A, status]
saved as CSV.

Usage
-----
    python sweep_kmo_vs_oa.py [--dist {uniform,lorentzian}] [--n_workers K]
                              [--out output.csv]
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import cauchy, pearsonr

_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# Frequency distributions
# ═══════════════════════════════════════════════════════════════════════════════

def sample_microscopic_frequencies(N, dist, omega0, Delta, rng, cauchy_clip=50.0):
    if dist == "uniform":
        return rng.uniform(omega0 - Delta, omega0 + Delta, N)
    elif dist == "lorentzian":
        # Truncated Cauchy: redraw any |ω - ω0| > cauchy_clip * Delta
        omega = omega0 + Delta * rng.standard_cauchy(N)
        bad = np.abs(omega - omega0) > cauchy_clip * Delta
        while bad.any():
            omega[bad] = omega0 + Delta * rng.standard_cauchy(bad.sum())
            bad = np.abs(omega - omega0) > cauchy_clip * Delta
        return omega


def ensemble_parameters(M, dist, omega0, Delta):
    """
    Build the (omega_m, Delta_m, w_m) parameters for the M-ensemble OA model.

    Returns
    -------
    omega_pop : (M,)
    delta_pop : (M,)
    w_pop     : (M,) ensemble weights (sum to 1)
    """
    if dist == "uniform":
        # Equidistant centres in (ω0-Δ, ω0+Δ); each ensemble equally weighted.
        # Cell centres of M equal bins.
        edges = np.linspace(omega0 - Delta, omega0 + Delta, M + 1)
        omega_pop = 0.5 * (edges[:-1] + edges[1:])
        delta_pop = np.full(M, Delta / (2 * M))
        w_pop     = np.full(M, 1.0 / M)
        return omega_pop, delta_pop, w_pop

    elif dist == "lorentzian":
        # Gast et al. PRE 2021, Eq. 22 (renamed from η to ω here).
        m = np.arange(1, M + 1)
        omega_pop = omega0 + Delta * np.tan(0.5 * np.pi * (2*m - M - 1) / (M + 1))
        delta_pop = Delta * (
            np.tan(0.5 * np.pi * (2*m - M - 0.5) / (M + 1))
            - np.tan(0.5 * np.pi * (2*m - M - 1.5) / (M + 1))
        )
        w_pop = np.full(M, 1.0 / M)
        return omega_pop, delta_pop, w_pop

    raise ValueError(f"Unknown dist={dist!r}")


def assign_oscillators_to_ensembles(omega_micro, omega_pop, delta_pop, w_pop, dist):
    """
    Hard-assign each microscopic oscillator k to one of M ensembles.

    For uniform: nearest centre by |ω_k - ω̄_m|.
    For lorentzian: most-likely component by w_m C(ω_k; ω̄_m, Δ_m).

    Returns
    -------
    labels : (N,) int array with values in [0, M)
    """
    if dist == "uniform":
        # Nearest centre
        diff = np.abs(omega_micro[:, None] - omega_pop[None, :])
        return np.argmin(diff, axis=1)

    elif dist == "lorentzian":
        # Posterior probability (log-space for numerical stability)
        log_post = np.stack([
            np.log(w_pop[I] + 1e-300)
            + cauchy.logpdf(omega_micro, omega_pop[I], delta_pop[I])
            for I in range(len(w_pop))
        ], axis=0)
        return np.argmax(log_post, axis=0)

    raise ValueError(f"Unknown dist={dist!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# ODEs
# ═══════════════════════════════════════════════════════════════════════════════

def hebbian(x):     return np.cos(x)
def antihebbian(x): return np.sin(x)


def km_ode(t, y, K, omega, mu, gamma, f):
    N     = len(omega)
    theta = y[:N]
    A     = y[N:].reshape(N, N)

    diff        = theta[np.newaxis, :] - theta[:, np.newaxis]
    interaction = np.sum(A * np.sin(diff), axis=1)
    dtheta      = omega + (K / N) * interaction

    dA = mu * f(diff) - gamma * A
    np.fill_diagonal(dA, 0.0)
    return np.concatenate([dtheta, dA.ravel()])


def km_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


def km_coarse_grain_labels(A_fine, labels, M):
    """Block-average the full N×N weight matrix using arbitrary labels."""
    A_cg = np.zeros((M, M))
    for I in range(M):
        idx_I = np.where(labels == I)[0]
        if idx_I.size == 0:
            continue
        for J in range(M):
            idx_J = np.where(labels == J)[0]
            if idx_J.size == 0:
                continue
            A_cg[I, J] = A_fine[np.ix_(idx_I, idx_J)].mean()
    return A_cg


def oa_ode(t, y, K, omega, delta, mu, gamma, f, w):
    """Weighted-population OA mean-field ODE."""
    M   = len(omega)
    r   = np.clip(y[:M], _EPS, 1.0 - _EPS)
    psi = y[M:2*M]
    A   = y[2*M:].reshape(M, M)

    dpsi  = psi[np.newaxis, :] - psi[:, np.newaxis]
    Ar    = A * r[np.newaxis, :]
    w_cos = Ar * np.cos(dpsi)
    w_sin = Ar * np.sin(dpsi)

    dr    = -delta * r + 0.5 * (1.0 - r**2) * K * (w_cos @ w)
    dpsi_ = omega     + 0.5 * (1.0 + r**2) / r * K * (w_sin @ w)

    rr = r[np.newaxis, :] * r[:, np.newaxis]
    dA = mu * rr * f(dpsi) - gamma * A
    return np.concatenate([dr, dpsi_, dA.ravel()])


def oa_order_parameter(r, w, psi):
    return np.abs(w @ (r * np.exp(1j * psi)))


# ═══════════════════════════════════════════════════════════════════════════════
# Single-trial KMO + per-(M, μ) OA helpers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KMResult:
    t_grid: np.ndarray      # shared time grid
    R_km:   np.ndarray      # R_KMO sampled on t_grid
    A_km_final: np.ndarray  # final N×N weight matrix
    theta0: np.ndarray
    omega_micro: np.ndarray


def run_km(N, T, K, mu, gamma, omega_micro, theta0, plasticity,
           n_eval, method, rtol, atol):
    """Run the microscopic Kuramoto simulation once for given (Δ, trial)."""
    f     = hebbian if plasticity == "hebbian" else antihebbian
    A_km0 = np.ones((N, N))
    y0_km = np.concatenate([theta0, A_km0.ravel()])

    # Linearly spaced sampling times on the closed interval [0, T].
    t_grid = np.linspace(0.0, T, n_eval)
    sol = solve_ivp(km_ode, (0, T), y0_km, method=method,
                    args=(K, omega_micro, mu, gamma, f),
                    rtol=rtol, atol=atol,
                    t_eval=t_grid, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"KMO solve_ivp failed: {sol.message}")

    y_grid = sol.y
    theta_grid = y_grid[:N]
    R_km   = km_order_parameter(theta_grid)
    A_final = y_grid[N:, -1].reshape(N, N)
    return KMResult(t_grid=t_grid, R_km=R_km, A_km_final=A_final,
                    theta0=theta0, omega_micro=omega_micro)


def run_oa_task(task):
    """
    Worker function executed in a multiprocessing pool.

    Each task contains everything needed to:
      - build OA initial conditions from the shared KMO phases,
      - integrate the OA ODE,
      - compute (rmse_R, corr_A) versus the shared KMO results.

    Returns a single-row dict.
    """
    (trial, dist, Delta, M, mu_val,
     omega0, gamma, plasticity, T, K,
     omega_micro, theta0,
     A_km_final, t_grid, R_km,
     N, n_eval, method, rtol, atol) = task

    try:
        f = hebbian if plasticity == "hebbian" else antihebbian

        # Ensemble parameters and oscillator assignments
        omega_pop, delta_pop, w_pop = ensemble_parameters(M, dist, omega0, Delta)
        labels = assign_oscillators_to_ensembles(
            omega_micro, omega_pop, delta_pop, w_pop, dist
        )

        # Initial OA state from empirical means
        r0   = np.empty(M)
        psi0 = np.empty(M)
        for I in range(M):
            mask = labels == I
            if mask.sum() == 0:
                r0[I], psi0[I] = _EPS, 0.0
                continue
            z = np.mean(np.exp(1j * theta0[mask]))
            r0[I]   = np.clip(np.abs(z), _EPS, 1.0 - _EPS)
            psi0[I] = np.angle(z)

        A_oa0 = np.ones((M, M))
        y0_oa = np.concatenate([r0, psi0, A_oa0.ravel()])

        sol = solve_ivp(oa_ode, (0, T), y0_oa, method=method,
                        args=(K, omega_pop, delta_pop, mu_val, gamma, f, w_pop),
                        rtol=rtol, atol=atol,
                        t_eval=t_grid, dense_output=False)
        if not sol.success:
            raise RuntimeError(f"OA solve_ivp failed: {sol.message}")

        y_grid = sol.y
        r_grid   = y_grid[:M]
        psi_grid = y_grid[M:2*M]
        A_oa_final = y_grid[2*M:, -1].reshape(M, M)
        R_oa = oa_order_parameter(r_grid, w_pop, psi_grid)

        # Metrics
        rmse_R = float(np.sqrt(np.mean((R_km - R_oa) ** 2)))

        A_km_cg = km_coarse_grain_labels(A_km_final, labels, M)
        a_km = A_km_cg.ravel()
        a_oa = A_oa_final.ravel()
        if np.std(a_km) < 1e-12 or np.std(a_oa) < 1e-12:
            corr_A = float("nan")
        else:
            corr_A, _ = pearsonr(a_km, a_oa)
            corr_A    = float(corr_A)

        return dict(
            trial=trial, dist=dist, plasticity=plasticity,
            Delta=Delta, M=M, mu=mu_val,
            rmse_R=rmse_R, corr_A=corr_A, status="ok", error="",
        )

    except Exception as e:
        return dict(
            trial=trial, dist=dist, plasticity=plasticity,
            Delta=Delta, M=M, mu=mu_val,
            rmse_R=float("nan"), corr_A=float("nan"),
            status="error", error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep driver
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    *,
    dist,
    n_trials,
    Delta_values,
    M_values,
    mu_values,
    plasticity_values,
    N, T, K, gamma, omega0,
    n_eval,
    method, rtol, atol,
    seed_base,
    n_workers,
    verbose=True,
):
    rows = []
    t_total = time.time()

    for trial in range(n_trials):
        seed = seed_base + trial
        rng  = np.random.default_rng(seed)

        # ── Draw initial phases once per trial; reused for all Δ ──────────────
        theta0 = rng.uniform(-np.pi, np.pi, N)

        for Delta in Delta_values:
            # Sample microscopic frequencies for THIS (trial, Δ)
            omega_micro = sample_microscopic_frequencies(
                N, dist, omega0, Delta, rng
            )

            for plasticity in plasticity_values:
                # ── Run KMO once for this (trial, Δ, plasticity) ─────────────
                t0 = time.time()
                try:
                    km_res = run_km(
                        N=N, T=T, K=K, mu=0.0, gamma=0.0,  # KMO weights frozen here
                        omega_micro=omega_micro, theta0=theta0,
                        plasticity=plasticity, n_eval=n_eval,
                        method=method, rtol=rtol, atol=atol,
                    )
                except Exception as e:
                    # If KMO failed, mark all (M, μ) entries as failed
                    for M, mu_val in product(M_values, mu_values):
                        rows.append(dict(
                            trial=trial, dist=dist,
                            plasticity=plasticity,
                            Delta=Delta, M=M, mu=mu_val,
                            rmse_R=float("nan"), corr_A=float("nan"),
                            status="error_km", error=str(e),
                        ))
                    continue
                t_km = time.time() - t0
                if verbose:
                    print(f"[trial {trial+1}/{n_trials}  Δ={Delta:g}  "
                          f"plast={plasticity}]  KMO done in {t_km:.1f}s")

                # NOTE: above call used mu=0, gamma=0 so weights stay at A=1.
                # If the comparison needs adaptive KMO weights, re-run with the
                # appropriate μ here.  See the main() wrapper.

                # ── Build all (M, μ) tasks for the OA sweep ─────────────────
                tasks = []
                for M, mu_val in product(M_values, mu_values):
                    tasks.append((
                        trial, dist, Delta, M, mu_val,
                        omega0, gamma, plasticity, T, K,
                        omega_micro, theta0,
                        km_res.A_km_final, km_res.t_grid, km_res.R_km,
                        N, n_eval, method, rtol, atol,
                    ))

                t0 = time.time()
                if n_workers > 1:
                    with Pool(n_workers) as pool:
                        chunk_rows = pool.map(run_oa_task, tasks)
                else:
                    chunk_rows = [run_oa_task(t) for t in tasks]
                t_oa = time.time() - t0

                rows.extend(chunk_rows)
                if verbose:
                    ok = sum(r["status"] == "ok" for r in chunk_rows)
                    print(f"   OA sweep over (M, μ) = "
                          f"{len(tasks)} tasks done in {t_oa:.1f}s  "
                          f"({ok}/{len(tasks)} ok)")

    if verbose:
        print(f"\nTotal wallclock: {time.time() - t_total:.1f}s")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTANT: the KMO/OA comparison requires the SAME μ in both models if
# weights are adaptive. The structure of run_sweep above runs ONE KMO per Δ
# (with μ_km fixed) and reuses it across all OA-μ values. That is only
# correct if the KMO weights don't depend on μ, i.e. if μ_km = 0.
#
# To compare *adaptive* KMO weights against adaptive OA weights, we instead
# need to run KMO once per (trial, Δ, μ).  This is implemented in
# run_sweep_adaptive() below, which is what the main entry point uses.
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep_adaptive(
    *,
    dist,
    n_trials,
    Delta_values,
    M_values,
    mu_values,
    plasticity_values,
    N, T, K, gamma, omega0,
    n_eval,
    method, rtol, atol,
    seed_base,
    n_workers,
    verbose=True,
):
    """Same as run_sweep but runs KMO for each (trial, Δ, μ, plasticity).
    The (M)-sweep is parallelised within each block."""
    rows = []
    t_total = time.time()

    for trial in range(n_trials):
        seed = seed_base + trial
        rng  = np.random.default_rng(seed)
        theta0 = rng.uniform(-np.pi, np.pi, N)

        for Delta in Delta_values:
            omega_micro = sample_microscopic_frequencies(
                N, dist, omega0, Delta, rng
            )

            for plasticity in plasticity_values:
                for mu_val in mu_values:
                    t0 = time.time()
                    try:
                        km_res = run_km(
                            N=N, T=T, K=K, mu=mu_val, gamma=gamma,
                            omega_micro=omega_micro, theta0=theta0,
                            plasticity=plasticity, n_eval=n_eval,
                            method=method, rtol=rtol, atol=atol,
                        )
                    except Exception as e:
                        for M in M_values:
                            rows.append(dict(
                                trial=trial, dist=dist,
                                plasticity=plasticity,
                                Delta=Delta, M=M, mu=mu_val,
                                rmse_R=float("nan"), corr_A=float("nan"),
                                status="error_km", error=str(e),
                            ))
                        continue
                    t_km = time.time() - t0

                    tasks = [
                        (trial, dist, Delta, M, mu_val,
                         omega0, gamma, plasticity, T, K,
                         omega_micro, theta0,
                         km_res.A_km_final, km_res.t_grid, km_res.R_km,
                         N, n_eval, method, rtol, atol)
                        for M in M_values
                    ]

                    t0 = time.time()
                    if n_workers > 1:
                        with Pool(n_workers) as pool:
                            chunk_rows = pool.map(run_oa_task, tasks)
                    else:
                        chunk_rows = [run_oa_task(t) for t in tasks]
                    t_oa = time.time() - t0

                    rows.extend(chunk_rows)
                    if verbose:
                        ok = sum(r["status"] == "ok" for r in chunk_rows)
                        print(f"[trial {trial+1}/{n_trials} Δ={Delta:g} "
                              f"plast={plasticity} μ={mu_val:g}] "
                              f"KMO {t_km:.1f}s | OA (M-sweep, {len(tasks)} tasks) "
                              f"{t_oa:.1f}s  [{ok}/{len(tasks)} ok]")

    if verbose:
        print(f"\nTotal wallclock: {time.time() - t_total:.1f}s")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dist", choices=["uniform", "lorentzian"],
                   default="lorentzian",
                   help="Microscopic frequency distribution")
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--N", type=int, default=500,
                   help="Number of microscopic oscillators")
    p.add_argument("--T", type=float, default=150.0)
    p.add_argument("--K", type=float, default=2.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--plasticity", choices=["hebbian", "antihebbian"],
                   nargs="+",
                   default=["hebbian", "antihebbian"],
                   help="Plasticity rule(s) to sweep over")

    # Sweep grids
    p.add_argument("--Delta", type=float, nargs="+",
                   default=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
                   help="Δ values to sweep")
    p.add_argument("--M", type=int, nargs="+",
                   default=[5, 10, 25, 50, 100, 250, 500],
                   help="M values to sweep")
    p.add_argument("--mu", type=float, nargs="+",
                   default=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08],
                   help="μ values to sweep")

    # Solver
    p.add_argument("--method", default="RK45")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--n_eval", type=int, default=1500)

    # Execution control
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--n_workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="kmo_sweep_results.csv")

    return p.parse_args()


def main():
    args = parse_args()
    print("Sweep configuration")
    print("-" * 60)
    for k, v in vars(args).items():
        print(f"  {k:12s} = {v}")
    print("-" * 60)
    n_tasks = (args.n_trials * len(args.Delta) * len(args.M)
               * len(args.mu) * len(args.plasticity))
    print(f"Total (trial, Δ, M, μ, plasticity) tasks: {n_tasks}")
    print(f"Using {args.n_workers} parallel worker(s)\n")

    df = run_sweep_adaptive(
        dist=args.dist,
        n_trials=args.n_trials,
        Delta_values=args.Delta,
        M_values=args.M,
        mu_values=args.mu,
        plasticity_values=args.plasticity,
        N=args.N, T=args.T, K=args.K, gamma=args.gamma, omega0=args.omega0,
        n_eval=args.n_eval,
        method=args.method, rtol=args.rtol, atol=args.atol,
        seed_base=args.seed_base,
        n_workers=args.n_workers,
    )

    df.to_csv(args.out, index=False, mode="a")
    print(f"\nResults saved → {args.out}  ({len(df)} rows)")

    # Compact summary
    if "status" in df.columns:
        ok = df[df["status"] == "ok"]
        if len(ok) > 0:
            grouped = (ok.groupby(["plasticity", "Delta", "M", "mu"])
                       [["rmse_R", "corr_A"]].mean()
                       .reset_index())
            print("\nMean metrics across trials:")
            print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()