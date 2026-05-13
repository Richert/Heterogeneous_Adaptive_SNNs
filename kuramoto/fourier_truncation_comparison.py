"""
Systematic K Sweep: KMO with |sin| Plasticity vs. OA Mean-Field
================================================================
For each trial:
    1. Draw microscopic frequencies and initial phases.
    2. For each K (with M fixed):
        - run the microscopic KMO with plasticity  Ȧ_ij = μ|sin(θ_j-θ_i)| - γ A_ij,
        - run the OA mean-field with the SAME plasticity rule under the
          truncated Fourier expansion (up to order n_terms),
        - compute rmse_R, corr_A, and the time-averaged within-ensemble
          coherence (mean of r_m(t)) versus the KMO trajectory.

OA plasticity rule (truncated at n = n_terms)
---------------------------------------------
    Ȧ_ml = μ[2/π - (4/π) Σ_{n=1}^{n_terms} (r_m r_l)^{2n} cos(2n(ψ_l-ψ_m)) / (4n²-1)]
           - γ A_ml

Parallelisation: the K sweep is the inner loop, parallelised with
multiprocessing.Pool inside each trial.  Each task self-contains everything
needed for one (trial, K) combination, including its own KMO and OA
simulations, so there is no shared state between workers.

Output: CSV with columns
    [trial, dist, Delta, K, M, mu, n_terms,
     rmse_R, corr_A, mean_r_km, mean_r_oa,
     status, error]

  - mean_r_km: time- and ensemble-averaged within-ensemble coherence in the
               microscopic network (each ensemble's coherence is the
               magnitude of the empirical mean of e^{iθ_k} over its
               assigned oscillators, then averaged over ensembles and time).
  - mean_r_oa: time- and ensemble-averaged r_m(t) from the OA model.

Usage
-----
    python sweep_K_M.py [--dist {uniform,lorentzian}] [--M 10] [--n_terms 5] ...
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import cauchy, pearsonr

_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# Frequency distributions and ensemble parameters
# ═══════════════════════════════════════════════════════════════════════════════

def sample_microscopic_frequencies(N, dist, omega0, Delta, rng):
    if dist == "uniform":
        return rng.uniform(omega0 - Delta, omega0 + Delta, N)
    if dist == "lorentzian":
        return omega0 + Delta * rng.standard_cauchy(N)
    raise ValueError(f"Unknown dist={dist!r}")


def ensemble_parameters(M, dist, omega0, Delta):
    if dist == "uniform":
        edges = np.linspace(omega0 - Delta, omega0 + Delta, M + 1)
        omega_pop = 0.5 * (edges[:-1] + edges[1:])
        delta_pop = np.full(M, Delta / (2 * M))
        w_pop     = np.full(M, 1.0 / M)
        return omega_pop, delta_pop, w_pop
    if dist == "lorentzian":
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
    if dist == "uniform":
        diff = np.abs(omega_micro[:, None] - omega_pop[None, :])
        return np.argmin(diff, axis=1)
    if dist == "lorentzian":
        log_post = np.stack([
            np.log(w_pop[I] + 1e-300)
            + cauchy.logpdf(omega_micro, omega_pop[I], delta_pop[I])
            for I in range(len(w_pop))
        ], axis=0)
        return np.argmax(log_post, axis=0)
    raise ValueError(f"Unknown dist={dist!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# ODEs — microscopic and OA, both with anti-Hebbian |sin| plasticity
# ═══════════════════════════════════════════════════════════════════════════════

def km_abs_sin_ode(t, y, K, omega, mu, gamma):
    """
    Microscopic Kuramoto network with anti-Hebbian |sin| plasticity:
        dθ_k/dt = ω_k + (K/N) Σ_l A_kl sin(θ_l - θ_k)
        dA_kl/dt = μ |sin(θ_l - θ_k)| - γ A_kl    (diagonal kept at zero)
    """
    N     = len(omega)
    theta = y[:N]
    A     = y[N:].reshape(N, N)

    diff        = theta[np.newaxis, :] - theta[:, np.newaxis]
    sin_diff    = np.sin(diff)
    interaction = np.sum(A * sin_diff, axis=1)
    dtheta      = omega + (K / N) * interaction

    dA = mu * np.abs(sin_diff) - gamma * A
    np.fill_diagonal(dA, 0.0)
    return np.concatenate([dtheta, dA.ravel()])


def km_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


def km_coarse_grain_labels(A_fine, labels, M):
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


def oa_abs_sin_ode(t, y, K, omega, delta, mu, gamma, w, fourier_coeffs):
    """
    OA mean-field ODE for the anti-Hebbian |sin| plasticity rule, with the
    plasticity term truncated at n_terms harmonics of the Fourier expansion:

        Ȧ_ml = μ[2/π + Σ_{n=1}^{n_terms} c_n (r_m r_l)^{2n} cos(2n(ψ_l-ψ_m))]
               - γ A_ml,

    where c_n = -4/(π(4n²-1)) is precomputed in `fourier_coeffs`.
    Element fourier_coeffs[0] holds the DC term 2/π so the sum is written
    uniformly as Σ_{n=0}^{n_terms} c_n (r_m r_l)^{2n} cos(2n Δψ).
    """
    M   = len(omega)
    r   = np.clip(y[:M], _EPS, 1.0 - _EPS)
    psi = y[M:2*M]
    A   = y[2*M:].reshape(M, M)

    dpsi  = psi[np.newaxis, :] - psi[:, np.newaxis]    # (M, M)
    Ar    = A * r[np.newaxis, :]
    w_cos = Ar * np.cos(dpsi)
    w_sin = Ar * np.sin(dpsi)

    dr    = -delta * r + 0.5 * (1.0 - r**2) * K * (w_cos @ w)
    dpsi_ = omega     + 0.5 * (1.0 + r**2) / r * K * (w_sin @ w)

    # Plasticity: Σ_{n=0}^{n_terms} c_n (r_m r_l)^{2n} cos(2n Δψ)
    rr = r[:, np.newaxis] * r[np.newaxis, :]            # (M, M)
    plast = np.full_like(A, fourier_coeffs[0])          # n=0 (DC) term
    for n, c_n in enumerate(fourier_coeffs[1:], start=1):
        plast += c_n * (rr ** (2 * n)) * np.cos(2 * n * dpsi)
    dA = mu * plast - gamma * A

    return np.concatenate([dr, dpsi_, dA.ravel()])


def oa_order_parameter(r, w, psi):
    return np.abs(w @ (r * np.exp(1j * psi)))


def build_fourier_coeffs(n_terms):
    """
    Coefficients of the Fourier series for |sin φ| / 1:
        |sin φ| = 2/π - (4/π) Σ_{n=1}^{∞} cos(2n φ) / (4n² - 1)

    Returns an array of length n_terms+1 where
        coeffs[0]  = +2/π
        coeffs[n]  = -4/(π(4n²-1))   for n = 1, ..., n_terms
    """
    coeffs = np.zeros(n_terms + 1)
    coeffs[0] = 2.0 / np.pi
    for n in range(1, n_terms + 1):
        coeffs[n] = -4.0 / (np.pi * (4 * n * n - 1))
    return coeffs


# ═══════════════════════════════════════════════════════════════════════════════
# Single (trial, K, M) task
# ═══════════════════════════════════════════════════════════════════════════════

def run_task(task):
    """
    Worker function executed in a multiprocessing pool.

    Receives a flat tuple describing one (trial, K) cell and:
        - runs the microscopic KMO with |sin| plasticity,
        - runs the OA mean-field with Fourier-truncated plasticity,
        - computes rmse_R, corr_A, and mean within-ensemble coherence.

    Returns a single-row dict.
    """
    (trial, dist, Delta, K, M, mu_val, gamma, omega0,
     n_terms, T, N, n_eval, method, rtol, atol,
     omega_micro, theta0) = task

    try:
        # ── KMO simulation ──────────────────────────────────────────────────
        A_km0 = np.ones((N, N))
        y0_km = np.concatenate([theta0, A_km0.ravel()])

        sol_km = solve_ivp(
            km_abs_sin_ode, (0, T), y0_km, method=method,
            args=(K, omega_micro, mu_val, gamma),
            rtol=rtol, atol=atol, dense_output=True,
        )
        if not sol_km.success:
            raise RuntimeError(f"KMO solve_ivp failed: {sol_km.message}")

        t_grid     = np.linspace(0.0, T, n_eval)
        y_km_grid  = sol_km.sol(t_grid)
        theta_grid = y_km_grid[:N]
        R_km       = km_order_parameter(theta_grid)
        A_km_final = y_km_grid[N:, -1].reshape(N, N)

        # ── Build OA initial conditions matched to the same θ_0 ───────────
        omega_pop, delta_pop, w_pop = ensemble_parameters(M, dist, omega0, Delta)
        labels = assign_oscillators_to_ensembles(
            omega_micro, omega_pop, delta_pop, w_pop, dist,
        )

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

        # ── OA simulation ───────────────────────────────────────────────────
        fourier_coeffs = build_fourier_coeffs(n_terms)
        sol_oa = solve_ivp(
            oa_abs_sin_ode, (0, T), y0_oa, method=method,
            args=(K, omega_pop, delta_pop, mu_val, gamma, w_pop, fourier_coeffs),
            rtol=rtol, atol=atol, dense_output=True,
        )
        if not sol_oa.success:
            raise RuntimeError(f"OA solve_ivp failed: {sol_oa.message}")

        y_oa_grid  = sol_oa.sol(t_grid)
        r_grid     = y_oa_grid[:M]
        psi_grid   = y_oa_grid[M:2*M]
        A_oa_final = y_oa_grid[2*M:, -1].reshape(M, M)
        R_oa       = oa_order_parameter(r_grid, w_pop, psi_grid)

        # ── Metrics ─────────────────────────────────────────────────────────
        rmse_R = float(np.sqrt(np.mean((R_km - R_oa) ** 2)))

        A_km_cg = km_coarse_grain_labels(A_km_final, labels, M)
        a_km = A_km_cg.ravel()
        a_oa = A_oa_final.ravel()
        if np.std(a_km) < 1e-12 or np.std(a_oa) < 1e-12:
            corr_A = float("nan")
        else:
            corr_A, _ = pearsonr(a_km, a_oa)
            corr_A    = float(corr_A)

        # Mean within-ensemble coherence, averaged over ensembles and time.
        # KMO: for each ensemble m, compute |mean_{k∈m} e^{iθ_k(t)}| at every
        # time step, then average across both ensembles and time.
        z_km = np.exp(1j * theta_grid)                      # (N, n_eval)
        r_km_per_ens = np.zeros((M, n_eval))
        for I in range(M):
            mask = labels == I
            if mask.sum() == 0:
                continue
            r_km_per_ens[I] = np.abs(np.mean(z_km[mask], axis=0))
        mean_r_km = float(np.mean(r_km_per_ens))

        # OA: r_grid is already (M, n_eval), so the mean is direct.
        mean_r_oa = float(np.mean(r_grid))

        return dict(
            trial=trial, dist=dist, Delta=Delta, K=K, M=M, mu=mu_val,
            n_terms=n_terms,
            rmse_R=rmse_R, corr_A=corr_A,
            mean_r_km=mean_r_km, mean_r_oa=mean_r_oa,
            status="ok", error="",
        )

    except Exception as e:
        return dict(
            trial=trial, dist=dist, Delta=Delta, K=K, M=M, mu=mu_val,
            n_terms=n_terms,
            rmse_R=float("nan"), corr_A=float("nan"),
            mean_r_km=float("nan"), mean_r_oa=float("nan"),
            status="error", error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep driver
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    *,
    dist, n_trials,
    K_values,
    M,
    Delta, mu, gamma, omega0, n_terms,
    N, T, n_eval, method, rtol, atol,
    seed_base, n_workers, verbose=True,
):
    rows = []
    t_total = time.time()

    for trial in range(n_trials):
        seed = seed_base + trial
        rng  = np.random.default_rng(seed)

        # Microscopic frequencies and initial phases drawn once per trial
        omega_micro = sample_microscopic_frequencies(N, dist, omega0, Delta, rng)
        theta0      = rng.uniform(-np.pi, np.pi, N)

        # ── Build one task per K value (M fixed) ──────────────────────────
        tasks = [
            (trial, dist, Delta, K, M, mu, gamma, omega0,
             n_terms, T, N, n_eval, method, rtol, atol,
             omega_micro, theta0)
            for K in K_values
        ]

        t0 = time.time()
        if n_workers > 1:
            with Pool(n_workers) as pool:
                chunk_rows = pool.map(run_task, tasks)
        else:
            chunk_rows = [run_task(t) for t in tasks]
        dt = time.time() - t0

        rows.extend(chunk_rows)
        if verbose:
            ok = sum(r["status"] == "ok" for r in chunk_rows)
            print(f"[trial {trial+1}/{n_trials}]  "
                  f"{len(tasks)} K tasks done in {dt:.1f}s  "
                  f"[{ok}/{len(tasks)} ok]")

    if verbose:
        print(f"\nTotal wallclock: {time.time() - t_total:.1f}s")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dist", choices=["uniform", "lorentzian"],
                   default="lorentzian")
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--N", type=int, default=500)
    p.add_argument("--T", type=float, default=150.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--Delta", type=float, default=1.0,
                   help="Frequency-distribution width (FIXED in this sweep)")
    p.add_argument("--mu", type=float, default=0.02,
                   help="Plasticity learning rate (FIXED in this sweep)")
    p.add_argument("--gamma", type=float, default=0.001)
    p.add_argument("--n_terms", type=int, default=1,
                   help="Fourier truncation order for the OA |sin| plasticity")

    # Sweep grid (K only) and fixed M
    p.add_argument("--K", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0, 4.0, 5.0],
                   help="K values to sweep")
    p.add_argument("--M", type=int, default=20,
                   help="Number of ensembles (FIXED in this sweep)")

    # Solver
    p.add_argument("--method", default="RK45")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--n_eval", type=int, default=400)

    # Execution control
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--n_workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="sweep_K_results.csv")

    return p.parse_args()


def main():
    args = parse_args()
    print("Sweep configuration")
    print("-" * 60)
    for k, v in vars(args).items():
        print(f"  {k:12s} = {v}")
    print("-" * 60)
    n_tasks = args.n_trials * len(args.K)
    print(f"Total (trial, K) tasks: {n_tasks}   (M fixed at {args.M})")
    print(f"Using {args.n_workers} parallel worker(s)")
    print(f"Fourier truncation: |sin| OA plasticity uses n=1..{args.n_terms}\n")

    df = run_sweep(
        dist=args.dist,
        n_trials=args.n_trials,
        K_values=args.K,
        M=args.M,
        Delta=args.Delta,
        mu=args.mu,
        gamma=args.gamma,
        omega0=args.omega0,
        n_terms=args.n_terms,
        N=args.N, T=args.T,
        n_eval=args.n_eval,
        method=args.method, rtol=args.rtol, atol=args.atol,
        seed_base=args.seed_base,
        n_workers=args.n_workers,
    )

    df.to_csv(args.out, index=False)
    print(f"\nResults saved → {args.out}  ({len(df)} rows)")

    if "status" in df.columns:
        ok = df[df["status"] == "ok"]
        if len(ok) > 0:
            grouped = (ok.groupby(["K"])
                       [["rmse_R", "corr_A", "mean_r_km", "mean_r_oa"]]
                       .mean().reset_index())
            print("\nMean metrics across trials (per K):")
            print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()