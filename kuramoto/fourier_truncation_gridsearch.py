"""
Systematic K × n_terms Sweep: KMO with |sin| Plasticity vs. OA Mean-Field
==========================================================================
For each trial × K × n_terms cell:
    1. Draw microscopic frequencies and initial phases (per trial).
    2. Run the microscopic KMO with plasticity  Ȧ_ij = μ|sin(θ_j-θ_i)| - γA_ij.
    3. Run the OA mean-field with the same plasticity rule under a Fourier
       truncation at order `n_terms`,
           Ȧ_ml = μ[2/π - (4/π) Σ_{n=1}^{n_terms}
                    (r_m r_l)^{2n} cos(2n(ψ_l-ψ_m)) / (4n² - 1)] - γA_ml.

Metrics recorded per cell
-------------------------
    corr_A       Pearson correlation between the final KMO coarse-grained
                 connectivity (averaged over the M ensembles induced by the
                 OA frequency partitioning) and the OA connectivity.

    nrmse_R      Normalised RMSE between the KMO and OA macroscopic phase
                 coherence traces R(t), with the RMSE divided by the time
                 mean of (R_km + R_oa)/2 so the metric is independent of the
                 absolute coherence level.

    nrmse_r      Same construction for the within-ensemble coherence r_m(t):
                 RMSE pooled over (ensemble, time) between the KMO empirical
                 r_m and the OA r_m, divided by the pooled mean of
                 (r_km + r_oa)/2.

    rmse_sbar    RMSE between the KMO empirical ensemble-averaged plasticity
                 drive  s̄_ml(t) = (1/|N_m||N_l|) Σ_{i∈N_m, j∈N_l}
                                          |sin(θ_j(t)-θ_i(t))|
                 and its OA Fourier-truncated counterpart
                 s̄_ml^trunc(t) = 2/π - (4/π) Σ_{n=1}^{n_terms}
                                       (r_m r_l)^{2n} cos(2n(ψ_l-ψ_m))/(4n²-1),
                 pooled across t > t_cutoff and all M² ensemble pairs.

Parallelisation: tasks are independent and run in a multiprocessing.Pool.

Memory: each worker uses `t_eval` (no dense_output) and streams the N×N
KMO connectivity slab one time slice at a time, coarse-graining and
computing s̄_ml on the fly.  Peak RSS scales like O(N² + N²·#workers)
instead of O(N²·n_eval).

Usage
-----
    python sweep_K_nterms.py --K 1 2 3 4 5 --n_terms 1 2 3 4
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


def oa_abs_sin_ode(t, y, K, omega, delta, mu, gamma, w, fourier_coeffs):
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

    rr = r[:, np.newaxis] * r[np.newaxis, :]
    plast = np.full_like(A, fourier_coeffs[0])
    for n, c_n in enumerate(fourier_coeffs[1:], start=1):
        plast += c_n * (rr ** (2 * n)) * np.cos(2 * n * dpsi)
    dA = mu * plast - gamma * A
    return np.concatenate([dr, dpsi_, dA.ravel()])


def oa_order_parameter(r, w, psi):
    return np.abs(w @ (r * np.exp(1j * psi)))


def build_fourier_coeffs(n_terms):
    coeffs = np.zeros(n_terms + 1)
    coeffs[0] = 2.0 / np.pi
    for n in range(1, n_terms + 1):
        coeffs[n] = -4.0 / (np.pi * (4 * n * n - 1))
    return coeffs


# ═══════════════════════════════════════════════════════════════════════════════
# Coarse-graining helper (no per-call np.where)
# ═══════════════════════════════════════════════════════════════════════════════

def _coarse_grain_fast(A_fine, ens_idx, ens_sizes):
    """
    Coarse-grain an N×N matrix to M×M by averaging over the ensemble blocks
    given by `ens_idx` (precomputed index arrays).  Empty ensembles → 0.
    """
    M = len(ens_idx)
    rowsum = np.zeros((M, A_fine.shape[1]))
    for I, idx in enumerate(ens_idx):
        if idx.size:
            rowsum[I] = A_fine[idx].sum(axis=0)
    A_cg = np.zeros((M, M))
    for J, idx in enumerate(ens_idx):
        if idx.size:
            A_cg[:, J] = rowsum[:, idx].sum(axis=1)
    denom = np.outer(ens_sizes, ens_sizes).astype(float)
    denom[denom == 0] = 1.0
    return A_cg / denom


# ═══════════════════════════════════════════════════════════════════════════════
# Single (trial, K, n_terms) task
# ═══════════════════════════════════════════════════════════════════════════════

def run_task(task):
    """
    Worker function: one (trial, K, n_terms) cell.
    Streams the KMO connectivity slab to avoid the O(N²·n_eval) memory
    footprint and to compute the empirical s̄_ml(t) on the fly.
    """
    (trial, dist, Delta, K, M, mu_val, gamma, omega0,
     n_terms, T, N, n_eval, method, rtol, atol, t_cutoff,
     omega_micro, theta0) = task

    try:
        t_grid = np.linspace(0.0, T, n_eval)

        # ── Ensemble partition & index arrays (needed for streaming CG) ────
        omega_pop, delta_pop, w_pop = ensemble_parameters(M, dist, omega0, Delta)
        labels = assign_oscillators_to_ensembles(
            omega_micro, omega_pop, delta_pop, w_pop, dist,
        )
        ens_idx   = [np.where(labels == I)[0] for I in range(M)]
        ens_sizes = np.array([idx.size for idx in ens_idx])

        # ── KMO simulation (t_eval, no dense_output) ────────────────────────
        A_km0 = np.ones((N, N))
        y0_km = np.concatenate([theta0, A_km0.ravel()])

        sol_km = solve_ivp(
            km_abs_sin_ode, (0, T), y0_km, method=method,
            args=(K, omega_micro, mu_val, gamma),
            t_eval=t_grid, rtol=rtol, atol=atol,
        )
        if not sol_km.success:
            raise RuntimeError(f"KMO solve_ivp failed: {sol_km.message}")

        theta_grid = sol_km.y[:N]                           # (N, n_eval)
        R_km       = km_order_parameter(theta_grid)

        # Per-ensemble KMO within-ensemble coherence (M, n_eval) — small.
        z_km = np.exp(1j * theta_grid)
        r_km_per_ens = np.zeros((M, n_eval))
        for I, idx in enumerate(ens_idx):
            if idx.size:
                r_km_per_ens[I] = np.abs(np.mean(z_km[idx], axis=0))
        del z_km

        # Streaming KMO connectivity + empirical s̄_ml(t)
        late_mask = t_grid > t_cutoff
        late_idx  = np.where(late_mask)[0]
        s_emp_late = np.empty((late_idx.size, M, M))

        A_km_flat  = sol_km.y[N:]                            # (N², n_eval)
        A_km_final_cg = None
        pos_in_late  = 0
        for k in range(n_eval):
            if late_mask[k]:
                theta_k = theta_grid[:, k]
                abs_sin_NN = np.abs(np.sin(theta_k[None, :] - theta_k[:, None]))
                s_emp_late[pos_in_late] = _coarse_grain_fast(
                    abs_sin_NN, ens_idx, ens_sizes,
                )
                pos_in_late += 1
            if k == n_eval - 1:
                A_slice = A_km_flat[:, k].reshape(N, N)
                A_km_final_cg = _coarse_grain_fast(A_slice, ens_idx, ens_sizes)
        del A_km_flat, sol_km

        # ── OA initial conditions matched to the same θ_0 ──────────────────
        r0   = np.empty(M)
        psi0 = np.empty(M)
        for I, idx in enumerate(ens_idx):
            if idx.size == 0:
                r0[I], psi0[I] = _EPS, 0.0
                continue
            z = np.mean(np.exp(1j * theta0[idx]))
            r0[I]   = np.clip(np.abs(z), _EPS, 1.0 - _EPS)
            psi0[I] = np.angle(z)

        A_oa0 = np.ones((M, M))
        y0_oa = np.concatenate([r0, psi0, A_oa0.ravel()])

        # ── OA simulation ───────────────────────────────────────────────────
        fourier_coeffs = build_fourier_coeffs(n_terms)
        sol_oa = solve_ivp(
            oa_abs_sin_ode, (0, T), y0_oa, method=method,
            args=(K, omega_pop, delta_pop, mu_val, gamma, w_pop, fourier_coeffs),
            t_eval=t_grid, rtol=rtol, atol=atol,
        )
        if not sol_oa.success:
            raise RuntimeError(f"OA solve_ivp failed: {sol_oa.message}")

        r_grid     = sol_oa.y[:M]
        psi_grid   = sol_oa.y[M:2*M]
        A_oa_final = sol_oa.y[2*M:, -1].reshape(M, M)
        R_oa       = oa_order_parameter(r_grid, w_pop, psi_grid)

        # OA-side Fourier-truncated s̄^trunc_ml at the same late times
        r_late   = r_grid[:, late_idx]
        psi_late = psi_grid[:, late_idx]
        s_oa_late = np.empty((late_idx.size, M, M))
        for kk in range(late_idx.size):
            r_k   = r_late[:, kk]
            psi_k = psi_late[:, kk]
            rr    = r_k[:, None] * r_k[None, :]
            dpsi  = psi_k[None, :] - psi_k[:, None]
            out   = np.full((M, M), fourier_coeffs[0])
            for n, c_n in enumerate(fourier_coeffs[1:], start=1):
                out += c_n * (rr ** (2 * n)) * np.cos(2 * n * dpsi)
            s_oa_late[kk] = out
        del sol_oa

        # ── Metrics ─────────────────────────────────────────────────────────
        # (1) Coupling-matrix correlation (KMO coarse-grained vs OA), final t
        a_km = A_km_final_cg.ravel()
        a_oa = A_oa_final.ravel()
        if np.std(a_km) < 1e-12 or np.std(a_oa) < 1e-12:
            corr_A = float("nan")
        else:
            corr_A, _ = pearsonr(a_km, a_oa)
            corr_A    = float(corr_A)

        # (2) Normalised RMSE of macroscopic R(t) between KMO and OA,
        # divided by the time-mean of (R_km + R_oa)/2 so the metric is
        # independent of the absolute coherence level.
        denom_R = 0.5 * float(np.mean(R_km + R_oa))
        if denom_R < 1e-12:
            nrmse_R = float("nan")
        else:
            nrmse_R = float(np.sqrt(np.mean((R_km - R_oa) ** 2)) / denom_R)

        # (3) Normalised RMSE of within-ensemble coherence r_m(t), pooled
        # over (ensemble, time), normalised by the pooled mean of
        # (r_km + r_oa)/2.
        denom_r = 0.5 * float(np.mean(r_km_per_ens + r_grid))
        if denom_r < 1e-12:
            nrmse_r = float("nan")
        else:
            nrmse_r = float(
                np.sqrt(np.mean((r_km_per_ens - r_grid) ** 2)) / denom_r
            )

        # (4) RMSE of empirical s̄_ml vs OA truncated s̄^trunc_ml on t>t_cutoff
        if late_idx.size > 0:
            rmse_sbar = float(np.sqrt(np.mean((s_emp_late - s_oa_late) ** 2)))
        else:
            rmse_sbar = float("nan")

        return dict(
            trial=trial, dist=dist, Delta=Delta, K=K, M=M, mu=mu_val,
            n_terms=n_terms, t_cutoff=t_cutoff,
            corr_A=corr_A,
            nrmse_R=nrmse_R, nrmse_r=nrmse_r,
            rmse_sbar=rmse_sbar,
            n_late=int(late_idx.size),
            status="ok", error="",
        )

    except Exception as e:
        return dict(
            trial=trial, dist=dist, Delta=Delta, K=K, M=M, mu=mu_val,
            n_terms=n_terms, t_cutoff=t_cutoff,
            corr_A=float("nan"),
            nrmse_R=float("nan"), nrmse_r=float("nan"),
            rmse_sbar=float("nan"),
            n_late=0,
            status="error", error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep driver
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    *,
    dist, n_trials,
    K_values, n_terms_values,
    M,
    Delta, mu, gamma, omega0,
    N, T, n_eval, method, rtol, atol, t_cutoff,
    seed_base, n_workers, verbose=True,
):
    rows = []
    t_total = time.time()

    for trial in range(n_trials):
        seed = seed_base + trial
        rng  = np.random.default_rng(seed)

        # Microscopic frequencies and initial phases drawn once per trial,
        # shared across all (K, n_terms) cells so that comparisons are
        # paired (same realisation, different sweep coordinates).
        omega_micro = sample_microscopic_frequencies(N, dist, omega0, Delta, rng)
        theta0      = rng.uniform(-np.pi, np.pi, N)

        tasks = [
            (trial, dist, Delta, K, M, mu, gamma, omega0,
             n_terms, T, N, n_eval, method, rtol, atol, t_cutoff,
             omega_micro, theta0)
            for K in K_values
            for n_terms in n_terms_values
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
                  f"{len(tasks)} (K, n_terms) tasks done in {dt:.1f}s  "
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
                   default="uniform")
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--N", type=int, default=500)
    p.add_argument("--T", type=float, default=120.0)
    p.add_argument("--omega0", type=float, default=0.0)
    p.add_argument("--Delta", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.001)

    # Two-dimensional sweep grid
    p.add_argument("--K", type=float, nargs="+",
                   default=[0.5, 1.0, 1.5],
                   help="K values to sweep")
    p.add_argument("--n_terms", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16],
                   help="Fourier truncation orders to sweep")
    p.add_argument("--M", type=int, default=50,
                   help="Number of ensembles (FIXED in this sweep)")

    p.add_argument("--method", default="RK45")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--n_eval", type=int, default=400)
    p.add_argument("--t_cutoff", type=float, default=20.0,
                   help="Drop t ≤ t_cutoff when building rmse_sbar")

    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--n_workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="sweep_K_nterms_results.csv")

    return p.parse_args()


def main():
    args = parse_args()
    print("Sweep configuration")
    print("-" * 60)
    for k, v in vars(args).items():
        print(f"  {k:12s} = {v}")
    print("-" * 60)
    n_tasks = args.n_trials * len(args.K) * len(args.n_terms)
    print(f"Total (trial, K, n_terms) tasks: {n_tasks}   "
          f"(M fixed at {args.M})")
    print(f"Using {args.n_workers} parallel worker(s)\n")

    df = run_sweep(
        dist=args.dist,
        n_trials=args.n_trials,
        K_values=args.K,
        n_terms_values=args.n_terms,
        M=args.M,
        Delta=args.Delta,
        mu=args.mu,
        gamma=args.gamma,
        omega0=args.omega0,
        N=args.N, T=args.T,
        n_eval=args.n_eval,
        method=args.method, rtol=args.rtol, atol=args.atol,
        t_cutoff=args.t_cutoff,
        seed_base=args.seed_base,
        n_workers=args.n_workers,
    )

    df.to_csv(args.out, index=False)
    print(f"\nResults saved → {args.out}  ({len(df)} rows)")

    if "status" in df.columns:
        ok = df[df["status"] == "ok"]
        if len(ok) > 0:
            grouped = (ok.groupby(["K", "n_terms"])
                       [["corr_A",
                         "nrmse_R", "nrmse_r",
                         "rmse_sbar"]]
                       .mean()
                       .reset_index())
            print("\nMean metrics across trials (per K, n_terms):")
            with pd.option_context("display.width", 200,
                                   "display.max_columns", 50):
                print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()