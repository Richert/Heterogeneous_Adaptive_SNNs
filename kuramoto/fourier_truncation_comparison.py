"""
Single-Run KMO vs. OA Mean-Field with |sin| Plasticity
=======================================================
Runs *one* parameter set of the adaptive-Kuramoto problem,

    dθ_k/dt  = ω_k + (K/N) Σ_l A_kl sin(θ_l - θ_k)
    dA_kl/dt = μ |sin(θ_l - θ_k)| - γ A_kl,

together with the Ott-Antonsen mean-field reduction in which the
plasticity rule is Fourier-expanded and truncated at `n_terms`,

    Ȧ_ml = μ [2/π - (4/π) Σ_{n=1}^{n_terms} (r_m r_l)^{2n}
                                          cos(2n(ψ_l - ψ_m)) / (4n² - 1)]
           - γ A_ml.

Then produces a 2×2 figure:

    (a) Connectivity matrices — KMO coarse-grained to M ensembles vs OA.
    (b) Global Kuramoto coherence R(t) for KMO and OA.
    (c) Coupling-element trajectories — coarse-grained KMO vs OA samples.
    (d) Ensemble-pair plasticity drive comparison.  For every time step
        t > t_cutoff and every ensemble pair (m, l), the KMO empirical
        average  s̄_ml = (1/|N_m||N_l|) Σ_{i∈N_m, j∈N_l} |sin(θ_j-θ_i)|
        is scattered against the OA-side n_terms-truncated Fourier
        prediction at the same time and pair.  Points on y = x indicate
        perfect mean-field reproduction.

Usage
-----
    python single_run_visualisation.py
    python single_run_visualisation.py --K 3.0 --n_terms 4 --dist uniform
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Reuse the model functions from the sweep script
from fourier_truncation_gridsearch import (
    _EPS,
    sample_microscopic_frequencies,
    ensemble_parameters,
    assign_oscillators_to_ensembles,
    km_abs_sin_ode,
    km_order_parameter,
    oa_abs_sin_ode,
    oa_order_parameter,
    build_fourier_coeffs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-run driver
# ═══════════════════════════════════════════════════════════════════════════════

def _coarse_grain_fast(A_fine, ens_idx, ens_sizes):
    """
    Coarse-grain an N×N matrix to M×M by averaging blocks defined by the
    membership lists in `ens_idx`.  Equivalent to km_coarse_grain_labels()
    but avoids recomputing np.where on every call.  Empty ensembles get 0.
    """
    M = len(ens_idx)
    A_cg = np.zeros((M, M))
    # Row-sum reduction first: rowsum[I, j] = sum_{i in ensemble I} A[i, j]
    rowsum = np.zeros((M, A_fine.shape[1]))
    for I, idx in enumerate(ens_idx):
        if idx.size:
            rowsum[I] = A_fine[idx].sum(axis=0)
    # Then column reduction
    for J, idx in enumerate(ens_idx):
        if idx.size:
            A_cg[:, J] = rowsum[:, idx].sum(axis=1)
    # Divide by block sizes
    denom = np.outer(ens_sizes, ens_sizes).astype(float)
    denom[denom == 0] = 1.0   # guard; corresponding entries stay 0
    return A_cg / denom


def run_single(
    *,
    dist, Delta, K, M, mu, gamma, omega0, n_terms,
    N, T, n_eval, method, rtol, atol, seed,
    t_cutoff=100.0,
):
    """Run one KMO + OA simulation. Returns a dict with all trajectories."""
    rng = np.random.default_rng(seed)

    omega_micro = sample_microscopic_frequencies(N, dist, omega0, Delta, rng)
    theta0      = rng.uniform(-np.pi, np.pi, N)
    t_grid      = np.linspace(0.0, T, n_eval)

    # ── Pre-compute ensemble labels and index arrays ───────────────────────
    # We need these *before* the KMO solve so we can coarse-grain the
    # connectivity on the fly and avoid storing an N×N×n_eval cube.
    omega_pop, delta_pop, w_pop = ensemble_parameters(M, dist, omega0, Delta)
    labels = assign_oscillators_to_ensembles(
        omega_micro, omega_pop, delta_pop, w_pop, dist,
    )
    ens_idx   = [np.where(labels == I)[0] for I in range(M)]
    ens_sizes = np.array([idx.size for idx in ens_idx])

    # ── KMO ────────────────────────────────────────────────────────────────
    # Memory note: the state vector has N + N² entries.  Storing it at every
    # t in t_grid is unavoidable while solving, but we
    #   (1) pass t_eval=t_grid instead of dense_output=True, so solve_ivp
    #       does not retain every internal RK45 step + interpolant, and
    #   (2) immediately reduce the N×N connectivity slices to M×M, dropping
    #       the full slab before the next step is requested.
    A_km0 = np.ones((N, N))
    y0_km = np.concatenate([theta0, A_km0.ravel()])

    sol_km = solve_ivp(
        km_abs_sin_ode, (0, T), y0_km, method=method,
        args=(K, omega_micro, mu, gamma),
        t_eval=t_grid,
        rtol=rtol, atol=atol,
    )
    if not sol_km.success:
        raise RuntimeError(f"KMO solve_ivp failed: {sol_km.message}")

    # sol_km.y has shape (N + N², n_eval).  Peel off θ first, then process
    # the connectivity slab one column (= one time slice) at a time so we
    # never hold more than two N×N matrices at once.
    theta_grid = sol_km.y[:N]                        # (N, n_eval)  — small
    R_km       = km_order_parameter(theta_grid)

    # Indices of late-time slices (t > t_cutoff) for the empirical
    # ensemble-averaged plasticity drive
    #     s̄_ml(t) = (1 / |N_m||N_l|) Σ_{i∈N_m} Σ_{j∈N_l} |sin(θ_j(t) - θ_i(t))|.
    # We compute this inside the streaming loop while θ for time-step k is
    # in scope, then store only the M×M scalars.
    late_mask = t_grid > t_cutoff
    late_idx  = np.where(late_mask)[0]
    s_emp_late = np.empty((late_idx.size, M, M))

    A_km_flat  = sol_km.y[N:]                        # (N², n_eval) view
    A_km_cg_grid = np.empty((M, M, n_eval))
    A_km_final   = None
    pos_in_late  = 0
    for k in range(n_eval):
        A_slice = A_km_flat[:, k].reshape(N, N)      # view, no copy
        A_km_cg_grid[..., k] = _coarse_grain_fast(A_slice, ens_idx, ens_sizes)
        if k == n_eval - 1:
            A_km_final = A_slice.copy()              # keep one full snapshot
        if late_mask[k]:
            # |sin(θ_j - θ_i)| for all pairs at this time, then M×M average.
            theta_k = theta_grid[:, k]
            abs_sin_NN = np.abs(np.sin(theta_k[None, :] - theta_k[:, None]))
            s_emp_late[pos_in_late] = _coarse_grain_fast(
                abs_sin_NN, ens_idx, ens_sizes,
            )
            pos_in_late += 1

    # Free the big arrays before the OA solve.
    del A_km_flat, sol_km

    # ── OA initial conditions matched to the same θ_0 ──────────────────────
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

    # ── OA ─────────────────────────────────────────────────────────────────
    # OA state is only 2M + M² floats; t_eval is enough, no dense_output.
    fourier_coeffs = build_fourier_coeffs(n_terms)
    sol_oa = solve_ivp(
        oa_abs_sin_ode, (0, T), y0_oa, method=method,
        args=(K, omega_pop, delta_pop, mu, gamma, w_pop, fourier_coeffs),
        t_eval=t_grid,
        rtol=rtol, atol=atol,
    )
    if not sol_oa.success:
        raise RuntimeError(f"OA solve_ivp failed: {sol_oa.message}")

    r_grid     = sol_oa.y[:M]
    psi_grid   = sol_oa.y[M:2*M]
    A_oa_grid  = sol_oa.y[2*M:].reshape(M, M, n_eval).copy()
    A_oa_final = A_oa_grid[..., -1]
    R_oa       = oa_order_parameter(r_grid, w_pop, psi_grid)

    # ── OA-side Fourier-truncated plasticity drive at the same late times ──
    # Mirroring the OA RHS (oa_abs_sin_ode), the predicted ensemble-averaged
    # drive is
    #     s̄_ml^trunc(t) = 2/π
    #                   - (4/π) Σ_{n=1}^{n_terms}
    #                       (r_m(t) r_l(t))^{2n} cos(2n(ψ_l(t)-ψ_m(t)))
    #                                                       / (4n² - 1).
    R_km_mean = float(R_km.mean())
    R_oa_mean = float(R_oa.mean())

    fourier_coeffs_eval = build_fourier_coeffs(n_terms)
    r_late   = r_grid[:, late_idx]                           # (M, T_late)
    psi_late = psi_grid[:, late_idx]                         # (M, T_late)
    s_oa_late = np.empty((late_idx.size, M, M))
    for kk in range(late_idx.size):
        r_k   = r_late[:, kk]
        psi_k = psi_late[:, kk]
        rr    = r_k[:, None] * r_k[None, :]                  # (M, M)
        dpsi  = psi_k[None, :] - psi_k[:, None]              # ψ_l - ψ_m
        out   = np.full((M, M), fourier_coeffs_eval[0])      # DC term
        for n, c_n in enumerate(fourier_coeffs_eval[1:], start=1):
            out += c_n * (rr ** (2 * n)) * np.cos(2 * n * dpsi)
        s_oa_late[kk] = out

    del sol_oa

    return dict(
        t_grid=t_grid, labels=labels,
        omega_micro=omega_micro, omega_pop=omega_pop, w_pop=w_pop,
        R_km=R_km, R_oa=R_oa,
        R_km_mean=R_km_mean, R_oa_mean=R_oa_mean,
        A_km_final=A_km_final, A_km_cg_grid=A_km_cg_grid,
        A_oa_grid=A_oa_grid, A_oa_final=A_oa_final,
        r_grid=r_grid, psi_grid=psi_grid,
        n_terms=n_terms,
        t_cutoff=t_cutoff, late_idx=late_idx,
        s_emp_late=s_emp_late, s_oa_late=s_oa_late,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(res, params, savepath):
    t      = res["t_grid"]
    R_km   = res["R_km"]
    R_oa   = res["R_oa"]
    Akm    = res["A_km_cg_grid"][..., -1]    # final coarse-grained KMO
    Aoa    = res["A_oa_final"]

    fig = plt.figure(figsize=(12.0, 9.5), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2)

    # ── (a) Connectivity matrices ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    vmax = max(Akm.max(), Aoa.max())
    vmin = min(Akm.min(), Aoa.min())
    # Show side-by-side as a single image with a separator column
    M = Akm.shape[0]
    sep = np.full((M, 1), np.nan)
    combined = np.hstack([Akm, sep, Aoa])
    im = ax_a.imshow(combined, cmap="viridis", vmin=vmin, vmax=vmax,
                     origin="lower", aspect="equal")
    ax_a.set_xticks([M / 2, M + 1 + M / 2])
    ax_a.set_xticklabels(["KMO (coarse-grained)", "OA"])
    ax_a.set_yticks([])
    ax_a.set_title("(a) Final connectivity  $A_{ml}$")
    cbar = fig.colorbar(im, ax=ax_a, shrink=0.85)
    cbar.set_label(r"$A_{ml}$")

    # ── (b) Macroscopic coherence R(t) ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(t, R_km, lw=1.8, label="KMO", color="C0")
    ax_b.plot(t, R_oa, lw=1.8, label="OA mean-field", color="C3",
              linestyle="--")
    ax_b.set_xlabel("time $t$")
    ax_b.set_ylabel(r"$R(t)$")
    ax_b.set_ylim(0.0, 1.02)
    ax_b.set_title("(b) Macroscopic phase coherence")
    ax_b.legend(loc="best", frameon=False)
    ax_b.grid(True, alpha=0.3)

    # ── (c) Coupling-element trajectories (sample of off-diagonal entries) ─
    ax_c = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    M_ = res["A_oa_grid"].shape[0]
    idx_pairs = []
    while len(idx_pairs) < 6:
        i = rng.integers(0, M_)
        j = rng.integers(0, M_)
        if i != j and (i, j) not in idx_pairs:
            idx_pairs.append((i, j))
    for k, (i, j) in enumerate(idx_pairs):
        ax_c.plot(t, res["A_km_cg_grid"][i, j], color=f"C{k}",
                  lw=1.4, alpha=0.85,
                  label=None if k else "KMO (cg)")
        ax_c.plot(t, res["A_oa_grid"][i, j], color=f"C{k}",
                  lw=1.4, linestyle="--", alpha=0.85,
                  label=None if k else "OA")
    ax_c.set_xlabel("time $t$")
    ax_c.set_ylabel(r"$A_{ml}(t)$")
    ax_c.set_title("(c) Sample coupling trajectories")
    ax_c.legend(loc="best", frameon=False)
    ax_c.grid(True, alpha=0.3)

    # ── (d) Ensemble-pair plasticity drive: KMO empirical vs OA truncation ─
    # For each time t > t_cutoff and each (m, l), scatter the empirical KMO
    # ensemble average  s̄_ml = (1/|N_m||N_l|) Σ_{i∈N_m, j∈N_l} |sin(θ_j-θ_i)|
    # against the OA-side n_terms-truncated Fourier prediction at the same
    # (t, m, l).  A perfect mean-field would put every point on y = x.
    ax_d = fig.add_subplot(gs[1, 1])

    s_emp = res["s_emp_late"]               # (T_late, M, M)
    s_oa  = res["s_oa_late"]                # (T_late, M, M)
    t_cutoff = res["t_cutoff"]
    n_late   = s_emp.shape[0]
    M_       = s_emp.shape[1]

    # Optionally drop diagonal (i.e. self-ensemble pairs) — keep them, but
    # mark with a different colour for clarity.
    diag_mask = np.eye(M_, dtype=bool)[None, :, :]
    diag_mask = np.broadcast_to(diag_mask, s_emp.shape)
    off_x = s_oa[~diag_mask].ravel()
    off_y = s_emp[~diag_mask].ravel()
    dia_x = s_oa[diag_mask].ravel()
    dia_y = s_emp[diag_mask].ravel()

    ax_d.scatter(off_x, off_y, s=4, alpha=0.25, color="C0",
                 label=fr"off-diagonal $(m \neq l)$, $n_\mathrm{{points}}={off_x.size}$")
    ax_d.scatter(dia_x, dia_y, s=8, alpha=0.45, color="C3",
                 label=fr"diagonal $(m = l)$, $n_\mathrm{{points}}={dia_x.size}$")

    # y = x reference
    lo = float(min(off_x.min(), off_y.min(), dia_x.min(), dia_y.min()))
    hi = float(max(off_x.max(), off_y.max(), dia_x.max(), dia_y.max()))
    pad = 0.02 * (hi - lo) if hi > lo else 0.02
    lims = (lo - pad, hi + pad)
    ax_d.plot(lims, lims, "k--", lw=1.0, alpha=0.7, label=r"$y=x$")
    ax_d.set_xlim(lims)
    ax_d.set_ylim(lims)
    ax_d.set_aspect("equal", adjustable="box")

    ax_d.set_xlabel(r"OA truncation  $\bar s_{ml}^{\,\mathrm{trunc}}(t)$")
    ax_d.set_ylabel(r"KMO empirical  $\bar s_{ml}(t)$")
    ax_d.set_title(fr"(d) Plasticity drive, $t>{t_cutoff:g}$  "
                   fr"($n_\mathrm{{terms}}={res['n_terms']}$)")
    ax_d.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax_d.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f"Adaptive Kuramoto: KMO vs. OA mean-field  "
        f"(dist={params['dist']}, K={params['K']}, M={params['M']}, "
        f"μ={params['mu']}, γ={params['gamma']}, "
        f"$n_\\mathrm{{terms}}$={params['n_terms']})",
        fontsize=11,
    )

    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {savepath}")
    fig.canvas.draw()
    plt.show()


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
    p.add_argument("--N", type=int, default=300)
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--T", type=float, default=150.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--Delta", type=float, default=1.0)
    p.add_argument("--K", type=float, default=0.6)
    p.add_argument("--mu", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.001)
    p.add_argument("--n_terms", type=int, default=1)

    p.add_argument("--method", default="RK45")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--n_eval", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t_cutoff", type=float, default=100.0,
                   help="discard t ≤ t_cutoff when building the "
                        "plasticity-drive scatter")

    p.add_argument("--out", default="single_run_results.png")
    return p.parse_args()


def main():
    args = parse_args()
    print("Single-run configuration")
    print("-" * 60)
    for k, v in vars(args).items():
        print(f"  {k:12s} = {v}")
    print("-" * 60)

    res = run_single(
        dist=args.dist, Delta=args.Delta, K=args.K, M=args.M,
        mu=args.mu, gamma=args.gamma, omega0=args.omega0,
        n_terms=args.n_terms,
        N=args.N, T=args.T, n_eval=args.n_eval,
        method=args.method, rtol=args.rtol, atol=args.atol,
        seed=args.seed, t_cutoff=args.t_cutoff,
    )

    print(f"\n  R_km  range: [{res['R_km'].min():.3f}, {res['R_km'].max():.3f}]"
          f"   <R_km>_t = {res['R_km_mean']:.3f}")
    print(f"  R_oa  range: [{res['R_oa'].min():.3f}, {res['R_oa'].max():.3f}]"
          f"   <R_oa>_t = {res['R_oa_mean']:.3f}")
    print(f"  rmse(R_km, R_oa) = "
          f"{np.sqrt(np.mean((res['R_km']-res['R_oa'])**2)):.4f}")
    n_late = res["s_emp_late"].shape[0]
    M_     = res["s_emp_late"].shape[1]
    print(f"  plasticity scatter: {n_late} time slices × {M_*M_} pairs "
          f"= {n_late * M_ * M_} points (t > {res['t_cutoff']:g})")

    params = vars(args)
    plot_results(res, params, args.out)


if __name__ == "__main__":
    main()