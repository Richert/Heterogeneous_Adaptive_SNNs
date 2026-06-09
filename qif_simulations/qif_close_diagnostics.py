"""
qif_closure_diagnostics.py
==========================

Measure the closure terms that the MPR ensemble mean-field DROPS when the
microscopic weight matrix A_ij is reduced to its block means A_mn.

Drop-in companion to `qif_ensemble_fitting.py`. It reuses that module's
population fitting, the `mpr_ode` right-hand side, and `coarse_grain_labels`,
and adds a *diagnostic copy* of the microscopic integrator that records, per
ensemble block (m, n) and per recorded time:

    D_true_mn : true block-averaged UNCLIPPED plasticity drive
                  = < a+ P_ij - a- Q_ij >_{i in m, j in n}
    D_mf_mn   : the mean-field drive built from block-mean activity only
                  (exactly what mpr_ode uses)
    D_real_mn : the realized block-averaged drive AFTER hard-clipping A to [0,1]
    Avar_mn   : within-block variance of A_ij
    svar_n    : within-ensemble variance of s_i

From these we form two gaps:

    activity_gap = D_true - D_mf
        Nonzero ONLY for drive terms that are nonlinear in a SINGLE neuron's
        variables -- those do not factor across the block. Products of two
        DISTINCT neurons factor exactly (a rectangular double sum is the
        product of two single sums), so for those terms this gap is ~0 to
        machine precision. With the rules as actually implemented:
            Hebbian      -> both terms are two-neuron products  => ~0
            anti-Hebbian -> x_j (pre only) and s_i*y_j (two-neuron) => ~0
            Oja          -> s_i*s_j (two-neuron) and s_i*y_i (post only)
                            => nonzero, via Cov_m(s_i, y_i) within ensemble m
        i.e. the diagnostic should confirm activity_gap ~ 0 for Hebbian /
        anti-Hebbian and != 0 for Oja. This is a built-in sanity check.

    clip_gap = D_real - D_true
        The Jensen-type gap through the hard clip: nonzero when pairs within a
        block saturate heterogeneously (some A_ij pinned at 0 or 1, others
        free), so the block-mean drive differs from the drive evaluated at the
        block mean. This is the *universal* mechanism here, and is driven by
        the within-block weight spread Avar_mn.

Decision rule
-------------
If both gaps stay small relative to ||D_true|| over the run, the block-mean MF
is self-consistent and no covariance / second-moment term is warranted. If a
gap is large, it tells you WHICH closure term to add: single-neuron activity
variance (activity_gap) vs. weight-saturation heterogeneity (clip_gap, tracked
via Avar_mn).

NOTE: the per-pair drives below mirror the ACTUAL implementation in
`simulate_qif_micro` / `mpr_ode` (constant a+/a- plus a hard clip, and for
anti-Hebbian / Oja the x,y traces rather than the raw s^2 of the docstrings).
The diagnostic measures what the code does, not what the docstrings say. If you
change the rules in the main file, mirror the change in `pair_drive` and
`mf_block_drive` below.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Reuse the building blocks from the main script (must be importable, i.e. on
# the same path / PYTHONPATH). The __main__ guard there prevents side effects.
from qif_ensemble_fitting import (
    sample_gaussian_mixture,
    fit_cauchy_mixture,
    assign_to_populations,
    coarse_grain_labels,
    mpr_ode,
    PLASTICITY_RULES,
)

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Block-averaging helpers
# ─────────────────────────────────────────────────────────────────────────────
def assignment_matrix(labels, M):
    """
    (N, M) column-normalized one-hot matrix S with S[i, m] = 1/N_m if
    label_i == m else 0. Then for any (N, N) matrix X:
        block_mean(X)[m, n] = (S.T @ X @ S)[m, n]
                            = (1/(N_m N_n)) sum_{i in m, j in n} X[i, j]
    and for any (N,) vector v: ensemble_mean(v) = S.T @ v.
    """
    N = len(labels)
    S = np.zeros((N, M))
    S[np.arange(N), labels] = 1.0
    counts = S.sum(axis=0)
    counts[counts == 0] = 1.0
    return S / counts[None, :]


def pair_drive(rule, s, x, y, a_plus, a_minus):
    """
    Full (N, N) per-pair UNCLIPPED plasticity drive dA_ij, rows = post i,
    cols = pre j. Mirrors `simulate_qif_micro` exactly.
    """
    N = s.shape[0]
    if rule == "Hebbian":
        pos = np.outer(s, x)                       # s_i x_j  (post, pre)
        neg = np.outer(y, s)                       # y_i s_j
    elif rule == "anti-Hebbian":
        pos = np.tile(x[None, :], (N, 1))          # x_j      (pre only)
        neg = np.outer(s, y)                       # s_i y_j
    elif rule == "Oja":
        pos = np.outer(s, s)                       # s_i s_j
        neg = np.tile((s * y)[:, None], (1, N))    # s_i y_i  (post only)
    else:
        raise ValueError(f"Invalid rule: {rule}")
    return a_plus * pos - a_minus * neg


def mf_block_drive(rule, s_m, x_m, y_m, a_plus, a_minus):
    """
    (M, M) mean-field drive built from block-mean activity only. Mirrors the
    drive computed inside `mpr_ode`.
    """
    M = s_m.shape[0]
    if rule == "Hebbian":
        pos = np.outer(s_m, x_m)
        neg = np.outer(y_m, s_m)
    elif rule == "anti-Hebbian":
        pos = np.tile(x_m[None, :], (M, 1))
        neg = np.outer(s_m, y_m)
    elif rule == "Oja":
        pos = np.outer(s_m, s_m)
        neg = np.tile((s_m * y_m)[:, None], (1, M))
    else:
        raise ValueError(f"Invalid rule: {rule}")
    return a_plus * pos - a_minus * neg


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic microscopic integrator (faithful copy of simulate_qif_micro + recorders)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_qif_micro_diag(eta_i, labels, A0_micro, T, J, a_minus, a_plus,
                            tau_s, plasticity, V_peak=50.0, dt=5e-4, seed=42,
                            tau_x=1.0, tau_y=1.0, alpha_spike=1.0,
                            record_dt=0.5, plast_update_every=10, verbose=False):
    rng = np.random.default_rng(seed)
    N = len(eta_i)
    M = int(labels.max()) + 1
    S = assignment_matrix(labels, M)

    V = -2.0 * np.ones(N) + 0.1 * rng.normal(size=N)
    s_i = np.zeros(N); B_i = np.zeros(N)
    x_i = np.zeros(N); y_i = np.zeros(N)
    A = A0_micro.copy()

    n_steps = int(T / dt)
    rec_steps = int(record_dt / dt)
    n_rec = n_steps // rec_steps
    t_rec = (np.arange(n_rec) + 0.5) * record_dt

    inv_tau_s = 1.0 / tau_s
    inv_tau_x = 1.0 / tau_x
    inv_tau_y = 1.0 / tau_y
    inv_N = 1.0 / N
    spike_bump = alpha_spike / tau_s
    N_per_ens = np.bincount(labels, minlength=M).astype(float)
    N_per_ens[N_per_ens == 0] = 1.0
    dt_plast = dt * plast_update_every

    # recordings
    r_m_rec   = np.zeros((M, n_rec))
    s_m_rec   = np.zeros((M, n_rec))
    svar_rec  = np.zeros((M, n_rec))
    Abar_rec  = np.zeros((M, M, n_rec))
    Avar_rec  = np.zeros((M, M, n_rec))
    Dtrue_rec = np.zeros((M, M, n_rec))
    Dmf_rec   = np.zeros((M, M, n_rec))
    Dreal_rec = np.zeros((M, M, n_rec))

    spike_count_bin = np.zeros(M, dtype=np.int64)
    rec_idx = 0

    if verbose:
        print(f"  [diag] N={N}, M={M}, T={T}, n_rec={n_rec}, rule={plasticity}")

    for step in range(n_steps):
        I_per_neuron = (J * inv_N) * (A @ s_i)
        V2 = V + dt * (V * V + eta_i + I_per_neuron)
        s2 = s_i + dt * (B_i * inv_tau_s)
        B2 = B_i + dt * ((-2.0 * B_i - s_i) * inv_tau_s)
        x2 = x_i + dt * (-x_i * inv_tau_x + s_i)
        y2 = y_i + dt * (-y_i * inv_tau_y + s_i)

        if (step + 1) % plast_update_every == 0:
            dA = pair_drive(plasticity, s_i, x_i, y_i, a_plus, a_minus)
            A += dt_plast * dA
            np.clip(A, 0.0, 1.0, out=A)

        V = V2; s_i = s2; B_i = B2; x_i = x2; y_i = y2

        spiked = V >= V_peak
        if spiked.any():
            V[spiked] = -V_peak
            B_i[spiked] += spike_bump
            spike_count_bin += np.bincount(labels[spiked], minlength=M)

        if (step + 1) % rec_steps == 0 and rec_idx < n_rec:
            s_m = S.T @ s_i
            x_m = S.T @ x_i
            y_m = S.T @ y_i

            r_m_rec[:, rec_idx]  = spike_count_bin / (N_per_ens * record_dt)
            s_m_rec[:, rec_idx]  = s_m
            svar_rec[:, rec_idx] = (S.T @ (s_i * s_i)) - s_m ** 2

            Abar = S.T @ A @ S
            Abar_rec[:, :, rec_idx] = Abar
            Avar_rec[:, :, rec_idx] = (S.T @ (A * A) @ S) - Abar ** 2

            dA_full = pair_drive(plasticity, s_i, x_i, y_i, a_plus, a_minus)
            Dtrue = S.T @ dA_full @ S
            Dtrue_rec[:, :, rec_idx] = Dtrue
            Dmf_rec[:, :, rec_idx]   = mf_block_drive(plasticity, s_m, x_m, y_m,
                                                      a_plus, a_minus)
            realized = (np.clip(A + dt_plast * dA_full, 0.0, 1.0) - A) / dt_plast
            Dreal_rec[:, :, rec_idx] = S.T @ realized @ S

            spike_count_bin[:] = 0
            rec_idx += 1

    return dict(
        t_rec=t_rec, M=M, N=N,
        r_m_rec=r_m_rec, s_m_rec=s_m_rec, svar_rec=svar_rec,
        Abar_rec=Abar_rec, Avar_rec=Avar_rec,
        Dtrue_rec=Dtrue_rec, Dmf_rec=Dmf_rec, Dreal_rec=Dreal_rec,
        A_final_micro=A.copy(),
    )


def run_mf(plasticity, w_pop, eta_pop, delta_pop, r0, v0, s0, A0_mf,
           M, T, J, a_minus, a_plus, tau_s, tau_x, tau_y,
           method="RK45", rtol=1e-6, atol=1e-8):
    B0 = np.zeros(M); x0 = np.zeros(M); y0 = np.zeros(M)
    y0_mf = np.concatenate([r0, v0, s0, B0, x0, y0, A0_mf.ravel()])
    sol = solve_ivp(
        mpr_ode, (0, T), y0_mf, method=method,
        args=(eta_pop, delta_pop, w_pop, J, a_minus, a_plus,
              tau_s, tau_x, tau_y, plasticity),
        rtol=rtol, atol=atol, max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"MF failed: {sol.message}")
    A_mf_final = sol.y[6 * M:].reshape(M, M, -1)[:, :, -1]
    return sol, A_mf_final


# ─────────────────────────────────────────────────────────────────────────────
# Metric assembly
# ─────────────────────────────────────────────────────────────────────────────
def _fro_t(arr_MMt):
    """Frobenius norm over the (M, M) block axes for each time -> (n_rec,)."""
    return np.sqrt(np.einsum("ijt,ijt->t", arr_MMt, arr_MMt))


def compute_for_rule(rule, *, eta_micro, labels, A0_micro,
                     w_pop, eta_pop, delta_pop, r0, v0, s0, A0_mf,
                     M, T, J, a_minus, a_plus, tau_s, tau_x, tau_y,
                     V_peak, dt_micro, record_dt, seed,
                     method, rtol, atol):
    print(f"\n▸ closure diagnostics — {rule}")
    diag = simulate_qif_micro_diag(
        eta_i=eta_micro, labels=labels, A0_micro=A0_micro, T=T,
        J=J, a_minus=a_minus, a_plus=a_plus, tau_s=tau_s,
        plasticity=rule, V_peak=V_peak, dt=dt_micro, seed=seed,
        tau_x=tau_x, tau_y=tau_y, record_dt=record_dt, verbose=True,
    )
    _, A_mf_final = run_mf(
        rule, w_pop, eta_pop, delta_pop, r0, v0, s0, A0_mf,
        M, T, J, a_minus, a_plus, tau_s, tau_x, tau_y,
        method=method, rtol=rtol, atol=atol,
    )
    A_qif_cg = coarse_grain_labels(diag["A_final_micro"], labels, M)

    t = diag["t_rec"]
    Dtrue = diag["Dtrue_rec"]; Dmf = diag["Dmf_rec"]; Dreal = diag["Dreal_rec"]
    denom = _fro_t(Dtrue) + _EPS

    act_gap_t  = _fro_t(Dtrue - Dmf)  / denom     # single-neuron-nonlinearity gap
    clip_gap_t = _fro_t(Dreal - Dtrue) / denom    # saturation-heterogeneity gap

    # within-block weight spread (rms std over blocks), absolute and relative
    Avar = diag["Avar_rec"]
    w_std_t = np.sqrt(np.clip(Avar, 0, None).mean(axis=(0, 1)))
    Abar_mean_t = np.abs(diag["Abar_rec"]).mean(axis=(0, 1)) + _EPS
    w_cv_t = w_std_t / Abar_mean_t

    # within-ensemble activity spread
    svar = diag["svar_rec"]
    s_std_t = np.sqrt(np.clip(svar, 0, None).mean(axis=0))
    s_mean_t = np.abs(diag["s_m_rec"]).mean(axis=0) + _EPS
    s_cv_t = s_std_t / s_mean_t

    final_mismatch = (np.sqrt(((A_qif_cg - A_mf_final) ** 2).sum())
                      / (np.sqrt((A_qif_cg ** 2).sum()) + _EPS))

    return dict(
        rule=rule, t=t,
        act_gap_t=act_gap_t, clip_gap_t=clip_gap_t,
        w_std_t=w_std_t, w_cv_t=w_cv_t, s_std_t=s_std_t, s_cv_t=s_cv_t,
        A_qif_cg=A_qif_cg, A_mf_final=A_mf_final,
        final_mismatch=final_mismatch,
        # time-averaged scalars for the summary table
        act_gap=np.nanmean(act_gap_t),
        clip_gap=np.nanmean(clip_gap_t),
        w_cv=np.nanmean(w_cv_t),
        s_cv=np.nanmean(s_cv_t),
    )


def run_all(**cfg):
    print(f"Sampling eta_i (N={cfg['N']}) and fitting Cauchy mixture (M={cfg['M']}) …")
    eta_micro = sample_gaussian_mixture(
        cfg["N"], cfg["gmm_means"], cfg["gmm_sigmas"], cfg["gmm_weights"],
        seed=cfg["seed"])
    w_pop, eta_pop, delta_pop = fit_cauchy_mixture(eta_micro, cfg["M"],
                                                   seed=cfg["seed"], verbose=False)
    labels = assign_to_populations(eta_micro, w_pop, eta_pop, delta_pop)
    M = cfg["M"]

    r0 = 0.05 * np.ones(M); v0 = -1.0 * np.ones(M); s0 = np.zeros(M)
    A0_micro = cfg["A0_value"] * np.ones((cfg["N"], cfg["N"]))
    A0_mf = cfg["A0_value"] * np.ones((M, M))

    results = {}
    for rule in PLASTICITY_RULES:
        results[rule] = compute_for_rule(
            rule, eta_micro=eta_micro, labels=labels, A0_micro=A0_micro,
            w_pop=w_pop, eta_pop=eta_pop, delta_pop=delta_pop,
            r0=r0, v0=v0, s0=s0, A0_mf=A0_mf,
            M=M, T=cfg["T"], J=cfg["J"], a_minus=cfg["a_minus"],
            a_plus=cfg["a_plus"], tau_s=cfg["tau_s"], tau_x=cfg["tau_x"],
            tau_y=cfg["tau_y"], V_peak=cfg["V_peak"], dt_micro=cfg["dt_micro"],
            record_dt=cfg["record_dt"], seed=cfg["seed"],
            method=cfg["method"], rtol=cfg["rtol"], atol=cfg["atol"],
        )

    # summary table
    print("\n" + "=" * 78)
    print(f"{'rule':<14}{'act_gap':>12}{'clip_gap':>12}{'weight CV':>12}"
          f"{'act CV':>12}{'final mism.':>14}")
    print("-" * 78)
    for rule in PLASTICITY_RULES:
        r = results[rule]
        print(f"{rule:<14}{r['act_gap']:>12.3e}{r['clip_gap']:>12.3e}"
              f"{r['w_cv']:>12.3e}{r['s_cv']:>12.3e}{r['final_mismatch']:>14.3e}")
    print("=" * 78)
    print("act_gap : ||D_true - D_mf|| / ||D_true||   (single-neuron nonlinearity)")
    print("clip_gap: ||D_real - D_true|| / ||D_true|| (within-block saturation)")
    print("Both are time-averaged. Large clip_gap or weight CV => a second-moment")
    print("(weight-variance) closure is what would help; large act_gap => add the")
    print("within-ensemble activity-variance term for that rule.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_diagnostics(results, savepath="qif_closure_diagnostics.svg"):
    rules = list(PLASTICITY_RULES)
    fig, axes = plt.subplots(len(rules), 3, figsize=(12.5, 3.1 * len(rules)),
                             constrained_layout=True)
    if len(rules) == 1:
        axes = axes[None, :]

    for ri, rule in enumerate(rules):
        r = results[rule]
        t = r["t"]

        ax = axes[ri, 0]
        ax.plot(t, r["act_gap_t"], color="#3a8e7c", lw=1.3,
                label="activity gap")
        ax.plot(t, r["clip_gap_t"], color="#c44e52", lw=1.3,
                label="clip gap")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_ylabel(f"{rule}\nrel. drive gap", fontsize=10)
        if ri == 0:
            ax.set_title("dropped-drive magnitude / ||D_true||")
        ax.legend(frameon=False, fontsize=8, loc="best")
        if ri == len(rules) - 1:
            ax.set_xlabel("time")

        ax = axes[ri, 1]
        ax.plot(t, r["w_cv_t"], color="#1f4e79", lw=1.3,
                label="weight CV (within block)")
        ax.plot(t, r["s_cv_t"], color="#888888", lw=1.1, ls="--",
                label="activity CV (within ens.)")
        if ri == 0:
            ax.set_title("within-group spread")
        ax.set_ylabel("coeff. of variation", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="best")
        if ri == len(rules) - 1:
            ax.set_xlabel("time")

        ax = axes[ri, 2]
        diff = np.abs(r["A_qif_cg"] - r["A_mf_final"])
        im = ax.imshow(diff, cmap="magma", aspect="equal",
                       interpolation="nearest")
        if ri == 0:
            ax.set_title(r"$|A^{\mathrm{qif}}_{mn}-A^{\mathrm{mf}}_{mn}|$ (final)")
        ax.set_xlabel("ensemble n"); ax.set_ylabel("ensemble m")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(savepath, bbox_inches="tight")
    print(f"Figure saved → {savepath}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Modest config for a quick pass; bump N / T / M to match your CONFIG.
    CFG = dict(
        N=200, M=8, T=200.0,
        J=2.0, a_minus=0.01, a_plus=0.003,
        tau_s=0.2, tau_x=0.5, tau_y=2.0,
        gmm_means=(-0.1, 0.0, 0.2),
        gmm_sigmas=(0.08, 0.1, 0.05),
        gmm_weights=(0.3, 0.5, 0.3),
        V_peak=100.0, dt_micro=5e-4,
        record_dt=0.5, seed=42,
        method="RK45", rtol=1e-6, atol=1e-8,
        A0_value=0.5,
    )
    res = run_all(**CFG)
    plot_diagnostics(res, savepath="/home/rgast/data/qif_plasticity/qif_closure_diagnostics.svg")
    plt.show()