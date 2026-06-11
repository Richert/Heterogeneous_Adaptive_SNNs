"""
Theta-Neuron Network with Gaussian-Mixture Excitability:
Microscopic vs. Ott-Antonsen Comparison Across Plasticity Rules
================================================================
Combines the ensemble-fitting workflow from `kuramoto_ensemble_fitting.py`
with the pulse-coupled adaptive theta-neuron model from
`theta_pulsed_adaptation.py`.

Workflow
--------
    1. Sample N microscopic excitabilities η_i from a 3-component Gaussian
       mixture (ground-truth distribution; not Lorentzian).
    2. Fit a sum of M Cauchy components to that sample (so the OA reduction
       has analytical r_m, ψ_m, Δ_m).  This gives (w_pop, η̄_pop, Δ_pop).
    3. Hard-assign each neuron to its most likely Cauchy component.
    4. For each plasticity rule ∈ {hebbian, antihebbian, oja}:
        a. Run the microscopic theta-neuron model.
        b. Run a weight-aware OA mean-field with the fitted ensemble
           parameters (the existing `oa_ode` in `theta_pulsed_adaptation`
           assumes uniform weights 1/M; here we replace it with one that
           takes the Cauchy-fit weights w_pop into account).
        c. Coarse-grain the microscopic coupling using the assignment labels.

Figure layout (PRX two-column-wide, matching `kuramoto_ensemble_fitting.py`)
---------------------------------------------------------------------------
    2 rows × 9 gridspec columns:

        row 0:  cols 0-1 ── η_i distribution + Cauchy fit
                cols 3-4 ── plasticity-rule shapes vs. φ
                cols 6-7 ── pulse rate s̄(t) for the Oja rule
        row 1:  per-rule [TN A_cg | OA A] blocks, plus a manually
                positioned colorbar right next to each OA matrix

    Connectivity matrices use a sequential (positive-only) colormap.  Each
    colorbar is positioned via `fig.add_axes` after the gridspec layout is
    finalised, which lets the OA→colorbar gap be tighter than the global
    `wspace` would allow while still keeping every TN and OA matrix at the
    same size (since they live in equal-width gridspec columns).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.stats import cauchy

# Reuse machinery from the two source scripts ----------------------------------
from kuramoto_ensemble_fitting import (
    sample_gaussian_mixture, gaussian_mixture_pdf, cauchy_mixture_pdf,
    fit_cauchy_mixture, assign_to_populations,
    set_prx_style, make_panel_label,
    C_KM, C_OA, C_GMM, C_FIT, CMAP_A,
)
from theta_pulsed_adaptation import (
    PLASTICITY_RULES, _EPS,
    coupling_norm, fourier_coeffs_s,
    oa_synaptic_mean, oa_pulse_squared_mean, s_micro,
    tn_ode, tn_order_parameter, tn_coarse_grain,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Weight-aware OA ODE for the theta-neuron model
# ═══════════════════════════════════════════════════════════════════════════════
# The OA in `theta_pulsed_adaptation.oa_ode` assumes ensembles are uniformly
# weighted (1/M).  When ensembles come from a Cauchy-mixture fit they carry
# component weights w_pop that sum to 1, so the mean-field synaptic input is
#     E_m = η̄_m + J Σ_n w_n A_mn S_n         (note: no /M factor)
# and the global order parameter is
#     R(t) = | Σ_m w_m R_m e^{iΨ_m} |.
# Setting w_pop = 1/M everywhere recovers the original module's behaviour.

def oa_ode_weighted(t, y, eta_pop, delta_pop, w_pop, J, mu, gamma,
                    n_pulse, s_hat, cn,
                    n1, s_hat1, cn1, s_hat1_sq,
                    n2, s_hat2, cn2,
                    n3, s_hat3, cn3,
                    plasticity):
    M = len(eta_pop)
    R   = np.clip(y[:M], _EPS, 1.0 - _EPS)
    Psi = y[M:2 * M]
    A   = y[2 * M:].reshape(M, M)

    # Mean-field synaptic input — weighted by w_pop instead of 1/M
    S = oa_synaptic_mean(n_pulse, R, Psi, s_hat, cn)
    E = eta_pop + J * (A @ (w_pop * S))

    cos_P = np.cos(Psi)
    sin_P = np.sin(Psi)

    dR = (-delta_pop * R
          - delta_pop * (1.0 + R ** 2) / 2.0 * cos_P
          + (E - 1.0) * (1.0 - R ** 2) / 2.0 * sin_P)

    dPsi = ((E + 1.0)
            + (E - 1.0) * (1.0 + R ** 2) / (2.0 * R) * cos_P
            + delta_pop * (1.0 - R ** 2) / (2.0 * R) * sin_P)

    # Pre/post-synaptic plasticity kernels — these are *expectations*, so
    # they depend on r_m, ψ_m and not on w_pop (the per-ensemble averages
    # are unweighted; w_pop only appears when summing across ensembles).
    S2 = oa_synaptic_mean(n2, R, Psi, s_hat2, cn2)
    S3 = oa_synaptic_mean(n3, R, Psi, s_hat3, cn3)
    hebb = np.outer(S2, S3)

    if plasticity == "hebbian":
        dA = mu * hebb - gamma * A
    elif plasticity == "antihebbian":
        S1 = oa_synaptic_mean(n1, R, Psi, s_hat1, cn1)
        dA = mu * (S1[np.newaxis, :] - A * hebb) - gamma * A
    elif plasticity == "oja":
        Ssq = oa_pulse_squared_mean(n1, R, Psi, s_hat1_sq, cn1)
        dA = mu * (hebb - A * Ssq[:, np.newaxis]) - gamma * A
    else:
        raise ValueError(f"Invalid plasticity='{plasticity}'. "
                         f"Choose from {PLASTICITY_RULES}.")

    return np.concatenate([dR, dPsi, dA.ravel()])


def oa_order_parameter_weighted(R_t, Psi_t, w_pop):
    """R(t) = |Σ_m w_m R_m e^{iΨ_m}| with shapes (M, T) → (T,)."""
    return np.abs(w_pop @ (R_t * np.exp(1j * Psi_t)))


def coarse_grain_labels(A_fine, labels, M):
    """Mean of A over (block_m, block_n) using assignment `labels`."""
    Ac = np.zeros((M, M))
    for I in range(M):
        idx_I = np.where(labels == I)[0]
        if idx_I.size == 0:
            continue
        for Jj in range(M):
            idx_J = np.where(labels == Jj)[0]
            if idx_J.size == 0:
                continue
            Ac[I, Jj] = A_fine[np.ix_(idx_I, idx_J)].mean()
    return Ac


def plasticity_kernel_shape(plasticity, phi, n1, n2, n3, A_ref=1.0):
    """
    Return Ȧ(φ) for one plasticity rule, with γ=0, μ=1, and A_ij = A_ref.

    Parametrisation
    ---------------
    A theta neuron fires when θ crosses π.  We anchor the post-synaptic
    neuron at its firing peak,  θ_i = π,  and let the pre-synaptic peak
    be displaced by φ:  θ_j = π + φ.  Then φ ∈ [-π, π] measures the
    pulse-peak lag, and the kernel shape is

        hebbian      : Ȧ = s(n2, π) · s(n3, π+φ)
        antihebbian  : Ȧ = s(n1, π+φ) - A_ref · s(n2, π) · s(n3, π+φ)
        oja          : Ȧ = s(n2, π) · s(n3, π+φ) - A_ref · s(n1, π)²

    The antihebbian and oja rules include an A-dependent term, so the
    plotted shape depends on the reference coupling A_ref.  A_ref = 1.0
    is a sensible default for a "saturated" weight.
    """
    cn1 = coupling_norm(n1)
    cn2 = coupling_norm(n2)
    cn3 = coupling_norm(n3)

    theta_i = np.pi
    theta_j = np.pi + phi

    s2_post = s_micro(n2, np.atleast_1d(theta_i), cn2)[0]   # scalar
    s3_pre  = s_micro(n3, theta_j, cn3)                     # vector over phi
    s1_pre  = s_micro(n1, theta_j, cn1)                     # vector over phi
    s1_post = s_micro(n1, np.atleast_1d(theta_i), cn1)[0]   # scalar

    if plasticity == "hebbian":
        return s2_post * s3_pre
    if plasticity == "antihebbian":
        return s1_pre - A_ref * s2_post * s3_pre
    if plasticity == "oja":
        return s2_post * s3_pre - A_ref * s1_post ** 2 * np.ones_like(phi)
    raise ValueError(f"Invalid plasticity='{plasticity}'")


# ═══════════════════════════════════════════════════════════════════════════════
# One full (microscopic + OA) run for a given plasticity rule
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_rule(plasticity, *,
                 eta_micro, theta0, A0_micro,
                 w_pop, eta_pop, delta_pop, labels, r0, psi0, A0_oa,
                 N, M, T, J, mu, gamma,
                 n_pulse, n1, n2, n3,
                 method, rtol, atol):
    """Run TN and weight-aware OA for one plasticity rule; return everything."""
    print(f"\n▸ Plasticity rule: {plasticity}")

    # Precompute pulse normalisations and Fourier coefficients
    cn,  s_hat   = coupling_norm(n_pulse), fourier_coeffs_s(n_pulse)
    cn1, s_hat1  = coupling_norm(n1),      fourier_coeffs_s(n1)
    cn2, s_hat2  = coupling_norm(n2),      fourier_coeffs_s(n2)
    cn3, s_hat3  = coupling_norm(n3),      fourier_coeffs_s(n3)
    s_hat1_sq    = fourier_coeffs_s(2 * n1)

    # ── Microscopic simulation ────────────────────────────────────────────────
    y0_tn = np.concatenate([theta0, A0_micro.ravel()])
    print("  Running TN (microscopic) …")
    sol_tn = solve_ivp(
        tn_ode, (0, T), y0_tn, method=method,
        args=(eta_micro, J, mu, gamma,
              n_pulse, cn,
              n1, cn1, n2, cn2, n3, cn3,
              plasticity),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_tn.success:
        raise RuntimeError(f"TN failed: {sol_tn.message}")
    t_tn   = sol_tn.t
    theta  = sol_tn.y[:N]
    A_tn   = sol_tn.y[N:].reshape(N, N, -1)
    R_tn   = tn_order_parameter(theta)
    print(f"    done — {sol_tn.t.size} steps, {sol_tn.nfev} evals")

    # ── OA simulation (weight-aware) ──────────────────────────────────────────
    y0_oa = np.concatenate([r0, psi0, A0_oa.ravel()])
    print("  Running OA (mean-field) …")
    sol_oa = solve_ivp(
        oa_ode_weighted, (0, T), y0_oa, method=method,
        args=(eta_pop, delta_pop, w_pop, J, mu, gamma,
              n_pulse, s_hat, cn,
              n1, s_hat1, cn1, s_hat1_sq,
              n2, s_hat2, cn2,
              n3, s_hat3, cn3,
              plasticity),
        rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol_oa.success:
        raise RuntimeError(f"OA failed: {sol_oa.message}")
    t_oa   = sol_oa.t
    R_oa_m = sol_oa.y[:M]
    Psi_oa = sol_oa.y[M:2 * M]
    A_oa   = sol_oa.y[2 * M:].reshape(M, M, -1)
    R_oa   = oa_order_parameter_weighted(R_oa_m, Psi_oa, w_pop)
    print(f"    done — {sol_oa.t.size} steps, {sol_oa.nfev} evals")

    # Coarse-grain the microscopic final matrix using the fit's labels
    A_tn_cg = coarse_grain_labels(A_tn[:, :, -1], labels, M)

    # ── Mean pulse rate s̄(t) = <s(n_pulse, θ)> for both TN and OA ──────────
    # TN: average over neurons → shape (T,)
    s_tn = cn * np.mean((1.0 - np.cos(theta)) ** n_pulse, axis=0)
    # OA: per-ensemble mean S_m(t), then weighted average over ensembles
    S_per_ens = np.empty_like(R_oa_m)
    for k in range(R_oa_m.shape[1]):
        S_per_ens[:, k] = oa_synaptic_mean(
            n_pulse, R_oa_m[:, k], Psi_oa[:, k], s_hat, cn,
        )
    s_oa = w_pop @ S_per_ens                                  # shape (T_oa,)

    return dict(
        plasticity=plasticity,
        t_tn=t_tn, R_tn=R_tn, A_tn_cg=A_tn_cg, s_tn=s_tn,
        t_oa=t_oa, R_oa=R_oa, A_oa_final=A_oa[:, :, -1], s_oa=s_oa,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Full sweep over the three plasticity rules
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_three_rules(
        N=500, M=12, T=150.0,
        J=-3.0, mu=0.01, gamma=0.001,
        n_pulse=10, n1=2, n2=2, n3=3,
        # Ground-truth 3-Gaussian-mixture parameters for η_i
        gmm_means=(-0.5, 0.5, 1.5),
        gmm_sigmas=(0.3, 0.25, 0.4),
        gmm_weights=(0.3, 0.4, 0.3),
        seed=42,
        method="RK45", rtol=1e-6, atol=1e-8,
):
    print(f"Sampling η_i from 3-Gaussian mixture  (N={N})")
    eta_micro = sample_gaussian_mixture(N, gmm_means, gmm_sigmas, gmm_weights,
                                        seed=seed)
    rng       = np.random.default_rng(seed + 1)
    theta0    = rng.uniform(-np.pi, np.pi, N)

    print(f"Fitting Cauchy mixture with M={M} components …")
    w_pop, eta_pop, delta_pop = fit_cauchy_mixture(eta_micro, M,
                                                   seed=seed, verbose=True)
    labels = assign_to_populations(eta_micro, w_pop, eta_pop, delta_pop)

    # Matched OA initial conditions
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

    A0_micro = np.ones((N, N))
    A0_oa    = np.ones((M, M))

    runs = {}
    for rule in PLASTICITY_RULES:
        runs[rule] = run_one_rule(
            rule,
            eta_micro=eta_micro, theta0=theta0, A0_micro=A0_micro,
            w_pop=w_pop, eta_pop=eta_pop, delta_pop=delta_pop,
            labels=labels, r0=r0, psi0=psi0, A0_oa=A0_oa,
            N=N, M=M, T=T, J=J, mu=mu, gamma=gamma,
            n_pulse=n_pulse, n1=n1, n2=n2, n3=n3,
            method=method, rtol=rtol, atol=atol,
        )

    return dict(
        N=N, M=M, T=T, J=J, mu=mu, gamma=gamma,
        n_pulse=n_pulse, n1=n1, n2=n2, n3=n3,
        eta_micro=eta_micro, theta0=theta0, labels=labels,
        w_pop=w_pop, eta_pop=eta_pop, delta_pop=delta_pop,
        gmm_means=gmm_means, gmm_sigmas=gmm_sigmas, gmm_weights=gmm_weights,
        runs=runs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2 × 9 PRX-style figure (per-rule matrices with manually placed colorbars,
# distribution fit, plasticity-rule shapes, Oja-only pulse rate)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_figure(res, savepath="theta_three_rules.pdf"):
    set_prx_style()

    rules = list(PLASTICITY_RULES)   # ("hebbian", "antihebbian", "oja")
    N, M, T = res["N"], res["M"], res["T"]
    n_pulse, n1, n2, n3 = (res["n_pulse"], res["n1"], res["n2"], res["n3"])

    # Per-rule colour scales for the matrix panels.  All final couplings are
    # non-negative (the plasticity rules produce purely positive drives at the
    # parameters used here), so we use a sequential colormap with vmin=0
    # rather than a diverging one that wastes the negative half of its range.
    CMAP_POS = "viridis"
    per_rule_norm = {}
    for rule in rules:
        d = res["runs"][rule]
        vmax = max(d["A_tn_cg"].max(), d["A_oa_final"].max()) or 1.0
        per_rule_norm[rule] = dict(cmap=CMAP_POS, vmin=0.0, vmax=vmax,
                                   interpolation="nearest", aspect="equal")

    # Layout: 2 rows × 9 gridspec columns ─ three rule blocks of
    # (TN | OA | cbar-slot) side by side.  The colorbar SLOTS in the
    # gridspec reserve space for layout balance, but each colorbar is
    # *positioned manually* (via fig.add_axes) right next to its OA panel
    # so the gap can be tighter than the global `wspace`.  The TN and OA
    # matrices remain identical in size because they live in equal-width
    # gridspec columns.
    #
    #   Row 0:  cols 0-1 ── η_i distribution + Cauchy fit
    #           cols 3-4 ── plasticity-rule shapes vs. φ
    #           cols 6-7 ── pulse rate s̄(t) for Oja (other rules omitted)
    #   Row 1:  per-rule [TN A_cg | OA A | manual cbar] blocks
    width_ratios = [1.0, 1.0, 0.35,
                    1.0, 1.0, 0.35,
                    1.0, 1.0, 0.35]
    # Row 0 is reduced by ~2/7 relative to row 1.  Figure height is shrunk
    # in step so the matrix row keeps roughly the same absolute size as
    # before; otherwise the matrices would balloon.
    fig = plt.figure(figsize=(13.0, 5.5))
    gs = gridspec.GridSpec(
        nrows=2, ncols=9, figure=fig,
        height_ratios=[0.679, 1.0],
        width_ratios=width_ratios,
        hspace=0.55, wspace=0.30,
        left=0.045, right=0.985, top=0.93, bottom=0.10,
    )

    rule_cols = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    panel_letters = iter("abcdefghijklmnop")

    # ── Row 0, cols 0-1: η_i distribution + Cauchy mixture fit ──────────────
    ax_f = fig.add_subplot(gs[0, 0:2])
    eta_micro = res["eta_micro"]
    means, sigmas, weights = (res["gmm_means"], res["gmm_sigmas"],
                              res["gmm_weights"])
    w_pop, eta_pop, delta_pop = (res["w_pop"], res["eta_pop"],
                                 res["delta_pop"])

    hi, lo = np.percentile(eta_micro, [99.5, 0.5])
    edges  = np.linspace(lo, hi, 60)
    ax_f.hist(eta_micro, bins=edges, density=True, color=C_KM,
              alpha=0.30, edgecolor="none", label="TN sample")

    x_grid = np.linspace(lo, hi, 600)
    pdf_g  = gaussian_mixture_pdf(x_grid, means, sigmas, weights)
    ax_f.plot(x_grid, pdf_g, color=C_GMM, lw=1.2, ls="-",
              label="Gaussian mix")

    params_flat = np.concatenate([w_pop, eta_pop, delta_pop])
    pdf_c = cauchy_mixture_pdf(x_grid, params_flat)
    ax_f.plot(x_grid, pdf_c, color=C_FIT, lw=1.2, ls="--",
              label=f"Cauchy fit (M={M})")

    if M <= 12:
        for I in range(M):
            comp = w_pop[I] * cauchy.pdf(x_grid, eta_pop[I], delta_pop[I])
            ax_f.plot(x_grid, comp, color=C_FIT, lw=0.6, alpha=0.5)

    ax_f.set_xlabel(r"$\eta$")
    ax_f.set_ylabel(r"$p(\eta)$")
    ax_f.set_xlim(lo, hi)
    ax_f.set_ylim(0.0, 2.0)
    ax_f.legend(loc="upper right", frameon=False, handlelength=2.0,
                borderaxespad=0.3, fontsize=9)
    make_panel_label(ax_f, f"({next(panel_letters)})", x=-0.10, y=1.04)

    # ── Row 0, cols 3-4: plasticity-rule shapes vs pulse-peak lag φ ─────────
    ax_shape = fig.add_subplot(gs[0, 3:5])
    phi = np.linspace(-np.pi, np.pi, 1001)
    rule_colors = {"hebbian":     "#1f4e79",
                   "antihebbian": "#3a8e7c",
                   "oja":         "#c44e52"}
    rule_styles = {"hebbian": "-", "antihebbian": "--", "oja": ":"}
    for rule in rules:
        y = plasticity_kernel_shape(rule, phi, n1=n1, n2=n2, n3=n3, A_ref=1.0)
        ax_shape.plot(phi, y, color=rule_colors[rule],
                      ls=rule_styles[rule], lw=1.4, label=rule)
    ax_shape.axhline(0.0, color="0.6", lw=0.5)
    ax_shape.set_xlabel(r"$\phi = \theta_j - \theta_i$  (pulse-peak lag)")
    ax_shape.set_ylabel(r"$\dot A(\phi)$  ($\mu{=}1,\, \gamma{=}0,\, A{=}1$)")
    ax_shape.set_xlim(-np.pi, np.pi)
    ax_shape.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    ax_shape.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0",
                              r"$\pi/2$", r"$\pi$"])
    ax_shape.legend(loc="best", frameon=False, handlelength=2.0,
                    borderaxespad=0.3, fontsize=9)
    make_panel_label(ax_shape, f"({next(panel_letters)})", x=-0.10, y=1.04)

    # ── Row 0, cols 6-7: mean pulse rate s̄(t) for the Oja rule only ────────
    d_oja = res["runs"]["oja"]
    ax_s = fig.add_subplot(gs[0, 6:8])
    ax_s.plot(d_oja["t_tn"], d_oja["s_tn"], color=C_KM, lw=1.4, label="TN")
    ax_s.plot(d_oja["t_oa"], d_oja["s_oa"], color=C_OA, lw=1.4, ls="--",
              label="OA")
    ax_s.set_xlim(0, T)
    ymax = max(np.max(d_oja["s_tn"]), np.max(d_oja["s_oa"])) * 1.05
    ax_s.set_ylim(0, ymax)
    ax_s.set_xlabel(r"time $t$")
    ax_s.set_ylabel(r"$\langle s(n,\theta)\rangle$  (oja)")
    ax_s.legend(loc="upper right", frameon=False, handlelength=2.0,
                borderaxespad=0.3, fontsize=9)
    make_panel_label(ax_s, f"({next(panel_letters)})", x=-0.10, y=1.04)

    # ── Row 1: per-rule TN + OA matrices ────────────────────────────────────
    # We keep references to the OA axes and the colorbar mappables so we can
    # position the colorbars *after* the gridspec layout is finalised.
    pending_cbars = []
    for r_idx, rule in enumerate(rules):
        col_tn, col_oa, col_cb = rule_cols[r_idx]
        d        = res["runs"][rule]
        norm_kw  = per_rule_norm[rule]

        ax_tn = fig.add_subplot(gs[1, col_tn])
        ax_oa = fig.add_subplot(gs[1, col_oa])

        ax_tn.imshow(d["A_tn_cg"], **norm_kw)
        im_oa = ax_oa.imshow(d["A_oa_final"], **norm_kw)

        ax_tn.set_title("TN (block-avg)")
        ax_oa.set_title("OA mean-field")

        # Integer ticks + xlabels on the connectivity matrices.  M is small
        # (~12), so we show every other tick to avoid crowding.
        tick_step = max(1, M // 6)
        ticks     = np.arange(0, M, tick_step)
        ax_tn.set_xticks(ticks); ax_tn.set_yticks(ticks)
        ax_oa.set_xticks(ticks); ax_oa.set_yticks(ticks)
        ax_tn.set_xticklabels(ticks); ax_tn.set_yticklabels(ticks)
        ax_oa.set_xticklabels(ticks); ax_oa.set_yticklabels(ticks)
        ax_tn.tick_params(axis="both", labelsize=9)
        ax_oa.tick_params(axis="both", labelsize=9)
        ax_tn.set_xlabel(r"ensemble $n$")
        ax_oa.set_xlabel(r"ensemble $n$")
        # ylabel scheme:
        #   TN column → bold rule name only (no "ensemble m")
        #   OA column → plain "ensemble m" matching the xlabel
        ax_tn.set_ylabel(rule, fontsize=12, fontweight="bold")
        ax_oa.set_ylabel(r"ensemble $m$", fontsize=10)

        pending_cbars.append((ax_oa, im_oa))

        # Only the TN columns get a panel label, so row 1 reads (d)(e)(f)
        # instead of (d)(e)(f)(g)(h)(i).
        make_panel_label(ax_tn, f"({next(panel_letters)})", x=-0.10, y=1.04)

    # Finalise layout, then position each colorbar in figure coordinates
    # right next to its OA panel.  Done after `draw` so positions are correct.
    fig.canvas.draw()
    CBAR_W   = 0.010   # colorbar width  in figure-fraction coords
    CBAR_PAD = 0.006   # gap from OA right edge to colorbar left edge
    for ax_oa, im_oa in pending_cbars:
        pos = ax_oa.get_position()
        cax = fig.add_axes([pos.x1 + CBAR_PAD, pos.y0, CBAR_W, pos.height])
        cb  = fig.colorbar(im_oa, cax=cax)
        cb.set_label(r"$A_{mn}$", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        cb.locator = plt.MaxNLocator(nbins=4)
        cb.update_ticks()

    fig.savefig(savepath, bbox_inches="tight")
    print(f"\nFigure saved → {savepath}")
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CONFIG = dict(
        N=500, M=5, T=150.0,
        J=-2.0, mu=0.02, gamma=0.001,
        n_pulse=10, n1=2, n2=2, n3=3,
        gmm_means=(-0.0, 0.5, 1.0),
        gmm_sigmas=(0.2, 0.15, 0.1),
        gmm_weights=(0.3, 0.4, 0.3),
        seed=42, method="RK45", rtol=1e-6, atol=1e-8,
    )

    res = simulate_three_rules(**CONFIG)
    plot_figure(res, savepath="theta_three_rules.svg")