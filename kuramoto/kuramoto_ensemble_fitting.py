"""
Figure 1: Ensemble Mean-Field vs. Kuramoto Network with Multi-Modal Frequencies
================================================================================
Compares the microscopic Kuramoto network to the OA mean-field for TWO ensemble
decompositions, obtained with the shared LMMF fitter (theory/lorentzian_mixture.py,
the same algorithm as grid_search/kmo_lorentzian_fit_sweep.py): a weighted mixture
of Lorentzians whose effective order M* is chosen by a penalized Cramér–von Mises
search. The two decompositions differ in a PER-ENSEMBLE WEIGHT-VARIANCE BUDGET: a
maximum allowed V_A/Ā² per ensemble, translated (Eq. 37 of the weight-variance
manuscript, cosine rule) into an upper bound on the ensemble width Δ. A tighter budget
=> narrower ensembles, each with lower internal weight variance. With Δ_min fixed, a
tighter Δ_max is a SUBSET feasible region, so its best fit at any M is no better than the
looser cap's => the tighter budget needs weakly MORE ensembles to reach the same goodness
of fit (M*_tight >= M*_loose). NB this monotonicity only holds if each fixed-M fit is
solved well: the CvM fit is non-convex, so too few `n_restarts` can land the looser cap in
a worse local minimum and spuriously invert the M* ordering — keep n_restarts generous.

The script sweeps a 2×2 grid of scenarios — 2 per-ensemble variance budgets × 2 adaptation
rates μ — and lays them out as a figure with the budgets as rows and μ as columns.

Workflow
--------
    1. Sample N microscopic frequencies from a Gaussian mixture (done once).
    2. For each μ in `mu_list`: run the KMO simulation once (μ sets its adaptation dynamics).
    3. For each (μ, variance budget) cell:
        a. Translate the budget into Δ_max (delta_max_for_vratio) for this μ, γ, K.
        b. Fit a weighted Lorentzian mixture with LMMF under Δ ≤ Δ_max; M* is automatic.
        c. Hard-assign each oscillator to its most-likely ensemble.
        d. Build matched OA initial conditions and simulate the OA mean-field.
        e. Coarse-grain the KMO weight matrix using THIS fit's labels.
    4. Plot a 2×2 grid (budget rows × μ columns); each cell = R(t) micro-vs-OA panel +
       fitted distribution + coarse-grained KMO / OA coupling matrices.

PRX figure constraints
----------------------
    - Two-column width: 7.0 inches (~17.8 cm).
    - Serif font (Computer Modern / STIX) for consistency with REVTeX.
    - 8 pt minimum for labels; 9 pt body, 10 pt panel labels.
    - Vector PDF output.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.stats import cauchy, norm

# the shared LMMF fitter (weighted-Lorentzian mixture with penalized CvM order selection),
# the same algorithm used by grid_search/kmo_lorentzian_fit_sweep.py; and the closed-form
# weight-variance relation V_A/Ā²(Δ) (Eq. 37) used to turn a per-ensemble variance budget
# into an upper bound on the ensemble width Δ.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "theory")))
import lorentzian_mixture as LM
import weight_variance_analysis as WV

_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Per-ensemble variance budget  ->  upper bound on the ensemble width Δ
# ═══════════════════════════════════════════════════════════════════════════════

def delta_max_for_vratio(vratio_max, mu, gamma, K, plasticity="cos", delta_min=1e-4, n=4000):
    r"""Largest ensemble width Δ whose steady-state relative weight variance V_A/Ā² does not
    exceed `vratio_max`, for the network's (μ, γ, K).

    Uses the closed form of the weight-variance manuscript (Eq. 37 via
    weight_variance_analysis.branches): V_A/Ā²(Δ) rises monotonically from 0 on the low-Δ part
    of the inverted-U, so a per-ensemble variance budget maps to a unique Δ_max on that rising
    branch. If the budget exceeds the peak of the inverted-U, the width is only limited by the
    synchronized-branch endpoint (fold / transcritical). Single-ensemble criterion (each fitted
    Lorentzian treated as its own population).

    NOTE — this closed form is the COSINE rule result (Eq. 37), consistent with the manuscript's
    Ȧ=μG+γ(1−A) adaptation used by the KMO/OA ODEs here. The "sin" and "|sin|" rules give a
    different V_A/Ā²(Δ) relation (manuscript App. B); those rule-specific relations are not yet
    implemented, so the variance-budget cap is currently defined only for plasticity='cos'."""
    if plasticity != "cos":
        raise NotImplementedError(
            "delta_max_for_vratio: the V_A/Ā²(Δ) closed form is currently the cosine-rule result "
            f"(Eq. 37); got plasticity={plasticity!r}. The 'sin'/'|sin|' relations differ (App. B) "
            "and are not implemented yet — pass an explicit Δ bound (delta_bounds) for those rules.")
    d_end = WV.sync_delta_end(mu, K, gamma)
    d = np.linspace(delta_min, 0.999 * d_end, n)
    br = WV.branches(d, mu, K, gamma)["sync"]
    rel = br["VA"] / br["A"] ** 2
    ipeak = int(np.nanargmax(rel))
    d_rise, rel_rise = d[:ipeak + 1], rel[:ipeak + 1]        # rising branch up to the peak
    if vratio_max >= np.nanmax(rel_rise):                    # budget above the peak -> no extra cap
        return float(d_rise[-1])
    j = int(np.searchsorted(rel_rise, vratio_max))
    if j <= 0:
        return float(d_rise[0])
    d0, d1, r0, r1 = d_rise[j - 1], d_rise[j], rel_rise[j - 1], rel_rise[j]
    return float(d0 + (d1 - d0) * (vratio_max - r0) / (r1 - r0))   # linear interpolation


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Frequency distribution: multi-modal Gaussian sample
# ═══════════════════════════════════════════════════════════════════════════════

def sample_gaussian_mixture(N, means, sigmas, weights, seed):
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    counts = rng.multinomial(N, weights)
    samples = np.concatenate([
        rng.normal(mu, sig, c) for mu, sig, c in zip(means, sigmas, counts)
    ])
    rng.shuffle(samples)
    return samples


def gaussian_mixture_pdf(x, means, sigmas, weights):
    w = np.asarray(weights, float)
    w = w / w.sum()
    return np.sum(
        [wi * norm.pdf(x, mu, sig)
         for wi, mu, sig in zip(w, means, sigmas)], axis=0,
    )


def phases_for_coherence(N, R0, rng):
    """Draw N initial phases θ_i with population coherence |⟨e^{iθ}⟩| ≈ R0 (wrapped-normal:
    R0 = exp(−σ²/2)). R0≈0 → uniform (incoherent); R0→1 → phase-aligned (coherent). The
    realised R(0) is R0 only up to O(1/√N) sampling noise."""
    R0 = float(np.clip(R0, 0.0, 1.0 - 1e-9))
    if R0 <= 1e-6:
        return rng.uniform(-np.pi, np.pi, N)
    sigma = np.sqrt(-2.0 * np.log(R0))
    return rng.normal(0.0, sigma, N)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Weighted-Lorentzian mixture fit via the shared LMMF algorithm
#    (theory/lorentzian_mixture.py — penalized Cramér–von Mises order selection)
# ═══════════════════════════════════════════════════════════════════════════════

def cauchy_mixture_pdf(x, params):
    """Weighted Cauchy/Lorentzian mixture density from a flat [w, μ, Δ] parameter vector
    (used only for plotting the fitted distribution; the fit itself is done by LMMF)."""
    M = len(params) // 3
    w  = np.asarray(params[0:M])
    mu = np.asarray(params[M:2*M])
    g  = np.asarray(params[2*M:3*M])
    w = np.abs(w); w = w / w.sum()
    g = np.abs(g) + 1e-6
    return np.sum(
        [wi * cauchy.pdf(x, m, gi) for wi, m, gi in zip(w, mu, g)], axis=0,
    )


def fit_lorentzian_mixture(samples, delta_bounds, fit_cfg, seed=0, verbose=True):
    r"""Fit a weighted Lorentzian mixture to `samples` with the shared LMMF algorithm.

    `delta_bounds` = (Δ_min, Δ_max) hard-bounds every ensemble width; here Δ_max is the
    per-ensemble variance budget translated by :func:`delta_max_for_vratio`, so a tighter Δ_max
    yields narrower ensembles (lower internal variance) and LMMF adapts M* to still fit.
    `fit_cfg` carries the shared LMMF settings
    (``M_max``, ``alpha``, ``lambda_M``, ``loss``, ``method``, ``n_restarts``, ``patience``).
    LMMF chooses the EFFECTIVE order M* by a greedy penalized Cramér–von Mises search; the
    returned mixture is already pruned to non-degenerate components. Returns (w, Ω, Δ) sorted by
    centre, plus the LMMF result dict."""
    res = LM.fit(samples, delta_bounds, M_max=fit_cfg["M_max"], alpha=fit_cfg["alpha"],
                 lambda_M=fit_cfg["lambda_M"], patience=fit_cfg["patience"],
                 loss=fit_cfg["loss"], n_restarts=fit_cfg["n_restarts"],
                 seed=seed, method=fit_cfg["method"])
    m = res["model"]
    order = np.argsort(m.Omega)
    w, mu, g = m.w[order], m.Omega[order], m.Delta[order]
    if verbose:
        print(f"LMMF fit (Δ≤{delta_bounds[1]:.4f}): M*={w.size}  "
              f"1−p={1 - res['pvalue']:.3f}  CvM D={res['data_loss']:.2e}")
        for I in range(w.size):
            print(f"  comp {I}:  w={w[I]:.3f}  Ω={mu[I]:+.3f}  Δ={g[I]:.4f}")
    return w, mu, g, res


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Initial conditions
# ═══════════════════════════════════════════════════════════════════════════════

def assign_to_populations(omega_micro, w_pop, mu_pop, g_pop):
    M = len(w_pop)
    log_post = np.stack([
        np.log(w_pop[I] + 1e-300)
        + cauchy.logpdf(omega_micro, mu_pop[I], g_pop[I])
        for I in range(M)
    ], axis=0)
    return np.argmax(log_post, axis=0)


def fit_and_assign(omega_micro, theta0, delta_bounds, fit_cfg, seed):
    """Fit a weighted-Lorentzian mixture with LMMF (order M* chosen automatically, ensemble
    widths capped by `delta_bounds`), hard-assign each oscillator to its most-likely ensemble,
    and build the matched OA initial state. Returns (labels, w, Ω, Δ, r0, psi0, M*)."""
    w_pop, mu_pop, g_pop, _ = fit_lorentzian_mixture(omega_micro, delta_bounds, fit_cfg, seed=seed)
    M = w_pop.size
    labels = assign_to_populations(omega_micro, w_pop, mu_pop, g_pop)

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
    return labels, w_pop, mu_pop, g_pop, r0, psi0, M


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ODEs
# ═══════════════════════════════════════════════════════════════════════════════

# adaptation kernels G(φ) in Eq. 4 (φ = θ_j − θ_i). "cos" is the manuscript main-text rule;
# "sin" and "|sin|" are the App. B generalizations.
PLASTICITY_KERNELS = {
    "cos":   np.cos,
    "sin":   np.sin,
    "|sin|": lambda x: np.abs(np.sin(x)),
}


def _kernel(plasticity):
    try:
        return PLASTICITY_KERNELS[plasticity]
    except KeyError:
        raise ValueError(f"unknown plasticity rule {plasticity!r}; "
                         f"expected one of {list(PLASTICITY_KERNELS)}")


def km_ode(t, y, K, omega, mu, gamma, f):
    N     = len(omega)
    theta = y[:N]
    A     = y[N:].reshape(N, N)

    diff        = theta[np.newaxis, :] - theta[:, np.newaxis]
    interaction = np.sum(A * np.sin(diff), axis=1)
    dtheta      = omega + (K / N) * interaction

    dA = mu * f(diff) + gamma * (1.0 - A)      # manuscript Eq. 4: Ȧ_ij = μ G(θ_j−θ_i) + γ(1−A_ij)
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


def oa_ode(t, y, K, omega, delta, mu, gamma, f, w):
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
    dA = mu * rr * f(dpsi) + gamma * (1.0 - A)   # manuscript Eq. 4 (ensemble form): +γ(1−A_ml)
    return np.concatenate([dr, dpsi_, dA.ravel()])


def oa_order_parameter(r, w, psi):
    return np.abs(w @ (r * np.exp(1j * psi)))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Simulation runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_km(N, T, K, mu, gamma, omega_micro, theta0, A_km0, plasticity,
           method, rtol, atol):
    """Run the microscopic Kuramoto simulation once, from the given initial coupling matrix A_km0."""
    f     = _kernel(plasticity)
    y0_km = np.concatenate([theta0, A_km0.ravel()])
    print("Running KMO …")
    sol_km = solve_ivp(km_ode, (0, T), y0_km, method=method,
                       args=(K, omega_micro, mu, gamma, f),
                       rtol=rtol, atol=atol, dense_output=False)
    assert sol_km.success, sol_km.message
    print(f"  {sol_km.t.size} steps, {sol_km.nfev} evaluations")

    return dict(
        t_km=sol_km.t,
        theta=sol_km.y[:N],
        A_km=sol_km.y[N:].reshape(N, N, -1),
        R_km=km_order_parameter(sol_km.y[:N]),
    )


def run_oa(M, T, K, mu, gamma, w_pop, mu_pop, g_pop, r0, psi0, A0_mean,
           plasticity, method, rtol, atol):
    """Run the OA mean-field simulation for a given M. The ensemble-pair weights start at the
    MEAN initial micro weight A0_mean (the OA A_ml tracks the mean; the micro weight variance
    A0_std averages out over pairs and is not part of the mean-field IC)."""
    f     = _kernel(plasticity)
    A_oa0 = A0_mean * np.ones((M, M))
    y0_oa = np.concatenate([r0, psi0, A_oa0.ravel()])
    print(f"Running OA (M={M}) …")
    sol_oa = solve_ivp(oa_ode, (0, T), y0_oa, method=method,
                       args=(K, mu_pop, g_pop, mu, gamma, f, w_pop),
                       rtol=rtol, atol=atol, dense_output=False)
    assert sol_oa.success, sol_oa.message
    print(f"  {sol_oa.t.size} steps, {sol_oa.nfev} evaluations")

    return dict(
        t_oa=sol_oa.t,
        r_oa=sol_oa.y[:M],
        psi_oa=sol_oa.y[M:2*M],
        A_oa=sol_oa.y[2*M:].reshape(M, M, -1),
        R_oa=oa_order_parameter(sol_oa.y[:M], w_pop, sol_oa.y[M:2*M]),
    )


def simulate_grid(
    N=600, mu_list=(0.05, 0.1), vratio_max_list=(0.05, 0.2), T=80.0, K=2.0,
    gamma=0.01,
    plasticity="cos",
    gmm_params=None,
    R0=0.0, A0_mean=1.0, A0_std=0.0,
    delta_min=1e-4, M_max=30, alpha=1e-2, lambda_M=1e-5,
    loss="cvm", fit_method="slsqp", n_restarts=30, patience=3,
    seed=42,
    method="RK45", rtol=1e-7, atol=1e-9,
    export_params=True, export_stem="oa_params",
):
    """2-D sweep: adaptation rate μ (`mu_list`) × per-ensemble variance budget (`vratio_max_list`).
    For each μ the microscopic KMO network is simulated once (μ changes its adaptation dynamics);
    for each (μ, budget) cell the LMMF fit, OA mean-field sim and KMO coarse-graining are done.
    Note μ enters the FIT too — via `delta_max_for_vratio` (the budget→Δ_max cap depends on μ/γ).

    Each variance budget is a maximum allowed relative weight variance V_A/Ā² PER ENSEMBLE; it is
    translated (via :func:`delta_max_for_vratio`, the Eq. 37 cosine-rule closed form for the
    network's μ, γ, K) into an upper bound Δ_max on every ensemble width, i.e. LMMF's
    `delta_bounds` = (`delta_min`, Δ_max). A TIGHTER budget => smaller Δ_max => narrower
    ensembles, each with lower internal weight variance, and (Δ_min fixed => subset feasible
    region) weakly MORE ensembles, M*_tight >= M*_loose — provided each fixed-M fit is solved
    well (non-convex CvM: too few `n_restarts` can invert the ordering via a bad local minimum).
    The LMMF order-selection knobs (`M_max`, `lambda_M`) are FIXED/shared — the
    variance budget is what varies the decomposition. `alpha`, `loss`, `fit_method` (LMMF's
    constrained solver, distinct from the ODE `method`), `n_restarts`, `patience` are the
    remaining shared LMMF settings.

    Initial conditions: `R0` sets the initial phase coherence R(0)≈R0 (0 → incoherent/uniform
    phases, →1 → aligned); the microscopic coupling weights start Gaussian, A_ij(0) ~
    N(`A0_mean`, `A0_std`²) (A0_std=0 reproduces the uniform A_ij(0)=A0_mean start), while the
    OA mean-field starts from the mean A_ml(0)=A0_mean (its state is the ensemble-pair mean).

    When ``export_params`` is true, the fitted OA mean-field parameters and initial conditions
    are written to ``<export_stem>_M<M*>.npz`` (+ ``.txt``) for each fit, for use by the
    bifurcation-analysis pipeline (see :func:`export_meanfield_params` / :func:`load_meanfield_params`)."""
    rng = np.random.default_rng(seed)
    means, sigmas, weights = gmm_params
    fit_cfg = dict(M_max=M_max, alpha=alpha, lambda_M=lambda_M,
                   loss=loss, method=fit_method, n_restarts=n_restarts, patience=patience)

    # ── Sample microscopic frequencies, phases (coherence R0) and Gaussian weights (shared)
    omega_micro = sample_gaussian_mixture(N, means, sigmas, weights, seed)
    theta0      = phases_for_coherence(N, R0, rng)
    A_km0       = rng.normal(A0_mean, A0_std, (N, N))
    np.fill_diagonal(A_km0, A0_mean)          # self-weights = mean (diagonal is dynamically inert)
    print(f"IC: R(0)≈{np.abs(np.exp(1j*theta0).mean()):.3f} (target {R0:g}); "
          f"A_ij(0) ~ N({A0_mean:g}, {A0_std:g}²)")

    # ── Sweep μ (KMO once per μ), then the variance budgets within each μ ─────────
    per_mu = []
    for mu in mu_list:
        print(f"\n══════ μ = {mu:g}  (μ/γ = {mu / gamma:g}) ══════")
        km = run_km(N, T, K, mu, gamma, omega_micro, theta0, A_km0,
                    plasticity, method, rtol, atol)
        A_km_final = km["A_km"][:, :, -1]

        cells = []
        for vratio_max in vratio_max_list:
            d_max = delta_max_for_vratio(vratio_max, mu, gamma, K, plasticity, delta_min)
            print(f"  ── budget V_A/Ā² ≤ {vratio_max:g}  ->  Δ_max={d_max:.4f} ──")
            labels, w_pop, mu_pop, g_pop, r0, psi0, M = \
                fit_and_assign(omega_micro, theta0, (delta_min, d_max), fit_cfg, seed)
            pop_sizes = np.bincount(labels, minlength=M)
            print(f"  M*={M}  population sizes: min={pop_sizes.min()}, "
                  f"max={pop_sizes.max()}, mean={pop_sizes.mean():.1f}")

            oa_res = run_oa(M, T, K, mu, gamma, w_pop, mu_pop, g_pop, r0, psi0, A0_mean,
                            plasticity, method, rtol, atol)

            # Export the OA mean-field parameters + ICs for the bifurcation pipeline
            # (stem tagged by μ and M* so the 2×2 cells don't overwrite one another).
            if export_params:
                export_meanfield_params(
                    f"{export_stem}_mu{mu:g}_M{M}", M, w_pop, mu_pop, g_pop,
                    r0, psi0, A0_mean * np.ones((M, M)), K, mu, gamma, plasticity)

            # Coarse-grain the final KM weight matrix using THIS cell's labels
            A_km_cg = km_coarse_grain_labels(A_km_final, labels, M)

            cells.append(dict(
                mu=mu, vratio_max=vratio_max, M=M, delta_max=d_max,
                labels=labels, pop_sizes=pop_sizes,
                w_pop=w_pop, mu_pop=mu_pop, g_pop=g_pop, r0=r0, psi0=psi0,
                A_km_cg=A_km_cg, A_oa_final=oa_res["A_oa"][:, :, -1],
                t_oa=oa_res["t_oa"], R_oa=oa_res["R_oa"],
            ))

        per_mu.append(dict(mu=mu, t_km=km["t_km"], R_km=km["R_km"], cells=cells))

    return dict(
        omega_micro=omega_micro, theta0=theta0,
        N=N, T=T, K=K, gamma=gamma, plasticity=plasticity, gmm_params=gmm_params,
        mu_list=list(mu_list), vratio_max_list=list(vratio_max_list),
        per_mu=per_mu,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. Export / load the OA mean-field network parameters + initial conditions
#     (handoff to the bifurcation-analysis pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def export_meanfield_params(stem, M, weights, omega, delta, r0, psi0, A0,
                            K, mu, gamma, plasticity):
    r"""Write the fitted OA mean-field model's parameters + initial conditions.

    The mean-field model (see :func:`oa_ode`) for the M weighted-Lorentzian
    ensembles fitted to the microscopic frequency sample is

        dr_i  = -δ_i r_i + ½(1-r_i²) K Σ_j w_j A_ij r_j cos(ψ_j-ψ_i)
        dψ_i  =  ω_i + ½(1+r_i²)/r_i K Σ_j w_j A_ij r_j sin(ψ_j-ψ_i)
        dA_ij =  μ r_i r_j f(ψ_j-ψ_i) + γ (1 - A_ij),

    with mixture weights w_j (Σ_j w_j = 1), centre frequencies ω_i, widths (HWHM) δ_i,
    and plasticity kernel f = G ∈ {cos, sin, |sin|} (manuscript Eq. 4; cos is the main text).
    Note this differs from ``kmo_macro_simulation`` in TWO ways the bifurcation
    model must reproduce: (i) the coupling is WEIGHTED by w_j (there is no 1/M
    normalisation), and (ii) the global/observable order parameter is the
    weighted sum R = |Σ_i w_i r_i e^{iψ_i}|.

    Writes ``<stem>.npz`` (machine-readable, for the bifurcation script via
    :func:`load_meanfield_params`) and ``<stem>.txt`` (human-readable summary).
    """
    weights = np.asarray(weights, float)
    omega = np.asarray(omega, float)
    delta = np.asarray(delta, float)
    r0 = np.asarray(r0, float)
    psi0 = np.asarray(psi0, float)
    A0 = np.asarray(A0, float)

    npz_path = f"{stem}.npz"
    np.savez(
        npz_path,
        M=np.int64(M),
        weights=weights, omega=omega, delta=delta,      # network parameters
        K=np.float64(K), mu=np.float64(mu), gamma=np.float64(gamma),
        plasticity=str(plasticity),
        r0=r0, psi0=psi0, A0=A0,                         # initial conditions
    )

    with open(f"{stem}.txt", "w") as fh:
        fh.write(f"# OA mean-field network parameters + initial conditions (M={M})\n")
        fh.write(f"# plasticity kernel G = {plasticity}   (Ȧ_ij = μ G(ψ_j−ψ_i) + γ(1−A_ij))\n")
        fh.write(f"# global params:  K={K}  mu={mu}  gamma={gamma}\n")
        fh.write(f"# global order parameter:  R = |sum_i w_i r_i exp(i psi_i)|\n")
        fh.write("#\n# i      w_i           omega_i        delta_i        "
                 "r0_i          psi0_i\n")
        for I in range(M):
            fh.write(f"{I:<4d} {weights[I]: .8e} {omega[I]: .8e} {delta[I]: .8e} "
                     f"{r0[I]: .8e} {psi0[I]: .8e}\n")
        fh.write("#\n# A0 (initial coupling matrix, row-major):\n")
        for I in range(M):
            fh.write("# " + " ".join(f"{A0[I, J]: .4e}" for J in range(M)) + "\n")

    print(f"  exported mean-field params → {npz_path}  (+ {stem}.txt)")
    return npz_path


def load_meanfield_params(npz_path):
    """Load parameters written by :func:`export_meanfield_params` into a dict
    (for the bifurcation-analysis script)."""
    d = np.load(npz_path, allow_pickle=False)
    return dict(
        M=int(d["M"]),
        weights=d["weights"], omega=d["omega"], delta=d["delta"],
        K=float(d["K"]), mu=float(d["mu"]), gamma=float(d["gamma"]),
        plasticity=str(d["plasticity"]),
        r0=d["r0"], psi0=d["psi0"], A0=d["A0"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PRX-style publication figure (2×2 grid: variance-budget rows × μ columns)
# ═══════════════════════════════════════════════════════════════════════════════

def set_prx_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["STIXGeneral", "Times New Roman", "Times"],
        "mathtext.fontset":   "stix",
        "font.size":          12,
        "axes.titlesize":     12,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "axes.linewidth":     0.7,
        "lines.linewidth":    1.2,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "savefig.dpi":        300,
        "figure.dpi":         150,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


C_KM   = "#1f4e79"
C_OA   = "#c44e52"
C_GMM  = "#4c4c4c"
C_FIT  = "#c44e52"
CMAP_A = "RdBu_r"


def make_panel_label(ax, label, *, x=-0.18, y=1.02, fontsize=12):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            ha="left", va="bottom")


def plot_figure(res, savepath="figure1.pdf", twindow=None):
    """2×2 (variance-budget rows × μ columns) figure. Each cell shows the micro-vs-OA R(t),
    the fitted frequency mixture, and the coarse-grained KMO / OA coupling matrices. The two
    matrices in a cell share a PER-CELL symmetric colour scale + colorbar (so weight magnitudes
    are read within each scenario, not across the whole grid).

    `twindow` = (t0, t1) sets the plotted R(t) time window (view only — the full traces are
    computed regardless); None shows the whole run [0, T]."""
    set_prx_style()

    t0, t1  = twindow if twindow is not None else (0.0, res["T"])
    N       = res["N"]
    per_mu  = res["per_mu"]
    mus     = res["mu_list"]
    budgets = res["vratio_max_list"]
    gamma   = res["gamma"]
    nmu, nb = len(mus), len(budgets)
    omega   = res["omega_micro"]
    means, sigmas, weights = res["gmm_params"]

    hi, lo = np.percentile(omega, [99.5, 0.5])
    edges  = np.linspace(lo, hi, 60)
    x_grid = np.linspace(lo, hi, 600)
    pdf_g  = gaussian_mixture_pdf(x_grid, means, sigmas, weights)

    # each budget row = 2 sub-rows (R(t) then panels); each μ column = 3 sub-cols (dist|KMO|OA)
    hr = []
    for _ in range(nb):
        hr += [0.72, 1.05]
    fig = plt.figure(figsize=(6.4 * nmu, 4.6 * nb), constrained_layout=True)
    gs = gridspec.GridSpec(2 * nb, 3 * nmu, figure=fig, height_ratios=hr,
                           width_ratios=[1.15, 1.0, 1.0] * nmu)

    letters = iter("abcdefghijklmnopqrstuvwxyz")

    for b, budget in enumerate(budgets):
        for m, mu in enumerate(mus):
            pm, cell = per_mu[m], per_mu[m]["cells"][b]
            M_val = cell["M"]
            c0 = 3 * m

            # ── R(t): micro vs OA mean field ────────────────────────────────
            axR = fig.add_subplot(gs[2 * b, c0:c0 + 3])
            axR.plot(pm["t_km"], pm["R_km"], color=C_KM, lw=1.3, label=fr"KMO ($N={N}$)")
            axR.plot(cell["t_oa"], cell["R_oa"], color=C_OA, lw=1.3, ls="--",
                     label=fr"OA ($M^*={M_val}$)")
            axR.set_xlim(t0, t1); axR.set_ylim(0, 1.02)
            axR.set_xlabel(r"time $t$")
            if m == 0:
                axR.set_ylabel(r"$R(t)$")
            axR.set_title(rf"$\mu={mu:g}$ ($\mu/\gamma={mu / gamma:g}$),  "
                          rf"$V_A/\bar A^2\!\leq\!{budget:g}$", fontsize=10.5, pad=3)
            axR.legend(loc="best", fontsize=8, frameon=False, handlelength=1.8)
            make_panel_label(axR, f"({next(letters)})", x=-0.05, y=1.05)

            # ── fitted frequency distribution ───────────────────────────────
            axf = fig.add_subplot(gs[2 * b + 1, c0])
            axf.hist(omega, bins=edges, density=True, color=C_KM, alpha=0.30, edgecolor="none")
            axf.plot(x_grid, pdf_g, color=C_GMM, lw=1.1, label="Gauss.")
            pdf_c = cauchy_mixture_pdf(
                x_grid, np.concatenate([cell["w_pop"], cell["mu_pop"], cell["g_pop"]]))
            axf.plot(x_grid, pdf_c, color=C_FIT, lw=1.2, ls="--", label=fr"LMMF $M^*={M_val}$")
            if M_val <= 12:
                for I in range(M_val):
                    axf.plot(x_grid, cell["w_pop"][I] * cauchy.pdf(x_grid, cell["mu_pop"][I],
                             cell["g_pop"][I]), color=C_FIT, lw=0.5, alpha=0.5)
            axf.set_xlabel(r"$\omega$")
            if m == 0:
                axf.set_ylabel(r"$p(\omega)$")
            axf.set_xlim(lo, hi); axf.set_ylim(0, None)
            axf.legend(loc="upper right", fontsize=7, frameon=False, handlelength=1.5)
            make_panel_label(axf, f"({next(letters)})", x=-0.30, y=1.06)

            # ── coarse-grained KMO & OA matrices (PER-CELL symmetric colour scale) ──
            vmax_c = max(np.abs(cell["A_km_cg"]).max(), np.abs(cell["A_oa_final"]).max()) or 1.0
            norm_kw = dict(cmap=CMAP_A, vmin=-vmax_c, vmax=vmax_c,
                           interpolation="nearest", aspect="equal")
            axK = fig.add_subplot(gs[2 * b + 1, c0 + 1])
            axK.imshow(cell["A_km_cg"], **norm_kw)
            axK.set_title(r"$A^{\mathrm{KO}}_{ml}$", fontsize=9, pad=2)
            axK.set_xlabel(r"$m$"); axK.set_ylabel(r"$l$")
            _set_matrix_ticks(axK, M_val)
            make_panel_label(axK, f"({next(letters)})", x=-0.32, y=1.08)

            axO = fig.add_subplot(gs[2 * b + 1, c0 + 2])
            imO = axO.imshow(cell["A_oa_final"], **norm_kw)
            axO.set_title(r"$A^{\mathrm{OA}}_{ml}$", fontsize=9, pad=2)
            axO.set_xlabel(r"$l$")
            _set_matrix_ticks(axO, M_val); axO.tick_params(labelleft=False)
            make_panel_label(axO, f"({next(letters)})", x=-0.14, y=1.08)

            # per-cell colorbar (shared by THIS cell's KMO & OA matrices only)
            cb = fig.colorbar(imO, ax=[axK, axO], location="right",
                              shrink=0.85, pad=0.012, fraction=0.06)
            cb.ax.tick_params(labelsize=7)

    fig.savefig(savepath)
    print(f"\nFigure saved → {savepath}")
    return fig


def _set_matrix_ticks(ax, M):
    """Adapt tick density to matrix size for readability."""
    if M <= 10:
        ax.set_xticks(range(M)); ax.set_yticks(range(M))
    else:
        # Show roughly 6 ticks
        step = max(1, M // 6)
        ticks = list(range(0, M, step))
        if ticks[-1] != M - 1:
            ticks.append(M - 1)
        ax.set_xticks(ticks); ax.set_yticks(ticks)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Multi-modal frequency distribution: 3 Gaussian components.
    # These are the means / sigmas / weights reconstructed from Fig. 1b of the
    # PRL manuscript (git commit d22ff83). The published figure additionally used
    # K=2.5, mu=0.02, gamma=0.001, plasticity="sin", N=500, seed=42 (and hand-picked
    # M values) — set those in CONFIG below to reproduce it.
    GMM_PARAMS = (
        # means         sigmas       weights
        [-2.0, 2.0],
        [ 1.0,  1.0],
        [ 0.5,  0.5],
    )

    CONFIG = dict(
        N          = 500,
        # per-ensemble weight-variance budgets to compare: each caps V_A/Ā² per ensemble and is
        # translated into an upper bound on the ensemble width Δ (via the Eq. 37 cosine-rule form
        # for this μ, γ, K). A TIGHTER budget => narrower ensembles (lower per-ensemble variance);
        # LMMF adapts M* to keep the fit adequate.
        vratio_max_list = [0.06, 0.006],   # loose vs. tight per-ensemble variance budget
        # initial conditions
        R0       = 0.9,     # initial phase coherence R(0) (0 = incoherent/uniform, →1 = aligned)
        A0_mean  = 1.0,     # mean initial coupling weight  A_ij(0) ~ N(A0_mean, A0_std²)
        A0_std   = 0.0,     # std  of initial coupling weight (0 = uniform A_ij(0)=A0_mean)
        # shared LMMF settings (same algorithm as grid_search/kmo_lorentzian_fit_sweep.py);
        # M_max / lambda_M are now FIXED — the variance budget is what varies the decomposition.
        delta_min    = 1e-4,
        M_max        = 20,         # generous cap; the Δ budget sets the granularity
        alpha        = 1e-2,       # CvM GoF acceptance level on (1−p)
        lambda_M     = 2e-4,     # tuned for the 2-Gaussian mix: loose budget -> M*=2, tight -> M*=6
        loss         = "cvm",
        fit_method   = "slsqp",    # LMMF constrained solver (NOT the ODE method below)
        n_restarts   = 30,
        patience     = 4,
        T          = 400.0,
        K          = 3.0,
        mu_list    = [0.005, 0.1],   # two adaptation rates to sweep (columns of the 2×2 figure)
        gamma      = 0.005,
        plasticity = "cos",     # G ∈ {"cos" (main text), "sin", "|sin|"} — Eq. 4
        gmm_params = GMM_PARAMS,
        seed       = 42,
        method     = "DOP853",     # ODE solver for the KMO/OA integration
        rtol       = 1e-6,
        atol       = 1e-8,
    )

    # plotting-only: time window (t0, t1) shown in the R(t) panels (None = whole run [0, T])
    PLOT_TWINDOW = (300.0, 400.0)

    res = simulate_grid(**CONFIG)
    fig = plot_figure(res, savepath="figure1_kmo_vs_oa_grid.pdf", twindow=PLOT_TWINDOW)
    fig.savefig("figure1_kmo_vs_oa_grid.png", dpi=200)
    print("PNG preview → figure1_kmo_vs_oa_grid.png")
    plt.show()
