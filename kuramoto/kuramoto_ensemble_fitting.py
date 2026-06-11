"""
Figure 1: Ensemble Mean-Field vs. Kuramoto Network with Multi-Modal Frequencies
================================================================================
Compares the microscopic Kuramoto network to the OA mean-field for TWO values
of M, the number of Cauchy components / ensembles, in a single PRX-style figure.

Workflow
--------
    1. Sample N microscopic frequencies from a Gaussian mixture (done once).
    2. Run the KMO simulation once (independent of M).
    3. For each M in {M1, M2}:
        a. Fit a sum of M Cauchy distributions to the microscopic sample.
        b. Hard-assign each oscillator to its most-likely Cauchy component.
        c. Build matched OA initial conditions and simulate the OA mean-field.
        d. Coarse-grain the KMO weight matrix using THIS M's labels.
    4. Plot a 4-row PRX-style figure:
        Rows 1–2:  M = M1   (R(t) panel + distribution + 2 matrix panels)
        Rows 3–4:  M = M2   (same layout)

PRX figure constraints
----------------------
    - Two-column width: 7.0 inches (~17.8 cm).
    - Serif font (Computer Modern / STIX) for consistency with REVTeX.
    - 8 pt minimum for labels; 9 pt body, 10 pt panel labels.
    - Vector PDF output.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import cauchy, norm
from sklearn.mixture import GaussianMixture

_EPS = 1e-12


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


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Fit a sum of M Cauchy distributions
# ═══════════════════════════════════════════════════════════════════════════════

def cauchy_mixture_pdf(x, params):
    M = len(params) // 3
    w  = np.asarray(params[0:M])
    mu = np.asarray(params[M:2*M])
    g  = np.asarray(params[2*M:3*M])
    w = np.abs(w); w = w / w.sum()
    g = np.abs(g) + 1e-6
    return np.sum(
        [wi * cauchy.pdf(x, m, gi) for wi, m, gi in zip(w, mu, g)], axis=0,
    )


def neg_log_likelihood(params, samples):
    pdf = cauchy_mixture_pdf(samples, params)
    return -np.sum(np.log(pdf + 1e-300))


def fit_cauchy_mixture(samples, M, seed=0, verbose=True):
    gmm = GaussianMixture(n_components=M, random_state=seed, n_init=4)
    gmm.fit(samples.reshape(-1, 1))
    w0  = gmm.weights_
    mu0 = gmm.means_.ravel()
    g0  = np.sqrt(gmm.covariances_.ravel())

    order = np.argsort(mu0)
    w0, mu0, g0 = w0[order], mu0[order], g0[order]
    w0 = w0 / w0.sum()

    bounds = [(0.0, 1.0)] * M + [(-np.inf, np.inf)] * M + [(1e-3, np.inf)] * M
    x0 = np.concatenate([w0, mu0, g0])
    res = minimize(neg_log_likelihood, x0, args=(samples,),
                   method="Nelder-Mead", bounds=bounds,
                   options=dict(maxiter=20000, xatol=1e-8, fatol=1e-8))

    w  = np.abs(res.x[0:M]);   w = w / w.sum()
    mu = res.x[M:2*M]
    g  = np.abs(res.x[2*M:3*M]) + 1e-6
    order = np.argsort(mu)
    w, mu, g = w[order], mu[order], g[order]

    if verbose:
        print(f"Cauchy mixture fit (M={M}):  NLL={res.fun:.4f}")
        for I in range(M):
            print(f"  comp {I}:  w={w[I]:.3f}  μ={mu[I]:+.3f}  γ={g[I]:.3f}")
    return w, mu, g


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


def fit_and_assign(omega_micro, theta0, M, seed):
    """Fit M Cauchy components and build the matched OA initial state."""
    w_pop, mu_pop, g_pop = fit_cauchy_mixture(omega_micro, M, seed=seed)
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
    return labels, w_pop, mu_pop, g_pop, r0, psi0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ODEs
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
    dA = mu * rr * f(dpsi) - gamma * A
    return np.concatenate([dr, dpsi_, dA.ravel()])


def oa_order_parameter(r, w, psi):
    return np.abs(w @ (r * np.exp(1j * psi)))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Simulation runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_km(N, T, K, mu, gamma, omega_micro, theta0, plasticity,
           method, rtol, atol):
    """Run the microscopic Kuramoto simulation once."""
    f     = hebbian if plasticity == "hebbian" else antihebbian
    A_km0 = np.ones((N, N))
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


def run_oa(M, T, K, mu, gamma, w_pop, mu_pop, g_pop, r0, psi0,
           plasticity, method, rtol, atol):
    """Run the OA mean-field simulation for a given M."""
    f     = hebbian if plasticity == "hebbian" else antihebbian
    A_oa0 = np.ones((M, M))
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


def simulate_two_M(
    N=600, M_list=(4, 50), T=80.0, K=2.0,
    mu=0.1, gamma=0.0,
    plasticity="antihebbian",
    gmm_params=None,
    seed=42,
    method="RK45", rtol=1e-7, atol=1e-9,
    export_params=True, export_stem="oa_params",
):
    """Run one KMO simulation and TWO OA simulations (one per M in M_list).

    When ``export_params`` is true, the fitted OA mean-field parameters and
    initial conditions are written to ``<export_stem>_M<M>.npz`` (+ ``.txt``) for
    each M, for use by the bifurcation-analysis pipeline (see
    :func:`export_meanfield_params` / :func:`load_meanfield_params`)."""
    rng = np.random.default_rng(seed)
    means, sigmas, weights = gmm_params

    # ── Sample microscopic frequencies + phases (shared across both M values)
    omega_micro = sample_gaussian_mixture(N, means, sigmas, weights, seed)
    theta0      = rng.uniform(-np.pi, np.pi, N)

    # ── Run the (expensive) KMO simulation once ──────────────────────────────
    km_res = run_km(N, T, K, mu, gamma, omega_micro, theta0,
                    plasticity, method, rtol, atol)

    # ── For each M: fit, assign, run OA, coarse-grain ────────────────────────
    per_M_results = []
    for M in M_list:
        print(f"\n── M = {M} ──")
        labels, w_pop, mu_pop, g_pop, r0, psi0 = \
            fit_and_assign(omega_micro, theta0, M, seed)
        pop_sizes = np.bincount(labels, minlength=M)
        print(f"Population sizes: min={pop_sizes.min()}, "
              f"max={pop_sizes.max()}, mean={pop_sizes.mean():.1f}")

        oa_res = run_oa(M, T, K, mu, gamma, w_pop, mu_pop, g_pop, r0, psi0,
                        plasticity, method, rtol, atol)

        # Export the OA mean-field parameters + initial conditions for the
        # bifurcation pipeline. A0 = ones((M, M)) is the IVP start used by run_oa
        # (the bifurcation script settles to the steady state from here).
        if export_params:
            export_meanfield_params(
                f"{export_stem}_M{M}", M, w_pop, mu_pop, g_pop,
                r0, psi0, np.ones((M, M)), K, mu, gamma, plasticity)

        # Coarse-grain the final KM weight matrix using THIS M's labels
        A_km_cg = km_coarse_grain_labels(km_res["A_km"][:, :, -1], labels, M)

        per_M_results.append(dict(
            M=M, labels=labels, pop_sizes=pop_sizes,
            w_pop=w_pop, mu_pop=mu_pop, g_pop=g_pop,
            r0=r0, psi0=psi0,
            A_km_cg=A_km_cg,
            **oa_res,
        ))

    return dict(
        omega_micro=omega_micro, theta0=theta0,
        N=N, T=T, K=K, mu=mu, gamma=gamma,
        plasticity=plasticity, gmm_params=gmm_params,
        **km_res,
        per_M=per_M_results,
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
        dA_ij =  μ r_i r_j f(ψ_j-ψ_i) - γ A_ij,

    with mixture weights w_j (Σ_j w_j = 1), centre frequencies ω_i, widths
    (HWHM) δ_i, and plasticity kernel f = cos (Hebbian) or sin (anti-Hebbian).
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
        fh.write(f"# plasticity = {plasticity}   (f = "
                 f"{'cos' if plasticity == 'hebbian' else 'sin'})\n")
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
# 6. PRX-style publication figure (4 rows)
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


def plot_figure(res, savepath="figure1.pdf"):
    set_prx_style()

    N         = res["N"]
    per_M     = res["per_M"]
    n_M       = len(per_M)
    assert n_M == 2, "This figure layout expects exactly two M values."

    # ── Shared colour scale across BOTH M panels and BOTH matrix types ──────
    # Using one scale lets the reader compare weight magnitudes between M
    # values directly.
    vmax = 0.0
    for d in per_M:
        vmax = max(vmax, np.abs(d["A_km_cg"]).max(),
                   np.abs(d["A_oa"][:, :, -1]).max())
    vmax = vmax or 1.0
    norm_kw = dict(cmap=CMAP_A, vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="equal")

    # ── Figure dimensions: two-column width, 4 rows ─────────────────────────
    # Each (M block) occupies 2 rows: top for R(t), bottom for the panels.
    # Original height_ratios totaled 4.1 over height 9.4". Shrinking rows 0 and 2
    # to 2/3 of their original height saves (2 - 4/3)/4.1 of the figure height,
    # keeping rows 1 and 3 (and all font sizes) at their original physical size.
    _orig_h = 9.4
    _new_h = _orig_h * (4/3 + 1.05 + 1.05) / (1.0 + 1.05 + 1.0 + 1.05)
    fig = plt.figure(figsize=(7.0, _new_h))

    gs = gridspec.GridSpec(
        nrows=4, ncols=3, figure=fig,
        height_ratios=[2/3, 1.05, 2/3, 1.05],
        width_ratios =[1.05, 1.0, 1.0],
        hspace=0.55, wspace=0.40,
        left=0.085, right=0.94,
        # Preserve top/bottom margins in absolute inches under the shorter figure
        top    = 1 - (1 - 0.965) * _orig_h / _new_h,
        bottom = 0.055 * _orig_h / _new_h,
    )

    panel_letters = iter("abcdefghij")

    for block, data in enumerate(per_M):
        M_val      = data["M"]
        row_top    = 2 * block      # R(t) row
        row_bottom = 2 * block + 1  # distribution + matrices row

        # ── R(t) panel (spans all 3 columns) ────────────────────────────────
        ax_R = fig.add_subplot(gs[row_top, :])
        ax_R.plot(res["t_km"], res["R_km"], color=C_KM, lw=1.4,
                  label=fr"KMO  ($N={N}$)")
        ax_R.plot(data["t_oa"], data["R_oa"], color=C_OA, lw=1.4, ls="--",
                  label=fr"OA mean-field ($M={M_val}$)")
        ax_R.set_xlim(0, res["T"])
        ax_R.set_ylim(0, 1.02)
        ax_R.set_xlabel(r"time $t$")
        ax_R.set_ylabel(r"$R(t)$")
        ax_R.legend(loc="lower right", frameon=False, handlelength=2.2,
                    borderaxespad=0.3)
        make_panel_label(ax_R, f"({next(panel_letters)})", x=-0.07, y=1.02)

        # ── Frequency distribution panel ─────────────────────────────────────
        ax_f = fig.add_subplot(gs[row_bottom, 0])
        omega = res["omega_micro"]
        w_pop, mu_pop, g_pop = data["w_pop"], data["mu_pop"], data["g_pop"]
        means, sigmas, weights = res["gmm_params"]

        hi, lo = np.percentile(omega, [99.5, 0.5])
        edges  = np.linspace(lo, hi, 60)
        ax_f.hist(omega, bins=edges, density=True, color=C_KM,
                  alpha=0.30, edgecolor="none", label="KMO sample")

        x_grid = np.linspace(lo, hi, 600)
        pdf_g  = gaussian_mixture_pdf(x_grid, means, sigmas, weights)
        ax_f.plot(x_grid, pdf_g, color=C_GMM, lw=1.2, ls="-",
                  label="Gaussian mix")

        params_flat = np.concatenate([w_pop, mu_pop, g_pop])
        pdf_c = cauchy_mixture_pdf(x_grid, params_flat)
        ax_f.plot(x_grid, pdf_c, color=C_FIT, lw=1.2, ls="--",
                  label=f"Cauchy fit (M={M_val})")

        # Individual Cauchy components (only show if M is small enough
        # that they're visually distinguishable)
        if M_val <= 10:
            for I in range(M_val):
                comp = w_pop[I] * cauchy.pdf(x_grid, mu_pop[I], g_pop[I])
                ax_f.plot(x_grid, comp, color=C_FIT, lw=0.6, alpha=0.5)

        ax_f.set_xlabel(r"$\omega$")
        ax_f.set_ylabel(r"$p(\omega)$")
        ax_f.legend(loc="upper right", frameon=False, handlelength=2.0,
                    borderaxespad=0.3)
        ax_f.set_xlim(lo, hi)
        ax_f.set_ylim(0.0, 1.0)
        make_panel_label(ax_f, f"({next(panel_letters)})", x=-0.22, y=1.02)

        # ── KMO coarse-grained weight matrix ─────────────────────────────────
        ax_AK = fig.add_subplot(gs[row_bottom, 1])
        ax_AK.imshow(data["A_km_cg"], **norm_kw)
        ax_AK.set_title(rf"${{A}}^{{\mathrm{{KO}}}}_{{ml}}$  "
                        rf"($M={M_val}$)", pad=3)
        ax_AK.set_xlabel(r"ensemble $m$")
        ax_AK.set_ylabel(r"ensemble $l$")
        # Adapt tick density to M
        _set_matrix_ticks(ax_AK, M_val)
        make_panel_label(ax_AK, f"({next(panel_letters)})", x=-0.22, y=1.02)

        # ── OA weight matrix ─────────────────────────────────────────────────
        ax_AO = fig.add_subplot(gs[row_bottom, 2])
        im_AO = ax_AO.imshow(data["A_oa"][:, :, -1], **norm_kw)
        ax_AO.set_title(rf"$A^{{\mathrm{{OA}}}}_{{ml}}$  ($M={M_val}$)", pad=3)
        ax_AO.set_xlabel(r"ensemble $l$")
        _set_matrix_ticks(ax_AO, M_val)
        ax_AO.tick_params(labelleft=False)
        make_panel_label(ax_AO, f"({next(panel_letters)})", x=-0.10, y=1.02)

        # ── Per-row colorbar ────────────────────────────────────────────────
        pos_AO = ax_AO.get_position()
        cax = fig.add_axes([pos_AO.x1 + 0.012, pos_AO.y0,
                            0.012, pos_AO.height])
        cb = fig.colorbar(im_AO, cax=cax)
        cb.ax.tick_params(labelsize=8, width=0.5, length=2.5)
        cb.outline.set_linewidth(0.5)
        cb.set_label(r"weight", fontsize=9, labelpad=2)

    fig.savefig(savepath, bbox_inches="tight")
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
    # K=2.5, mu=0.02, gamma=0.001, plasticity="antihebbian", M_list=(5, 25),
    # N=500, seed=42 — set those in CONFIG below to reproduce it exactly.
    GMM_PARAMS = (
        # means         sigmas       weights
        [-0.2, -0.05, 0.4],
        [ 0.2,  0.5,  0.4],
        [ 0.5,  1.0,  0.6],
    )

    CONFIG = dict(
        N          = 400,
        M_list     = (5, 20),     # two values of M to compare
        T          = 500.0,
        K          = 1.0,
        mu         = 0.05,
        gamma      = 0.01,
        plasticity = "hebbian",
        gmm_params = GMM_PARAMS,
        seed       = 42,
        method     = "DOP853",
        rtol       = 1e-6,
        atol       = 1e-8,
    )

    res = simulate_two_M(**CONFIG)
    fig = plot_figure(res, savepath="figure1_kmo_vs_oa_twoM.pdf")
    fig.savefig("figure1_kmo_vs_oa_twoM.png", dpi=300, bbox_inches="tight")
    print("PNG preview → figure1_kmo_vs_oa_twoM.png")
    plt.show()
