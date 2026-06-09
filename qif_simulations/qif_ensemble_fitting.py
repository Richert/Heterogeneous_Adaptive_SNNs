"""
QIF Network with Gaussian-Mixture Excitability:
Microscopic vs. MPR Mean-Field Comparison Across Plasticity Rules
=================================================================

QIF analog of `theta_ensemble_fitting.py`. The QIF microscopic dynamics
match PRL 2026 (Gast, Schmidt, Takasu, Benna, Kennedy) eq. 25-26, with an
alpha-kernel synapse and trace-based STDP plasticity (eq. 30-32).

  Microscopic (per neuron i):
      dV_i / dt = V_i^2 + eta_i + (J/N) sum_j A_ij s_j           (eq. 25)
      tau_s ds_i / dt = B_i                                       (alpha kernel
      tau_s dB_i / dt = -2 B_i - s_i + alpha sum_k delta(t-t_i^k)  see lab report)
      spike when V_i >= V_peak: V_i -> -V_peak, B_i -> B_i + alpha/tau_s
      tau_x dx_i / dt = -x_i + s_i             (pre-trace, eq. 30)
      tau_y dy_i / dt = -y_i + s_i             (post-trace, eq. 31)

  MPR mean-field (per ensemble m):
      dr_m / dt    = Delta_m / pi + 2 r_m v_m
      dv_m / dt    = v_m^2 + eta_bar_m - (pi r_m)^2 + J sum_n A_mn w_n s_n
      tau_s ds_m   = B_m
      tau_s dB_m   = -2 B_m - s_m + alpha r_m
      tau_x dx_m   = -x_m + s_m
      tau_y dy_m   = -y_m + s_m
    (alpha = 1, so each unit of mean firing rate adds 1 to int s_m dt.)

  Plasticity rules (PRL eq. 32 and the user's anti-Hebbian/oja modifications):
      Hebbian:     dA_ij = a_+(A_ij) x_j s_i  -  a_-(A_ij) y_i s_j
      anti-Hebbian: dA_ij = a_+(A_ij) s_j^2    -  a_-(A_ij) y_j s_i
      oja:         dA_ij = a_+(A_ij) s_i s_j  -  a_-(A_ij) s_i^2
      with bounded a_+(A) = mu(1-A), a_-(A) = mu A to keep A in [0, 1].

  At the ensemble level, replace (i, j) with (m, n) and use the ensemble-level
  s_m, x_m, y_m. Both micro and MF have hard clipping A in [0, 1] as a safety net.

Figure layout (matches theta_three_rules.svg):
    2 rows x 9 gridspec columns:
        row 0:  cols 0-1 -- eta_i distribution + Cauchy fit
                cols 3-4 -- STDP kernel Delta A(t_post - t_pre)
                cols 6-7 -- mean synaptic activation <s>(t) for Oja
        row 1:  per-rule [QIF block-avg A | MPR-OA A | colorbar] blocks
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import cauchy


# ═══════════════════════════════════════════════════════════════════════════════
# Style and color helpers (lifted from kuramoto_ensemble_fitting style)
# ═══════════════════════════════════════════════════════════════════════════════
C_QIF = "#1f4e79"     # microscopic QIF (was C_KM)
C_MF  = "#c44e52"     # MPR mean-field (was C_OA)
C_GMM = "#3a8e7c"     # Gaussian-mixture ground truth
C_FIT = "#c44e52"     # Cauchy-mixture fit

PLASTICITY_RULES = ("Hebbian", "anti-Hebbian", "Symmetric")
_EPS = 1e-9


def set_prx_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    })


def make_panel_label(ax, label, x=-0.10, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# ═══════════════════════════════════════════════════════════════════════════════
# Excitability distribution helpers
# ═══════════════════════════════════════════════════════════════════════════════
def sample_gaussian_mixture(N, means, sigmas, weights, seed=42):
    """Sample N values from a Gaussian mixture model."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    comp = rng.choice(len(means), size=N, p=weights)
    return np.array([rng.normal(means[c], sigmas[c]) for c in comp])


def gaussian_mixture_pdf(x, means, sigmas, weights):
    """Density of a Gaussian mixture at points x."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    pdf = np.zeros_like(x, dtype=float)
    for mu, sg, w in zip(means, sigmas, weights):
        pdf += w * np.exp(-0.5 * ((x - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
    return pdf


def cauchy_mixture_pdf(x, params_flat):
    """
    Density of a Cauchy mixture.
    params_flat is concatenated [weights (M), centers (M), widths (M)].
    """
    M = len(params_flat) // 3
    w = params_flat[:M]
    c = params_flat[M:2 * M]
    d = params_flat[2 * M:]
    pdf = np.zeros_like(x, dtype=float)
    for wi, ci, di in zip(w, c, d):
        pdf += wi * (di / np.pi) / ((x - ci) ** 2 + di ** 2)
    return pdf


def fit_cauchy_mixture(samples, M, seed=42, verbose=False,
                        em_iters=200, nm_polish=True):
    """
    Fit M-component Cauchy mixture to samples using EM for the weights
    (with fixed centers/widths each step, then update centers and widths),
    optionally polishing the result with Nelder-Mead on neg-log-likelihood.

    The EM iteration for a Cauchy mixture:
        E-step: gamma_{im} = w_m p(x_i | c_m, d_m) / sum_n w_n p(x_i | c_n, d_n)
        M-step:
          w_m   <- (1/N) sum_i gamma_{im}
          c_m, d_m <- weighted Cauchy-MLE for component m
                      (no closed form; do a single Newton step on the
                       weighted -log-lik for (c_m, d_m))

    Returns (weights, centers, widths) each of shape (M,).
    """
    rng = np.random.default_rng(seed)
    N = len(samples)
    samples = np.asarray(samples)

    # Init: spread centers over quantiles
    sorted_s = np.sort(samples)
    centers = np.quantile(sorted_s, (np.arange(M) + 0.5) / M)
    widths = np.full(M, max(np.std(samples) / max(1, np.sqrt(M)), 0.02))
    weights = np.full(M, 1.0 / M)

    def cauchy_pdf(x, c, d):
        return (d / np.pi) / ((x - c) ** 2 + d ** 2)

    def neg_log_lik_full(w, c, d):
        # (N, M)
        diff = samples[:, None] - c[None, :]
        comp = (d[None, :] / np.pi) / (diff ** 2 + d[None, :] ** 2)
        density = (comp * w[None, :]).sum(axis=1)
        density = np.clip(density, 1e-300, None)
        return -np.log(density).sum()

    prev_nll = neg_log_lik_full(weights, centers, widths)
    for it in range(em_iters):
        # E-step
        diff = samples[:, None] - centers[None, :]
        comp = (widths[None, :] / np.pi) / (diff ** 2 + widths[None, :] ** 2)
        gamma = weights[None, :] * comp                       # (N, M)
        Z = gamma.sum(axis=1, keepdims=True) + 1e-300
        gamma = gamma / Z

        # M-step: weights (closed-form)
        Nk = gamma.sum(axis=0)                                # (M,)
        weights = Nk / N

        # M-step: centers and widths (weighted Cauchy MLE via fixed-point
        # iteration, one step per EM iteration)
        for k in range(M):
            if Nk[k] < 1e-6:
                continue
            xk = samples
            wk = gamma[:, k]
            # IRLS-style fixed-point update for Cauchy location/scale:
            # weighted mean of x weighted by gamma * 1/((x-c)^2 + d^2)
            #   c <- sum(wk * 2 * x / ((x-c)^2 + d^2)) / sum(wk * 2 / ((x-c)^2 + d^2))
            #   d^2 <- sum(wk * (x-c)^2 / ((x-c)^2 + d^2)) / sum(wk / ((x-c)^2 + d^2))
            # Reference: weighted Cauchy MLE iterations
            for _ in range(5):
                r2 = (xk - centers[k]) ** 2 + widths[k] ** 2
                u = wk / r2
                centers[k] = (u * xk).sum() / u.sum()
                r2 = (xk - centers[k]) ** 2 + widths[k] ** 2
                v_sum = (wk * (xk - centers[k]) ** 2 / r2).sum()
                u_sum = (wk / r2).sum()
                widths[k] = max(np.sqrt(v_sum / max(u_sum, 1e-12)), 1e-4)

        nll = neg_log_lik_full(weights, centers, widths)
        if it > 5 and abs(prev_nll - nll) < 1e-5 * max(abs(prev_nll), 1.0):
            if verbose:
                print(f"    EM converged at iter {it+1}, nLL = {nll:.4f}")
            break
        prev_nll = nll
    else:
        if verbose:
            print(f"    EM ran for full {em_iters} iters, "
                  f"final nLL = {prev_nll:.4f}")

    # Sort by center
    order = np.argsort(centers)
    weights = weights[order]
    centers = centers[order]
    widths = widths[order]

    if verbose:
        print(f"    Cauchy mixture fit (EM):")
        for k in range(M):
            print(f"      comp {k}: w={weights[k]:+.4f}, "
                  f"c={centers[k]:+.4f}, d={widths[k]:.4f}")

    return weights, centers, widths


def assign_to_populations(samples, weights, centers, widths):
    """Hard-assign each sample to its highest-likelihood Cauchy component."""
    diff = samples[:, None] - centers[None, :]
    comp_density = (widths[None, :] / np.pi) / (diff ** 2 + widths[None, :] ** 2)
    posterior = comp_density * weights[None, :]
    return np.argmax(posterior, axis=1)


def coarse_grain_labels(A_fine, labels, M):
    """Mean of A over (block_m, block_n) using assignment labels."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Plasticity rules
# ═══════════════════════════════════════════════════════════════════════════════
# The microscopic and mean-field plasticity drives are now implemented inline
# in `simulate_qif_micro` and `mpr_ode` respectively, since they involve
# multiple state variables (s, x, y) and the bounded a_+, a_- functions.
# See those functions for the per-rule equations.


def plasticity_kernel_shape(rule, dt_arr, a_minus, a_plus, tau_s=1.0, tau_x=None, tau_y=None,
                            alpha_spike=1.0):
    """
    Total weight change Delta A from a single isolated pre-post spike pair as
    a function of dt = t_post - t_pre, for the trace-based STDP rules in PRL
    eq. 30-32 (with the modifications for anti-Hebbian / oja noted in the
    user's message).

    Alpha-kernel synapse:
        tau_s ds/dt = B
        tau_s dB/dt = -2B - s + alpha_spike * delta(t - t_spike)
    Trace filters:
        tau_x dx/dt = -x + s
        tau_y dy/dt = -y + s
    Plasticity (with A held fixed at A_ref):
        Hebbian:     dA = a_+(A) x_pre s_post  -  a_-(A) y_post s_pre
        anti-Hebbian: dA = a_+(A) s_pre^2       -  a_-(A) y_pre  s_post
        oja:         dA = a_+(A) s_post s_pre  -  a_-(A) s_post^2
        with a_+(A) = (1-A), a_-(A) = A (mu set to 1 for shape).

    The kernel is computed by direct numerical integration on a fine grid:
    the pre-spike fires at t=0, the post-spike at t=dt. We simulate s_pre,
    s_post, B_pre, B_post, x_pre, y_post, y_pre (etc. as needed by the rule)
    over a fixed window and integrate the plasticity drive.
    """
    if tau_x is None:
        tau_x = tau_s
    if tau_y is None:
        tau_y = tau_s

    # Numerical-integration window. Cover ~5 tau_max on either side of the
    # spike pair so we capture the full trace decay.
    tau_max = max(tau_s, tau_x, tau_y)
    pad = 8.0 * tau_max
    deltaA = np.zeros_like(dt_arr, dtype=float)

    for k, dt_pair in enumerate(dt_arr):
        # Window: from -pad before the earlier spike to +pad after the later
        t_start = min(0.0, dt_pair) - pad
        t_end   = max(0.0, dt_pair) + pad
        # Time grid (fine relative to tau_s)
        n_pts = max(int((t_end - t_start) / (0.005 * tau_s)), 200)
        t_grid = np.linspace(t_start, t_end, n_pts)
        h = t_grid[1] - t_grid[0]

        # Initialize state for both spike trains
        s_pre = 0.0;  B_pre = 0.0
        s_post = 0.0; B_post = 0.0
        x_pre = 0.0
        x_post = 0.0
        y_post = 0.0
        y_pre = 0.0     # used by anti-Hebbian rule

        # Track plasticity drive contributions
        cum_dA = 0.0
        fired_pre = False
        fired_post = False

        inv_ts = 1.0 / tau_s
        inv_tx = 1.0 / tau_x
        inv_ty = 1.0 / tau_y

        for i_t in range(n_pts):
            t = t_grid[i_t]
            # Trigger spikes when we cross 0 and dt_pair
            if not fired_pre and t >= 0.0:
                B_pre += alpha_spike / tau_s
                fired_pre = True
            if not fired_post and t >= dt_pair:
                B_post += alpha_spike / tau_s
                fired_post = True

            # Compute plasticity drive at this instant
            if rule == "Hebbian":
                drive = a_plus * x_pre * s_post - a_minus * y_post * s_pre
            elif rule == "anti-Hebbian":
                drive = a_plus * x_pre - a_minus * y_pre * s_post
            elif rule == "Oja":
                drive = a_plus * s_post * s_pre - a_minus * (s_post*y_post)
            elif rule == "Symmetric":
                drive = a_plus * x_pre * x_post - a_minus * y_pre * y_post
            else:
                raise ValueError(f"Invalid rule: {rule}")
            cum_dA += drive * h

            # Advance state (forward Euler)
            ds_pre  = B_pre * inv_ts
            dB_pre  = (-2.0 * B_pre - s_pre) * inv_ts
            ds_post = B_post * inv_ts
            dB_post = (-2.0 * B_post - s_post) * inv_ts
            dx_post = (-x_post + s_post) * inv_tx
            dx_pre  = (-x_pre + s_pre) * inv_tx
            dy_post = (-y_post + s_post) * inv_ty
            dy_pre  = (-y_pre + s_pre) * inv_ty

            s_pre  += h * ds_pre;   B_pre  += h * dB_pre
            s_post += h * ds_post;  B_post += h * dB_post
            x_post += h * dx_post
            x_pre  += h * dx_pre
            y_post += h * dy_post
            y_pre  += h * dy_pre

        deltaA[k] = cum_dA

    return deltaA


# ═══════════════════════════════════════════════════════════════════════════════
# Microscopic QIF network
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_qif_micro(eta_i, labels, A0_micro, T, J, a_minus, a_plus, tau_s,
                        plasticity, V_peak=50.0, dt=5e-4, seed=42,
                        tau_x=1.0, tau_y=1.0, alpha_spike=1.0,
                        record_dt=0.05, plast_update_every=10, verbose=False):
    """
    Direct integration of N QIF neurons with alpha-kernel synaptic activations
    and a plastic NxN coupling matrix A_ij with trace-based STDP plasticity.

    Per neuron (PRL eq. 25-26, with the synaptic ODE recast as a 2nd-order
    alpha-kernel form, see lab report eq. for "alpha kernel" - the
    relation tau_A_dot_s = B, tau_A_dot_B = -2B - s + alpha * delta(spikes)):

        dV_i / dt = V_i^2 + eta_i + (J/N) sum_j A_ij s_j
        tau_s * ds_i/dt = B_i                                  (continuous)
        tau_s * dB_i/dt = -2 B_i - s_i                         (continuous)
        on spike of neuron i:  V_i -> -V_peak,  B_i -> B_i + alpha_spike/tau_s

    The pair (s_i, B_i) implements an alpha-kernel convolution
        s_i(t) = sum_k alpha_spike * G_A(t - t_i^(k))
    with G_A(tau) = tau_A^-2 * tau * exp(-tau/tau_A) and tau_A = tau_s.
    The integral int G_A dtau = 1, so each spike contributes a "credit"
    of alpha_spike to int s_i(t) dt (default 1).

    Pre/post trace variables (PRL eq. 30-31):
        tau_x dx_i/dt = -x_i + s_i
        tau_y dy_i/dt = -y_i + s_i

    Plasticity rules (PRL eq. 32 with hard-bounding a_+, a_-):
        Hebbian:     dA_ij/dt = a_+(A_ij) x_j s_i  -  a_-(A_ij) y_i s_j
        anti-Hebbian: dA_ij/dt = a_+(A_ij) s_j^2    -  a_-(A_ij) y_j s_i
        oja:         dA_ij/dt = a_+(A_ij) s_i s_j  -  a_-(A_ij) s_i^2
        with a_+(A) = mu * (1 - A), a_-(A) = mu * A   (bounds A in [0, 1]).

    The per-pair plasticity update is the dominant cost (NxN per step). We
    update A every plast_update_every steps via a multi-step Euler increment.

    Parameters
    ----------
    tau_x, tau_y : float, optional
        Time constants for the pre/post trace filters. Default = tau_s.
    alpha_spike : float
        Multiplier on the spike delta (default 1.0 = unit integrated trace
        contribution per spike).
    gamma : float
        Legacy linear decay; with the bounded a_+ / a_- form, this should
        usually be 0.0 (the bounding already keeps weights well-behaved).

    Returns dict with:
        t_rec         -- recording times
        r_m_rec       -- (M, T) per-ensemble firing rate
        s_m_rec       -- (M, T) per-ensemble mean synaptic activation
        A_final_micro -- (N, N) final coupling
    """

    rng = np.random.default_rng(seed)
    N = len(eta_i)
    M = int(labels.max()) + 1

    V = -2.0 * np.ones(N) + 0.1 * rng.normal(size=N)
    s_i = np.zeros(N)
    B_i = np.zeros(N)
    x_i = np.zeros(N)
    y_i = np.zeros(N)
    A = A0_micro.copy()              # (N, N) microscopic coupling

    n_steps = int(T / dt)
    rec_steps = int(record_dt / dt)
    n_rec = n_steps // rec_steps
    t_rec = (np.arange(n_rec) + 0.5) * record_dt

    r_m_rec = np.zeros((M, n_rec))
    s_m_rec = np.zeros((M, n_rec))

    inv_tau_s = 1.0 / tau_s
    inv_tau_x = 1.0 / tau_x
    inv_tau_y = 1.0 / tau_y
    inv_N = 1.0 / N
    spike_bump = alpha_spike / tau_s   # impulsive jump on B_i per spike
    N_per_ens = np.bincount(labels, minlength=M).astype(float)
    N_per_ens[N_per_ens == 0] = 1.0

    spike_count_bin = np.zeros(M, dtype=np.int64)
    rec_idx = 0

    dt_plast = dt * plast_update_every

    if verbose:
        print(f"  QIF micro sim: N={N}, M={M}, T={T}, dt={dt}, "
              f"n_steps={n_steps}, n_rec={n_rec}, "
              f"plast every {plast_update_every} steps; "
              f"tau_s={tau_s}, tau_x={tau_x}, tau_y={tau_y}")

    for step in range(n_steps):

        # Per-neuron synaptic drive
        I_per_neuron = (J * inv_N) * (A @ s_i)

        # Update V
        V2 = V + dt * (V * V + eta_i + I_per_neuron)

        # Alpha-kernel synapse: two-equation form
        #   tau_s ds = B,    tau_s dB = -2B - s + alpha * delta(spikes)
        s2 = s_i + dt * (B_i * inv_tau_s)
        B2 = B_i + dt * ((-2.0 * B_i - s_i) * inv_tau_s)

        # Trace filters
        x2 = x_i + dt * (-x_i*inv_tau_x + s_i)
        y2 = y_i + dt * (-y_i*inv_tau_y + s_i)

        # Plasticity update (multi-step Euler).
        # Compute drive matrices: pre/post outer products depending on rule.
        if (step + 1) % plast_update_every == 0:

            if plasticity == "Hebbian":
                # dA_ij = a_+ x_j s_i  -  a_- y_i s_j
                # outer indexing: row=i (post), col=j (pre).
                pos_drive = np.outer(s_i, x_i)        # s_i (post, rows) * x_j (pre, cols)
                neg_drive = np.outer(y_i, s_i)        # y_i (post, rows) * s_j (pre, cols)
            elif plasticity == "anti-Hebbian":
                # dA_ij = a_+ s_j^2  -  a_- y_j s_i
                # outer indexing: row=i (post), col=j (pre)
                pos_drive = np.tile((x_i)[None, :], (N, 1))   # s_j^2 along columns
                neg_drive = np.outer(s_i, y_i)                     # s_i * y_j
            elif plasticity == "Oja":
                # dA_ij = a_+ s_i s_j  -  a_- s_i^2
                pos_drive = np.outer(s_i, s_i)                     # s_i * s_j
                neg_drive = np.tile((s_i * y_i)[:, None], (1, N))   # s_i^2 along rows
            elif plasticity == "Symmetric":
                pos_drive = np.outer(x_i, x_i)  # s_i (post, rows) * x_j (pre, cols)
                neg_drive = np.outer(y_i, y_i)  # y_i (post, rows) * s_j (pre, cols)
            else:
                raise ValueError(f"Invalid plasticity rule: {plasticity}")

            dA = a_plus * pos_drive - a_minus * neg_drive
            A += dt_plast * dA
            np.clip(A, 0.0, 1.0, out=A)   # hard clip as a safety net

        # state variable updates
        V = V2
        s_i = s2
        B_i = B2
        x_i = x2
        y_i = y2

        # Spike detection
        spiked = V >= V_peak
        if spiked.any():
            V[spiked] = -V_peak
            B_i[spiked] += spike_bump
            spike_count_bin += np.bincount(labels[spiked], minlength=M)

        if (step + 1) % rec_steps == 0 and rec_idx < n_rec:
            r_m_rec[:, rec_idx] = spike_count_bin / (N_per_ens * record_dt)
            s_m_rec[:, rec_idx] = np.bincount(labels, weights=s_i,
                                                minlength=M) / N_per_ens
            spike_count_bin[:] = 0
            rec_idx += 1

    return dict(
        t_rec=t_rec, r_m_rec=r_m_rec, s_m_rec=s_m_rec,
        A_final_micro=A.copy(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MPR mean-field for M ensembles
# ═══════════════════════════════════════════════════════════════════════════════
def mpr_ode(t, y, eta_pop, delta_pop, w_pop, J, a_minus, a_plus,
             tau_s, tau_x, tau_y, plasticity):
    """
    MPR ensemble equations with alpha-kernel synaptic activations, trace
    variables x_m, y_m, and trace-based STDP plasticity rules with bounded
    weights.

    State layout (length 6M + M*M):
        y[0    : M  ] = r       (firing rates)
        y[M    : 2M ] = v       (mean membrane potentials)
        y[2M   : 3M ] = s       (alpha-kernel-filtered firing rates)
        y[3M   : 4M ] = B       (alpha-kernel auxiliary variable)
        y[4M   : 5M ] = x       (pre-trace = LP(s; tau_x))
        y[5M   : 6M ] = y_var   (post-trace = LP(s; tau_y))
        y[6M   :    ] = A.flat  (M x M coupling)

    Equations:
        dr/dt = Delta_m / pi + 2 r v
        dv/dt = v^2 + eta_bar_m - (pi r)^2 + J sum_n A_mn w_n s_n
        tau_s ds/dt = B
        tau_s dB/dt = -2B - s + alpha * r       (alpha=1 by convention)
        tau_x dx/dt = -x + s
        tau_y dy/dt = -y + s

    Plasticity rules:
        Hebbian:     dA_mn = a_+(A_mn) x_n s_m  -  a_-(A_mn) y_m s_n
        anti-Hebbian: dA_mn = a_+(A_mn) s_n^2    -  a_-(A_mn) y_n s_m
        oja:         dA_mn = a_+(A_mn) s_m s_n  -  a_-(A_mn) s_m^2
        with a_+(A) = mu(1-A), a_-(A) = mu A.
    """
    M = len(eta_pop)
    r   = y[0 * M : 1 * M]
    v   = y[1 * M : 2 * M]
    s   = y[2 * M : 3 * M]
    B   = y[3 * M : 4 * M]
    x_pre  = y[4 * M : 5 * M]
    y_post = y[5 * M : 6 * M]
    A   = y[6 * M:].reshape(M, M)
    A_c = np.clip(A, 0.0, 1.0)

    inv_tau_s = 1.0 / tau_s
    inv_tau_x = 1.0 / tau_x
    inv_tau_y = 1.0 / tau_y

    # Synaptic input to each ensemble
    I_syn = J * (A_c @ (w_pop * s))     # shape (M,)

    # MPR equations
    dr = delta_pop / np.pi + 2.0 * r * v
    dv = v * v + eta_pop - (np.pi * r) ** 2 + I_syn

    # Alpha-kernel synapse driven by r (mean-field analog of spike train)
    ds = B * inv_tau_s
    dB = (-2.0 * B - s + r) * inv_tau_s    # alpha = 1

    # Trace filters
    dx = -x_pre * inv_tau_x  + s
    dy = -y_post * inv_tau_y + s

    if plasticity == "Hebbian":
        # dA_mn = a_+ x_n s_m  -  a_- y_m s_n
        pos_drive = np.outer(s, x_pre)             # s_m * x_n
        neg_drive = np.outer(y_post, s)            # y_m * s_n
    elif plasticity == "anti-Hebbian":
        # dA_mn = a_+ s_n^2  -  a_- y_n s_m
        pos_drive = np.tile((x_pre)[None, :], (M, 1))
        neg_drive = np.outer(s, y_post)            # s_m * y_n
    elif plasticity == "Oja":
        # dA_mn = a_+ s_m s_n  -  a_- s_m^2
        pos_drive = np.outer(s, s)
        neg_drive = np.tile((s * y_post)[:, None], (1, M))
    elif plasticity == "Symmetric":
        pos_drive = np.outer(x_pre, x_pre)  # s_m * x_n
        neg_drive = np.outer(y_post, y_post)  # y_m * s_n
    else:
        raise ValueError(f"Invalid plasticity rule: {plasticity}")
    dA = a_plus * pos_drive - a_minus * neg_drive

    return np.concatenate([dr, dv, ds, dB, dx, dy, dA.ravel()])


# ═══════════════════════════════════════════════════════════════════════════════
# One full (microscopic + mean-field) run for a given plasticity rule
# ═══════════════════════════════════════════════════════════════════════════════
def run_one_rule(plasticity, *,
                 eta_micro, labels, A0_micro,
                 w_pop, eta_pop, delta_pop, r0, v0, s0, A0_mf,
                 N, M, T, J, a_minus, a_plus, tau_s, tau_x, tau_y,
                 V_peak, dt_micro,
                 method, rtol, atol):
    print(f"\n▸ Plasticity rule: {plasticity}")

    # ── Microscopic simulation ─────────────────────────────────────────────
    print("  Running QIF micro …")
    micro = simulate_qif_micro(
        eta_i=eta_micro, labels=labels, A0_micro=A0_micro, T=T,
        J=J, a_minus=a_minus, a_plus=a_plus, tau_s=tau_s,
        tau_x=tau_x, tau_y=tau_y,
        plasticity=plasticity,
        V_peak=V_peak, dt=dt_micro, verbose=True,
    )
    t_qif = micro["t_rec"]
    r_qif_m = micro["r_m_rec"]
    s_qif_m = micro["s_m_rec"]
    A_qif_final_NxN = micro["A_final_micro"]    # (N, N)
    r_qif_macro = (w_pop @ r_qif_m)
    s_qif_macro = (w_pop @ s_qif_m)
    print(f"    done")

    # ── MPR mean-field ─────────────────────────────────────────────────────
    print("  Running MPR mean-field …")
    # State layout: r (M), v (M), s (M), B (M), x (M), y (M), A (M*M)
    B0 = np.zeros(M)
    x0 = np.zeros(M)
    y0_trace = np.zeros(M)
    y0_mf = np.concatenate([r0, v0, s0, B0, x0, y0_trace, A0_mf.ravel()])
    sol_mf = solve_ivp(
        mpr_ode, (0, T), y0_mf, method=method,
        args=(eta_pop, delta_pop, w_pop, J, a_minus, a_plus,
              tau_s, tau_x, tau_y, plasticity),
        rtol=rtol, atol=atol, dense_output=False,
        max_step=0.05,
    )
    if not sol_mf.success:
        raise RuntimeError(f"MF failed: {sol_mf.message}")
    t_mf = sol_mf.t
    r_mf_m = sol_mf.y[0 * M : 1 * M]
    s_mf_m = sol_mf.y[2 * M : 3 * M]
    A_mf = sol_mf.y[6 * M:].reshape(M, M, -1)
    A_mf_final = A_mf[:, :, -1]
    r_mf_macro = w_pop @ r_mf_m
    s_mf_macro = w_pop @ s_mf_m
    print(f"    done — {sol_mf.t.size} steps")

    # Coarse-grain the microscopic NxN A_ij to ensemble-level MxM
    A_qif_cg = coarse_grain_labels(A_qif_final_NxN, labels, M)

    return dict(
        plasticity=plasticity,
        t_qif=t_qif, r_qif=r_qif_macro, s_qif=s_qif_macro,
        A_qif_cg=A_qif_cg,
        t_mf=t_mf, r_mf=r_mf_macro, s_mf=s_mf_macro,
        A_mf_final=A_mf_final,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Full sweep across plasticity rules
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_three_rules(
        N=500, M=12, T=150.0,
        J=-2.0, a_minus=0.02, a_plus=0.02,
        tau_s=1.0, tau_x=None, tau_y=None,
        gmm_means=(-0.0, 0.5, 1.0),
        gmm_sigmas=(0.2, 0.15, 0.1),
        gmm_weights=(0.3, 0.4, 0.3),
        V_peak=50.0, dt_micro=5e-4,
        seed=42,
        method="RK45", rtol=1e-6, atol=1e-8,
        A0_value=0.5,
):
    if tau_x is None:
        tau_x = tau_s
    if tau_y is None:
        tau_y = tau_s

    print(f"Sampling eta_i from 3-Gaussian mixture  (N={N})")
    eta_micro = sample_gaussian_mixture(N, gmm_means, gmm_sigmas, gmm_weights,
                                          seed=seed)

    print(f"Fitting Cauchy mixture with M={M} components …")
    w_pop, eta_pop, delta_pop = fit_cauchy_mixture(eta_micro, M,
                                                    seed=seed, verbose=True)
    labels = assign_to_populations(eta_micro, w_pop, eta_pop, delta_pop)

    # Matched initial conditions: small firing rate, no synaptic activity
    r0 = 0.05 * np.ones(M)
    v0 = -1.0 * np.ones(M)
    s0 = np.zeros(M)

    # Initial coupling: a value in (0, 1) so both a_+ (potentiation) and a_-
    # (depression) terms are active from the start. A0 = 0.5 sits in the middle
    # of the bounded interval.
    A0_micro = A0_value * np.ones((N, N))
    A0_mf    = A0_value * np.ones((M, M))

    runs = {}
    for rule in PLASTICITY_RULES:
        runs[rule] = run_one_rule(
            rule,
            eta_micro=eta_micro, labels=labels, A0_micro=A0_micro,
            w_pop=w_pop, eta_pop=eta_pop, delta_pop=delta_pop,
            r0=r0, v0=v0, s0=s0, A0_mf=A0_mf,
            N=N, M=M, T=T, J=J, a_minus=a_minus, a_plus=a_plus,
            tau_s=tau_s, tau_x=tau_x, tau_y=tau_y,
            V_peak=V_peak, dt_micro=dt_micro,
            method=method, rtol=rtol, atol=atol,
        )

    return dict(
        N=N, M=M, T=T, J=J, a_minus=a_minus, a_plus=a_plus,
        tau_s=tau_s, tau_x=tau_x, tau_y=tau_y,
        eta_micro=eta_micro, labels=labels,
        w_pop=w_pop, eta_pop=eta_pop, delta_pop=delta_pop,
        gmm_means=gmm_means, gmm_sigmas=gmm_sigmas, gmm_weights=gmm_weights,
        runs=runs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2 × 9 figure (matches theta_three_rules.svg layout)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_figure(res, a_minus, a_plus, savepath="qif_three_rules.svg"):
    set_prx_style()

    rules = list(PLASTICITY_RULES)
    N, M, T = res["N"], res["M"], res["T"]
    tau_s = res.get("tau_s", 1.0)

    # Per-rule colour scales
    CMAP_POS = "viridis"
    per_rule_norm = {}
    for rule in rules:
        d = res["runs"][rule]
        vmax = max(d["A_qif_cg"].max(), d["A_mf_final"].max()) or 1.0
        per_rule_norm[rule] = dict(cmap=CMAP_POS, vmin=0.0, vmax=vmax,
                                    interpolation="nearest", aspect="equal")

    # Layout
    width_ratios = [1.0, 1.0, 0.35,
                    1.0, 1.0, 0.35,
                    1.0, 1.0, 0.35]
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

    # ── Row 0, cols 0-1: eta_i distribution + Cauchy mixture fit ─────────
    ax_f = fig.add_subplot(gs[0, 0:2])
    eta_micro = res["eta_micro"]
    means, sigmas, weights = (res["gmm_means"], res["gmm_sigmas"],
                                res["gmm_weights"])
    w_pop, eta_pop, delta_pop = (res["w_pop"], res["eta_pop"],
                                  res["delta_pop"])

    hi, lo = np.percentile(eta_micro, [99.5, 0.5])
    edges = np.linspace(lo, hi, 60)
    ax_f.hist(eta_micro, bins=edges, density=True, color=C_QIF,
              alpha=0.30, edgecolor="none", label="QIF sample")

    x_grid = np.linspace(lo, hi, 600)
    pdf_g = gaussian_mixture_pdf(x_grid, means, sigmas, weights)
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
    ymax = 1.5 * pdf_g.max()
    ax_f.set_ylim(0.0, ymax)
    ax_f.legend(loc="upper right", frameon=False, handlelength=2.0,
                borderaxespad=0.3, fontsize=9)
    make_panel_label(ax_f, f"({next(panel_letters)})", x=-0.10, y=1.04)

    # ── Row 0, cols 3-4: STDP-style plasticity-rule kernels Delta A(dt) ───
    # dt = t_post - t_pre, with a window centered on 0. Use a moderately
    # coarse grid for the numerical integration (the kernel itself is smooth).
    tau_x = res.get("tau_x", tau_s)
    tau_y = res.get("tau_y", tau_s)
    window = 6.0 * max(tau_s, tau_x, tau_y)
    ax_shape = fig.add_subplot(gs[0, 3:5])
    dt_grid = np.linspace(-window, window, 81)
    rule_colors = {"Hebbian":     "#1f4e79",
                   "anti-Hebbian": "#3a8e7c",
                   "Oja":         "#c44e52",
                   "Symmetric":   "#c44e52",
                   }
    rule_styles = {"Hebbian": "-", "anti-Hebbian": "--", "Oja": ":", "Symmetric": ":"}
    for rule in rules:
        y = plasticity_kernel_shape(rule, dt_grid, a_minus=a_minus, a_plus=a_plus,
                                    tau_s=tau_s, tau_x=tau_x, tau_y=tau_y,
                                    )
        ax_shape.plot(dt_grid, y, color=rule_colors[rule],
                       ls=rule_styles[rule], lw=1.4, label=rule)
    ax_shape.axhline(0.0, color="0.6", lw=0.5)
    ax_shape.axvline(0.0, color="0.6", lw=0.5)
    ax_shape.set_xlabel(r"$t_{\mathrm{post}} - t_{\mathrm{pre}}$")
    ax_shape.set_ylabel(r"$\Delta A$  ($A_{\mathrm{ref}}{=}0.5$)")
    ax_shape.set_xlim(-window, window)
    ax_shape.legend(loc="best", frameon=False, handlelength=2.0,
                     borderaxespad=0.3, fontsize=9)
    make_panel_label(ax_shape, f"({next(panel_letters)})", x=-0.10, y=1.04)

    # ── Row 0, cols 6-7: mean synaptic activity s(t) for Symmetric rule only ───
    d_oja = res["runs"]["Symmetric"]
    ax_s = fig.add_subplot(gs[0, 6:8])
    ax_s.plot(d_oja["t_qif"], d_oja["s_qif"], color=C_QIF, lw=1.4,
              label="QIF")
    ax_s.plot(d_oja["t_mf"], d_oja["s_mf"], color=C_MF, lw=1.4, ls="--",
              label="MF")
    ax_s.set_xlim(0, T)
    ymax = 1.05 * max(np.max(d_oja["s_qif"]), np.max(d_oja["s_mf"]))
    if ymax > 0:
        ax_s.set_ylim(0, ymax)
    ax_s.set_xlabel(r"time $t$")
    ax_s.set_ylabel(r"$\langle s\rangle$")
    ax_s.legend(loc="upper right", frameon=False, handlelength=2.0,
                 borderaxespad=0.3, fontsize=9)
    make_panel_label(ax_s, f"({next(panel_letters)})", x=-0.10, y=1.04)

    # ── Row 1: per-rule QIF + MF matrices ───────────────────────────────
    pending_cbars = []
    for r_idx, rule in enumerate(rules):
        col_qif, col_mf, col_cb = rule_cols[r_idx]
        d = res["runs"][rule]
        norm_kw = per_rule_norm[rule]

        ax_qif = fig.add_subplot(gs[1, col_qif])
        ax_mf  = fig.add_subplot(gs[1, col_mf])

        ax_qif.imshow(d["A_qif_cg"], **norm_kw)
        im_mf = ax_mf.imshow(d["A_mf_final"], **norm_kw)

        ax_qif.set_title("QIF (block-avg)")
        ax_mf.set_title("MF (OA)")

        tick_step = max(1, M // 6)
        ticks = np.arange(0, M, tick_step)
        ax_qif.set_xticks(ticks); ax_qif.set_yticks(ticks)
        ax_mf.set_xticks(ticks); ax_mf.set_yticks(ticks)
        ax_qif.set_xticklabels(ticks); ax_qif.set_yticklabels(ticks)
        ax_mf.set_xticklabels(ticks); ax_mf.set_yticklabels(ticks)
        ax_qif.tick_params(axis="both", labelsize=9)
        ax_mf.tick_params(axis="both", labelsize=9)
        ax_qif.set_xlabel(r"ensemble $n$")
        ax_mf.set_xlabel(r"ensemble $n$")
        ax_qif.set_ylabel(rule, fontsize=12, fontweight="bold")
        ax_mf.set_ylabel(r"ensemble $m$", fontsize=10)

        pending_cbars.append((ax_mf, im_mf))
        make_panel_label(ax_qif, f"({next(panel_letters)})", x=-0.10, y=1.04)

    # Finalise layout and place colorbars
    fig.canvas.draw()
    CBAR_W = 0.010
    CBAR_PAD = 0.006
    for ax_mf, im_mf in pending_cbars:
        pos = ax_mf.get_position()
        cax = fig.add_axes([pos.x1 + CBAR_PAD, pos.y0, CBAR_W, pos.height])
        cb = fig.colorbar(im_mf, cax=cax)
        cb.set_label(r"$A_{mn}$", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        cb.locator = plt.MaxNLocator(nbins=4)
        cb.update_ticks()

    fig.savefig(savepath, bbox_inches="tight")
    print(f"\nFigure saved → {savepath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    a_m, a_p = 0.005, 0.01
    CONFIG = dict(
        N=200, M=12, T=200.0,
        J=10.0, a_minus=a_m, a_plus=a_p,
        tau_s=0.1, tau_x=2.0, tau_y=3.0,
        gmm_means=(-0.1, 0.0, 0.2),
        gmm_sigmas=(0.08, 0.1, 0.05),
        gmm_weights=(0.3, 0.5, 0.3),
        V_peak=100.0, dt_micro=5e-4,
        seed=42, method="RK45", rtol=1e-6, atol=1e-8,
    )

    res = simulate_three_rules(**CONFIG)
    plot_figure(res, a_m, a_p, savepath="/home/rgast/data/qif_plasticity/qif_three_rules.svg")
    # plot_figure(res, savepath="/mnt/user-data/outputs/qif_three_rules.png")
    plt.show()
