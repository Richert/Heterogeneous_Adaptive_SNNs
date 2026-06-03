"""
Parameter sweep comparing three models of a recurrently coupled spiking
network with Hebbian STDP, across (tau_s, M):

  (1) QIF microscopic network
        dV_i/dt   = V_i^2 + eta_i + (J/N) * sum_j A_{ij} s_j        (PRL eq. 25)
        tau_s ds_i = B_i,   tau_s dB_i = -2 B_i - s_i + alpha * Sigma_delta
                                                                    (PRL eq. 26
                                                                     as alpha
                                                                     kernel ODE
                                                                     pair, lab
                                                                     report)
        tau_x dx_i = -x_i + s_i, tau_y dy_i = -y_i + s_i           (PRL eq. 30-31)
        dA_ij/dt  = a_+(A_ij) x_j s_i - a_-(A_ij) y_i s_j          (PRL eq. 32)

  (2) MPR-OA mean-field ensembles (one set of equations per ensemble m)
        dr_m/dt   = Delta_m / pi + 2 r_m v_m                       (PRL eq. 33)
        dv_m/dt   = v_m^2 + bar_eta_m - (pi r_m)^2
                    + J * sum_n A_mn w_n s_n                       (PRL eq. 34)
        tau_s ds_m = B_m,   tau_s dB_m = -2 B_m - s_m + r_m        (PRL eq. 35
                                                                     w/ alpha
                                                                     kernel)
        tau_x dx_m = -x_m + s_m, tau_y dy_m = -y_m + s_m           (PRL eq. 35-36)
        dA_mn/dt  = a_+(A_mn) x_n s_m - a_-(A_mn) y_m s_n          (PRL eq. 38)

  (3) Wilson-Cowan rate model (same synaptic + trace + plasticity equations
        but with v_m collapsed: the firing rate r_m follows a sigmoid-like
        closed form)
        dr_m/dt   = -r_m + (1/sqrt(2 pi))
                    * sqrt(bar_eta_m + I(t)
                            + sqrt((bar_eta_m + I(t))^2 + Delta_m^2))
                  with I(t) = J * sum_n A_mn w_n s_n               (PRL eq. 39)

The QIF microscopic ensemble has uniformly-distributed eta_i in some interval
[eta_min, eta_max]. The mean-field ensembles approximate this uniform
distribution by a Lorentzian mixture with M components: each Lorentzian sits
at the midpoint of one of M equal subintervals, with HWHM equal to half the
subinterval width, and weight w_m = 1/M (PRL eq. 17).

For each (tau_s, M) cell we save:
    - A_final_qif_cg : (M, M) block-averaged final QIF coupling
    - A_final_mpr    : (M, M) final mean-field coupling
    - A_final_wc     : (M, M) final WC coupling
    - t_rec          : (T_rec,) common record times
    - s_mean_qif     : (T_rec,) network-mean synaptic activation in QIF
    - s_mean_mpr     : (T_rec,) network-mean synaptic activation in MPR
    - s_mean_wc      : (T_rec,) network-mean synaptic activation in WC
    - r_mean_qif     : (T_rec,) network-mean firing rate in QIF (binned)
    - r_mean_mpr     : (T_rec,) network-mean firing rate in MPR
    - r_mean_wc      : (T_rec,) network-mean firing rate in WC
    - eta_pop, delta_pop, w_pop : (M,) ensemble parameters

Plus shared scalars: N, T, J, eta range, a_plus/a_minus identities, etc.

The results are pickled to a single .pkl file (numpy arrays); a plotting/
analysis step can consume that file downstream.
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _default_a_plus(A):
    return 0.02 * (1.0 - A)


def _default_a_minus(A):
    return 0.02 * A


def make_uniform_eta(N, eta_min, eta_max, kind="equispaced", seed=0):
    """
    Generate N excitabilities from a uniform distribution on [eta_min, eta_max].

    kind = "equispaced": evenly-spaced grid (smallest finite-N variance)
    kind = "random"   : iid uniform samples
    """
    if kind == "equispaced":
        # Quasi-uniform: midpoints of equal-area bins, avoids edge-pile-up
        u = (np.arange(N) + 0.5) / N
        return eta_min + (eta_max - eta_min) * u
    elif kind == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(eta_min, eta_max, size=N)
    else:
        raise ValueError(f"Unknown kind={kind}")


def make_lorentzian_mixture(M, eta_min, eta_max):
    """
    Build an M-component Lorentzian mixture approximating U(eta_min, eta_max).

    Each component sits at the midpoint of one of M equal subintervals of
    [eta_min, eta_max], with HWHM equal to half the subinterval width, and
    weight w_m = 1/M.

    The HWHM choice is a heuristic: making the Lorentzian narrower than the
    subinterval gives a more "delta-like" ensemble (low intra-ensemble
    heterogeneity); making it wider blurs across subintervals. Half the
    subinterval width gives a good visual match to the uniform density in
    the bulk (per component). For large M this naturally narrows.
    """
    edges = np.linspace(eta_min, eta_max, M + 1)
    eta_pop = 0.5 * (edges[:-1] + edges[1:])
    sub_w = (eta_max - eta_min) / M
    delta_pop = np.full(M, 0.5 * sub_w)
    w_pop = np.full(M, 1.0 / M)
    return eta_pop, delta_pop, w_pop


def assign_to_ensembles_uniform(eta_arr, eta_min, eta_max, M):
    """
    Assign neurons (with excitabilities eta_arr) to one of M ensembles by
    which subinterval of [eta_min, eta_max] they belong to.

    Returns an int array of shape (N,) with values in [0, M-1]. Neurons that
    fall exactly on eta_max go to ensemble M-1.
    """
    sub_w = (eta_max - eta_min) / M
    labels = np.floor((eta_arr - eta_min) / sub_w).astype(int)
    labels = np.clip(labels, 0, M - 1)
    return labels


def block_average(A_fine, labels, M):
    """
    Coarse-grain an N x N matrix to M x M by averaging within label blocks.
    """
    Ac = np.zeros((M, M))
    for I in range(M):
        idx_I = np.where(labels == I)[0]
        if idx_I.size == 0:
            continue
        for J_ in range(M):
            idx_J = np.where(labels == J_)[0]
            if idx_J.size == 0:
                continue
            Ac[I, J_] = A_fine[np.ix_(idx_I, idx_J)].mean()
    return Ac


# ═══════════════════════════════════════════════════════════════════════════════
# (1) QIF microscopic simulator
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_qif(eta_i, labels, A0_micro, *,
                  J, tau_s, tau_x=None, tau_y=None,
                  a_plus=None, a_minus=None,
                  alpha_spike=1.0,
                  T=80.0, dt=5e-4, V_peak=50.0,
                  plast_update_every=10, record_dt=0.1,
                  seed=42, verbose=False):
    """
    Forward-Euler simulation of N QIF neurons with alpha-kernel synapses,
    pre/post traces, and Hebbian-STDP plasticity for an N x N coupling A.

    Returns dict with t_rec, s_mean (network mean s), r_mean (per-bin),
    r_per_pop (per-ensemble rate via labels), s_per_pop, A_final_NxN.
    """
    if tau_x is None:
        tau_x = tau_s
    if tau_y is None:
        tau_y = tau_s
    if a_plus is None:
        a_plus = _default_a_plus
    if a_minus is None:
        a_minus = _default_a_minus

    rng = np.random.default_rng(seed)
    N = len(eta_i)
    M = int(labels.max()) + 1

    V = -2.0 * np.ones(N) + 0.1 * rng.normal(size=N)
    s = np.zeros(N)
    B = np.zeros(N)
    x = np.zeros(N)
    y = np.zeros(N)
    A = A0_micro.copy()
    eta = eta_i.astype(float)

    n_steps = int(T / dt)
    rec_steps = max(1, int(record_dt / dt))
    n_rec = n_steps // rec_steps
    t_rec = (np.arange(n_rec) + 0.5) * record_dt

    s_mean_hist = np.zeros(n_rec)
    r_mean_hist = np.zeros(n_rec)
    s_per_pop_hist = np.zeros((M, n_rec))
    r_per_pop_hist = np.zeros((M, n_rec))

    inv_ts = 1.0 / tau_s
    inv_tx = 1.0 / tau_x
    inv_ty = 1.0 / tau_y
    inv_N = 1.0 / N
    spike_bump = alpha_spike / tau_s
    dt_plast = dt * plast_update_every

    spike_count_total = 0
    spike_count_per_pop = np.zeros(M, dtype=np.int64)
    rec_idx = 0
    N_per_pop = np.bincount(labels, minlength=M).astype(float)
    N_per_pop[N_per_pop == 0] = 1.0

    t0 = time.time()
    if verbose:
        print(f"    QIF: N={N}, M={M}, T={T}, dt={dt}, n_steps={n_steps}, "
              f"plast every {plast_update_every} steps")

    for k in range(n_steps):
        # Synaptic input: (J/N) * sum_j A_{ij} s_j
        I_syn = (J * inv_N) * (A @ s)

        # Update V
        V += dt * (V * V + eta + I_syn)

        # Continuous parts of alpha-kernel and traces
        s += dt * (B * inv_ts)
        B += dt * ((-2.0 * B - s) * inv_ts)
        x += dt * ((-x + s) * inv_tx)
        y += dt * ((-y + s) * inv_ty)

        # Spike detection & reset
        spiked = V >= V_peak
        if spiked.any():
            V[spiked] = -V_peak
            B[spiked] += spike_bump
            spike_count_total += spiked.sum()
            spike_count_per_pop += np.bincount(labels[spiked], minlength=M)

        # Plasticity update (multi-step Euler)
        if (k + 1) % plast_update_every == 0:
            A_clip = np.clip(A, 0.0, 1.0)
            a_plus_mat = a_plus(A_clip)
            a_minus_mat = a_minus(A_clip)
            # Hebbian: dA_ij = a_+ * x_j s_i  -  a_- * y_i s_j
            pos_drive = np.outer(s, x)         # s_i (rows) * x_j (cols)
            neg_drive = np.outer(y, s)         # y_i (rows) * s_j (cols)
            dA = a_plus_mat * pos_drive - a_minus_mat * neg_drive
            A += dt_plast * dA
            np.clip(A, 0.0, 1.0, out=A)

        # Record
        if (k + 1) % rec_steps == 0 and rec_idx < n_rec:
            s_mean_hist[rec_idx] = s.mean()
            r_mean_hist[rec_idx] = spike_count_total / (N * record_dt)
            s_per_pop_hist[:, rec_idx] = np.bincount(labels, weights=s,
                                                       minlength=M) / N_per_pop
            r_per_pop_hist[:, rec_idx] = spike_count_per_pop / (N_per_pop * record_dt)
            spike_count_total = 0
            spike_count_per_pop[:] = 0
            rec_idx += 1

    if verbose:
        print(f"    QIF done in {time.time() - t0:.1f}s")

    return dict(
        t_rec=t_rec,
        s_mean=s_mean_hist, r_mean=r_mean_hist,
        s_per_pop=s_per_pop_hist, r_per_pop=r_per_pop_hist,
        A_final_NxN=A,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# (2) MPR-OA mean-field
# ═══════════════════════════════════════════════════════════════════════════════
def _mpr_rhs(t, y, M, eta_pop, delta_pop, w_pop, J,
              tau_s, tau_x, tau_y, a_plus, a_minus):
    r   = y[0 * M : 1 * M]
    v   = y[1 * M : 2 * M]
    s   = y[2 * M : 3 * M]
    B   = y[3 * M : 4 * M]
    x   = y[4 * M : 5 * M]
    yt  = y[5 * M : 6 * M]
    A   = y[6 * M:].reshape(M, M)
    A_clip = np.clip(A, 0.0, 1.0)

    I_syn = J * (A_clip @ (w_pop * s))
    dr = delta_pop / np.pi + 2.0 * r * v
    dv = v * v + eta_pop - (np.pi * r) ** 2 + I_syn
    ds = B / tau_s
    dB = (-2.0 * B - s + r) / tau_s
    dx = (-x + s) / tau_x
    dy = (-yt + s) / tau_y

    ap = a_plus(A_clip)
    am = a_minus(A_clip)
    pos = np.outer(s, x)         # s_m (row) * x_n (col)
    neg = np.outer(yt, s)        # y_m (row) * s_n (col)
    dA = ap * pos - am * neg

    return np.concatenate([dr, dv, ds, dB, dx, dy, dA.ravel()])


def simulate_mpr(eta_pop, delta_pop, w_pop, A0_mf, *,
                  J, tau_s, tau_x=None, tau_y=None,
                  a_plus=None, a_minus=None,
                  T=80.0, record_dt=0.1,
                  method="RK45", rtol=1e-6, atol=1e-8, verbose=False):
    """
    Integrate the MPR-OA mean-field equations for M ensembles with Hebbian-STDP
    plasticity on the M x M coupling matrix.
    """
    if tau_x is None:
        tau_x = tau_s
    if tau_y is None:
        tau_y = tau_s
    if a_plus is None:
        a_plus = _default_a_plus
    if a_minus is None:
        a_minus = _default_a_minus

    M = len(eta_pop)
    n_rec = int(T / record_dt)
    t_eval = (np.arange(n_rec) + 0.5) * record_dt

    # Initial conditions: small positive r, slightly negative v, zero synaptic
    r0 = 0.05 * np.ones(M)
    v0 = -1.0 * np.ones(M)
    s0 = np.zeros(M)
    B0 = np.zeros(M)
    x0 = np.zeros(M)
    y0_tr = np.zeros(M)
    y0 = np.concatenate([r0, v0, s0, B0, x0, y0_tr, A0_mf.ravel()])

    t0 = time.time()
    sol = solve_ivp(
        _mpr_rhs, (0, T), y0, t_eval=t_eval, method=method,
        args=(M, eta_pop, delta_pop, w_pop, J, tau_s, tau_x, tau_y,
              a_plus, a_minus),
        rtol=rtol, atol=atol, max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"MPR ODE failed: {sol.message}")
    if verbose:
        print(f"    MPR done in {time.time() - t0:.1f}s "
              f"({sol.t.size} pts, {sol.nfev} evals)")

    r_per_pop = sol.y[0 * M : 1 * M, :]
    s_per_pop = sol.y[2 * M : 3 * M, :]
    A_traj = sol.y[6 * M:, :].reshape(M, M, -1)
    A_final = A_traj[:, :, -1]

    # Network means (weighted by w_pop)
    s_mean = (w_pop[:, None] * s_per_pop).sum(axis=0)
    r_mean = (w_pop[:, None] * r_per_pop).sum(axis=0)

    return dict(
        t_rec=sol.t, s_mean=s_mean, r_mean=r_mean,
        s_per_pop=s_per_pop, r_per_pop=r_per_pop,
        A_final=np.clip(A_final, 0.0, 1.0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# (3) Wilson-Cowan rate model (PRL eq. 39)
# ═══════════════════════════════════════════════════════════════════════════════
def _wc_rhs(t, y, M, eta_pop, delta_pop, w_pop, J,
             tau_s, tau_x, tau_y, a_plus, a_minus):
    # State layout: r (M), s (M), B (M), x (M), yt (M), A (M^2)  -- no v!
    r  = y[0 * M : 1 * M]
    s  = y[1 * M : 2 * M]
    B  = y[2 * M : 3 * M]
    x  = y[3 * M : 4 * M]
    yt = y[4 * M : 5 * M]
    A  = y[5 * M:].reshape(M, M)
    A_clip = np.clip(A, 0.0, 1.0)

    I_syn = J * (A_clip @ (w_pop * s))

    # PRL eq. 39: closed-form sigmoidal nonlinearity for r.
    # Note: the displayed factor 1/sqrt(2 pi) in the PRL is ambiguous to
    # interpret from the OCR; the MPR derivation gives 1/(pi*sqrt(2)) at
    # steady state, which matches the MPR fixed point exactly. We use that
    # factor here so WC ~ MPR at steady state; if the PRL truly uses
    # 1/sqrt(2 pi), the WC values will differ by a factor sqrt(pi).
    #   dr/dt = -r + (1/(pi*sqrt(2)))
    #            * sqrt( bar_eta + I + sqrt((bar_eta+I)^2 + Delta^2) )
    arg_outer = eta_pop + I_syn
    arg_inner = np.sqrt(arg_outer * arg_outer + delta_pop ** 2)
    # Clip non-negativity of sqrt argument (it should always be >= 0)
    rad = arg_outer + arg_inner
    rad = np.maximum(rad, 0.0)
    r_inf = np.sqrt(rad) / (np.pi * np.sqrt(2.0))
    dr = -r + r_inf

    ds = B / tau_s
    dB = (-2.0 * B - s + r) / tau_s
    dx = (-x + s) / tau_x
    dy = (-yt + s) / tau_y

    ap = a_plus(A_clip)
    am = a_minus(A_clip)
    pos = np.outer(s, x)
    neg = np.outer(yt, s)
    dA = ap * pos - am * neg

    return np.concatenate([dr, ds, dB, dx, dy, dA.ravel()])


def simulate_wc(eta_pop, delta_pop, w_pop, A0_mf, *,
                 J, tau_s, tau_x=None, tau_y=None,
                 a_plus=None, a_minus=None,
                 T=80.0, record_dt=0.1,
                 method="RK45", rtol=1e-6, atol=1e-8, verbose=False):
    """
    Wilson-Cowan rate-model integration: identical to MPR except dr_m/dt
    follows the closed-form sigmoid (PRL eq. 39) instead of the (r, v)
    coupled ODEs.
    """
    if tau_x is None:
        tau_x = tau_s
    if tau_y is None:
        tau_y = tau_s
    if a_plus is None:
        a_plus = _default_a_plus
    if a_minus is None:
        a_minus = _default_a_minus

    M = len(eta_pop)
    n_rec = int(T / record_dt)
    t_eval = (np.arange(n_rec) + 0.5) * record_dt

    # Initial conditions: r at the stationary value at A_0 input level,
    # other variables zero. For simplicity just use r0 = small positive.
    r0 = 0.05 * np.ones(M)
    s0 = np.zeros(M); B0 = np.zeros(M)
    x0 = np.zeros(M); y0_tr = np.zeros(M)
    y0 = np.concatenate([r0, s0, B0, x0, y0_tr, A0_mf.ravel()])

    t0 = time.time()
    sol = solve_ivp(
        _wc_rhs, (0, T), y0, t_eval=t_eval, method=method,
        args=(M, eta_pop, delta_pop, w_pop, J, tau_s, tau_x, tau_y,
              a_plus, a_minus),
        rtol=rtol, atol=atol, max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"WC ODE failed: {sol.message}")
    if verbose:
        print(f"    WC done in {time.time() - t0:.1f}s "
              f"({sol.t.size} pts, {sol.nfev} evals)")

    r_per_pop = sol.y[0 * M : 1 * M, :]
    s_per_pop = sol.y[1 * M : 2 * M, :]
    A_traj = sol.y[5 * M:, :].reshape(M, M, -1)
    A_final = A_traj[:, :, -1]

    s_mean = (w_pop[:, None] * s_per_pop).sum(axis=0)
    r_mean = (w_pop[:, None] * r_per_pop).sum(axis=0)

    return dict(
        t_rec=sol.t, s_mean=s_mean, r_mean=r_mean,
        s_per_pop=s_per_pop, r_per_pop=r_per_pop,
        A_final=np.clip(A_final, 0.0, 1.0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Driver: sweep over (tau_s, M)
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# Per-cell visualization
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cell_comparison(cell):
    """
    Two-row figure for a single sweep cell:
        row 1: network-mean synaptic activation <s>(t) for QIF, MPR, WC,
                overlaid in distinct colors on the same axis.
        row 2: three final M x M coupling matrices (QIF block-averaged, MPR,
                WC) side by side, sharing a common color scale.

    Blocks (via plt.show()) until the user closes the figure. The figure is
    NOT saved -- this is intended for inline inspection during a sweep.

    Parameters
    ----------
    cell : dict
        One element of sweep_results, as returned by run_sweep. Expected keys:
        t_rec, s_mean_{qif,mpr,wc}, A_final_{qif_cg,mpr,wc}, tau_s, M.
    """
    tab10 = plt.get_cmap("tab10")
    c_qif, c_mpr, c_wc = tab10(0), tab10(1), tab10(2)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        width_ratios=[1.0, 1.0, 1.0, 0.06],
        height_ratios=[1.0, 1.2],
        hspace=0.35, wspace=0.30,
        left=0.07, right=0.94, top=0.92, bottom=0.09,
    )

    # ── Row 1: <s>(t) overlay ──────────────────────────────────────────────
    ax_s = fig.add_subplot(gs[0, :3])
    t = cell["t_rec"]
    ax_s.plot(t, cell["s_mean_qif"], color=c_qif, lw=1.4, label="QIF")
    ax_s.plot(t, cell["s_mean_mpr"], color=c_mpr, lw=1.4, ls="--", label="MPR-OA")
    ax_s.plot(t, cell["s_mean_wc"],  color=c_wc,  lw=1.4, ls=":",  label="WC")
    ax_s.set_xlabel(r"time $t$")
    ax_s.set_ylabel(r"$\langle s\rangle (t)$")
    ax_s.legend(loc="best", frameon=False)
    ax_s.grid(alpha=0.25)
    ax_s.set_title(
        fr"Sweep cell: $\tau_s={cell['tau_s']}$, $M={cell['M']}$"
    )

    # ── Row 2: final A matrices, shared color scale ─────────────────────────
    A_qif = cell["A_final_qif_cg"]
    A_mpr = cell["A_final_mpr"]
    A_wc  = cell["A_final_wc"]
    A_all = np.stack([A_qif, A_mpr, A_wc])
    vmin, vmax = float(A_all.min()), float(A_all.max())
    # Center the scale around 0.5 (the resting weight) if the data crosses it
    if vmin < 0.5 < vmax:
        half = max(0.5 - vmin, vmax - 0.5)
        vmin, vmax = 0.5 - half, 0.5 + half
        cmap = "RdBu_r"
    else:
        cmap = "viridis"

    titles = ["QIF (block-avg)", "MPR-OA", "WC"]
    mats   = [A_qif, A_mpr, A_wc]
    last_im = None
    for k, (T_, A_) in enumerate(zip(titles, mats)):
        ax = fig.add_subplot(gs[1, k])
        last_im = ax.imshow(A_, cmap=cmap, vmin=vmin, vmax=vmax,
                            interpolation="nearest", aspect="equal",
                            origin="upper")
        ax.set_title(T_)
        ax.set_xlabel(r"ensemble $n$ (pre)")
        if k == 0:
            ax.set_ylabel(r"ensemble $m$ (post)")
        M_ = A_.shape[0]
        tick_step = max(1, M_ // 6)
        ticks = np.arange(0, M_, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    cax = fig.add_subplot(gs[1, 3])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label(r"$A_{mn}$")

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Driver: sweep over (tau_s, M)
# ═══════════════════════════════════════════════════════════════════════════════
def run_sweep(
        tau_values, M_values, *,
        N=500, eta_min=0.5, eta_max=2.0,
        eta_kind="equispaced",
        tau_x=1.0, tau_y=1.0,
        J=2.0, A0_value=0.5,
        a_plus=None, a_minus=None,
        T=80.0, dt_qif=5e-4, V_peak=50.0,
        record_dt=0.1, plast_update_every=10,
        seed=42, verbose=True,
        plot_each=False,
):
    """
    Sweep over (tau_s, M) pairs running QIF, MPR-OA, and WC for each cell.
    Returns a dict of results keyed by (tau_s, M).

    Parameters
    ----------
    plot_each : bool
        If True, call `plot_cell_comparison(cell)` after each (tau_s, M)
        cell completes. This blocks the sweep via plt.show() until the
        user closes the figure window. The plot is NOT saved to file.
        Default False.
    """
    if a_plus is None:
        a_plus = _default_a_plus
    if a_minus is None:
        a_minus = _default_a_minus

    # The microscopic eta sample is fixed across the sweep (so all MF
    # comparisons see the same QIF reference).
    eta_micro = make_uniform_eta(N, eta_min, eta_max, kind=eta_kind, seed=seed)

    sweep_results = {}
    cell_idx = 0
    n_cells = len(tau_values) * len(M_values)

    for tau_s in tau_values:
        for M in M_values:
            cell_idx += 1
            print(f"\n[{cell_idx}/{n_cells}] tau_s={tau_s}, M={M}")

            # Ensemble parameters (Lorentzian mixture approximation of uniform)
            eta_pop, delta_pop, w_pop = make_lorentzian_mixture(M, eta_min, eta_max)
            labels = assign_to_ensembles_uniform(eta_micro, eta_min, eta_max, M)

            # Initial couplings (NxN for QIF, MxM for MF models)
            A0_micro = A0_value * np.ones((N, N))
            A0_mf    = A0_value * np.ones((M, M))

            # (1) QIF
            print("  Running QIF microscopic...")
            qif = simulate_qif(
                eta_micro, labels, A0_micro, tau_x=tau_x, tau_y=tau_y,
                J=J, tau_s=tau_s, a_plus=a_plus, a_minus=a_minus,
                T=T, dt=dt_qif, V_peak=V_peak,
                plast_update_every=plast_update_every,
                record_dt=record_dt,
                seed=seed, verbose=verbose,
            )

            # (2) MPR
            print("  Running MPR-OA mean-field...")
            mpr = simulate_mpr(
                eta_pop, delta_pop, w_pop, A0_mf, tau_x=tau_x, tau_y=tau_y,
                J=J, tau_s=tau_s, a_plus=a_plus, a_minus=a_minus,
                T=T, record_dt=record_dt, verbose=verbose,
            )

            # (3) WC
            print("  Running Wilson-Cowan rate...")
            wc = simulate_wc(
                eta_pop, delta_pop, w_pop, A0_mf, tau_x=tau_x, tau_y=tau_y,
                J=J, tau_s=tau_s, a_plus=a_plus, a_minus=a_minus,
                T=T, record_dt=record_dt, verbose=verbose,
            )

            # Coarse-grain QIF coupling
            A_qif_cg = block_average(qif["A_final_NxN"], labels, M)

            sweep_results[(tau_s, M)] = dict(
                tau_s=tau_s, M=M,
                eta_pop=eta_pop, delta_pop=delta_pop, w_pop=w_pop,
                labels=labels,
                # Final connectivity
                A_final_qif_cg=A_qif_cg,
                A_final_qif_NxN=qif["A_final_NxN"],
                A_final_mpr=mpr["A_final"],
                A_final_wc=wc["A_final"],
                # Common time grid
                t_rec=qif["t_rec"],
                # Network-mean synaptic activations
                s_mean_qif=qif["s_mean"],
                s_mean_mpr=mpr["s_mean"],
                s_mean_wc=wc["s_mean"],
                # Network-mean firing rates
                r_mean_qif=qif["r_mean"],
                r_mean_mpr=mpr["r_mean"],
                r_mean_wc=wc["r_mean"],
                # Per-population traces (in case they are useful later)
                s_per_pop_qif=qif["s_per_pop"],
                s_per_pop_mpr=mpr["s_per_pop"],
                s_per_pop_wc=wc["s_per_pop"],
                r_per_pop_qif=qif["r_per_pop"],
                r_per_pop_mpr=mpr["r_per_pop"],
                r_per_pop_wc=wc["r_per_pop"],
            )
            print(f"  Final s_mean (last 1s): "
                  f"QIF={qif['s_mean'][-10:].mean():.3f}, "
                  f"MPR={mpr['s_mean'][-10:].mean():.3f}, "
                  f"WC={wc['s_mean'][-10:].mean():.3f}")

            if plot_each:
                plot_cell_comparison(sweep_results[(tau_s, M)])

    return dict(
        config=dict(
            tau_values=list(tau_values), M_values=list(M_values),
            N=N, eta_min=eta_min, eta_max=eta_max, eta_kind=eta_kind,
            J=J, A0_value=A0_value, T=T, dt_qif=dt_qif, V_peak=V_peak,
            record_dt=record_dt, plast_update_every=plast_update_every,
            seed=seed,
        ),
        eta_micro=eta_micro,
        cells=sweep_results,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Sweep parameters
    tau_values = (0.1, 0.2, 0.4, 0.8, 1.6)        # synaptic time constants
    M_values   = (6, 12, 25, 50, 100)            # mean-field ensemble counts

    # Plasticity functions (bounded weights in [0, 1])
    a_plus  = lambda A: 0.1
    a_minus = lambda A: 0.1

    # Default sizing: N=300, T=40 chosen for tractable runtime in this sweep.
    # Each QIF cell takes ~20s at this size on a single core; for a heavier
    # comparison, increase N (e.g. 500-1000) and/or T (e.g. 80-160).
    out = run_sweep(
        tau_values=tau_values,
        M_values=M_values,
        N=500,
        eta_min=-1.0, eta_max=1.0,        # uniform eta in [0.5, 2.0]
        eta_kind="equispaced",
        tau_x=1.0, tau_y=1.0,
        J=5.0, A0_value=0.5,
        a_plus=a_plus, a_minus=a_minus,
        T=20.0, dt_qif=5e-4, V_peak=100.0,
        record_dt=0.1, plast_update_every=10,
        seed=42, verbose=True,
        plot_each=False,        # set False to disable per-cell plot
    )

    savepath = "/home/rgast/data/qif_plasticity/qif_mf_sweep.pkl"
    with open(savepath, "wb") as fh:
        pickle.dump(out, fh)
    print(f"\nSweep results saved to {savepath}")

    # Quick summary
    print(f"\nSummary: {len(out['cells'])} cells")
    for (tau_s, M), cell in out["cells"].items():
        print(f"  tau_s={tau_s:>5}, M={M:>3}: "
              f"A_qif_cg mean={cell['A_final_qif_cg'].mean():.4f}, "
              f"A_mpr mean={cell['A_final_mpr'].mean():.4f}, "
              f"A_wc  mean={cell['A_final_wc'].mean():.4f}, "
              f"s_qif={cell['s_mean_qif'][-10:].mean():.3f}, "
              f"s_mpr={cell['s_mean_mpr'][-10:].mean():.3f}, "
              f"s_wc={cell['s_mean_wc'][-10:].mean():.3f}")