#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Ensemble mean-field vs. microscopic network for theta-neurons with
phase-difference-dependent plasticity (PDDP).

This reproduces the scenario of Fig. 1 in
    Fennelly, Neff, Lambiotte, Keane & Byrne,
    "Mean-field approximation for networks with synchrony-driven adaptive
     coupling", Chaos 35, 013152 (2025),  arXiv:2407.21393
and extends their single global mean-field coupling (which corresponds to
M = 1 here) to the M-ensemble mean-field reduction of the accompanying
manuscript.

For a *fixed* mean drive eta0 the script
  (1) simulates ONCE the microscopic network of N theta-neurons with
      pairwise ("local") PDDP updates  (the ground-truth system);
  (2) for each user-specified number of ensembles M, integrates the
      ensemble mean-field equations and measures the mismatch with the
      microscopic network in
        * the STRUCTURAL coupling: how well the M-ensemble coupling matrix
          reconstructs the microscopic coupling matrix block-averaged to a
          fixed fine reference resolution (this is what the adaptive coupling
          makes M-dependent, even for a Lorentzian drive)
        * the GLOBAL coupling strength k_hat(t) (= filtered alpha R^2), a
          scalar that is tied to the synchrony R
  (3) produces a PRL single-column figure:
        (a) mismatch vs M: structural matrix RMSE and scalar k_hat RMSE
            (twin y-axes),
        (b) mean coupling strength k_hat(t): micro vs mean-field (M=1, M=16),
        (c) phase coherence R(t): micro vs mean-field (M=1, M=16),
        (d) coupling matrices: micro (block-averaged) vs mean-field (M=16).

----------------------------------------------------------------------------
Model (microscopic, theta-neuron form; spike when theta crosses pi)
    tau_m d theta_j/dt = (1-cos th_j) + (1+cos th_j)(eta_j + s_j v_syn)
                          - s_j sin th_j
    tau_s d x_j/dt      = -x_j + (1/N) sum_l sum_spk k_{lj} delta(t-t_l^spk)
    tau_s d s_j/dt      = -s_j + x_j                                  (alpha kernel)
    d k_{jl}/dt         = eps( -k_{jl} + alpha cos(theta_l - theta_j) )   (PDDP)
with eta_j ~ Lorentzian(eta0, Delta), v_syn the synaptic reversal potential.
The synaptic activation s_j is the convolution of the spike train with an
alpha kernel  g(t) = (t/tau_s^2) e^{-t/tau_s},  realised as the two coupled
first-order ODEs above (an intermediate "rise" variable x_j feeding s_j; equal
rise/decay time tau_s).  Setting s_j = x_j directly recovers the previous
mono-exponential kernel.

Ensemble mean-field (M Lorentzian ensembles; z_m = R_m e^{i psi_m}):
    tau_m dz_m/dt = -i (z_m-1)^2/2
                    + (z_m+1)^2/2 [ -Delta_m + i eta_m + i s_m v_syn ]
                    - (z_m^2-1)/2 s_m
    tau_s dx_m/dt = -x_m + sum_n kbar_{mn} w_n r_n,
                    r_n = (1/(pi tau_m)) (1-|z_n|^2) / |1+z_n|^2
    tau_s ds_m/dt = -s_m + x_m                                        (alpha kernel)
    d kbar_{mn}/dt = eps( -kbar_{mn} + alpha Re(z_n conj(z_m)) )           (PDDP)
The M=1 case recovers Eqs. (4)-(6) of Fennelly et al. exactly when the alpha
kernel is collapsed to the mono-exponential one.

Global observables compared between the two descriptions:
    mean coupling : micro  = (1/N^2) sum_{jl} k_{jl}
                    MF     = sum_{mn} w_m w_n kbar_{mn}
    synchrony     : micro  = | (1/N) sum_j e^{i theta_j} |
                    MF     = | sum_m w_m z_m |
----------------------------------------------------------------------------
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS  (edit here)
# =============================================================================
PARAMS = dict(
    # --- drive distribution rho(eta) ----------------------------------------
    #   Lorentzian drive: the *bare* phase dynamics are Ott-Antonsen-exact at
    #   M=1.  The adaptive (PDDP) coupling, however, builds drive-organised
    #   coupling structure that a single population cannot represent, so the
    #   M-ensemble reduction still improves the agreement -- this is the regime
    #   that exhibits that effect (a structured "spiral" state with appreciable
    #   heterogeneity).  Options: 'gaussian' | 'bimodal' | 'lorentzian'.
    dist="lorentzian",
    eta0=10.0,      # mean background drive (fixed)
    Delta=0.5,      # spread: half-width for 'lorentzian', sigma for 'gaussian'
    # --- neuron / synapse parameters (Fennelly et al. Fig. 1) ---------------
    v_syn=-8.0,
    tau_m=1.0,
    tau_s=1.0,
    # --- plasticity (PDDP) parameters ---------------------------------------
    alpha=2.0,      # plasticity strength / upper bound on coupling
    eps=0.1,        # plasticity rate
    k0=1.0,         # initial coupling strength (all pairs / ensembles)
    # --- simulation control --------------------------------------------------
    N=1000,         # number of neurons in the microscopic network
    T=120.0,        # total integration time
    dt=0.01,        # micro time step (Euler); MF uses adaptive RK45
    seed=1,         # RNG seed for the initial phases
    t_warmup=20.0,  # time discarded before computing the RMSE mismatch
)

# values of M to sweep over for the mismatch curve
M_SWEEP = [1, 5, 10, 25, 50]
# the two values of M whose full dynamics are shown in panels (b)-(d)
M_SHOW = (1, 50)
# fixed fine reference resolution for the *structural* coupling metric.  The
# block-averaged microscopic coupling matrix is computed once at this
# resolution; the M-ensemble mean field is then scored on how well it
# reconstructs that fixed reference.  Keeping the reference fixed (rather than
# re-blocking at each M) means finite-N noise does not grow with M, so the
# curve reflects genuine structural refinement.  Must exceed max(M_SWEEP).
M_REF = 100


# =============================================================================
# 2. DRIVE DISTRIBUTION AND ENSEMBLE CONSTRUCTION
#
#    The Ott-Antonsen ansatz makes a *single Lorentzian* drive M=1-exact, so a
#    single Lorentzian leaves no room for the ensemble reduction to improve.
#    The ensemble method targets the realistic case of heterogeneity that one
#    Lorentzian cannot represent (Gaussian, bimodal / multi-cluster, ...): one
#    then fits rho(eta) by a sum of M Lorentzians (the manuscript's "Cauchy
#    mix").  Here we build the M ensembles *data-driven*: the deterministically
#    sampled drives eta_j are sorted and split into M equal groups; ensemble m
#    inherits the m-th sub-population and is summarised by a Lorentzian whose
#    centre/half-width are the robust (median / half-IQR) statistics of the
#    group.  As M grows each group becomes more homogeneous (Delta_m -> 0) and
#    the reduction becomes exact; M=1 is the single-Lorentzian (Fennelly) limit.
# =============================================================================
def draw_eta(P):
    """Deterministic (quantile) sample of the drive distribution, length N."""
    N = P["N"]
    eta0, Delta = P["eta0"], P["Delta"]
    dist = P.get("dist", "gaussian")
    p = (np.arange(N) + 0.5) / N
    if dist == "lorentzian":
        eta = eta0 + Delta * np.tan(np.pi * (p - 0.5))
    elif dist == "gaussian":
        from scipy.special import ndtri          # standard-normal quantile
        eta = eta0 + Delta * ndtri(p)            # Delta plays the role of sigma
    elif dist == "bimodal":
        from scipy.special import ndtri
        sep = P.get("sep", 4.0)                  # half-distance between modes
        eta = np.empty(N)
        half = N // 2
        p1 = (np.arange(half) + 0.5) / half
        p2 = (np.arange(N - half) + 0.5) / (N - half)
        eta[:half] = (eta0 - sep) + Delta * ndtri(p1)
        eta[half:] = (eta0 + sep) + Delta * ndtri(p2)
    else:
        raise ValueError("unknown dist %r" % dist)
    return eta


def make_ensembles(eta_sorted, M):
    """Split the sorted drive sample into M equal groups -> Lorentzian params.

    Returns eta_m (centre), Delta_m (half-width), w_m (weight), groups (index
    arrays into the *sorted* sample).
    """
    groups = np.array_split(np.arange(eta_sorted.size), M)
    eta_m = np.empty(M)
    Delta_m = np.empty(M)
    w_m = np.empty(M)
    for m, g in enumerate(groups):
        vals = eta_sorted[g]
        eta_m[m] = np.median(vals)
        q25, q75 = np.percentile(vals, [25.0, 75.0])
        Delta_m[m] = 0.5 * (q75 - q25)
        w_m[m] = g.size / eta_sorted.size
    Delta_m = np.maximum(Delta_m, 1e-3)          # keep OA reduction regular
    return eta_m, Delta_m, w_m, groups


# =============================================================================
# 3. MICROSCOPIC NETWORK  (theta-neurons + pairwise PDDP)  -- run once
# =============================================================================
def simulate_micro(P, progress=False):
    from scipy.linalg.blas import dger  # in-place rank-1 update: A += a*x*y^T

    v_syn, tau_m, tau_s = P["v_syn"], P["tau_m"], P["tau_s"]
    alpha, eps, k0 = P["alpha"], P["eps"], P["k0"]
    N, T, dt = P["N"], P["T"], P["dt"]

    rng = np.random.default_rng(P["seed"])

    eta = np.sort(draw_eta(P))
    # rng.shuffle(eta)                      # de-correlate index from drive value

    theta = rng.uniform(0.0, 2.0 * np.pi, N)        # incoherent initial phases
    s = np.zeros(N)                                 # synaptic activation (alpha)
    x = np.zeros(N)                                 # intermediate "rise" variable
    K = np.asfortranarray(np.full((N, N), k0))      # K[l, j] = coupling l -> j

    nsteps = int(round(T / dt))
    t = np.arange(nsteps + 1) * dt
    R = np.empty(nsteps + 1)                        # synchrony |z|
    khat = np.empty(nsteps + 1)                     # global mean coupling
    R[0] = np.abs(np.exp(1j * theta).mean())
    khat[0] = K.mean()

    a = dt * eps                       # PDDP decay factor per step
    b = dt * eps * alpha               # PDDP drive factor per step
    inv_Ntau = 1.0 / (N * tau_s)
    decay_s = dt / tau_s
    two_pi = 2.0 * np.pi
    for it in range(1, nsteps + 1):
        c = np.cos(theta)
        sn = np.sin(theta)
        # --- theta-neuron drift -------------------------------------------
        dtheta = ((1.0 - c) + (1.0 + c) * (eta + s * v_syn) - s * sn) / tau_m
        theta_new = theta + dt * dtheta
        # --- spike detection: theta crossing pi (mod 2pi), upward ----------
        n_old = np.floor((theta - np.pi) / two_pi)
        n_new = np.floor((theta_new - np.pi) / two_pi)
        spikes = np.clip(n_new - n_old, 0.0, None)  # upward crossings only
        theta = theta_new
        # --- synaptic activation: alpha kernel as two coupled ODEs --------
        #     tau_s x' = -x + (spike kicks);   tau_s s' = -s + x
        s += decay_s * (x - s)                      # s integrates x (uses x(t))
        x -= decay_s * x                            # x decays
        idx = np.nonzero(spikes)[0]
        if idx.size:
            # spike kicks enter the rise variable x (input to neuron j)
            x += inv_Ntau * (spikes[idx] @ K[idx, :])
        # --- PDDP rank-2 update: cos(th_l - th_j) = c_j c_l + s_j s_l ------
        c = np.cos(theta)
        sn = np.sin(theta)
        K *= (1.0 - a)
        dger(b, c, c, a=K, overwrite_a=1)           # K += b * c c^T
        dger(b, sn, sn, a=K, overwrite_a=1)         # K += b * sn sn^T
        # --- observables ---------------------------------------------------
        R[it] = np.abs(np.exp(1j * theta).mean())
        khat[it] = K.mean()
        if progress and it % max(1, nsteps // 10) == 0:
            print("    micro %3d%%" % (100 * it // nsteps), flush=True)

    return dict(t=t, R=R, khat=khat, eta=eta, K_final=np.array(K))


# =============================================================================
# 4. ENSEMBLE MEAN-FIELD  (M ensembles)
# =============================================================================
def simulate_meanfield(P, M, z0_ensembles):
    v_syn, tau_m, tau_s = P["v_syn"], P["tau_m"], P["tau_s"]
    alpha, eps, k0 = P["alpha"], P["eps"], P["k0"]
    T = P["T"]

    eta_sorted = np.sort(draw_eta(P))
    eta_m, Delta_m, w_m, _ = make_ensembles(eta_sorted, M)

    # state vector: [Re z_m, Im z_m (M each), x_m (M), s_m (M), kbar_mn (M*M)]
    def pack(z, x, s, K):
        return np.concatenate([z.real, z.imag, x, s, K.reshape(-1)])

    def unpack(y):
        zr = y[:M]
        zi = y[M:2 * M]
        x = y[2 * M:3 * M]
        s = y[3 * M:4 * M]
        K = y[4 * M:4 * M + M * M].reshape(M, M)
        return zr + 1j * zi, x, s, K

    inv_pi_tau = 1.0 / (np.pi * tau_m)

    def rhs(t, y):
        z, x, s, K = unpack(y)
        # firing rate of each ensemble (Luke-Barreto-So / Montbrio formula)
        r = inv_pi_tau * (1.0 - np.abs(z) ** 2) / np.abs(1.0 + z) ** 2
        # z dynamics (Fennelly Eq. 4, per ensemble)
        dz = (-1j * (z - 1.0) ** 2 / 2.0
              + (z + 1.0) ** 2 / 2.0 * (-Delta_m + 1j * eta_m + 1j * s * v_syn)
              - (z ** 2 - 1.0) / 2.0 * s) / tau_m
        # alpha kernel as two coupled ODEs:
        #   tau_s x' = -x + sum_n kbar_{mn} w_n r_n;   tau_s s' = -s + x
        dx = (-x + K @ (w_m * r)) / tau_s
        ds = (-s + x) / tau_s
        # PDDP: d kbar_mn = eps(-kbar + alpha Re(z_n conj z_m))
        ReZ = np.real(np.outer(np.conj(z), z))       # [m,n] = Re(conj z_m z_n)
        dK = eps * (-K + alpha * ReZ)
        return pack(dz, dx, ds, dK)

    z0 = z0_ensembles
    x0 = np.zeros(M)
    s0 = np.zeros(M)
    K0 = np.full((M, M), k0)
    y0 = pack(z0, x0, s0, K0)

    t_eval = np.linspace(0.0, T, int(round(T / P["dt"])) + 1)  # match micro grid
    sol = solve_ivp(rhs, (0.0, T), y0, t_eval=t_eval,
                    method="RK45", rtol=1e-7, atol=1e-9, max_step=0.5)

    Zg = np.empty(sol.t.size, dtype=complex)
    khat = np.empty(sol.t.size)
    for k in range(sol.t.size):
        z, x, s, K = unpack(sol.y[:, k])
        Zg[k] = np.sum(w_m * z)
        khat[k] = np.sum(np.outer(w_m, w_m) * K)
    _, _, _, K_final = unpack(sol.y[:, -1])
    return dict(t=sol.t, R=np.abs(Zg), khat=khat,
                eta_m=eta_m, Delta_m=Delta_m, w_m=w_m,
                K_final=np.array(K_final))


def ensemble_z0_from_micro(P, M):
    """Matched initial ensemble order parameters z_m(0) from the micro IC.

    Reproduces the microscopic initial condition exactly.  simulate_micro uses
    eta = sort(draw_eta(P)) (no shuffle), so neuron j carries the j-th smallest
    drive and the same rng draw produces the same initial phases.  Neurons are
    therefore already ordered by drive, and the M ensembles are the M contiguous
    equal blocks used by make_ensembles().
    """
    rng = np.random.default_rng(P["seed"])
    _ = np.sort(draw_eta(P))                      # deterministic; consumes no rng
    theta0 = rng.uniform(0.0, 2.0 * np.pi, P["N"])
    z0 = np.empty(M, dtype=complex)
    for m, idx in enumerate(np.array_split(np.arange(P["N"]), M)):
        z0[m] = np.exp(1j * theta0[idx]).mean()
    return z0


# =============================================================================
# 5. MISMATCH METRICS
# =============================================================================
#  (A) Global coupling: RMSE between the network-average coupling strength
#      k_hat(t) of the micro network and of the mean field.  NOTE: averaging
#      the PDDP rule gives  d k_hat/dt = eps(-k_hat + alpha R^2)  exactly, so
#      k_hat is slaved to the synchrony R and is *blind* to coupling structure.
#  (A') Structural coupling: RMSE between the block-averaged microscopic
#      coupling matrix and the mean-field coupling matrix kbar_{mn}.  This sees
#      the between-ensemble structure that the adaptive coupling builds and that
#      the global average (A) discards; it is the quantity the M-ensemble
#      reduction actually improves.
#  (B) Phase coherence: RMSE between the amplitude spectra |FFT| of the micro
#      and mean-field synchrony signals R(t).
# =============================================================================
def micro_labels(P, M):
    """Per-neuron ensemble label (0..M-1), consistent with make_ensembles.

    simulate_micro orders neurons by drive (eta = sort(draw_eta(P))), so the M
    ensembles are the M contiguous equal blocks of neuron indices.
    """
    labels = np.empty(P["N"], dtype=int)
    for m, idx in enumerate(np.array_split(np.arange(P["N"]), M)):
        labels[idx] = m
    return labels


def block_average(A_fine, labels, M):
    """Block-average a full N x N matrix into an M x M between-group matrix.

    A_cg[I, J] = mean of A_fine over rows in group I and columns in group J.
    """
    N = A_fine.shape[0]
    B = np.zeros((N, M))
    B[np.arange(N), labels] = 1.0
    counts = B.sum(axis=0)
    S = B.T @ A_fine @ B
    return S / np.outer(counts, counts)


def coupling_matrix_rmse(K_micro_final, labels, M, K_mf_final):
    """RMSE between block-averaged micro coupling matrix and MF coupling matrix.

    Both M x M matrices are symmetric (cos-PDDP rule), and micro block I matches
    mean-field ensemble I by construction of `labels`.
    """
    A_cg = block_average(K_micro_final, labels, M)
    rmse = float(np.sqrt(np.mean((A_cg - K_mf_final) ** 2)))
    return rmse, A_cg


def timeseries_rmse(t_ref, x_ref, t_other, x_other, t_warmup):
    """RMSE between two scalar time series on the (uniform) micro grid.

    The second signal is resampled onto the reference grid and only samples
    past t_warmup are used.
    """
    x_on_ref = np.interp(t_ref, t_other, x_other)
    mask = t_ref >= t_warmup
    return float(np.sqrt(np.mean((x_on_ref[mask] - x_ref[mask]) ** 2)))


def coupling_reconstruction_rmse(P, K_mf_final, M, M_ref, A_ref):
    """Structural coupling error of the M-ensemble mean field against a FIXED
    fine reference.

    A_ref is the microscopic coupling matrix block-averaged once to M_ref groups
    (M_ref > M).  The M-ensemble mean-field matrix is expanded back onto the
    M_ref grid (each fine block inherits the coupling of the coarse ensemble it
    belongs to) and compared to A_ref.  Because the reference resolution is
    fixed, finite-N noise does not grow with M and the RMSE measures genuine
    structural refinement (it falls as M -> M_ref).
    """
    labels_M = micro_labels(P, M)
    e = np.empty(M_ref, dtype=int)
    for I, idx in enumerate(np.array_split(np.arange(P["N"]), M_ref)):
        e[I] = np.bincount(labels_M[idx], minlength=M).argmax()
    A_pred = K_mf_final[np.ix_(e, e)]
    return float(np.sqrt(np.mean((A_ref - A_pred) ** 2))), A_pred


def amplitude_spectrum(t_ref, x_ref, t_other, x_other, t_warmup):
    """Hann-windowed single-sided amplitude spectra of two signals.

    The mean-field signal is resampled onto the (uniform) micro time grid past
    t_warmup; the DC component is removed so the comparison is about the
    oscillatory content (frequency/amplitude), not the steady-state offset.

    Returns (freqs, A_ref, A_other).
    """
    mask = t_ref >= t_warmup
    tt = t_ref[mask]
    dt = tt[1] - tt[0]
    xr = x_ref[mask]
    xo = np.interp(tt, t_other, x_other)
    xr = xr - xr.mean()
    xo = xo - xo.mean()
    win = np.hanning(tt.size)
    norm = np.sum(win)
    A_ref = np.abs(np.fft.rfft(xr * win)) / norm
    A_oth = np.abs(np.fft.rfft(xo * win)) / norm
    freqs = np.fft.rfftfreq(tt.size, d=dt)
    return freqs, A_ref, A_oth


def spectral_rmse(t_ref, R_ref, t_mf, R_mf, t_warmup):
    """RMSE between the amplitude spectra of micro and mean-field R(t)."""
    freqs, A_ref, A_mf = amplitude_spectrum(t_ref, R_ref, t_mf, R_mf, t_warmup)
    return float(np.sqrt(np.mean((A_ref - A_mf) ** 2))), freqs, A_ref, A_mf


# =============================================================================
# 6. PRL-STYLE FIGURE  (single column)
# =============================================================================
def _set_prl_style():
    matplotlib.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "axes.unicode_minus": False,
        "savefig.dpi": 400,
    })


# colour scheme shared across panels
C_MICRO = "#1a1a1a"      # microscopic network (ground truth)
C_M1 = "#d1495b"         # single-population MF  (M = 1, Fennelly limit)
C_M10 = "#2e6f95"        # ensemble MF           (M = 10)
C_KHAT = "#c1121f"       # coupling-weight mismatch curve
C_SYNC = "#264653"       # synchrony mismatch curve


def make_figure(P, M_sweep, M_show, outdir="/mnt/user-data/outputs",
                M_ref=M_REF, layout_pad=None):
    """Build the PRL figure.

    layout_pad : dict or None
        Padding controls passed to matplotlib's constrained-layout engine
        (ConstrainedLayoutEngine.set).  Recognised keys:
            'w_pad'  padding around axes, horizontal  [inches]  (default 0.04)
            'h_pad'  padding around axes, vertical    [inches]  (default 0.04)
            'wspace' width  space between subplots  [fraction]  (default 0.03)
            'hspace' height space between subplots  [fraction]  (default 0.06)
        Increase these (e.g. larger 'w_pad') if any label is clipped.
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    _set_prl_style()

    pad = dict(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.06)
    if layout_pad:
        pad.update(layout_pad)

    warmup = P["t_warmup"]

    # --- ground-truth microscopic network (run once) -----------------------
    print("simulating microscopic network (N=%d, T=%g) ..." % (P["N"], P["T"]),
          flush=True)
    micro = simulate_micro(P, progress=True)

    # fixed fine reference for the structural coupling metric (computed once)
    A_ref = block_average(micro["K_final"], micro_labels(P, M_ref), M_ref)

    # --- sweep over M -------------------------------------------------------
    rmse_struct, rmse_khat = [], []     # structural matrix RMSE, scalar k_hat RMSE
    dyn = {}                            # cached mean-field dicts for M in M_show
    for M in M_sweep:
        z0 = ensemble_z0_from_micro(P, M)
        mf = simulate_meanfield(P, M, z0)
        rstr, _ = coupling_reconstruction_rmse(P, mf["K_final"], M, M_ref, A_ref)
        rk = timeseries_rmse(micro["t"], micro["khat"], mf["t"], mf["khat"],
                             warmup)
        rmse_struct.append(rstr)
        rmse_khat.append(rk)
        if M in M_show:
            dyn[M] = mf
        print("  M=%2d   RMSE_structural=%.4e   RMSE_khat=%.4e"
              % (M, rstr, rk), flush=True)
    rmse_struct = np.array(rmse_struct)
    rmse_khat = np.array(rmse_khat)

    for M in M_show:                # ensure M_show dynamics exist
        if M not in dyn:
            z0 = ensemble_z0_from_micro(P, M)
            dyn[M] = simulate_meanfield(P, M, z0)

    # =====================================================================
    # layout: PRL single column; 4 logical rows, constrained layout (no label
    # clipping).  Paddings are user-controllable via the `layout_pad` argument.
    #   (a) mismatch vs M: structural matrix RMSE & scalar k_hat (twin axes)
    #   (b) mean coupling strength k_hat(t)        (micro vs MF, M_show)
    #   (c) phase coherence R(t)                   (micro vs MF, M_show)
    #   (d) coupling matrices: micro block-avg vs MF (M_show[1])
    # =====================================================================
    fig = plt.figure(figsize=(3.5, 6.8), layout="constrained")
    fig.get_layout_engine().set(w_pad=pad["w_pad"], h_pad=pad["h_pad"],
                                wspace=pad["wspace"], hspace=pad["hspace"])
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.85, 0.85, 0.95])
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2])
    gsD = gs[3].subgridspec(1, 2, wspace=0.10)
    axD1 = fig.add_subplot(gsD[0])
    axD2 = fig.add_subplot(gsD[1])

    Mx = np.asarray(M_sweep)
    m0, m1 = M_show
    tw = warmup

    # ---- (a) mismatch vs M  (two separate y-scales via twinx) --------------
    #   left  : STRUCTURAL coupling-matrix RMSE (the adaptive-coupling effect)
    #   right : GLOBAL average coupling k_hat RMSE (= filtered alpha R^2)
    axA2 = axA.twinx()
    l1, = axA.plot(Mx, rmse_struct, "o-", color=C_KHAT, ms=3.5, mfc=C_KHAT,
                   mec=C_KHAT, lw=1.1, label=r"coupling structure $\bar{k}_{mn}$")
    l2, = axA2.plot(Mx, rmse_khat, "s-", color=C_SYNC, ms=3.2, mfc="white",
                    mec=C_SYNC, lw=1.1, label=r"global average $\hat{k}$")
    axA.set_yscale("log")
    axA2.set_yscale("log")
    axA.set_xlabel(r"number of ensembles $M$")
    axA.set_xlim(0, Mx[-1] + 1)
    axA.set_ylabel(r"structural mismatch", color=C_KHAT)
    axA2.set_ylabel(r"global $\hat{k}$ mismatch", color=C_SYNC)
    axA.tick_params(axis="y", colors=C_KHAT, which="both")
    axA2.tick_params(axis="y", colors=C_SYNC, which="both")
    axA.spines["left"].set_color(C_KHAT)
    axA2.spines["right"].set_color(C_SYNC)
    axA2.spines["left"].set_visible(False)
    axA.legend(handles=[l1, l2], loc="lower left",
               handlelength=1.6, borderaxespad=0.5)
    axA.annotate(r"(a)", xy=(0, 1), xycoords="axes fraction",
                 xytext=(-32, 4), textcoords="offset points",
                 fontsize=9, fontweight="bold", va="bottom")

    # ---- (b) mean coupling strength k_hat(t) ------------------------------
    axB.plot(micro["t"], micro["khat"], "-", color=C_MICRO, lw=1.2,
             label=r"network")
    axB.plot(dyn[m0]["t"], dyn[m0]["khat"], "--", color=C_M1, lw=1.1,
             label=r"MF $M=%d$" % m0)
    axB.plot(dyn[m1]["t"], dyn[m1]["khat"], "-", color=C_M10, lw=1.0,
             label=r"MF $M=%d$" % m1)
    axB.axvline(tw, color="0.6", ls=":", lw=0.7)
    axB.set_xlabel(r"time $t$")
    axB.set_ylabel(r"mean coupling $\hat{k}$")
    axB.set_xlim(0, P["T"])
    axB.legend(loc="upper right", handlelength=1.6, borderaxespad=0.3)
    axB.annotate(r"(b)", xy=(0, 1), xycoords="axes fraction",
                 xytext=(-32, 4), textcoords="offset points",
                 fontsize=9, fontweight="bold", va="bottom")

    # ---- (c) phase coherence time series R(t) -----------------------------
    axC.plot(micro["t"], micro["R"], "-", color=C_MICRO, lw=1.2,
             label=r"network")
    axC.plot(dyn[m0]["t"], dyn[m0]["R"], "--", color=C_M1, lw=1.1,
             label=r"MF $M=%d$" % m0)
    axC.plot(dyn[m1]["t"], dyn[m1]["R"], "-", color=C_M10, lw=1.0,
             label=r"MF $M=%d$" % m1)
    axC.axvline(tw, color="0.6", ls=":", lw=0.7)
    axC.set_xlabel(r"time $t$")
    axC.set_ylabel(r"phase coherence $R=|z|$")
    axC.set_xlim(0, P["T"])
    axC.legend(loc="upper right", handlelength=1.6, borderaxespad=0.3)
    axC.annotate(r"(c)", xy=(0, 1), xycoords="axes fraction",
                 xytext=(-32, 4), textcoords="offset points",
                 fontsize=9, fontweight="bold", va="bottom")

    # ---- (d) coupling matrices: micro block-avg vs MF at M=m1 -------------
    A_micro = block_average(micro["K_final"], micro_labels(P, m1), m1)
    A_mf = dyn[m1]["K_final"]
    vmin = min(A_micro.min(), A_mf.min())
    vmax = max(A_micro.max(), A_mf.max())
    imargs = dict(cmap="magma", vmin=vmin, vmax=vmax,
                  origin="upper", aspect="equal", interpolation="nearest")
    axD1.imshow(A_micro, **imargs)
    im = axD2.imshow(A_mf, **imargs)
    axD1.set_title(r"network (block-avg.)", fontsize=6.8, pad=2)
    axD2.set_title(r"mean-field", fontsize=6.8, pad=2)
    for ax in (axD1, axD2):
        ax.set_xticks([0, m1 - 1]); ax.set_yticks([0, m1 - 1])
        ax.set_xticklabels([1, m1]); ax.set_yticklabels([1, m1])
        ax.tick_params(length=2)
        ax.set_xlabel(r"ensemble $n$", fontsize=6.8, labelpad=1)
    axD1.set_ylabel(r"ensemble $m$", fontsize=6.8, labelpad=1)
    cb = fig.colorbar(im, ax=[axD1, axD2], location="right",
                      fraction=0.045, pad=0.03, aspect=18)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label(r"$\bar{k}_{mn}$", fontsize=7, labelpad=2)
    axD1.annotate(r"(d)", xy=(0, 1), xycoords="axes fraction",
                  xytext=(-32, 6), textcoords="offset points",
                  fontsize=9, fontweight="bold", va="bottom")
    axD1.annotate(r"$M=%d$" % m1, xy=(1.05, 1), xycoords="axes fraction",
                  xytext=(0, 6), textcoords="offset points",
                  fontsize=6.8, va="bottom", ha="center", color="#444444")

    png = os.path.join(outdir, "ensemble_mf_sweep.png")
    pdf = os.path.join(outdir, "ensemble_mf_sweep.pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print("saved:\n  %s\n  %s" % (png, pdf), flush=True)
    return micro, dyn, rmse_struct, rmse_khat


if __name__ == "__main__":
    make_figure(PARAMS, M_SWEEP, M_SHOW, outdir="/home/rgast/data/qif_plasticity")