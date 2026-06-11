#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Microscopic theta-neuron network vs. Ott-Antonsen (complex-Z) ensemble
mean-field, swept over the number of populations M.

Model (from theta_comparison_simulation2.py)
--------------------------------------------
Micro (N neurons):
    dtheta_k/dt = (1 - cos th_k) + (1 + cos th_k)[ eta_k + (J/N) sum_l A_kl s(n,th_l) ]
    dA_kl/dt    = mu f(th_l - th_k) - gamma A_kl
    s(n,theta)  = c_n (1 - cos theta)^n ,   c_n = 2^n (n!)^2 / (2n)!
Macro (M = N/d populations, complex Z_m):
    Zdot_m      = -1/2 [ (Delta_m - i E_m)(1+Z_m)^2 + i (1-Z_m)^2 ],  E_m = eta_m + (J/M) sum_n A_mn <s>_{Z_n}
    Adot_mn     = mu |Z_m||Z_n| f(arg Z_n - arg Z_m) - gamma A_mn
with <s>_Z = c_n Re[ sum_p shat[p] Z^p ] on the OA manifold.

One microscopic network is fixed (drives = sorted quantile sample of a single
Lorentzian, fixed phases/coupling).  For each M the Lorentzian is split into M
analytic sub-Lorentzians (population centres/widths), neurons are grouped by
drive rank into M equal blocks, and the OA reduction is integrated and compared
to the same micro.
"""

import numpy as np
from scipy.integrate import solve_ivp
from math import factorial, comb
import matplotlib

try:                                   # optional acceleration
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_EPS = 1e-12

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = dict(
    N=500,
    T=150.0,
    J=-8.0,
    mu=0.02,
    gamma=0.002,
    eta0=5.0,
    Delta0=2.0,
    n_pulse=10,
    plasticity="antihebbian",   # 'hebbian' (cos) or 'antihebbian' (sin)
    seed=42,
    n_save=600,                 # time points stored (micro & OA share this grid)
    t_warmup=0.0,              # discarded before spectral RMSE of R(t)
    rtol=1e-6, atol=1e-8, method="RK45",
    use_numba=False,            # set True to use the numba-accelerated micro RHS
    matrix_metric="correlation",    # coupling comparison: 'fourier' (RMSE) | 'correlation'
)
M_SWEEP = [1, 2, 5, 10, 25, 50, 100]   # must divide N
M_SHOW = (5, 50)                       # two M for rows 2-3 (d=60 vs d=20)

# ---- figure style (single-column PRL) --------------------------------------
# All font sizes are collected here so they can be tuned in one place.
FONTS = dict(
    tick=7,       # tick-label size (and y-axis offset "x10^-1" text)
    label=8,      # x/y axis-label size
    legend=7,     # legend text size
    panel=9,      # panel letters (a), (b), ...
    colhdr=8,     # matrix column headers ("network" / "OA")
    cbar=6,       # colour-bar tick + label size
)
# PRL single-column width is 3.375 in (8.6 cm); height set for the 4 panel rows
# (top two rows reduced to ~2/3 height; matrix rows keep their size).
FIG_SIZE = (3.375, 5.4)


# =============================================================================
# Model primitives (from the attached script)
# =============================================================================
def coupling_norm(n):
    return (2 ** n * factorial(n) ** 2) / factorial(2 * n)


def fourier_coeffs_s(n):
    """Real cosine Fourier coefficients of (1-cos th)^n, length n+1."""
    c = np.zeros(2 * n + 1, dtype=complex)
    for k in range(n + 1):
        binom_nk = comb(n, k) * (-1) ** k / 2 ** k
        for j in range(k + 1):
            p = 2 * j - k
            c[p + n] += binom_nk * comb(k, j)
    s_hat = np.zeros(n + 1)
    s_hat[0] = c[n].real
    for p in range(1, n + 1):
        s_hat[p] = 2.0 * c[p + n].real
    return s_hat


def oa_synaptic_mean_complex(n_pulse, Z, s_hat, cn):
    S = np.full(len(Z), s_hat[0], dtype=complex)
    Zp = Z.copy()
    for p in range(1, n_pulse + 1):
        S += s_hat[p] * Zp
        Zp *= Z
    return cn * S.real


def s_micro(n_pulse, theta, cn):
    return cn * (1.0 - np.cos(theta)) ** n_pulse


def hebbian(x):
    return np.cos(x)


def antihebbian(x):
    return np.sin(x)


# =============================================================================
# Drive distribution: fixed micro sample + analytic M-population decomposition
# =============================================================================
def lorentzian_quantiles(n, x0, gamma):
    """Deterministic quantile sample of a Lorentzian(x0, gamma), length n."""
    p = (np.arange(n) + 0.5) / n
    return x0 + gamma * np.tan(np.pi * (p - 0.5))


def lorentzian_populations(eta0, Delta0, M):
    """Analytic split of Lorentzian(eta0,Delta0) into M sub-Lorentzians."""
    n = np.arange(1, M + 1)
    eta_pop = eta0 + Delta0 * np.tan(0.5 * np.pi * (2 * n - M - 1) / (M + 1))
    delta_pop = Delta0 * (np.tan(0.5 * np.pi * (2 * n - M - 0.5) / (M + 1))
                          - np.tan(0.5 * np.pi * (2 * n - M - 1.5) / (M + 1)))
    return eta_pop, delta_pop


def make_micro_ic(P):
    """Fixed microscopic drives (sorted) and initial phases."""
    rng = np.random.default_rng(P["seed"])
    eta = np.sort(lorentzian_quantiles(P["N"], P["eta0"], P["Delta0"]))
    theta0 = rng.uniform(-np.pi, np.pi, P["N"])
    return eta, theta0


# =============================================================================
# Microscopic and macroscopic ODEs
# =============================================================================
def tn_ode(t, y, eta, J, mu, gamma, n_pulse, cn, anti):
    """Microscopic RHS (optimized, pure NumPy).

    The plasticity term avoids evaluating sin/cos of the full N x N phase-
    difference matrix.  Using the angle-subtraction identities with the
    per-neuron vectors c=cos(theta), s=sin(theta):
        antihebbian:  sin(theta_l - theta_k) = c_k s_l - s_k c_l  ->  c (x) s - s (x) c
        hebbian:      cos(theta_l - theta_k) = c_k c_l + s_k s_l  ->  c (x) c + s (x) s
    so f(diff) becomes two rank-1 outer products (no N^2 transcendentals).
    The result is written straight into a freshly allocated output vector
    (solve_ivp keeps references to the returned array between RK stages, so it
    must not be a reused buffer)."""
    N = eta.shape[0]
    theta = y[:N]
    A = y[N:].reshape(N, N)
    c = np.cos(theta)
    s = np.sin(theta)
    s_vec = cn * (1.0 - c) ** n_pulse
    dydt = np.empty(N + N * N)
    dydt[:N] = (1.0 - c) + (1.0 + c) * (eta + (J / N) * (A @ s_vec))
    dA = dydt[N:].reshape(N, N)
    if anti:
        np.outer(c, s, out=dA); dA -= np.outer(s, c)
    else:
        np.outer(c, c, out=dA); dA += np.outer(s, s)
    dA *= mu
    dA -= gamma * A
    np.fill_diagonal(dA, 0.0)
    return dydt


if _HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _tn_rhs_numba(y, eta, J, mu, gamma, n_pulse, cn, anti):
        N = eta.shape[0]
        c = np.empty(N); s = np.empty(N); sv = np.empty(N)
        for i in range(N):
            ci = np.cos(y[i]); si = np.sin(y[i])
            c[i] = ci; s[i] = si
            sv[i] = cn * (1.0 - ci) ** n_pulse
        dydt = np.empty(N + N * N)
        JN = J / N
        for k in prange(N):                # rows are independent -> parallel
            base = N + k * N
            acc = 0.0
            for l in range(N):
                acc += y[base + l] * sv[l]
            ck = c[k]; sk = s[k]
            dydt[k] = (1.0 - ck) + (1.0 + ck) * (eta[k] + JN * acc)
            for l in range(N):
                if anti:
                    fval = ck * s[l] - sk * c[l]
                else:
                    fval = ck * c[l] + sk * s[l]
                dydt[base + l] = 0.0 if k == l else (mu * fval - gamma * y[base + l])
        return dydt

    def tn_ode_numba(t, y, eta, J, mu, gamma, n_pulse, cn, anti):
        return _tn_rhs_numba(y, eta, J, mu, gamma, n_pulse, cn, anti)


def oa_ode_complex(t, y, eta_pop, delta_pop, J, mu, gamma, n_pulse, s_hat, cn, f):
    M = len(eta_pop)
    Z = y[:M] + 1j * y[M:2 * M]
    A = y[2 * M:].reshape(M, M)
    absZ = np.abs(Z)
    Z = np.where(absZ > 1.0 - _EPS, Z / np.maximum(absZ, _EPS) * (1.0 - _EPS), Z)
    S_vec = oa_synaptic_mean_complex(n_pulse, Z, s_hat, cn)
    E = eta_pop + (J / M) * (A @ S_vec)
    coeff = delta_pop - 1j * E
    Zdot = -0.5 * (coeff * (1.0 + Z) ** 2 + 1j * (1.0 - Z) ** 2)
    R = np.abs(Z)
    Psi = np.angle(Z)
    dPsi = Psi[np.newaxis, :] - Psi[:, np.newaxis]
    rr = R[:, np.newaxis] * R[np.newaxis, :]
    dA = mu * rr * f(dPsi) - gamma * A
    return np.concatenate([Zdot.real, Zdot.imag, dA.ravel()])


def coarse_grain(A_fine, M):
    """Block-average an N x N matrix into M x M (contiguous equal blocks)."""
    N = A_fine.shape[0]
    d = N // M
    return A_fine.reshape(M, d, M, d).mean(axis=(1, 3))


def expand_mf(A_oa, d):
    """Expand an M x M mean-field matrix to N x N (N = M*d) with block-constant
    entries: every microscopic pair (k, l) inherits the coupling of the
    population pair it belongs to.  This lets the coarse mean-field matrix be
    compared directly against the *full* microscopic coupling matrix."""
    return np.kron(A_oa, np.ones((d, d)))


# =============================================================================
# Simulations
# =============================================================================
def run_micro(P, eta, theta0):
    cn = coupling_norm(P["n_pulse"])
    anti = (P["plasticity"] == "antihebbian")
    N = P["N"]
    rhs = tn_ode
    if P.get("use_numba", False):
        if _HAVE_NUMBA:
            rhs = tn_ode_numba
        else:
            print("  [use_numba requested but numba not installed -> NumPy RHS]",
                  flush=True)
    t_eval = np.linspace(0.0, P["T"], P["n_save"])
    y0 = np.concatenate([theta0, np.ones(N * N)])
    sol = solve_ivp(rhs, (0.0, P["T"]), y0, method=P["method"],
                    args=(eta, P["J"], P["mu"], P["gamma"], P["n_pulse"], cn, anti),
                    rtol=P["rtol"], atol=P["atol"], t_eval=t_eval)
    if not sol.success:
        raise RuntimeError("micro failed: " + sol.message)
    theta = sol.y[:N]
    R = np.abs(np.mean(np.exp(1j * theta), axis=0))
    A_final = sol.y[N:, -1].reshape(N, N)
    return dict(t=sol.t, R=R, A_final=A_final)


def run_oa(P, M, eta, theta0):
    cn = coupling_norm(P["n_pulse"])
    s_hat = fourier_coeffs_s(P["n_pulse"])
    f = antihebbian if P["plasticity"] == "antihebbian" else hebbian
    N = P["N"]
    d = N // M
    eta_pop, delta_pop = lorentzian_populations(P["eta0"], P["Delta0"], M)
    # matched initial Z from the (drive-sorted) micro blocks
    Z0 = np.array([np.mean(np.exp(1j * theta0[m * d:(m + 1) * d]))
                   for m in range(M)])
    t_eval = np.linspace(0.0, P["T"], P["n_save"])
    y0 = np.concatenate([Z0.real, Z0.imag, np.ones(M * M)])
    sol = solve_ivp(oa_ode_complex, (0.0, P["T"]), y0, method=P["method"],
                    args=(eta_pop, delta_pop, P["J"], P["mu"], P["gamma"],
                          P["n_pulse"], s_hat, cn, f),
                    rtol=P["rtol"], atol=P["atol"], t_eval=t_eval)
    if not sol.success:
        raise RuntimeError("OA failed (M=%d): %s" % (M, sol.message))
    Z = sol.y[:M] + 1j * sol.y[M:2 * M]
    R = np.abs(np.mean(Z, axis=0))
    A_final = sol.y[2 * M:, -1].reshape(M, M)
    return dict(t=sol.t, R=R, A_final=A_final)


# =============================================================================
# Mismatch metrics
# =============================================================================
def matrix_compare(A_micro_full, A_mf_expanded, metric):
    """Compare the full microscopic coupling matrix with the block-constant
    expansion of the mean-field matrix (both N x N).

    metric = "fourier"      -> RMSE between the energy-normalised 2D spatial-
                               frequency magnitude spectra (phase-insensitive;
                               low bins test the coarse inter-population
                               structure, high bins the within-population
                               fluctuations the mean field cannot represent).
    metric = "correlation"  -> Pearson correlation coefficient between the two
                               matrices (1 = perfect agreement; insensitive to
                               an overall scale/offset)."""
    if metric == "correlation":
        return float(np.corrcoef(A_micro_full.ravel(),
                                 A_mf_expanded.ravel())[0, 1])
    if metric == "fourier":
        Fm = np.abs(np.fft.fft2(A_micro_full, norm="ortho"))
        Fo = np.abs(np.fft.fft2(A_mf_expanded, norm="ortho"))
        return float(np.sqrt(np.mean((Fm - Fo) ** 2)))
    raise ValueError("matrix_metric must be 'fourier' or 'correlation'")


def coherence_spectrum_rmse(t, R_micro, R_oa, t_warmup):
    """RMSE between Hann-windowed amplitude spectra of R(t).

    The DC component is retained (not mean-subtracted): the dominant model
    mismatch here is the steady coherence *level*, which the full Fourier
    decomposition of the time series should capture, alongside the oscillatory
    content at higher frequencies.
    """
    mask = t >= t_warmup
    tt = t[mask]
    xm = R_micro[mask]
    xo = R_oa[mask]
    w = np.hanning(tt.size)
    nrm = w.sum()
    Am = np.abs(np.fft.rfft(xm * w)) / nrm
    Ao = np.abs(np.fft.rfft(xo * w)) / nrm
    return float(np.sqrt(np.mean((Am - Ao) ** 2)))


# =============================================================================
# Driver + figure
# =============================================================================
def make_figure(P, M_sweep, M_show, outdir="/mnt/user-data/outputs",
                fonts=FONTS, fig_size=FIG_SIZE):
    import os
    os.makedirs(outdir, exist_ok=True)

    eta, theta0 = make_micro_ic(P)
    print("running micro (N=%d, T=%g) ..." % (P["N"], P["T"]), flush=True)
    micro = run_micro(P, eta, theta0)

    mat_metric, rmse_coh = [], []
    oa_cache = {}
    for M in M_sweep:
        oa = run_oa(P, M, eta, theta0)
        A_exp = expand_mf(oa["A_final"], P["N"] // M)        # MF -> full N x N
        mat_metric.append(matrix_compare(micro["A_final"], A_exp,
                                         P["matrix_metric"]))
        rmse_coh.append(coherence_spectrum_rmse(micro["t"], micro["R"], oa["R"],
                                                P["t_warmup"]))
        if M in M_show:
            oa_cache[M] = (oa, coarse_grain(micro["A_final"], M))
        print("  M=%2d  matrix(%s)=%.4e  coherence_spectrum_RMSE=%.4e"
              % (M, P["matrix_metric"], mat_metric[-1], rmse_coh[-1]), flush=True)
    for M in M_show:
        if M not in oa_cache:
            oa = run_oa(P, M, eta, theta0)
            oa_cache[M] = (oa, coarse_grain(micro["A_final"], M))

    # ---------------- figure (single-column PRL) ----------------
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "font.size": fonts["tick"],
        "axes.titlepad": 3.0,
    })

    def panel_letter(ax, s):
        ax.annotate(s, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-22, 3), textcoords="offset points",
                    fontsize=fonts["panel"], fontweight="bold",
                    va="bottom", ha="left", annotation_clip=False)

    def sci_yaxis(ax):
        """Linear y-axis with the common power shown once at the top."""
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                            useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(fonts["tick"])

    fig = plt.figure(figsize=fig_size, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.015,
                                    wspace=0.02, hspace=0.03)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.667, 0.667, 0.92, 0.92])

    Mx = np.asarray(M_sweep)
    m0, m1 = M_show
    C_MIC, C0, C1 = "#1a1a1a", "#d1495b", "#2e6f95"

    # ---- row 0: (a) coupling- & coherence-spectrum RMSE vs M (twin axes) ----
    axA = fig.add_subplot(gs[0])
    axR = axA.twinx()
    cL, cR = "#c1121f", "#264653"
    axA.plot(Mx, mat_metric, "o-", color=cL, ms=3.2, lw=1.0)
    axR.plot(Mx, rmse_coh, "s-", color=cR, ms=3.0, lw=1.0, mfc="white")
    axA.set_xlabel(r"ensembles $M$", fontsize=fonts["label"])
    if P["matrix_metric"] == "correlation":
        axA.set_ylabel(r"corr$(A_{ij})$", fontsize=fonts["label"], color=cL)
    else:
        axA.set_ylabel(r"RMSE$(A_{ij})$", fontsize=fonts["label"], color=cL)
    axR.set_ylabel(r"RMSE$(R(t))$", fontsize=fonts["label"], color=cR)
    axA.tick_params(axis="x", labelsize=fonts["tick"])
    axA.tick_params(axis="y", labelsize=fonts["tick"], colors=cL)
    axR.tick_params(axis="y", labelsize=fonts["tick"], colors=cR)
    if P["matrix_metric"] != "correlation":   # sci offset only for RMSE-scale data
        sci_yaxis(axA)
        axA.yaxis.get_offset_text().set_color(cL)
    sci_yaxis(axR)
    axR.yaxis.get_offset_text().set_color(cR)
    axA.spines["left"].set_color(cL)
    axA.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.spines["right"].set_color(cR)
    panel_letter(axA, "(a)")

    # ---- row 1: (c) phase-coherence dynamics --------------------------------
    axC = fig.add_subplot(gs[1])
    axC.plot(micro["t"], micro["R"], "-", color=C_MIC, lw=1.1, label="network")
    axC.plot(oa_cache[m0][0]["t"], oa_cache[m0][0]["R"], "--", color=C0, lw=1.0,
             label=r"OA $M{=}%d$" % m0)
    axC.plot(oa_cache[m1][0]["t"], oa_cache[m1][0]["R"], "-", color=C1, lw=0.9,
             label=r"OA $M{=}%d$" % m1)
    axC.axvline(P["t_warmup"], color="0.6", ls=":", lw=0.6)
    axC.set_xlabel(r"time $t$", fontsize=fonts["label"])
    axC.set_ylabel(r"$R(t)$", fontsize=fonts["label"])
    axC.set_xlim(0, P["T"])
    axC.tick_params(labelsize=fonts["tick"])
    axC.legend(ncol=1, fontsize=fonts["legend"], loc="lower left",
               handlelength=1.5, borderaxespad=0.4, labelspacing=0.3,
               framealpha=0.85)
    panel_letter(axC, "(b)")

    # ---- rows 2-3: final coupling matrices (network vs OA) for the two M ----
    def matrix_row(gs_row, M, letter, top):
        sub = gs_row.subgridspec(1, 3, width_ratios=[1, 1, 0.075], wspace=0.10)
        A_net = coarse_grain(micro["A_final"], M)
        A_oa = oa_cache[M][0]["A_final"]
        v = max(np.abs(A_net).max(), np.abs(A_oa).max()) or 1.0
        ima = dict(cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
                   aspect="auto", origin="upper")
        ticks = [0, M - 1]
        axn = fig.add_subplot(sub[0])
        axn.imshow(A_net, **ima)
        axn.set_ylabel(r"$M{=}%d$" % M + "\n" + r"ensemble $m$", fontsize=fonts["label"])
        axn.set_xticks(ticks); axn.set_yticks(ticks)
        axn.tick_params(labelsize=fonts["cbar"], length=2)
        axo = fig.add_subplot(sub[1])
        im = axo.imshow(A_oa, **ima)
        axo.set_xticks(ticks); axo.set_yticks(ticks)
        axo.tick_params(labelsize=fonts["cbar"], length=2, labelleft=False)
        if top:
            axn.set_title(r"network", fontsize=fonts["colhdr"])
            axo.set_title(r"OA", fontsize=fonts["colhdr"])
        else:
            axn.set_xlabel(r"ensemble $n$", fontsize=fonts["label"])
            axo.set_xlabel(r"ensemble $n$", fontsize=fonts["label"])
        cax = fig.add_subplot(sub[2])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=fonts["cbar"], length=2)
        cb.set_label(r"$A_{mn}$", fontsize=fonts["cbar"])
        panel_letter(axn, letter)

    matrix_row(gs[2], m0, "(c)", top=True)
    matrix_row(gs[3], m1, "(d)", top=False)

    png = os.path.join(outdir, "theta_M_comparison.png")
    pdf = os.path.join(outdir, "theta_M_comparison.pdf")
    fig.savefig(png, dpi=400)
    fig.savefig(pdf)
    plt.close(fig)
    print("saved:", png, flush=True)
    return micro, oa_cache, np.array(mat_metric), np.array(rmse_coh)


if __name__ == "__main__":
    make_figure(CONFIG, M_SWEEP, M_SHOW, outdir="/home/rgast/data/qif_plasticity")