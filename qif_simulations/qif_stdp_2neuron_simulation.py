"""
Two-neuron QIF network with alpha-kernel synapse and trace-based Hebbian STDP.

Setup: neuron 1 projects synaptically to neuron 2 (no other connections).
    dV_1/dt = V_1^2 + eta_1
    dV_2/dt = V_2^2 + eta_2 + J * A_21 * s_1
With per-neuron alpha-kernel synapses (lab report eq.; PRL eq. 26 cast as a
2nd-order ODE):
    tau_s ds_i/dt = B_i
    tau_s dB_i/dt = -2 B_i - s_i + alpha_spike * sum_k delta(t - t_i^(k))
And pre/post trace variables (PRL eq. 30-31):
    tau_x dx_i/dt = -x_i + s_i
    tau_y dy_i/dt = -y_i + s_i
Hebbian plasticity for A_21 (PRL eq. 32, post=2, pre=1):
    dA_21/dt = a_+(A_21) * x_1 * s_2  -  a_-(A_21) * y_2 * s_1
with bounded a_+(A) = mu (1 - A), a_-(A) = mu A.

We sweep tau_s in {0.1, 0.4, 1.6} and produce a 3 x 1 figure:
    row 1 : spike times of neuron 1 (dashed verticals) + s_1(t) per tau_s
    row 2 : spike times of neuron 2 (dashed verticals, color-coded by tau_s)
            + LTP term x_1(t) s_2(t) (normalized per tau_s)
    row 3 : A_21(t) for each tau_s
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_two_qif(eta1, eta2, tau_s, *,
                      J=2.0, A0=0.5,
                      tau_x=None, tau_y=None,
                      a_plus=None, a_minus=None,
                      alpha_spike=1.0,
                      T=80.0, dt=1e-4, V_peak=30.0,
                      seed=0):
    """
    Integrate the 2-neuron QIF + alpha-kernel + trace + Hebbian-STDP system
    by forward Euler.

    Returns a dict of time-series arrays sampled on a uniform grid (dt).
    """
    if tau_x is None:
        tau_x = tau_s
    if tau_y is None:
        tau_y = tau_s
    if a_plus is None:
        a_plus = lambda A: 0.02 * (1.0 - A)
    if a_minus is None:
        a_minus = lambda A: 0.02 * A

    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt

    # State (we record everything at every step since N=2; cheap)
    V = np.array([-1.0, -1.0]) + 0.05 * rng.normal(size=2)
    s = np.zeros(2)
    B = np.zeros(2)
    x = np.zeros(2)
    y = np.zeros(2)
    A_21 = float(A0)
    eta = np.array([eta1, eta2])

    V_hist = np.zeros((n_steps, 2))
    s_hist = np.zeros((n_steps, 2))
    x_hist = np.zeros((n_steps, 2))
    y_hist = np.zeros((n_steps, 2))
    A_hist = np.zeros(n_steps)
    spike_mask = np.zeros((n_steps, 2), dtype=bool)

    spike_bump = alpha_spike / tau_s
    inv_ts = 1.0 / tau_s
    inv_tx = 1.0 / tau_x
    inv_ty = 1.0 / tau_y

    for k in range(n_steps):
        V_hist[k] = V
        s_hist[k] = s
        x_hist[k] = x
        y_hist[k] = y
        A_hist[k] = A_21

        # Synaptic input: only neuron 2 receives, from neuron 1
        I = np.array([0.0, J * A_21 * s[0]])

        # Update V (forward Euler)
        V = V + dt * (V * V + eta + I)

        # Update alpha-kernel synapse, traces (continuous parts)
        s = s + dt * (B * inv_ts)
        B = B + dt * ((-2.0 * B - s) * inv_ts)
        x = x + dt * ((-x + s) * inv_tx)
        y = y + dt * ((-y + s) * inv_ty)

        # Spike detection + reset
        spiked = V >= V_peak
        if spiked.any():
            V[spiked] = -V_peak
            B[spiked] += spike_bump
            spike_mask[k, spiked] = True

        # Plasticity (Hebbian for A_21):
        # dA_21 = a_+(A_21) * x_1 * s_2  -  a_-(A_21) * y_2 * s_1
        Ac = np.clip(A_21, 0.0, 1.0)
        dA = a_plus(Ac) * x[0] * s[1] - a_minus(Ac) * y[1] * s[0]
        A_21 = float(np.clip(A_21 + dt * dA, 0.0, 1.0))

    return dict(
        t=t, V=V_hist, s=s_hist, x=x_hist, y=y_hist,
        A_21=A_hist, spike_mask=spike_mask,
        eta1=eta1, eta2=eta2, tau_s=tau_s, J=J, A0=A0, V_peak=V_peak,
    )


def _blank_at_resets(t, V, spike_mask, V_peak):
    """
    Replace the reset transition (V_peak -> -V_peak) with NaNs so that
    matplotlib doesn't draw a vertical line across the plot for each spike.
    Returns a copy of V with NaN inserted at the spike time-steps.

    Also drives the V at the spike time-step up to V_peak (visual "spike"),
    then NaN, then resume at -V_peak.
    """
    Vp = V.copy()
    Vp[spike_mask] = np.nan
    return Vp


def plot_two_qif(results, savepath="two_qif_stdp.png", t_window=None,
                  fontsize=10, vline_lw=0.7, trace_lw=1.2, dpi=150):
    """
    Build the 3 x 1 figure for a list of `results` dicts (one per tau_s).
        row 1: spike times of neuron 1 (dashed vertical lines, neutral color,
               same across all tau_s since V_1 has no input), with s_1(t)
               overlaid for each tau_s in distinct tab10 colors.
        row 2: spike times of neuron 2 (dashed vertical lines, colored per
               tau_s), with x_1(t) s_2(t) overlaid in the same colors --
               the LTP term in the Hebbian rule dA_21 = a_+ x_1 s_2 - a_- y_2 s_1.
        row 3: A_21(t) for each tau_s.

    Parameters
    ----------
    fontsize : float
        Base font size for axes labels, tick labels, legend, and title.
    vline_lw : float
        Line width for the dashed vertical spike-time lines.
    trace_lw : float
        Line width for the solid time-series traces (s_1, x_1 s_2, A_21).
        Row-3 (A_21) traces use 1.2 * trace_lw for slightly heavier weight.
    dpi : int
        DPI for the saved figure.
    """
    # tab10 gives distinct categorical colors for small N
    tab10 = plt.get_cmap("tab10")
    tau_values = [r["tau_s"] for r in results]
    n_tau = len(tau_values)
    tau_colors = [tab10(i) for i in range(n_tau)]

    # Local rc context so font-size changes don't leak to other figures
    rc = {
        "font.size":       fontsize,
        "axes.labelsize":  fontsize,
        "axes.titlesize":  fontsize,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "legend.fontsize": fontsize,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)

        if t_window is None:
            t_lo, t_hi = 0.0, results[0]["t"][-1]
        else:
            t_lo, t_hi = t_window

        r0 = results[0]
        t = r0["t"]
        mask_t = (t >= t_lo) & (t <= t_hi)

        # ── Row 1: neuron-1 spike times (dashed vert.) + s_1(t) per tau_s ────
        ax = axes[0]
        # Neuron 1's spike times are tau_s-independent (no input). Take from r0.
        spike_idx_1 = np.where(r0["spike_mask"][:, 0])[0]
        spike_t_1 = t[spike_idx_1]
        spike_t_1 = spike_t_1[(spike_t_1 >= t_lo) & (spike_t_1 <= t_hi)]
        for ts in spike_t_1:
            ax.axvline(ts, color="black", lw=vline_lw, ls="--",
                        alpha=0.8, zorder=1)
        for r, c in zip(results, tau_colors):
            ax.plot(r["t"][mask_t], r["s"][mask_t, 0],
                    color=c, lw=trace_lw,
                    label=fr"$\tau_s={r['tau_s']}$",
                    zorder=2)
        ax.set_ylabel(r"$s_1(t)$")
        ax.set_ylim(bottom=0.0)
        ax.set_yticks([0.0, 1.0, 2.0, 3.0], labels=["0.0", "1.0", "2.0", "3.0"])
        # ax.legend(loc="upper right", frameon=False, ncol=n_tau)
        ax.set_title("Two-neuron QIF + STDP")

        # ── Row 2: neuron-2 spike times + LTP term x_1(t) s_2(t), per tau_s ──
        ax = axes[1]
        # Vertical lines for neuron-2 spikes, color-matched to each tau_s
        for r, c in zip(results, tau_colors):
            spike_idx_2 = np.where(r["spike_mask"][:, 1])[0]
            spike_t_2 = t[spike_idx_2]
            spike_t_2 = spike_t_2[(spike_t_2 >= t_lo) & (spike_t_2 <= t_hi)]
            for ts in spike_t_2:
                ax.axvline(ts, color=c, lw=vline_lw, ls="--",
                            alpha=0.8, zorder=1)
        # Then plot the LTP term on top (normalized so max = 1 per tau_s)
        for r, c in zip(results, tau_colors):
            ltp = r["x"][:, 0] * r["s"][:, 1]
            ltp_max = ltp[mask_t].max() if ltp[mask_t].size else 1.0
            ltp_norm = ltp / ltp_max if ltp_max > 0 else ltp
            ax.plot(r["t"][mask_t], ltp_norm[mask_t],
                    color=c, lw=trace_lw,
                    label=fr"$\tau_s={r['tau_s']}$",
                    zorder=2)
        ax.set_ylabel(r"$x_1(t)\, s_2(t)$")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.3, 0.6, 0.9])
        # ax.legend(loc="upper right", frameon=False, ncol=n_tau)

        # ── Row 3: A_21(t) for each tau_s ────────────────────────────────────
        ax = axes[2]
        for r, c in zip(results, tau_colors):
            ax.plot(r["t"][mask_t], r["A_21"][mask_t],
                    color=c, lw=1.2 * trace_lw,
                    label=fr"$\tau_s={r['tau_s']}$")
        ax.set_ylabel(r"$A_{21}(t)$")
        ax.set_xlabel(r"time $t$")
        ax.legend(loc="best", frameon=False, ncol=n_tau)
        ax.set_ylim(0.4, 0.7)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7])

        for ax in axes:
            ax.grid(alpha=0.25)
            ax.set_xlim(t_lo, t_hi)

        fig.tight_layout()
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved -> {savepath}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # User-configurable excitabilities for the two neurons.
    # With eta > 0, neuron fires at intrinsic period ~ pi / sqrt(eta).
    # Pick asymmetric values so neuron 1 (pre) fires noticeably faster than
    # neuron 2 (post). This biases the trace-based STDP toward net LTP.
    eta1 = 0.5     # pre-synaptic neuron, faster
    eta2 = 0.2     # post-synaptic neuron, slower intrinsically

    tau_values = (0.1, 0.4, 1.6)

    # Larger mu so the slow weight dynamics are clearly visible on the T=80
    # window. Switch to mu = 0.02 for the slower/realistic learning regime.
    mu = 0.2
    a_plus  = lambda A: mu if A <= 1.0 else 0.0
    a_minus = lambda A: mu if A >= 0.0 else 0.0

    SIM = dict(
        eta1=eta1, eta2=eta2,
        J=1.0,           # excitatory coupling
        A0=0.5,          # initial weight in the middle of [0, 1]
        T=40.0,
        dt=1e-4,
        V_peak=100.0,
        a_plus=a_plus, a_minus=a_minus,
    )

    results = []
    for tau_s in tau_values:
        print(f"Simulating tau_s = {tau_s} ...")
        r = simulate_two_qif(tau_s=tau_s, **SIM)
        print(f"  Final A_21 = {r['A_21'][-1]:.4f}")
        results.append(r)

    plot_two_qif(
        results,
        savepath="/home/rgast/data/qif_plasticity/two_qif_stdp.svg",
        fontsize=20.0,
        vline_lw=2.0,
        trace_lw=1.2,
        dpi=200,
    )
    plt.show()
