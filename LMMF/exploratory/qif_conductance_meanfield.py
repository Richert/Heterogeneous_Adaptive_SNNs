r"""
Conductance-based QIF network with homeostatic synaptic conductance — spiking net vs. mean field
=================================================================================================

Self-contained script for simulating a recurrent network of quadratic
integrate-and-fire (QIF) neurons with a *conductance* synapse whose gain is
regulated by a slow, activity-dependent homeostatic variable, together with the
corresponding low-dimensional mean-field (Ott–Antonsen / Montbrió–Pazó–Roxin)
description. Both the spiking network and the mean field are built with PyRates
(the same PopulationTemplate + get_run_func recipe used in
``allen_qif_meanfield.py``); the spike reset, the time-varying input and the
recurrent coupling are injected into PyRates parameters inside the integrator.

Model (microscopic)
-------------------
For i = 1..N,

    C dV_i/dt = k (V_i - V_r)(V_i - V_θ) + η_i + I(t)
                + (g_i / N) Σ_j W_ij s_j (V_i - E)

    τ_s ds_i/dt : s_i is the alpha-kernel convolution of neuron i's own spike train
                  (cascade of two 1st-order filters, τ_s; impulse response
                  (t/τ_s²)e^{-t/τ_s}, DC gain 1 → in steady state s_i ≈ firing rate of i)

    dg_i/dt = (1 - g_i)/τ_g + (g_m - g_i) S - g_i s_i   (activity-gated conductance)

    Steady state (τ_g ≫ 1, leak negligible): g_i ≈ g_m S/(S + s_i) — a strictly
    positive conductance in [0, g_m] that is driven toward g_m by the set-point S and
    closed by the neuron's own activity s_i (synaptic-depression-like homeostasis).

C, k, V_r, V_θ, E, τ_s, τ_g, S, g_m are constants; η_i is a neuron-specific excitability
drawn from a Lorentzian L(η̄, Δ) (centre η̄, half-width-at-half-maximum Δ). We use
all-to-all coupling W_ij = J, so (1/N) Σ_j W_ij s_j = J s̄ with s̄ = (1/N) Σ_j s_j.
Note the synaptic term is written exactly as given, +g s (V-E); choose the sign of
J for excitatory vs. inhibitory recurrence in this convention.

The alpha synapse is realised as two cascaded first-order filters:
    da_i/dt = -a_i/τ_s ,   ds_i/dt = (a_i - s_i)/τ_s ,
with every spike of neuron i adding 1/τ_s to a_i (equivalent to driving a_i with
the neuron's instantaneous spike rate). Averaged over the population this gives
    da/dt = (r - a)/τ_s ,  ds/dt = (a - s)/τ_s   (r = population firing rate).

Mean field
----------
Heterogeneity is in the additive term η_i ~ L(η̄, Δ). Applying the Lorentzian
ansatz to C V̇ = k V² + A V + B + η with A = -k(V_r+V_θ) + gJs and
B = k V_r V_θ - gJsE, and using the QIF firing-rate/flux relation r = k x /(πC)
(x = half-width of the voltage distribution, i.e. x = πC r/k), the pole
z = v - iπC r/k obeys C ż = k z² + A z + B + η̄ - iΔ. Splitting into real/imaginary
parts gives the closed mean-field system (population means r, v, a, s, g):

    C ṙ = k Δ/(πC) + r [ k (2v - V_r - V_θ) + g J s ]
    C v̇ = k (v - V_r)(v - V_θ) - (πC r)²/k + η̄ + I(t) + g J s (v - E)
    τ_s ȧ = r - a ,   τ_s ṡ = a - s
    ġ = (1 - g)/τ_g + (g_m - g) S - g s

i.e. the standard MPR reduction (constant Δ-flux term in ṙ, no Δ term in v̇,
because the heterogeneity is additive in η) plus a conductance coupling gJs and a
population-mean conductance g driven by the mean synaptic activation s. This is a
mean-field approximation for the conductance: the per-neuron g_i (correlated with
V_i and with the neuron's own s_i) is replaced by the population mean g, and the
bilinear closure ⟨g_i s_i⟩ ≈ g s is used. Both are justified when τ_g ≫ 1: the
conductance is then slow relative to the fast phase dynamics, so within each OA
ensemble g_i is approximately common (decorrelated from the fast fluctuations of
V_i and s_i), and the closure becomes exact.

Run in the ``allen`` conda env (PyRates 1.2.3 + numba + scipy):
    PATH="$HOME/conda/envs/allen/bin:$PATH" python qif_conductance_meanfield.py
CLI (optional):  python qif_conductance_meanfield.py <J> <I1>
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
from prl_style import set_prl_style as _set_prl_style
# ---------------------------------------------------------------------------
import os
import sys
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

from pyrates import (OperatorTemplate, NodeTemplate, CircuitTemplate,
                     PopulationTemplate, clear)

OUT = os.path.join(_HERE, "qif_conductance_meanfield")

# ════════════════════════════════════════════════════════════════════════════
#  configuration  (edit these — every symbol above maps to one entry here)
# ════════════════════════════════════════════════════════════════════════════
P = dict(
    # --- QIF membrane (all constants; biophysical Izhikevich scale: pF, mV, ms) ---
    C=100.0,              # membrane capacitance
    k=0.7,                # quadratic gain
    v_r=-60.0,            # lower root of the quadratic (≈ resting potential)
    v_th=-40.0,           # upper root of the quadratic (≈ intrinsic threshold)
    E=0.0,                # synaptic reversal potential (0 ⇒ excitatory; with E=0, J<0 excites)
    # --- excitability heterogeneity η_i ~ Lorentzian(η̄, Δ) ---
    eta_bar=100.0,        # centre of the excitability distribution (> rheobase ≈ 70 ⇒ tonic)
    Delta=1.0,            # half-width-at-half-maximum
    # --- synapse + homeostatic conductance ---
    J=-40.0,              # global recurrent coupling (W_ij = J); with E=0, J<0 ⇒ excitatory here
    tau_s=10.0,           # alpha-synapse time constant
    tau_g=500.0,          # conductance leak time constant; the mean field needs τ_g ≫ 1
    S=0.05,               # set-point: opening rate of g toward g_m
    g_m=2.0,              # maximum conductance (g relaxes into [0, g_m])
    # --- global input I(t): baseline I0, rectangular pulse to I1 on [t_on, t_off] ---
    I0=0.0, I1=40.0, t_on=500.0, t_off=800.0,
    # --- simulation ---
    T=1200.0,
    dt=1e-3,              # forward-Euler step (micro); keep small (Euler overshoots v̇∼v²)
    dts=1.0,              # recording step
    N=10000,              # number of QIF neurons
    v_peak=400.0,         # spike detection threshold (≈ +∞)
    v_reset=-600.0,       # reset potential (≈ −∞)
    eta_clip=400.0,       # clip Lorentzian-sampled η_i to ±clip (numerical safety on heavy tails)
    seed=0,
)
if len(sys.argv) > 1:
    P["J"] = float(sys.argv[1])
if len(sys.argv) > 2:
    P["I1"] = float(sys.argv[2])


# ════════════════════════════════════════════════════════════════════════════
#  PRL figure style
# ════════════════════════════════════════════════════════════════════════════
set_prl_style = functools.partial(_set_prl_style, "diagnostic",
                                 **{})


# ════════════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════════════
def sample_eta(eta_bar, Delta, N, clip, rng):
    """Draw N excitabilities η_i ~ Lorentzian(η̄, Δ) via inverse-CDF (tan) sampling."""
    x = eta_bar + Delta * np.tan(np.pi * (rng.random(N) - 0.5))
    return np.clip(x, eta_bar - clip, eta_bar + clip)


def make_input(P):
    I0, I1, t_on, t_off = P["I0"], P["I1"], P["t_on"], P["t_off"]
    def I(t):
        return I1 if (t_on <= t < t_off) else I0
    return I


def _state_slice(vmap, suffix):
    """Index range of a state variable in the flat state vector."""
    v = next(val for k, val in vmap.items() if k.endswith(suffix))
    if isinstance(v, (tuple, list)):
        return slice(int(v[0]), int(v[1]))
    return slice(int(v), int(v) + 1)


def _param(args, keys, suffix):
    """The (mutable) PyRates parameter array whose key ends in `suffix`."""
    return args[keys.index(next(k for k in keys if k.endswith(suffix)))]


# ════════════════════════════════════════════════════════════════════════════
#  micro: spiking QIF network — PyRates vector field + forward Euler
# ════════════════════════════════════════════════════════════════════════════
def build_micro(P, eta):
    """One population of N QIF neurons; each unit carries its own alpha synapse (a, s)
    and its own homeostatic conductance g. Operator wiring inside the node:
        alpha_syn: s (per neuron, kicked by that neuron's spikes) → cond & (via s_mean) mem
        cond:      g (per neuron, driven by own s)                → mem
        qif_mem:   v ; reads g, injected s_mean (= J·s̄ coupling) and injected Iext
    `s_mean` and `Iext` are unconnected inputs → 0-d PyRates parameters we overwrite each
    Euler step; the per-neuron synapse is driven by adding 1/τ_s to a_i on each spike.
    NB E→E_r, S→S_g: `E` and `S` are names reserved by PyRates."""
    N = eta.size
    mem_op = OperatorTemplate(name="qif_mem",
        equations=["v' = (k*(v - vr)*(v - vth) + eta + Iext + g*s_mean*(v - E_r))/C"],
        variables={"v": "output(-70.0)", "vth": P["v_th"], "vr": P["v_r"], "E_r": P["E"],
                   "k": P["k"], "C": P["C"], "eta": P["eta_bar"],
                   "g": "input(1.0)", "s_mean": "input(0.0)", "Iext": "input(0.0)"})
    cond_op = OperatorTemplate(name="cond",
        equations=["g' = (1 - g)/tau_g + (g_m - g)*S_g - g*s"],
        variables={"g": "output(1.0)", "s": "input(0.0)", "tau_g": P["tau_g"], "S_g": P["S"], "g_m": P["g_m"]})
    syn_op = OperatorTemplate(name="alpha_syn",
        equations=["a' = -a/tau_s", "s' = (a - s)/tau_s"],
        variables={"s": "output(0.0)", "a": "variable(0.0)", "tau_s": P["tau_s"]})
    pop = PopulationTemplate(name="net",
        node=NodeTemplate(name="m", operators=[mem_op, cond_op, syn_op]), n=N,
        params={"qif_mem/eta": eta, "qif_mem/v": np.full(N, P["v_r"])})
    net = CircuitTemplate("micro", populations={"net": pop})
    func, args, keys, vmap = net.get_run_func("micro_vf", step_size=P["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="float64")
    return func, args, keys, vmap, net


def _euler_loop(f, y, extra, p_I, p_sm, v0, v1, a0, s0, s1, g0, g1, J, N, dt, steps, sr,
                vp, vreset, kick, I0, I1, t_on, t_off):
    """Forward-Euler stepping of the spiking network. Spike reset, per-neuron synapse
    kick, I(t) and the recurrent coupling drive s_mean = J·mean(s) are done in-loop.
    p_I / p_sm are size-1 views of the (0-d) PyRates parameter arrays (shared memory)."""
    n_save = steps // sr + 1
    t_rec = np.empty(n_save); r_rec = np.empty(n_save); s_rec = np.empty(n_save)
    v_rec = np.empty(n_save); g_rec = np.empty(n_save)
    ny = y.shape[0]
    spike_accum = 0
    ss = 0
    for k in range(steps):
        t = k * dt
        p_I[0] = I1 if (t_on <= t < t_off) else I0
        # recurrent coupling: g_i * (J * mean_j s_j) * (v_i - E); inject J*s̄ as s_mean
        smean = 0.0
        for i in range(s0, s1):
            smean += y[i]
        p_sm[0] = J * smean / N
        dy = f(t, y, *extra)
        for i in range(ny):
            y[i] += dt * dy[i]
        nsp = 0
        for i in range(v0, v1):
            if y[i] >= vp:
                y[i] = vreset
                y[a0 + (i - v0)] += kick          # kick this neuron's own synapse (alpha)
                nsp += 1
        spike_accum += nsp
        if k % sr == 0:
            t_rec[ss] = t
            r_rec[ss] = spike_accum / (N * sr * dt)
            s_rec[ss] = np.mean(y[s0:s1])
            v_rec[ss] = np.median(y[v0:v1])          # median ≈ Lorentzian centre = MF v
            g_rec[ss] = np.mean(y[g0:g1])
            spike_accum = 0
            ss += 1
    return t_rec[:ss], r_rec[:ss], s_rec[:ss], v_rec[:ss], g_rec[:ss]


_euler_loop_jit = njit(_euler_loop)


def run_micro(P, eta, loop=None):
    func, args, keys, vmap, net = build_micro(P, eta)
    N = eta.size
    extra = tuple(args[2:])
    f = njit(func)
    f(0.0, np.asarray(args[1], float), *extra)            # warm-up compile
    clear(net)

    y = np.asarray(args[1], float).copy()
    v_sl = _state_slice(vmap, "qif_mem/v")
    a_sl = _state_slice(vmap, "alpha_syn/a")
    s_sl = _state_slice(vmap, "alpha_syn/s")
    g_sl = _state_slice(vmap, "cond/g")
    p_I = _param(args, keys, "qif_mem/Iext").reshape(1)    # injected: global input I(t)
    p_sm = _param(args, keys, "qif_mem/s_mean").reshape(1)  # injected: J·mean(s) coupling

    dt, steps = P["dt"], int(round(P["T"] / P["dt"]))
    sr = max(1, int(round(P["dts"] / dt)))
    loop = loop or _euler_loop_jit
    return loop(f, y, extra, p_I, p_sm, v_sl.start, v_sl.stop, a_sl.start,
                s_sl.start, s_sl.stop, g_sl.start, g_sl.stop, P["J"], N, dt, steps, sr,
                P["v_peak"], P["v_reset"], 1.0 / P["tau_s"],
                P["I0"], P["I1"], P["t_on"], P["t_off"])


# ════════════════════════════════════════════════════════════════════════════
#  mean field: single Ott–Antonsen population — PyRates vector field + solve_ivp
# ════════════════════════════════════════════════════════════════════════════
def build_mf(P):
    """One OA population (r, v) + alpha synapse (a, s) + homeostatic conductance (g).
    Wiring: syn outputs s → cond & oa; cond outputs g → oa. The rate r that drives the
    synapse is INJECTED each step (rather than connected oa→syn) to break the otherwise
    cyclic oa→syn→oa operator graph, which PyRates forbids within a node. The global
    input Iext is injected too. NB E→E_r, S→S_g (PyRates-reserved names)."""
    oa_op = OperatorTemplate(name="oa_op", equations=[
        "r' = (k*Delta/(pi*C) + r*(k*(2*v - vr - vth) + g*J*s))/C",
        "v' = (k*(v - vr)*(v - vth) - (pi*C*r)^2/k + eta_bar + Iext + g*J*s*(v - E_r))/C"],
        variables={"r": "output(0.0)", "v": "variable(-70.0)",
                   "vth": P["v_th"], "vr": P["v_r"], "E_r": P["E"], "k": P["k"], "C": P["C"],
                   "Delta": P["Delta"], "eta_bar": P["eta_bar"], "J": P["J"],
                   "g": "input(1.0)", "s": "input(0.0)", "Iext": "input(0.0)"})
    cond_op = OperatorTemplate(name="cond",
        equations=["g' = (1 - g)/tau_g + (g_m - g)*S_g - g*s"],
        variables={"g": "output(1.0)", "s": "input(0.0)", "tau_g": P["tau_g"], "S_g": P["S"], "g_m": P["g_m"]})
    syn_op = OperatorTemplate(name="alpha_syn",
        equations=["a' = (rin - a)/tau_s", "s' = (a - s)/tau_s"],
        variables={"s": "output(0.0)", "a": "variable(0.0)", "rin": "input(0.0)", "tau_s": P["tau_s"]})
    pop = PopulationTemplate(name="mf",
        node=NodeTemplate(name="e", operators=[oa_op, cond_op, syn_op]), n=1,
        params={"oa_op/v": np.array([P["v_r"]])})
    net = CircuitTemplate("mf", populations={"mf": pop})
    func, args, keys, vmap = net.get_run_func("mf_vf", step_size=P["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="float64")
    return func, args, keys, vmap, net


def run_mf(P):
    func, args, keys, vmap, net = build_mf(P)
    extra = args[2:]
    f = njit(func)
    y0 = np.asarray(args[1], float)
    f(0.0, y0, *extra)                                    # warm-up compile
    clear(net)

    r_sl = _state_slice(vmap, "oa_op/r")
    v_sl = _state_slice(vmap, "oa_op/v")
    s_sl = _state_slice(vmap, "alpha_syn/s")
    g_sl = _state_slice(vmap, "cond/g")
    p_I = _param(args, keys, "oa_op/Iext")                # injected: global input I(t)
    p_r = _param(args, keys, "alpha_syn/rin")             # injected: rate r driving the synapse
    I = make_input(P)

    def rhs(t, y):
        p_I.fill(I(t))
        p_r.fill(y[r_sl][0])                              # break oa→syn cycle: inject rate
        return np.array(f(t, y, *extra))

    t_eval = np.arange(0.0, P["T"], P["dts"])
    sol = solve_ivp(rhs, (0.0, P["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=1e-8, atol=1e-10, max_step=P["tau_s"])
    return sol.t, sol.y[r_sl][0], sol.y[s_sl][0], sol.y[v_sl][0], sol.y[g_sl][0]


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(P["seed"])
    eta = sample_eta(P["eta_bar"], P["Delta"], P["N"], P["eta_clip"], rng)
    print(f"QIF conductance network: N={P['N']}, η̄={P['eta_bar']}, Δ={P['Delta']}, "
          f"J={P['J']}, τ_g={P['tau_g']}, S={P['S']}")

    t0 = perf_counter()
    print(f"[micro] spiking QIF network, N={P['N']} ...")
    tm, rm, sm, vm, gm = run_micro(P, eta)
    print(f"   done in {perf_counter()-t0:.1f}s")

    t0 = perf_counter()
    print("[mean field] Ott–Antonsen reduction ...")
    tf, rf, sf, vf, gf = run_mf(P)
    print(f"   done in {perf_counter()-t0:.1f}s")

    Iarr = np.array([make_input(P)(t) for t in tf])
    np.savez(OUT + ".npz",
             t_micro=tm, r_micro=rm, s_micro=sm, v_micro=vm, g_micro=gm,
             t_mf=tf, r_mf=rf, s_mf=sf, v_mf=vf, g_mf=gf,
             t_input=tf, input=Iarr, **{k: float(v) for k, v in P.items()})
    print(f"[saved] {OUT}.npz")

    # ── figure ──────────────────────────────────────────────────────────────
    set_prl_style()
    C_MICRO, C_MF = "0.25", "#c1121f"
    fig, axes = plt.subplots(5, 1, figsize=(5.0, 7.2), sharex=True, layout="constrained")

    axes[0].plot(tf, Iarr, color="0.4", lw=1.0)
    axes[0].set_ylabel(r"input $I(t)$")
    axes[0].set_title("(a)  global input", loc="left")

    axes[1].plot(tm, rm, color=C_MICRO, lw=1.0, label="spiking network")
    axes[1].plot(tf, rf, color=C_MF, lw=1.2, ls="--", label="mean field")
    axes[1].set_ylabel(r"rate $r(t)$")
    axes[1].set_title("(b)  population firing rate", loc="left")
    axes[1].legend(loc="best")

    axes[2].plot(tm, sm, color=C_MICRO, lw=1.0)
    axes[2].plot(tf, sf, color=C_MF, lw=1.2, ls="--")
    axes[2].set_ylabel(r"synaptic $s(t)$")
    axes[2].set_title("(c)  mean synaptic activation", loc="left")

    axes[3].plot(tm, gm, color=C_MICRO, lw=1.0)
    axes[3].plot(tf, gf, color=C_MF, lw=1.2, ls="--")
    axes[3].set_ylabel(r"conductance $g(t)$")
    axes[3].set_title("(d)  homeostatic conductance", loc="left")

    axes[4].plot(tm, vm, color=C_MICRO, lw=1.0)
    axes[4].plot(tf, vf, color=C_MF, lw=1.2, ls="--")
    axes[4].set_ylabel(r"median $v(t)$")
    axes[4].set_title("(e)  membrane potential (median)", loc="left")
    axes[4].set_xlabel("time")
    axes[4].set_xlim(0, P["T"])

    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
