r"""
QIF network with Allen-fitted threshold heterogeneity — spiking net vs. mean field
===================================================================================

Loads the Lorentzian-mixture fit of the Allen excitability gap v_θ − v_r produced
by ``data_fitting/allen_lorentzian_fit.py`` and simulates

  1. a spiking network of N quadratic integrate-and-fire (QIF) neurons
         v̇_i = (v_i − v_r)(v_i − v_{θ,i}) + I(t) + J s(t),
     with neuron-specific thresholds v_{θ,i} = v_r + x_i; s(t) is the recurrent synaptic
     input, an alpha-kernel convolution of the population spike rate; I(t) a global input;
     J a global coupling constant. Integrated with forward Euler (spike at v≥v_peak →
     reset). This network is simulated TWICE: once drawing the x_i from the fitted
     Lorentzian mixture, and once drawing them from the EMPIRICAL distribution (the saved
     normalized histogram interpolated into a PDF and inverse-CDF sampled), to test how
     faithfully the Lorentzian fit reproduces the network drawn from the real data.

  2. the corresponding mean-field equations (Gast, Solla & Kennedy, Phys. Rev. E
     107, 024306 (2023), Eqs. 22–25 with b=κ=0 ⇒ no recovery variable, C=k=1, and
     the additive current Js instead of a conductance synapse).  Each Lorentzian
     component m of the threshold distribution (weight w_m, centre Ω_m, width Δ_m)
     becomes an Ott–Antonsen ensemble with mean threshold v̄_{θ,m}=v_r+Ω_m:
         ṙ_m = (Δ_m/π)(v_m − v_r) + r_m (2 v_m − v_r − v̄_{θ,m})
         v̇_m = (v_m − v_r)(v_m − v̄_{θ,m}) − (π r_m)² − π Δ_m r_m + I(t) + J s
     all ensembles share s, the alpha-kernel convolution of the total rate
     r = Σ_m w_m r_m.  Integrated with scipy.solve_ivp.

All model equations (membrane, alpha synapse, mean field) are implemented in PyRates
via PopulationTemplate (vectorised populations: n=N QIF neurons / n=M ensembles, each
node also carrying the shared alpha synapse). PyRates only emits the continuous vector
field; three things are handled in the integrator and injected into PyRates parameters
each step: the spike reset of the QIF network, the time-varying input I(t), and the
population firing rate that drives the synapse (the spike flux in the network; the
total rate Σ_m w_m r_m in the mean field). Injecting the rate keeps both right-hand
sides element-wise and therefore njit-compilable.

The alpha synapse is a cascade of two first-order filters with time constant τ_s
(impulse response (t/τ_s²)e^{−t/τ_s}, DC gain 1):  ȧ = (r − a)/τ_s,  ṡ = (a − s)/τ_s.

Run in the ``allen`` conda env (PyRates 1.2.3 PopulationTemplate/Connectivity + numba
+ scipy + the Allen fit .npz):
    PATH="$HOME/conda/envs/allen/bin:$PATH" python allen_qif_meanfield.py
"""
import os
import sys
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

from pyrates import (OperatorTemplate, NodeTemplate, CircuitTemplate,
                     PopulationTemplate, clear)

_HERE = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
def _tag(cell_class, layer):
    """Filename tag matching data_fitting/allen_lorentzian_fit.py (e.g. 'pyramidal_L23')."""
    c = cell_class.split("+")[0].split()[0].lower()       # Pyramidal→pyramidal, PV+ int→pv
    return f"{c}_{layer.replace('/', '').replace(' ', '')}"


# cell class / layer: CLI args override the defaults -> `python allen_qif_meanfield.py "Pyramidal" "L5/6"`
CELL_CLASS = sys.argv[1] if len(sys.argv) > 1 else "Pyramidal"   # | "PV+ interneuron" | "SOM interneuron"
LAYER = sys.argv[2] if len(sys.argv) > 2 else "L2/3"             # "L2/3" | "L5/6"
_TAG = _tag(CELL_CLASS, LAYER)
FIT_NPZ = os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{_TAG}.npz")
OUT = os.path.join(_HERE, f"allen_qif_meanfield_{_TAG}")

P = dict(
    v_r=-70.0,            # resting potential (one root of the quadratic), fixed
    J=100.0,                # global recurrent coupling constant
    tau_s=2.0,           # alpha-synapse time constant
    # global input I(t): baseline I0, rectangular pulse to I1 on [t_on, t_off]
    I0=200.0, I1=400.0, t_on=150.0, t_off=450.0,
    # simulation
    T=700.0,
    dt=2e-4,              # forward-Euler step (micro). MUST be small: Euler overshoots the
                          # convex QIF upstroke (v̇∼v²), so a coarse dt makes neurons cross
                          # threshold too early → inflated rate, amplified by strong J. dt=1e-3
                          # already biases r/v noticeably here; 2e-4 converges. NB a LARGER
                          # v_peak lengthens the v→∞ excursion and needs an even smaller dt.
    dts=0.5,              # recording step
    # spiking network
    N=10000,              # number of QIF neurons
    v_peak=430.0,         # spike detection threshold (≈ +∞)
    v_reset=-570.0,       # reset potential (≈ −∞)
    theta_clip=50.0,     # clip Lorentzian-sampled x_i to ±clip (numerical safety on heavy tails)
    seed=0,
)


# ════════════════════════════════════════════════════════════════════════════
#  PRL figure style
# ════════════════════════════════════════════════════════════════════════════
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 140,
    })


# ════════════════════════════════════════════════════════════════════════════
#  fit loading + helpers
# ════════════════════════════════════════════════════════════════════════════
def load_fit(path):
    d = np.load(path, allow_pickle=False)
    w = np.asarray(d["weights"], float)
    Omega = np.asarray(d["omega"], float)        # centres of v_θ − v_r (mV)
    Delta = np.asarray(d["delta"], float)        # half-widths (mV)
    return w / w.sum(), Omega, Delta, int(d["M"])


def load_empirical(path):
    """Empirical distribution of v_θ − v_r as a normalized histogram (centres, density)."""
    d = np.load(path, allow_pickle=False)
    return np.asarray(d["emp_centers"], float), np.asarray(d["emp_pdf"], float)


def sample_thresholds(w, Omega, Delta, N, v_r, clip, rng):
    """Draw N thresholds v_{θ,i} = v_r + x_i, x_i ~ Σ_m w_m Lorentzian(Ω_m, Δ_m)."""
    comp = rng.choice(len(w), size=N, p=w)
    x = Omega[comp] + Delta[comp] * np.tan(np.pi * (rng.random(N) - 0.5))
    return v_r + np.clip(x, -clip, clip)


def sample_empirical(centers, pdf, N, v_r, rng):
    """Interpolate the normalized histogram counts into a continuous empirical PDF, then
    inverse-CDF sample N excitability gaps x_i and return the thresholds v_{θ,i}=v_r+x_i."""
    xs = np.linspace(centers[0], centers[-1], 4000)
    p = np.clip(np.interp(xs, centers, pdf), 0.0, None)   # linear interp of the histogram
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    x = np.interp(rng.random(N), cdf, xs)                 # inverse-CDF sampling
    return v_r + x


def make_input(P):
    I0, I1, t_on, t_off = P["I0"], P["I1"], P["t_on"], P["t_off"]
    def I(t):
        return I1 if (t_on <= t < t_off) else I0
    return I


def _state_slice(vmap, suffix):
    """Index range of a state variable in the flat state vector (scalar int or (a,b) tuple)."""
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
def build_micro(P, v_th):
    N = v_th.size
    # one population of N QIF neurons; each unit also carries the (shared) alpha synapse.
    # syn output `s` auto-connects to the membrane input `s`; `rin` (rate) and `Iext` (input)
    # are unconnected → PyRates parameters that we overwrite each Euler step.
    mem_op = OperatorTemplate(name="qif_mem",
        equations=["v' = (v - vr)*(v - vth) + Iext + J*s"],
        variables={"v": "output(-70.0)", "vth": -32.0, "vr": P["v_r"], "J": P["J"],
                   "Iext": "input(0.0)", "s": "input(0.0)"})
    syn_op = OperatorTemplate(name="alpha_syn",
        equations=["a' = (rin - a)/tau_s", "s' = (a - s)/tau_s"],
        variables={"s": "output(0.0)", "a": "variable(0.0)", "rin": "input(0.0)", "tau_s": P["tau_s"]})
    pop = PopulationTemplate(name="net", node=NodeTemplate(name="m", operators=[mem_op, syn_op]), n=N,
                             params={"qif_mem/vth": v_th, "qif_mem/v": np.full(N, P["v_r"])})
    net = CircuitTemplate("micro", populations={"net": pop})
    func, args, keys, vmap = net.get_run_func("micro_vf", step_size=P["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="float64")
    return func, args, keys, vmap, net


# njit decorator for the forward-Euler driver. BENCHMARK (N=10000, 400k steps):
#   pure-Python loop 28.1 s → njit loop 21.8 s  (≈1.3× faster; the loop overhead drops
#   from 10.4 s to 4.1 s). The remaining 17.7 s is the PyRates RHS `f` itself (already
#   njit'd), which dominates ~80% of the runtime and no loop decorator can reduce.
# Among DECORATOR ARGUMENTS none gave a further speedup (all within ~3%): fastmath,
# nogil and error_model="numpy" were neutral (the heavy math lives in `f`); parallel=True
# was slightly SLOWER (a sequential time-stepping loop has nothing to parallelise).
# ⇒ plain @njit is the best choice.
EULER_NJIT_KWARGS = {}


def _euler_loop(f, y, extra, p_I, p_r, v0, v1, s0, s1, N, dt, steps, sr,
                vp, vreset, I0, I1, t_on, t_off):
    """Forward-Euler time stepping of the spiking QIF network, with spike reset and
    the I(t)/rate injection done in-loop. ``f`` is the njit'd PyRates RHS (called via
    f(t, y, *extra)); ``p_I``/``p_r`` are size-1 VIEWS of the (0-d) PyRates parameter
    arrays for the input and the synapse-driving rate — they share memory with the arrays
    stored in ``extra``, so writing element [0] here is what f reads. njit-compiled (see
    EULER_NJIT_KWARGS / _euler_loop_jit)."""
    n_save = steps // sr + 1
    t_rec = np.empty(n_save); r_rec = np.empty(n_save); s_rec = np.empty(n_save)
    v_rec = np.empty(n_save); vmed_rec = np.empty(n_save)
    ny = y.shape[0]
    spike_accum = 0
    rate = 0.0
    ss = 0
    for k in range(steps):
        t = k * dt
        p_I[0] = I1 if (t_on <= t < t_off) else I0
        p_r[0] = rate                                     # drive synapse with last step's rate
        dy = f(t, y, *extra)
        for i in range(ny):
            y[i] += dt * dy[i]
        nsp = 0
        for i in range(v0, v1):
            if y[i] >= vp:
                y[i] = vreset
                nsp += 1
        rate = nsp / (N * dt)
        spike_accum += nsp
        if k % sr == 0:
            t_rec[ss] = t
            r_rec[ss] = spike_accum / (N * sr * dt)       # mean rate over the recording window
            s_rec[ss] = np.mean(y[s0:s1])
            v_rec[ss] = np.mean(y[v0:v1])                 # arithmetic mean (heavy-tail biased)
            vmed_rec[ss] = np.median(y[v0:v1])            # median ≈ Lorentzian centre = MF v
            spike_accum = 0
            ss += 1
    return t_rec[:ss], r_rec[:ss], s_rec[:ss], v_rec[:ss], vmed_rec[:ss]


_euler_loop_jit = njit(**EULER_NJIT_KWARGS)(_euler_loop)


def run_micro(P, v_th, loop=None):
    func, args, keys, vmap, net = build_micro(P, v_th)
    N = v_th.size
    extra = tuple(args[2:])
    f = njit(func)                                        # RHS (fastmath gave no measurable gain)
    f(0.0, np.asarray(args[1], float), *extra)            # warm-up compile
    clear(net)

    y = np.asarray(args[1], float).copy()
    v_sl = _state_slice(vmap, "qif_mem/v")
    s_sl = _state_slice(vmap, "alpha_syn/s")              # replicated across units (all identical)
    # injected params are 0-d (unconnected inputs); pass size-1 VIEWS (shared memory)
    # so the njit loop can write them (0-d arrays aren't writable in nopython mode).
    p_I = _param(args, keys, "qif_mem/Iext").reshape(1)   # injected: global input I(t)
    p_r = _param(args, keys, "alpha_syn/rin").reshape(1)  # injected: population spike rate

    dt, steps = P["dt"], int(round(P["T"] / P["dt"]))
    sr = max(1, int(round(P["dts"] / dt)))
    loop = loop or _euler_loop_jit
    return loop(f, y, extra, p_I, p_r, v_sl.start, v_sl.stop, s_sl.start, s_sl.stop,
                N, dt, steps, sr, P["v_peak"], P["v_reset"],
                P["I0"], P["I1"], P["t_on"], P["t_off"])


# ════════════════════════════════════════════════════════════════════════════
#  mean field: M Ott–Antonsen ensembles — PyRates vector field + solve_ivp
# ════════════════════════════════════════════════════════════════════════════
def build_mf(P, vth_bar, Delta):
    M = Delta.size
    # one population of M Ott–Antonsen ensembles; each node also carries the (shared)
    # alpha synapse. The synapse input `rin` (= total rate Σ_m w_m r_m) and `Iext` are
    # unconnected → parameters we inject; `s` auto-connects to the oa-equation input `s`.
    oa_op = OperatorTemplate(name="oa_op", equations=[
        "r' = Delta/pi*(v - vr) + r*(2*v - vr - vthbar)",
        "v' = (v - vr)*(v - vthbar) - (pi*r)^2 - pi*Delta*r + Iext + J*s"],
        variables={"r": "output(0.0)", "v": "variable(-70.0)",
                   "Delta": 1.0, "vthbar": -32.0, "vr": P["v_r"], "J": P["J"],
                   "Iext": "input(0.0)", "s": "input(0.0)"})
    syn_op = OperatorTemplate(name="alpha_syn",
        equations=["a' = (rin - a)/tau_s", "s' = (a - s)/tau_s"],
        variables={"s": "output(0.0)", "a": "variable(0.0)", "rin": "input(0.0)", "tau_s": P["tau_s"]})
    pop = PopulationTemplate(name="mf", node=NodeTemplate(name="e", operators=[oa_op, syn_op]), n=M,
                             params={"oa_op/Delta": Delta, "oa_op/vthbar": vth_bar,
                                     "oa_op/v": np.full(M, P["v_r"])})
    net = CircuitTemplate("mf", populations={"mf": pop})
    func, args, keys, vmap = net.get_run_func("mf_vf", step_size=P["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="float64")
    return func, args, keys, vmap, net


def run_mf(P, w, vth_bar, Delta):
    func, args, keys, vmap, net = build_mf(P, vth_bar, Delta)
    extra = args[2:]
    f = njit(func)
    y0 = np.asarray(args[1], float)
    f(0.0, y0, *extra)                                    # warm-up compile
    clear(net)

    r_sl = _state_slice(vmap, "oa_op/r")
    v_sl = _state_slice(vmap, "oa_op/v")
    s_sl = _state_slice(vmap, "alpha_syn/s")              # replicated across ensembles (identical)
    p_I = _param(args, keys, "oa_op/Iext")                # injected: global input I(t)
    p_r = _param(args, keys, "alpha_syn/rin")             # injected: total rate Σ_m w_m r_m
    I = make_input(P)

    def rhs(t, y):
        p_I.fill(I(t))
        p_r.fill(w @ y[r_sl])                             # total population rate drives the synapse
        return np.array(f(t, y, *extra))

    t_eval = np.arange(0.0, P["T"], P["dts"])
    sol = solve_ivp(rhs, (0.0, P["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=1e-7, atol=1e-9, max_step=P["tau_s"])
    r = w @ sol.y[r_sl, :]                                # total rate Σ w_m r_m
    s = sol.y[s_sl, :].mean(axis=0)                       # global synaptic activation
    v = w @ sol.y[v_sl, :]                                # mean potential
    return sol.t, r, s, v


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    w, Omega, Delta, M = load_fit(FIT_NPZ)
    vth_bar = P["v_r"] + Omega
    rng = np.random.default_rng(P["seed"])
    v_th = sample_thresholds(w, Omega, Delta, P["N"], P["v_r"], P["theta_clip"], rng)

    print(f"Allen fit: M={M} ensembles, ⟨v_θ⟩={vth_bar @ w:.2f} mV, v_r={P['v_r']}")
    print(f"  centres v̄_θ,m = {np.round(vth_bar, 2)}")
    print(f"  widths  Δ_m    = {np.round(Delta, 3)}")
    print(f"  weights w_m    = {np.round(w, 3)}")

    # empirical-distribution thresholds (interpolated normalized histogram → inverse-CDF)
    emp_centers, emp_pdf = load_empirical(FIT_NPZ)
    v_th_emp = sample_empirical(emp_centers, emp_pdf, P["N"], P["v_r"], rng)

    t0 = perf_counter()
    print(f"[micro-fit] spiking QIF network (fitted Lorentzian mixture), N={P['N']} ...")
    tm, rm, sm, vm, vmed = run_micro(P, v_th)
    print(f"   done in {perf_counter()-t0:.1f}s")

    t0 = perf_counter()
    print(f"[micro-emp] spiking QIF network (empirical distribution), N={P['N']} ...")
    te, re, se, ve, vmede = run_micro(P, v_th_emp)
    print(f"   done in {perf_counter()-t0:.1f}s")

    t0 = perf_counter()
    print("[mean field] M-ensemble Ott–Antonsen ...")
    tf, rf, sf, vf = run_mf(P, w, vth_bar, Delta)
    print(f"   done in {perf_counter()-t0:.1f}s")

    # ── save the rate dynamics for the summary figure (bifurcation_analysis/allen_qif_figure.py) ──
    Iarr = np.array([make_input(P)(t) for t in tf])
    np.savez(OUT + ".npz",
             cell_class=CELL_CLASS, layer=LAYER,
             t_micro=tm, r_micro=rm,                       # spiking net, fitted-Lorentzian thresholds
             t_micro_emp=te, r_micro_emp=re,               # spiking net, empirical-distribution thresholds
             t_mf=tf, r_mf=rf, s_mf=sf, v_mf=vf,           # mean field
             t_input=tf, input=Iarr,
             I0=float(P["I0"]), I1=float(P["I1"]),
             t_on=float(P["t_on"]), t_off=float(P["t_off"]), J=float(P["J"]))
    print(f"[saved] {OUT}.npz")

    # ── figure ────────────────────────────────────────────────────────────────
    set_prl_style()
    C_FIT, C_EMP, C_MF = "0.25", "#2ca02c", "#c1121f"
    I = make_input(P)
    Iarr = np.array([I(t) for t in tf])
    fig, axes = plt.subplots(4, 1, figsize=(5.4, 6.6), sharex=True, layout="constrained")

    axes[0].plot(tf, Iarr, color="0.4", lw=1.0)
    axes[0].set_ylabel(r"input $I(t)$")
    axes[0].set_title("(a)  global input", loc="left")

    axes[1].plot(tm, rm, color=C_FIT, lw=1.0, label="QIF (fitted Lorentzian)")
    axes[1].plot(te, re, color=C_EMP, lw=1.0, label="QIF (empirical)")
    axes[1].plot(tf, rf, color=C_MF, lw=1.2, ls="--", label="mean field (fit)")
    axes[1].set_ylabel(r"firing rate $r(t)$")
    axes[1].set_title("(b)  population firing rate", loc="left")
    axes[1].legend(loc="best", fontsize=6.5)

    axes[2].plot(tm, sm, color=C_FIT, lw=1.0)
    axes[2].plot(te, se, color=C_EMP, lw=1.0)
    axes[2].plot(tf, sf, color=C_MF, lw=1.2, ls="--")
    axes[2].set_ylabel(r"synaptic $s(t)$")
    axes[2].set_title("(c)  recurrent synaptic input", loc="left")

    axes[3].plot(tm, vmed, color=C_FIT, lw=1.0, label="QIF (fitted)")
    axes[3].plot(te, vmede, color=C_EMP, lw=1.0, label="QIF (empirical)")
    axes[3].plot(tf, vf, color=C_MF, lw=1.2, ls="--", label="mean field")
    axes[3].set_ylabel(r"median $v(t)$ (mV)")
    axes[3].set_title("(d)  membrane potential (median)", loc="left")
    axes[3].set_xlabel("time")
    axes[3].set_xlim(0, P["T"])
    axes[3].legend(loc="best", ncol=3, fontsize=6.5)

    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
