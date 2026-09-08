r"""
Structured-coupling conductance-QIF network vs. Lorentzian-mixture mean field (LMMF)
====================================================================================

Companion to ``qif_conductance_meanfield.py`` (all-to-all coupling, single Lorentzian η).
Here the excitabilities η_i come from a GAUSSIAN MIXTURE, the recurrent weights W_ij are
STRUCTURED (a deterministic function of the excitabilities of the pre- and post-synaptic
neuron, plus noise), and the mean field is obtained by the Lorentzian-Mixture Mean Field
(LMMF): a Lorentzian mixture is fitted to the empirical η distribution and each component
becomes an Ott–Antonsen ensemble. The LMMF equations keep the DETERMINISTIC part of the
coupling, reducing the microscopic W_ij to their ensemble averages (the noise averages out
∝ 1/√N).

Model (microscopic)
-------------------
For i = 1..N,

    C dV_i/dt = k (V_i - V_r)(V_i - V_θ) + η_i + I(t) + (g_i / N) Σ_j W_ij s_j (E - V_i)
    da_i/dt     = -a_i/τ_s ,  ds_i/dt = (a_i - s_i)/τ_s          (alpha synapse; spike ⇒ a_i += 1/τ_s)
    τ_g dh_i/dt = 1 - h_i + (S - s_i) h_i (g_m - h_i) ,  τ_g dg_i/dt = h_i - g_i   (conductance, τ_g ≫ 1)

The synapse is a standard conductance term +(g_i/N)Σ_j W_ij s_j (E - V_i) (current toward the
reversal E); with E = 0 and V_i < E, a positive coupling drive is depolarising, so J > 0 is
excitatory. The conductance g_i is SECOND-ORDER, built like the alpha synapse as a cascade of two
first-order filters with time constant τ_g: the first stage h_i carries the activity-gating and is
logistically confined to [0, g_m] — the term (S - s_i) h_i (g_m - h_i) pushes h_i up toward g_m
when the neuron's activity s_i is below the set-point S and down toward 0 when it is above — and the
second stage low-passes it, g_i → h_i. This gives a smooth, critically-damped (alpha-like) g_i with
the SAME steady state as the first-order version. τ_g ≫ 1 keeps g slow, which the closure needs.

η_i ~ Gaussian mixture Σ_c π_c N(μ_c, σ_c²). Define the standardised excitability
x_i = (η_i - η̄)/σ_η (η̄, σ_η the sample mean/std). The weights are strictly positive, have unit
global average, and factorise (rank-1) with independent control of the incoming (row) and
outgoing (column) strength, times multiplicative lognormal noise:

    W_ij = f_in(x_i) · f_out(x_j) · exp(σ_W ξ_ij) / ⟨·⟩ ,   ξ_ij ~ N(0,1) i.i.d.,  ⟨W⟩_ij = 1
    f_in(x)  = exp(c_in · x) ,   f_out(x) = exp(c_out · x)

so the in-strength Σ_j W_ij ∝ f_in(x_i) and the out-strength Σ_i W_ij ∝ f_out(x_j) are each a
neuron-specific function of η_i (cf. the k_i k_j scheme of
LMMF/exploratory/kmo_random_coupling_sweep.py, generalised to two functions and made positive).
c_in = c_out = 0 recovers the all-to-all network (⟨W⟩ = 1) of the companion script.

Mean field (LMMF)
-----------------
Fit ρ(η) ≈ Σ_m w_m Lorentzian(Ω_m, Δ_m) (shared/lorentzian_mixture.py, CvM). Each ensemble m
carries mean excitability Ω_m and, from its centre, x_m = (Ω_m - η̄)/σ_η and the (mean-
normalised) coupling coefficients f_in,m = f_in(x_m)/⟨f_in⟩, f_out,m = f_out(x_m)/⟨f_out⟩ (the
⟨·⟩ over the η sample; this carries the ⟨W⟩ = 1 rescaling into the reduction). Reducing W_ij to
its ensemble averages, ⟨W⟩_{mn} = f_in,m f_out,n (the multiplicative noise averages to a
constant that the rescaling absorbs), the drive to ensemble m closes on a single out-strength-
weighted collective synaptic mode

    S_eff = Σ_n w_n f_out,n s_n ,   coupling_m = g_m J f_in,m S_eff .

The per-ensemble mean-field (additive-η MPR reduction; see the companion script for the
derivation) is, with r = Σ_m w_m r_m the total rate. Note the synaptic term enters as +coupling_m
(E - v_m), so its contribution to the coefficient of v_m is -coupling_m, which is why it appears
with a MINUS sign in the ṙ_m equation:

    C ṙ_m = k Δ_m/(πC) + r_m [ k (2 v_m - V_r - V_θ) - coupling_m ]
    C v̇_m = k (v_m - V_r)(v_m - V_θ) - (πC r_m)²/k + Ω_m + I(t) + coupling_m (E - v_m)
    τ_s ȧ_m = r_m - a_m ,  τ_s ṡ_m = a_m - s_m
    τ_g ḣ_m = 1 - h_m + (S - s_m) h_m (g_m^max - h_m) ,  τ_g ġ_m = h_m - g_m

Implementation (PyRates edges close the recurrence)
--------------------------------------------------
Both models are PyRates circuits (get_run_func + njit). Each node is a feed-forward operator
cascade and the recurrent coupling is closed with a Connectivity EDGE, the idiomatic PyRates
way (cf. LMMF/kmo_lorentzian_fit_sweep.py):
  * micro — node cascade alpha_syn → cond → qif_mem; a matrix edge
        Connectivity(alpha_syn/s → qif_mem/coupling_in, weights=W/N)
    delivers (1/N) Σ_j W_ij s_j to each neuron (the full structured matrix, noise included);
    qif_mem multiplies by g_i J (E-V_i). The synapse is kicked (a_i += 1/τ_s) on each spike.
    Integrated with forward Euler (spike reset in the loop; I(t) injected as a parameter).
  * mean field — node cascade oa_op(r,v,g) → alpha_syn (rate r flows forward into the
    synapse); the synaptic output s is fed back to the membrane by two edges:
        Connectivity(alpha_syn/s → oa_op/coupling_in, weights A_mn=f_in,m w_n f_out,n)  ⇒ f_in,m S_eff
        Connectivity(alpha_syn/s → oa_op/s_self,     weights = I_M)                     ⇒ own s (for ġ)
    oa_op multiplies coupling_in by g_m J. Integrated with scipy.solve_ivp (I(t) injected).

Minimal Python environment
--------------------------
Pure-Python/CPU; no GPU, no compiled Auto/PyCoBi backend needed (unlike the bifurcation
scripts). Reference versions are the ``allen`` conda env this was developed in:

    Python  >= 3.10        (developed on 3.11)
    pyrates == 1.2.3       (dev/vector frontend: PopulationTemplate, Connectivity, get_run_func;
                            the matrix/scalar Connectivity edge API used here is 1.2.x-specific)
    numba   >= 0.57        (njit; developed on 0.65)
    numpy   >= 1.23
    scipy   >= 1.10        (scipy.integrate.solve_ivp)
    matplotlib >= 3.6      (constrained_layout + add_gridspec)

Also requires the repo-local module ``shared/lorentzian_mixture.py`` (imported as ``LM`` via the
sys.path insert above) — it has no extra third-party dependencies beyond numpy/scipy.

Run (developed in the ``allen`` conda env):
    PATH="$HOME/conda/envs/allen/bin:$PATH" python qif_structured_coupling_lmmf.py
CLI (optional):  python qif_structured_coupling_lmmf.py <c_in> <c_out> <sigma_W>
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
                     PopulationTemplate, Connectivity, clear)

import lorentzian_mixture as LM          # CvM Lorentzian-mixture fitter (shared/lorentzian_mixture.py)

OUT = os.path.join(_HERE, "qif_structured_coupling_lmmf")

# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
P = dict(
    # --- QIF membrane (biophysical Izhikevich scale: pF, mV, ms) ---
    C=100.0, k=0.7, v_r=-60.0, v_th=-40.0, E=0.0,
    # --- excitability η_i ~ Gaussian mixture Σ_c π_c N(μ_c, σ_c) (μ_c > rheobase ≈ 70) ---
    gmm_means=(90.0, 100.0, 120.0),
    gmm_stds=(6.0, 10.0, 6.0),
    gmm_weights=(0.3, 0.5, 0.2),
    # --- structured coupling W_ij ∝ f_in(x_i) f_out(x_j)·exp(σ_W ξ), f=exp(c·x), x=(η-η̄)/σ_η;
    #     positive weights, rescaled to global mean ⟨W⟩=1 ---
    c_in=-0.0,             # log-slope of the incoming (row-sum) strength vs. standardised η
    c_out=0.0,           # log-slope of the outgoing (col-sum) strength vs. standardised η
    sigma_W=0.5,          # multiplicative lognormal weight noise (averages out ∝ 1/√N in the MF)
    J=20.0,               # global coupling strength (with the +gs(E-v) synapse, E=0, J>0 ⇒ excitatory)
    tau_s=0.5, tau_g=100.0, S=0.05, g_m=10.0,
    # --- LMMF fit of the empirical η-distribution (shared/lorentzian_mixture.py) ---
    # delta_bounds' lower bound keeps ensembles broad: narrow Lorentzians give weakly-damped
    # OA ensembles that ring, whereas broader Δ_m match the finite-N network dynamics better.
    delta_bounds=(0.1, 10.0), fit_M_max=20, fit_alpha=1e-3, fit_lambda=1e-5,
    fit_restarts=10, fit_method="slsqp",
    # --- global input I(t): baseline I0, rectangular pulse to I1 on [t_on, t_off] ---
    I0=0.0, I1=40.0, t_on=400.0, t_off=700.0,
    # --- simulation ---
    T=1000.0, dt=1e-3, dts=1.0,
    N=1000,               # QIF neurons (structured N×N coupling ⇒ matvec each Euler step)
    v_peak=400.0, v_reset=-600.0, seed=0,
)
if len(sys.argv) > 1:
    P["c_in"] = float(sys.argv[1])
if len(sys.argv) > 2:
    P["c_out"] = float(sys.argv[2])
if len(sys.argv) > 3:
    P["sigma_W"] = float(sys.argv[3])


# ════════════════════════════════════════════════════════════════════════════
#  PRL figure style
# ════════════════════════════════════════════════════════════════════════════
set_prl_style = functools.partial(_set_prl_style, "diagnostic",
                                 **{})


# ════════════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════════════
def sample_eta_gmm(means, stds, weights, N, rng):
    """Draw N excitabilities from a Gaussian mixture Σ_c weights_c N(means_c, stds_c)."""
    means, stds, weights = map(np.asarray, (means, stds, weights))
    comp = rng.choice(len(weights), size=N, p=weights / weights.sum())
    return rng.normal(means[comp], stds[comp])


def build_coupling(eta, eta_bar, sigma_eta, c_in, c_out, sigma_W, rng):
    """Structured, strictly POSITIVE weights with unit global average:
        W_ij = f_in(x_i) f_out(x_j) · exp(σ_W ξ_ij) ,  then  W /= ⟨W⟩ ,
    with x=(η-η̄)/σ_η, f_in=exp(c_in x), f_out=exp(c_out x) (both > 0). The multiplicative
    lognormal noise (ξ~N(0,1)) keeps every weight positive, and the final rescaling sets the
    global mean ⟨W⟩_ij = 1. In-strength Σ_j W_ij ∝ f_in(x_i), out-strength Σ_i W_ij ∝
    f_out(x_j); c_in = c_out = 0 ⇒ all-to-all (W_ij ≡ noise, ⟨W⟩=1). Returns (W, f_in, f_out)."""
    x = (eta - eta_bar) / sigma_eta
    f_in = np.exp(c_in * x)          # per-neuron incoming-strength factor (> 0)
    f_out = np.exp(c_out * x)        # per-neuron outgoing-strength factor (> 0)
    N = eta.size
    W = np.outer(f_in, f_out) * np.exp(sigma_W * rng.standard_normal((N, N)))
    W /= W.mean()                    # global weight average = 1
    return W, f_in, f_out


def make_input(P):
    I0, I1, t_on, t_off = P["I0"], P["I1"], P["t_on"], P["t_off"]
    def I(t):
        return I1 if (t_on <= t < t_off) else I0
    return I


def _state_slice(vmap, suffix):
    v = next(val for k, val in vmap.items() if k.endswith(suffix))
    if isinstance(v, (tuple, list)):
        return slice(int(v[0]), int(v[1]))
    return slice(int(v), int(v) + 1)


def _param(args, keys, suffix):
    return args[keys.index(next(k for k in keys if k.endswith(suffix)))]


# ════════════════════════════════════════════════════════════════════════════
#  micro: structured-coupling spiking QIF network — PyRates circuit + forward Euler
# ════════════════════════════════════════════════════════════════════════════
def build_micro(P, eta, W):
    """Feed-forward node cascade alpha_syn → cond → qif_mem; the structured coupling
    (g_i/N) Σ_j W_ij s_j (V_i-E) is closed by a matrix edge Connectivity(s → coupling_in,
    weights=W/N) that delivers (1/N) Σ_j W_ij s_j to each neuron. The synapse is kicked
    (a_i += 1/τ_s) per spike in the Euler loop. NB E→E_r, S→S_g (reserved PyRates names)."""
    N = eta.size
    syn_op = OperatorTemplate(name="alpha_syn",
        equations=["a' = -a/tau_s", "s' = (a - s)/tau_s"],
        variables={"s": "output(0.0)", "a": "variable(0.0)", "tau_s": P["tau_s"]})
    cond_op = OperatorTemplate(name="cond",       # 2nd-order (alpha-like) activity-gated conductance
        equations=["h' = (1 - h + (S_g - s)*h*(g_m-h))/tau_g", "g' = (h - g)/tau_g"],
        variables={"g": "output(1.0)", "h": "variable(1.0)", "s": "input(0.0)",
                   "tau_g": P["tau_g"], "S_g": P["S"], "g_m": P["g_m"]})
    mem_op = OperatorTemplate(name="qif_mem",
        equations=["v' = (k*(v - vr)*(v - vth) + eta + Iext + g*J*coupling_in*(E_r-v))/C"],
        variables={"v": "output(-60.0)", "vth": P["v_th"], "vr": P["v_r"], "E_r": P["E"],
                   "k": P["k"], "C": P["C"], "J": P["J"], "eta": P["v_r"],
                   "g": "input(1.0)", "coupling_in": "input(0.0)", "Iext": "input(0.0)"})
    pop = PopulationTemplate(name="net",
        node=NodeTemplate(name="m", operators=[syn_op, cond_op, mem_op]), n=N,
        params={"qif_mem/eta": eta, "qif_mem/v": np.full(N, P["v_r"])})
    conn = Connectivity("net/alpha_syn/s", "net/qif_mem/coupling_in", weights=np.ascontiguousarray(W / N))
    net = CircuitTemplate("micro", populations={"net": pop}, connections=[conn])
    func, args, keys, vmap = net.get_run_func("micro_vf", step_size=P["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="float64")
    return func, args, keys, vmap, net


def _euler_loop(f, y, extra, p_I, v0, v1, a0, s0, s1, g0, g1, N, dt, steps, sr,
                vp, vreset, kick, I0, I1, t_on, t_off):
    """Forward-Euler stepping with spike reset, per-neuron synapse kick and I(t) injection.
    The structured coupling is computed inside the PyRates RHS `f` (the Connectivity matvec)."""
    n_save = steps // sr + 1
    t_rec = np.empty(n_save); r_rec = np.empty(n_save); s_rec = np.empty(n_save)
    v_rec = np.empty(n_save); g_rec = np.empty(n_save)
    ny = y.shape[0]
    spike_accum = 0
    ss = 0
    for k in range(steps):
        t = k * dt
        p_I[0] = I1 if (t_on <= t < t_off) else I0
        dy = f(t, y, *extra)
        for i in range(ny):
            y[i] += dt * dy[i]
        nsp = 0
        for i in range(v0, v1):
            if y[i] >= vp:
                y[i] = vreset
                y[a0 + (i - v0)] += kick
                nsp += 1
        spike_accum += nsp
        if k % sr == 0:
            t_rec[ss] = t
            r_rec[ss] = spike_accum / (N * sr * dt)
            s_rec[ss] = np.mean(y[s0:s1])
            v_rec[ss] = np.median(y[v0:v1])
            g_rec[ss] = np.mean(y[g0:g1])
            spike_accum = 0
            ss += 1
    return t_rec[:ss], r_rec[:ss], s_rec[:ss], v_rec[:ss], g_rec[:ss]


_euler_loop_jit = njit(_euler_loop)


def run_micro(P, eta, W, loop=None):
    func, args, keys, vmap, net = build_micro(P, eta, W)
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
    p_I = _param(args, keys, "qif_mem/Iext").reshape(1)

    dt, steps = P["dt"], int(round(P["T"] / P["dt"]))
    sr = max(1, int(round(P["dts"] / dt)))
    loop = loop or _euler_loop_jit
    return loop(f, y, extra, p_I, v_sl.start, v_sl.stop, a_sl.start,
                s_sl.start, s_sl.stop, g_sl.start, g_sl.stop, N, dt, steps, sr,
                P["v_peak"], P["v_reset"], 1.0 / P["tau_s"],
                P["I0"], P["I1"], P["t_on"], P["t_off"])


# ════════════════════════════════════════════════════════════════════════════
#  LMMF fit
# ════════════════════════════════════════════════════════════════════════════
def fit_lmmf(eta, P):
    """Fit a Lorentzian mixture to the empirical η samples (CvM, greedy M)."""
    res = LM.fit(eta, P["delta_bounds"], M_max=P["fit_M_max"], alpha=P["fit_alpha"],
                 lambda_M=P["fit_lambda"], n_restarts=P["fit_restarts"],
                 seed=P["seed"], method=P["fit_method"])
    m = res["model"]
    return np.asarray(m.w, float) / m.w.sum(), np.asarray(m.Omega, float), np.asarray(m.Delta, float), m.M


# ════════════════════════════════════════════════════════════════════════════
#  mean field: M LMMF ensembles — PyRates circuit (edges) + solve_ivp
# ════════════════════════════════════════════════════════════════════════════
def build_mf(P, Omega, Delta, w, f_in_m, f_out_m):
    """Feed-forward node cascade oa_op(r,v,g) → alpha_syn (rate r flows forward into the
    synapse); the synaptic output s is fed back to the membrane by two edges: a matrix edge
    A_mn=f_in,m w_n f_out,n (⇒ coupling_in = f_in,m S_eff) and an identity edge (⇒ s_self, the
    own-ensemble s for ġ). oa_op multiplies coupling_in by g_m J. NB E→E_r, S→S_g."""
    M = Delta.size
    oa_op = OperatorTemplate(name="oa_op", equations=[
        "h' = (1 - h + (S_g - s_self)*h*(g_m-h))/tau_g",   # 2nd-order (alpha-like) conductance
        "g' = (h - g)/tau_g",
        "r' = (k*Delta/(pi*C) + r*(k*(2*v - vr - vth) - g*J*coupling_in))/C",
        "v' = (k*(v - vr)*(v - vth) - (pi*C*r)^2/k + eta_bar + Iext + g*J*coupling_in*(E_r-v))/C"],
        variables={"r": "output(0.0)", "v": "variable(-60.0)", "g": "variable(1.0)", "h": "variable(1.0)",
                   "vth": P["v_th"], "vr": P["v_r"], "E_r": P["E"], "k": P["k"], "C": P["C"], "J": P["J"],
                   "Delta": 1.0, "eta_bar": P["v_r"], "tau_g": P["tau_g"], "S_g": P["S"], "g_m": P["g_m"],
                   "s_self": "input(0.0)", "coupling_in": "input(0.0)", "Iext": "input(0.0)"})
    syn_op = OperatorTemplate(name="alpha_syn",
        equations=["a' = (r - a)/tau_s", "s' = (a - s)/tau_s"],
        variables={"s": "output(0.0)", "a": "variable(0.0)", "r": "input(0.0)", "tau_s": P["tau_s"]})
    pop = PopulationTemplate(name="mf",
        node=NodeTemplate(name="e", operators=[oa_op, syn_op]), n=M,
        params={"oa_op/Delta": Delta, "oa_op/eta_bar": Omega, "oa_op/v": np.full(M, P["v_r"])})
    A = f_in_m[:, None] * (w * f_out_m)[None, :]          # A_mn = f_in,m w_n f_out,n
    conn_coup = Connectivity("mf/alpha_syn/s", "mf/oa_op/coupling_in", weights=np.ascontiguousarray(A))
    conn_self = Connectivity("mf/alpha_syn/s", "mf/oa_op/s_self", weights=np.eye(M))
    net = CircuitTemplate("mf", populations={"mf": pop}, connections=[conn_coup, conn_self])
    func, args, keys, vmap = net.get_run_func("mf_vf", step_size=P["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="float64")
    return func, args, keys, vmap, net


def run_mf(P, w, Omega, Delta, f_in_m, f_out_m):
    func, args, keys, vmap, net = build_mf(P, Omega, Delta, w, f_in_m, f_out_m)
    extra = args[2:]
    f = njit(func)
    y0 = np.asarray(args[1], float)
    f(0.0, y0, *extra)                                    # warm-up compile
    clear(net)

    r_sl = _state_slice(vmap, "oa_op/r")
    v_sl = _state_slice(vmap, "oa_op/v")
    s_sl = _state_slice(vmap, "alpha_syn/s")
    g_sl = _state_slice(vmap, "cond/g" if any("cond/g" in k for k in vmap) else "oa_op/g")
    p_I = _param(args, keys, "oa_op/Iext")
    I = make_input(P)

    def rhs(t, y):
        p_I.fill(I(t))
        return np.array(f(t, y, *extra))

    t_eval = np.arange(0.0, P["T"], P["dts"])
    sol = solve_ivp(rhs, (0.0, P["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=1e-8, atol=1e-10, max_step=P["tau_s"])
    r = w @ sol.y[r_sl]                                   # total rate Σ w_m r_m
    s = w @ sol.y[s_sl]
    v = w @ sol.y[v_sl]
    g = w @ sol.y[g_sl]
    return sol.t, r, s, v, g


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(P["seed"])
    eta = sample_eta_gmm(P["gmm_means"], P["gmm_stds"], P["gmm_weights"], P["N"], rng)
    eta_bar, sigma_eta = float(eta.mean()), float(eta.std())
    W, f_in, f_out = build_coupling(eta, eta_bar, sigma_eta, P["c_in"], P["c_out"], P["sigma_W"], rng)
    print(f"structured-coupling QIF: N={P['N']}, η̄={eta_bar:.1f}, σ_η={sigma_eta:.1f}, "
          f"c_in={P['c_in']}, c_out={P['c_out']}, σ_W={P['sigma_W']}, J={P['J']}")

    # LMMF fit of the empirical η distribution
    w, Omega, Delta, M = fit_lmmf(eta, P)
    x_m = (Omega - eta_bar) / sigma_eta                  # standardised ensemble centres
    # per-ensemble coupling coefficients, normalised by the sample means ⟨f_in⟩,⟨f_out⟩ so the
    # ensemble-average weights reproduce the unit-mean (⟨W⟩=1) normalisation of the micro W
    f_in_m = np.exp(P["c_in"] * x_m) / f_in.mean()
    f_out_m = np.exp(P["c_out"] * x_m) / f_out.mean()
    print(f"LMMF fit: M={M} ensembles, Ω_m={np.round(Omega, 1)}, Δ_m={np.round(Delta, 2)}, "
          f"w_m={np.round(w, 3)}")

    t0 = perf_counter()
    print(f"[micro] structured-coupling spiking network, N={P['N']} ...")
    tm, rm, sm, vm, gm = run_micro(P, eta, W)
    print(f"   done in {perf_counter()-t0:.1f}s")

    t0 = perf_counter()
    print("[mean field] LMMF (deterministic ensemble-average coupling) ...")
    tf, rf, sf, vf, gf = run_mf(P, w, Omega, Delta, f_in_m, f_out_m)
    print(f"   done in {perf_counter()-t0:.1f}s")

    Iarr = np.array([make_input(P)(t) for t in tf])
    np.savez(OUT + ".npz",
             eta=eta, eta_bar=eta_bar, sigma_eta=sigma_eta, f_in=f_in, f_out=f_out, W=W,
             w=w, Omega=Omega, Delta=Delta, M=M, f_in_m=f_in_m, f_out_m=f_out_m,
             t_micro=tm, r_micro=rm, s_micro=sm, v_micro=vm, g_micro=gm,
             t_mf=tf, r_mf=rf, s_mf=sf, v_mf=vf, g_mf=gf, t_input=tf, input=Iarr,
             **{k: float(v) for k, v in P.items()
                if k not in ("gmm_means", "gmm_stds", "gmm_weights", "delta_bounds", "fit_method")})
    print(f"[saved] {OUT}.npz")

    # ── figure ──────────────────────────────────────────────────────────────
    set_prl_style()
    C_MICRO, C_MF, C_FIT, C_MIX = "0.25", "#c1121f", "0.8", "#2e6f95"
    fig = plt.figure(figsize=(7.0, 6.6), layout="constrained")
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.7])

    axd = fig.add_subplot(gs[0:2, 0])
    gx = np.linspace(np.percentile(eta, 0.5), np.percentile(eta, 99.5), 600)
    axd.hist(eta, bins=60, density=True, color=C_FIT, label="η samples")
    mix = sum(w[k] * (Delta[k] / np.pi) / ((gx - Omega[k]) ** 2 + Delta[k] ** 2) for k in range(M))
    axd.plot(gx, mix, color=C_MIX, lw=1.4, label=f"LMMF (M={M})")
    axd.set_xlabel(r"excitability $\eta$"); axd.set_ylabel("density")
    axd.set_title("(a)  η distribution + LMMF", loc="left"); axd.legend()

    order = np.argsort(eta)                               # rows & cols sorted by η so the
    Wsorted = W[np.ix_(order, order)]                     # f_in(η_i)×f_out(η_j) structure shows
    axw = fig.add_subplot(gs[2:4, 0])
    im = axw.imshow(Wsorted, aspect="auto", cmap="magma", vmin=0.0, vmax=np.percentile(W, 99.5))
    axw.set_xlabel(r"presynaptic $j$ (sorted by $\eta_j$)")
    axw.set_ylabel(r"postsynaptic $i$ (sorted by $\eta_i$)")
    axw.set_title(r"(b)  weights $W_{ij}$", loc="left")
    fig.colorbar(im, ax=axw, fraction=0.046)

    titles = [("(c)  population firing rate", rm, rf, r"rate $r(t)$"),
              ("(d)  mean synaptic activation", sm, sf, r"synaptic $s(t)$"),
              ("(e)  homeostatic conductance", gm, gf, r"conductance $g(t)$"),
              ("(f)  membrane potential (median)", vm, vf, r"median $v(t)$")]
    axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    for ax, (title, ym, yf, ylab) in zip(axes, titles):
        ax.plot(tm, ym, color=C_MICRO, lw=1.0, label="spiking network")
        ax.plot(tf, yf, color=C_MF, lw=1.2, ls="--", label="LMMF")
        ax.set_ylabel(ylab); ax.set_title(title, loc="left"); ax.set_xlim(0, P["T"])
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time [ms]")

    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
