r"""
QIF network with inhibitory synapses — three-model comparison
=============================================================

Compares, for a single simulation without extrinsic input, three descriptions of
the SAME network of quadratic integrate-and-fire (QIF) neurons with inhibitory,
plastic synapses (see prl_draft.pdf):

  1. micro  – the spiking QIF network            (Eqs. 25, 26, 30, 31)
  2. OA     – the Ott–Antonsen / MPR mean field  (Eqs. 32–36)
  3. WC     – the Wilson–Cowan reduced rate model (Eqs. 34–37)

All three share the same heterogeneous excitabilities (one distribution per
ensemble), the same inhibitory coupling J<0, the same synaptic kernel and the same
USER-DEFINED plasticity rule, so they can be compared directly.

Model correspondence
---------------------
micro coupling   (J/N) Σ_j A_ij s_j          (Eq. 25)
MF coupling      J Σ_n A_mn w_n s_n  (w_n=1/M) (Eqs. 33/37)
synaptic act.    ṡ = (drive − s)/τ_s          (Eq. 26/35, DC gain 1)
traces           τ_x ẋ = −x + s,  τ_y ẏ = −y + s   (Eq. 30/34)
plasticity       Ȧ_ij = a_p(A_ij)·P_ij − a_d(A_ij)·D_ij
                 P, D = user LTP/LTD terms;  a_p(A), a_d(A) = soft or hard weight bounds:
                   soft:  a_p(A)=a·a_p·(A_max−A),     a_d(A)=a·a_d·(A−A_min)
                   hard:  a_p(A)=a·a_p·𝟙[A<A_max],    a_d(A)=a·a_d·𝟙[A>A_min]

The LTP and LTD terms are *defined by the user* (P["ltp"], P["ltd"]) as Python
expressions in the building blocks
    s_i, s_j   – post / pre synaptic activation
    x_i, x_j   – post / pre fast trace  (time constant τ_x)
    y_i, y_j   – post / pre slow trace  (time constant τ_y)
The same expression strings are injected into the PyRates edge equations and
compiled (via njit/numpy) for the spiking network and the trace read-outs, so the
rule is guaranteed identical across all three models. Use Python syntax (``**`` for
powers, bare ``exp``/``sin``/``cos``/``sqrt``/``abs``).

The micro N×N weights are block-averaged to M×M for the matrix comparison.

Run in the ``sbi`` conda env (the only one with both PyRates and numba):
    PATH="$HOME/conda/envs/sbi/bin:$PATH" python qif_mf_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import perf_counter
from numba import njit, prange
from scipy.integrate import solve_ivp

from pyrates import OperatorTemplate, NodeTemplate, EdgeTemplate, CircuitTemplate, clear

import sys, os

from scipy.stats import norm, uniform

sys.path.append(os.path.abspath(".."))
from config.utility_functions import lorentzian2


# ════════════════════════════════════════════════════════════════════════════
#  parameters
# ════════════════════════════════════════════════════════════════════════════
P = dict(
    # network size
    M=10,                 # number of ensembles (mean-field nodes)
    n_per=50,             # QIF neurons per ensemble  (N = M*n_per)
    # excitability distribution (one per ensemble)
    dist="uniform",
    eta_bar=0.5,          # global centre of ensemble centres
    eta_spread=1.0,       # spread of the ensemble centres
    Delta=0.2,            # within-ensemble half-width
    # coupling (inhibitory)
    J=-10.0,
    # synapse + plasticity traces
    tau_s=0.5,
    tau_x=0.5,            # fast trace x time constant
    tau_y=1.0,            # slow trace y time constant
    # plasticity rule:  Ȧ = a_p(A)·P − a_d(A)·D
    a=0.01,               # base learning rate
    a_p=1.5,              # LTP rate constant
    a_d=1.0,              # LTD rate constant
    A0=0.5,               # initial coupling weight
    bounds="soft",        # "soft": a_p(A)=a·a_p·(A_max−A),  a_d(A)=a·a_d·(A−A_min)
                          # "hard": rates held at a·a_p / a·a_d until A reaches A_max/A_min, then 0
    A_min=0.0, A_max=1.0, # weight bounds
    # USER-DEFINED LTP/LTD coincidence terms P, D in s_i,s_j,x_i,x_j,y_i,y_j (see header)
    ltp="s_i*x_j",        # default P: post activation × pre fast trace
    ltd="y_i*s_j",        # default D: post slow trace × pre activation
    # simulation
    T=200.0,
    dt=1e-3,              # micro integration step
    dts=0.5,              # sampling step for the recorded traces
    w_stride=10,          # update the N×N micro weight matrix every w_stride steps
    v_peak=100.0,         # QIF spike threshold / |reset|
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
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 140,
    })


# ════════════════════════════════════════════════════════════════════════════
#  user-defined plasticity rule -> callables (numpy + numba) and PyRates strings
# ════════════════════════════════════════════════════════════════════════════
_RULE_NS = {"np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos,
            "sqrt": np.sqrt, "abs": np.abs, "tanh": np.tanh, "pi": np.pi}


def make_rule(ltp_expr, ltd_expr):
    """Compile the user's LTP/LTD expressions into (numpy fn, numba fn) pairs and
    the PyRates-syntax equation strings. The fn signature is f(s_i,s_j,x_i,x_j,y_i,y_j)
    and works element-wise on scalars (numba loop) or broadcast arrays (read-outs)."""
    sig = "lambda s_i, s_j, x_i, x_j, y_i, y_j:"
    ltp_py = eval(f"{sig} ({ltp_expr})", _RULE_NS)
    ltd_py = eval(f"{sig} ({ltd_expr})", _RULE_NS)
    ltp_jit, ltd_jit = njit(ltp_py), njit(ltd_py)
    # PyRates uses ^ for powers
    ltp_pr = ltp_expr.replace("**", "^")
    ltd_pr = ltd_expr.replace("**", "^")
    return ltp_py, ltd_py, ltp_jit, ltd_jit, ltp_pr, ltd_pr


def mean_ltp_ltd(s_traj, x_traj, y_traj, ltp_py, ltd_py):
    """Population-mean LTP term ⟨P⟩ and LTD term ⟨D⟩ over time, from recorded (K, T)
    traces. K = N (micro) or M (mean field); the mean is over all (post i, pre j)
    pairs. These are the raw user-defined terms (without the a_p(A)/a_d(A) rates)."""
    T = s_traj.shape[1]
    lt = np.empty(T); ld = np.empty(T)
    for t in range(T):
        si, xi, yi = s_traj[:, t], x_traj[:, t], y_traj[:, t]
        lt[t] = ltp_py(si[:, None], si[None, :], xi[:, None], xi[None, :],
                       yi[:, None], yi[None, :]).mean()
        ld[t] = ltd_py(si[:, None], si[None, :], xi[:, None], xi[None, :],
                       yi[:, None], yi[None, :]).mean()
    return lt, ld


# ════════════════════════════════════════════════════════════════════════════
#  micro QIF network  (Eqs. 25, 26, 30, 31) — fully JIT'd
# ════════════════════════════════════════════════════════════════════════════
@njit(parallel=True, fastmath=True)
def simulate_qif(V, s, up, ud, w, eta, J_over_N, tau_s, tau_x, tau_y,
                 a, a_p, a_d, A_min, A_max, soft, dt, dt_w, w_stride, v_peak,
                 steps, sr, ltp_fn, ltd_fn):
    N = V.shape[0]
    n_save = steps // sr
    rate_rec = np.zeros(n_save)
    s_rec = np.zeros((n_save, N))
    x_rec = np.zeros((n_save, N))
    y_rec = np.zeros((n_save, N))
    spikes = np.zeros(N)
    ss = 0
    for step in range(steps):
        # synaptic coupling current  (J/N) Σ_j A_ij s_j
        inp = J_over_N * np.dot(w, s)
        # QIF membrane update + threshold/reset
        V = V + dt * (V * V + eta + inp)
        spikes[:] = 0.0
        for i in prange(N):
            if V[i] >= v_peak:
                V[i] = -v_peak
                spikes[i] = 1.0 / dt
        # synaptic activation + fast/slow traces (DC gain 1)
        s = s + dt * (spikes - s) / tau_s
        up = up + dt * (s - up) / tau_x
        ud = ud + dt * (s - ud) / tau_y
        # plastic weights — coarse stride (plasticity is slow)
        if step % w_stride == 0:
            for i in prange(N):
                for j in range(N):
                    Pt = ltp_fn(s[i], s[j], up[i], up[j], ud[i], ud[j])
                    Dt = ltd_fn(s[i], s[j], up[i], up[j], ud[i], ud[j])
                    wij = w[i, j]
                    if soft:                                   # soft bounds
                        gp = A_max - wij
                        gd = wij - A_min
                    else:                                      # hard bounds (Heaviside)
                        gp = 0.5 * (np.sign(A_max - wij) + 1.0)
                        gd = 0.5 * (np.sign(wij - A_min) + 1.0)
                    dw = a * a_p * gp * Pt - a * a_d * gd * Dt
                    w[i, j] = wij + dt_w * dw
        if step % sr == 0:
            rate_rec[ss] = np.mean(s)
            s_rec[ss] = s; x_rec[ss] = up; y_rec[ss] = ud
            ss += 1
    return rate_rec, w, s_rec, x_rec, y_rec


def run_micro(P, eta_micro, ltp_jit, ltd_jit):
    N = P["M"] * P["n_per"]
    steps = int(P["T"] / P["dt"])
    sr = int(P["dts"] / P["dt"])
    soft = (P["bounds"] == "soft")
    rng = np.random.default_rng(0)
    V = rng.uniform(-2.5, -1.5, N)
    s = np.zeros(N); up = np.zeros(N); ud = np.zeros(N)
    w = np.full((N, N), P["A0"])
    rate, w_final, s_rec, x_rec, y_rec = simulate_qif(
        V, s, up, ud, w, eta_micro.astype(np.float64), P["J"] / N,
        P["tau_s"], P["tau_x"], P["tau_y"], P["a"], P["a_p"], P["a_d"],
        P["A_min"], P["A_max"], soft,
        P["dt"], P["dt"] * P["w_stride"], P["w_stride"], P["v_peak"], steps, sr,
        ltp_jit, ltd_jit)
    # block-average N×N -> M×M (neurons are ordered ensemble-by-ensemble)
    M, n = P["M"], P["n_per"]
    W_block = w_final.reshape(M, n, M, n).mean(axis=(1, 3))
    return rate, W_block, s_rec.T, x_rec.T, y_rec.T


# ════════════════════════════════════════════════════════════════════════════
#  mean-field models via PyRates
# ════════════════════════════════════════════════════════════════════════════
def _trace_ops(tau_x, tau_y):
    ltp = OperatorTemplate(name="ltp_op", equations=["u_p' = (s - u_p)/tau_p"],
                           variables={"u_p": "output(0.0)", "s": "input(0.0)", "tau_p": tau_x})
    ltd = OperatorTemplate(name="ltd_op", equations=["u_d' = (s - u_d)/tau_d"],
                           variables={"u_d": "output(0.0)", "s": "input(0.0)", "tau_d": tau_y})
    return ltp, ltd


def _syn_op(tau_s):
    return OperatorTemplate(name="syn_op", equations=["s' = (r - s)/tau_s"],
                            variables={"s": "output(0.0)", "r": "input(0.0)", "tau_s": tau_s})


def _stdp_edge(ltp_pr, ltd_pr, a, a_p, a_d, A_min, A_max, bounds, A0):
    # bound-dependent LTP/LTD rate factors a_p(A), a_d(A)
    if bounds == "soft":
        gp, gd = "(A_max - w)", "(w - A_min)"
    elif bounds == "hard":                              # Heaviside via sign()
        gp, gd = "(sign(A_max - w) + 1)/2", "(sign(w - A_min) + 1)/2"
    else:
        raise ValueError(f"Invalid bounds: {bounds}")
    stdp = OperatorTemplate(name="stdp_op", equations=[
        f"ltp = {ltp_pr}",
        f"ltd = {ltd_pr}",
        f"w' = a*a_p*{gp}*ltp - a*a_d*{gd}*ltd",
        "s_out = s_in*w"],
        variables={"s_out": "output(0.0)", "w": f"variable({A0})",
                   "ltp": "variable(0.0)", "ltd": "variable(0.0)",
                   "a": a, "a_p": a_p, "a_d": a_d, "A_min": A_min, "A_max": A_max,
                   "s_in": "input(0.0)",
                   "s_i": "input(0.0)", "s_j": "input(0.0)",
                   "x_i": "input(0.0)", "x_j": "input(0.0)",
                   "y_i": "input(0.0)", "y_j": "input(0.0)"})
    return EdgeTemplate(name="stdp_edge", operators=[stdp])


def build_mf_net(kind, P, eta_bar, ltp_pr, ltd_pr):
    """Build the OA ('oa') or Wilson–Cowan ('wc') mean-field network in PyRates."""
    M = P["M"]
    if kind == "oa":                                   # MPR / Ott–Antonsen (Eqs. 32-33)
        node_op = OperatorTemplate(name="qif_op", equations=[
            "r' = Delta/pi + 2*r*v",
            "v' = v^2 + eta - (pi*r)^2 + J*s_in"],
            variables={"r": "output(0.1)", "v": "variable(-2.0)",
                       "Delta": P["Delta"], "eta": P["eta_bar"], "J": P["J"], "s_in": "input(0.0)"})
        rate_op = "qif_op"
    elif kind == "wc":                                  # Wilson–Cowan reduced rate (Eq. 37)
        node_op = OperatorTemplate(name="fre_op", equations=[
            "inp = eta + J*s_in",
            "r' = -r + sqrt(inp + sqrt(inp^2 + Delta^2))/(sqrt(2)*pi)"],
            variables={"r": "output(0.1)", "inp": "variable(0.0)",
                       "Delta": P["Delta"], "eta": P["eta_bar"], "J": P["J"], "s_in": "input(0.0)"})
        rate_op = "fre_op"
    else:
        raise ValueError(kind)

    syn = _syn_op(P["tau_s"]); ltp, ltd = _trace_ops(P["tau_x"], P["tau_y"])
    node = NodeTemplate(name=f"{kind}_pop", operators=[node_op, syn, ltp, ltd])
    edge = _stdp_edge(ltp_pr, ltd_pr, P["a"], P["a_p"], P["a_d"],
                      P["A_min"], P["A_max"], P["bounds"], P["A0"])

    edges = []
    for i in range(M):
        for j in range(M):
            edges.append((f"p{j}/syn_op/s", f"p{i}/{rate_op}/s_in", deepcopy(edge),
                          {"weight": 1.0 / M,
                           "stdp_edge/stdp_op/s_in": f"p{j}/syn_op/s",   # routing (pre)
                           "stdp_edge/stdp_op/s_i": f"p{i}/syn_op/s",    # post activation
                           "stdp_edge/stdp_op/s_j": f"p{j}/syn_op/s",    # pre  activation
                           "stdp_edge/stdp_op/x_i": f"p{i}/ltp_op/u_p",  # post fast trace
                           "stdp_edge/stdp_op/x_j": f"p{j}/ltp_op/u_p",  # pre  fast trace
                           "stdp_edge/stdp_op/y_i": f"p{i}/ltd_op/u_d",  # post slow trace
                           "stdp_edge/stdp_op/y_j": f"p{j}/ltd_op/u_d"})) # pre slow trace
    net = CircuitTemplate(name=kind, nodes={f"p{i}": node for i in range(M)}, edges=edges)
    net.update_var(node_vars={f"all/{rate_op}/eta": eta_bar})
    return net, rate_op


def run_mf(kind, P, eta_bar, ltp_pr, ltd_pr):
    M = P["M"]
    net, rate_op = build_mf_net(kind, P, eta_bar, ltp_pr, ltd_pr)
    # float64 throughout so the njit'd vector field's internal `dot` sees matching
    # dtypes (solve_ivp hands the state to the RHS as float64).
    func, args, keys, vmap = net.get_run_func(f"{kind}_vf", step_size=P["dt"],
                                              backend="numpy", vectorize=True, clear=False,
                                              float_precision="float64")

    def block(suffix):
        return np.arange(*next(v for k, v in vmap.items() if k.endswith(suffix)))
    r_idx = block(f"{rate_op}/r")
    s_idx = block("syn_op/s")
    x_idx = block("ltp_op/u_p")
    y_idx = block("ltd_op/u_d")
    w_idx = block("stdp_op/w")

    # PyRates' run func has signature func(t, y, dy, *params): it writes the vector
    # field into the shared `dy` buffer and returns it. JIT it for speed, and copy
    # the result so the solver's RK stages don't alias the same buffer.
    y0 = np.asarray(args[1], dtype=float)
    extra = args[2:]                              # (dy_buffer, *params)
    func_jit = njit(func)
    func_jit(0.0, y0, *extra)                     # warm-up compile

    def f(t, y):
        return np.array(func_jit(t, y, *extra))

    t_eval = np.arange(0.0, P["T"], P["dts"])
    sol = solve_ivp(f, (0.0, P["T"]), y0, method="RK45",
                    t_eval=t_eval, rtol=1e-6, atol=1e-8)
    clear(net)

    rate = sol.y[r_idx, :].mean(axis=0)           # weighted mean (equal ensembles)
    W = sol.y[w_idx, -1].reshape(M, M)            # final M×M coupling matrix
    s_traj, x_traj, y_traj = sol.y[s_idx], sol.y[x_idx], sol.y[y_idx]
    return rate, W, s_traj, x_traj, y_traj


def get_dist(d: str):
    if d == "lorentzian":
        return lambda loc, scale, size: lorentzian2(size, loc, scale)
    elif d == "gaussian":
        return norm.rvs
    elif d == "uniform":
        return uniform.rvs
    else:
        raise ValueError(f"Invalid distribution: {d}")


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    M, n, distribution = P["M"], P["n_per"], P["dist"]
    f = get_dist(distribution)
    # ensemble centres + per-neuron excitabilities
    eta_bar = f(P["eta_bar"], P["eta_spread"], M)
    eta_micro = np.concatenate([f(eta_bar[m], P["Delta"], n) for m in range(M)])

    # compile the user-defined plasticity rule once, shared by all three models
    ltp_py, ltd_py, ltp_jit, ltd_jit, ltp_pr, ltd_pr = make_rule(P["ltp"], P["ltd"])

    print(f"QIF inhibitory three-model comparison — M={M}, N={M*n}, J={P['J']}")
    print(f"  plasticity:  P = {P['ltp']}   D = {P['ltd']}   ({P['bounds']} bounds)")
    print(f"  ensemble centres η̄_m ∈ [{eta_bar.min():.2f}, {eta_bar.max():.2f}]")

    t0 = perf_counter()
    print("[micro] spiking QIF network ...")
    r_micro, W_micro, s_mi, x_mi, y_mi = run_micro(P, eta_micro, ltp_jit, ltd_jit)
    print(f"   done in {perf_counter()-t0:.1f}s")

    t0 = perf_counter()
    print("[OA] Ott–Antonsen mean field ...")
    r_oa, W_oa, s_oa, x_oa, y_oa = run_mf("oa", P, eta_bar, ltp_pr, ltd_pr)
    print(f"   done in {perf_counter()-t0:.1f}s")

    t0 = perf_counter()
    print("[WC] Wilson–Cowan rate model ...")
    r_wc, W_wc, s_wc, x_wc, y_wc = run_mf("wc", P, eta_bar, ltp_pr, ltd_pr)
    print(f"   done in {perf_counter()-t0:.1f}s")

    # population-mean LTP/LTD terms over time (same rule applied to each model)
    ltp_mi, ltd_mi = mean_ltp_ltd(s_mi, x_mi, y_mi, ltp_py, ltd_py)
    ltp_oa, ltd_oa = mean_ltp_ltd(s_oa, x_oa, y_oa, ltp_py, ltd_py)
    ltp_wc, ltd_wc = mean_ltp_ltd(s_wc, x_wc, y_wc, ltp_py, ltd_py)

    # ── figure ──────────────────────────────────────────────────────────────
    set_prl_style()
    time = np.arange(0.0, P["T"], P["dts"])
    models = [("QIF network", "0.25", "-"),
              ("Ott–Antonsen", "#1F77B4", "-"),
              ("Wilson–Cowan", "#D62728", "--")]
    fig = plt.figure(figsize=(7.0, 8.2), layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.5])

    def ts_panel(row, series, ylabel, tag):
        ax = fig.add_subplot(gs[row, :])
        npts = min([len(time)] + [len(y) for y in series])
        for y, (lbl, col, ls) in zip(series, models):
            ax.plot(time[:npts], y[:npts], color=col, ls=ls, lw=1.1, label=lbl)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, P["T"])
        ax.set_title(tag, loc="left")
        return ax

    ax_r = ts_panel(0, [r_micro, r_oa, r_wc], r"mean firing rate $\langle r \rangle$",
                    "(a)  average firing rate dynamics")
    ax_r.legend(loc="best", ncol=3)
    ts_panel(1, [ltp_mi, ltp_oa, ltp_wc], r"$\langle\mathrm{LTP}\rangle$", "(b)  LTP term")
    ax_d = ts_panel(2, [ltd_mi, ltd_oa, ltd_wc], r"$\langle\mathrm{LTD}\rangle$", "(c)  LTD term")
    ax_d.set_xlabel("time")

    # row 4 — the three final coupling matrices, one shared colorbar
    mats = [W_micro, W_oa, W_wc]
    titles = ["QIF (block-avg.)", "Ott–Antonsen", "Wilson–Cowan"]
    vmin = min(m.min() for m in mats)             # common scale, data range for contrast
    vmax = max(m.max() for m in mats)
    axes2 = [fig.add_subplot(gs[3, k]) for k in range(3)]
    for k, (ax, Wm, ttl) in enumerate(zip(axes2, mats, titles)):
        im = ax.imshow(Wm, vmin=vmin, vmax=vmax, cmap="magma", origin="upper",
                       aspect="equal", interpolation="nearest")
        ax.set_title(f"(d{k+1})  {ttl}" if k == 0 else ttl)
        ax.set_xlabel("ensemble $n$")
        if k == 0:
            ax.set_ylabel("ensemble $m$")
        else:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=axes2, location="right", shrink=0.9, fraction=0.05, pad=0.02)
    cb.set_label(r"coupling weight $A_{mn}$")

    fig.savefig("qif_mf_comparison.pdf", bbox_inches="tight")
    fig.savefig("qif_mf_comparison.png", dpi=300, bbox_inches="tight")
    print("[plot] saved qif_mf_comparison.{pdf,png}")


if __name__ == "__main__":
    main()
