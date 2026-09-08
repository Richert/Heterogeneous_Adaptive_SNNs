r"""
Pyramidal journal figure (updated) — rate dynamics in the multi-stable regime
=============================================================================
Regime J=100, tau_s=0.5, width heterogeneity h_Delta=0.1 (centre spread = data fit).  Both layers
are driven by the SAME piecewise-constant input protocol I(t) chosen (from the fold structure) so
that the L2/3 model is walked through THREE of its coexisting stable fixed points (low / intermediate
"inner-cusp" / high), while the L5/6 model only toggles between TWO (low / high).  For each layer we
run the spiking QIF network (forward Euler) and the Ott-Antonsen mean field (solve_ivp) and save r(t).

    python pyramidal_fig_rate.py            # both layers
Run in the ``allen`` conda env (PyRates PopulationTemplate + numba + scipy).
"""
import os
import sys
from time import perf_counter

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import allen_qif_meanfield as MF

CELL_CLASS = "Pyramidal"
LAYERS = ["L2/3", "L5/6"]
HD = 0.1                                  # width-heterogeneity knob h_Delta
# piecewise-constant input protocol: (I level, duration). 100 -> 160 -> 260 -> 360 -> 100,
# five equal phases over T=500 (both models start on the lower branch at I=100)
PROTO = [(150.0, 100.0), (175.0, 100.0), (150.0, 100.0), (250.0, 100.0), (300.0, 100.0), (250.0, 100.0), (150.0, 100.0)]

P = dict(v_r=-70.0, J=100.0, tau_s=0.5,
         dt=2e-4, dts=0.5, N=10000, v_peak=400.0, v_reset=-600.0, theta_clip=50.0, seed=0)
P["T"] = float(sum(d for _, d in PROTO))


def proto_arrays():
    """(level, t_start, t_end) rows of the protocol."""
    rows = []; acc = 0.0
    for I, d in PROTO:
        rows.append((I, acc, acc + d)); acc += d
    return rows


def I_of_t(t):
    for I, t0, t1 in proto_arrays():
        if t0 <= t < t1:
            return I
    return PROTO[-1][0]


def input_per_step(dt, steps):
    rows = proto_arrays()
    I = np.empty(steps)
    for k in range(steps):
        t = k * dt; I[k] = rows[-1][0]
        for lev, t0, t1 in rows:
            if t0 <= t < t1:
                I[k] = lev; break
    return I


# ── spiking QIF network: forward Euler with a per-step input array ──────────────
def _euler_loop(f, y, extra, p_I, p_r, v0, v1, s0, s1, N, dt, steps, sr, vp, vreset, I_arr):
    n_save = steps // sr + 1
    t_rec = np.empty(n_save); r_rec = np.empty(n_save)
    ny = y.shape[0]; spike_accum = 0; rate = 0.0; ss = 0
    for k in range(steps):
        p_I[0] = I_arr[k]
        p_r[0] = rate
        dy = f(k * dt, y, *extra)
        for i in range(ny):
            y[i] += dt * dy[i]
        nsp = 0
        for i in range(v0, v1):
            if y[i] >= vp:
                y[i] = vreset; nsp += 1
        rate = nsp / (N * dt); spike_accum += nsp
        if k % sr == 0:
            t_rec[ss] = k * dt; r_rec[ss] = spike_accum / (N * sr * dt)
            spike_accum = 0; ss += 1
    return t_rec[:ss], r_rec[:ss]


_euler_jit = njit(_euler_loop)


def run_micro(P, v_th):
    func, args, keys, vmap, net = MF.build_micro(P, v_th)
    from pyrates import clear
    N = v_th.size; extra = tuple(args[2:]); f = njit(func)
    f(0.0, np.asarray(args[1], float), *extra); clear(net)
    y = np.asarray(args[1], float).copy()
    v_sl = MF._state_slice(vmap, "qif_mem/v"); s_sl = MF._state_slice(vmap, "alpha_syn/s")
    p_I = MF._param(args, keys, "qif_mem/Iext").reshape(1)
    p_r = MF._param(args, keys, "alpha_syn/rin").reshape(1)
    dt, steps = P["dt"], int(round(P["T"] / P["dt"])); sr = max(1, int(round(P["dts"] / dt)))
    I_arr = input_per_step(dt, steps)
    return _euler_jit(f, y, extra, p_I, p_r, v_sl.start, v_sl.stop, s_sl.start, s_sl.stop,
                      N, dt, steps, sr, P["v_peak"], P["v_reset"], I_arr)


def run_mf(P, w, vth_bar, Delta):
    func, args, keys, vmap, net = MF.build_mf(P, vth_bar, Delta)
    from pyrates import clear
    extra = args[2:]; f = njit(func); y0 = np.asarray(args[1], float)
    f(0.0, y0, *extra); clear(net)
    r_sl = MF._state_slice(vmap, "oa_op/r"); s_sl = MF._state_slice(vmap, "alpha_syn/s")
    p_I = MF._param(args, keys, "oa_op/Iext"); p_r = MF._param(args, keys, "alpha_syn/rin")

    def rhs(t, y):
        p_I.fill(I_of_t(t)); p_r.fill(w @ y[r_sl])
        return np.array(f(t, y, *extra))

    t_eval = np.arange(0.0, P["T"], P["dts"])
    sol = solve_ivp(rhs, (0.0, P["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=1e-7, atol=1e-9, max_step=P["tau_s"])
    return sol.t, w @ sol.y[r_sl, :]


def main():
    rng = np.random.default_rng(P["seed"])
    for layer in LAYERS:
        tag = MF._tag(CELL_CLASS, layer)
        fit = os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{tag}.npz")
        w, Om, De, M = MF.load_fit(fit)
        De_h = HD * De                         # width heterogeneity h_Delta; centres unchanged
        vth_bar = P["v_r"] + Om
        print(f"== {layer}  M={M}  J={P['J']}  tau_s={P['tau_s']}  hD={HD} ==")

        v_th = MF.sample_thresholds(w, Om, De_h, P["N"], P["v_r"], P["theta_clip"], rng)
        t0 = perf_counter(); tm, rm = run_micro(P, v_th)
        print(f"   micro done in {perf_counter()-t0:.1f}s")
        t0 = perf_counter(); tf, rf = run_mf(P, w, vth_bar, De_h)
        print(f"   mean field done in {perf_counter()-t0:.1f}s")

        Iarr = np.array([I_of_t(t) for t in tf])
        out = os.path.join(_HERE, f"pyramidal_fig_rate_{tag}.npz")
        np.savez(out, cell_class=CELL_CLASS, layer=layer, hd=float(HD),
                 t_micro=tm, r_micro=rm, t_mf=tf, r_mf=rf, t_input=tf, input=Iarr,
                 proto=np.array(PROTO, float), J=float(P["J"]), tau_s=float(P["tau_s"]))
        print(f"   [saved] {os.path.basename(out)}")


if __name__ == "__main__":
    main()
