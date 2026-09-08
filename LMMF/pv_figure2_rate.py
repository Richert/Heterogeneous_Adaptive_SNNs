r"""
PV+ interneuron — figure 2, rate-dynamics part: heterogeneity step (QIF net vs. mean field)
============================================================================================

Fixed input I=I_fix and coupling J (per the two-knob oscillatory regime); instead of an
input pulse we STEP THE HETEROGENEITY down at t_step.  Both knobs go from the data fit
(h_eta=1, h_Delta=1, quiescent) into the oscillatory wedge (h_eta=0, h_Delta=h_lo):

    centre_m(h_eta) = Ombar + h_eta (Omega_m - Ombar),   Ombar = sum_m w_m Omega_m
    width_m(h_Delta) = h_Delta * Delta_m

so reducing heterogeneity collapses the threshold centres onto their mean and narrows the
component widths -> the near-homogeneous inhibitory population switches on a synchronous
(inhibition-based) oscillation.

Mean field: inject time-varying ``oa_op/Delta`` and ``oa_op/vthbar`` each RHS call.
QIF net:   each neuron i keeps its base centre Omega_{m_i} and original deviate
           dev_i = Delta_{m_i} tan(...) ; its threshold is recomputed at the step as
           v_{th,i} = v_r + Ombar + h_eta (Omega_{m_i}-Ombar) + h_Delta dev_i.

ONE LAYER PER PROCESS:
    python pv_figure2_rate.py "PV+ Interneuron" "L2/3"
    python pv_figure2_rate.py "PV+ Interneuron" "L5/6"
Run in the ``allen`` conda env.
"""
import os
import sys
from time import perf_counter

import numpy as np
from numba import njit

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import allen_qif_meanfield as MF       # reuse build_micro/build_mf, _state_slice, _param, load_fit, _tag

# fixed regime (matches pv_figure2_bifurcation.py): J=-100, tau_s=0.5, I=I_fix per layer.
REGIME = {("PV+ Interneuron", "L2/3"): dict(I_fix=430.0, h_lo=0.05),   # common input I; both layers
          ("PV+ Interneuron", "L5/6"): dict(I_fix=430.0, h_lo=0.05)}   # differ ONLY in distribution

P = dict(v_r=-70.0, J=-100.0, tau_s=0.5,
         T=400.0, t_step=120.0, dt=2e-4, dts=0.1,
         N=10000, v_peak=400.0, v_reset=-600.0, theta_clip=50.0, seed=0)


def sample_thresholds(w, Omega, Delta, N, v_r, clip, rng):
    """Draw N thresholds AND keep the per-neuron component centre + original deviate so the
    heterogeneity step can rescale them: v_th = v_r + Omega_{m} + dev,  dev = Delta_m tan(...)."""
    comp = rng.choice(len(w), size=N, p=w)
    dev = Delta[comp] * np.tan(np.pi * (rng.random(N) - 0.5))
    dev = np.clip(Omega[comp] + dev, -clip, clip) - Omega[comp]   # clip the *threshold*, keep centre
    base_centre = Omega[comp]
    v_th = v_r + base_centre + dev
    return v_th, base_centre, dev


def _euler_loop_hstep(f, y, extra, p_I, p_r, p_vth, base_centre, dev,
                      v0, v1, s0, s1, N, dt, steps, sr, vp, vreset,
                      I_fix, Ombar, vr, t_step, he0, he1, hd0, hd1):
    """Forward Euler with a one-shot heterogeneity step at t_step (recompute v_th in place)."""
    n_save = steps // sr + 1
    t_rec = np.empty(n_save); r_rec = np.empty(n_save); s_rec = np.empty(n_save); v_rec = np.empty(n_save)
    ny = y.shape[0]; spike_accum = 0; rate = 0.0; ss = 0; stepped = False
    # initial thresholds at (he0, hd0)
    for i in range(N):
        p_vth[i] = vr + Ombar + he0 * (base_centre[i] - Ombar) + hd0 * dev[i]
    for k in range(steps):
        t = k * dt
        if (not stepped) and t >= t_step:
            for i in range(N):
                p_vth[i] = vr + Ombar + he1 * (base_centre[i] - Ombar) + hd1 * dev[i]
            stepped = True
        p_I[0] = I_fix
        p_r[0] = rate
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
            r_rec[ss] = spike_accum / (N * sr * dt)
            s_rec[ss] = np.mean(y[s0:s1])
            v_rec[ss] = np.median(y[v0:v1])
            spike_accum = 0; ss += 1
    return t_rec[:ss], r_rec[:ss], s_rec[:ss], v_rec[:ss]


_euler_loop_hstep_jit = njit(_euler_loop_hstep)


def run_micro_hstep(P, v_th, base_centre, dev, Ombar, I_fix, he0, he1, hd0, hd1):
    func, args, keys, vmap, net = MF.build_micro(P, v_th)
    N = v_th.size
    extra = tuple(args[2:])
    f = njit(func)
    f(0.0, np.asarray(args[1], float), *extra)
    MF.clear(net)
    y = np.asarray(args[1], float).copy()
    v_sl = MF._state_slice(vmap, "qif_mem/v")
    s_sl = MF._state_slice(vmap, "alpha_syn/s")
    p_I = MF._param(args, keys, "qif_mem/Iext").reshape(1)
    p_r = MF._param(args, keys, "alpha_syn/rin").reshape(1)
    p_vth = MF._param(args, keys, "qif_mem/vth")
    dt, steps = P["dt"], int(round(P["T"] / P["dt"]))
    sr = max(1, int(round(P["dts"] / dt)))
    return _euler_loop_hstep_jit(f, y, extra, p_I, p_r, p_vth, base_centre, dev,
                                 v_sl.start, v_sl.stop, s_sl.start, s_sl.stop, N, dt, steps, sr,
                                 P["v_peak"], P["v_reset"], I_fix, Ombar, P["v_r"],
                                 P["t_step"], he0, he1, hd0, hd1)


def run_mf_hstep(P, w, Omega, Delta, I_fix, he0, he1, hd0, hd1):
    """Mean field with the heterogeneity step injected into Delta_m(t) and vthbar_m(t)."""
    from scipy.integrate import solve_ivp
    Ombar = float(w @ Omega)
    vth0 = P["v_r"] + Ombar + he0 * (Omega - Ombar)          # only to build the template
    func, args, keys, vmap, net = MF.build_mf(P, vth0, hd0 * Delta)
    extra = args[2:]
    f = njit(func)
    y0 = np.asarray(args[1], float)
    f(0.0, y0, *extra)
    MF.clear(net)
    r_sl = MF._state_slice(vmap, "oa_op/r"); v_sl = MF._state_slice(vmap, "oa_op/v")
    s_sl = MF._state_slice(vmap, "alpha_syn/s")
    p_I = MF._param(args, keys, "oa_op/Iext")
    p_r = MF._param(args, keys, "alpha_syn/rin")
    p_De = MF._param(args, keys, "oa_op/Delta")
    p_vt = MF._param(args, keys, "oa_op/vthbar")
    t_step = P["t_step"]

    def rhs(t, y):
        he, hd = (he1, hd1) if t >= t_step else (he0, hd0)
        p_De[:] = hd * Delta
        p_vt[:] = P["v_r"] + Ombar + he * (Omega - Ombar)
        p_I.fill(I_fix)
        p_r.fill(w @ y[r_sl])
        return np.array(f(t, y, *extra))

    t_eval = np.arange(0.0, P["T"], P["dts"])
    sol = solve_ivp(rhs, (0.0, P["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=1e-7, atol=1e-9, max_step=P["tau_s"])
    r = w @ sol.y[r_sl, :]; s = sol.y[s_sl, :].mean(axis=0); v = w @ sol.y[v_sl, :]
    return sol.t, r, s, v


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cell_class = args[0] if args else "PV+ Interneuron"
    layer = args[1] if len(args) > 1 else "L2/3"
    tag = MF._tag(cell_class, layer)
    reg = REGIME[(cell_class, layer)]
    I_fix, h_lo = reg["I_fix"], reg["h_lo"]
    fit = os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{tag}.npz")
    w, Omega, Delta, M = MF.load_fit(fit)
    Ombar = float(w @ Omega)
    he0, he1, hd0, hd1 = 1.0, h_lo, 1.0, h_lo        # single knob: h_eta = h_Delta = h, steps 1 -> h_lo
    print(f"== {cell_class} {layer}  M={M}  I={I_fix}  J={P['J']}  tau_s={P['tau_s']} ==")
    print(f"   step at t={P['t_step']}: single knob h {1.0} -> {h_lo}  (Ombar={Ombar:.2f})")

    rng = np.random.default_rng(P["seed"])
    v_th, base_centre, dev = sample_thresholds(w, Omega, Delta, P["N"], P["v_r"], P["theta_clip"], rng)

    t0 = perf_counter()
    print(f"[mean field] M-ensemble OA, heterogeneity step ...")
    tf, rf, sf, vf = run_mf_hstep(P, w, Omega, Delta, I_fix, he0, he1, hd0, hd1)
    print(f"   done in {perf_counter()-t0:.1f}s;  r(before)~{rf[(tf<P['t_step'])][-1]:.3f}, r(end)~{rf[-1]:.3f}")

    t0 = perf_counter()
    print(f"[micro] spiking QIF net N={P['N']}, heterogeneity step ...")
    tm, rm, sm, vm = run_micro_hstep(P, v_th, base_centre, dev, Ombar, I_fix, he0, he1, hd0, hd1)
    print(f"   done in {perf_counter()-t0:.1f}s")

    out = os.path.join(_HERE, f"pv_fig2_rate_{tag}.npz")
    np.savez(out, cell_class=cell_class, layer=layer, tag=tag, M=np.int64(M),
             t_micro=tm, r_micro=rm, s_micro=sm, t_mf=tf, r_mf=rf, s_mf=sf, v_mf=vf,
             I_fix=float(I_fix), J=float(P["J"]), tau_s=float(P["tau_s"]),
             t_step=float(P["t_step"]), he0=he0, he1=he1, hd0=hd0, hd1=hd1)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
