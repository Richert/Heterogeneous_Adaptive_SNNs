r"""
PV+ interneuron — exploratory Hopf check at J=-100, tau_s=0.5
============================================================
Question: do we find Hopf bifurcations for a continuation in the CENTRE-spread knob h_eta (= hC)
alone (widths fixed at the data fit, hD=1), instead of the lumped single knob h (which scales both
centres and widths)?  Regime J=-100, tau_s=0.5.  For context we also run the lumped-h continuation
and the width-only (hD, hC=1) continuation at the same operating point.

ONE LAYER PER PROCESS (two ODESystems collide on shared continuation names):
    python pv_heta_check.py "PV+ Interneuron" "L2/3"
    python pv_heta_check.py "PV+ Interneuron" "L5/6"
Run in the ``pycobi`` conda env.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import allen_qif_bifurcation as B
from pycobi import ODESystem

CONFIG = dict(v_r=-70.0, J=-100.0, tau_s=0.5, I_fix=430.0, r0=0.05, T_settle=1000.0, H_MAX=1.5)


def _settle(ode, cfg):
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: cfg["T_settle"]}, STOP={"UZ1"})


def _cont_knob(ode, cfg, icp, name):
    """Equilibrium continuation in one heterogeneity knob from the data-fit settle; report HBs/LPs."""
    eq, _ = ode.run(origin="time", starting_point="UZ1", name=name, c="eq",
                    ICP=icp, bidirectional=True, RL0=3e-3, RL1=cfg["H_MAX"],
                    IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=200,
                    DSMIN=1e-9, DSMAX=5e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, get_stability=True)
    hbs = B._bif_vals(eq, "HB", icp)
    lps = B._bif_vals(eq, "LP", icp)
    return eq, hbs, lps


def run_layer(cfg, cell_class, layer):
    tag = B._tag(cell_class, layer)
    w, Om, De, M = B.load_fit(os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{tag}.npz"))
    v_r = cfg["v_r"]
    print(f"\n==== {cell_class} {layer}  M={M}  J={cfg['J']}  tau_s={cfg['tau_s']}  I={cfg['I_fix']} ====")
    r0 = np.full(M, cfg["r0"]); v0 = np.full(M, v_r)

    results = {}
    # ---- (A) lumped single knob h (centres AND widths together) ------------------------------
    circ = B.build_circuit(M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], cfg["I_fix"], r0, v0,
                           combined=True, h0=1.0)
    ode = ODESystem.from_template(circ, auto_dir=B.AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))
    _settle(ode, cfg)
    _, hb, lp = _cont_knob(ode, cfg, "h", "eq_h")
    print(f"[lumped h ]  HB at h   = {[round(x, 4) for x in hb]}   (LP at {[round(x,4) for x in lp]})")
    results["h"] = hb
    ode.close_session(clear_files=True)

    # ---- (B) centre-spread only: vary hC, widths fixed at data (hD=1) ------------------------
    circ = B.build_circuit(M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], cfg["I_fix"], r0, v0,
                           combined=False, hD0=1.0, hC0=1.0)
    ode = ODESystem.from_template(circ, auto_dir=B.AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))
    _settle(ode, cfg)
    _, hb, lp = _cont_knob(ode, cfg, "hC", "eq_hC")
    print(f"[h_eta=hC ]  HB at hC  = {[round(x, 4) for x in hb]}   (LP at {[round(x,4) for x in lp]})")
    results["hC"] = hb
    ode.close_session(clear_files=True)

    # ---- (C) width only: vary hD, centres fixed at data (hC=1) — for context -----------------
    circ = B.build_circuit(M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], cfg["I_fix"], r0, v0,
                           combined=False, hD0=1.0, hC0=1.0)
    ode = ODESystem.from_template(circ, auto_dir=B.AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))
    _settle(ode, cfg)
    _, hb, lp = _cont_knob(ode, cfg, "hD", "eq_hD")
    print(f"[h_Delta  ]  HB at hD  = {[round(x, 4) for x in hb]}   (LP at {[round(x,4) for x in lp]})")
    results["hD"] = hb
    ode.close_session(clear_files=True)

    print(f"---- summary {layer}: lumped-h HB {len(results['h'])}, hC HB {len(results['hC'])}, "
          f"hD HB {len(results['hD'])} ----")
    return results


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        raise SystemExit('usage: pv_heta_check.py "<cell_class>" "<layer>"')
    run_layer(CONFIG, args[0], args[1])


if __name__ == "__main__":
    main()
