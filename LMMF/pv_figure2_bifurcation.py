r"""
PV+ interneuron — figure 2, bifurcation part: SINGLE-knob heterogeneity continuation
====================================================================================

Pure PyCoBi/Auto-07p. ONE heterogeneity knob h (build_circuit combined=True): h scales BOTH the
centre spread and the widths, so h=0 is fully homogeneous (one delta at the weighted mean), h=1 is
the data fit, h>1 is more heterogeneous than the data.  Fixed regime J=-100, tau_s=0.5, common
I=430 (the two layers differ ONLY in their fitted distribution).  Per layer it computes
  - the 1-D bifurcation s(h): equilibrium + Hopf + limit cycle (+ period-doubling),
  - the Hopf loci in the (control parameter x h) planes: (J,h), (I,h), (tau_s,h),
and saves them to a self-contained .npy for the figure-assembly scripts.

ONE LAYER PER PROCESS (two ODESystems in one process collide on shared continuation names):
    python pv_figure2_bifurcation.py "PV+ Interneuron" "L2/3"
    python pv_figure2_bifurcation.py "PV+ Interneuron" "L5/6"
Run in the ``pycobi`` conda env.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import allen_qif_bifurcation as B
from pycobi import ODESystem

REGIME = {("PV+ Interneuron", "L2/3"): dict(I_fix=430.0),    # common input: the two layers
          ("PV+ Interneuron", "L5/6"): dict(I_fix=430.0)}    # differ ONLY in their distribution
CONFIG = dict(v_r=-70.0, J=-100.0, tau_s=0.5, r0=0.05, T_settle=1000.0, H_MAX=1.5)

# Hopf loci in (control parameter x h) planes (x-axis = J | I | tau_s, y-axis = h).
# RL0/RL1 bound the primary control parameter; UZSTOP bounds h; DSMAX scales to the control param.
LOCI_BOUNDS = {"J": (-400.0, -5.0), "Iext": (0.0, 900.0), "tau_s": (0.1, 2.0), "h": (2e-3, 1.5)}
LOCI_DSMAX = {"J": 4.0, "Iext": 9.0, "tau_s": 0.05}


def _arr(ode, cont, xname, yname, with_stab=True):
    """(x, y_min, y_max, stability, bifurcation) from a continuation summary; limit-cycle branches
    carry two columns per state var (min, max) -> envelope, equilibria one (min==max)."""
    summ = ode.get_summary(cont)
    head = lambda n: [c for c in summ.columns if (c[0] if isinstance(c, tuple) else c) == n]
    x = np.asarray(summ[head(xname)[0]], float)
    Y = np.column_stack([np.asarray(summ[c], float) for c in head(yname)])
    bif = np.asarray(summ[head("bifurcation")[0]]).astype(str)
    sc = head("stability")
    stab = np.asarray(summ[sc[0]], bool) if (with_stab and sc) else np.ones(x.size, bool)
    return dict(x=x, ymin=Y.min(axis=1), ymax=Y.max(axis=1), stab=stab, bif=bif)


def run_layer(cfg, cell_class, layer):
    tag = B._tag(cell_class, layer)
    I_fix = REGIME[(cell_class, layer)]["I_fix"]
    w, Om, De, M = B.load_fit(os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{tag}.npz"))
    v_r = cfg["v_r"]
    print(f"== {cell_class} {layer}  M={M}  J={cfg['J']}  tau_s={cfg['tau_s']}  I={I_fix}  (single-h) ==")
    r0 = np.full(M, cfg["r0"]); v0 = np.full(M, v_r)
    circuit = B.build_circuit(M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], I_fix, r0, v0,
                              combined=True, h0=1.0)                # single heterogeneity knob h
    ode = ODESystem.from_template(circuit, auto_dir=B.AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))

    # (1) settle at h = 1 (the data fit)
    print(f"[1] IVP settle (T={cfg['T_settle']}) at h=1")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: cfg["T_settle"]}, STOP={"UZ1"})

    # (2) equilibrium continuation in h -> Hopf  (RL0=3e-3 avoids the spurious h=0 boundary HB)
    print("[2] equilibrium continuation in h")
    eq, _ = ode.run(origin="time", starting_point="UZ1", name="eq_h", c="eq",
                    ICP="h", bidirectional=True, RL0=3e-3, RL1=cfg["H_MAX"],
                    IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=10,
                    DSMIN=1e-9, DSMAX=5e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, get_stability=True)
    hbs = B._bif_vals(eq, "HB", "h")
    print(f"   HB at h = {[round(x, 4) for x in hbs]}")
    h_star = hbs[0] if hbs else np.nan

    # (3) limit cycle from each Hopf (detect period-doubling)
    lc_conts = []; pd_h = None
    n_hb = len(eq[eq[("bifurcation", "")] == "HB"])
    for k in range(1, n_hb + 1):
        nm = f"lc_{k}"
        try:
            # NMX/DSMAX raised so the cycle is tracked through the period-diverging
            # (homoclinic-type) approach all the way down to h->0 (RL0=2e-3).
            lc, _ = ode.run(origin="eq_h", starting_point=f"HB{k}", name=nm,
                            IPS=2, ISW=-1, ICP=["h", 11], NTST=400, NCOL=4, ILP=1, ISP=2,
                            NMX=40000, NPR=10, DS=1e-3, DSMIN=1e-12, DSMAX=5e-3,
                            EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, RL0=2e-3, RL1=cfg["H_MAX"],
                            get_stability=True)
            lc_conts.append(nm)
            print(f"   {nm}: limit cycle from HB{k}, {len(lc)} pts;  bifs={dict(lc['bifurcation'].value_counts())}")
            if nm == "lc_1":
                _pd = B._bif_vals(lc, "PD", "h"); pd_h = _pd[0] if _pd else None
                if pd_h is not None:
                    print(f"       PD at h = {pd_h:.4f}")
        except Exception as e:
            print(f"   {nm}: FAILED ({type(e).__name__}: {e})")

    # (4) Hopf loci in the (control parameter x h) planes, all from the regime Hopf HB1
    loci = {}

    def _locus(p1, seed):
        nm = f"locus_{p1}_h"
        try:
            ode.run(origin="eq_h", starting_point="HB1", name=nm, c="eq", ICP=[p1, "h"],
                    IPS=1, ISW=2, ISP=0, ILP=0, bidirectional=True,
                    RL0=LOCI_BOUNDS[p1][0], RL1=LOCI_BOUNDS[p1][1], UZSTOP={"h": list(LOCI_BOUNDS["h"])},
                    NMX=8000, NPR=1, DS=1e-2, DSMIN=1e-10, DSMAX=LOCI_DSMAX[p1],
                    EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, get_stability=False)
            a = _arr(ode, nm, p1, "h", with_stab=False)
            loci[f"{p1}_h"] = dict(x=a["x"], y=a["ymax"], seed=(float(seed[0]), float(seed[1])))
            print(f"   {nm}: {a['x'].size} pts;  {p1}∈[{a['x'].min():.2f},{a['x'].max():.2f}]  "
                  f"h∈[{a['ymax'].min():.3f},{a['ymax'].max():.3f}]")
        except Exception as e:
            print(f"   {nm}: FAILED ({type(e).__name__}: {e})")

    _locus("J", (cfg["J"], h_star))
    _locus("Iext", (I_fix, h_star))
    _locus("tau_s", (cfg["tau_s"], h_star))

    data = dict(cell_class=cell_class, layer=layer, tag=tag, M=int(M), I_fix=float(I_fix),
                J=float(cfg["J"]), tau_s=float(cfg["tau_s"]), h_star=float(h_star),
                eq=_arr(ode, "eq_h", "h", "s"),
                lc=[_arr(ode, nm, "h", "s") for nm in lc_conts],
                loci=loci)
    np.save(os.path.join(_HERE, f"pv_fig2_bif_{tag}.npy"), data, allow_pickle=True)
    print(f"   [saved] pv_fig2_bif_{tag}.npy")
    return data


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        raise SystemExit('usage: pv_figure2_bifurcation.py "<cell_class>" "<layer>"')
    run_layer(CONFIG, args[0], args[1])


if __name__ == "__main__":
    main()
