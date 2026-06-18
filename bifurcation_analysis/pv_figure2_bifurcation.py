r"""
PV+ interneuron — figure 2, bifurcation part: TWO-knob heterogeneity continuation
=================================================================================

Pure PyCoBi/Auto-07p. Two heterogeneity knobs (build_circuit default, combined=False):
  h_Delta (hD) = width scaling,   h_eta (hC) = centre-spread scaling   (both = 1 at the data fit).
Fixed regime J=-200, tau_s=2, I=I_fix (per layer). Per layer it computes
  - the 1-D bifurcation r(h_Delta) at collapsed centres (h_eta=0): equilibrium + Hopf + limit cycle,
  - the 2-D Hopf (+ period-doubling) loci in the (h_eta, h_Delta) heterogeneity plane,
and saves them to a self-contained .npy for the figure-assembly script.

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

REGIME = {("PV+ Interneuron", "L2/3"): dict(I_fix=490.6),
          ("PV+ Interneuron", "L5/6"): dict(I_fix=387.8)}
CONFIG = dict(v_r=-70.0, J=-200.0, tau_s=2.0, r0=0.05, T_settle=1000.0,
              HD_MAX=1.0, HC_MAX=1.5)      # h_Delta sweep 0..1 (1-D); h_eta up to 1.5 (2-D plane)

# extra Hopf loci in (drive/coupling x heterogeneity) planes (x-axis = J or I).  The h_Delta
# planes are traced at h_eta=0 from the regime Hopf HB1; the h_eta planes are traced at a fixed
# mid-wedge h_Delta=h_lo (the rate-sim value) where the oscillatory region has real h_eta extent,
# seeded by an equilibrium continuation in h_eta to its Hopf.
# RL0/RL1 bound the primary (J/I); UZSTOP bounds the heterogeneity knob; DSMAX scales to J/I.
LOCI_BOUNDS = {"J": (-400.0, -5.0), "Iext": (0.0, 900.0), "hD": (2e-3, 1.2), "hC": (-1e-3, 1.5)}
LOCI_DSMAX = {"J": 4.0, "Iext": 9.0}
LOCI_HDFIX = {"pv_L23": 0.07, "pv_L56": 0.10}    # fixed h_Delta for the h_eta planes (matches rate sim)


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
    print(f"== {cell_class} {layer}  M={M}  J={cfg['J']}  tau_s={cfg['tau_s']}  I={I_fix}  (two-knob) ==")
    r0 = np.full(M, cfg["r0"]); v0 = np.full(M, v_r)
    circuit = B.build_circuit(M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], I_fix, r0, v0,
                              hD0=1.0, hC0=0.0)                 # centres collapsed for the 1-D cut
    ode = ODESystem.from_template(circuit, auto_dir=B.AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))

    # (1) settle at h_Delta = 1, h_eta = 0
    print(f"[1] IVP settle (T={cfg['T_settle']}) at h_Delta=1, h_eta=0")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: cfg["T_settle"]}, STOP={"UZ1"})

    # (2) equilibrium continuation in h_Delta -> Hopf
    print("[2] equilibrium continuation in h_Delta")
    h_lo = LOCI_HDFIX[tag]
    eq, _ = ode.run(origin="time", starting_point="UZ1", name="eq_hD", c="eq",
                    ICP="hD", bidirectional=True, RL0=0.0, RL1=cfg["HD_MAX"],
                    IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=10, UZR={"hD": h_lo},
                    DSMIN=1e-9, DSMAX=5e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, get_stability=True)
    hbs = B._bif_vals(eq, "HB", "hD")
    print(f"   HB at h_Delta = {[round(x, 4) for x in hbs]}")

    # (3) limit cycle from the first Hopf (detect PD)
    lc_conts = []; pd_hd = None
    n_hb = len(eq[eq[("bifurcation", "")] == "HB"])
    for k in range(1, n_hb + 1):
        nm = f"lc_{k}"
        try:
            lc, _ = ode.run(origin="eq_hD", starting_point=f"HB{k}", name=nm,
                            IPS=2, ISW=-1, ICP=["hD", 11], NTST=400, NCOL=4, ILP=1, ISP=2,
                            NMX=8000, NPR=10, DS=1e-3, DSMIN=1e-12, DSMAX=2e-3,
                            EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, RL0=0.0, RL1=cfg["HD_MAX"],
                            get_stability=True)
            lc_conts.append(nm)
            print(f"   {nm}: limit cycle from HB{k}, {len(lc)} pts;  bifs={dict(lc['bifurcation'].value_counts())}")
            if nm == "lc_1":
                _pd = B._bif_vals(lc, "PD", "hD"); pd_hd = _pd[0] if _pd else None
                if pd_hd is not None:
                    print(f"       PD at h_Delta = {pd_hd:.4f}")
        except Exception as e:
            print(f"   {nm}: FAILED ({type(e).__name__}: {e})")

    # (4) 2-D loci in the (h_eta, h_Delta) heterogeneity plane: Hopf, and PD if the cycle has one.
    #     RL0=3e-3 on hD (the zero-width limit is singular); hC bounded to [0, HC_MAX].
    two_d = {}
    common = dict(ICP=["hD", "hC"], bidirectional=True, RL0=3e-3, RL1=cfg["HD_MAX"],
                  NPR=10, DS=1e-3, DSMIN=1e-12, DSMAX=2e-2,
                  EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, UZSTOP={"hC": [-1e-3, cfg["HC_MAX"]]})
    periodic = dict(IPS=2, ISW=2, ISP=0, ILP=0, NTST=200, NCOL=4, NMX=4000, get_stability=False)
    try:
        hb2, _ = ode.run(origin="eq_hD", starting_point="HB1", name="hopf_2d", c="eq",
                         IPS=1, ISW=2, ISP=0, ILP=0, NMX=8000, get_stability=False, **common)
        two_d["hopf"] = "hopf_2d"
        print(f"   hopf_2d: {len(hb2)} pts, hD∈[{hb2[B._pcol(hb2,'hD')].min():.3f},{hb2[B._pcol(hb2,'hD')].max():.3f}]"
              f" hC∈[{hb2[B._pcol(hb2,'hC')].min():.3f},{hb2[B._pcol(hb2,'hC')].max():.3f}]")
    except Exception as e:
        print(f"   hopf_2d: FAILED ({type(e).__name__}: {e})")
    if pd_hd is not None:
        try:
            pd2d, _ = ode.run(origin="lc_1", starting_point="PD1", name="pd_2d", **periodic, **common)
            two_d["pd"] = "pd_2d"
            print(f"   pd_2d: {len(pd2d)} pts, hD∈[{pd2d[B._pcol(pd2d,'hD')].min():.3f},{pd2d[B._pcol(pd2d,'hD')].max():.3f}]"
                  f" hC∈[{pd2d[B._pcol(pd2d,'hC')].min():.3f},{pd2d[B._pcol(pd2d,'hC')].max():.3f}]")
        except Exception as e:
            print(f"   pd_2d: FAILED ({type(e).__name__}: {e})")

    # (5) Hopf loci in the (drive/coupling x heterogeneity) planes (x-axis J or I).
    loci = {}

    def _locus(origin, start, p1, knob, seed):
        """Continue a Hopf in (p1, knob) bidirectionally; p1 -> x, knob -> y. `seed` is the
        starting point (p1, knob) used to bridge the bidirectional join when plotting."""
        nm = f"locus_{p1}_{knob}"
        try:
            ode.run(origin=origin, starting_point=start, name=nm, c="eq", ICP=[p1, knob],
                    IPS=1, ISW=2, ISP=0, ILP=0, bidirectional=True,
                    RL0=LOCI_BOUNDS[p1][0], RL1=LOCI_BOUNDS[p1][1], UZSTOP={knob: list(LOCI_BOUNDS[knob])},
                    NMX=8000, NPR=1, DS=1e-2, DSMIN=1e-10, DSMAX=LOCI_DSMAX[p1],
                    EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, get_stability=False)
            a = _arr(ode, nm, p1, knob, with_stab=False)
            loci[f"{p1}_{knob}"] = dict(x=a["x"], y=a["ymax"], seed=(float(seed[0]), float(seed[1])))
            print(f"   {nm}: {a['x'].size} pts;  {p1}∈[{a['x'].min():.2f},{a['x'].max():.2f}]  "
                  f"{knob}∈[{a['ymax'].min():.3f},{a['ymax'].max():.3f}]")
        except Exception as e:
            print(f"   {nm}: FAILED ({type(e).__name__}: {e})")

    hD_star = hbs[0] if hbs else np.nan                          # regime Hopf width (h_eta=0)
    # h_Delta planes (h_eta=0): seed from the regime Hopf HB1 on eq_hD
    _locus("eq_hD", "HB1", "J", "hD", seed=(cfg["J"], hD_star))
    _locus("eq_hD", "HB1", "Iext", "hD", seed=(I_fix, hD_star))
    # h_eta planes (h_Delta=h_lo): from the unstable equilibrium at hD=h_lo (UZ on eq_hD),
    # continue in h_eta to its Hopf, then trace the Hopf in (J/I, h_eta).
    try:
        eqc, _ = ode.run(origin="eq_hD", starting_point="UZ1", name="eq_hC", c="eq", ICP="hC",
                         RL0=-1e-3, RL1=LOCI_BOUNDS["hC"][1], IPS=1, ILP=0, ISP=2, ISW=1,
                         NMX=8000, NPR=10, DSMIN=1e-10, DSMAX=2e-3,
                         EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, get_stability=True)
        hbc = B._bif_vals(eqc, "HB", "hC")
        print(f"   eq_hC (hD={h_lo}): HB at h_eta = {[round(x, 4) for x in hbc]}")
        if hbc:
            _locus("eq_hC", "HB1", "J", "hC", seed=(cfg["J"], hbc[0]))
            _locus("eq_hC", "HB1", "Iext", "hC", seed=(I_fix, hbc[0]))
    except Exception as e:
        print(f"   eq_hC: FAILED ({type(e).__name__}: {e})")

    data = dict(cell_class=cell_class, layer=layer, tag=tag, M=int(M), I_fix=float(I_fix),
                J=float(cfg["J"]), hd_fix=float(h_lo),
                eq=_arr(ode, "eq_hD", "hD", "s"),
                lc=[_arr(ode, nm, "hD", "s") for nm in lc_conts],
                hopf=(_arr(ode, "hopf_2d", "hD", "hC", with_stab=False) if "hopf" in two_d else None),
                pd=(_arr(ode, "pd_2d", "hD", "hC", with_stab=False) if "pd" in two_d else None),
                loci=loci, seeds={})
    if "hopf" in two_d and hbs:
        data["seeds"]["hopf"] = (float(hbs[0]), 0.0)            # (h_Delta*, h_eta=0)
    if "pd" in two_d and pd_hd is not None:
        data["seeds"]["pd"] = (float(pd_hd), 0.0)
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
