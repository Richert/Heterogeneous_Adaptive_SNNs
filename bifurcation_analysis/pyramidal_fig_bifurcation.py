r"""
Pyramidal journal figure (updated) — bifurcation data
======================================================
Regime J=100, tau_s=0.5, two-knob heterogeneity with the WIDTH knob h_Delta as the second
continuation parameter (centre spread h_C fixed at the data fit = 1).  Per layer it produces,
into a self-contained .npz read by the figure script (so the figure needs no PyCoBi):

  * the 1-D equilibrium branch s(I) at h_Delta = HD_CUT (the multi-stable slice shown in the
    figure + used by the rate simulation), with recomputed stability + fold markers,
  * the fold loci in the (I, h_Delta) plane (the cusps that bound the bi-/multi-stable regions),
  * a stable-equilibrium COUNT grid n(I, h_Delta): #stable equilibria at each grid point.  The
    equilibria form one connected S-curve and (no Hopf in the excitatory case) stability
    alternates at folds, so #stable = (#crossings of the I-continuation branch with I=I_col + 1)//2
    — counted from the branch geometry alone, no eigenvalue recompute needed.

ONE LAYER PER PROCESS (two ODESystems collide on shared continuation names):
    python pyramidal_fig_bifurcation.py "Pyramidal" "L2/3"
    python pyramidal_fig_bifurcation.py "Pyramidal" "L5/6"
Run in the ``pycobi`` conda env.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import allen_qif_bifurcation as B
from pycobi import ODESystem

CONFIG = dict(v_r=-70.0, J=100.0, tau_s=0.5, I_settle=0.0, r0=0.05, T_settle=2000.0,
              I_lo=0.0, I_hi=400.0, HD_CUT=0.1, H_MAX=1.5,
              # count grid
              H_lo=0.05, H_hi=1.05, n_h=60, n_I=240)


def _branch_I(df):
    """Iext along a continuation in continuation order (drop NaNs)."""
    I = df[B._pcol(df, "Iext")].to_numpy(float)
    return I[np.isfinite(I)]


def _crossings(I_branch, I_col):
    """# times the (continuation-ordered) branch crosses the vertical line I=I_col."""
    d = I_branch - I_col
    s = np.sign(d)
    s[s == 0] = 1.0                       # treat exact hits as + (avoids double count)
    return int(np.count_nonzero(np.diff(s) != 0))


def run_layer(cfg, cell_class, layer):
    tag = B._tag(cell_class, layer)
    w, Om, De, M = B.load_fit(os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{tag}.npz"))
    v_r = cfg["v_r"]
    print(f"== {cell_class} {layer}  M={M}  J={cfg['J']}  tau_s={cfg['tau_s']}  (vary h_Delta) ==")
    r0 = np.full(M, cfg["r0"]); v0 = np.full(M, v_r)
    circuit = B.build_circuit(M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], cfg["I_settle"], r0, v0,
                              combined=False, hD0=1.0, hC0=1.0)
    ode = ODESystem.from_template(circuit, auto_dir=B.AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))

    I_eq = dict(IPS=1, ILP=1, ISP=2, ISW=1, NMX=50000, NPR=2000,
                DS=1e-2, DSMIN=1e-8, DSMAX=0.1, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5)

    # (1) settle at the data fit (hD=hC=1), I=I_settle
    print(f"[1] IVP settle (T={cfg['T_settle']}) at I={cfg['I_settle']}, hD=hC=1")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: cfg["T_settle"]}, STOP={"UZ1"})

    def _to_hd(hd, nm):
        """Equilibrium continuation in hD from the data-fit settle down to hd (stop at UZ)."""
        ode.run(origin="time", starting_point="UZ1", name=nm, c="eq", ICP="hD",
                RL0=hd - 1e-3, RL1=1.0, IPS=1, ILP=0, ISP=0, ISW=1, NMX=8000, NPR=2000,
                DS=-1e-3, DSMIN=1e-9, DSMAX=2e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                UZR={"hD": hd}, STOP={"UZ1"})

    def _in_I(origin, nm, get_stab=False, npr=None):
        kw = dict(I_eq)
        if npr is not None:
            kw["NPR"] = npr
        return ode.run(origin=origin, starting_point="UZ1", name=nm, c="eq", ICP="Iext",
                       bidirectional=True, RL0=cfg["I_lo"], RL1=cfg["I_hi"],
                       get_stability=get_stab, **kw)

    # (2) 1-D branch at HD_CUT (with stability + fold markers) for the figure's 1-D panel
    hc = cfg["HD_CUT"]
    print(f"[2] 1-D branch at hD={hc}")
    _to_hd(hc, "eq_hD_cut")
    eqc, _ = _in_I("eq_hD_cut", "eq_I_cut", get_stab=True, npr=10)   # dense for the 1-D panel
    B.recompute_stability(eqc, M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], hD=hc, hC=1.0)
    folds = B._bif_vals(eqc, "LP", "Iext")
    print(f"   hD={hc} folds LP at I = {[round(x, 1) for x in folds]}")
    branch_I = eqc[B._pcol(eqc, "Iext")].to_numpy(float)
    branch_s = eqc[B._pcol(eqc, "s")].to_numpy(float)
    branch_stab = eqc[("stability", "")].to_numpy(bool)
    lp_mask = eqc[("bifurcation", "")].to_numpy() == "LP"     # (I,s) of each fold, SAME order
    lp_I = eqc[B._pcol(eqc, "Iext")].to_numpy(float)[lp_mask]
    lp_s = eqc[B._pcol(eqc, "s")].to_numpy(float)[lp_mask]

    # (3) fold loci in (I, hD), seeded from every fold of the HD_CUT slice
    n_lp = int(lp_mask.sum())
    loci = []
    for k in range(1, min(n_lp, 8) + 1):
        nm = f"fold_{k}_IhD"
        try:
            df, _ = ode.run(origin="eq_I_cut", starting_point=f"LP{k}", name=nm, c="eq",
                            ICP=["Iext", "hD"], bidirectional=True, IPS=1, ISW=2, ISP=2, ILP=0,
                            RL0=cfg["I_lo"], RL1=cfg["I_hi"], NMX=20000, NPR=10,
                            DS=1e-2, DSMIN=1e-8, DSMAX=0.05, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                            UZSTOP={"hD": [2e-3, cfg["H_MAX"]]})
            xa = df[B._pcol(df, "Iext")].to_numpy(float); ya = df[B._pcol(df, "hD")].to_numpy(float)
            m = np.isfinite(xa) & np.isfinite(ya)
            loci.append((xa[m], ya[m]))
        except Exception as e:
            print(f"   {nm}: FAILED ({type(e).__name__}: {e})")
    print(f"   traced {len(loci)} fold loci")

    # (4) stable-equilibrium count grid n(I, hD) via branch-crossing parity
    hgrid = np.linspace(cfg["H_lo"], cfg["H_hi"], cfg["n_h"])
    Igrid = np.linspace(cfg["I_lo"], cfg["I_hi"], cfg["n_I"])
    ncount = np.zeros((cfg["n_h"], cfg["n_I"]), int)
    print(f"[4] count grid {cfg['n_h']}x{cfg['n_I']} ...")
    for j, hv in enumerate(hgrid):
        try:
            if hv >= 0.99:                              # data-fit row: continue in I from the settle
                df, _ = _in_I("time", f"I_{j}")
            else:
                _to_hd(float(hv), f"hd_{j}")
                df, _ = _in_I(f"hd_{j}", f"I_{j}")
            Ib = _branch_I(df)
            ncount[j] = [(_crossings(Ib, Ic) + 1) // 2 for Ic in Igrid]
        except Exception as e:
            print(f"   row hD={hv:.3f}: FAILED ({type(e).__name__}: {e}); reuse prev")
            ncount[j] = ncount[j - 1] if j else 1
        if j % 10 == 0:
            print(f"   hD={hv:.3f}: max #stable = {ncount[j].max()}")

    out = os.path.join(_HERE, f"pyramidal_fig_bif_{tag}.npz")
    data = dict(cell_class=cell_class, layer=layer, tag=tag, M=np.int64(M),
                J=float(cfg["J"]), tau_s=float(cfg["tau_s"]), hd_cut=float(hc),
                branch_I=branch_I, branch_s=branch_s, branch_stab=branch_stab,
                lp_I=lp_I, lp_s=lp_s, n_loci=np.int64(len(loci)),
                count_I=Igrid, count_h=hgrid, count_n=ncount)
    for k, (xa, ya) in enumerate(loci):
        data[f"loci_{k}_I"] = xa; data[f"loci_{k}_h"] = ya
    np.savez(out, **data)
    print(f"   [saved] {os.path.basename(out)}")
    ode.close_session(clear_files=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        raise SystemExit('usage: pyramidal_fig_bifurcation.py "<cell_class>" "<layer>"')
    run_layer(CONFIG, args[0], args[1])


if __name__ == "__main__":
    main()
