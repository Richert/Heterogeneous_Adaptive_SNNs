r"""
TEMPORARY exploration — Pyramidal QIF, new regime tau_s=0.5, J=50, combined heterogeneity h.
============================================================================================
Mirrors the PV figure-2 single-knob approach but for the EXCITATORY pyramidal case, where the
relevant bifurcation is the FOLD (saddle-node / bistability), not the Hopf.  For one layer:
  1. settle the IVP at the data fit (h=1),
  2. 1-D equilibrium continuation in the external input I  ->  do folds survive at J=50?
  3. MAIN 2-D continuation: the fold locus in the (I, h) plane (h = combined heterogeneity knob).
Saves temp figures tmp_pyramidal_ih_<tag>.png (1-D | 2-D side by side).

    python tmp_pyramidal_ih.py "Pyramidal" "L2/3"
    python tmp_pyramidal_ih.py "Pyramidal" "L5/6"
Run in the ``pycobi`` conda env.
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import allen_qif_bifurcation as B
from pycobi import ODESystem

CONFIG = dict(v_r=-70.0, J=100.0, tau_s=0.5, I_settle=0.0, r0=0.05, T_settle=2000.0,
              I_min=-200.0, I_max=2000.0, H_MAX=1.5, hd_seed=0.3)
# second continuation knob is h_Delta (width scaling); hC (centre spread) stays at the data fit = 1.
# hd_seed = a reduced-width slice shown alongside the data fit in the 1-D panel.


def _arr(ode, cont, xname, yname):
    """(x, ymin, ymax, stab, bif) from a continuation summary (LC carries min/max columns)."""
    summ = ode.get_summary(cont)
    head = lambda n: [c for c in summ.columns if (c[0] if isinstance(c, tuple) else c) == n]
    x = np.asarray(summ[head(xname)[0]], float)
    Y = np.column_stack([np.asarray(summ[c], float) for c in head(yname)])
    bif = np.asarray(summ[head("bifurcation")[0]]).astype(str)
    return dict(x=x, ymin=Y.min(axis=1), ymax=Y.max(axis=1), bif=bif)


def run_layer(cfg, cell_class, layer):
    tag = B._tag(cell_class, layer)
    w, Om, De, M = B.load_fit(os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{tag}.npz"))
    v_r = cfg["v_r"]
    print(f"== {cell_class} {layer}  M={M}  J={cfg['J']}  tau_s={cfg['tau_s']}  (two-knob, vary h_Delta) ==")
    r0 = np.full(M, cfg["r0"]); v0 = np.full(M, v_r)
    circuit = B.build_circuit(M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], cfg["I_settle"], r0, v0,
                              combined=False, hD0=1.0, hC0=1.0)       # two knobs: hD (width), hC (centre)
    ode = ODESystem.from_template(circuit, auto_dir=B.AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))

    # (1) settle at the data fit (hD=hC=1), I=I_settle
    print(f"[1] IVP settle (T={cfg['T_settle']}) at I={cfg['I_settle']}, hD=hC=1")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: cfg["T_settle"]}, STOP={"UZ1"})

    # (2a) equilibrium continuation in I at the DATA FIT (hD=1) -> folds?
    print(f"[2a] equilibrium continuation in I at hD=1  (J={cfg['J']}, tau_s={cfg['tau_s']})")
    eq1, _ = ode.run(origin="time", starting_point="UZ1", name="eq_I1", c="eq",
                     ICP="Iext", bidirectional=True, RL0=cfg["I_min"], RL1=cfg["I_max"],
                     IPS=1, ILP=1, ISP=2, ISW=1, NMX=50000, NPR=1000,
                     DS=1e-2, DSMIN=1e-8, DSMAX=0.1, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                     get_stability=True)
    B.recompute_stability(eq1, M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], hD=1.0, hC=1.0)
    print(f"   hD=1 fold(s) LP at I = {[round(x, 2) for x in B._bif_vals(eq1, 'LP', 'Iext')]}")

    # (2b) reduced-width slice hD=hd_seed (hC stays at 1) for the 1-D panel comparison
    hs = cfg["hd_seed"]
    print(f"[2b] equilibrium continuation in hD: 1 -> {hs}")
    ode.run(origin="time", starting_point="UZ1", name="eq_hD", c="eq", ICP="hD",
            RL0=hs - 1e-3, RL1=1.0, IPS=1, ILP=0, ISP=0, ISW=1, NMX=8000, NPR=1000,
            DS=-1e-3, DSMIN=1e-9, DSMAX=2e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
            UZR={"hD": hs}, STOP={"UZ1"})
    print(f"[2c] equilibrium continuation in I at hD={hs}")
    eq2, _ = ode.run(origin="eq_hD", starting_point="UZ1", name="eq_I2", c="eq",
                     ICP="Iext", bidirectional=True, RL0=cfg["I_min"], RL1=cfg["I_max"],
                     IPS=1, ILP=1, ISP=2, ISW=1, NMX=50000, NPR=1000,
                     DS=1e-2, DSMIN=1e-8, DSMAX=0.1, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                     get_stability=True)
    B.recompute_stability(eq2, M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], hD=hs, hC=1.0)
    print(f"   hD={hs} fold(s) LP at I = {[round(x, 2) for x in B._bif_vals(eq2, 'LP', 'Iext')]}")

    # (3) MAIN 2-D continuation: fold loci (cusps) in the (I, h_Delta) plane.  Seed from EVERY fold
    #     of the small-hD slice (the multi-stable regime has >2 folds: extra cusps appear as hD drops),
    #     so each fold branch — including the inner ones that annihilate at higher-hD cusp tips — is traced.
    n_lp = len(eq2[eq2[("bifurcation", "")] == "LP"])
    loci = []
    for k in range(1, min(n_lp, 8) + 1):
        nm = f"fold_{k}_IhD"
        try:
            df, _ = ode.run(origin="eq_I2", starting_point=f"LP{k}", name=nm, c="eq", ICP=["Iext", "hD"],
                            bidirectional=True, IPS=1, ISW=2, ISP=2, ILP=0,
                            RL0=cfg["I_min"], RL1=cfg["I_max"], NMX=20000, NPR=10,
                            DS=1e-2, DSMIN=1e-8, DSMAX=0.05, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                            UZSTOP={"hD": [2e-3, cfg["H_MAX"]]})
            xa = df[B._pcol(df, "Iext")].to_numpy(float); ya = df[B._pcol(df, "hD")].to_numpy(float)
            loci.append((xa, ya))
            print(f"   {nm}: {xa.size} pts;  I∈[{xa.min():.1f},{xa.max():.1f}]  "
                  f"hD∈[{ya.min():.3f},{ya.max():.3f}]")
        except Exception as e:
            print(f"   {nm}: FAILED ({type(e).__name__}: {e})")

    # ── temp figures: (a) 1-D s(I) at hD=1 vs hd_seed, (b) fold cusp in (I, h_Delta) ─────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for eqd, hlab, col in ((eq1, "1.0", B.C_EQ), (eq2, f"{hs:g}", B.C_HOPF)):
        B._plot_branch(ax1, eqd[B._pcol(eqd, "Iext")].to_numpy(float),
                       eqd[B._pcol(eqd, "s")].to_numpy(float),
                       eqd[("stability", "")].to_numpy(bool), col)
        B.add_markers(ax1, eqd, "LP", B.M_FOLD, "Iext", "s", B.C_FOLD)
    ax1.set_xlabel(r"external input $I$"); ax1.set_ylabel(r"synaptic activation $s$")
    ax1.set_title(f"(a) 1-D in $I$  ({cell_class} {layer}, $J$={cfg['J']:g}, $\\tau_s$={cfg['tau_s']:g})")
    ax1.legend(handles=[Line2D([0], [0], color=B.C_EQ, lw=2, label="$h_\\Delta$=1 (data fit)"),
                        Line2D([0], [0], color=B.C_HOPF, lw=2, label=f"$h_\\Delta$={hs:g}"),
                        Line2D([0], [0], marker=B.M_FOLD, color=B.C_FOLD, lw=0, mfc="none", label="fold")],
               loc="best", fontsize=8)

    for xa, ya in loci:
        ax2.plot(xa, ya, color=B.C_FOLD, lw=1.8)
    ax2.axhline(1.0, color="0.6", ls=":", lw=1.0, label="$h_\\Delta$=1 (data fit)")
    ax2.set_xlabel(r"external input $I$"); ax2.set_ylabel(r"width scaling $h_\Delta$")
    ax2.set_title("(b) MAIN 2-D: fold cusp in the $(I, h_\\Delta)$ plane")
    ax2.set_ylim(0, cfg["H_MAX"]); ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    out = os.path.join(_HERE, f"tmp_pyramidal_ih_{tag}.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"   [saved] {os.path.basename(out)}")
    ode.close_session(clear_files=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        raise SystemExit('usage: tmp_pyramidal_ih.py "<cell_class>" "<layer>"')
    run_layer(CONFIG, args[0], args[1])


if __name__ == "__main__":
    main()
