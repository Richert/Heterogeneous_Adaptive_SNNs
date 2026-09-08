r"""
PV+ interneuron heterogeneity bifurcation — PURE PyCoBi version  [WORK IN PROGRESS]
===================================================================================

Goal: reproduce the 1-D bifurcation diagram of allen_qif_heterogeneity.py panel (c) — equilibrium
branch + Hopf + emergent limit cycle as a function of the width-heterogeneity knob h_Delta — but
entirely with PyCoBi/Auto-07p (equilibrium continuation + Hopf detection + periodic-orbit
continuation), for both PV L2/3 and L5/6. No numpy eigenvalue / solve_ivp crutches.

Fixed regime (matches the heterogeneity study): centres collapsed (h_eta=0), J=-200, tau_s=2,
I = I_fix (per layer); bifurcation parameter = h_Delta (Auto param `hD`), swept 1 -> 0.

Run in the ``pycobi`` conda env (Auto-07p):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python allen_qif_heterogeneity_pycobi.py [cell_class] [layer]
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import allen_qif_bifurcation as B          # reuse build_circuit (hD/hC params), load_fit, helpers
from pycobi import ODESystem

# ── per-layer regime (from allen_qif_heterogeneity.py) ───────────────────────
REGIME = {
    ("PV+ Interneuron", "L2/3"): dict(I_fix=490.6),
    ("PV+ Interneuron", "L5/6"): dict(I_fix=387.8),
}
CONFIG = dict(cell_class="PV+ Interneuron", layer="L2/3",
              v_r=-70.0, J=-200.0, tau_s=2.0, hC=0.0,    # centres collapsed
              r0=0.05, T_settle=1000.0,
              hD_min=0.0, hD_max=1.0)
if len(sys.argv) > 2:
    CONFIG["cell_class"], CONFIG["layer"] = sys.argv[1], sys.argv[2]

C_EQ = "#1f4e79"; C_LC = "#c1121f"; C_PD2 = "#7b2cbf"           # equilibrium / cycle / period-2
C_HOPF = "#1f4e79"; C_PD = "#c1121f"; C_LPC = "#2e7d32"          # 2-D loci
C_HB = "#e07b00"


def _branch_arrays(ode, cont, yname, with_stab=True):
    """Extract (h_Delta, y_min, y_max, stability, bifurcation) arrays straight from the continuation
    summary, so the figure is drawn from plain numpy (no live ODESystem -> avoids the cross-layer
    name clash, and lets the figure be re-rendered from the saved .npy). For limit-cycle branches
    reduce_limit_cycle stores TWO columns per variable — (yname,0)=min and (yname,1)=max — which we
    keep as the cycle envelope; equilibria have a single column (min==max)."""
    summ = ode.get_summary(cont)
    head = lambda name: [c for c in summ.columns if (c[0] if isinstance(c, tuple) else c) == name]
    hD = np.asarray(summ[head("hD")[0]], float)
    Y = np.column_stack([np.asarray(summ[c], float) for c in head(yname)])     # (N, 1) or (N, 2)
    bif = np.asarray(summ[head("bifurcation")[0]]).astype(str)
    sc = head("stability")
    stab = np.asarray(summ[sc[0]], bool) if (with_stab and sc) else np.ones(hD.size, bool)
    return dict(hD=hD, ymin=Y.min(axis=1), ymax=Y.max(axis=1), stab=stab, bif=bif)


def run_layer(cfg, cell_class, layer):
    """Run the full PyCoBi pipeline for one layer; return the live ODESystem + continuation names."""
    tag = B._tag(cell_class, layer)
    I_fix = REGIME[(cell_class, layer)]["I_fix"]
    w, Om, De, M = B.load_fit(os.path.join(_HERE, "..", "data_fitting", f"allen_lorentzian_{tag}.npz"))
    v_r = cfg["v_r"]
    print(f"== {cell_class} {layer}  M={M}  J={cfg['J']}  tau_s={cfg['tau_s']}  "
          f"h_eta={cfg['hC']}  I_fix={I_fix} ==")

    r0 = np.full(M, cfg["r0"]); v0 = np.full(M, v_r)
    circuit = B.build_circuit(M, Om, De, w, v_r, cfg["tau_s"], cfg["J"], I_fix, r0, v0,
                              hD0=1.0, hC0=cfg["hC"])
    ode = ODESystem.from_template(circuit, auto_dir=B.AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))

    # ── Step 1: settle to the (stable) equilibrium at h_Delta = 1 ───────────
    print(f"[1] IVP settle (T={cfg['T_settle']}) at h_Delta=1")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: cfg["T_settle"]}, STOP={"UZ1"})

    # ── Step 2: equilibrium continuation in h_Delta (1 -> 0): expect a Hopf ──
    print("[2] equilibrium continuation in h_Delta")
    eq, _ = ode.run(origin="time", starting_point="UZ1", name="eq_hD", c="eq",
                    ICP="hD", bidirectional=True, RL0=cfg["hD_min"], RL1=cfg["hD_max"],
                    IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=10,
                    DSMIN=1e-9, DSMAX=5e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                    get_stability=True)
    print("   bifurcations:", dict(eq["bifurcation"].value_counts()))
    hbs = B._bif_vals(eq, "HB", "hD"); lps = B._bif_vals(eq, "LP", "hD")
    print(f"   HB at h_Delta = {[round(x, 4) for x in hbs]}")
    print(f"   LP at h_Delta = {[round(x, 4) for x in lps]}")

    # ── Step 3: limit-cycle continuation from each Hopf point ───────────────
    #   ISP=2 -> detect PD (period doubling) & TR (torus);  ILP=1 -> detect LPC (fold of cycles)
    n_hb = len(eq[eq[("bifurcation", "")] == "HB"])
    lc_conts = []; pd_hd = None; lc1_has_lp = False
    for k in range(1, n_hb + 1):
        nm = f"lc_{k}"
        try:
            lc, _ = ode.run(origin="eq_hD", starting_point=f"HB{k}", name=nm,
                            IPS=2, ISW=-1, ICP=["hD", 11], NTST=400, NCOL=4,
                            ILP=1, ISP=2, NMX=8000, NPR=10, DS=1e-3, DSMIN=1e-12, DSMAX=2e-3,
                            EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, RL0=cfg["hD_min"], RL1=cfg["hD_max"],
                            get_stability=True)
            lc_conts.append(nm)
            print(f"   {nm}: limit cycle from HB{k}, {len(lc)} pts;  "
                  f"bifs={dict(lc['bifurcation'].value_counts())}")
            for bif in ("PD", "LP", "TR", "BP"):
                vals = B._bif_vals(lc, bif, "hD")
                if vals:
                    print(f"       {bif} at h_Delta = {[round(x, 4) for x in vals]}")
            if nm == "lc_1":
                _pd = B._bif_vals(lc, "PD", "hD")
                pd_hd = _pd[0] if _pd else None
                lc1_has_lp = bool(B._bif_vals(lc, "LP", "hD"))
        except Exception as e:
            print(f"   {nm}: FAILED ({type(e).__name__}: {e})")

    # ── Step 4: period-doubled branch from the first PD -> its fold-of-cycle ─
    #   only bother if the MAIN cycle continuation actually hit a fold-of-cycle (LP); otherwise
    #   we keep just the Hopf + PD loci (per request).
    pd2_conts = []; pd2_df = None
    if "lc_1" in lc_conts and lc1_has_lp:
        try:
            pd2_df, _ = ode.run(origin="lc_1", starting_point="PD1", name="lc_pd2",
                                IPS=2, ISW=-1, ICP=["hD", 11], NTST=400, NCOL=4,
                                ILP=1, ISP=2, NMX=8000, NPR=20, DS=1e-3, DSMIN=1e-12, DSMAX=5e-3,
                                EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, RL0=cfg["hD_min"], RL1=cfg["hD_max"],
                                get_stability=True)
            pd2 = pd2_df
            pd2_conts.append("lc_pd2")
            print(f"   lc_pd2: period-2 branch from PD1, {len(pd2)} pts;  "
                  f"bifs={dict(pd2['bifurcation'].value_counts())}")
            for bif in ("LP", "PD", "TR"):
                vals = B._bif_vals(pd2, bif, "hD")
                if vals:
                    print(f"       {bif} at h_Delta = {[round(x, 4) for x in vals]}")
        except Exception as e:
            print(f"   lc_pd2: FAILED ({type(e).__name__}: {e})")

    # ── Step 5: 2-parameter loci in (h_Delta, I) of HB, PD and the first LPC ─
    two_d = {}
    #   RL0=3e-3 (not 0): the homogeneous limit h_Delta->0 is singular; continuing into it makes the
    #   curve mis-track / wind. Stop just short of it for a clean locus.
    common = dict(ICP=["hD", "Iext"], bidirectional=True, RL0=3e-3, RL1=cfg["hD_max"],
                  NPR=10, DS=1e-3, DSMIN=1e-12, DSMAX=5e-2,
                  EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, UZSTOP={"Iext": [0.0, 2500.0]})
    periodic = dict(IPS=2, ISW=2, ISP=0, ILP=0, NTST=200, NCOL=4, NMX=4000, get_stability=False)
    try:
        hb2, _ = ode.run(origin="eq_hD", starting_point="HB1", name="hopf_2d", c="eq",
                         IPS=1, ISW=2, ISP=0, ILP=0, NMX=8000, get_stability=False, **common)
        two_d["hopf"] = "hopf_2d"
        print(f"   hopf_2d: {len(hb2)} pts, hD∈[{hb2[B._pcol(hb2,'hD')].min():.3f},{hb2[B._pcol(hb2,'hD')].max():.3f}]"
              f" I∈[{hb2[B._pcol(hb2,'Iext')].min():.0f},{hb2[B._pcol(hb2,'Iext')].max():.0f}]")
    except Exception as e:
        print(f"   hopf_2d: FAILED ({type(e).__name__}: {e})")
    try:
        pd2d, _ = ode.run(origin="lc_1", starting_point="PD1", name="pd_2d", **periodic, **common)
        two_d["pd"] = "pd_2d"
        print(f"   pd_2d: {len(pd2d)} pts, hD∈[{pd2d[B._pcol(pd2d,'hD')].min():.3f},{pd2d[B._pcol(pd2d,'hD')].max():.3f}]"
              f" I∈[{pd2d[B._pcol(pd2d,'Iext')].min():.0f},{pd2d[B._pcol(pd2d,'Iext')].max():.0f}]")
    except Exception as e:
        print(f"   pd_2d: FAILED ({type(e).__name__}: {e})")
    # pick the first fold-of-cycle AFTER the PD (largest hD among the period-2 LPs below the PD)
    if pd2_df is not None:
        pd2 = pd2_df
        lp_rows = pd2[pd2[("bifurcation", "")] == "LP"]
        lp_hd = lp_rows[B._pcol(pd2, "hD")].to_numpy(float)
        print(f"   period-2 LP labels (branch order) hD = {[round(x,4) for x in lp_hd]}  (PD at {pd_hd})")
        thr = (pd_hd if pd_hd is not None else lp_hd.max()) - 1e-4
        cand = [(i + 1, h) for i, h in enumerate(lp_hd) if h < thr]
        if cand:
            lab = max(cand, key=lambda t: t[1])[0]                 # first fold below the PD
            try:
                lpc2d, _ = ode.run(origin="lc_pd2", starting_point=f"LP{lab}", name="lpc_2d",
                                   **periodic, **common)
                two_d["lpc"] = "lpc_2d"
                print(f"   lpc_2d (from LP{lab}): {len(lpc2d)} pts, "
                      f"hD∈[{lpc2d[B._pcol(lpc2d,'hD')].min():.3f},{lpc2d[B._pcol(lpc2d,'hD')].max():.3f}]"
                      f" I∈[{lpc2d[B._pcol(lpc2d,'Iext')].min():.0f},{lpc2d[B._pcol(lpc2d,'Iext')].max():.0f}]")
            except Exception as e:
                print(f"   lpc_2d: FAILED ({type(e).__name__}: {e})")

    # extract everything to arrays NOW (before the next layer's ODESystem is built)
    data = dict(cell_class=cell_class, layer=layer, tag=tag, M=int(M), I_fix=float(I_fix),
                eq=_branch_arrays(ode, "eq_hD", "s"),
                lc=[_branch_arrays(ode, nm, "s") for nm in lc_conts],
                pd2=(_branch_arrays(ode, "lc_pd2", "s") if pd2_conts else None))
    for key in ("hopf", "pd", "lpc"):
        data[key] = _branch_arrays(ode, two_d[key], "Iext", with_stab=False) if key in two_d else None
    # seed (h_Delta, I) each 2-D locus started from — used to bridge the bidirectional join cleanly
    data["seeds"] = {}
    if "hopf" in two_d and hbs:
        data["seeds"]["hopf"] = (float(hbs[0]), float(I_fix))
    if "pd" in two_d and pd_hd is not None:
        data["seeds"]["pd"] = (float(pd_hd), float(I_fix))
    np.save(os.path.join(_HERE, f"allen_qif_heterogeneity_pycobi_{tag}.npy"), data, allow_pickle=True)
    print(f"   [saved] allen_qif_heterogeneity_pycobi_{tag}.npy")
    return data


def _one_seg(ax, x, y, stab, color, lw):
    """One curve y(h_Delta): solid (stable) / dashed (unstable) contiguous-stability runs. The
    1-D branches are single continuations (no bidirectional join), so plot them whole — NO jump
    guard (a median-based guard shreds the cycle near the Hopf, where amplitude grows from ~0)."""
    flips = list(np.where(np.diff(stab.astype(int)) != 0)[0] + 1)
    for a, b in zip([0] + flips, flips + [stab.size]):
        e = min(b + 1, stab.size)
        ax.plot(x[a:e], y[a:e], color=color, lw=lw, ls="-" if stab[a] else "--", zorder=3)


def _seg(ax, d, color, lw):
    """Plot a branch as solid/dashed; for limit cycles draw BOTH envelope rails (y_min, y_max)."""
    x, stab = d["hD"], d["stab"].astype(bool)
    _one_seg(ax, x, d["ymax"], stab, color, lw)
    if np.any(np.abs(d["ymax"] - d["ymin"]) > 1e-9):
        _one_seg(ax, x, d["ymin"], stab, color, lw)


def _markers(ax, d, kinds):
    for t, marker, color in kinds:
        idx = np.where(d["bif"] == t)[0]
        if idx.size:
            ax.plot(d["hD"][idx], d["ymax"][idx], marker, color=color, ms=5, ls="none",
                    mec="k", mew=0.4, zorder=6)


def _locus(ax, d, color, ls, lab, seed=None):
    """2-parameter locus in (h_Delta, I). The bidirectional halves meet at the seed (the
    bifurcation point on the 1-D branch); coarse steps leave a gap there, so we INSERT the exact
    seed point at the join (largest step) rather than break the line — closing the gap cleanly."""
    x = np.asarray(d["hD"], float).copy(); y = np.asarray(d["ymax"], float).copy()
    if seed is not None and x.size > 2:
        rx = np.ptp(x) or 1.0; ry = np.ptp(y) or 1.0
        j = int(np.argmin(((x - seed[0]) / rx) ** 2 + ((y - seed[1]) / ry) ** 2))  # junction = closest to seed
        x = np.insert(x, j, seed[0]); y = np.insert(y, j, seed[1])
    ax.plot(x, y, color=color, ls=ls, lw=1.6, label=lab)


def make_figure(results, cfg, out_stem):
    """2×2 figure: columns = layers, row 0 = 1-D bifurcation in h_Delta, row 1 = 2-D loci (h_Delta, I)."""
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), squeeze=False)
    panel = "abcd"
    for j, r in enumerate(results):
        name = f"{r['cell_class'].split('+')[0]} {r['layer']}"
        # ── row 0: 1-D bifurcation diagram s(h_Delta) ──
        ax = axs[0][j]
        _seg(ax, r["eq"], C_EQ, 1.7)
        for lc in r["lc"]:
            _seg(ax, lc, C_LC, 1.4)
        if r["pd2"] is not None:
            _seg(ax, r["pd2"], C_PD2, 1.2)
        _markers(ax, r["eq"], [("HB", "o", C_HB)])
        for lc in r["lc"]:
            _markers(ax, lc, [("PD", "s", C_PD2)])
        if r["pd2"] is not None:
            _markers(ax, r["pd2"], [("LP", "D", C_LPC)])
        ax.set_xlim(-0.01, 1.0)
        ax.set_xlabel(r"width heterogeneity $h_\Delta$"); ax.set_ylabel(r"synaptic activation $s$")
        ax.set_title(rf"({panel[j]}) {name}:  1-D bifurcation in $h_\Delta$")
        if j == 0:
            has_pd2 = any(rr["pd2"] is not None for rr in results)
            handles = [Line2D([], [], color=C_EQ, lw=1.7, label="equilibrium"),
                       Line2D([], [], color=C_LC, lw=1.4, label="limit cycle")]
            if has_pd2:
                handles.append(Line2D([], [], color=C_PD2, lw=1.2, label="period-2 cycle"))
            handles += [Line2D([], [], color=C_HB, marker="o", ls="none", mec="k", mew=0.4, label="Hopf"),
                        Line2D([], [], color=C_PD2, marker="s", ls="none", mec="k", mew=0.4, label="period-doubling")]
            if has_pd2:
                handles.append(Line2D([], [], color=C_LPC, marker="D", ls="none", mec="k", mew=0.4, label="fold of cycles"))
            ax.legend(handles=handles, fontsize=5.5, loc="upper left", ncol=2, handlelength=1.4)

        # ── row 1: 2-D loci in the (h_Delta, I) plane ──
        ax = axs[1][j]
        seeds = r.get("seeds", {})
        for key, color, ls, lab in [("hopf", C_HOPF, "-", "Hopf"),
                                    ("pd", C_PD, "--", "period-doubling"),
                                    ("lpc", C_LPC, "-.", "fold of cycles")]:
            if r[key] is not None:
                _locus(ax, r[key], color, ls, lab, seed=seeds.get(key))
        ax.axhline(r["I_fix"], color="0.5", ls=":", lw=1)
        ax.text(0.99, r["I_fix"], " 1-D slice", color="0.4", ha="right", va="bottom",
                fontsize=6, transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.005, 0.25)
        ax.set_xlabel(r"width heterogeneity $h_\Delta$"); ax.set_ylabel(r"external drive $I$")
        ax.set_title(rf"({panel[j + 2]}) {name}:  $(h_\Delta, I)$ loci")
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(r"PV$^+$ interneuron: width-heterogeneity control of inhibition-based oscillations "
                 rf"($h_\eta=0$, $J={cfg['J']:.0f}$, $\tau_s={cfg['tau_s']:.0f}$ ms)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_stem + ".png", dpi=200); fig.savefig(out_stem + ".pdf")
    print(f"[saved] {os.path.basename(out_stem)}.png / .pdf")


def main():
    cfg = CONFIG
    layers = [("PV+ Interneuron", "L2/3"), ("PV+ Interneuron", "L5/6")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_stem = os.path.join(_HERE, "allen_qif_heterogeneity_pycobi_PV")

    if "--fig-only" in sys.argv:
        # render from the per-layer .npy files (fast; no Auto)
        results = [np.load(os.path.join(_HERE, f"allen_qif_heterogeneity_pycobi_{B._tag(cc, ly)}.npy"),
                           allow_pickle=True).item() for cc, ly in layers]
        make_figure(results, cfg, out_stem)
    elif len(args) >= 2:
        # run ONE layer (one ODESystem per process -> no cross-layer name clash) and save its .npy
        run_layer(cfg, args[0], args[1])
    else:
        raise SystemExit("usage: <cell_class> <layer>   (run one layer)   |   --fig-only   (build figure)")


if __name__ == "__main__":
    main()
