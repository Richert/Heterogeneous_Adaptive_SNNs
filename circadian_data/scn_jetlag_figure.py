r"""
Write-up figure: jet lag (re-entrainment after an instantaneous shift of the 24-h light cycle), SCN1 vs SCN2
==========================================================================================================

Single-column PRL figure, 2 × 2 (protocol and readouts: scn_jetlag.py; F = 0.06 rad/h, 12:12, β = 4.2; pooled-lobe
mixtures "pfloor" (noise floor) and "psingle" (single Lorentzian); for SCN2 both are the same single Lorentzian):
  (a) SCN1 at K_mid, 8-h eastward shift: phase error ψ(t) of the population rhythm for both models and of the two
      clusters (mixture components) of the noise-floor model — the clusters re-entrain at different speeds
      (internal desynchronisation); inset: coherence R(t)
  (b) coherence dip ΔR vs K / K_c (K_c of the noise-floor mixture) after 8-h shifts east (solid) and west (dashed)
  (c) re-entrainment time (|ψ| < 1 h for good) vs K / K_c, same shifts; × = re-entrains the long way round
  (d) re-entrainment time vs shift size X at K_post (2-h grid; X > 0 east = advance, X < 0 west = delay; ±12 h are the
      same shift)
K values: scn_jetlag.K_grid (30 values in [1.05 K_c, K_pre]); K_mid, K_post from scn_entrainment (K_protocol.json).
Panels (a) and (d) are simulated here (seconds; cached), (b) and (c) read the scn_jetlag outputs.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_jetlag_figure.py [--resim] [--show]
Output: data_paths.MPMF/circadian_data/writeup/scn_jetlag_scn12.{pdf,png}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import data_paths as dp
import scn_jetlag as J
import scn_entrainment as E

WRITEUP = dp.ensure(dp.mpmf("circadian_data", "writeup"))
F = J.F_DEF
X_EX = 8.0                                     # example / panels (b), (c): ±8 h
SHIFTS_D = tuple(range(-10, 13, 2))            # panel (d)
CURVES = ((1, "pfloor", "#c0392b", "SCN1, noise floor"), (1, "psingle", "0.35", "SCN1, single Lorentzian"),
          (2, "pfloor", "#2166ac", "SCN2 (both models)"))


def cached(name, fun):
    f = os.path.join(J.OUT, name)
    if "--resim" in sys.argv or not os.path.exists(f):
        np.savez(f, **fun())
    return dict(np.load(f, allow_pickle=True))


def main():
    show = "--show" in sys.argv
    Kmeta = E.load_K()
    grid = {(i, k): cached(f"jetlag_scn{i}_{k}_F{F:g}.npz", lambda i=i, k=k: J.run_model(i, k, F))
            for i in (1, 2) for k in ("pfloor", "psingle")}
    ex = {k: cached(f"jetlag_example_scn1_{k}_Kmid_X{X_EX:g}.npz",
                    lambda k=k: J.run_model(1, k, F, Ks=[Kmeta["1"]["K"][1]], shifts=[X_EX]))
          for k in ("pfloor", "psingle")}
    scan = {(i, k): cached(f"jetlag_shiftscan_scn{i}_{k}_Kpost.npz",
                           lambda i=i, k=k: J.run_model(i, k, F, Ks=[Kmeta[str(i)]["K"][2]], shifts=SHIFTS_D))
            for i, k, *_ in CURVES}

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from prl_style import set_prl_style, panel_label, COL_SINGLE
    set_prl_style("prl")
    fig, axs = plt.subplots(2, 2, figsize=(COL_SINGLE, 3.35),
                            gridspec_kw=dict(hspace=0.55, wspace=0.42, left=0.13, right=0.98, top=0.95, bottom=0.21))
    # (a) example: SCN1, K_mid, 8 h east
    a = axs[0, 0]
    for k, col, ls in (("psingle", "0.35", "-"), ("pfloor", "#c0392b", "-")):
        d = ex[k]
        a.plot(d["t_rec"], d["psi_rec"][:, 0, 0], color=col, ls=ls, lw=1.1, zorder=3)
    d = ex["pfloor"]; fr = d["Om"] * 24 / (2 * np.pi)
    for m, cc in zip(np.argsort(fr), ("#f4a582", "#67001f")):
        a.plot(d["t_rec"], d["psi_comp_rec"][:, 0, 0, m], color=cc, lw=0.7, zorder=2)
        a.text(0.97, 0.06 + 0.13 * (cc == "#67001f"), f"cluster {fr[m]:.2f} c/d", color=cc, fontsize=5.2,
               transform=a.transAxes, ha="right")
    lo = min(-1.5, np.nanmin(ex["pfloor"]["psi_comp_rec"][:, 0, 0]) - 1)
    a.axhspan(-J.PHASE_TOL, J.PHASE_TOL, color="0.9", zorder=0)
    for off in (-24, 24):
        a.axhspan(off - J.PHASE_TOL, off + J.PHASE_TOL, color="0.9", zorder=0)
    a.set_xlim(0, 20); a.set_ylim(max(lo, -26), 3)
    a.set_xlabel("days after the shift"); a.set_ylabel(r"phase error $\psi$ (h)")
    a.set_title(rf"SCN1, $K_{{mid}}$, 8 h east", fontsize=6.5, pad=3)
    ins = a.inset_axes([0.5, 0.42, 0.47, 0.3])
    for k, col, ls in (("psingle", "0.35", "-"), ("pfloor", "#c0392b", "-")):
        ins.plot(ex[k]["t_rec"], ex[k]["R_rec"][:, 0, 0], color=col, ls=ls, lw=0.7)
    ins.set_xlim(0, 20); ins.set_ylim(0, 1); ins.set_yticks([0, 1]); ins.set_xticks([])
    ins.tick_params(labelsize=5, length=1.5, pad=1); ins.set_ylabel("$R$", fontsize=5.5, labelpad=0)
    # (b), (c) vs K / K_c
    for c, (key, ylab) in enumerate((("dR", r"coherence dip $\Delta R$"), ("T1h", "re-entrainment (days)"))):
        a = axs[0, 1] if c == 0 else axs[1, 0]
        for i, k, col, lab in CURVES:
            d = grid[(i, k)]; x = d["K"] / d["K_c"]
            for X, ls, lw in ((X_EX, "-", 1.1), (-X_EX, "--", 0.9)):
                s = list(d["shifts"]).index(X); y = d[key][s]
                a.plot(x, y, color=col, ls=ls, lw=lw)
                if key == "T1h":
                    ar = (d["route"][s] == "around") & np.isfinite(y)
                    a.plot(x[ar], y[ar], "x", color=col, ms=3, mew=0.8)
        for i, col, ha in ((1, "#c0392b", "right"), (2, "#2166ac", "left")):
            Kc_ = grid[(i, "pfloor")]["K_c"]
            for j_, nm in ((1, "mid"), (2, "post")):
                xv = Kmeta[str(i)]["K"][j_] / Kc_
                a.axvline(xv, color=col, lw=0.5, ls=":", alpha=0.8)
                if c == 0 and (nm == "post" or i == 1):                 # K_mid / K_c ≈ 1.7 in both slices
                    a.text(xv, 1.01, rf"$K_{{{nm}}}$", color=col if nm == "post" else "0.2", fontsize=5,
                           ha=ha if nm == "post" else "center", va="bottom", transform=a.get_xaxis_transform())
        a.set_xscale("log"); a.set_xlabel(r"$K / K_c$"); a.set_ylabel(ylab); a.set_ylim(bottom=0)
    # (d) re-entrainment time vs shift size at K_post
    a = axs[1, 1]
    for i, k, col, lab in CURVES:
        d = scan[(i, k)]; X = d["shifts"]; y = d["T1h"][:, 0]
        a.plot(X, y, color=col, lw=1.1, marker="o", ms=2.2)
        ar = d["route"][:, 0] == "around"
        a.plot(X[ar], y[ar], "x", color=col, ms=4, mew=0.9)
    a.axvline(0, color="0.7", lw=0.5)
    a.set_xticks([-8, -4, 0, 4, 8, 12]); a.set_xlabel("shift $X$ (h): west  |  east"); a.set_ylabel(
        r"re-entrainment at $K_{post}$ (d)"); a.set_ylim(bottom=0)
    for r, c, l in ((0, 0, "a"), (0, 1, "b"), (1, 0, "c"), (1, 1, "d")):
        panel_label(axs[r, c], l, dx=-22, dy=3)
    hd = ([Line2D([], [], color=col, lw=1.1, label=lab)
           for i, k, col, lab in CURVES] +
          [Line2D([], [], color="k", lw=1.1, label="8 h east"), Line2D([], [], color="k", lw=0.9, ls="--",
                                                                           label="8 h west"),
           Line2D([], [], ls="none", marker="x", color="k", ms=4, mew=0.9, label="long way round")])
    fig.legend(handles=hd, loc="lower center", ncol=3, frameon=False, fontsize=5.5, bbox_to_anchor=(0.53, 0.0),
               handlelength=1.8, columnspacing=1.0)
    out = os.path.join(WRITEUP, "scn_jetlag_scn12")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
