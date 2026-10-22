r"""
Main-text figure: bifurcation structure and entrainment of a bimodal (SCN1) vs a unimodal (SCN2) slice
=====================================================================================================

Two-column PRL figure, one row per slice (default SCN1, SCN2), four panels per row. Panels (1)–(3) are those of the
bifurcation figure (scn_bif_ramp_figure.py), panel (4) is from the entrainment analysis (scn_entrainment.py). Both
mixtures are fitted to the slice's POOLED intrinsic frequencies (scn_pycobi_bifurcation kinds "pfloor" = noise-floor
Lorentzian mixture, "psingle" = one Lorentzian):
  (1) intrinsic frequencies under TTX and the two mixture densities
  (2) 1-D bifurcation diagram R*(K) (PyCoBi): stable / unstable equilibria, folds, Hopf points, limit-cycle R min/max;
      the intermediate coupling K_mid of panel (4) marked
  (3) K staircase 0.06 → 0 (steps of 0.01, 30-day holds): R(t) of both mixtures, grey = stable R* (noise floor)
  (4) coherence between driver and the noise-floor model at the driving frequency (brute-force map) at K_mid, with
      the tongue boundaries (PyCoBi) of BOTH models: folds of the entrained cycle (solid) and torus bifurcations
      (dashed); bold where the entrained cycle changes stability and the coherence of the respective model changes
      across the curve, thin otherwise (partial entrainment of a cluster). The K_mid curves contain no interior folds.
Requires the outputs of scn_pycobi_bifurcation.py (run --no-loci, pooled kinds), scn_bif_ramp_figure.py (scn_steps.npz)
and scn_entrainment.py (sim, run).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_bif_entrain_figure.py [--slices 1,2] [--show]
Output: data_paths.MPMF/circadian_data/writeup/scn_bif_entrain_scn<slices>.{pdf,png}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import data_paths as dp
import scn_pycobi_bifurcation as B
import scn_bif_ramp_figure as BR
import scn_entrainment as E

J_MID = E.K_LABELS.index("K_mid")
KINDS = (("pfloor", "#c0392b", "-", "noise floor"), ("psingle", "0.3", "--", "single Lorentzian"))
OUTLINE = {"pfloor": ("#39d5ff", "noise floor"), "psingle": ("white", "single Lorentzian")}
C_LC = "#1a9850"
CPD = 24 / (2 * np.pi)


def tongue_curves(i, kind):
    """K_mid tongue boundaries of one model with the bold mask (stability change & coherence contrast of that model's
    own map); duplicates removed."""
    g = np.load(os.path.join(E.OUT, f"tongue3_scn{i}_{kind}_K{J_MID}.npy"), allow_pickle=True).item()
    m = np.load(os.path.join(E.OUT, f"grid_scn{i}_{kind}_K{J_MID}.npz"))
    out, drawn = [], []
    for nm, cur in g["curves"].items():
        fd, Fc = cur[:, 0] * CPD, cur[:, 1]
        uv = np.column_stack([fd / 0.0276, Fc / 0.0052])
        if any(np.mean(np.min(np.linalg.norm(uv[:, None] - d[None], axis=2), axis=1) < 0.5) > 0.9 for d in drawn):
            continue
        drawn.append(uv)
        bold = (E.contrast(m, fd, Fc) > E.CONTRAST) & bool(g["meta"][nm].get("stability_change", True))
        out.append((fd, np.where(Fc >= 0, Fc, np.nan), bold, g["meta"][nm]["type"]))
    return out


def main():
    rows = (tuple(int(s) for s in sys.argv[sys.argv.index("--slices") + 1].split(","))
            if "--slices" in sys.argv else (1, 2))
    show = "--show" in sys.argv
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from prl_style import set_prl_style, panel_label, COL_DOUBLE
    import scn_targets as TG
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center"})
    S = np.load(os.path.join(B.OUT, "scn_steps.npz"))
    Ks, hold = S["K_steps"], float(S["hold_days"])
    Kmeta = E.load_K()
    H = 1.55 * len(rows) + 0.55
    fig, axs = plt.subplots(len(rows), 4, figsize=(COL_DOUBLE, H), squeeze=False,
                            gridspec_kw=dict(width_ratios=[0.85, 1, 1.25, 1.05], hspace=0.62, wspace=0.42,
                                             left=0.055, right=0.985, top=1 - 0.3 / H, bottom=0.62 / H))
    im = None
    for r, i in enumerate(rows):
        f_cells = TG.load(i)["om"] * CPD
        res = {k: np.load(os.path.join(B.OUT, f"scn{i}_{k}.npy"), allow_pickle=True).item() for k, *_ in KINDS}
        K_mid = Kmeta[str(i)]["K"][J_MID]
        # (1) frequency distribution
        a = axs[r, 0]
        g = np.linspace(f_cells.min() - 0.03, f_cells.max() + 0.03, 800)
        a.hist(f_cells, bins=np.linspace(g[0], g[-1], 32), density=True, color="0.85", lw=0)
        for k, col, ls, lab in KINDS:
            mod = B.pooled_mixture(i, k); w = mod["w"] / mod["w"].sum(); gg = g / CPD
            pdf = (w[:, None] * mod["De"][:, None] / np.pi / ((gg[None] - mod["Om"][:, None]) ** 2
                                                              + mod["De"][:, None] ** 2)).sum(0) / CPD
            a.plot(g, pdf, color=col, ls=ls, lw=1.0)
        a.set_yticks([]); a.set_xlabel("intrinsic frequency (c/d)")
        a.set_ylabel(f"SCN{i}\ndensity")
        a.text(0.97, 0.96, "\n".join(f"M = {res[k]['M']}" for k, *_ in KINDS), transform=a.transAxes, ha="right",
               va="top", fontsize=5.5, linespacing=1.3)
        # (2) bifurcation diagram
        a = axs[r, 1]
        for k, col, ls, lab in KINDS:
            gq = res[k]; eq = gq["eq"]
            a.plot(eq["K"], np.where(eq["stable"], eq["R"], np.nan), color=col, ls=ls, lw=1.1)
            a.plot(eq["K"], np.where(~eq["stable"], eq["R"], np.nan), color=col, ls=":", lw=0.9)
            a.plot([0, gq["kc"]["K_c"]], [0, 0], color=col, ls=ls, lw=1.1)
            a.plot([gq["kc"]["K_c"], 0.07], [0, 0], color=col, ls=":", lw=0.9)
            for b_, mk in (("LP", "o"), ("HB", "s")):
                s_ = eq["bif"] == b_
                a.plot(eq["K"][s_], eq["R"][s_], mk, ms=3, mfc="white", mec="k", mew=0.7, zorder=4)
            for lc in filter(None, map(B.trim_lc, gq["lc"])):
                for key in ("Rmax", "Rmin"):
                    a.plot(lc["K"], np.where(lc["stable"], lc[key], np.nan), color=C_LC, lw=0.9)
                    a.plot(lc["K"], np.where(~lc["stable"], lc[key], np.nan), color=C_LC, lw=0.7, ls=":")
        a.axvline(K_mid, color="#39d5ff", lw=0.8, ls="-.", zorder=0)
        a.text(K_mid, 1.03, r"$K_{mid}$", color="#1f8fb3", fontsize=6, ha="center", va="bottom")
        a.set_xlim(-0.001, 0.063); a.set_ylim(-0.02, 1.02)
        a.set_xlabel(r"coupling $K$ (rad/h)"); a.set_ylabel(r"$R^*$")
        # (3) K staircase
        a = axs[r, 2]
        for j, Kj in enumerate(Ks):
            if j % 2 == 0:
                a.axvspan(j * hold, (j + 1) * hold, color="0.95", lw=0, zorder=0)
            a.text((j + 0.5) * hold, 1.03, f"{abs(Kj):.2f}", ha="center", va="bottom", fontsize=5, color="0.3")
            Rs = BR.stable_Rstar(res["pfloor"], Kj)
            if np.isfinite(Rs):
                a.plot([j * hold, (j + 1) * hold], [Rs, Rs], color="0.55", lw=0.6, zorder=1)
        for k, col, ls, lab in KINDS:
            a.plot(S[f"s{i}_{k}_t"] / 24, S[f"s{i}_{k}_R"], color=col, ls=ls, lw=0.9, zorder=2)
        a.set_xlim(0, Ks.size * hold); a.set_ylim(-0.02, 1.02); a.set_xticks(np.arange(0, Ks.size * hold + 1, 60))
        a.set_xlabel("time (days)"); a.set_ylabel(r"$R(t)$")
        # (4) entrainment at K_mid: noise-floor coherence + outlines of both models
        a = axs[r, 3]
        m = np.load(os.path.join(E.OUT, f"grid_scn{i}_pfloor_K{J_MID}.npz"))
        im = a.pcolormesh(m["freqs"], m["F"], m["C"], cmap="magma", vmin=0, vmax=1, shading="nearest",
                          rasterized=True)
        for kind in ("psingle", "pfloor"):
            col = OUTLINE[kind][0]
            pe_ = [pe.withStroke(linewidth=2.2, foreground="0.15")] if kind == "psingle" else None
            for fd, Fc, bold, typ in tongue_curves(i, kind):
                ls = "-" if typ == "LP" else (0, (3, 1.5))
                a.plot(fd, Fc, color=col, lw=0.6, alpha=0.8, ls=ls, zorder=5)
                a.plot(fd, np.where(bold, Fc, np.nan), color=col, lw=1.4, ls=ls, zorder=5, path_effects=pe_)
        if np.isfinite(m["f_c"]):
            a.plot(m["f_c"], 0, marker="v", color="c", mec="k", mew=0.4, ms=4.5, clip_on=False, zorder=6)
        a.set_xlim(*E.FREQ_RANGE); a.set_ylim(0, E.F_MAX)
        a.set_xlabel(r"driving frequency $f_d$ (c/d)"); a.set_ylabel(r"light $F$ (rad/h)")
        a.set_title(rf"$K_{{mid}}$ = {K_mid:.3f}", fontsize=6.5, pad=2)
    titles = ("frequency distribution", "bifurcation in $K$", "coupling staircase",
              "entrainment (noise-floor map)")
    for c, t in enumerate(titles):
        axs[0, c].annotate(t, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 12 if c == 3 else 11),
                           textcoords="offset points", ha="center", va="bottom", fontsize=7, weight="bold")
    for r in range(len(rows)):
        for c in range(4):
            panel_label(axs[r, c], "abcdefghijkl"[4 * r + c], dx=-20 if c else -24, dy=3)
    cax = axs[0, 3].inset_axes([1.03, 0.0, 0.04, 1.0])
    cb = fig.colorbar(im, cax=cax); cb.set_label("coherence", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    hd = ([Patch(fc="0.85", label="cells (TTX)")] +
          [Line2D([], [], color=c, ls=ls, lw=1.0, label=lab) for _, c, ls, lab in KINDS] +
          [Line2D([], [], color="0.3", ls=":", lw=0.9, label="unstable"),
           Line2D([], [], color=C_LC, lw=0.9, label="limit cycle"),
           Line2D([], [], ls="none", marker="o", mfc="white", mec="k", ms=3, label="fold"),
           Line2D([], [], ls="none", marker="s", mfc="white", mec="k", ms=3, label="Hopf"),
           Line2D([], [], color="0.55", lw=0.6, label=r"stable $R^*$"),
           Line2D([], [], color="#39d5ff", lw=1.4, label="tongue edge, noise floor"),
           Line2D([], [], color="white", lw=1.4, label="tongue edge, single",
                  path_effects=[pe.withStroke(linewidth=2.2, foreground="0.15")]),
           Line2D([], [], color="0.4", lw=1.0, label="fold of entrained cycle"),
           Line2D([], [], color="0.4", lw=1.0, ls=(0, (3, 1.5)), label="torus bifurcation")])
    fig.legend(handles=hd, loc="lower center", ncol=7, frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.0)
    out = os.path.join(dp.ensure(dp.mpmf("circadian_data", "writeup")), "scn_bif_entrain_scn" + "".join(map(str, rows)))
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
