r"""
RMSE vs. distance to the fold (K_SN) and to the Hopf point (K_H) — two figures + bifurcation table.
Reads kmo_bifurcation_cauchy.py outputs.
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_bifurcation_cauchy_figure.py

Each figure: left panel CvM fits, right panel CvM+Cauchy fits (ε = 0.1, β = 1); colour = M,
line style = side of the (true) bifurcation (solid: K below, dashed: K above); black = true-ρ
continuum mean field (finite-size reference). Mean spectral RMSE vs. network over N_TRIALS
realisations, shading = s.e.m.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared"), os.path.join(_HERE, "continuity")]
import functools
import numpy as np
import matplotlib.pyplot as plt
from prl_style import set_prl_style as _set_prl_style, panel_label
import kmo_bifurcation_accuracy as BA
import kmo_bifurcation_cauchy as BC

set_prl_style = functools.partial(_set_prl_style, "prl")
_lab = functools.partial(panel_label, dx=-16, dy=4)
C_M = {2: "#eda100", 4: "#1baf7a", 6: "#2a78d6"}
LS = {"left": "-", "right": "--"}


def spectral_rmse(a, b):
    n = min(a.size, b.size)
    return float(np.sqrt(np.mean((np.abs(np.fft.rfft(a[:n])) - np.abs(np.fft.rfft(b[:n]))) ** 2)) / n)


def main():
    fits = {f["idx"]: f for f in BC.load_fits()}
    s = np.load(BC.SIMS_NPZ)
    tr = np.load(BA.TRUTH_NPZ)
    sel = [fits[i] for i in s["fit_idx"]]
    nK, nT = s["R_net"].shape[:2]
    # RMSE[k, trial, model]; model -1 = true-ρ continuum
    E = np.zeros((nK, nT, len(sel) + 1))
    for k in range(nK):
        for t in range(nT):
            Rn = s["R_net"][k, t]
            for j in range(len(sel)):
                E[k, t, j] = spectral_rmse(Rn, s["R_fit"][k, t, j])
            E[k, t, -1] = spectral_rmse(Rn, s["R_cont"][k])          # continuum: once per K
    mean, sem = E.mean(1), E.std(1, ddof=1) / np.sqrt(nT)

    # bifurcation table
    print(f"true: K_H = {float(tr['K_H']):.4f}   K_SN = {float(tr['K_SN']):.4f}")
    for f in BC.load_fits():
        b = np.load(os.path.join(BA.OUT, f"cauchy_bif_{f['idx']:02d}.npy"), allow_pickle=True).item()
        print(f"  {f['name']:34s} K_H = {b['K_H']:.4f} ({100 * (b['K_H'] / float(tr['K_H']) - 1):+5.1f}%)   "
              f"K_SN = {b['K_SN']:.4f} ({100 * (b['K_SN'] / float(tr['K_SN']) - 1):+5.1f}%)")
    # mean RMSE summary per bifurcation / side
    for bif in ("hopf", "fold"):
        for side in ("left", "right"):
            m = (s["bif"] == bif) & (s["side"] == side)
            near = m & (s["delta"] <= 0.1); far = m & (s["delta"] >= 0.45)
            txt = "  ".join(f"{f['name'].split(' (')[0]}: {mean[near, j].mean():.1e}/{mean[far, j].mean():.1e}"
                            for j, f in enumerate(sel))
            print(f"[{bif:4s} {side:5s}] near(δ≤0.1)/far(δ≥0.45):  {txt}  |  true ρ: "
                  f"{mean[near, -1].mean():.1e}/{mean[far, -1].mean():.1e}")

    set_prl_style()
    for bif, Kname, fname in (("fold", r"K_{SN}", "rmse_vs_fold"), ("hopf", r"K_H", "rmse_vs_hopf")):
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, layout="constrained")
        for ax, kind, ttl, lab in zip(axs, ("cvm", "cauchy"), ("CvM fits", r"CvM + Cauchy-transform loss"), "ab"):
            for side in ("left", "right"):
                m = (s["bif"] == bif) & (s["side"] == side)
                d = s["delta"][m]; o = np.argsort(d)
                for j, f in enumerate(sel):
                    if f["kind"] != kind:
                        continue
                    y, e = mean[m, j][o], sem[m, j][o]
                    ax.plot(d[o], y, LS[side], color=C_M[f["M"]], lw=1.2, marker="o", ms=2.2)
                    ax.fill_between(d[o], y - e, y + e, color=C_M[f["M"]], alpha=0.15, lw=0)
                y, e = mean[m, -1][o], sem[m, -1][o]
                ax.plot(d[o], y, LS[side], color="0.1", lw=1.2, marker="s", ms=2.0)
                ax.fill_between(d[o], y - e, y + e, color="0.1", alpha=0.12, lw=0)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(rf"distance to the true ${Kname}$, $|K-{Kname}|$")
            ax.set_title(ttl, fontsize=7, pad=2)
            _lab(ax, lab)
        axs[0].set_ylabel(f"spectral RMSE vs network\n(mean of {nT} realisations)")
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color=C_M[M], lw=1.2, label=f"LMMF $M={M}$") for M in (2, 4, 6)]
        h += [Line2D([], [], color="0.1", lw=1.2, label=r"true $\rho$ (continuum)"),
              Line2D([], [], color="0.5", ls="-", lw=1.0, label=rf"$K<{Kname}$"),
              Line2D([], [], color="0.5", ls="--", lw=1.0, label=rf"$K>{Kname}$")]
        axs[1].legend(handles=h, fontsize=5.5, ncol=2, loc="best")
        out = os.path.join(BA.OUT, fname)
        fig.savefig(out + ".png", dpi=250); fig.savefig(out + ".svg")
        plt.close(fig)
        print(f"[saved] {out}.png/.svg")


if __name__ == "__main__":
    main()
