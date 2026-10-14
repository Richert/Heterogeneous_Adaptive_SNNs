r"""
Frequency-locking overview for the illustration samples (draft, for discussion)
===============================================================================

Rows: gauss, g8, dgauss, bimodal (N = 1000, illustration.py). Columns:
  network   log10 of the locked fraction ℓ (backward ramp, second half of every holding window), exact
            incoherence threshold of ρ (dashed) and the ℓ = ELL_C contour of every LMMF (fit colours)
  pop / floor / nofloor   log10 ℓ of each LMMF's test oscillators, network ℓ = ELL_C contour (black)
Printed: mean |ℓ_fit − ℓ_net| over the grid and over the transition band (0.01 < ℓ_net < 0.5).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python fig_locking_illustration.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import illustration as IL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from prl_style import set_prl_style

ELL_C, FLOOR = 0.05, 1e-3
C_FIT = {"pop": "#2a78d6", "floor": "#eb6834", "nofloor": "#7b2cbf"}
TITLE = {"pop": r"LMMF, fit to true $\rho$", "floor": "LMMF, noise floor", "nofloor": "LMMF, no floor"}
LAB = {"gauss": "Gaussian", "g8": r"rational $g_8$", "dgauss": "double Gaussian", "bimodal": "bimodal"}


def main():
    set_prl_style("prl")
    fig, axs = plt.subplots(len(IL.DISTS), 4, figsize=(7.0, 1.55 * len(IL.DISTS) + 0.4), layout="constrained")
    norm = Normalize(np.log10(FLOOR), 0.0)
    for r, d in enumerate(IL.DISTS):
        L = IL.locking_maps(d, "bwd")
        K, A = L["K"], L["alpha"]
        ext = [K[0] - 0.5 * (K[1] - K[0]), K[-1] + 0.5 * (K[1] - K[0]), A[0] - 0.025, A[-1] + 0.025]
        ex = IL.exact_boundary(d); ex = ex[np.isfinite(ex[:, 0]) & (ex[:, 0] <= K[-1]) & (ex[:, 1] >= 0)]
        maps = [L["ell_net"]] + [L["ell_fit"][j] for j in range(len(IL.KINDS))]
        for c, (ax, M) in enumerate(zip(axs[r], maps)):
            im = ax.imshow(np.log10(np.maximum(M, FLOOR)), origin="lower", aspect="auto", extent=ext,
                           cmap="viridis", norm=norm, interpolation="nearest")
            ax.plot(ex[:, 0], ex[:, 1], ".", ms=0.6, color="w")
            if c == 0:
                for j, k in enumerate(IL.KINDS):
                    ax.contour(K, A, L["ell_fit"][j], levels=[ELL_C], colors=[C_FIT[k]], linewidths=0.9)
                ax.set_ylabel(LAB[d] + "\n" + r"$\alpha$", labelpad=2)
            else:
                ax.contour(K, A, L["ell_net"], levels=[ELL_C], colors=["k"], linewidths=0.8)
                ax.contour(K, A, M, levels=[ELL_C], colors=[C_FIT[IL.KINDS[c - 1]]], linewidths=0.9)
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title("network" if c == 0 else TITLE[IL.KINDS[c - 1]], fontsize=6.5, pad=3)
            if r == len(IL.DISTS) - 1:
                ax.set_xlabel(r"$K$", labelpad=1)
        band = (L["ell_net"] > 0.01) & (L["ell_net"] < 0.5)
        print(f"{d:8s} mean |ℓ_fit − ℓ_net| over the grid / transition band ({band.sum()} cells): "
              + ", ".join(f"{k} {np.abs(L['ell_fit'][j] - L['ell_net']).mean():.3f} / "
                          f"{np.abs(L['ell_fit'][j] - L['ell_net'])[band].mean():.3f}" for j, k in enumerate(IL.KINDS)))
    cb = fig.colorbar(im, ax=axs, shrink=0.5, pad=0.01)
    cb.set_label(r"$\log_{10}$ locked fraction $\ell$")
    h = [Line2D([], [], color=C_FIT[k], lw=0.9, label=rf"{TITLE[k]}: $\ell={ELL_C:g}$") for k in IL.KINDS]
    h += [Line2D([], [], color="k", lw=0.8, label=rf"network: $\ell={ELL_C:g}$"),
          Line2D([], [], ls="none", marker=".", color="0.5", label=r"exact threshold (true $\rho$)")]
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=6)
    out = os.path.join(IL.OUT, "fig_locking_illustration")
    fig.savefig(out + ".png", dpi=250); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    main()
