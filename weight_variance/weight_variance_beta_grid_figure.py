r"""
Δ–β plane of the adaptation phase lag: network grid vs. mean field and quasi-static theory
=========================================================================================

Diagnostic two-row figure (weight_variance_beta_grid.npz + weight_variance_beta_plane.npz):
  top    — R (coherent init), R (asynchronous init), Ω (coherent init);
  bottom — largest frequency-locked fraction, phase coherence of that cluster, V_A/Ā² (coherent init).
Overlays: mean-field transcritical point Δ = K/2 (dotted), mean-field fold (dashed), end of the
quasi-static synchronized branch (solid), sin β* = γ/μ (thin horizontal).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_beta_grid_figure.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
from prl_style import set_prl_style as _set_prl_style
from prl_style import panel_label, COL_DOUBLE
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

GRID_NPZ = dp.mpmf("weight_variance_beta_grid.npz")
PLANE_NPZ = dp.mpmf("weight_variance_beta_plane.npz")
OUT = dp.mpmf("weight_variance_beta_grid")

set_prl_style = functools.partial(_set_prl_style, "prl", **{"legend.fontsize": 5.2})
_panel_label = functools.partial(panel_label, dx=-22, dy=3)


def overlay(ax, p, K):
    bd = np.degrees(p["betas"])
    ax.axvline(K / 2, color="k", ls=":", lw=0.8)
    ax.plot(p["fold_mf"], bd, color="k", ls="--", lw=0.9)
    ax.plot(p["qs_last"], bd, color="w", lw=1.6)
    ax.plot(p["qs_last"], bd, color="k", lw=0.8)
    ax.axhline(np.degrees(float(p["beta_star"])), color="0.4", lw=0.5)


def main():
    d, p = np.load(GRID_NPZ), np.load(PLANE_NPZ)
    K = float(d["K"])
    bd, D = np.degrees(d["betas"]), d["deltas"]
    set_prl_style()
    fig, axes = plt.subplots(2, 3, figsize=(COL_DOUBLE, 4.2), constrained_layout=True, sharex=True, sharey=True)
    panels = [(d["R"][..., 0], r"$R$ (coherent init)", "viridis", (0, 1)),
              (d["R"][..., 1], r"$R$ (asynchronous init)", "viridis", (0, 1)),
              (d["Omega"][..., 0], r"$\Omega$ (coherent init)", "RdBu_r", None),
              (d["lock_frac"][..., 0], "largest locked fraction", "magma", (0, 1)),
              (d["R_cluster"][..., 0], "coherence of locked cluster", "magma", (0, 1)),
              (d["VArel"][..., 0], r"$V_A/\bar A^2$ (coherent init)", "cividis", None)]
    for ax, (Z, title, cmap, lim), l in zip(axes.ravel(), panels, "abcdef"):
        if title.startswith(r"$\Omega$"):
            v = np.nanmax(np.abs(Z))
            lim = (-v, v)
        im = ax.pcolormesh(D, bd, Z, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None,
                           shading="nearest")
        fig.colorbar(im, ax=ax, shrink=0.85)
        overlay(ax, p, K)
        ax.set_title(title, fontsize=6.5)
        ax.set_xlim(D.min() - 0.04, D.max() + 0.04)
        ax.set_ylim(-3, 93)
        _panel_label(ax, l)
    for ax in axes[1]:
        ax.set_xlabel(r"$\Delta$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\beta$ (deg)")
    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".png", dpi=300)
    print(f"saved {OUT}.png/.svg")


if __name__ == "__main__":
    main()
