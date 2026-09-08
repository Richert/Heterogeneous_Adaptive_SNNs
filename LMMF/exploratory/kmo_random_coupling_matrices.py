r"""
Quick look: structured coupling matrices A_ij over the full (σ, c) sweep grid
=============================================================================

Builds the frequency-assortative coupling A_ij = c²(ω_i−ω̄)(ω_j−ω̄) + N(1,σ) for a small network
(N=50) at every (σ, c) combination in kmo_random_coupling_sweep.CONFIG and tiles them: rows = σ,
columns = c.  One shared colourbar per σ row (so the c-progression is comparable within a noise level).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_random_coupling_matrices.py
"""
import numpy as np
import matplotlib.pyplot as plt

import kmo_random_coupling_sweep as S

N = 50
SEED = 1
CMAP = "magma"
OUT = "/home/rgast/data/mpmf_simulations/kmo_random_coupling_matrices"


def main():
    cfg = S.CONFIG
    sigmas, cs = cfg["sigma_sweep"], cfg["c_sweep"]
    rng = np.random.default_rng(SEED)
    omega = np.sort(rng.uniform(-cfg["omega_max"], cfg["omega_max"], N))   # one shared ω-sample (sorted)

    nr, nc = len(sigmas), len(cs)
    mats = {(si, ci): S.build_coupling(omega, c, s, rng)
            for si, s in enumerate(sigmas) for ci, c in enumerate(cs)}

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    })

    fig, axes = plt.subplots(nr, nc, figsize=(1.05 * nc + 0.5, 1.05 * nr + 0.3),
                             squeeze=False, layout="constrained")
    for si, s in enumerate(sigmas):
        row = [mats[(si, ci)] for ci in range(nc)]
        vmin, vmax = min(M.min() for M in row), max(M.max() for M in row)   # shared scale per σ row
        for ci, c in enumerate(cs):
            ax = axes[si][ci]
            im = ax.imshow(mats[(si, ci)], origin="lower", aspect="equal", cmap=CMAP,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if si == 0:
                ax.set_title(rf"$c={c:g}$", pad=2)
            if ci == 0:
                ax.set_ylabel(rf"$\sigma={s:g}$", labelpad=3)
        cb = fig.colorbar(im, ax=list(axes[si]), location="right", fraction=0.025, pad=0.012)
        cb.ax.tick_params(labelsize=4.5, pad=0.6, length=1.5)

    fig.suptitle(rf"$A_{{ij}} = c^2(\omega_i-\bar\omega)(\omega_j-\bar\omega) + \mathcal{{N}}(1,\sigma)$"
                 rf",  $N={N}$  (oscillators sorted by $\omega$)", fontsize=7.5)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
