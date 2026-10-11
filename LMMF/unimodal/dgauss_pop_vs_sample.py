r"""
Double Gaussian: LMMF incoherence boundary in (K, α) — fit to the true ρ vs. fit to samples (N = 1000).
======================================================================================================

Fits (plain CvM): (i) the noise-floor rule (λ = 1e-5) and (ii) fixed M = 7, each to
  * the true ρ: quantile nodes of ρ (no sampling noise) with N = 1000 (same resolution as the
    samples) and N = 20000 (finer reference),
  * 5 independent samples of N = 1000.
For every fit: the FULL LMMF incoherence boundary (zero contour of the largest eigenvalue real part at
z = 0 on a fine (K, α) grid, so restabilisation is captured), compared with the exact boundary of the
true ρ. Runtime: ~1 min.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python dgauss_pop_vs_sample.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import lorentzian_mixture as LM
import unimodal_kalpha as UK

NAME, N_S, N_SAMPLES = "dgauss", 1000, 5
A_FINE = np.linspace(0.0, 1.45, 146)


def quantile_nodes(n):
    g = np.linspace(-8, 8, 800001)
    F = UK.TAU * norm.cdf(g) + (1 - UK.TAU) * norm.cdf(g, 0, UK.DELTA_DG)
    return np.interp((np.arange(n) + 0.5) / n, F, g)


def fit(x, M=None):
    if M is None:
        m = LM.fit(x, (1e-4, 1e2), M_max=16, lambda_M=UK.LAM, patience=3, n_restarts=10, seed=1,
                   method="slsqp", floor_c=1.0)["model"]
    else:
        prev = None
        for k in range(1, M + 1):
            m, _ = LM.fit_fixed_M(x, k, (1e-4, 1e2), n_restarts=10, seed=1, method="slsqp", init_model=prev)
            m = LM._prune_mixture(m); prev = m
    return dict(w=m.w, Om=m.Omega, De=m.Delta, M=m.M)


K_FINE = np.linspace(0.02, 4.0, 200)


def stability_map(f):
    """Largest real part of the linearisation at z = 0 on the (α, K) grid. Its zero contour is the
    FULL incoherence boundary, including restabilisation (several K crossings per α)."""
    A0 = np.diag(1j * f["Om"] - f["De"]); P = 0.5 * np.outer(np.ones(f["M"]), f["w"])
    G = np.empty((A_FINE.size, K_FINE.size))
    for i, a in enumerate(A_FINE):
        Pa = np.exp(-1j * a) * P
        for j, K in enumerate(K_FINE):
            G[i, j] = np.max(np.linalg.eigvals(A0 + K * Pa).real)
    return G


def summary(G):
    """K_c at α = 0 and the dip: smallest α at which incoherence is stable for some K ∈ [1.3, 1.9]
    while it is unstable at smaller K (restabilisation)."""
    k0 = K_FINE[np.argmax(G[0] > 0)]
    sel = (K_FINE > 1.3) & (K_FINE < 1.9)
    rows = [i for i in range(A_FINE.size) if (G[i, sel] < 0).any() and (G[i, K_FINE < 1.3] > 0).any()]
    return k0, (A_FINE[min(rows)] if rows else np.nan)


def draw(ax, G, **kw):
    ax.contour(K_FINE, A_FINE, G, levels=[0.0], **kw)


def main():
    ex = np.load(UK._f(NAME, "boundary_exact") + ".npy"); ex = ex[np.isfinite(ex[:, 0])]
    sel = (ex[:, 0] > 1.3) & (ex[:, 0] < 1.9)
    print(f"exact:  K_c(α=0) = {ex[np.argmin(np.abs(ex[:, 1])), 0]:.3f}   dip (min α for K∈[1.3,1.9]) = "
          f"{np.abs(ex[sel, 1]).min():.3f}")
    rng = np.random.default_rng(2026)
    samples = [UK.dists()[NAME]["sample"](N_S, rng) for _ in range(N_SAMPLES)]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True, layout="constrained")
    res = {}
    for ax, (lab, M) in zip(axs, (("noise-floor rule", None), ("fixed M = 7", 7))):
        ax.plot(ex[:, 0], np.abs(ex[:, 1]), "--", color="0.1", lw=1.3, label=r"exact (true $\rho$)")
        for src, x, col, lw in ((f"true ρ, N={N_S} nodes", quantile_nodes(N_S), "#2a78d6", 1.6),
                                ("true ρ, N=20000 nodes", quantile_nodes(20000), "#1baf7a", 1.2)):
            f = fit(x, M); G = stability_map(f); k0, dip = summary(G)
            draw(ax, G, colors=[col], linewidths=lw)
            ax.plot([], [], color=col, lw=lw, label=f"{src} (M={f['M']})")
            print(f"[{lab:16s}] {src:22s} M={f['M']:2d}  K_c(α=0)={k0:.3f}  dip={dip:.3f}")
            res[(lab, src)] = (f["M"], k0, dip)
        stats = []
        for i, x in enumerate(samples):
            f = fit(np.asarray(x, float), M); G = stability_map(f); k0, dip = summary(G); stats.append((f["M"], k0, dip))
            draw(ax, G, colors=["#eb6834"], linewidths=0.8, alpha=0.8)
            if i == 0:
                ax.plot([], [], color="#eb6834", lw=0.8, label=f"samples N={N_S}")
        st = np.array(stats)
        print(f"[{lab:16s}] {'samples N=' + str(N_S):22s} M={st[:, 0].astype(int)}  "
              f"K_c(α=0)={st[:, 1].mean():.3f}±{st[:, 1].std():.3f}  dip={np.nanmean(st[:, 2]):.3f}±{np.nanstd(st[:, 2]):.3f}")
        ax.set_xlim(0, 4); ax.set_ylim(0, 1.45); ax.set_xlabel(r"coupling $K$")
        ax.set_title(lab, fontsize=7)
        ax.legend(fontsize=5.5, loc="lower right")
    axs[0].set_ylabel(r"phase lag $\alpha$")
    out = os.path.join(UK.OUT, "fig_dgauss_pop_vs_sample")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg")
    print(f"[saved] {out}.png/.svg")


if __name__ == "__main__":
    main()
