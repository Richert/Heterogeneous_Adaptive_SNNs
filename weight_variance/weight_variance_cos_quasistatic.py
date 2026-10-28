r"""
Symmetric adaptation (G = cos): structure-aware quasi-static self-consistency along the synchronized branch
==========================================================================================================

Computes, for each adaptation rate μ and on a Δ grid, the synchronized branch of the quasi-static
self-consistency of weight_variance_quasistatic.py (slaved weights A_ij = 1 + κ Re(c_j c_i*), κ = μ/γ)
on the same N = 500 Lorentzian quantiles as the network simulations: R, W = ⟨c²⟩, S = ⟨|c|²⟩, Ā,
V_A/Ā², the effective locking band b_eff, and the last Δ before the branch is lost (fold).

    python weight_variance_cos_quasistatic.py      # → weight_variance_cos_quasistatic.npz (~2 min)
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import time, warnings
import numpy as np

import weight_variance_quasistatic as qs
import weight_variance_rule_micro as base

CONFIG = dict(N=500, K=1.0, gamma=0.01,
              mus=[0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
              deltas=np.round(np.arange(0.01, 1.7, 0.01), 4),
              out=dp.mpmf("weight_variance_cos_quasistatic.npz"))


def main(cfg=CONFIG):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mus, deltas, K, g = np.array(cfg["mus"]), cfg["deltas"], cfg["K"], cfg["gamma"]
    keys = ("R", "W", "S", "Abar", "VArel", "b_eff")
    out = {k: np.full((mus.size, deltas.size), np.nan) for k in keys}
    for m, mu in enumerate(mus):
        t0, kap = time.time(), mu / g
        br = qs.sync_branch(lambda D: base.lorentzian_quantiles(cfg["N"], D), deltas, kap, K)
        for j, (D, s) in enumerate(zip(deltas, br)):
            if s is None:
                continue
            ws = qs.weight_stats(s, kap, mu, g, D)
            for k, v in (("R", s["R"]), ("W", s["W"]), ("S", s["S"]), ("Abar", ws["Abar"]),
                         ("VArel", ws["VArel"]), ("b_eff", s["b_eff"])):
                out[k][m, j] = v
        last = deltas[np.isfinite(out["R"][m])].max()
        print(f"μ={mu}: synchronized branch up to Δ={last:.2f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(cfg["out"], mus=mus, deltas=deltas, K=K, gamma=g, N=cfg["N"], **out)
    print(f"saved {cfg['out']}")


if __name__ == "__main__":
    main()
