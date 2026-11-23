r"""
Δ–β plane of the adaptation phase lag: mean field vs. structure-aware quasi-static self-consistency
==================================================================================================

G_β(φ) = sin(φ + β) interpolates between the antisymmetric (β = 0) and the symmetric (β = π/2) rule.
At fixed μ, γ, K this script computes, for each β on a grid,
  * the mean-field bifurcations: transcritical point Δ_T = K/2 (all β) and, for μ sin β > γ, the fold
    Δ_SN = K(γ + μ sin β)² / (8γμ sin β) of the synchronized branch, which merges with Δ_T at
    sin β* = γ/μ (degenerate transcritical point);
  * the mean-field synchronized branch R_MF(Δ) (stable node);
  * the synchronized branch of the quasi-static self-consistency (R, Ω, W, Ā, V_A/Ā²), continued in Δ
    from Δ_min until it is lost, on the same N = 500 frequency quantiles as the network simulations.

    python weight_variance_beta_plane.py      # → weight_variance_beta_plane.npz (a few minutes)
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

CONFIG = dict(N=500, K=1.0, gamma=0.01, mu=0.03,
              betas=np.linspace(0.0, np.pi / 2, 13),
              deltas=np.round(np.arange(0.02, 1.4001, 0.02), 4),
              out=dp.mpmf("weight_variance_beta_plane.npz"))


def mf_fold(mu, g, K, beta):
    ms = mu * np.sin(beta)
    return K * (g + ms) ** 2 / (8 * g * ms) if ms > g else np.nan


def main(cfg=CONFIG):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    K, g, mu = cfg["K"], cfg["gamma"], cfg["mu"]
    kap = mu / g
    betas, deltas = cfg["betas"], cfg["deltas"]
    nb, nd = betas.size, deltas.size
    out = {k: np.full((nb, nd), np.nan) for k in ("qs_R", "qs_Omega", "qs_Wabs", "qs_Abar", "qs_VArel",
                                                 "mf_R", "mf_Abar", "mf_VArel")}
    fold_mf = np.array([mf_fold(mu, g, K, b) for b in betas])
    last = np.full(nb, np.nan)
    for a, beta in enumerate(betas):
        t0 = time.time()
        x = None
        for j, D in enumerate(deltas):
            mf = qs.meanfield_beta(D, mu, g, K, beta)
            if mf is not None:
                out["mf_R"][a, j], out["mf_Abar"][a, j], out["mf_VArel"][a, j] = mf["R"], mf["Abar"], mf["VArel"]
            om = base.lorentzian_quantiles(cfg["N"], D)
            if x is None:                                              # start from the Kuramoto state
                R0 = np.sqrt(max(1 - 2 * D / K, 0.05))
                x = (R0, 0.0, complex(R0 ** 2))
            st = qs.solve_beta(om, kap, K, beta, x)
            if st is None:
                break
            x = (st["R"], st["Omega"], st["W"])
            ws = qs.weight_stats_beta(st, kap, mu, g, D, beta)
            out["qs_R"][a, j], out["qs_Omega"][a, j] = st["R"], st["Omega"]
            out["qs_Wabs"][a, j], out["qs_Abar"][a, j], out["qs_VArel"][a, j] = abs(st["W"]), ws["Abar"], ws["VArel"]
            last[a] = D
        j = np.nonzero(np.isfinite(out["qs_R"][a]))[0]
        jl = j[-1] if j.size else None
        print(f"β={beta:.3f} ({np.degrees(beta):5.1f}°, κcosβ={kap * np.cos(beta):.2f}, κsinβ={kap * np.sin(beta):.2f}): "
              f"QS branch up to Δ={last[a]:.2f}" + (f" (R={out['qs_R'][a, jl]:.3f}, Ω={out['qs_Omega'][a, jl]:+.4f})" if jl is not None else "")
              + f" | MF fold {fold_mf[a]:.3f}, transcritical {K / 2:.2f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(cfg["out"], betas=betas, deltas=deltas, mu=mu, gamma=g, K=K, N=cfg["N"], fold_mf=fold_mf,
             qs_last=last, beta_star=np.arcsin(min(g / mu, 1.0)), **out)
    print(f"saved {cfg['out']}")


if __name__ == "__main__":
    main()
