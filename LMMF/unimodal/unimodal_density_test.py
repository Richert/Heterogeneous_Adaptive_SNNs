r"""
Unimodal threshold bias: direct density-matching term (population fits).
=========================================================================

The LMMF onset of synchrony is a POINTWISE density property (for a symmetric fit with a stationary
critical mode, K_c = 2/(π ρ_M(0)) exactly; off-centre ripples can trigger oscillatory modes earlier).
Neither the CvM loss nor the ε-smoothed Cauchy-transform loss constrains ripples narrower than ε.
Here the fit minimises

    J = D_CvM / D0 + β · Q / Q0,     Q = ∫ (ρ_M(ω) − ρ̂(ω))² dω,

with ρ̂ a Gaussian kernel density estimate of the data (Silverman bandwidth) and D0, Q0 the values at
the CvM fit, on a fixed grid with analytic gradients; M is that of the CvM fit (Δ_min = 1e-4).
Runtime: ~1–5 s per refit (9 refits in total).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_density_test.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import lorentzian_mixture as LM
import unimodal_threshold_test as U
import unimodal_cauchy_test as C

BETAS = [1.0, 10.0, 100.0]
GRID = np.linspace(-4, 4, 1601)


def fit_density(x, w0, Om0, De0, beta, dbounds=(1e-4, 1e2), n_jitter=4, seed=0):
    xs = np.sort(x); n = xs.size; u = (np.arange(n) + 0.5) / n; M = w0.size
    g = GRID; dg = g[1] - g[0]
    rho_hat = gaussian_kde(x)(g)

    def Q_and_grad(p):
        w, Om, De = p[:M], p[M:2 * M], p[2 * M:]
        dx = g[:, None] - Om[None, :]; den = dx ** 2 + De ** 2
        P = (De / np.pi) / den                                   # (G, M) component densities
        res = P @ w - rho_hat
        dP_dOm = P * 2 * dx / den
        dP_dDe = P * (1 / De - 2 * De / den)
        Q = float(np.sum(res ** 2) * dg)
        gq = 2 * dg * np.concatenate([P.T @ res, w * (dP_dOm.T @ res), w * (dP_dDe.T @ res)])
        return Q, gq

    p0 = np.concatenate([w0, Om0, De0])
    D0 = LM._cvm_obj_natural(p0, M, xs, n, u)[0]; Q0 = Q_and_grad(p0)[0]

    def J(p):
        d, gd = LM._cvm_obj_natural(p, M, xs, n, u)
        q, gq = Q_and_grad(p)
        return d / D0 + beta * q / Q0, gd / D0 + beta * gq / Q0

    rng = np.random.default_rng(seed)
    starts = [p0] + [np.concatenate([w0, Om0 + rng.normal(0, 0.05, M), De0 * np.exp(rng.normal(0, 0.2, M))])
                     for _ in range(n_jitter)]
    bounds = [(0, 1)] * M + [(None, None)] * M + [dbounds] * M
    eq = dict(type="eq", fun=lambda p: p[:M].sum() - 1)
    best = min((minimize(J, s, method="SLSQP", jac=True, bounds=bounds, constraints=[eq],
                         options=dict(maxiter=1000, ftol=1e-12)) for s in starts), key=lambda r: r.fun)
    p = best.x; w = np.clip(p[:M], 0, None); w /= w.sum()
    return w, p[M:2 * M], p[2 * M:]


def main():
    refs = C._refs()
    for name, d in U.make_dists().items():
        x = U.quantile_nodes(d, U.N)
        b = np.load(os.path.join(C.OUT, C._tag(name, 1e-4) + ".npy"), allow_pickle=True).item()
        print(f"\n== {name}  (CvM only: K_c err={100 * b['Kc_err']:+.1f}%  mean|Δr|={b['branch_err']:.4f}  "
              f"D/floor={b['D_floor']:.2f}  bump={b['bump']:.2f})")
        for beta in BETAS:
            w, Om, De = fit_density(x, b["w"], b["Om"], b["De"], beta)
            mt = C._metrics(d, refs[name], w, Om, De, x)
            print(f"  +density β={beta:<5g} M={w.size}  K_c err={100 * mt['Kc_err']:+6.1f}%  "
                  f"mean|Δr|={mt['branch_err']:.4f}  D/floor={mt['D_floor']:5.2f}  bump={mt['bump']:.2f}",
                  flush=True)
            np.save(os.path.join(C.OUT, f"{name}_density_beta{beta:g}.npy"),
                    dict(dist=name, beta=beta, w=w, Om=Om, De=De, **mt), allow_pickle=True)


if __name__ == "__main__":
    main()
