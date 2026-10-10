r"""
Unimodal symmetric distributions: does a lower bound on the Lorentzian widths fix the threshold bias?
=====================================================================================================

Distributions (unit scale): Gaussian N(0,1), Skardal g_4 (Δ = 1), uniform U(−1, 1) (= g_n, n → ∞).
Exact references (α = 0, symmetric unimodal ρ):
  * onset of synchrony   K_c = 2 / (π ρ(0))
  * partially locked branch r(K) from the self-consistency  K(u) = 1/G(u), r = u G(u),
    G(u) = ∫_{−π/2}^{π/2} cos²θ ρ(u sinθ) dθ.
LMMF fits: LM.fit (warm start, noise-floor cap, λ = 1e-5, N = 5000) with width bounds
(Δ_min, 100) for a grid of Δ_min, both to a sample ("sample") and to N quantile nodes ("population").
Per fit: K_c(M) (first eigenvalue crossing of the linearisation at z = 0), the steady-state LMMF r(K)
on a K grid (integrated from a synchronized start), the CvM loss, and the peak density ρ_M(0).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_threshold_test.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.stats import norm
import lorentzian_mixture as LM
import skardal_benchmark_simulate as SK

N, LAM, SEED = 5000, 1e-5, 1
DMIN_GRID = [1e-4, 0.05, 0.1, 0.2, 0.3, 0.5]


# ── distributions ────────────────────────────────────────────────────────────
def make_dists():
    g4 = lambda w: SK.gn_density(np.atleast_1d(w), 4, 1.0)
    return {
        "gauss": dict(pdf=lambda w: norm.pdf(np.atleast_1d(w)), cdf=lambda w: norm.cdf(w),
                      sample=lambda n, rng: rng.normal(0, 1, n), support=None),
        "g4": dict(pdf=g4, cdf=None, sample=lambda n, rng: SK.sample_gn(4, 1.0, n, rng), support=None),
        "uniform": dict(pdf=lambda w: np.where(np.abs(np.atleast_1d(w)) <= 1, 0.5, 0.0),
                        cdf=lambda w: np.clip((np.asarray(w) + 1) / 2, 0, 1),
                        sample=lambda n, rng: rng.uniform(-1, 1, n), support=(-1, 1)),
    }


def quantile_nodes(d, n):
    g = np.linspace(-60, 60, 1_200_001)
    F = d["cdf"](g) if d["cdf"] is not None else np.cumsum(d["pdf"](g)) * (g[1] - g[0])
    F = F / F[-1]
    return np.interp((np.arange(n) + 0.5) / n, F, g)


# ── exact references ─────────────────────────────────────────────────────────
def exact_branch(d, Ks):
    rho0 = float(d["pdf"](0.0)[0])
    pts = [] if d["support"] is None else [-np.pi / 2, np.pi / 2]
    def G(u):
        f = lambda th: np.cos(th) ** 2 * float(d["pdf"](u * np.sin(th))[0])
        if d["support"] is not None and u > 1:          # integrand vanishes where |u sinθ| > 1
            a = np.arcsin(1 / u)
            return quad(f, -a, a, limit=400)[0]
        return quad(f, -np.pi / 2, np.pi / 2, limit=400)[0]
    us = np.concatenate([np.linspace(1e-4, 3, 600), np.linspace(3, 60, 600)])
    Kb = np.array([1 / G(u) for u in us]); rb = us / Kb
    # stable branch: r as a function of K on the part where K increases with r beyond the minimum
    i0 = int(np.argmin(Kb))
    Ku, ru = Kb[i0:], rb[i0:]
    r_of_K = np.interp(Ks, Ku, ru, left=0.0, right=ru[-1])
    return 2 / (np.pi * rho0), r_of_K, float(Kb[i0])


# ── LMMF quantities ──────────────────────────────────────────────────────────
def lmmf_Kc(w, Om, De, lo=0.01, hi=50.0):
    A0 = np.diag(1j * Om - De); P = 0.5 * np.outer(np.ones(w.size), w)
    g = lambda K: np.max(np.linalg.eigvals(A0 + K * P).real)
    for _ in range(80):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def lmmf_r(w, Om, De, Ks, T=400.0):
    a = 1j * Om - De
    out = []
    for K in Ks:
        rhs = lambda t, z: a * z + 0.5 * K * ((w @ z) - np.conj(w @ z) * z * z)
        sol = solve_ivp(rhs, (0, T), np.full(w.size, 0.95 + 0j), rtol=1e-8, atol=1e-10, t_eval=[T])
        out.append(abs(w @ sol.y[:, -1]))
    return np.array(out)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for name, d in make_dists().items():
        x_samp = np.asarray(d["sample"](N, rng), float)
        x_pop = quantile_nodes(d, N)
        Kc_true, _, _ = exact_branch(d, np.array([1.0]))
        Ks = Kc_true * np.linspace(1.05, 3.0, 14)
        _, r_true, _ = exact_branch(d, Ks)
        print(f"\n== {name}: K_c = {Kc_true:.4f}, ρ(0) = {float(d['pdf'](0.0)[0]):.4f}", flush=True)
        for src, x in (("population", x_pop), ("sample", x_samp)):
            for dmin in DMIN_GRID:
                r = LM.fit(x, (dmin, 1e2), M_max=16, lambda_M=LAM, patience=3, n_restarts=10, seed=1,
                           method="slsqp", floor_c=1.0)
                m = r["model"]
                kc = lmmf_Kc(m.w, m.Omega, m.Delta)
                rM = lmmf_r(m.w, m.Omega, m.Delta, Ks)
                err_r = float(np.mean(np.abs(rM - r_true)))
                rows.append(dict(dist=name, src=src, dmin=dmin, M=m.M, D=r["data_loss"],
                                 floor=LM.noise_floor(N), Kc=kc, Kc_err=kc / Kc_true - 1,
                                 rho0=float(m.pdf(0.0)[0]), rho0_true=float(d["pdf"](0.0)[0]),
                                 branch_err=err_r, min_width=float(m.Delta.min())))
                q = rows[-1]
                print(f"  {src:10s} Δmin={dmin:<6g} M={q['M']:2d}  D/floor={q['D'] / q['floor']:5.2f}  "
                      f"ρ_M(0)/ρ(0)={q['rho0'] / q['rho0_true']:.3f}  K_c err={100 * q['Kc_err']:+6.1f}%  "
                      f"mean|Δr| on [1.05,3]K_c={err_r:.4f}", flush=True)
    np.save(os.path.join(_HERE, "unimodal_threshold_test.npy"), rows, allow_pickle=True)


if __name__ == "__main__":
    main()
