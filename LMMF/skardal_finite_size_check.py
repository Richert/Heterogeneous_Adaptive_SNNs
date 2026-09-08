r"""
Skardal benchmark — finite-size validation
===========================================

Question (for the PRL): why does the Skardal mean field deviate from the microscopic model
more than the Lorentzian-mixture ("best fit") ensemble does, at large n?

Hypothesis: the Skardal reduction is EXACT in the thermodynamic limit (N→∞) on the
Ott–Antonsen manifold — no extra approximation. The benchmark runs the microscopic model at
FINITE N with frequencies SAMPLED from g_n, near the synchronisation threshold (K=1.2 just
below K_c=2/(π g_n(0))≈1.27–1.41), where the decay rate is very sensitive to the realised
frequency set. The "best fit" ensemble is fit to those SAME samples, so it tracks the finite-N
run; Skardal represents the ideal g_n that the finite-N system only approaches.

Decisive test: increase N. If the deviation is finite-size, micro(N) -> Skardal as N grows.
We report ‖R_micro − R_Skardal‖ vs N, and the empirical density mismatch at ω=0 (which sets
the effective K_c).  Run in the ``pycobi`` env.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "grid_search"))
import kmo_lorentzian_fit_sweep as KFS
import skardal_benchmark_simulate as SK

Delta, K, sigma0 = 1.0, 1.2, 0.5
cfg = dict(T=40.0, dt=1e-2, dts=0.1, rtol=1e-6, atol=1e-8)
N_LIST = [5000, 20000, 80000]
N_LIST_REF = 320000          # "almost continuum" reference sample


def main():
    for n in [20, 200]:
        Kc = 2.0 / (np.pi * SK.gn_density(np.array([0.0]), n, Delta)[0])
        print(f"\n=== n={n}:  g_n(0)={SK.gn_density(np.array([0.0]),n,Delta)[0]:.4f}, "
              f"K_c=2/(π g(0))={Kc:.4f}, K={K}  ({'below' if K<Kc else 'above'} threshold) ===")
        # Skardal reference (uses the ideal g_n; IC R0 = continuum value exp(-sigma0^2/2))
        R0_inf = float(np.exp(-0.5 * sigma0 ** 2))
        t_sk, R_sk = SK.simulate_skardal(n, Delta, K, R0_inf, cfg, tag=f"sk{n}")
        print(f"  Skardal R(end)={R_sk[-1]:.4f}   (dim={n})")
        for N in N_LIST:
            rng = np.random.default_rng(1)
            omega = SK.sample_gn(n, Delta, N, rng)
            theta0 = rng.normal(0.0, sigma0, N)
            R0 = float(np.abs(np.exp(1j * theta0).mean()))
            # empirical density at 0 (KDE-free: fraction in a small window / width)
            hw = 0.05 * Delta
            g0_emp = np.mean(np.abs(omega) < hw) / (2 * hw)
            t_m, R_m = KFS.simulate_micro(omega, K, theta0, cfg, tag=f"fs{n}_{N}")
            L = min(len(R_m), len(R_sk))
            dev = float(np.sqrt(np.mean((R_m[:L] - R_sk[:L]) ** 2)))
            print(f"  N={N:>7}: R0={R0:.4f}  ĝ(0)={g0_emp:.4f} (exact {SK.gn_density(np.array([0.0]),n,Delta)[0]:.4f})  "
                  f"micro R(end)={R_m[-1]:.4f}  ‖R_micro−R_Skardal‖={dev:.4f}")


if __name__ == "__main__":
    main()
