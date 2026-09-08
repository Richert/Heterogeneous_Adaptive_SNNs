r"""
Adaptive-coupling Kuramoto: (K, μ) mean-field sweep of coupling-weight statistics
=================================================================================

Mean-field analogue of ``weight_variance_coherence_sweep.py``, following ``weight_variance_meanfield.py``:
it computes, for every (K, μ) on the microscopic sweep's grid, two mean-field predictions:

  * MICRO-DRIVEN — *takes the microscopic phase coherence R* as given and only simulates the
    coupling-weight statistics driven by it (Eqs. 10, 71–73 for Ā, C_S, C_F, V_A; the R equation,
    Eq. 8, is dropped), holding R at the measured value from ``weight_variance_coherence_sweep.npz``.
    Because R is the micro value, each point sits at the same R as its micro counterpart — a clean
    "at the observed coherence, what weight statistics does the mean field predict?" comparison;
  * FULL — closes the mean field self-consistently (adds Eq. 8 for R), from an initial condition
    comparable to the micro run (R(0) = e^{-σ₀²/2}, the expected coherence of θ_i(0) ~ N(0, σ₀); Ā=1;
    C_S=C_F=V_A=0), so its coherence R is predicted rather than imposed.

Both integrate from uniform weights and window-average exactly as the micro sweep does.

Saves one .npz sharing the micro schema: the micro-driven fields (R passed through, plus Ā, V_A, C_A)
and the full-MF fields suffixed ``_full`` (R_full, Abar_full, VA_full, CA_full) per (K, μ).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_coherence_meanfield.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import os
import numpy as np
from scipy.integrate import solve_ivp

from weight_variance_meanfield import order_parameter_S            # on-manifold S = ⟨|c|²⟩
import weight_variance_coherence_sweep as S                        # shared params (out_dir, etc.)

MICRO_NPZ = dp.kmo_adaptive("weight_variance_coherence_sweep.npz")
CONFIG = dict(S.CONFIG, out_name="weight_variance_coherence_meanfield.npz")


def _window_average(arr, W_rec):
    """Mean over 50 %-overlap windows of size W_rec records (first window = transient, skipped)."""
    n = len(arr)
    starts = np.arange(W_rec, n - W_rec + 1, W_rec // 2)
    return float(np.mean([arr[int(s):int(s) + W_rec].mean() for s in starts]))


def simulate_weights(K, mu, R, p):
    """Integrate the coupling-weight statistics (Ā, C_S, C_F, V_A) at the given, constant microscopic
    coherence R; return window-averaged Ā, V_A, C_A = C_S + C_F."""
    g, delta = p["gamma"], p["delta"]
    te = np.arange(0.0, p["T_total"], p["dts"])
    W_rec = int(round(p["window"] / p["dts"]))

    def rhs(t, y):
        A, CS, CF, V = y
        Sq = float(order_parameter_S(R, A, K, delta))
        sS2, sF2 = 0.5 * (Sq ** 2 - R ** 4), 0.5 * (1.0 - Sq ** 2)   # Eqs. 74, 75
        return [mu * R ** 2 + g * (1.0 - A),                         # Eq. 10
                -g * CS + mu * sS2,                                  # Eq. 71
                -(g + 2.0 * delta) * CF + mu * sF2,                  # Eq. 72
                2.0 * mu * (CS + CF) - 2.0 * g * V]                  # Eq. 73

    sol = solve_ivp(rhs, (0.0, te[-1]), [1.0, 0.0, 0.0, 0.0], t_eval=te, method="RK45",
                    rtol=1e-7, atol=1e-9, max_step=1.0)
    A, CS, CF, V = sol.y
    return dict(Abar=_window_average(A, W_rec), VA=_window_average(V, W_rec),
                CA=_window_average(CS + CF, W_rec))


def simulate_full_mf(K, mu, p):
    """Integrate the full self-consistent MF system (Eqs. 8, 10, 71–73) from a coherent IC comparable
    to the micro run; return window-averaged R, Ā, V_A, C_A = C_S + C_F (R is predicted, not imposed)."""
    g, delta = p["gamma"], p["delta"]
    te = np.arange(0.0, p["T_total"], p["dts"])
    W_rec = int(round(p["window"] / p["dts"]))
    R0 = float(np.exp(-p["sigma0"] ** 2 / 2.0))                   # expected coherence of N(0, σ₀) IC

    def rhs(t, y):
        R, A, CS, CF, V = y
        Sq = float(order_parameter_S(R, A, K, delta))
        sS2, sF2 = 0.5 * (Sq ** 2 - R ** 4), 0.5 * (1.0 - Sq ** 2)
        return [-delta * R + (K * A / 2.0) * R * (1.0 - R ** 2),  # Eq. 8
                mu * R ** 2 + g * (1.0 - A),                      # Eq. 10
                -g * CS + mu * sS2,                               # Eq. 71
                -(g + 2.0 * delta) * CF + mu * sF2,               # Eq. 72
                2.0 * mu * (CS + CF) - 2.0 * g * V]               # Eq. 73

    sol = solve_ivp(rhs, (0.0, te[-1]), [R0, 1.0, 0.0, 0.0, 0.0], t_eval=te, method="RK45",
                    rtol=1e-7, atol=1e-9, max_step=1.0)
    R, A, CS, CF, V = sol.y
    return dict(R=_window_average(R, W_rec), Abar=_window_average(A, W_rec),
                VA=_window_average(V, W_rec), CA=_window_average(CS + CF, W_rec))


def main(cfg=CONFIG):
    dm = np.load(MICRO_NPZ)
    Ks, mus, Rmic = dm["Ks"], dm["mus"], dm["R"]
    p = dict(gamma=float(dm["gamma"]), delta=float(dm["delta"]), sigma0=float(dm["sigma0"]),
             window=float(dm["window"]), T_total=float(dm["T_total"]), dts=float(dm["dts"]))
    print(f"(K,μ) mean-field sweep — Δ={p['delta']}, γ={p['gamma']}, micro-driven + full (σ₀={p['sigma0']}), "
          f"window={p['window']}, T_total={p['T_total']}, {len(mus)}×{len(Ks)} points")

    shape = (len(mus), len(Ks))
    R, Abar, VA, CA = Rmic.copy(), np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    R_full, Abar_full, VA_full, CA_full = (np.full(shape, np.nan) for _ in range(4))
    for i, mu in enumerate(mus):
        for j, K in enumerate(Ks):
            r = simulate_weights(float(K), float(mu), float(Rmic[i, j]), p)
            Abar[i, j], VA[i, j], CA[i, j] = r["Abar"], r["VA"], r["CA"]
            f = simulate_full_mf(float(K), float(mu), p)
            R_full[i, j], Abar_full[i, j], VA_full[i, j], CA_full[i, j] = f["R"], f["Abar"], f["VA"], f["CA"]
            print(f"  μ={mu:7.4f} K={K:5.2f} -> R_mic={Rmic[i, j]:.3f} R_full={f['R']:.3f} | "
                  f"driven V_A/Ā²={r['VA']/r['Abar']**2:.3f}  full V_A/Ā²={f['VA']/f['Abar']**2:.3f}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], cfg["out_name"])
    np.savez(out, Ks=Ks, mus=mus, R=R, Abar=Abar, VA=VA, CA=CA,
             R_full=R_full, Abar_full=Abar_full, VA_full=VA_full, CA_full=CA_full,
             gamma=p["gamma"], delta=p["delta"], window=p["window"], T_total=p["T_total"], dts=p["dts"])
    print(f"[saved] {out}  (grid {shape})")


if __name__ == "__main__":
    main()
