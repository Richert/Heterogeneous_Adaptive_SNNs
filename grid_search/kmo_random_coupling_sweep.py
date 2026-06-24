r"""
Kuramoto with structured coupling vs. mean-only & correlation-aware LMMF
=======================================================================

A globally connected Kuramoto network with a UNIFORM natural-frequency distribution and
frequency-assortative coupling is compared to TWO Lorentzian-mixture mean fields (LMMF) that
SHARE one mixture fit of the frequency distribution:
  * mean-only LMMF       — knows only the mean coupling μ (= 1);
  * correlation-aware LMMF — uses the ANALYTICAL inter-ensemble coupling A_ml = 1 + c²μ_mμ_l.

Coupling construction (frequency-assortative, rank-1 + uniform baseline):
  * A_ij = c²·(ω_i − ω̄)(ω_j − ω̄) + N(1, σ)   (= 1 + c²(ω_i−ω̄)(ω_j−ω̄) + N(0, σ)), ω̄ = mean ω.
    The baseline 1 is a uniform all-to-all term; the rank-1 part couples like-offset oscillators
    positively.  ⟨A_ij⟩ = 1 (the rank-1 part has zero mean since Σ_j(ω_j−ω̄)=0), so μ = 1 and the
    effective global coupling is K·μ = K.  NOTE the row-sum Σ_j A_ij ≈ N is ω-INDEPENDENT here
    (Σ_j(ω_j−ω̄)=0), so the structure is NOT a strength–frequency correlation but a frequency
    GRADING of the field.  c enters as c², so the sign of c is immaterial (sweep c ≥ 0).

Models (all integrated with scipy.integrate.solve_ivp), with a global coupling strength K:
  * micro:  θ̇_i = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i)  — the FULL structured matrix (numpy RHS).
  * mean-only LMMF: ż_m = (iΩ_m−Δ_m) z_m + ½(h − h* z_m²), h = Kμ Σ_l w_l z_l (KFS.simulate_ensemble, K=Kμ).
  * corr.-aware LMMF (ANALYTIC): the responsibility-weighted block average of A is, in closed form,
    A_ml = 1 + c²μ_mμ_l with μ_m = Ω_m − ω̄ (the fitted Lorentzian centre, offset by the mixture mean).
    The field then closes on TWO collective modes Z0 = Σ_l w_l z_l (baseline) and Z1 = Σ_l w_l μ_l z_l
    (frequency-graded):  ż_m = (iΩ_m−Δ_m) z_m + ½(H_m − H_m* z_m²),  H_m = K(Z0 + c²μ_m Z1).
    (No realised matrix is used — μ_m comes straight from the shared LMMF fit.)

Sweep: σ × c, n_trials independent trials — each draws a fresh UNIFORM ω-sample
(ω_i ~ U[−omega_max, omega_max]), refits the (shared) LMMF mixture, and uses a fresh coherent
phase IC θ_i(0)~N(0,σ0).  Comparison metric (downstream) = time-domain RMSE between R_micro(t)
and each MF R(t), reported as mean ± std over trials.  Output: one tidy CSV by `quantity`.

Run in the ``pycobi`` conda env (dev PyRates 1.2.2 + scipy + pandas):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_random_coupling_sweep.py
"""
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import kmo_lorentzian_fit_sweep as KFS          # reuse LMMF fit (KFS.LM) + ensemble MF


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    N=500,                          # microscopic oscillators (full N×N coupling)
    omega_max=1.0,                  # uniform ω ∈ [-omega_max, omega_max]
    K=1.2,                          # global coupling strength (effective coupling = K·⟨A⟩ = K·μ = K;
                                    #   uniform ω ⇒ K_c = 2/(π g(0)) = 4·omega_max/π ≈ 1.27)
    # sweep: additive-noise std σ × coupling coefficient c   (A_ij = c²(ω_i−ω̄)(ω_j−ω̄) + N(1,σ))
    #   structured part ≤ c²·omega_max² ≈ c², baseline = 1, so keep σ ≲ O(1); c enters as c² (c ≥ 0)
    sigma_sweep=[0.0, 0.1, 0.5],
    c_sweep=[0.0, 0.25, 0.5, 0.75, 1.0],
    n_trials=10,
    # coherent initial condition: θ_i(0) ~ N(0, sigma0);  LMMF z_m(0)=R(0)
    sigma0=0.5,
    # shared Lorentzian-mixture fit of the ω-sample (used by BOTH the mean-only and corr.-aware LMMF)
    delta_bounds=(0.01, 1.0), fit_M_max=20, fit_lambda=1e-6, fit_alpha=1e-3,
    fit_restarts=10, fit_loss="cvm", fit_method="slsqp",
    # integration (solve_ivp / RK45) — keys reused by KFS.simulate_ensemble
    T=100.0, dt=1e-2, dts=0.1, rtol=1e-6, atol=1e-8,
    # storage
    save_res=50,                    # block-average the (example) coupling matrix to this size
    seed=1,
    out_csv="/home/rgast/data/mpmf_simulations/kmo_random_coupling_sweep.csv",
)


# ════════════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════════════
def block_average(M, res):
    n = M.shape[0]
    if n <= res:
        return M
    b = n // res
    return M[:res * b, :res * b].reshape(res, b, res, b).mean(axis=(1, 3))


def block_average_1d(v, res):
    n = v.size
    if n <= res:
        return v
    b = n // res
    return v[:res * b].reshape(res, b).mean(axis=1)


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    d = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    return float((xm * ym).sum() / d) if d > 0 else 0.0


def build_coupling(omega, c, sigma, rng):
    """Frequency-assortative coupling A_ij = c²(ω_i−ω̄)(ω_j−ω̄) + N(1, σ)
    (= 1 + c²(ω_i−ω̄)(ω_j−ω̄) + N(0, σ)): uniform baseline + rank-1 frequency-product structure."""
    N = omega.size
    om = omega - omega.mean()
    return 1.0 + (c ** 2) * np.outer(om, om) + sigma * rng.standard_normal((N, N))


def simulate_micro(omega, A, theta0, cfg):
    """Full structured-coupling Kuramoto network (numpy RHS + solve_ivp). Returns t, R(t)."""
    N = omega.size
    KinvN = cfg["K"] / N

    def f(t, theta):
        e = np.exp(1j * theta)
        field = KinvN * (A @ e)                      # (K/N) Σ_j A_ij e^{iθ_j}
        return omega + np.imag(field * np.conj(e))   # ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i)

    t_eval = np.arange(0.0, cfg["T"], cfg["dts"])
    sol = solve_ivp(f, (0.0, cfg["T"]), theta0, method="RK45", t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["dt"])
    R = np.abs(np.exp(1j * sol.y).mean(axis=0))
    return t_eval, R


# ── correlation-aware LMMF (ANALYTIC inter-ensemble coupling) ─────────────────────────────────
#   The responsibility-weighted block average of A_ij = c²(ω_i−ω̄)(ω_j−ω̄) + N(1,σ) is, in closed
#   form, A_ml = 1 + c² μ_m μ_l with μ_m = ⟨ω−ω̄⟩_m ≈ Ω_m − ω̄ (the fitted Lorentzian centre).
#   The micro field (K/N) Σ_j A_ij e^{iθ_j} then averages to H_m = K Σ_l w_l A_ml z_l, which closes
#   on TWO collective modes:  Z0 = Σ_l w_l z_l (uniform baseline) and Z1 = Σ_l w_l μ_l z_l
#   (frequency-graded),  H_m = K(Z0 + c² μ_m Z1).  Per Lorentzian ensemble m:
#       ż_m = (iΩ_m − Δ_m) z_m + ½(H_m − H_m* z_m²).
#   c → 0 (H_m → K Z0) recovers the mean-only LMMF exactly.  No realised matrix is used.
def simulate_ensemble_analytic(model, mu_m, K, c, R0, cfg):
    """Analytic correlation-aware ensemble OA mean field for A_ml = 1 + c²μ_mμ_l (numpy solve_ivp,
    complex). Two-mode forcing H_m = K(Z0 + c²μ_m Z1). Returns t, R(t)."""
    w, coef = model.w, 1j * model.Omega - model.Delta
    wmu = w * mu_m                           # weights for the frequency-graded mode Z1
    g = (c ** 2) * mu_m                       # per-ensemble grading factor c²μ_m

    def f(t, z):
        Z0 = w @ z                            # baseline order parameter Σ_l w_l z_l
        Z1 = wmu @ z                          # frequency-graded order parameter Σ_l w_l μ_l z_l
        H = K * (Z0 + g * Z1)                 # H_m = K(Z0 + c²μ_m Z1)   (vector over ensembles)
        return coef * z + 0.5 * (H - np.conj(H) * z * z)

    t_eval = np.arange(0.0, cfg["T"], cfg["dts"])
    sol = solve_ivp(f, (0.0, cfg["T"]), np.full(w.size, R0 + 0.0j), method="RK45", t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["dt"])
    return t_eval, np.abs(w @ sol.y)         # observable = plain order parameter |Σ_l w_l z_l|


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])
    N, res, K = cfg["N"], cfg["save_res"], cfg["K"]
    mu = 1.0                       # ⟨A⟩ = 1 (rank-1 part has zero mean); effective coupling = K·μ = K

    print(f"frequency-assortative coupling sweep — N={N}, K={K}, ω∈[±{cfg['omega_max']}] "
          f"(resampled + refit per trial), n_trials={cfg['n_trials']}")

    rows = []

    def base():
        return dict(N=N, K=K, omega_max=cfg["omega_max"])

    for trial in range(cfg["n_trials"]):
        omega = np.sort(rng.uniform(-cfg["omega_max"], cfg["omega_max"], N))   # random ω (sorted ↑)
        theta0 = rng.normal(0.0, cfg["sigma0"], N)
        R0 = float(np.abs(np.exp(1j * theta0).mean()))
        # ONE shared LMMF fit of this trial's sampled frequencies (used by both MF models)
        model = KFS.LM.fit(omega, cfg["delta_bounds"], M_max=cfg["fit_M_max"], alpha=cfg["fit_alpha"],
                           lambda_M=cfg["fit_lambda"], patience=3, loss=cfg["fit_loss"],
                           n_restarts=cfg["fit_restarts"], seed=cfg["seed"] + trial,
                           method=cfg["fit_method"])["model"]
        t_mean, R_mean = KFS.simulate_ensemble(model.w, model.Omega, model.Delta, K * mu, R0, cfg,
                                               tag=f"mf{trial}")     # h = Kμ Σ_l w_l z_l
        wbar = float(omega.mean())                 # ω̄ used in the coupling (and to offset μ_m)
        mu_m = model.Omega - wbar                  # ensemble frequency offsets μ_m = Ω_m − ω̄
        print(f"--- trial {trial + 1}/{cfg['n_trials']}  R(0)={R0:.3f}  M={model.M}  "
              f"R_mean(end)={R_mean[-1]:.3f} ---")

        # per-trial ω axis (block-averaged) + the shared fitted mixture
        for k, om in enumerate(block_average_1d(omega, res)):
            rows.append({**base(), "quantity": "omega", "trial": trial, "idx": k, "value": float(om)})
        for k in range(model.M):
            rows.append({**base(), "quantity": "mixture", "trial": trial, "idx": k, "w": float(model.w[k]),
                         "Omega": float(model.Omega[k]), "Delta": float(model.Delta[k])})

        for sigma in cfg["sigma_sweep"]:
            for c in cfg["c_sweep"]:
                A = build_coupling(omega, c, sigma, rng)  # A_ij = c²(ω_i−ω̄)(ω_j−ω̄) + N(1, σ)
                t_m, R_m = simulate_micro(omega, A, theta0, cfg)
                t_c, R_c = simulate_ensemble_analytic(model, mu_m, K, c, R0, cfg)   # H_m=K(Z0+c²μ_m Z1)
                om = omega - wbar                         # realised structure↔coupling correlation:
                c_real = _pearson(A.ravel(), np.outer(om, om).ravel())  # Pearson(A_ij, (ω_i−ω̄)(ω_j−ω̄))
                print(f"  [t{trial}] σ={sigma:<4} c={c:<5}({c_real:+.2f}) -> R_mic={R_m[-1]:.3f} "
                      f"mean={R_mean[-1]:.3f} corr={R_c[-1]:.3f}")

                meta = {**base(), "sigma": sigma, "c": c, "trial": trial}
                for t, v in zip(t_m, R_m):
                    rows.append({**meta, "quantity": "R_micro", "time": float(t), "value": float(v)})
                for name, (tf, Rf) in (("mean", (t_mean, R_mean)), ("corr", (t_c, R_c))):
                    for t, v in zip(tf, Rf):
                        rows.append({**meta, "quantity": "R_mf", "model": name,
                                     "time": float(t), "value": float(v)})
                rows.append({**meta, "quantity": "corr", "c_real": c_real})
                Ab = block_average(A, res)
                for i in range(Ab.shape[0]):
                    for j in range(Ab.shape[1]):
                        rows.append({**meta, "quantity": "A_final", "row": i, "col": j,
                                     "value": float(Ab[i, j])})

    df = pd.DataFrame(rows).reindex(columns=[
        "quantity", "model", "sigma", "c", "trial", "time", "idx", "row", "col", "value",
        "c_real", "w", "Omega", "Delta", "N", "K", "omega_max"])
    os.makedirs(os.path.dirname(cfg["out_csv"]) or ".", exist_ok=True)
    df.to_csv(cfg["out_csv"], index=False)
    print(f"[saved] {cfg['out_csv']}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
