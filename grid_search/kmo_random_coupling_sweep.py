r"""
Kuramoto with structured coupling vs. mean-only & correlation-aware LMMF
=======================================================================

A globally connected Kuramoto network with a UNIFORM natural-frequency distribution and
structured coupling is compared to TWO Lorentzian-mixture mean fields (LMMF) that SHARE one
mixture fit of the frequency distribution:
  * mean-only LMMF       — knows only the mean coupling μ;
  * correlation-aware LMMF — additionally uses the per-ensemble in/out coupling strengths.

Coupling construction:
  * each oscillator i (sorted by its frequency ω_i) gets a scalar strength k_i = 1 + c·p_i,
    p_i ∈ [−1, 1] linear in the frequency-sorted index, so MEDIAN k = 1 and the slope c is swept;
  * A_ij = k_i · k_j + N(0, σ)   (rank-1 structure + additive per-edge Gaussian noise).
    ⟨A_ij⟩ = ⟨k⟩² = 1, so the effective global coupling is K·μ = K; the in/out strength Σ_j A_ij ∝ k_i
    is linear in ω-rank, so c sets the strength–frequency correlation and σ the disorder.  Since
    A ≈ k_i k_j is exactly the separable form the correlation-aware reduction assumes, that MF is
    near-exact.

Models (all integrated with scipy.integrate.solve_ivp), with a global coupling strength K:
  * micro:  θ̇_i = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i)  — the FULL structured matrix (numpy RHS).
  * mean-only LMMF: ż_m = (iΩ_m−Δ_m) z_m + ½(h − h* z_m²), h = Kμ Σ_l w_l z_l (KFS.simulate_ensemble, K=Kμ).
  * corr.-aware LMMF: ż_m = (iΩ_m−Δ_m) z_m + (K a_m/2)(z_out − z_out* z_m²), z_out = Σ_l w_l b_l z_l,
    with a_m, b_m the responsibility-weighted ensemble in/out strengths of the realised A.

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
    # sweep: additive-noise std σ × strength-slope c   (k_i = 1 + c·p_i, p_i∈[-1,1])
    sigma_sweep=[0.0, 0.5, 1.0],
    c_sweep=[-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9],
    n_trials=10,
    # coherent initial condition: θ_i(0) ~ N(0, sigma0);  LMMF z_m(0)=R(0)
    sigma0=0.5,
    # shared Lorentzian-mixture fit of the ω-sample (used by BOTH the mean-only and corr.-aware LMMF)
    delta_bounds=(0.01, 1.0), fit_M_max=20, fit_lambda=1e-6, fit_alpha=1e-3,
    fit_restarts=10, fit_loss="cvm", fit_method="slsqp",
    # integration (solve_ivp / RK45) — keys reused by KFS.simulate_ensemble
    T=50.0, dt=1e-2, dts=0.1, rtol=1e-6, atol=1e-8,
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


def build_strengths(N, slope):
    """Per-oscillator coupling strength k_i = 1 + slope·p_i, with p_i ∈ [−1, 1] linear in the
    frequency-sorted index (so MEDIAN k = 1).  `slope` (= c) is the swept parameter."""
    return 1.0 + slope * np.linspace(-1.0, 1.0, N)


def build_coupling(k, sigma, rng):
    """Structured coupling A_ij = k_i k_j + N(0, σ) (rank-1 strength structure + edge noise)."""
    N = k.size
    return np.outer(k, k) + sigma * rng.standard_normal((N, N))


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


# ── correlation-aware LMMF (knows the frequency–coupling correlation) ─────────────────────────
#   Annealed reduction A_ij ≈ s_i^in s_j^out / s_total ⇒  θ̇_i = ω_i + a_i Im[e^{-iθ_i} z_out],
#   a_i = (Σ_j A_ij)/N (normalised in-strength, mean μ),  z_out = ⟨b e^{iθ}⟩ = Σ_l w_l b_l z_l,
#   b_i = (Σ_i A_ij)/(Nμ) (normalised out-strength, mean 1).  Per Lorentzian ensemble m:
#       ż_m = (iΩ_m − Δ_m) z_m + (a_m/2)(z_out − z_out* z_m²),
#   with a_m, b_m the responsibility-weighted ensemble strengths of the REALISED matrix A.
#   a_m→μ, b_m→1 (no correlation) recovers the mean-only LMMF exactly.
def ensemble_strengths(A, omega, model, mu):
    """Responsibility-weighted normalised in-/out-strength (a_m, b_m) per Lorentzian ensemble."""
    N = omega.size
    a_i = A.sum(axis=1) / N                 # in-strength  Σ_j A_ij / N        (mean μ)
    b_i = A.sum(axis=0) / (N * mu)          # out-strength Σ_i A_ij / (Nμ)     (mean 1)
    comp = model.w[None, :] * KFS.LM._comp_pdf(omega, model.Omega, model.Delta)   # (N, M)
    r = comp / np.where(comp.sum(axis=1, keepdims=True) > 0, comp.sum(axis=1, keepdims=True), 1.0)
    wsum = np.where(r.sum(axis=0) > 0, r.sum(axis=0), 1.0)
    return (r * a_i[:, None]).sum(axis=0) / wsum, (r * b_i[:, None]).sum(axis=0) / wsum


def simulate_ensemble_corr(model, a, b, R0, cfg):
    """Correlation-aware ensemble OA mean field (numpy solve_ivp, complex). Returns t, R(t)."""
    w, coef, wb = model.w, 1j * model.Omega - model.Delta, model.w * b

    def f(t, z):
        z_out = wb @ z                       # out-strength-weighted field Σ_l w_l b_l z_l
        return coef * z + 0.5 * a * (z_out - np.conj(z_out) * z * z)

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
    mu = 1.0                       # ⟨A⟩ = ⟨k⟩² = 1 (median/mean k = 1); effective coupling = K·μ = K

    print(f"structured-coupling sweep — N={N}, K={K}, ω∈[±{cfg['omega_max']}] "
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
                k = build_strengths(N, c)                 # k_i = 1 + c·p_i  (frequency-sorted index)
                A = build_coupling(k, sigma, rng)         # A_ij = k_i k_j + N(0, σ)
                t_m, R_m = simulate_micro(omega, A, theta0, cfg)
                a_m, b_m = ensemble_strengths(A, omega, model, mu)   # corr.-aware ensemble strengths
                t_c, R_c = simulate_ensemble_corr(model, K * a_m, b_m, R0, cfg)   # forcing K·a_m·z_out
                c_real = _pearson(A.sum(axis=1), omega)   # realised strength↔frequency correlation
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
