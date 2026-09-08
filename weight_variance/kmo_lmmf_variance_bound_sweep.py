r"""
LMMF vs. adaptive Kuramoto — discrepancy vs. the weight-variance budget V_A/Ā²
==============================================================================

Systematically contrasts the *Lorentzian-mixture mean field* (LMMF) against the
microscopic system of adaptively coupled Kuramoto oscillators it approximates, as a
function of the **upper bound on the predicted relative weight variance** V_A/Ā²
(the manuscript's weight-heterogeneity budget, Eqs. 32/37).

Idea (why a variance bound sets the ensemble widths)
----------------------------------------------------
The microscopic population has a *uniform* natural-frequency distribution ω ~ U(−a, a). A
single adaptive ensemble neglects the coupling-weight heterogeneity that plasticity builds
up *within* the ensemble — captured by the relative weight variance V_A/Ā². The LMMF
reduces this error by tiling the population with M narrower Lorentzian ensembles: each
ensemble then carries a *smaller* within-ensemble variance, and the between-ensemble
variance is represented explicitly by the M×M mean-coupling matrix Ā_{ml}. The closed-form
V_A/Ā² prediction only has to hold *per ensemble* (each ensemble is locally Lorentzian-like).

So the control knob is a budget `rv_max` on the predicted per-ensemble V_A/Ā². For each
budget we invert the closed-form relative-variance curve rv(Δ) (weight_variance_analysis,
synchronized branch, manuscript Eqs. 32/37) to the ensemble width Δ_max whose predicted
V_A/Ā² equals the budget (first crossing of the inverted-U). The Lorentzian-mixture fit
(shared/lorentzian_mixture) then tiles the uniform distribution with ensembles of width
Δ_m = Δ_max (widths *pinned* at the budget, cf. `pin_widths`; the fit still selects the
number of ensembles M and their centres/weights by penalized goodness-of-fit). A smaller
budget ⇒ narrower Δ_max ⇒ more ensembles M* ⇒ the LMMF captures more of the true weight
variance. NB: pinning is essential for a uniform target — with only an *upper* width bound
the CvM fit always chooses narrow ensembles far below the cap, so the budget would not bind
and the sweep would be degenerate. The same Δ_max (canonical cos-rule prediction) is used
for all three rules so the ensemble decomposition — hence the heterogeneity budget — is
directly comparable across rules; only the adaptation dynamics differ.

Every parameter combination is repeated over `n_trials` independent random realizations of
the uniform frequencies (and hence independent LMMF fits and initial conditions).

Models (weight-variance convention: K/N coupling, decay γ toward Ā=1; Eqs. 1/4/8)
--------------------------------------------------------------------------------
* micro: all-to-all adaptive Kuramoto, per-pair weights A_ij (reused from
    ``kmo_adaptive_single_sweep.simulate_micro``):
        θ̇_i = ω_i + (K/N) Σ_j A_ij sin(θ_j−θ_i),  Ȧ_ij = μ G(θ_j−θ_i) + γ(1−A_ij).
* LMMF: M adaptive OA ensembles (built here; the M=1 limit reproduces the single-ensemble
    mean field of ``kmo_adaptive_single_sweep`` and the closed form of
    ``weight_variance_analysis``):
        ż_m = (iΩ_m−Δ_m) z_m + ½(h_m − h_m* z_m²),   h_m = K Σ_l w_l Ā_{ml} z_l,
        Ā̇_{ml} = μ D_{ml}(z) + γ(1−Ā_{ml}),
    with the rule-specific ensemble-mean drive (Daido moments Z_k=z^k on the OA manifold):
        cos  : D_{ml} = Re(z_m* z_l)
        sin  : D_{ml} = Im(z_m* z_l)
        |sin|: D_{ml} = 2/π − (4/π) Σ_{n=1}^{n_trunc} Re((z_m* z_l)^{2n})/(4n²−1)   (App. B2).
    The LMMF weight statistics are the between-block moments Ā = wᵀĀw and
    V_A = wᵀ(Ā∘Ā)w − Ā² (the within-block variance is exactly what the LMMF drops).

Per sweep point the tidy CSV (discriminated by `quantity`, tagged by `trial`) stores, for
BOTH models, the average phase coherence R(t), the average weight Ā(t) and the weight
variance V_A(t); for the microscopic model also the final (frequency-sorted, block-averaged)
coupling matrix A_ij; plus the fitted Lorentzian-mixture parameters per (trial, μ, rv_max).
The micro run depends only on (trial, μ, rule) — not on rv_max — so it is simulated once per
(trial, μ, rule).

Run in the ``pycobi`` conda env (dev PyRates 1.2.2 + scipy + pandas):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_lmmf_variance_bound_sweep.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# microscopic adaptive Kuramoto + shared helpers (weight-variance convention)
from kmo_adaptive_single_sweep import simulate_micro, block_average, block_average_1d
# closed-form relative-variance curve rv(Δ) and Lorentzian-mixture fit
import weight_variance_analysis as WVA
import lorentzian_mixture as LM


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    # microscopic population: uniform natural frequencies ω ~ U(−a, a)
    N=500,                          # number of microscopic oscillators
    omega_halfwidth=1.0,            # uniform support half-width a (on the synchronized branch)
    n_trials=5,                     # independent random realizations of ω (and thus LMMF fits)
    # coupling / adaptation (manuscript values)
    K=1.2, gamma=0.001,
    A0=1.0,                         # initial coupling weight (uniform A_ij(0)=A0)
    sigma0=0.5,                     # coherent IC: θ_i(0) ~ N(0, sigma0)
    # sweep axes
    mu_sweep=[0.001, 0.005],        # adaptation rate μ (μ=γ and μ=5γ)
    G_A_rules=["cos", "sin", "|sin|"],
    # 5 upper bounds on the predicted per-ensemble relative weight variance V_A/Ā² (descending
    # ⇒ narrower Δ_max ⇒ more ensembles). Each maps to Δ_max via the cos-rule rv(Δ) inversion;
    # for a=1.0, K=1.2 these give M*≈2,3,5,9,10 (see delta_max_for + `pin_widths`).
    rv_bounds=[0.08, 0.04, 0.02, 0.01, 0.005],
    # Lorentzian-mixture fit (shared/lorentzian_mixture.fit)
    pin_widths=True,                # pin every ensemble width at Δ_max (the budget); False =>
                                    # (delta_min, Δ_max) upper bound only (degenerate: the CvM
                                    # fit then picks narrow widths and M* stops tracking Δ_max)
    delta_min=1e-3,                 # lower width bound when pin_widths=False
    M_max=20,                       # max ensembles (tightest budget needs ~9 — keep headroom)
    alpha=1e-2, lambda_M=1e-5, patience=3, n_restarts=8, loss="cvm", fit_method="slsqp",
    n_trunc=10,                     # Fourier-truncation order for the |sin| mean field
    # integration (shared time grid for micro & LMMF); T ≫ 1/γ so the slow weights settle.
    # `method` (solve_ivp integrator) is also read by the imported micro simulator's _run.
    T=5000.0, dts=4.0,
    method="RK45", rtol=1e-6, atol=1e-8,
    # storage
    save_res=100,                   # block-average the final A matrix / ω axis to this size
    seed=1,
    out_csv=dp.mpmf("kmo_lmmf_variance_bound_sweep.csv"),
)


# ════════════════════════════════════════════════════════════════════════════
#  rv(Δ) inversion: weight-variance budget -> ensemble width cap
# ════════════════════════════════════════════════════════════════════════════
def delta_max_for(rv_max, mu, K, gamma, n=6000):
    """Largest ensemble width Δ_max whose predicted relative weight variance V_A/Ā² stays
    below `rv_max`. rv(Δ) is the cos-rule synchronized-branch curve (WVA.branches, Eqs.
    32/37); it is an inverted-U (0 at Δ→0 and at the branch end), so the *first* crossing
    from below is the narrow-ensemble cap. If `rv_max` exceeds the peak, the whole branch
    is admissible ⇒ return (just below) the branch endpoint."""
    d_end = WVA.sync_delta_end(mu, K, gamma)
    dlt = np.linspace(1e-4, 0.999 * d_end, n)
    br = WVA.branches(dlt, mu, K, gamma)["sync"]
    rv = br["VA"] / br["A"] ** 2
    above = np.where(rv >= rv_max)[0]
    return float(dlt[above[0]]) if above.size else float(dlt[-1])


# ════════════════════════════════════════════════════════════════════════════
#  LMMF: M adaptive Ott–Antonsen ensembles (numpy RHS + solve_ivp)
# ════════════════════════════════════════════════════════════════════════════
def _lmmf_rhs(t, y, M, w, Om, De, K, mu, gamma, rule, ntrunc, c0, c1, coef):
    """State y = [Re z (M), Im z (M), Ā (M²)]. Field h_m = K Σ_l w_l Ā_{ml} z_l; the
    rule-specific drive uses G_{ml} = z_m* z_l (Daido moments Z_k=z^k on the OA manifold)."""
    z = y[:M] + 1j * y[M:2 * M]
    Ab = y[2 * M:].reshape(M, M)
    h = K * (Ab @ (w * z))                                   # h_m = K Σ_l w_l Ā_{ml} z_l
    dz = (1j * Om - De) * z + 0.5 * (h - np.conj(h) * z * z)
    G = np.outer(np.conj(z), z)                              # G_{ml} = z_m* z_l
    if rule == "cos":
        D = G.real
    elif rule == "sin":
        D = G.imag
    else:                                                    # |sin| via Fourier truncation
        D = np.full((M, M), c0)
        Gp = np.ones_like(G)
        for k in range(ntrunc):
            Gp = Gp * G * G                                  # G^{2(k+1)}
            D = D - c1 * coef[k] * Gp.real
    dAb = mu * D + gamma * (1.0 - Ab)
    out = np.empty_like(y)
    out[:M] = dz.real
    out[M:2 * M] = dz.imag
    out[2 * M:] = dAb.ravel()
    return out


def simulate_lmmf(w, Om, De, K, mu, gamma, rule, R0, A0, cfg, return_final_A=False):
    """Integrate the M-ensemble adaptive LMMF from a coherent IC (z_m(0)=R0, Ā_{ml}(0)=A0).
    Returns (t, R(t), Ā(t), V_A(t)) with Ā = wᵀĀw and V_A = wᵀ(Ā∘Ā)w − Ā² (between-block).
    With `return_final_A`, also returns the final M×M mean-coupling matrix Ā_{ml}(T)."""
    M = w.size
    c0, c1 = 2.0 / np.pi, 4.0 / np.pi
    coef = np.array([1.0 / (4 * n * n - 1) for n in range(1, cfg["n_trunc"] + 1)])
    z0 = np.full(M, float(R0) + 0.0j)
    y0 = np.concatenate([z0.real, z0.imag, np.full(M * M, float(A0))])
    t_eval = np.arange(0.0, cfg["T"], cfg["dts"])
    sol = solve_ivp(_lmmf_rhs, (0.0, cfg["T"]), y0, method=cfg["method"], t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"],
                    args=(M, w, Om, De, K, mu, gamma, rule, cfg["n_trunc"], c0, c1, coef))
    z = sol.y[:M] + 1j * sol.y[M:2 * M]                      # (M, n_t)
    Ab = sol.y[2 * M:].reshape(M, M, -1)                     # (M, M, n_t)
    R = np.abs(w @ z)                                        # |Σ_m w_m z_m|
    Abar = np.einsum("m,mlk,l->k", w, Ab, w)                 # wᵀĀw
    VA = np.einsum("m,mlk,l->k", w, Ab ** 2, w) - Abar ** 2  # between-block variance
    if return_final_A:
        return t_eval, R, Abar, VA, Ab[:, :, -1]
    return t_eval, R, Abar, VA


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    N, K, gamma, res = cfg["N"], cfg["K"], cfg["gamma"], cfg["save_res"]
    A0, a, n_trials = cfg["A0"], cfg["omega_halfwidth"], cfg["n_trials"]
    dbnd = (lambda dmax: (dmax, dmax) if cfg["pin_widths"] else (cfg["delta_min"], dmax))

    print(f"LMMF vs. adaptive Kuramoto — N={N}, K={K}, γ={gamma}, ω~U(±{a}), "
          f"n_trials={n_trials}, pin_widths={cfg['pin_widths']}")
    print(f"  rv_max bounds {cfg['rv_bounds']}  rules {cfg['G_A_rules']}  μ {cfg['mu_sweep']}")

    # width caps Δ_max per (μ, rv_max) via the cos-rule rv(Δ) inversion (trial-independent)
    delta_caps = {mu: {rv: delta_max_for(rv, mu, K, gamma) for rv in cfg["rv_bounds"]}
                  for mu in cfg["mu_sweep"]}
    for mu in cfg["mu_sweep"]:
        caps = ", ".join(f"{rv:g}->Δ={delta_caps[mu][rv]:.3f}" for rv in cfg["rv_bounds"])
        print(f"  μ={mu:<6}: {caps}")

    rows = []

    def base():
        return dict(K=K, N=N, gamma=gamma, omega_halfwidth=a)

    for trial in range(n_trials):
        # independent uniform-frequency realization (sorted ⇒ A_final is frequency-ordered)
        # + coherent phase IC; a distinct RNG stream per trial for reproducibility
        rng = np.random.default_rng(cfg["seed"] + trial)
        omega = np.sort(rng.uniform(-a, a, N))
        theta0 = rng.normal(0.0, cfg["sigma0"], N)
        R0 = float(np.abs(np.exp(1j * theta0).mean()))
        print(f"=== trial {trial + 1}/{n_trials}  R(0)={R0:.3f} ===")

        # microscopic oscillator frequencies (block-averaged axis)
        for k, om in enumerate(block_average_1d(omega, res)):
            rows.append({**base(), "quantity": "omega", "trial": trial, "idx": k, "value": float(om)})

        # Lorentzian-mixture fits per (μ, rv_max)  [rule-independent: uses the cos-rule Δ_max]
        fits = {}
        for mu in cfg["mu_sweep"]:
            for rv in cfg["rv_bounds"]:
                dmax = delta_caps[mu][rv]
                r = LM.fit(omega, dbnd(dmax), M_max=cfg["M_max"], alpha=cfg["alpha"],
                           lambda_M=cfg["lambda_M"], patience=cfg["patience"], loss=cfg["loss"],
                           n_restarts=cfg["n_restarts"], seed=cfg["seed"] + trial,
                           method=cfg["fit_method"], verbose=False)
                fits[(mu, rv)] = (r["model"], r["M"], dmax, float(r["pvalue"]))
                m = r["model"]
                print(f"  fit μ={mu:<6} rv≤{rv:<6g} Δ_max={dmax:.3f} -> M*={r['M']:2d}  "
                      f"p={r['pvalue']:.3f}")
                for k in range(m.M):
                    rows.append({**base(), "quantity": "mixture", "trial": trial, "mu": mu,
                                 "rv_max": rv, "M_star": m.M, "Delta_max": dmax, "idx": k,
                                 "w": float(m.w[k]), "Omega": float(m.Omega[k]),
                                 "Delta": float(m.Delta[k])})

        # sweep: micro once per (μ, rule); LMMF once per (μ, rule, rv_max)
        for rule in cfg["G_A_rules"]:
            for mu in cfg["mu_sweep"]:
                # ---- microscopic model (independent of rv_max) ----
                t_m, R_m, Ab_m, VA_m, A_fin = simulate_micro(theta0, A0, omega, K, mu, gamma,
                                                             rule, cfg)
                print(f"  [micro] G={rule:6s} μ={mu:<6} -> R={R_m[-1]:.3f} Ā={Ab_m[-1]:.3f} "
                      f"V_A={VA_m[-1]:.4f} V_A/Ā²={VA_m[-1] / Ab_m[-1] ** 2:.4f}")
                mmeta = {**base(), "G_A": rule, "mu": mu, "trial": trial}
                for q, tr in (("R_micro", R_m), ("Abar_micro", Ab_m), ("VA_micro", VA_m)):
                    for t, v in zip(t_m, tr):
                        rows.append({**mmeta, "quantity": q, "time": float(t), "value": float(v)})
                Ab_mat = block_average(A_fin, res)
                for i in range(Ab_mat.shape[0]):
                    for j in range(Ab_mat.shape[1]):
                        rows.append({**mmeta, "quantity": "A_final", "row": i, "col": j,
                                     "value": float(Ab_mat[i, j])})

                # ---- LMMF model per rv_max ----
                for rv in cfg["rv_bounds"]:
                    model, Mstar, dmax, pval = fits[(mu, rv)]
                    t_f, R_f, Ab_f, VA_f = simulate_lmmf(model.w, model.Omega, model.Delta,
                                                         K, mu, gamma, rule, R0, A0, cfg)
                    print(f"    [lmmf] G={rule:6s} μ={mu:<6} rv≤{rv:<6g} (M*={Mstar:2d}) -> "
                          f"R={R_f[-1]:.3f} Ā={Ab_f[-1]:.3f} V_A={VA_f[-1]:.4f} "
                          f"V_A/Ā²={VA_f[-1] / Ab_f[-1] ** 2:.4f}")
                    fmeta = {**base(), "G_A": rule, "mu": mu, "trial": trial, "rv_max": rv,
                             "M_star": Mstar, "Delta_max": dmax}
                    for q, tr in (("R_mf", R_f), ("Abar_mf", Ab_f), ("VA_mf", VA_f)):
                        for t, v in zip(t_f, tr):
                            rows.append({**fmeta, "quantity": q, "time": float(t), "value": float(v)})

    df = pd.DataFrame(rows).reindex(columns=[
        "quantity", "trial", "G_A", "mu", "rv_max", "M_star", "Delta_max", "time", "idx", "row",
        "col", "value", "w", "Omega", "Delta", "K", "N", "gamma", "omega_halfwidth"])
    os.makedirs(os.path.dirname(cfg["out_csv"]) or ".", exist_ok=True)
    df.to_csv(cfg["out_csv"], index=False)
    print(f"[saved] {cfg['out_csv']}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
