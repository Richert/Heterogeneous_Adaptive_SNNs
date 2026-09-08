r"""
Adaptive-coupling Kuramoto: microscopic Δ×μ sweep initialized ON THE OTT–ANTONSEN MANIFOLD
==========================================================================================

Same microscopic network and sweep as ``weight_variance_rule_micro.py`` (Eqs. 1, 2, 4; rule G ∈
{cos, sin}, adaptation rate μ, heterogeneity Δ), but instead of a generic coherent start the initial
state is placed ON THE OTT–ANTONSEN MANIFOLD, so the phases carry no off-manifold transient and the
run tests the mean field from a state it can itself represent.

Default initialization (``init="uniform"``):

  * WEIGHTS uniform, A_ij(0) = A0 — hence V_A(0) = C_A(0) = 0, and the whole build-up of the weight
    variance is part of what the mean-field closure has to reproduce.
  * PHASES on the OA manifold with the coherence the mean field predicts for those weights,
    R(0)² = 1 − 2Δ/(K·A0) (0 where negative) — the fixed point of Ṙ = −ΔR + (K Ā/2) R(1−R²) at
    Ā = A0. Sampling at order parameter R (Ψ=0, b=K·A0·R): locked oscillators (|ω|<b) sit at the
    stable fixed point θ*=arcsin(ω/b); drifting oscillators (|ω|>b) are drawn from the wrapped-Cauchy
    invariant density with modulus r(ω)=(|ω|−√(ω²−b²))/b and mean angle sgn(ω)·π/2 (App. B, Eq. B3);
    b=0 ⇒ uniform phases.

With ``init="mf_state"`` the previous behaviour is recovered: phases at the mean-field steady state
R* and STRUCTURED weights realizing (Ā*, V_A*, C_A*) via ``oa_weights``, which tests whether the
microscopic system *stays* at the mean-field fixed point.

Records the same R(t), Ā(t), V_A(t) traces + final A as the base script (so it feeds the same
``weight_variance_rule_meanfield.py`` figure), plus the (R, Ā, V_A, C_A) of the initial state. Reuses
the base script's numba kernel / helpers; run in an env with numba (`allen`/`sbi`) for the speedup.
    PATH="$HOME/conda/envs/allen/bin:$PATH" python weight_variance_rule_micro_oainit.py
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

import weight_variance_rule_micro as base                      # reuse kernel + helpers
from weight_variance_analysis import S_order, branches          # noqa: E402

HAS_NUMBA = base.HAS_NUMBA
lorentzian_quantiles = base.lorentzian_quantiles
block_average = base.block_average
block_average_1d = base.block_average_1d

CONFIG = dict(base.CONFIG)                                      # inherit the base sweep configuration
CONFIG["init"] = "uniform"                                      # "uniform" | "mf_state" (see init_state)
CONFIG["out"] = dp.mpmf("weight_variance_rule_micro_oainit.npz")


# ════════════════════════════════════════════════════════════════════════════
#  mean-field steady state + Ott–Antonsen-manifold initial condition
# ════════════════════════════════════════════════════════════════════════════
def mf_state(rule, Delta, mu, gamma, K):
    """Coherent-IC-selected mean-field steady state (R*, Ā*, V_A*, C_A*)."""
    da = np.array([Delta])
    if rule == "cos":
        br = branches(da, mu, K, gamma)["sync"]                 # synchronized branch (nan if unphysical)
        R, A = float(br["R"][0]), float(br["A"][0])
        if not np.isfinite(R):
            R, A = 0.0, 1.0                                     # → asynchronous branch
    else:                                                       # sin: Ā ≡ 1
        R2 = 1.0 - 2.0 * Delta / K
        R, A = (float(np.sqrt(R2)), 1.0) if R2 > 0 else (0.0, 1.0)
    R2 = R * R
    S = float(np.clip(S_order(np.array([R2]), np.array([A]), da, K), 0.0, 1.0)[0])
    VA = mu ** 2 / (2 * gamma ** 2 * (gamma + 2 * Delta)) * (gamma * (1.0 - R2 ** 2) + 2 * Delta * (S ** 2 - R2 ** 2))
    CA = (gamma / mu) * VA                                      # steady-state balance V̇_A=2μC_A−2γV_A=0
    return R, A, VA, CA


def oa_phases(omega, R, A, K, rng):
    """Sample N phases on the OA manifold with global order parameter R (Ψ=0, ω̄=0)."""
    N = omega.size
    b = K * A * R
    if b <= 1e-12:                                              # R=0 → fully incoherent
        return rng.uniform(-np.pi, np.pi, N)
    theta = np.empty(N)
    locked = np.abs(omega) < b
    theta[locked] = np.arcsin(np.clip(omega[locked] / b, -1.0, 1.0))    # stable locked phase θ*
    od = omega[~locked]                                        # drifting: wrapped Cauchy (Eq. B3)
    r = (np.abs(od) - np.sqrt(od ** 2 - b ** 2)) / b           # modulus ∈ (0,1)
    mu_ph = np.sign(od) * (np.pi / 2.0)                        # mean angle ±π/2
    U = rng.uniform(0.0, 1.0, od.size)
    theta[~locked] = mu_ph + 2.0 * np.arctan(((1.0 - r) / (1.0 + r)) * np.tan(np.pi * (U - 0.5)))
    return theta


def oa_weights(theta, rule, Abar, VA, CA, rng):
    """Weights realizing ⟨A⟩=Ā*, Var(A)=V_A*, Cov(A,G)=C_A* (G = drive at the OA phases)."""
    N = theta.size
    e = np.exp(1j * theta)
    gph = np.conj(e)[:, None] * e[None, :]                     # e^{i(θ_j − θ_i)}
    G = np.real(gph) if rule == "cos" else np.imag(gph)        # G(θ_j − θ_i)
    off = ~np.eye(N, dtype=bool)
    Gbar, sG2 = G[off].mean(), G[off].var()
    if sG2 < 1e-12:
        beta, noise_var = 0.0, max(VA, 0.0)
    else:
        beta = CA / sG2
        noise_var = VA - beta * beta * sG2
        if noise_var < 0.0:                                    # correlation would exceed 1 → cap
            beta, noise_var = np.sqrt(max(VA, 0.0) / sG2), 0.0
    A = Abar + beta * (G - Gbar) + rng.normal(0.0, np.sqrt(noise_var), (N, N))
    return np.ascontiguousarray(A, dtype=float)


def init_state(rule, Delta, mu, cfg, omega, rng):
    """Ott–Antonsen initial condition (θ, A) and the mean-field values it realizes.

    ``cfg["init"]``:
      * ``"uniform"`` (default) — UNIFORM weights A_ij = A0, and phases on the OA manifold at the
        coherence the mean field predicts for those weights: R(0)² = 1 − 2Δ/(K·A0) (0 if negative),
        the fixed point of Ṙ = −ΔR + (K Ā/2) R(1−R²) at Ā = A0. The weight moments therefore start
        at V_A = C_A = 0 and the run tests the closure over the whole transient, not just at the
        fixed point.
      * ``"mf_state"`` — the previous behaviour: phases at the mean-field steady state R* and
        STRUCTURED weights realizing (Ā*, V_A*, C_A*) from ``oa_weights``.
    """
    K = cfg["K"]
    if cfg["init"] == "uniform":
        A0 = float(cfg["A0"])
        R0 = float(np.sqrt(max(1.0 - 2.0 * Delta / (K * A0), 0.0)))
        theta = oa_phases(omega, R0, A0, K, rng)
        A = np.full((omega.size, omega.size), A0, dtype=float)
        return np.ascontiguousarray(theta), np.ascontiguousarray(A), (R0, A0, 0.0, 0.0)
    if cfg["init"] == "mf_state":
        R_mf, A_mf, VA_mf, CA_mf = mf_state(rule, Delta, mu, cfg["gamma"], K)
        theta = oa_phases(omega, R_mf, A_mf, K, rng)
        return theta, oa_weights(theta, rule, A_mf, VA_mf, CA_mf, rng), (R_mf, A_mf, VA_mf, CA_mf)
    raise ValueError(f"unknown init {cfg['init']!r} (use 'uniform' or 'mf_state')")


# ════════════════════════════════════════════════════════════════════════════
#  simulation (OA-manifold IC, then the shared integrator)
# ════════════════════════════════════════════════════════════════════════════
def simulate(rule, Delta, mu, cfg):
    """Initialize on the OA manifold (see ``init_state``), then integrate. Returns
    (t_rec, R(t), Ā(t), V_A(t), A_final, mf_init) with mf_init = (R, Ā, V_A, C_A) at t=0."""
    N, K, g, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["dt"]
    nsteps = int(cfg["T"] / dt)
    rng = np.random.default_rng(cfg["seed"])
    omega = lorentzian_quantiles(N, Delta)
    theta, A, (R_mf, A_mf, VA_mf, CA_mf) = init_state(rule, Delta, mu, cfg, omega, rng)
    rec_steps = np.unique(np.linspace(0, nsteps - 1, cfg["n_rec"]).astype(int))
    t_rec = rec_steps * dt

    if HAS_NUMBA:
        rec_mask = np.zeros(nsteps, dtype=np.bool_)
        rec_mask[rec_steps] = True
        nr = rec_steps.size
        Rs, Abars, VAs = np.empty(nr), np.empty(nr), np.empty(nr)
        c_cos = 1.0 if rule == "cos" else 0.0
        base._simulate_numba(theta, A, omega, float(K), float(mu), float(g), float(dt),
                             int(nsteps), rec_mask, c_cos, Rs, Abars, VAs)
    else:
        KinvN, n_off = K / N, N * N - N
        diag = np.arange(N)
        rec_set = set(int(s) for s in rec_steps)
        Rs, Abars, VAs = [], [], []
        for k in range(nsteps):
            e = np.exp(1j * theta)
            field = A @ e
            theta = theta + dt * (omega + KinvN * np.imag(np.conj(e) * field))
            gph = np.conj(e)[:, None] * e[None, :]
            G = np.real(gph) if rule == "cos" else np.imag(gph)
            A = A + dt * (mu * G + g * (1.0 - A))
            if k in rec_set:
                s1 = A.sum() - A[diag, diag].sum()
                s2 = (A * A).sum() - (A[diag, diag] ** 2).sum()
                Abar = s1 / n_off
                Rs.append(float(np.abs(e.mean()))); Abars.append(float(Abar))
                VAs.append(float(s2 / n_off - Abar ** 2))
        Rs, Abars, VAs = np.array(Rs), np.array(Abars), np.array(VAs)

    A_final = block_average(A, cfg["save_res"])
    return t_rec, Rs, Abars, VAs, A_final, (R_mf, A_mf, VA_mf, CA_mf)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rules = cfg["rules"]
    mus = np.asarray(cfg["mus"], float)
    deltas = np.asarray(cfg["deltas"], float)
    nr, nm, nd, res = len(rules), mus.size, deltas.size, cfg["save_res"]
    print(f"micro OA-manifold-init sweep [{'numba' if HAS_NUMBA else 'numpy'}] — N={cfg['N']}, "
          f"K={cfg['K']}, γ={cfg['gamma']}; rules={rules}; μ∈{list(mus)}; "
          f"{nd} Δ∈[{deltas.min():g},{deltas.max():g}]")

    t_ref = None
    R = np.full((nr, nm, nd, cfg["n_rec"]), np.nan)
    Abar = np.full_like(R, np.nan); VA = np.full_like(R, np.nan)
    A_final = np.full((nr, nm, nd, res, res), np.nan)
    omega_axis = np.full((nd, res), np.nan)
    mf_init = np.full((nr, nm, nd, 4), np.nan)                  # (R*, Ā*, V_A*, C_A*)

    for j, D in enumerate(deltas):
        omega_axis[j] = block_average_1d(lorentzian_quantiles(cfg["N"], D), res)
    for m, mu in enumerate(mus):
        for i, rule in enumerate(rules):
            for j, D in enumerate(deltas):
                t_rec, Rs, Abars, VAs, A_fin, mfi = simulate(rule, D, mu, cfg)
                n = Rs.size
                if t_ref is None:
                    t_ref = t_rec
                R[i, m, j, :n], Abar[i, m, j, :n], VA[i, m, j, :n] = Rs, Abars, VAs
                A_final[i, m, j] = A_fin
                mf_init[i, m, j] = mfi
                print(f"  G={rule:<4} μ={mu:<6g} Δ={D:6.3f} | init R*={mfi[0]:.3f} Ā*={mfi[1]:.3f} "
                      f"V_A*={mfi[2]:.4f} -> end R={Rs[-1]:.3f} Ā={Abars[-1]:.3f} V_A={VAs[-1]:.4f}")

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    np.savez(cfg["out"],
             rules=np.array(rules), mus=mus, deltas=deltas, t=t_ref,
             R=R, Abar=Abar, VA=VA, A_final=A_final, omega_axis=omega_axis, mf_init=mf_init,
             K=cfg["K"], gamma=cfg["gamma"], N=cfg["N"], T=cfg["T"],
             dt=cfg["dt"], trans_frac=cfg["trans_frac"], sigma0=cfg["sigma0"], save_res=res)
    print(f"[saved] {cfg['out']}")


if __name__ == "__main__":
    main()
