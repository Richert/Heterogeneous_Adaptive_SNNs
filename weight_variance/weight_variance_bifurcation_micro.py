r"""
Adaptive-coupling Kuramoto: microscopic sims along the Δ-bifurcation branches (per rule, per μ)
===============================================================================================

Companion to ``weight_variance_bifurcation_rules.py``. For the same (K, γ) and the two adaptation
rates μ used in the bifurcation figure, it simulates the microscopic adaptive network (Eqs. 1, 2, 4)
across a range of heterogeneities Δ, initializing each run ON THE OTT–ANTONSEN MANIFOLD at a STABLE
mean-field steady state, and records the tail-averaged (R, Ā, V_A/Ā²).

The stable states at (rule, Δ, μ) are:
  * SYNCHRONIZED branch — cos: the upper root Ā₊ of γĀ²−(γ+μ)Ā+2μΔ/K (physical where R²≥0 & D≥0);
    sin: Ā≡1, R²=1−2Δ/K (physical for Δ<K/2);
  * ASYNCHRONOUS branch — R=0, Ā=1, stable for Δ>K/2.
In the cos BISTABLE window K/2 < Δ < Δ_SN both are stable, so the run is repeated from EACH initial
condition; results are stored under the branch that seeded them (last axis: 0=sync, 1=async).

Reuses the numba Euler kernel + Lorentzian quantiles from ``weight_variance_rule_micro`` and the
OA-manifold sampler from ``weight_variance_rule_micro_oainit``; run in an env with numba
(`allen`/`sbi`) for the speedup:
    PATH="$HOME/conda/envs/allen/bin:$PATH" python weight_variance_bifurcation_micro.py
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

import weight_variance_rule_micro as base                        # numba kernel + helpers
import weight_variance_rule_micro_oainit as oai                  # OA-manifold IC sampler
from weight_variance_analysis import branches, S_order           # noqa: E402

HAS_NUMBA = base.HAS_NUMBA

CONFIG = dict(
    K=1.0, gamma=0.01,
    mus=[0.05, 0.003],                                           # = (mu_bi, mu_mono) of the bif figure
    rules=["cos", "sin"],
    deltas=list(np.round(np.linspace(0.02, 1.2, 25), 4)),        # spans sync / bistable / async
    N=900, T=5000.0, dt=0.05, trans_frac=0.5, n_rec=200, seed=2,
    out=dp.mpmf("weight_variance_bif_micro_large.npz"),
)

BRANCH_LABELS = ["sync", "async"]                                # storage order on the last axis


# ════════════════════════════════════════════════════════════════════════════
#  stable mean-field states + OA-manifold-initialized simulation
# ════════════════════════════════════════════════════════════════════════════
def va_ca(R, A, Delta, mu, g, K):
    """Full-closure V_A* and the matching covariance C_A*=(γ/μ)V_A* for the OA-manifold init."""
    R2 = R * R
    S = float(np.clip(S_order(np.array([R2]), np.array([A]), np.array([Delta]), K), 0.0, 1.0)[0])
    VA = mu ** 2 / (2 * g ** 2 * (g + 2 * Delta)) * (g * (1.0 - R2 ** 2) + 2 * Delta * (S ** 2 - R2 ** 2))
    return VA, (g / mu) * VA


def stable_states(rule, Delta, mu, g, K):
    """Stable mean-field states at (rule, Δ, μ): list of (branch_idx, R*, Ā*, V_A*, C_A*).
    sync branch where physical (idx 0); asynchronous R=0, Ā=1 where stable, Δ>K/2 (idx 1).
    In the cos bistable window K/2 < Δ < Δ_SN both are returned."""
    states = []
    if rule == "cos":
        br = branches(np.array([Delta]), mu, K, g)["sync"]
        R, A = float(br["R"][0]), float(br["A"][0])
        sync = np.isfinite(R) and np.isfinite(A)
    else:                                                        # sin: Ā ≡ 1
        R2 = 1.0 - 2.0 * Delta / K
        sync = R2 > 0
        R, A = (float(np.sqrt(R2)), 1.0) if sync else (np.nan, np.nan)
    if sync:
        VA, CA = va_ca(R, A, Delta, mu, g, K)
        states.append((0, R, A, VA, CA))
    if Delta > K / 2.0:                                          # asynchronous branch stable
        VA, CA = va_ca(0.0, 1.0, Delta, mu, g, K)
        states.append((1, 0.0, 1.0, VA, CA))
    return states


def simulate_state(rule, Delta, mu, R0, A0, VA0, CA0, cfg):
    """Init the micro network on the OA manifold at (R0, Ā0, V_A0, C_A0), integrate, and return the
    tail-averaged steady (R, Ā, V_A)."""
    N, K, g, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["dt"]
    nsteps = int(cfg["T"] / dt)
    rng = np.random.default_rng(cfg["seed"])
    omega = base.lorentzian_quantiles(N, Delta)
    theta = oai.oa_phases(omega, R0, A0, K, rng)
    A = oai.oa_weights(theta, rule, A0, VA0, CA0, rng)
    rec_steps = np.unique(np.linspace(int(cfg["trans_frac"] * nsteps), nsteps - 1, cfg["n_rec"]).astype(int))

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
                Ab = s1 / n_off
                Rs.append(float(np.abs(e.mean()))); Abars.append(float(Ab))
                VAs.append(float(s2 / n_off - Ab ** 2))
        Rs, Abars, VAs = np.array(Rs), np.array(Abars), np.array(VAs)
    return float(np.mean(Rs)), float(np.mean(Abars)), float(np.mean(VAs))


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rules = cfg["rules"]
    mus = np.asarray(cfg["mus"], float)
    deltas = np.asarray(cfg["deltas"], float)
    g, K = cfg["gamma"], cfg["K"]
    nr, nm, nd = len(rules), mus.size, deltas.size
    print(f"micro bifurcation sims [{'numba' if HAS_NUMBA else 'numpy'}] — N={cfg['N']}, K={K}, "
          f"γ={g}; rules={rules}; μ∈{list(mus)}; {nd} Δ∈[{deltas.min():g},{deltas.max():g}]")

    R = np.full((nr, nm, nd, 2), np.nan)                         # (rule, μ, Δ, branch)
    Abar = np.full_like(R, np.nan)
    VArel = np.full_like(R, np.nan)                              # V_A/Ā² (micro estimate)

    for i, rule in enumerate(rules):
        for m, mu in enumerate(mus):
            for j, D in enumerate(deltas):
                for bi, R0, A0, VA0, CA0 in stable_states(rule, float(D), float(mu), g, K):
                    Rss, Ass, Vss = simulate_state(rule, float(D), float(mu), R0, A0, VA0, CA0, cfg)
                    R[i, m, j, bi] = Rss
                    Abar[i, m, j, bi] = Ass
                    VArel[i, m, j, bi] = Vss / Ass ** 2
                    print(f"  G={rule:<4} μ={mu:<6g} Δ={D:6.3f} [{BRANCH_LABELS[bi]:>5}] "
                          f"init R*={R0:.3f} -> R={Rss:.3f} Ā={Ass:.3f} V_A/Ā²={Vss / Ass ** 2:.4f}")

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    np.savez(cfg["out"], rules=np.array(rules), mus=mus, deltas=deltas,
             branches=np.array(BRANCH_LABELS), R=R, Abar=Abar, VArel=VArel,
             K=K, gamma=g, N=cfg["N"], T=cfg["T"], dt=cfg["dt"], trans_frac=cfg["trans_frac"])
    print(f"[saved] {cfg['out']}")


if __name__ == "__main__":
    main()
