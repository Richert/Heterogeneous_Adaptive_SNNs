r"""
Adaptive-coupling Kuramoto: microscopic Δ-sweep for two adaptation rules
========================================================================

Simulates the microscopic adaptively-coupled Kuramoto network (manuscript Eqs. 1, 2, 4)

    θ̇_i  = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i),
    Ȧ_ij = μ G(θ_j − θ_i) + γ (1 − A_ij),          G ∈ { cos, sin },

with a Lorentzian intrinsic-frequency distribution (centre 0, HWHM Δ; deterministic quantiles).
K and γ are held constant (configurable); the sweep is over the adaptation rule G ∈ {cos, sin}, the
adaptation rate μ, and the heterogeneity Δ (a range spanning the ASYNCHRONOUS and SYNCHRONOUS regime).

For every (rule, μ, Δ) it records the time courses of
    R(t)   = |⟨e^{iθ}⟩|                     (phase coherence),
    Ā(t)   = ⟨A_ij⟩_{i≠j}                    (average coupling weight, off-diagonal),
    V_A(t) = ⟨A_ij²⟩_{i≠j} − Ā(t)²           (coupling-weight variance, off-diagonal),
and the final coupling matrix A_ij (frequency-sorted, block-averaged to `save_res`).

Everything is written to a single .npz consumed by ``weight_variance_rule_meanfield.py`` (which
adds the mean-field predictions and makes the V_A/Ā²-vs-Δ figure). The Euler integrator uses a fused
numba kernel when numba is importable (~10× faster; the `allen` / `sbi` envs have it), and falls
back to an identical pure-numpy loop otherwise (e.g. in `pycobi`):
    PATH="$HOME/conda/envs/allen/bin:$PATH" python weight_variance_rule_micro.py   # numba (fast)
    python weight_variance_rule_micro.py                                            # numpy fallback
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import os
import numpy as np

try:                                    # optional ~10× speedup; falls back to numpy if absent
    import numba as nb                  # present in the `allen` / `sbi` envs (not `pycobi`)
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    # network parameters (constant across the sweep)
    K=1.0, gamma=0.01,
    # sweep axes: adaptation rule G × adaptation rate μ × heterogeneity Δ
    rules=["cos", "sin"],
    mus=[0.005, 0.01, 0.02, 0.04],
    deltas=list(np.round(np.linspace(0.01, 1.0, 20), 4)),   # spans sync (small Δ) → async (large Δ)
    # microscopic network + Euler integration
    N=500, T=5000.0, dt=0.05, trans_frac=0.0,               # steady state = last (1−trans_frac) of T
    sigma0=0.3,                                             # coherent IC: θ_i(0) ~ N(0, sigma0)
    A0=1.0,                                                 # uniform initial weights A_ij(0)=A0
    # storage
    n_rec=500,                                              # recorded time points per run
    save_res=100,                                           # block-average final A / ω axis to this size
    seed=2,
    out=dp.mpmf("weight_variance_rule_micro.npz"),
)


# ════════════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════════════
def lorentzian_quantiles(N, Delta):
    """Deterministic Lorentzian quantiles (ascending), centre 0, HWHM Δ."""
    p = (np.arange(N) + 0.5) / N
    return Delta * np.tan(np.pi * (p - 0.5))


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


# ════════════════════════════════════════════════════════════════════════════
#  microscopic simulation (full N×N adaptive network; numba-accelerated if available)
# ════════════════════════════════════════════════════════════════════════════
if HAS_NUMBA:
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _simulate_numba(theta, A, omega, K, mu, g, dt, nsteps, rec_mask, c_cos, Rs, Abars, VAs):
        """Fused Euler step: precomputed cos/sin, no N×N temporaries, prange over rows. theta and
        A are updated in place; Rs/Abars/VAs are filled at the steps where rec_mask is True.
        c_cos = 1 (cos drive) or 0 (sin drive). Results match the numpy path."""
        N = theta.shape[0]
        KinvN = K / N
        n_off = N * N - N
        ct = np.empty(N); st = np.empty(N); dth = np.empty(N)
        ridx = 0
        for k in range(nsteps):
            for i in range(N):
                ct[i] = np.cos(theta[i]); st[i] = np.sin(theta[i])
            for i in nb.prange(N):                          # rows are independent
                ci = ct[i]; si = st[i]; acc = 0.0
                for j in range(N):
                    sij = st[j] * ci - ct[j] * si           # sin(θ_j − θ_i)
                    cij = ct[j] * ci + st[j] * si           # cos(θ_j − θ_i)
                    acc += A[i, j] * sij
                    Gd = c_cos * cij + (1.0 - c_cos) * sij  # cos / sin adaptation drive
                    A[i, j] += dt * (mu * Gd + g * (1.0 - A[i, j]))
                dth[i] = omega[i] + KinvN * acc
            for i in range(N):
                theta[i] += dt * dth[i]
            if rec_mask[k]:
                cs = 0.0; ss = 0.0
                for i in range(N):
                    cs += ct[i]; ss += st[i]                # R from θ at step start (matches numpy)
                Rs[ridx] = np.sqrt(cs * cs + ss * ss) / N
                s1 = 0.0; s2 = 0.0
                for i in range(N):
                    for j in range(N):
                        a = A[i, j]; s1 += a; s2 += a * a
                for i in range(N):
                    dd = A[i, i]; s1 -= dd; s2 -= dd * dd    # drop the diagonal
                Abar = s1 / n_off
                Abars[ridx] = Abar; VAs[ridx] = s2 / n_off - Abar * Abar
                ridx += 1


def simulate(rule, Delta, mu, cfg):
    """Integrate the adaptive Kuramoto network from a coherent IC. Returns
    (t_rec, R(t), Ā(t), V_A(t), A_final) with off-diagonal weight statistics. Uses the numba
    kernel when available (~10× faster; bit-comparable), else a pure-numpy Euler loop."""
    N, K, g, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["dt"]
    nsteps = int(cfg["T"] / dt)
    rng = np.random.default_rng(cfg["seed"])                # fixed → identical IC across (rule, Δ)
    omega = lorentzian_quantiles(N, Delta)
    theta = rng.normal(0.0, cfg["sigma0"], N)
    A = cfg["A0"] * np.ones((N, N))
    rec_steps = np.unique(np.linspace(0, nsteps - 1, cfg["n_rec"]).astype(int))
    t_rec = rec_steps * dt

    if HAS_NUMBA:
        rec_mask = np.zeros(nsteps, dtype=np.bool_)
        rec_mask[rec_steps] = True
        nr = rec_steps.size
        Rs, Abars, VAs = np.empty(nr), np.empty(nr), np.empty(nr)
        c_cos = 1.0 if rule == "cos" else 0.0
        _simulate_numba(theta, A, omega, float(K), float(mu), float(g), float(dt),
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
            gph = np.conj(e)[:, None] * e[None, :]          # e^{i(θ_j − θ_i)}
            G = np.real(gph) if rule == "cos" else np.imag(gph)  # cos / sin drive
            A = A + dt * (mu * G + g * (1.0 - A))
            if k in rec_set:
                s1 = A.sum() - A[diag, diag].sum()
                s2 = (A * A).sum() - (A[diag, diag] ** 2).sum()
                Abar = s1 / n_off
                Rs.append(float(np.abs(e.mean()))); Abars.append(float(Abar))
                VAs.append(float(s2 / n_off - Abar ** 2))
        Rs, Abars, VAs = np.array(Rs), np.array(Abars), np.array(VAs)

    A_final = block_average(A, cfg["save_res"])
    return t_rec, Rs, Abars, VAs, A_final


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rules = cfg["rules"]
    mus = np.asarray(cfg["mus"], float)
    deltas = np.asarray(cfg["deltas"], float)
    nr, nm, nd, res = len(rules), mus.size, deltas.size, cfg["save_res"]
    print(f"micro adaptive-Kuramoto sweep [{'numba' if HAS_NUMBA else 'numpy'}] — N={cfg['N']}, "
          f"K={cfg['K']}, γ={cfg['gamma']}; rules={rules}; μ∈{list(mus)}; "
          f"{nd} Δ∈[{deltas.min():g},{deltas.max():g}]")

    t_ref = None
    R = np.full((nr, nm, nd, cfg["n_rec"]), np.nan)         # (rule, μ, Δ, time)
    Abar = np.full_like(R, np.nan)
    VA = np.full_like(R, np.nan)
    A_final = np.full((nr, nm, nd, res, res), np.nan)
    omega_axis = np.full((nd, res), np.nan)                 # μ- and rule-independent

    for j, D in enumerate(deltas):
        omega_axis[j] = block_average_1d(lorentzian_quantiles(cfg["N"], D), res)
    for m, mu in enumerate(mus):
        for i, rule in enumerate(rules):
            for j, D in enumerate(deltas):
                t_rec, Rs, Abars, VAs, A_fin = simulate(rule, D, mu, cfg)
                n = Rs.size
                if t_ref is None:
                    t_ref = t_rec
                R[i, m, j, :n], Abar[i, m, j, :n], VA[i, m, j, :n] = Rs, Abars, VAs
                A_final[i, m, j] = A_fin
                print(f"  G={rule:<4} μ={mu:<6g} Δ={D:6.3f} -> R={Rs[-1]:.3f}  Ā={Abars[-1]:.3f}  "
                      f"V_A={VAs[-1]:.4f}  V_A/Ā²={VAs[-1]/Abars[-1]**2:.4f}")

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    np.savez(cfg["out"],
             rules=np.array(rules), mus=mus, deltas=deltas, t=t_ref,
             R=R, Abar=Abar, VA=VA, A_final=A_final, omega_axis=omega_axis,
             K=cfg["K"], gamma=cfg["gamma"], N=cfg["N"], T=cfg["T"],
             dt=cfg["dt"], trans_frac=cfg["trans_frac"], sigma0=cfg["sigma0"], save_res=res)
    print(f"[saved] {cfg['out']}")


if __name__ == "__main__":
    main()
