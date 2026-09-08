r"""
Adaptive-coupling Kuramoto: Δ-sweep of the connectivity structure (cos & sin, μ=0.04)
======================================================================================

Simulates the microscopic adaptive Kuramoto network (Eqs. 1, 2, 4) at fixed K=1, γ=0.01, μ=0.04
across a range of heterogeneities Δ, for BOTH adaptation rules G ∈ {cos, sin}. Each run is
INITIALIZED AT THE MEAN-FIELD STEADY STATE: phases on the Ott–Antonsen manifold at the predicted R*,
and UNIFORM weights A_ij = Ā* (the coherent-IC-selected branch: synchronized where physical, else
asynchronous R=0, Ā=1). For every (rule, Δ) it records, after an initial transient:

  * the FINAL coupling matrix A_ij (raw N×N, frequency-sorted; float32),
  * the complex pairwise resultant between every oscillator pair,
        Z_ij = ⟨ e^{i(θ_i(t) − θ_j(t))} ⟩_t ,
    from which the pairwise coherence r_ij = |Z_ij| and the coherence-weighted mean phase difference
    ψ_ij = arg Z_ij follow (accumulated over the tail, subsampled every `pair_stride` steps),
  * the global phase coherence R(t) = |⟨e^{iθ}⟩| at EVERY post-transient step (t ≥ trans_frac·T),
  * the raw global weight variance V_A and average weight Ā.

Consumed by ``weight_variance_connectivity_figure.py``. Uses a dedicated numba Euler kernel (which
also accumulates the pairwise coherence) when numba is importable (`allen`/`sbi`), else an identical
numpy loop:
    PATH="$HOME/conda/envs/allen/bin:$PATH" python weight_variance_connectivity_sweep.py
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

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from weight_variance_rule_micro import lorentzian_quantiles     # noqa: E402
import weight_variance_rule_micro_oainit as oai                 # mf_state + OA-manifold phases

CONFIG = dict(
    K=1.0, gamma=0.01, mu=0.04,
    rules=["cos", "sin"],
    deltas=list(np.round(np.linspace(0.02, 1.2, 25), 4)),        # spans sync → async
    N=500, T=5000.0, dt=0.05, trans_frac=0.8,
    pair_stride=10,                                              # subsample tail steps for r_ij
    seed=2,
    out=dp.mpmf("weight_variance_connectivity.npz"),
)


# ════════════════════════════════════════════════════════════════════════════
#  microscopic simulation with pairwise-coherence accumulation
# ════════════════════════════════════════════════════════════════════════════
if HAS_NUMBA:
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _simulate_conn(theta, A, omega, K, mu, g, dt, nsteps, rec_start, stride, c_cos,
                       pre, pim, Rt):
        """Fused Euler step (as in weight_variance_rule_micro) that additionally accumulates, over
        the tail steps (every `stride`), the pairwise phase factor pre[i,j]+=cos(θ_i−θ_j),
        pim[i,j]+=sin(θ_i−θ_j), and records the global coherence R(t) at EVERY post-transient step
        (k≥rec_start) into Rt. theta and A are updated in place."""
        N = theta.shape[0]
        KinvN = K / N
        ct = np.empty(N); st = np.empty(N); dth = np.empty(N)
        ridx = 0
        for k in range(nsteps):
            for i in range(N):
                ct[i] = np.cos(theta[i]); st[i] = np.sin(theta[i])
            if k >= rec_start:                                  # global coherence R(t), full tail
                cs = 0.0; ss = 0.0
                for i in range(N):
                    cs += ct[i]; ss += st[i]
                Rt[ridx] = np.sqrt(cs * cs + ss * ss) / N
                ridx += 1
            for i in nb.prange(N):                              # rows independent
                ci = ct[i]; si = st[i]; acc = 0.0
                for j in range(N):
                    sij = st[j] * ci - ct[j] * si               # sin(θ_j − θ_i)
                    cij = ct[j] * ci + st[j] * si               # cos(θ_j − θ_i)
                    acc += A[i, j] * sij
                    Gd = c_cos * cij + (1.0 - c_cos) * sij      # cos / sin adaptation drive
                    A[i, j] += dt * (mu * Gd + g * (1.0 - A[i, j]))
                dth[i] = omega[i] + KinvN * acc
            for i in range(N):
                theta[i] += dt * dth[i]
            if k >= rec_start and (k - rec_start) % stride == 0:
                for i in nb.prange(N):
                    ci = ct[i]; si = st[i]
                    for j in range(N):
                        pre[i, j] += ci * ct[j] + si * st[j]    # cos(θ_i − θ_j)
                        pim[i, j] += si * ct[j] - ci * st[j]    # sin(θ_i − θ_j)


def simulate(rule, Delta, cfg):
    """MF-matched IC (OA phases at R*, uniform weights Ā*), integrate; return
    (V_A, Ā, r_pair[N,N], A_final[N,N], R(t), t_rec)."""
    N, K, g, mu, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["mu"], cfg["dt"]
    nsteps = int(cfg["T"] / dt)
    rec_start = int(cfg["trans_frac"] * nsteps)
    stride = cfg["pair_stride"]
    n_pair = len(range(rec_start, nsteps, stride))
    n_rt = nsteps - rec_start                                   # R(t) recorded every post-transient step
    t_rec = np.arange(rec_start, nsteps) * dt
    rng = np.random.default_rng(cfg["seed"])
    omega = lorentzian_quantiles(N, Delta)
    R_mf, A_mf, _, _ = oai.mf_state(rule, Delta, mu, g, K)      # mean-field steady state
    theta = oai.oa_phases(omega, R_mf, A_mf, K, rng)            # phases on OA manifold at R*
    A = A_mf * np.ones((N, N))                                  # UNIFORM weights at Ā*

    if HAS_NUMBA:
        pre, pim = np.zeros((N, N)), np.zeros((N, N))
        Rt = np.empty(n_rt)
        c_cos = 1.0 if rule == "cos" else 0.0
        _simulate_conn(theta, A, omega, float(K), float(mu), float(g), float(dt),
                       int(nsteps), int(rec_start), int(stride), c_cos, pre, pim, Rt)
        Z = (pre + 1j * pim) / n_pair                          # ⟨e^{i(θ_i−θ_j)}⟩
    else:
        KinvN = K / N
        pacc = np.zeros((N, N), dtype=complex)
        Rt = []
        for k in range(nsteps):
            e = np.exp(1j * theta)
            if k >= rec_start:
                Rt.append(float(np.abs(e.mean())))
            field = A @ e
            theta = theta + dt * (omega + KinvN * np.imag(np.conj(e) * field))
            gph = np.conj(e)[:, None] * e[None, :]
            G = np.real(gph) if rule == "cos" else np.imag(gph)
            A = A + dt * (mu * G + g * (1.0 - A))
            if k >= rec_start and (k - rec_start) % stride == 0:
                pacc += e[:, None] * np.conj(e)[None, :]        # e^{i(θ_i − θ_j)}
        Z = pacc / n_pair
        Rt = np.array(Rt)

    diag = np.arange(N)
    n_off = N * N - N
    s1 = A.sum() - A[diag, diag].sum()
    s2 = (A * A).sum() - (A[diag, diag] ** 2).sum()
    Abar = s1 / n_off
    VA = s2 / n_off - Abar ** 2                                 # global (off-diagonal) weight variance
    return float(VA), float(Abar), Z.astype(np.complex64), A.astype(np.float32), Rt, t_rec


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    rules = cfg["rules"]
    deltas = np.asarray(cfg["deltas"], float)
    N = cfg["N"]
    nr, nd = len(rules), deltas.size
    print(f"connectivity Δ-sweep [{'numba' if HAS_NUMBA else 'numpy'}] — N={N}, K={cfg['K']}, "
          f"γ={cfg['gamma']}, μ={cfg['mu']}; rules={rules}; {nd} Δ∈[{deltas.min():g},{deltas.max():g}]")

    VA = np.full((nr, nd), np.nan)
    Abar = np.full((nr, nd), np.nan)
    Zpair = np.full((nr, nd, N, N), np.nan, dtype=np.complex64)  # pairwise resultant ⟨e^{i(θ_i−θ_j)}⟩
    Amat = np.full((nr, nd, N, N), np.nan, dtype=np.float32)     # raw final coupling matrices
    omega_axis = np.full((nd, N), np.nan)                        # frequency-sorted ω (rule-independent)
    for j, D in enumerate(deltas):
        omega_axis[j] = lorentzian_quantiles(N, D)
    Rt_arr, t_rec = None, None                                   # R(t) traces (allocated lazily)

    for i, rule in enumerate(rules):
        for j, D in enumerate(deltas):
            V, Ab, Z, A_fin, Rt, t_r = simulate(rule, float(D), cfg)
            if Rt_arr is None:
                t_rec = t_r
                Rt_arr = np.full((nr, nd, Rt.size), np.nan, dtype=np.float32)
            VA[i, j], Abar[i, j] = V, Ab
            Zpair[i, j], Amat[i, j], Rt_arr[i, j] = Z, A_fin, Rt
            print(f"  G={rule:<4} Δ={D:6.3f} -> Ā={Ab:.3f}  V_A={V:.4f}  ⟨r_ij⟩={np.abs(Z).mean():.3f}  "
                  f"R(0)={Rt[0]:.3f} R(T)={Rt[-1]:.3f}")

    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    np.savez_compressed(cfg["out"], rules=np.array(rules), deltas=deltas,
                        VA=VA, Abar=Abar, Zpair=Zpair, Amat=Amat, omega_axis=omega_axis,
                        Rt=Rt_arr, t_rec=t_rec,
                        K=cfg["K"], gamma=cfg["gamma"], mu=cfg["mu"], N=N, T=cfg["T"],
                        dt=cfg["dt"], trans_frac=cfg["trans_frac"])
    print(f"[saved] {cfg['out']}")


if __name__ == "__main__":
    main()
