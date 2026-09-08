r"""
Saturating (logistic) adaptive coupling: microscopic sweep + moment hierarchy
=============================================================================

Model — Kuramoto oscillators whose coupling weights adapt by a LOGISTIC rule, so
that A_ij is confined to [0, A_m] by the rule itself rather than by a decay term:

    dθ_i/dt  = ω_i + (K/N) Σ_{j≠i} A_ij sin(θ_j − θ_i)
    dA_ij/dt = μ G(θ_j − θ_i) A_ij (A_m − A_ij),        G = cos

This differs structurally from the ``weight_variance_*`` rule dA = μG + γ(1−A):
there each pair has a unique fixed point A* = G/γ, here EVERY pair has two fixed
points (0 and A_m) and the sign of G selects which is stable.  The weight law
therefore saturates towards a two-atom (binary) law instead of staying narrow.

What is recorded
----------------
The off-diagonal (i≠j) pair averages appearing in the exact moment hierarchy

    Ā'   =   Ā(A_m−Ā) Ḡ + (A_m−2Ā) C_A −  P               [' ≡ (1/μ) d/dt]
    V_A' = 2 Ā(A_m−Ā) C_A + 2(A_m−2Ā) P  − 2Q

with a = A − Ā, C_A = ⟨G a⟩, P = ⟨G a²⟩, Q = ⟨G a³⟩; plus the drive moments
⟨G^k⟩, k=1..4, and the coherence R.  Note Ḡ cancels identically from V_A'.

In addition, at a few snapshot times, the per-pair LATENT DRIVE

    u_ij = ⟨G_ij⟩_t   (trailing window average)

is stored together with A_ij on a sub-block of the network.  The latent is what
the closures in ``logistic_closure_meanfield.py`` model: for a frozen u the rule
integrates exactly to A_ij = A_m σ(λ u_ij + b), a sigmoid of the latent.

Usage (run in the ``allen`` env — needs numba)
---------------------------------------------
    PATH="$HOME/conda/envs/allen/bin:$PATH" python logistic_closure_micro.py
    PATH="$HOME/conda/envs/allen/bin:$PATH" python logistic_closure_micro.py partial

Writes one <out>/logistic_closure_<regime>.npz per regime.
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import numpy as np
from numba import njit

# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    N=400,                       # oscillators (pair averages over N(N-1) edges)
    Am=1.0,                      # weight ceiling
    A0=0.5,                      # uniform initial weight -> V_A(0)=C_A(0)=0
    dt=0.01,
    T=1500.0,
    rec_every=200,               # moment-hierarchy sampling interval (steps)
    n_snap=3,                    # latent snapshots (equally spaced over the 2nd half)
    snap_window=100.0,           # trailing window (time units) for u_ij = <G_ij>_t
    n_sub=150,                   # store the latent on an n_sub x n_sub sub-block
    seed=0,
    out_dir=dp.KMO_ADAPTIVE,
)

#: regimes spanning asynchronous -> partially locked -> strongly locked
REGIMES = dict(
    async_=dict(K=1.0, Delta=0.8, mu=0.02),      # R ~ 0.03, weights barely move
    partial=dict(K=2.0, Delta=0.4, mu=0.02),     # R ~ 0.42, strong saturation
    slow=dict(K=2.0, Delta=0.4, mu=0.005),       # same phases, slower adaptation
    sync=dict(K=4.0, Delta=0.2, mu=0.02),        # R ~ 0.9
)


# ════════════════════════════════════════════════════════════════════════════
#  microscopic integrator (numba)
# ════════════════════════════════════════════════════════════════════════════
@njit(cache=True, fastmath=True)
def _moments(theta, A, N):
    """Off-diagonal pair moments of the drive G and of a = A - Ā."""
    Abar = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                Abar += A[i, j]
    npair = N * (N - 1)
    Abar /= npair

    g1 = g2 = g3 = g4 = 0.0
    VA = CA = P = Q = 0.0
    for i in range(N):
        ci, si = np.cos(theta[i]), np.sin(theta[i])
        for j in range(N):
            if i == j:
                continue
            G = np.cos(theta[j]) * ci + np.sin(theta[j]) * si      # cos(θ_j − θ_i)
            a = A[i, j] - Abar
            g1 += G; g2 += G * G; g3 += G * G * G; g4 += G * G * G * G
            VA += a * a
            CA += G * a
            P += G * a * a
            Q += G * a * a * a
    return (Abar, VA / npair, CA / npair, P / npair, Q / npair,
            g1 / npair, g2 / npair, g3 / npair, g4 / npair)


@njit(cache=True, fastmath=True)
def _integrate(omega, A, theta, K, mu, Am, dt, nsteps, rec_every,
               snap_steps, snap_win_steps, n_sub):
    """Euler integration; returns (moment record, latent snapshots, weight snapshots)."""
    N = omega.shape[0]
    nrec = nsteps // rec_every + 1
    rec = np.zeros((nrec, 11))
    n_snap = snap_steps.shape[0]
    u_snap = np.zeros((n_snap, n_sub, n_sub))
    A_snap = np.zeros((n_snap, n_sub, n_sub))
    Gacc = np.zeros((N, N))                    # trailing-window accumulator for u_ij
    nacc = 0
    dth = np.zeros(N)
    k = 0
    si_snap = 0

    for n in range(nsteps + 1):
        if n % rec_every == 0:
            Abar, VA, CA, P, Q, g1, g2, g3, g4 = _moments(theta, A, N)
            zr = zi = 0.0
            for i in range(N):
                zr += np.cos(theta[i]); zi += np.sin(theta[i])
            rec[k, 0] = n * dt
            rec[k, 1] = np.sqrt(zr * zr + zi * zi) / N
            rec[k, 2] = Abar; rec[k, 3] = VA; rec[k, 4] = CA
            rec[k, 5] = P;    rec[k, 6] = Q
            rec[k, 7] = g1;   rec[k, 8] = g2; rec[k, 9] = g3; rec[k, 10] = g4
            k += 1

        # ---- accumulate the latent drive inside the trailing window
        if si_snap < n_snap and n > snap_steps[si_snap] - snap_win_steps:
            for i in range(N):
                ci, si = np.cos(theta[i]), np.sin(theta[i])
                for j in range(N):
                    Gacc[i, j] += np.cos(theta[j]) * ci + np.sin(theta[j]) * si
            nacc += 1
        if si_snap < n_snap and n == snap_steps[si_snap]:
            for i in range(n_sub):
                for j in range(n_sub):
                    u_snap[si_snap, i, j] = Gacc[i, j] / nacc
                    A_snap[si_snap, i, j] = A[i, j]
            Gacc[:, :] = 0.0
            nacc = 0
            si_snap += 1

        # ---- phase update
        for i in range(N):
            acc = 0.0
            ci, si = np.cos(theta[i]), np.sin(theta[i])
            for j in range(N):
                if i != j:
                    acc += A[i, j] * (np.sin(theta[j]) * ci - np.cos(theta[j]) * si)
            dth[i] = omega[i] + K / N * acc
        # ---- weight update (logistic: A and A_m-A both gate the drive)
        for i in range(N):
            ci, si = np.cos(theta[i]), np.sin(theta[i])
            for j in range(N):
                if i != j:
                    G = np.cos(theta[j]) * ci + np.sin(theta[j]) * si
                    A[i, j] += dt * mu * G * A[i, j] * (Am - A[i, j])
        for i in range(N):
            theta[i] += dt * dth[i]

    return rec[:k], u_snap, A_snap


def lorentzian_quantiles(N, centre, width, rng, cutoff=30.0):
    """Deterministic Lorentzian sample (quantile spacing), as in weight_variance_rule_micro."""
    q = (np.arange(1, N + 1) - 0.5) / N
    return np.clip(centre + width * np.tan(np.pi * (q - 0.5)), -cutoff, cutoff)


def simulate(K, Delta, mu, cfg=CONFIG):
    N, Am, dt = cfg["N"], cfg["Am"], cfg["dt"]
    rng = np.random.default_rng(cfg["seed"])
    omega = lorentzian_quantiles(N, 0.0, Delta, rng)
    rng.shuffle(omega)
    theta = 2 * np.pi * rng.random(N)
    A = np.full((N, N), cfg["A0"] * Am)
    nsteps = int(cfg["T"] / dt)
    snap = np.linspace(nsteps // 2, nsteps, cfg["n_snap"]).astype(np.int64)
    rec, u_snap, A_snap = _integrate(
        omega, A, theta, K, mu, Am, dt, nsteps, cfg["rec_every"],
        snap, int(cfg["snap_window"] / dt), cfg["n_sub"])
    cols = "t R Abar VA CA P Q g1 g2 g3 g4".split()
    out = {c: rec[:, i] for i, c in enumerate(cols)}
    out.update(u_snap=u_snap, A_snap=A_snap, snap_t=snap * dt,
               omega=omega, K=K, Delta=Delta, mu=mu, Am=Am, A0=cfg["A0"], N=N)
    return out


# ════════════════════════════════════════════════════════════════════════════
#  entry point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    which = sys.argv[1:] or list(REGIMES)
    dp.ensure(CONFIG["out_dir"])
    for name in which:
        if name not in REGIMES:
            raise SystemExit(f"unknown regime {name!r}; choose from {list(REGIMES)}")
        p = REGIMES[name]
        print(f"[{name}] K={p['K']} Δ={p['Delta']} μ={p['mu']} ... ", end="", flush=True)
        res = simulate(**p)
        f = os.path.join(CONFIG["out_dir"], f"logistic_closure_{name}.npz")
        np.savez_compressed(f, **res)
        Ab, VA, Am = res["Abar"][-1], res["VA"][-1], res["Am"]
        print(f"R={res['R'][-1]:.3f} Ā={Ab:.3f} V_A={VA:.4f} "
              f"s={VA / (Ab * (Am - Ab)):.3f}\n   -> {f}")
