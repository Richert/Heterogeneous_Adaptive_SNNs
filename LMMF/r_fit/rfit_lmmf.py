r"""
Fitting the LMMF from phase-coherence traces R(t) alone (no access to the frequency distribution)
================================================================================================

Ground truth: the Fig. 1 network (Gaussian-mixture frequencies, N = 5000, seed 1, K = 3, alpha = 0),
optionally driven by a phase-referenced periodic forcing

    dθ_i/dt = ω_i + K Im(Z e^{-iθ_i}) + ε sin(Ω_f t − θ_i) = ω_i + Im(e^{-iθ_i} H),
    H = K Z + ε e^{iΩ_f t}.

Initial conditions: θ_i(0) i.i.d. wrapped Cauchy (OA manifold) with <e^{iθ}> = R0 (mean phase 0,
known, i.e. aligned with the forcing phase at t = 0). The model starts from the OBSERVED R(0).

Model (LMMF in coupling-time units, see conversation): fitted parameters
    p = (w_m [softmax], Ω̃_m, log Δ̃_m),  log K,
with Ω_m = K Ω̃_m, Δ_m = K Δ̃_m, i.e. K acts as a pure time scale on a K-free shape model:
    dz_m/dt = (iΩ_m − Δ_m) z_m + ½ (H − H̄ z_m²),  H = K Σ w_l z_l + ε e^{iΩ_f t},  z_m(0) = R0.
Without forcing, the mean frequency is unobservable from R (rotation symmetry); it is gauged to
Σ w_m Ω̃_m = 0.

Scenarios (train / validation split; validation traces select M):
  S1  no input, several initial conditions                 R0 ∈ {0.1, 0.9} / {0.5}
  S2  forcing, several periods, identical initial condition Ω_f ∈ {0.5, 2.5, 3.5} / {1.5}, R0 = 0.9
  S3  forcing, several periods × several initial conditions (Ω_f, R0) ∈ {0.5,2.5,3.5}×{0.1,0.9} /
      all with Ω_f = 1.5 or R0 = 0.5
Common test set (never used for fitting or selection):
      (no forcing, R0 = 0.3), (Ω_f = 1.0, R0 = 0.3), (Ω_f = 3.0, R0 = 0.3)

Fitting: loss = mean squared error of R(t) over [0, T] (all traces of a split), fixed-step RK4 in
JAX (float64, exact gradients) + L-BFGS-B. Per M: warm starts from the (M−1)-fit by duplicating each
component (identical halves reproduce the (M−1)-dynamics exactly, so the training loss cannot
increase with M) and random restarts with a horizon curriculum (T = 5 → 10 → 20 → 30) against
phase-drift local minima. Greedy over M; stop when the validation loss has not improved for
`patience` steps; M* = argmin validation loss.

Baselines on the test set: the distribution-informed LMMF (CvM fit of ρ, Fig. 1 algorithm, true K)
and the continuum mean field of the true ρ (finite-N reference).

Run in the `sbi` conda env (jax):
    $HOME/conda/envs/sbi/bin/python rfit_lmmf.py [data|S1|S2|S3|baselines]
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import lorentzian_mixture as LM

# ── ground truth (Fig. 1) ────────────────────────────────────────────────────
GMM = dict(means=[-2.0, 2.0], stds=[0.6, 0.6], weights=[0.5, 0.5])
N, K_TRUE, SEED = 5000, 3.0, 1
EPS = 1.0                                   # forcing amplitude (known)
T, DTS, DT = 30.0, 0.1, 0.01                # horizon, sampling step, RK4 step
TS = np.arange(0.0, T + 1e-9, DTS)
STRIDE = int(round(DTS / DT))

UNFORCED = None
SCEN = {
    "S1": dict(train=[(UNFORCED, 0.1), (UNFORCED, 0.9)], val=[(UNFORCED, 0.5)], gauge=True),
    "S2": dict(train=[(0.5, 0.9), (2.5, 0.9), (3.5, 0.9)], val=[(1.5, 0.9)], gauge=False),
    "S3": dict(train=[(f, r) for f in (0.5, 2.5, 3.5) for r in (0.1, 0.9)],
               val=[(1.5, r) for r in (0.1, 0.5, 0.9)] + [(f, 0.5) for f in (0.5, 2.5, 3.5)],
               gauge=False),
}
TEST = [(UNFORCED, 0.3), (1.0, 0.3), (3.0, 0.3)]
DATA = os.path.join(_HERE, "rfit_data.npz")


def _key(cond):
    f, r = cond
    return f"f{'none' if f is None else f'{f:g}'}_r{r:g}"


def all_conditions():
    out = []
    for s in SCEN.values():
        out += s["train"] + s["val"]
    out += TEST
    return sorted(set(out), key=lambda c: (-1 if c[0] is None else c[0], c[1]))


# ── network data (scipy, independent of the JAX model integrator) ───────────
def sample_omega():
    rng = np.random.default_rng(SEED)
    m, s, w = (np.asarray(GMM[k], float) for k in ("means", "stds", "weights"))
    comp = rng.choice(len(m), size=N, p=w / w.sum())
    return rng.normal(m[comp], s[comp])


def wrapped_cauchy(n, r, rng):
    return (-np.log(r)) * np.tan(np.pi * (rng.random(n) - 0.5))      # <e^{iθ}> = r


def simulate_network(omega, cond, seed):
    f, r0 = cond
    rng = np.random.default_rng(seed)
    th0 = wrapped_cauchy(omega.size, r0, rng)
    eps, wf = (0.0, 0.0) if f is None else (EPS, f)

    def rhs(t, th):
        Z = np.exp(1j * th).mean()
        H = K_TRUE * Z + eps * np.exp(1j * wf * t)
        return omega + np.imag(np.exp(-1j * th) * H)

    sol = solve_ivp(rhs, (0, T), th0, t_eval=TS, rtol=1e-7, atol=1e-9, method="RK45")
    return np.abs(np.exp(1j * sol.y).mean(0))


def make_data():
    omega = sample_omega()
    out = dict(omega=omega, t=TS)
    for i, c in enumerate(all_conditions()):
        t0 = time.time()
        out[_key(c)] = simulate_network(omega, c, seed=1000 + i)
        print(f"[data] {_key(c)}  R(0)={out[_key(c)][0]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(DATA, **out)


def load_data():
    return dict(np.load(DATA))


# ── LMMF model in JAX ────────────────────────────────────────────────────────
def _rhs(z, t, a, w, K, eps, wf):
    H = K * jnp.sum(w * z) + eps * jnp.exp(1j * wf * t)
    return a * z + 0.5 * (H - jnp.conj(H) * z * z)


def _simulate_one(a, w, K, eps, wf, r0):
    """RK4 from z_m(0) = r0; returns R at the sample times TS."""
    def step(carry, _):
        z, t = carry
        k1 = _rhs(z, t, a, w, K, eps, wf)
        k2 = _rhs(z + 0.5 * DT * k1, t + 0.5 * DT, a, w, K, eps, wf)
        k3 = _rhs(z + 0.5 * DT * k2, t + 0.5 * DT, a, w, K, eps, wf)
        k4 = _rhs(z + DT * k3, t + DT, a, w, K, eps, wf)
        return (z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + DT), None

    def sample(carry, _):
        carry, _ = jax.lax.scan(step, carry, None, length=STRIDE)
        return carry, jnp.abs(jnp.sum(w * carry[0]))

    z0 = jnp.full(a.shape, r0 + 0j)
    _, R = jax.lax.scan(sample, (z0, 0.0), None, length=TS.size - 1)
    return jnp.concatenate([jnp.array([r0]), R])


_simulate_batch = jax.vmap(_simulate_one, in_axes=(None, None, None, 0, 0, 0))


def unpack(p, M, gauge):
    w = jax.nn.softmax(p[:M])
    Om_t = p[M:2 * M]
    if gauge:
        Om_t = Om_t - jnp.sum(w * Om_t)
    K = jnp.exp(p[3 * M])
    return w, K * Om_t, K * jnp.exp(p[2 * M:3 * M]), K


def model_R(p, M, gauge, eps, wf, r0):
    w, Om, De, K = unpack(p, M, gauge)
    return _simulate_batch(1j * Om - De, w, K, eps, wf, r0)


def conds_arrays(conds, data):
    """Forcing amplitude/frequency per trace and the OBSERVED initial coherence R(0)."""
    eps = np.array([0.0 if f is None else EPS for f, _ in conds])
    wf = np.array([0.0 if f is None else f for f, _ in conds])
    r0 = np.array([data[_key(c)][0] for c in conds])
    return jnp.asarray(eps), jnp.asarray(wf), jnp.asarray(r0)


def make_loss(M, gauge, conds, data):
    eps, wf, r0 = conds_arrays(conds, data)
    Robs = jnp.asarray(np.stack([data[_key(c)] for c in conds]))

    @jax.jit
    def loss(p, mask):
        R = model_R(p, M, gauge, eps, wf, r0)
        return jnp.sum(mask * (R - Robs) ** 2) / (jnp.sum(mask) * len(conds))

    vg = jax.jit(jax.value_and_grad(loss))
    return loss, vg


# ── fitting ──────────────────────────────────────────────────────────────────
HORIZONS = (5.0, 10.0, 20.0, 30.0)


def _bounds(M, n):
    return ([(-20, 20)] * M + [(-8, 8)] * M + [(np.log(1e-3), np.log(10.0))] * M
            + [(np.log(0.2), np.log(30.0))])


def _lbfgs(vg, p0, mask, M, maxiter=400):
    f = lambda p: tuple(np.asarray(v, float) for v in vg(jnp.asarray(p), mask))
    r = minimize(f, np.asarray(p0, float), jac=True, method="L-BFGS-B", bounds=_bounds(M, p0.size),
                 options=dict(maxiter=maxiter, ftol=1e-14, gtol=1e-10))
    return r.x, float(r.fun)


def _pack(w, Om_phys, De_phys, K):
    return np.concatenate([np.log(np.clip(w, 1e-8, None)), np.asarray(Om_phys) / K,
                           np.log(np.asarray(De_phys) / K), [np.log(K)]])


def _random_init(M, rng):
    K0 = float(np.exp(rng.uniform(np.log(1.0), np.log(10.0))))
    Om = np.sort(rng.uniform(-4, 4, M))
    De = rng.uniform(0.1, 1.0, M)
    return _pack(np.full(M, 1.0 / M), Om, De, K0)


def _duplicate_inits(p_prev, M_prev, gauge):
    """Split each component of the (M-1)-fit into two identical halves (tiny centre offset)."""
    w, Om, De, K = (np.asarray(v) for v in unpack(jnp.asarray(p_prev), M_prev, gauge))
    K = float(K)
    out = []
    for k in range(M_prev):
        d = 1e-3 * De[k]
        w2 = np.concatenate([w, [0.5 * w[k]]]); w2[k] *= 0.5
        Om2 = np.concatenate([Om, [Om[k] + d]]); Om2[k] -= d
        De2 = np.concatenate([De, [De[k]]])
        out.append(_pack(w2, Om2, De2, K))
    return out


def fit_fixed_M(M, scen, data, p_prev=None, n_random=4, seed=0):
    gauge = scen["gauge"]
    _, vg = make_loss(M, gauge, scen["train"], data)
    masks = {h: jnp.asarray((TS <= h + 1e-9).astype(float)) for h in HORIZONS}
    rng = np.random.default_rng(seed + 17 * M)
    best = (None, np.inf)
    starts = [("warm", p) for p in (_duplicate_inits(p_prev, M - 1, gauge) if p_prev is not None else [])]
    starts += [("random", _random_init(M, rng)) for _ in range(n_random)]
    for kind, p in starts:
        hs = HORIZONS[-1:] if kind == "warm" else HORIZONS          # curriculum for random starts
        for h in hs:
            p, fval = _lbfgs(vg, p, masks[h], M)
        if fval < best[1]:
            best = (p, fval)
    return best


def evaluate(p, M, gauge, conds, data):
    eps, wf, r0 = conds_arrays(conds, data)
    R = np.asarray(model_R(jnp.asarray(p), M, gauge, eps, wf, r0))
    Robs = np.stack([data[_key(c)] for c in conds])
    return R, float(np.sqrt(np.mean((R - Robs) ** 2)))


def run_scenario(name, data, M_max=8, patience=2):
    scen = SCEN[name]
    trace, p_prev, best_val, stall = [], None, np.inf, 0
    for M in range(1, M_max + 1):
        t0 = time.time()
        p, ftrain = fit_fixed_M(M, scen, data, p_prev)
        _, val_rmse = evaluate(p, M, scen["gauge"], scen["val"], data)
        w, Om, De, K = (np.asarray(v) for v in unpack(jnp.asarray(p), M, scen["gauge"]))
        trace.append(dict(M=M, p=p, train_rmse=float(np.sqrt(ftrain)), val_rmse=val_rmse,
                          w=w, Om=Om, De=De, K=float(K)))
        print(f"[{name}] M={M}  train RMSE={np.sqrt(ftrain):.4f}  val RMSE={val_rmse:.4f}  "
              f"K={float(K):.3f}  ({time.time() - t0:.0f}s)", flush=True)
        p_prev = p
        if val_rmse < best_val - 1e-6:
            best_val, stall = val_rmse, 0
        else:
            stall += 1
            if stall >= patience:
                break
    best = min(trace, key=lambda e: e["val_rmse"])
    R_test, test_rmse = evaluate(best["p"], best["M"], scen["gauge"], TEST, data)
    print(f"[{name}] selected M*={best['M']}  K={best['K']:.3f} (true {K_TRUE})  "
          f"test RMSE={test_rmse:.4f}", flush=True)
    np.save(os.path.join(_HERE, f"rfit_{name}.npy"),
            dict(trace=trace, best=best, R_test=R_test, test_rmse=test_rmse), allow_pickle=True)


# ── baselines on the test set ────────────────────────────────────────────────
def baselines(data):
    omega = data["omega"]
    out = {}
    # (i) distribution-informed LMMF: CvM fit of ρ (Fig. 1 algorithm), true K
    r = LM.fit(omega, (1e-4, 1e2), M_max=16, lambda_M=1e-5, patience=3, n_restarts=10, seed=1,
               method="slsqp", floor_c=1.0)
    m = r["model"]
    p = _pack(m.w, m.Omega, m.Delta, K_TRUE)
    R, rmse = evaluate(p, m.M, False, TEST, data)
    out["rho_fit"] = dict(M=m.M, w=m.w, Om=m.Omega, De=m.Delta, R_test=R, test_rmse=rmse)
    print(f"[baseline] ρ-fit LMMF (M={m.M}, true K): test RMSE={rmse:.4f}", flush=True)
    # (ii) continuum mean field of the TRUE ρ (quantile nodes), true K: finite-N reference
    from scipy.stats import norm
    g = np.linspace(-10, 10, 200001)
    F = sum(wk * norm.cdf((g - mk) / sk) for mk, sk, wk in zip(GMM["means"], GMM["stds"], GMM["weights"]))
    nodes = np.interp((np.arange(20000) + 0.5) / 20000, F, g)
    eps, wf, r0 = conds_arrays(TEST, data)
    R = np.asarray(_simulate_batch(1j * jnp.asarray(nodes), jnp.full(nodes.size, 1 / nodes.size),
                                   K_TRUE, eps, wf, r0))
    Robs = np.stack([data[_key(c)] for c in TEST])
    rmse = float(np.sqrt(np.mean((R - Robs) ** 2)))
    out["true_rho"] = dict(R_test=R, test_rmse=rmse)
    print(f"[baseline] continuum MF of true ρ (true K): test RMSE={rmse:.4f}", flush=True)
    np.save(os.path.join(_HERE, "rfit_baselines.npy"), out, allow_pickle=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["data", "baselines", "S1", "S2", "S3"]
    for w in which:
        if w == "data":
            make_data()
        elif w == "baselines":
            baselines(load_data())
        else:
            run_scenario(w, load_data())
