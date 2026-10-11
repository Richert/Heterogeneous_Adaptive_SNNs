r"""
Unimodal symmetric distributions in the (K, α) plane: LMMF loss variants vs. network
=====================================================================================

Kuramoto–Sakaguchi network  dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i − α) = ω_i + Im(e^{−iθ_i} H),
H = K e^{−iα} Z;  LMMF  dz_m/dt = (iω̄_m − Δ_m) z_m + ½ (H − H̄ z_m²),  H = K e^{−iα} Σ_l w_l z_l.

Distributions (unit scale, one network sample of N = 5000 each, fixed seed):
  gauss    N(0, 1)                    g4       Skardal g_4 (Δ = 1)
  uniform  U(−1, 1)                   dgauss   τ N(0,1) + (1−τ) N(0, δ²), τ = 0.6, δ = 0.1
                                               (Omel'chenko & Wolfrum, Physica D 263 (2013))
Loss variants (same M, from the CvM fit):  a = CvM (noise-floor rule, λ = 1e-5),
  b = CvM + density term β = 1,  c = CvM + density term β = 10   (unimodal_density_test.fit_density).

Stages (resumable; results in data_paths.MPMF/unimodal_kalpha):
  fits     fit variants a–c for every distribution                                   (~1 min)
  loci     LMMF incoherence boundary (eigenvalues at z = 0, fine α grid) + exact boundary of
           the true ρ (e^{iα}/K = J(Ω), Omel'chenko & Wolfrum)                         (~1–2 min)
  folds    PyCoBi two-parameter fold curves of the locked states (sequential)      (~2–5 min/fit)
  sims     quasi-static K-ramps of network + LMMF (all variants) for every α row: forward from an
           incoherent start (R0 = 0.05), backward from a coherent start (R0 = 0.9), holding time HOLD per
           step, state carried. Classification per (K, α): both ramps async -> monostable async, both
           sync -> monostable sync, disagree -> multistable (oscillatory flagged). One job per
           (distribution, direction, α chunk), all α rows of a chunk vectorised; RK4, dt = 0.05.
  figure   unimodal_kalpha_figure.py

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_kalpha.py fits|loci|folds
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_kalpha.py sims [--procs N] [--dist NAME]
                                                                [--chunks C] [--hold H]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_kalpha.py time     # timing probe
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_kalpha.py lmmf_only [--procs N]
        # recompute only the LMMF traces of existing ramp files (e.g. after changing KICK)
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import subprocess
import time
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import data_paths as dp
import lorentzian_mixture as LM
import skardal_benchmark_simulate as SK
from unimodal_density_test import fit_density

OUT = dp.ensure(dp.mpmf("unimodal_kalpha"))
N, SEED, LAM = 5000, 7, 1e-5
TAU, DELTA_DG = 0.6, 0.1
VARIANTS = {"a": None, "b": 1.0, "c": 10.0}          # density-term weight β (None = CvM only)
K_GRID = np.round(np.linspace(0.1, 4.0, 40), 3)
A_GRID = np.round(np.arange(0.0, 1.4001, 0.05), 3)
DT, DTS = 0.05, 0.5                                  # RK4 step, sampling step
R_SYNC, R_OSC = 0.1, 0.03                            # sync if mean R > R_SYNC; oscillatory if std R > R_OSC
AUTO_DIR = "~/PycharmProjects/auto-07p"


# ════════════════════════════════════════════════════════════════════════════
#  distributions
# ════════════════════════════════════════════════════════════════════════════
N_RATIONAL = 8                                       # exponent of the rational density in the figure ("g8")


def _gn(n):
    """Skardal's rational density g_n (Δ = 1) and sampler; overflow of ω^{2n} far outside [−1, 1] is
    benign (g_n -> 0 there)."""
    def pdf(w):
        with np.errstate(over="ignore"):
            return SK.gn_density(np.atleast_1d(np.asarray(w, float)), n, 1.0)

    def smp(k, r):
        with np.errstate(over="ignore"):
            return SK.sample_gn(n, 1.0, k, r)
    return dict(pdf=pdf, sample=smp)


def dists():
    g4 = lambda w: SK.gn_density(np.atleast_1d(np.asarray(w, float)), 4, 1.0)
    dg_pdf = lambda w: TAU * norm.pdf(w) + (1 - TAU) * norm.pdf(w, 0, DELTA_DG)
    return {
        "gauss": dict(pdf=lambda w: norm.pdf(w), sample=lambda n, r: r.normal(0, 1, n)),
        "g4": dict(pdf=g4, sample=lambda n, r: SK.sample_gn(4, 1.0, n, r)),
        f"g{N_RATIONAL}": _gn(N_RATIONAL),
        "uniform": dict(pdf=lambda w: np.where(np.abs(w) <= 1, 0.5, 0.0), sample=lambda n, r: r.uniform(-1, 1, n)),
        "dgauss": dict(pdf=dg_pdf, sample=lambda n, r: np.where(r.random(n) < TAU, r.normal(0, 1, n),
                                                                 r.normal(0, DELTA_DG, n))),
    }


def sample(name):
    return np.asarray(dists()[name]["sample"](N, np.random.default_rng(SEED)), float)


def _f(name, *parts):
    return os.path.join(OUT, f"{name}_" + "_".join(str(p) for p in parts))


# ════════════════════════════════════════════════════════════════════════════
#  fits
# ════════════════════════════════════════════════════════════════════════════
def run_fits():
    for name in dists():
        x = sample(name)
        r = LM.fit(x, (1e-4, 1e2), M_max=16, lambda_M=LAM, patience=3, n_restarts=10, seed=1,
                   method="slsqp", floor_c=1.0)
        m = r["model"]
        for v, beta in VARIANTS.items():
            w, Om, De = (m.w, m.Omega, m.Delta) if beta is None else fit_density(x, m.w, m.Omega, m.Delta, beta)
            np.save(_f(name, "fit", v) + ".npy", dict(w=w, Om=Om, De=De, M=w.size, beta=beta), allow_pickle=True)
            print(f"[fits] {name} {v}: M={w.size}", flush=True)


def load_fit(name, v):
    return np.load(_f(name, "fit", v) + ".npy", allow_pickle=True).item()


# ════════════════════════════════════════════════════════════════════════════
#  incoherence boundaries
# ════════════════════════════════════════════════════════════════════════════
def lmmf_Kc(f, alpha, lo=1e-3, hi=40.0):
    """Incoherence threshold of the LMMF at phase lag α: first K with an unstable eigenvalue of
    dz/dt = (iω̄ − Δ) z + (K/2) e^{−iα} Σ w z  (z = 0). Returns (K_c, critical frequency)."""
    A0 = np.diag(1j * f["Om"] - f["De"]); P = 0.5 * np.exp(-1j * alpha) * np.outer(np.ones(f["M"]), f["w"])
    g = lambda K: np.max(np.linalg.eigvals(A0 + K * P).real)
    if g(hi) < 0:
        return np.nan, np.nan
    for _ in range(70):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    ev = np.linalg.eigvals(A0 + hi * P)
    return hi, float(ev[np.argmax(ev.real)].imag)


def exact_boundary(name, n_om=400):
    """Exact incoherence boundary of the true ρ: e^{iα}/K = J(Ω) = (π/2) ρ(Ω) + (i/2) PV∫ ρ(ω)/(ω−Ω) dω,
    parameterised by the critical frequency Ω: K = 1/|J|, α = arg J (Omel'chenko & Wolfrum 2013)."""
    pdf = lambda w: float(np.asarray(dists()[name]["pdf"](w)).ravel()[0])
    out = []
    # Ω ≤ 0 gives α ≥ 0 with this sign convention (PV∫ρ/(ω−Ω) > 0 for Ω < 0 and unimodal ρ);
    # for symmetric ρ the Ω ≥ 0 branch is its mirror image (K equal, α → −α).
    for Om in np.linspace(-3.0, 0.0, n_om):
        pv = quad(pdf, -30, 30, weight="cauchy", wvar=Om, limit=400, points=None)[0]
        J = 0.5 * np.pi * pdf(Om) + 0.5j * pv
        if abs(J) > 0:
            out.append((1 / abs(J), np.angle(J), Om))
    return np.array(out)                                  # columns: K, α, Ω


A_FINE = np.linspace(0.0, 1.45, 146)
K_FINE = np.linspace(0.02, 4.0, 200)


def lmmf_stability_map(f):
    """Largest real part of the linearisation at z = 0 on the (A_FINE × K_FINE) grid. Its zero contour
    is the FULL incoherence boundary, including restabilisation (several K crossings per α) — unlike
    lmmf_Kc, which only returns the first crossing."""
    A0 = np.diag(1j * f["Om"] - f["De"]); P = 0.5 * np.outer(np.ones(f["M"]), f["w"])
    G = np.empty((A_FINE.size, K_FINE.size))
    for i, a in enumerate(A_FINE):
        Pa = np.exp(-1j * a) * P
        for j, K in enumerate(K_FINE):
            G[i, j] = np.max(np.linalg.eigvals(A0 + K * Pa).real)
    return G


def run_loci(keys=None, names=None):
    for name in (names or dists()):
        ex = exact_boundary(name)
        np.save(_f(name, "boundary_exact") + ".npy", ex)
        ks = keys or list(VARIANTS) + [k.split("_fit_")[1][:-4] for k in os.listdir(OUT)
                                       if k.startswith(f"{name}_fit_M")]
        for v in ks:
            if not os.path.exists(_f(name, "fit", v) + ".npy"):
                continue
            f = load_fit(name, v)
            kc = np.array([lmmf_Kc(f, a) for a in A_FINE])
            np.save(_f(name, "boundary", v) + ".npy",
                    dict(alpha=A_FINE, K=kc[:, 0], freq=kc[:, 1], K_grid=K_FINE, G=lmmf_stability_map(f)),
                    allow_pickle=True)
        print(f"[loci] {name}: exact boundary {len(ex)} pts, LMMF boundaries for {ks}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  PyCoBi fold curves of the locked states (co-rotating frame, with phase lag)
# ════════════════════════════════════════════════════════════════════════════
def _corot_equations(f):
    """Co-rotating Cartesian LMMF with phase lag (pin y_0 := 0; Ω = Im F_0 / x_0), dim 2M−1.
    H = K e^{−iα} Σ w z  ->  Re H = K(cosα X + sinα Y),  Im H = K(cosα Y − sinα X)."""
    M = f["M"]; w = [float(v) for v in f["w"]]; om = [float(v) for v in f["Om"]]; dl = [float(v) for v in f["De"]]
    xj = lambda j: f"x_{j}"
    yj = lambda j: "0" if j == 0 else f"y_{j}"
    X = "(" + " + ".join(f"{w[j]}*{xj(j)}" for j in range(M)) + ")"
    Y = "(" + (" + ".join(f"{w[j]}*{yj(j)}" for j in range(1, M)) if M > 1 else "0") + ")"
    rH = f"K*(cos(alpha)*{X} + sin(alpha)*{Y})"
    iH = f"K*(cos(alpha)*{Y} - sin(alpha)*{X})"

    def ReF(i):
        xi, yi = xj(i), yj(i)
        rezz = f"(x_0^2)*({rH})" if i == 0 else f"(({xi})^2-({yi})^2)*({rH}) + 2*{xi}*{yi}*({iH})"
        bias = f"-{dl[i]}*{xi}" if i == 0 else f"-{dl[i]}*{xi} - ({om[i]})*{yi}"
        return f"({bias} + 0.5*(({rH}) - ({rezz})))"

    def ImF(i):
        xi, yi = xj(i), yj(i)
        if i == 0:
            imzz, bias = f"-(x_0^2)*({iH})", f"({om[0]})*x_0"
        else:
            imzz = f"2*{xi}*{yi}*({rH}) - (({xi})^2-({yi})^2)*({iH})"
            bias = f"-{dl[i]}*{yi} + ({om[i]})*{xi}"
        return f"({bias} + 0.5*(({iH}) - ({imzz})))"

    Omega = f"(({ImF(0)})/x_0)"
    eqs = [f"d/dt * x_0 = {ReF(0)}"]
    for i in range(1, M):
        eqs += [f"d/dt * x_{i} = {ReF(i)} + {Omega}*y_{i}", f"d/dt * y_{i} = {ImF(i)} - {Omega}*x_{i}"]
    return eqs


# α values at which the K-continuation looks for fold points, tried IN ORDER; the next one is only used
# if the previous continuation found no fold to trace
FOLD_SEEDS_A = (0.3, 0.8, 1.1)


def _dedupe(curves, tol=1e-3):
    """Drop fold curves that retrace an already found one (Auto lists LPs once per direction)."""
    out = []
    for c in curves:
        if not any(abs(c[:, 0].min() - o[:, 0].min()) < tol and abs(c[:, 0].max() - o[:, 0].max()) < tol and
                   abs(c[:, 1].min() - o[:, 1].min()) < tol and abs(c[:, 1].max() - o[:, 1].max()) < tol
                   for o in out):
            out.append(c)
    return out


def fold_curves(name, v, seeds=FOLD_SEEDS_A):
    """Fold curves from the first seed α whose K-continuation finds any fold (single process; the
    CLI path run_fold_one uses one Auto session per seed instead)."""
    for a0 in seeds:
        curves = _fold_curves_from(name, v, K0=4.0, a0=a0)
        if curves:
            return _dedupe(curves)
    return []


def _fold_curves_from(name, v, K0=4.0, a0=0.3):
    """Settle the locked state at (K0, α0), continue it in K (LP/HB), then trace every LP in (K, α)."""
    from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
    from pycobi import ODESystem
    f = load_fit(name, v); M = f["M"]
    var = {"x_0": "output(0.8)"}
    for i in range(1, M):
        var[f"x_{i}"] = "variable(0.8)"; var[f"y_{i}"] = "variable(0.0)"
    var["K"] = K0; var["alpha"] = a0
    op = OperatorTemplate(name="lk_op", equations=_corot_equations(f), variables=var)
    circ = CircuitTemplate(name=f"kmo_lag_{int(round(100 * a0))}",
                           nodes={"p": NodeTemplate(name="lk_node", operators=[op])})
    ode = ODESystem.from_template(circ, auto_dir=AUTO_DIR, init_cont=False, analytical_jacobian=True,
                                  auto_constants=("ivp", "eq"))
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000, EPSL=1e-8, EPSU=1e-8,
            EPSS=1e-6, UZR={14: 2000.0}, STOP={"UZ1"})
    eq, _ = ode.run(origin="time", starting_point="UZ1", name="eqK", c="eq", ICP="K", bidirectional=True,
                    RL0=0.05, RL1=6.0, IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=20, DSMIN=1e-9,
                    DSMAX=5e-3, EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, STOP={"BP1"}, get_stability=True)
    bif = np.asarray(eq[[c for c in eq.columns if (c[0] if isinstance(c, tuple) else c) == "bifurcation"][0]]).astype(str)
    curves = []                                       # (bidirectional runs may list an LP twice)
    for k in range(1, int(np.sum(bif == "LP")) + 1):
        try:
            nm = f"lp2d_{k}"
            ode.run(origin="eqK", starting_point=f"LP{k}", name=nm, c="eq", ICP=["K", "alpha"],
                    bidirectional=True, RL0=0.05, RL1=6.0, UZSTOP={"alpha": [0.0, 1.45]}, IPS=1, ISW=2,
                    ILP=0, ISP=0, NMX=8000, NPR=20, DS=1e-3, DSMIN=1e-10, DSMAX=2e-2, EPSL=1e-7, EPSU=1e-7,
                    EPSS=1e-6, get_stability=False)
            s = ode.get_summary(nm)
            col = lambda n: np.asarray(s[[c for c in s.columns if (c[0] if isinstance(c, tuple) else c) == n][0]], float)
            curves.append(np.column_stack([col("K"), col("alpha")]))
        except Exception as e:
            print(f"   {name} {v} LP{k}: 2-D continuation failed ({type(e).__name__})", flush=True)
    ode.close_session(clear_files=True); clear(circ)
    return curves


def run_folds(variants=None, only=None, redo=False):
    """Sequential (one Auto session at a time), one subprocess per fit (and per seed α)."""
    for name in ([only] if only else dists()):
        for v in (variants or VARIANTS):
            out = _f(name, "folds", v) + ".npy"
            if os.path.exists(out) and not redo:
                continue
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "fold_one", name, v],
                               capture_output=True, text=True)
            print(f"[folds] {name} {v}: " + (r.stdout.strip().splitlines()[-1] if r.returncode == 0 and
                                              r.stdout.strip() else f"FAILED\n{r.stderr[-600:]}"), flush=True)


def run_fold_one(name, v, a0=None):
    """One Auto session per process: with a0, trace the folds from that seed into a temporary file;
    without a0, spawn one subprocess per seed and merge (deduplicated) into <name>_folds_<v>.npy."""
    os.chdir(dp.local_work("unimodal_kalpha_auto"))       # executables must not live on the share
    if a0 is not None:
        curves = _fold_curves_from(name, v, a0=a0)
        np.save(_f(name, "folds", v, f"seed{a0:g}") + ".npy", np.array(curves, dtype=object), allow_pickle=True)
        print(f"{len(curves)} raw fold curve(s) from α0={a0:g}")
        return
    curves, used = [], None
    for s0 in FOLD_SEEDS_A:                          # next seed only if the previous found no fold
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "fold_one", name, v, str(s0)],
                           capture_output=True, text=True)
        tmp = _f(name, "folds", v, f"seed{s0:g}") + ".npy"
        if os.path.exists(tmp):
            curves = list(np.load(tmp, allow_pickle=True)); os.remove(tmp)
        else:
            print(f"   seed α0={s0:g} failed: {r.stderr[-300:]}")
        if curves:
            used = s0
            break
    curves = _dedupe(curves)
    np.save(_f(name, "folds", v) + ".npy", np.array(curves, dtype=object), allow_pickle=True)
    print(f"{len(curves)} fold curve(s) (deduplicated; seed α0={used})")


# ════════════════════════════════════════════════════════════════════════════
#  simulations on the (K, α) grid
# ════════════════════════════════════════════════════════════════════════════
def _rk4(rhs, y, n_steps, stride, obs):
    """Fixed-step RK4; records only the observable obs(y) every `stride` steps (memory-light)."""
    out = [obs(y)]
    for s in range(1, n_steps + 1):
        k1 = rhs(y); k2 = rhs(y + 0.5 * DT * k1); k3 = rhs(y + 0.5 * DT * k2); k4 = rhs(y + DT * k3)
        y = y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s % stride == 0:
            out.append(obs(y))
    return np.array(out).T


HOLD = 100.0                                         # holding time per ramp step
# finite-size seed: a network of N oscillators never has R = 0 but fluctuates around N^{-1/2}. The
# deterministic LMMF instead decays to the exact incoherent fixed point z = 0 below threshold and then
# needs ~ln(1/|z|)/growth-rate to leave it above threshold, which on a forward ramp mimics spurious
# hysteresis. Kick every LMMF state by KICK at the start of each ramp step (negligible when synchronized).
KICK = 1.0 / np.sqrt(N)
DIRECTIONS = {"fwd": 0.05, "bwd": 0.9}               # ramp direction -> R0 of the wrapped-Cauchy start


def _net_rhs(om, Kc):
    """θ has shape (nα, N); Kc = K e^{−iα} with shape (nα, 1)."""
    return lambda th: om[None, :] + np.imag(np.exp(-1j * th) * (Kc * np.exp(1j * th).mean(1, keepdims=True)))


def _lmmf_rhs(f, Kc):
    a = 1j * f["Om"] - f["De"]; w = f["w"]

    def rhs(z):                                      # z has shape (nα, M)
        H = Kc * (z @ w)[:, None]
        return a[None, :] * z + 0.5 * (H - np.conj(H) * z * z)
    return rhs


def ramp(name, direction, ia_idx, hold=HOLD):
    """Quasi-static ramp over K_GRID (fwd: increasing from an incoherent start, bwd: decreasing from a
    coherent start), carrying the state; all α rows in ia_idx integrated together.
    Returns R_net (nα, nK, ns) and R_lmmf (nVar, nα, nK, ns) per holding window, K in ascending order."""
    alphas = A_GRID[ia_idx][:, None]
    om = sample(name)
    R0 = DIRECTIONS[direction]
    th = (-np.log(R0)) * np.tan(np.pi * (np.random.default_rng(100).random(N) - 0.5))
    R0_obs = float(np.abs(np.exp(1j * th).mean()))
    th = np.broadcast_to(th, (alphas.size, N)).copy()
    fits = [load_fit(name, v) for v in VARIANTS]
    zs = [np.full((alphas.size, f["M"]), R0_obs + 0j) for f in fits]
    order = range(K_GRID.size) if direction == "fwd" else range(K_GRID.size - 1, -1, -1)
    n_steps, stride = int(round(hold / DT)), int(round(DTS / DT))
    ns = n_steps // stride + 1
    Rn = np.zeros((alphas.size, K_GRID.size, ns)); Rl = np.zeros((len(fits), alphas.size, K_GRID.size, ns))
    for ik in order:
        Kc = K_GRID[ik] * np.exp(-1j * alphas)
        Rn[:, ik] = _rk4_carry(_net_rhs(om, Kc), th, n_steps, stride,
                               lambda y: np.abs(np.exp(1j * y).mean(1)))
        th = _LAST[0]
        Rl[:, :, ik] = _lmmf_step(fits, zs, Kc, n_steps, stride)
    return Rn, Rl, R0_obs


def _lmmf_step(fits, zs, Kc, n_steps, stride, kick=KICK):
    """One ramp step for all LMMF fits (states in zs are updated in place), with the finite-size seed."""
    out = []
    for j, f in enumerate(fits):
        z = zs[j] + kick
        # keep |z| <= 1 (a local order parameter cannot exceed 1): zero-width nodes (the true-ρ continuum)
        # sit exactly on |z| = 1 when locked, and the kick would push them outside the unit disc, where the
        # dynamics are unstable. Acts only when |z| + kick > 1.
        r = np.abs(z)
        zs[j] = np.where(r > 1.0, z / np.maximum(r, 1e-300), z)
        out.append(_rk4_carry(_lmmf_rhs(f, Kc), zs[j], n_steps, stride, lambda y: np.abs(y @ f["w"])))
        zs[j] = _LAST[0]
    return np.array(out)


def lmmf_ramp_only(path, kick=KICK):
    """Recompute ONLY the LMMF traces of an existing ramp file (network traces are reused)."""
    d = dict(np.load(path)); name = os.path.basename(path).split("_ramp_")[0]
    direction = "fwd" if "_ramp_fwd_" in path else "bwd"
    alphas = A_GRID[d["ia"]][:, None]; hold = float(d["hold"])
    fits = [load_fit(name, v) for v in VARIANTS]
    zs = [np.full((alphas.size, f["M"]), float(d["R0"]) + 0j) for f in fits]
    n_steps, stride = int(round(hold / DT)), int(round(DTS / DT))
    Rl = np.zeros_like(d["R_lmmf"], dtype=float)
    order = range(K_GRID.size) if direction == "fwd" else range(K_GRID.size - 1, -1, -1)
    for ik in order:
        Rl[:, :, ik] = _lmmf_step(fits, zs, K_GRID[ik] * np.exp(-1j * alphas), n_steps, stride, kick)
    if "R_lmmf_nokick" not in d:                      # keep the original (unseeded) traces once
        d["R_lmmf_nokick"] = d["R_lmmf"]
    d["R_lmmf"] = Rl.astype(np.float32); d["kick"] = kick
    np.savez_compressed(path, **d)
    return path


def run_lmmf_only(procs):
    from multiprocessing import Pool
    files = sorted(os.path.join(OUT, f) for f in os.listdir(OUT) if "_ramp_" in f and f.endswith(".npz"))
    t0 = time.time()
    with Pool(procs) as p:
        for i, out in enumerate(p.imap_unordered(lmmf_ramp_only, files, chunksize=1), 1):
            if i % 8 == 0 or i == len(files):
                print(f"[lmmf] {i}/{len(files)} files ({(time.time() - t0) / 60:.1f} min)", flush=True)


_LAST = [None]


def _rk4_carry(rhs, y, n_steps, stride, obs):
    """_rk4 that also leaves the final state in _LAST[0] (to carry it to the next ramp step)."""
    out = [obs(y)]
    for s_ in range(1, n_steps + 1):
        k1 = rhs(y); k2 = rhs(y + 0.5 * DT * k1); k3 = rhs(y + 0.5 * DT * k2); k4 = rhs(y + DT * k3)
        y = y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s_ % stride == 0:
            out.append(obs(y))
    _LAST[0] = y
    return np.array(out).T


def _chunks(n_chunks):
    return [np.array(c) for c in np.array_split(np.arange(A_GRID.size), n_chunks)]


def _ramp_job(args):
    name, direction, ci, idx, hold = args
    tag = "" if hold == HOLD else f"_hold{hold:g}"
    out = _f(name, "ramp", direction, f"c{ci:02d}") + tag + ".npz"
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    Rn, Rl, R0 = ramp(name, direction, idx, hold)
    np.savez_compressed(out, R_net=Rn.astype(np.float32), R_lmmf=Rl.astype(np.float32), ia=idx,
                        alpha=A_GRID[idx], K=K_GRID, R0=R0, hold=hold, variants=np.array(list(VARIANTS)))
    return out, time.time() - t0


def run_sims(procs, only=None, n_chunks=1, hold=HOLD):
    from multiprocessing import Pool
    names = [only] if only else list(dists())
    jobs = [(n, d, ci, idx, hold) for n in names for d in DIRECTIONS for ci, idx in enumerate(_chunks(n_chunks))]
    t0 = time.time()
    with Pool(procs) as p:
        for i, (out, dt) in enumerate(p.imap_unordered(_ramp_job, jobs, chunksize=1), 1):
            el = time.time() - t0
            print(f"[ramps] {i}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; {el / 60:.1f} min elapsed, "
                  f"~{el / i * (len(jobs) - i) / 60:.1f} min left)", flush=True)


def classify(R_win):
    """R over a holding window (..., ns) -> 0 async, 1 sync (stationary), 2 oscillatory; uses the
    second half of the window."""
    tail = R_win[..., R_win.shape[-1] // 2:]
    m, sd = tail.mean(-1), tail.std(-1)
    return np.where(sd > R_OSC, 2, np.where(m > R_SYNC, 1, 0))


def run_time_probe():
    """Time ONE ramp step (all 29 α rows, hold = HOLD) for network + one LMMF fit; a full ramp job is
    len(K_GRID) = 40 such steps."""
    om = sample("gauss"); th = np.zeros((A_GRID.size, N)); Kc = 2.0 * np.exp(-1j * A_GRID[:, None])
    n_steps, stride = int(round(HOLD / DT)), int(round(DTS / DT))
    t0 = time.time(); _rk4_carry(_net_rhs(om, Kc), th, n_steps, stride, lambda y: np.abs(np.exp(1j * y).mean(1)))
    tn = time.time() - t0
    f = dict(Om=np.linspace(-1, 1, 8), De=np.full(8, 0.3), w=np.full(8, 1 / 8), M=8)
    t0 = time.time(); _rk4_carry(_lmmf_rhs(f, Kc), np.full((A_GRID.size, 8), 0.5 + 0j), n_steps, stride,
                                 lambda y: np.abs(y @ f["w"]))
    tl = time.time() - t0
    job = len(K_GRID) * (tn + len(VARIANTS) * tl)
    print(f"[time] one ramp step: network {tn:.1f}s, LMMF {tl:.2f}s  ->  one ramp job ≈ {job / 60:.1f} min; "
          f"8 jobs (4 dists × 2 directions) on 8 cores ≈ {job / 60:.0f} min")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mode = sys.argv[1]
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    if mode == "lmmf_only":
        run_lmmf_only(arg("--procs", os.cpu_count(), int))
    elif mode == "sims":
        run_sims(arg("--procs", os.cpu_count(), int), arg("--dist", None), arg("--chunks", 1, int),
                 arg("--hold", HOLD, float))
    elif mode == "folds":
        run_folds(arg("--variants", None, lambda s_: s_.split(",")), arg("--dist", None),
                  "--redo" in sys.argv)
    elif mode == "loci":
        run_loci(names=arg("--dist", None, lambda s_: s_.split(",")))
    elif mode == "fold_one":
        run_fold_one(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        {"fits": run_fits, "time": run_time_probe}[mode]()
