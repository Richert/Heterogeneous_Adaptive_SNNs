r"""
Fit nested LMMF model variants to the SCN coherence targets (epoch-peak pipeline, ceiling, mask)
================================================================================================

Model = scn_lmmf_sim (identical build and integrator), for a cell i in lobe L, TTX level u(t):
    θ̇_i = ω_i − δω (1 − u) + K_s(t) Im(e^{−iα_s} Z e^{−iθ_i}) + K_i Im(e^{−iα_i} Z_L e^{−iθ_i})  [+ Cauchy noise γ u]
    phase scrambling z_m → e^{−σ} z_m at TTX application (σ_ttx) / washout (σ_wash);
    TTX-induced quenched disorder γ_q (contained in the measured distribution, absent outside TTX).
Two shapes of the TTX-sensitive coupling K_s(t) (scn_lmmf_sim.coupling_profile):
    step     K_pre | K_ttx | K_wash                                       (K_min = K_ttx, K_max = K_wash)
    profile  K_pre | K_min → K_ttx (τ_ttx) | K_max → K_post (τ_wash):  K_s follows the spiking rate — a sudden drop
             at TTX application with partial recovery under TTX, a rebound at washout that decays to a (possibly
             lower) post-washout level. K_post is stored as K_wash. Parametrised as K_ttx = f_ttx K_pre,
             K_min = f_min K_ttx (so K_min < K_ttx < K_pre), K_max = K_post + f_max (K_MAX − K_post) (K_max > K_post),
             f ∈ (0, 1), τ ∈ (TAU_MIN, TAU_MAX) h (log-uniform transform); K_post vs K_pre is free.
LMMF from one Lorentzian mixture per lobe (scn_lmmf_sim.build; noise floor or not), started from the measured
state at the first hour with ≥ 90 % defined phases and run continuously to the end of the valid window.

Target (scn_targets): data R of the slice and both lobes vs the model c(t)|Z(t)| (measurement ceiling), over the
mask (valid window without ±MASK_H h around the treatment changes):
    loss = Σ_t mask_t [(c|Z| − R)² + (c_A|Z_A| − R_A)² + (c_B|Z_B| − R_B)²] / (3 Σ mask)
           + Σ_coupled stretches ((ν_model − ν_data) / SIGMA_NU)² / Σ mask
(ν = collective frequency in the pre-TTX and post-washout stretches outside the mask).

Models (MODELS; all parameters not listed fixed at 0; ONE harmonic coupling term unless +Ki):
    base         step: K_pre, K_ttx, K_wash, α_s          prof-base   profile: K_pre, f_ttx, f_min, τ_ttx, K_wash,
    reset        base + σ_ttx                                                  f_max, τ_wash, α_s
    noise        base + γ                                 prof-<mech> the profile + the same mechanisms
    quenched     base + γ_q  (γ_q < GQ_FRAC × the narrowest component width, so no width is clipped)
    reset+noise  base + σ_ttx + γ
    shift, full, <name>+Ki   (not in the defaults) base + δω | σ_ttx, σ_wash, γ, δω | + lobe-local K_i
NESTED optimisation in STAGES over all (slice, mixture kind, model family) units at once; every start of a stage
is a separate task of one process pool (scales to many cores):
  1 base           --n-base generic starts (the hand-picked STARTS, then uniform random points in the
                   transformed parameter space)
  2 single mech.   reset / noise / quenched: the base optimum with the mechanism near 0 + --n-ext generic starts
  3 reset+noise    the reset, noise and base optima (missing mechanism near 0) + --n-ext generic starts
  4 base polish    base from every extended optimum with its mechanism switched off
  5 refits         every extended model that ended above a model nested in it, from that model's optimum
Each start is a Nelder–Mead run; the best run of each search is polished once (tighter tolerances). Every model
is then at least as good as the models nested in it.
--optimizer basinhopping: each search runs scipy basinhopping from its first --bh-starts starts (the warm starts
come first; --bh-niter hops, --bh-T temperature relative to the start's loss, --bh-step step in the transformed
space, --bh-maxiter-per-dim for the local Nelder–Mead); results in model_fit/nested_bh/, to check the 'nm' optima.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_model_fit.py fit figure [--slices 1,2,3,4]
        [--models base,reset,...,prof-base,...] [--kinds floor] [--procs 8] [--dt 0.5] [--maxiter-per-dim 300]
        [--n-base 16] [--n-ext 8] [--optimizer nm|basinhopping] [--bh-niter 20] [--bh-T 0.02] [--bh-step 0.75]
        [--bh-maxiter-per-dim 150] [--bh-starts 8]
    e.g. the optimiser check:  ... fit figure --optimizer basinhopping --models prof-base,prof-reset+noise
    (time a single loss evaluation: python scn_model_fit.py time --slices 1)
Results: data_paths.MPMF/circadian_data/model_fit/nested/{scn<i>_<kind>_<model>.npy, scn_model_fit.{png,pdf},
         scn_model_fit_summary.csv}   (mixtures cached in model_fit/)
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import csv
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize, basinhopping
import data_paths as dp
import scn_common as S
import scn_lmmf_sim as SIM
import scn_targets as TG

MIX = dp.ensure(os.path.join(S.OUT, "model_fit"))
OUT = dp.ensure(os.path.join(MIX, "nested"))
K_MAX, DW_MAX, G_MAX, S_MAX = 0.3, 0.03, 0.05, 3.0
TAU_MIN, TAU_MAX = 1.0, 240.0
GQ_FRAC = 0.95
SIGMA_NU = 0.005
LAM, WIDTH_K = 1e-6, 10.0
DT_FIT = 0.5
MAXITER_PER_DIM = 300
OPT = dict(optimizer="nm", bh_niter=20, bh_T=0.02, bh_step=0.75, bh_mpd=150, bh_starts=8, seed=1)  # --optimizer
N_BASE, N_EXT = 16, 8          # generic starts per base / extended model (+ the nested warm starts); see --n-base
X_SPREAD = 2.5                 # random starts: uniform in the transformed space, |x| < X_SPREAD
STEP = ("K_pre", "K_ttx", "K_wash", "alpha_s")
PROF = ("K_pre", "f_ttx", "f_min", "tau_ttx", "K_wash", "f_max", "tau_wash", "alpha_s")
MECH = {"base": (), "reset": ("reset_ttx",), "noise": ("gamma",), "quenched": ("gamma_q",),
        "reset+noise": ("reset_ttx", "gamma"), "shift": ("dw",), "full": ("reset_ttx", "reset_wash", "gamma", "dw")}
MODELS = {}
for _m, _x in MECH.items():
    MODELS[_m] = STEP + _x
    MODELS["prof-" + _m] = PROF + _x
    MODELS[_m + "+Ki"] = STEP + ("K_i",) + _x + (("alpha_i",) if _m == "full" else ())
DEFAULT_MODELS = tuple(p + m for p in ("", "prof-") for m in ("base", "reset", "noise", "reset+noise", "quenched"))
DEFAULT_SLICES = (1, 2, 3, 4)                              # SCN5 set aside (outlier) for now
DEFAULTS = dict(K_pre=0.0, K_ttx=0.0, K_wash=0.0, K_i=0.0, alpha_s=0.0, alpha_i=0.0, dw=0.0, gamma=0.0,
                gamma_q=0.0, reset_ttx=0.0, reset_wash=0.0, f_ttx=1.0, f_min=1.0, tau_ttx=24.0, f_max=0.0,
                tau_wash=24.0)
SMALL = dict(reset_ttx=1e-3, reset_wash=1e-3, gamma=1e-4, gamma_q=1e-5, dw=0.0, K_i=1e-4)
STARTS = [dict(K_pre=0.10, K_ttx=0.006, K_wash=0.08, alpha_s=0.0, f_ttx=0.1, f_min=0.2, tau_ttx=48.0, f_max=0.3,
               tau_wash=12.0, reset_ttx=0.3, gamma=0.003, gamma_q=0.3),
          dict(K_pre=0.25, K_ttx=0.02, K_wash=0.15, alpha_s=0.2, f_ttx=0.3, f_min=0.1, tau_ttx=96.0, f_max=0.5,
               tau_wash=24.0, reset_ttx=0.7, gamma=0.01, gamma_q=0.6),
          dict(K_pre=0.06, K_ttx=0.002, K_wash=0.05, alpha_s=-0.2, f_ttx=0.05, f_min=0.5, tau_ttx=24.0, f_max=0.1,
               tau_wash=48.0, reset_ttx=0.1, gamma=0.001, gamma_q=0.1),
          dict(K_pre=0.28, K_ttx=0.05, K_wash=0.2, alpha_s=0.0, f_ttx=0.5, f_min=0.05, tau_ttx=150.0, f_max=0.7,
               tau_wash=6.0, reset_ttx=0.5, gamma=0.02, gamma_q=0.8)]
for _s in STARTS:                                           # common defaults of the remaining parameters
    for _k, _v in dict(K_i=0.01, alpha_i=0.0, dw=0.0, reset_wash=0.1).items():
        _s.setdefault(_k, _v)
KINDS = ("floor", "nofloor")
C_MODEL = {"base": "0.35", "reset": "#b2182b", "noise": "#2166ac", "shift": "#5aae61", "quenched": "#e08214",
           "reset+noise": "#762a83", "full": "#01665e"}
_GQ_MAX = [G_MAX]                                          # set per job from the mixture (γ_q < narrowest width)


def mech_of(name):
    return name.replace("prof-", "").replace("+Ki", "")


def family_of(name):
    return ("prof-" if name.startswith("prof-") else "") + ("|+Ki" if name.endswith("+Ki") else "")


# ─── parameter transforms ────────────────────────────────────────────────────────────────────────
def _bounds(k):
    if k.startswith("K_"):
        return 0.0, K_MAX
    if k.startswith("alpha"):
        return -0.5 * np.pi, 0.5 * np.pi
    if k == "dw":
        return -DW_MAX, DW_MAX
    if k == "gamma":
        return 0.0, G_MAX
    if k == "gamma_q":
        return 0.0, _GQ_MAX[0]
    if k.startswith("f_"):
        return 0.0, 1.0
    if k.startswith("tau"):
        return np.log(TAU_MIN), np.log(TAU_MAX)
    return 0.0, S_MAX


def to_x(p, keys):
    x = []
    for k in keys:
        lo, hi = _bounds(k); v = np.log(p[k]) if k.startswith("tau") else p[k]
        q = np.clip((v - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        x.append(np.log(q / (1 - q)))
    return np.array(x)


def from_x(x, keys):
    """Raw parameters (the fitted ones) and the derived couplings K_min, K_ttx (profile), K_max."""
    p = dict(DEFAULTS)
    for k, v in zip(keys, x):
        lo, hi = _bounds(k); q = lo + (hi - lo) / (1 + np.exp(-v))
        p[k] = np.exp(q) if k.startswith("tau") else q
    if "f_ttx" in keys:
        p["K_ttx"] = p["f_ttx"] * p["K_pre"]
        p["K_min"] = p["f_min"] * p["K_ttx"]
        p["K_max"] = p["K_wash"] + p["f_max"] * (K_MAX - p["K_wash"])
    else:
        p["K_min"], p["K_max"] = p["K_ttx"], p["K_wash"]
    return p


def start_params(s):
    """A generic start as raw parameters (γ_q given as a fraction of its bound)."""
    p = dict(DEFAULTS); p.update(s); p["gamma_q"] = s["gamma_q"] * _GQ_MAX[0]
    return p


# ─── data, model, loss ───────────────────────────────────────────────────────────────────────────
def mixture(i, kind, d):
    f = os.path.join(MIX, f"scn{i}_{kind}_mixture.npy")
    if os.path.exists(f):
        return np.load(f, allow_pickle=True).item()
    m = SIM.build(d, LAM, kind == "floor", WIDTH_K)
    np.save(f, m, allow_pickle=True)
    return m


def collective_frequency(Z):
    ph = np.unwrap(np.angle(Z)); t = np.arange(Z.size)
    return np.polyfit(t, ph, 1)[0]


def prepare(d):
    T = int(d["T"]); v0, v1 = S.valid_window(T); ttx, wash, mh = int(d["ttx"]), int(d["wash"]), int(d["mask_h"])
    ok = np.isfinite(d["PH"]); E = np.where(ok, np.exp(1j * np.nan_to_num(d["PH"])), 0)
    Dg = E.sum(1) / np.maximum(ok.sum(1), 1)
    t0 = TG.start_hour(d, v0)                               # first hour with ≥ 90 % defined phases
    stretches = [(max(v0, t0), ttx - mh), (wash + mh, v1)]
    nu = [collective_frequency(Dg[a:b]) for a, b in stretches]
    return dict(t0=t0, t1=v1 - 1, stretches=stretches, nu=nu)


def Kfun(p, d):
    return SIM.coupling_profile(p["K_pre"], p["K_min"], p["K_ttx"], p["tau_ttx"], p["K_max"], p["K_wash"],
                                p["tau_wash"], int(d["ttx"]), int(d["wash"]))


def predict(p, d, model, prep, dt):
    ttx, wash = int(d["ttx"]), int(d["wash"])
    uf = SIM.ttx_level(ttx, wash, np.inf, np.inf)
    t0, t1 = prep["t0"], prep["t1"]
    return SIM.simulate(model, SIM.initial_state(model, d["PH"][t0]), Kfun(p, d), uf, p["K_i"], p["alpha_s"],
                        p["alpha_i"], False, p["dw"], p["gamma"], t0, t1, p["gamma_q"],
                        resets=((ttx, p["reset_ttx"]), (wash, p["reset_wash"])), dt=dt)


def loss(x, keys, d, model, prep, dt, detail=False):
    p = from_x(x, keys)
    tm, Zg, ZA, ZB = predict(p, d, model, prep, dt)
    ti = tm.astype(int); w = d["mask"][ti]
    e = ((d["c"][ti] * np.abs(Zg) - d["R"][ti]) ** 2 + (d["cA"][ti] * np.abs(ZA) - d["RA"][ti]) ** 2
         + (d["cB"][ti] * np.abs(ZB) - d["RB"][ti]) ** 2) / 3
    e = np.where(w > 0, e, 0.0); n = w.sum()
    if not np.all(np.isfinite(e)):
        return (1e3, None) if detail else 1e3
    fterm = sum(((collective_frequency(Zg[a - prep["t0"]:b - prep["t0"]]) - nu) / SIGMA_NU) ** 2
                for (a, b), nu in zip(prep["stretches"], prep["nu"]))
    L = (e.sum() + fterm) / n
    if detail:
        rm = lambda M, R, c: np.sqrt(np.sum(np.where(w > 0, (c[ti] * np.abs(M) - R[ti]) ** 2, 0)) / n)
        return L, dict(params=p, tm=tm, Zg=Zg, ZA=ZA, ZB=ZB, rmse=rm(Zg, d["R"], d["c"]),
                       rmse_A=rm(ZA, d["RA"], d["cA"]), rmse_B=rm(ZB, d["RB"], d["cB"]), loss=L)
    return L


# ─── nested optimisation ─────────────────────────────────────────────────────────────────────────
def warm(p, add=(), drop=()):
    q = dict(p)
    for k in add:
        q[k] = SMALL.get(k, 1e-3)
    for k in drop:
        q[k] = DEFAULTS[k]
    return q


# ─── staged, start-parallel nested optimisation ──────────────────────────────────────────────────
_CTX = {}                                                  # per-process cache: (slice, kind, dt) -> context


def context(i, kind, dt):
    key = (i, kind, dt)
    if key not in _CTX:
        d = TG.load(i); model = mixture(i, kind, d)
        _CTX[key] = (d, model, prepare(d), dt, GQ_FRAC * float(model["De"].min()))
    return _CTX[key]


def generic_starts(keys, n, seed):
    """The hand-picked STARTS, then uniform samples in the transformed space (x ∈ (−X_SPREAD, X_SPREAD))."""
    X = [to_x(start_params(s), keys) for s in STARTS][:n]
    rng = np.random.default_rng(seed)
    while len(X) < n:
        X.append(rng.uniform(-X_SPREAD, X_SPREAD, len(keys)))
    return X


def _task(args):
    """One local search: (tag, x, loss). mode 'nm' (Nelder–Mead), 'bh' (basinhopping) or 'polish'."""
    tag, i, kind, name, x0, mode, mpd, dt, opt = args
    d, model, prep, dt_, gq = context(i, kind, dt)
    _GQ_MAX[0] = gq; OPT.update(opt)
    keys = MODELS[name]; ctx = (d, model, prep, dt)
    if mode == "bh":
        L0 = loss(x0, keys, *ctx)
        r = basinhopping(loss, x0, niter=OPT["bh_niter"], T=OPT["bh_T"] * L0, stepsize=OPT["bh_step"],
                         minimizer_kwargs=dict(args=(keys,) + ctx, method="Nelder-Mead",
                                               options=dict(maxiter=OPT["bh_mpd"] * len(keys), xatol=1e-3,
                                                            fatol=1e-7)),
                         seed=OPT["seed"] + 7919 * tag[2] + 31 * i)
        return tag, r.x, r.fun
    tol = (1e-4, 1e-8) if mode == "polish" else (1e-3, 1e-7)
    r = minimize(loss, x0, args=(keys,) + ctx, method="Nelder-Mead",
                 options=dict(maxiter=mpd * len(keys), xatol=tol[0], fatol=tol[1]))
    return tag, r.x, r.fun


class Stage:
    """Collects the starts of several (unit, model) searches and runs them in one pool map; the best run of each
    search is then polished once (a second map)."""
    def __init__(self, pool, mpd, dt):
        self.pool, self.mpd, self.dt, self.jobs = pool, mpd, dt, []

    def add(self, unit, name, X):
        i, kind = unit[:2]
        mode = "bh" if OPT["optimizer"] == "basinhopping" else "nm"
        if mode == "bh":                                    # basinhopping from the most promising starts only
            X = X[:OPT["bh_starts"]]
        for k, x in enumerate(X):
            self.jobs.append(((unit, name, k), i, kind, name, np.asarray(x, float), mode, self.mpd, self.dt,
                              dict(OPT)))

    def run(self, label):
        t0 = time.time(); best = {}
        for (unit, name, _), x, f in self.pool.imap_unordered(_task, self.jobs, chunksize=1):
            if (unit, name) not in best or f < best[unit, name][1]:
                best[unit, name] = (x, f)
        pol = [((unit, name, -1), unit[0], unit[1], name, x, "polish", self.mpd, self.dt, dict(OPT))
               for (unit, name), (x, f) in best.items()]
        for (unit, name, _), x, f in self.pool.imap_unordered(_task, pol, chunksize=1):
            if f < best[unit, name][1]:
                best[unit, name] = (x, f)
        print(f"[stage] {label}: {len(self.jobs)} starts + {len(pol)} polishes, {(time.time() - t0) / 60:.1f} min",
              flush=True)
        return best


def run(slices, kinds, models, procs, dt, mpd, n_base, n_ext):
    """Nested fits of all requested models; stages = base | single mechanisms | reset+noise | base re-polish |
    refits of extended models above a nested one. Every start of a stage is a separate pool task."""
    gq = {}
    for i in slices:                                       # targets and mixtures, serially (cached)
        d = TG.load(i)
        for k in kinds:
            gq[i, k] = GQ_FRAC * float(mixture(i, k, d)["De"].min())
    fams = {}
    for m in models:
        fams.setdefault(family_of(m), []).append(m)
    units = []                                             # (slice, kind, prefix, suffix, names)
    for i in slices:
        for k in kinds:
            for f, ms in fams.items():
                pre = "prof-" if f.startswith("prof-") else ""; suf = "+Ki" if f.endswith("+Ki") else ""
                units.append((i, k, pre, suf, tuple(ms)))
    name = lambda u, m: u[2] + m + u[3]
    seed = lambda u, m: (1000 * u[0] + 10 * KINDS.index(u[1]) + sum(map(ord, name(u, m)))) % (2 ** 31)

    def conv(u, p, n):                                     # raw parameter dict -> x for model n of unit u
        _GQ_MAX[0] = gq[u[0], u[1]]
        return to_x(p, MODELS[n])

    def P(u, n):
        _GQ_MAX[0] = gq[u[0], u[1]]
        return from_x(res[u, n][0], MODELS[n])

    extra = lambda n, m: [k for k in MODELS[n] if k not in MODELS[m]]
    res = {}
    t_all = time.time()
    with Pool(procs) as pool:
        st = Stage(pool, mpd, dt)                          # 1: base
        for u in units:
            b = name(u, "base")
            _GQ_MAX[0] = gq[u[0], u[1]]
            st.add(u, b, generic_starts(MODELS[b], n_base, seed(u, "base")))
        res.update(st.run("base"))
        st = Stage(pool, mpd, dt)                          # 2: single mechanisms
        singles = [(u, name(u, m)) for u in units for m in u[4] if mech_of(name(u, m)) not in ("base", "reset+noise")]
        for u, n in singles:
            b = name(u, "base")
            _GQ_MAX[0] = gq[u[0], u[1]]
            st.add(u, n, [conv(u, warm(P(u, b), add=extra(n, b)), n)]
                   + generic_starts(MODELS[n], n_ext, seed(u, mech_of(n))))
        if singles:
            res.update(st.run("single mechanisms"))
        st = Stage(pool, mpd, dt)                          # 3: reset+noise (and other composite models)
        comps = [(u, name(u, m)) for u in units for m in u[4] if mech_of(name(u, m)) == "reset+noise"]
        for u, n in comps:
            X = [conv(u, warm(P(u, name(u, "base")), add=extra(n, name(u, "base"))), n)]
            for s_ in ("reset", "noise"):
                if (u, name(u, s_)) in res:
                    X.insert(0, conv(u, warm(P(u, name(u, s_)), add=extra(n, name(u, s_))), n))
            _GQ_MAX[0] = gq[u[0], u[1]]
            st.add(u, n, X + generic_starts(MODELS[n], n_ext, seed(u, mech_of(n))))
        if comps:
            res.update(st.run("reset+noise"))
        st = Stage(pool, mpd, dt)                          # 4: base re-polished from the extended optima
        for u in units:
            b = name(u, "base")
            X = [res[u, b][0]] + [conv(u, warm(P(u, n), drop=extra(n, b)), b)
                                  for (uu, n) in res if uu == u and n != b]
            st.add(u, b, X)
        for k_, v in st.run("base re-polish").items():
            if v[1] < res[k_][1]:
                res[k_] = v
        st = Stage(pool, mpd, dt)                          # 5: refit extended models above a nested model
        todo = 0
        for (u, n) in [k_ for k_ in res if mech_of(k_[1]) != "base"]:
            nested = [name(u, "base")] + ([name(u, s_) for s_ in ("reset", "noise") if (u, name(u, s_)) in res]
                                          if mech_of(n) == "reset+noise" else [])
            m = min(nested, key=lambda q: res[u, q][1])
            if res[u, n][1] > res[u, m][1]:
                st.add(u, n, [conv(u, warm(P(u, m), add=extra(n, m)), n), res[u, n][0]]); todo += 1
        if todo:
            for k_, v in st.run(f"refits ({todo})").items():
                if v[1] < res[k_][1]:
                    res[k_] = v
    print(f"[fit] all stages: {(time.time() - t_all) / 60:.1f} min", flush=True)
    for (u, n), (x, f) in sorted(res.items(), key=lambda kv: (kv[0][0][0], kv[0][0][1], kv[0][1])):
        i, k = u[0], u[1]
        if n not in u[4] and not (mech_of(n) == "base" and name(u, "base") == n):
            continue
        d, model, prep, _, g = context(i, k, dt); _GQ_MAX[0] = g
        L, r = loss(x, MODELS[n], d, model, prep, dt, detail=True)
        r.update(slice=i, kind=k, model_name=n, keys=MODELS[n], x=x, n_free=len(MODELS[n]), model=model, dt=dt,
                 gq_max=g, optimizer=dict(OPT), n_base=n_base, n_ext=n_ext)
        np.save(os.path.join(OUT, f"scn{i}_{k}_{n}.npy"), r, allow_pickle=True)
        p = r["params"]
        shown = list(r["keys"]) + (["K_ttx", "K_min", "K_max"] if "f_ttx" in r["keys"] else [])
        print(f"[fit] SCN{i} {k:7s} {n:16s} RMSE {r['rmse']:.3f} (A {r['rmse_A']:.3f}, B {r['rmse_B']:.3f}) "
              f"loss {r['loss']:.6f} | " + ", ".join(f"{q} {p[q]:.4g}" for q in shown), flush=True)


def timing(i, dt, kind="floor"):
    d = TG.load(i); model = mixture(i, kind, d); prep = prepare(d); _GQ_MAX[0] = GQ_FRAC * float(model["De"].min())
    keys = MODELS["prof-reset+noise"]; x = to_x(start_params(STARTS[0]), keys)
    t0 = time.time()
    for _ in range(5):
        loss(x, keys, d, model, prep, dt)
    el = (time.time() - t0) / 5
    ref = loss(x, keys, d, model, prep, 0.25, detail=True)[1]
    cur = loss(x, keys, d, model, prep, dt, detail=True)[1]
    print(f"SCN{i} {kind}: M = {model['Om'].size}, one loss evaluation at dt = {dt}: {el:.3f} s; "
          f"max |ΔR| vs dt = 0.25: {np.max(np.abs(np.abs(cur['Zg']) - np.abs(ref['Zg']))):.2e}")


# ─── output ──────────────────────────────────────────────────────────────────────────────────────
def collect(slices, kinds, models):
    R = {}
    for i in slices:
        for k in kinds:
            for m in models:
                f = os.path.join(OUT, f"scn{i}_{k}_{m}.npy")
                if os.path.exists(f):
                    R[i, k, m] = np.load(f, allow_pickle=True).item()
    return R


def summary(R, slices, kinds, models):
    print("\nRMSE (slice, masked, ceiling-corrected) per slice  [loss]")
    for k in kinds:
        for pre in ("", "prof-"):
            ms = [m for m in models if m.startswith(pre) and (pre or not m.startswith("prof-"))]
            if not ms:
                continue
            print(f"  ── {k}, {'profile' if pre else 'step'}:  " + "  ".join(f"{mech_of(m):>18s}" for m in ms))
            for i in slices:
                print(f"  SCN{i}  " + "  ".join(f"{R[i, k, m]['rmse']:7.3f} [{R[i, k, m]['loss']:.5f}]"
                                              if (i, k, m) in R else f"{'—':>18s}" for m in ms))
    rows = []
    for (i, k, m), r in R.items():
        rows.append(dict(slice=i, kind=k, model=m, n_free=r["n_free"], loss=r["loss"], rmse=r["rmse"],
                         rmse_A=r["rmse_A"], rmse_B=r["rmse_B"],
                         **{p: r["params"][p] for p in list(DEFAULTS) + ["K_min", "K_max"]}))
    if rows:
        with open(os.path.join(OUT, "scn_model_fit_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def figure(R, slices, kinds, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kind = "floor" if "floor" in kinds else kinds[0]
    fig, axs = plt.subplots(5, len(slices), figsize=(3.2 * len(slices), 11.0), layout="constrained", squeeze=False)
    CL = ["#1b7837", "#762a83"]
    for j, i in enumerate(slices):
        d = TG.load(i); T = int(d["T"]); td = np.arange(T) / 24; v0, v1 = S.valid_window(T)
        vw = lambda x: np.where((np.arange(T) >= v0) & (np.arange(T) < v1), x, np.nan)
        fits = {m: R[i, kind, m] for m in models if (i, kind, m) in R}
        best = min(fits, key=lambda m: fits[m]["loss"]) if fits else None
        for row in range(5):
            a = axs[row, j]
            a.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
            if row != 2:
                for t_c in (d["ttx"], d["wash"]):
                    a.axvspan((t_c - d["mask_h"]) / 24, (t_c + d["mask_h"]) / 24, color="none", hatch="////",
                              ec="0.75", lw=0, zorder=0)
                a.set_ylim(0, 1.02)
        for row, pre in ((0, ""), (1, "prof-")):
            a = axs[row, j]
            a.plot(td, vw(d["R"]), color="k", lw=1.3, label="data")
            for m, r in fits.items():
                if (m.startswith("prof-")) != bool(pre):
                    continue
                ti = r["tm"].astype(int)
                a.plot(r["tm"] / 24, d["c"][ti] * np.abs(r["Zg"]), color=C_MODEL[mech_of(m)], lw=0.9,
                       label=f"{mech_of(m)} ({r['rmse']:.3f})")
            a.set_title(f"SCN{i}: slice R, {'profile' if pre else 'step'} K_s ({kind})", fontsize=8)
            a.legend(fontsize=5, frameon=False, loc="lower left")
        a = axs[2, j]
        tt = np.arange(T, dtype=float)
        for m, r in fits.items():
            if m.startswith("prof-") or m == "base":
                K = Kfun(r["params"], d)
                a.plot(tt / 24, [K(x) for x in tt], color=C_MODEL[mech_of(m)], lw=0.9,
                       ls="-" if m.startswith("prof-") else ":", label=("" if m.startswith("prof-") else "step ")
                       + mech_of(m))
        a.set_ylabel(r"$K_s(t)$ (rad/h)" if j == 0 else ""); a.set_title("TTX-sensitive coupling", fontsize=8)
        a.legend(fontsize=5, frameon=False, loc="upper right")
        for row, (lab, key, ck, zk) in ((3, ("lobe A", "RA", "cA", "ZA")), (4, ("lobe B", "RB", "cB", "ZB"))):
            a = axs[row, j]
            a.plot(td, vw(d[key]), color=CL[row - 3], lw=1.2, label=f"data, {lab}")
            if best:
                r = fits[best]; ti = r["tm"].astype(int)
                a.plot(r["tm"] / 24, d[ck][ti] * np.abs(r[zk]), color=C_MODEL[mech_of(best)], lw=0.9, ls="--",
                       label=f"best: {best}")
            a.set_title(f"SCN{i}: {lab}", fontsize=8); a.legend(fontsize=5.5, frameon=False, loc="lower left")
        axs[4, j].set_xlabel("time (days)")
    axs[0, 0].set_ylabel(r"$R$ vs $c(t)|Z|$"); axs[1, 0].set_ylabel(r"$R$ vs $c(t)|Z|$")
    axs[3, 0].set_ylabel("lobe A"); axs[4, 0].set_ylabel("lobe B")
    out = os.path.join(OUT, f"scn_model_fit_{kind}")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    lst = lambda s_: [v for v in s_.split(",")]
    slices = arg("--slices", list(DEFAULT_SLICES), lambda s_: [int(v) for v in s_.split(",")])
    models = arg("--models", list(DEFAULT_MODELS), lst); kinds = arg("--kinds", ["floor"], lst)
    dt = arg("--dt", DT_FIT, float); mpd = arg("--maxiter-per-dim", MAXITER_PER_DIM, int)
    OPT.update(optimizer=arg("--optimizer", "nm"), bh_niter=arg("--bh-niter", OPT["bh_niter"], int),
               bh_T=arg("--bh-T", OPT["bh_T"], float), bh_step=arg("--bh-step", OPT["bh_step"], float),
               bh_mpd=arg("--bh-maxiter-per-dim", OPT["bh_mpd"], int), bh_starts=arg("--bh-starts", 8, int),
               seed=arg("--seed", OPT["seed"], int))
    if OPT["optimizer"] == "basinhopping":                # separate results, for the comparison with 'nm'
        OUT = dp.ensure(os.path.join(MIX, "nested_bh"))
    if "--out-tag" in sys.argv:                           # e.g. separate floor / no-floor runs on the server
        OUT = dp.ensure(os.path.join(OUT, arg("--out-tag", "")))
    if "time" in sys.argv:
        for i in slices:
            for k in kinds:
                timing(i, dt, k)
    if "fit" in sys.argv:
        run(slices, kinds, models, arg("--procs", os.cpu_count(), int), dt, mpd, arg("--n-base", N_BASE, int),
            arg("--n-ext", N_EXT, int))
    if "figure" in sys.argv or "summary" in sys.argv:
        R = collect(slices, kinds, models)
        summary(R, slices, kinds, models)
        if "figure" in sys.argv:
            figure(R, slices, kinds, models)
