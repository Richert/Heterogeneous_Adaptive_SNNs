r"""
LMMF accuracy close to bifurcation points — Fig. 1 example (bimodal Gaussian mixture)
=====================================================================================

Question: how well do LMMF fits of different complexity M locate the bifurcations of the true system,
and how does the spectral mismatch between LMMF and network dynamics depend on the distance to them?

Bifurcations in K for ρ = ½N(−2, 0.6²) + ½N(2, 0.6²):
  * K_H  : the incoherent state z = 0 loses stability (Hopf; oscillation frequency ν* ≈ ±1.9).
           Exact for ρ from the dispersion relation 1 = (K/2) Φ_ρ(iν):  K_H = 2/(π ρ(ν*)),
           with ν* the zero of the Hilbert transform of ρ near the cluster peaks.
  * K_SN : fold at which the fully locked (partially synchronized, both clusters) state appears.
           Exact for ρ from the Kuramoto self-consistency: K(u) = 1/G(u), r = u G(u),
           G(u) = ∫ cos²θ ρ(u sinθ) dθ;  K_SN = min_u K(u).
For each LMMF fit both are obtained by PyCoBi continuation in K:
  * incoherent branch z = 0 in lab-frame Cartesian coordinates (2M dims) -> first HB = K_H(M),
    cross-checked against the eigenvalues of the linearisation at z = 0;
  * locked branch in co-rotating Cartesian coordinates (2M−1 dims, kmo_heterogeneity_bifurcation
    equations with h = 1) -> LP/HB where the stable locked state ends = K_SN(M).

Fits (revised algorithm: warm start + noise-floor cap, absolute λ grid), two data sources:
  * "sample"     : the N = 5000 frequency sample of Fig. 1 (seed 1),
  * "population" : N = 5000 quantile nodes of ρ (no sampling noise, same floor/λ settings).
Dynamics: network (Fig. 1 sample, N = 5000, Fig. 1 wrapped-Cauchy initial condition), continuum
mean field of the true ρ, and every LMMF fit, simulated on a K grid; spectral RMSE of R(t) on [0, T].

Modes (pycobi env; PyCoBi runs are SEQUENTIAL, one ODESystem per process):
    python kmo_bifurcation_accuracy.py fits
    python kmo_bifurcation_accuracy.py truth
    python kmo_bifurcation_accuracy.py auto_all        # spawns `auto <idx> <incoh|sync>` sequentially
    python kmo_bifurcation_accuracy.py sims
"""
# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared"), os.path.join(_HERE, "continuity")]
import data_paths as dp
# ---------------------------------------------------------------------------
import subprocess
import time
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

import lorentzian_mixture as LM
from kmo_lorentzian_fit_sweep import CONFIG as FIG1, sample_gaussian_mixture

OUT = dp.ensure(dp.mpmf("kmo_bif_accuracy"))
FITS_NPZ = os.path.join(OUT, "fits.npz")
TRUTH_NPZ = os.path.join(OUT, "truth.npz")
SIMS_NPZ = os.path.join(OUT, "sims.npz")
AUTO_DIR = "~/PycharmProjects/auto-07p"

CFG = dict(
    lambdas=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    M_max=16, floor_c=1.0, patience=3, n_restarts=10, delta_bounds=(1e-4, 1e2), method="slsqp",
    K_grid=np.round(np.arange(0.5, 5.01, 0.1), 2),
    T=100.0, dts=0.1,
    K_lo=0.2, K_hi=8.0,          # continuation range in K
    K_settle_sync=7.0,           # locked state settled here before continuing down in K
)
MU, SD, CW = (np.asarray(FIG1[k], float) for k in ("gmm_means", "gmm_stds", "gmm_weights"))


def rho_true(w):
    w = np.atleast_1d(w)[:, None]
    return (CW * norm.pdf((w - MU) / SD) / SD).sum(1)


def fig1_sample():
    """Exactly the Fig. 1 network: ω from seed 1, then wrapped-Cauchy θ(0) from the same stream."""
    rng = np.random.default_rng(FIG1["seed"])
    om = sample_gaussian_mixture(FIG1["gmm_means"], FIG1["gmm_stds"], FIG1["gmm_weights"], FIG1["N"], rng)
    gamma = 0.5 * FIG1["sigma0"] ** 2
    th0 = gamma * np.tan(np.pi * (rng.random(FIG1["N"]) - 0.5))
    return om, th0


def quantile_nodes(n):
    g = np.linspace(-12, 12, 400001)
    F = (CW * norm.cdf((g[:, None] - MU) / SD)).sum(1)
    return np.interp((np.arange(n) + 0.5) / n, F, g)


# ════════════════════════════════════════════════════════════════════════════
#  fits
# ════════════════════════════════════════════════════════════════════════════
def make_fits(cfg=CFG):
    om, _ = fig1_sample()
    sources = {"sample": om, "population": quantile_nodes(FIG1["N"])}
    rows, seen = [], set()
    for src, x in sources.items():
        for lam in cfg["lambdas"]:
            r = LM.fit(x, cfg["delta_bounds"], M_max=cfg["M_max"], lambda_M=lam, patience=cfg["patience"],
                       n_restarts=cfg["n_restarts"], seed=1, method=cfg["method"], floor_c=cfg["floor_c"])
            m = r["model"]
            key = (src, m.M, tuple(np.round(m.Omega, 6)))
            dup = key in seen
            seen.add(key)
            rows.append(dict(source=src, lam=lam, M=m.M, w=m.w, Om=m.Omega, De=m.Delta,
                             data_loss=r["data_loss"], dup=dup))
            print(f"[fits] {src:10s} λ={lam:<7g} M={m.M:2d}  D={r['data_loss']:.2e}"
                  f"{'  (duplicate)' if dup else ''}", flush=True)
    np.savez(FITS_NPZ, source=np.array([r["source"] for r in rows]), lam=np.array([r["lam"] for r in rows]),
             M=np.array([r["M"] for r in rows]), dup=np.array([r["dup"] for r in rows]),
             data_loss=np.array([r["data_loss"] for r in rows]),
             w=np.concatenate([r["w"] for r in rows]), Om=np.concatenate([r["Om"] for r in rows]),
             De=np.concatenate([r["De"] for r in rows]))


def load_fits():
    d = np.load(FITS_NPZ)
    off = np.concatenate([[0], np.cumsum(d["M"])])
    return [dict(idx=i, source=str(d["source"][i]), lam=float(d["lam"][i]), M=int(d["M"][i]),
                 dup=bool(d["dup"][i]), data_loss=float(d["data_loss"][i]),
                 w=d["w"][off[i]:off[i + 1]], Om=d["Om"][off[i]:off[i + 1]], De=d["De"][off[i]:off[i + 1]])
            for i in range(d["M"].size)]


# ════════════════════════════════════════════════════════════════════════════
#  exact references for the true ρ
# ════════════════════════════════════════════════════════════════════════════
def _rho_scalar(w):
    return float((CW * norm.pdf((w - MU) / SD) / SD).sum())


def make_truth():
    Hr = lambda nu: quad(_rho_scalar, -30, 30, weight="cauchy", wvar=nu, limit=400)[0]
    nu_s = brentq(Hr, 1.5, 2.3)
    K_H = 2.0 / (np.pi * _rho_scalar(nu_s))
    G = lambda u: quad(lambda th: np.cos(th) ** 2 * _rho_scalar(u * np.sin(th)), -np.pi / 2, np.pi / 2,
                       limit=200)[0]
    us = np.linspace(0.02, 12.0, 1200)
    Gs = np.array([G(u) for u in us])
    i = int(np.argmin(1 / Gs))
    res = minimize_scalar(lambda u: 1 / G(u), bracket=(us[i - 3], us[i], us[i + 3]))
    print(f"[truth] K_H = {K_H:.6f} (ν* = {nu_s:.6f});  K_SN = {res.fun:.6f} (r = {res.x * G(res.x):.4f})")
    np.savez(TRUTH_NPZ, K_H=K_H, nu_star=nu_s, K_SN=res.fun, r_SN=res.x * G(res.x),
             branch_K=1 / Gs, branch_r=us * Gs)


# ════════════════════════════════════════════════════════════════════════════
#  PyCoBi continuations (one ODESystem per process)
# ════════════════════════════════════════════════════════════════════════════
def _lab_circuit(f, K0):
    from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate
    M = f["M"]
    w, Om, De = ([float(v) for v in f[k]] for k in ("w", "Om", "De"))     # plain floats for the parser
    Hr = "K*(" + " + ".join(f"{w[j]!r}*x_{j}" for j in range(M)) + ")"
    Hi = "K*(" + " + ".join(f"{w[j]!r}*y_{j}" for j in range(M)) + ")"
    eqs, var = [], {}
    for m in range(M):
        x, y = f"x_{m}", f"y_{m}"
        eqs.append(f"d/dt * {x} = -{De[m]!r}*{x} - ({Om[m]!r})*{y} + 0.5*(({Hr}) - "
                   f"((({x})^2 - ({y})^2)*({Hr}) + 2*{x}*{y}*({Hi})))")
        eqs.append(f"d/dt * {y} = -{De[m]!r}*{y} + ({Om[m]!r})*{x} + 0.5*(({Hi}) - "
                   f"(2*{x}*{y}*({Hr}) - (({x})^2 - ({y})^2)*({Hi})))")
        var[x] = "output(0.0)" if m == 0 else "variable(0.0)"
        var[y] = "variable(0.0)"
    var["K"] = float(K0)
    op = OperatorTemplate(name="lab_op", equations=eqs, variables=var)
    return CircuitTemplate(name="kmo_lab", nodes={"p": NodeTemplate(name="lab_node", operators=[op])})


def _summary_cols(ode, cont):
    s = ode.get_summary(cont)
    head = lambda n: [c for c in s.columns if (c[0] if isinstance(c, tuple) else c) == n]
    col = lambda n: np.asarray(s[head(n)[0]], float) if head(n) else None
    bif = np.asarray(s[head("bifurcation")[0]]).astype(str)
    stab = np.asarray(s[head("stability")[0]], bool) if head("stability") else None
    return s, col, bif, stab


def auto_incoherent(f, cfg=CFG):
    """Continue z = 0 in K (lab frame) -> Hopf points of the incoherent state."""
    from pycobi import ODESystem
    from pyrates import clear
    circ = _lab_circuit(f, 0.3)
    ode = ODESystem.from_template(circ, auto_dir=AUTO_DIR, init_cont=False, analytical_jacobian=True,
                                  auto_constants=("ivp", "eq"))
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=0.5, NMX=20000,
            UZR={14: 5.0}, STOP={"UZ1"})
    ode.run(origin="time", starting_point="UZ1", name="incoh", c="eq", ICP="K",
            RL0=cfg["K_lo"], RL1=cfg["K_hi"], IPS=1, ILP=0, ISP=2, ISW=1, NMX=8000, NPR=50,
            DS=1e-3, DSMIN=1e-9, DSMAX=5e-3, EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, get_stability=True)
    s, col, bif, stab = _summary_cols(ode, "incoh")
    K = col("K")
    K_hb = np.unique(np.round(K[bif == "HB"], 6))
    ode.close_session(clear_files=True); clear(circ)
    return dict(K_hb=K_hb)


def auto_locked(f, cfg=CFG):
    """Continue the locked state in K (co-rotating frame) -> LP/HB bounding its stability."""
    from pycobi import ODESystem
    from pyrates import clear
    import kmo_heterogeneity_bifurcation as KHB
    M = f["M"]
    if M == 1:   # single Lorentzian: locked state = OA fixed point R² = 1 − 2Δ/K, born at K = 2Δ (BP)
        return dict(K=np.array([]), R=np.array([]), stab=np.array([]), K_lp=np.array([]),
                    K_hb=np.array([]), K_SN=float(2 * f["De"][0]))
    # h is fixed at 1, so the weighted mean frequency cancels: pass 0.0 (see KHB.build_equations)
    circ = KHB.build_circuit(M, cfg["K_settle_sync"], f["Om"], f["De"], f["w"], np.full(M, 0.8),
                             combined=True, h0=1.0, name="kmo_lock", ombar=0.0)
    ode = ODESystem.from_template(circ, auto_dir=AUTO_DIR, init_cont=False, analytical_jacobian=True,
                                  auto_constants=("ivp", "eq"))
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: 2000.0}, STOP={"UZ1"})
    ode.run(origin="time", starting_point="UZ1", name="lock", c="eq", ICP="K", bidirectional=True,
            RL0=cfg["K_lo"], RL1=cfg["K_hi"], IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=20,
            DSMIN=1e-9, DSMAX=5e-3, EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, STOP={"BP1"}, get_stability=True)
    s, col, bif, stab = _summary_cols(ode, "lock")
    K = col("K")
    z = np.zeros(K.size, complex)
    for m in range(M):
        z += f["w"][m] * (col(f"x_{m}") + 1j * (col(f"y_{m}") if m else 0.0))
    R = np.abs(z)
    # bidirectional runs list each special point once per direction -> de-duplicate
    K_lp = np.unique(np.round(K[bif == "LP"], 6))
    K_hb = np.unique(np.round(K[bif == "HB"], 6))
    # onset of the stable LOCKED state: lower end of the stable high-coherence branch (R > 0.5),
    # snapped to the fold (LP) that bounds it. (Stable points near R ~ 0, where the co-rotating
    # frame becomes singular, are excluded.)
    good = stab & (R > 0.5) if stab is not None else np.zeros(K.size, bool)
    K_SN = float(K[good].min()) if good.any() else np.nan
    if good.any() and K_lp.size and np.min(np.abs(K_lp - K_SN)) < 0.05:
        K_SN = float(K_lp[np.argmin(np.abs(K_lp - K_SN))])
    ode.close_session(clear_files=True); clear(circ)
    return dict(K=K, R=R, stab=stab, K_lp=K_lp, K_hb=K_hb, K_SN=K_SN)


def eig_KH(f, lo=0.05, hi=20.0):
    """Cross-check: K at which the linearisation at z = 0 first has an eigenvalue with Re > 0."""
    A0 = np.diag(1j * f["Om"] - f["De"]); P = 0.5 * np.outer(np.ones(f["M"]), f["w"])
    g = lambda K: np.max(np.linalg.eigvals(A0 + K * P).real)
    if g(hi) < 0:
        return np.nan
    for _ in range(80):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def run_auto(idx, which):
    f = load_fits()[idx]
    wd = dp.ensure(os.path.join(OUT, "auto_work")); os.chdir(wd)      # keep Auto/PyRates files out of the repo
    out = os.path.join(OUT, f"auto_{idx:02d}_{which}.npy")
    t0 = time.time()
    res = auto_incoherent(f) if which == "incoh" else auto_locked(f)
    res.update(idx=idx, which=which, K_H_eig=eig_KH(f))
    np.save(out, res, allow_pickle=True)
    print(f"[auto] fit {idx:2d} ({f['source']}, M={f['M']}) {which}: "
          + (f"HB={np.round(res['K_hb'], 4)} (eig {res['K_H_eig']:.4f})" if which == "incoh"
             else f"LP={np.round(res['K_lp'], 4)} HB={np.round(res['K_hb'], 4)} K_SN={res['K_SN']:.4f}")
          + f"  ({time.time() - t0:.0f}s)", flush=True)


def run_auto_all():
    fits = load_fits()
    for f in fits:
        if f["dup"]:
            continue
        for which in ("incoh", "sync"):
            out = os.path.join(OUT, f"auto_{f['idx']:02d}_{which}.npy")
            if os.path.exists(out):
                continue
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "auto", str(f["idx"]), which],
                               capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("[auto]")]
            print(line[-1] if line else f"[auto] fit {f['idx']} {which}: FAILED\n{r.stderr[-1500:]}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  dynamics on the K grid: network, true-ρ continuum, all LMMF fits
# ════════════════════════════════════════════════════════════════════════════
def _riccati(nu, q, z0, K, T, dts):
    nu = np.asarray(nu, complex); q = np.asarray(q, float); a = 1j * nu
    t_eval = np.arange(0.0, T + 1e-9, dts)

    def rhs(t, z):
        Z = q @ z
        return a * z + 0.5 * K * (Z - np.conj(Z) * z * z)
    sol = solve_ivp(rhs, (0, T), np.broadcast_to(np.asarray(z0, complex), nu.shape).copy(),
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return np.abs(q @ sol.y)


def _network(om, th0, K, T, dts):
    t_eval = np.arange(0.0, T + 1e-9, dts)

    def rhs(t, th):
        Z = np.exp(1j * th).mean()
        return om + K * np.imag(Z * np.exp(-1j * th))
    sol = solve_ivp(rhs, (0, T), th0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return np.abs(np.exp(1j * sol.y).mean(0))


def run_sims(cfg=CFG):
    om, th0 = fig1_sample()
    R0 = float(np.abs(np.exp(1j * th0).mean()))
    nodes = quantile_nodes(20000)
    fits = [f for f in load_fits() if not f["dup"]]
    Ks = cfg["K_grid"]
    R_net = np.zeros((Ks.size, int(cfg["T"] / cfg["dts"]) + 1)); R_cont = np.zeros_like(R_net)
    R_fit = np.zeros((len(fits),) + R_net.shape)
    for iK, K in enumerate(Ks):
        t0 = time.time()
        R_net[iK] = _network(om, th0, K, cfg["T"], cfg["dts"])
        R_cont[iK] = _riccati(nodes, np.full(nodes.size, 1 / nodes.size), R0, K, cfg["T"], cfg["dts"])
        for j, f in enumerate(fits):
            R_fit[j, iK] = _riccati(f["Om"] + 1j * f["De"], f["w"], R0, K, cfg["T"], cfg["dts"])
        print(f"[sims] K={K:.2f}  R_net(end)={R_net[iK, -1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(SIMS_NPZ, K=Ks, t=np.arange(0.0, cfg["T"] + 1e-9, cfg["dts"]), R_net=R_net, R_cont=R_cont,
             R_fit=R_fit, fit_idx=np.array([f["idx"] for f in fits]), R0=R0)
    print(f"[saved] {SIMS_NPZ}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fits":
        make_fits()
    elif mode == "truth":
        make_truth()
    elif mode == "auto":
        run_auto(int(sys.argv[2]), sys.argv[3])
    elif mode == "auto_all":
        run_auto_all()
    elif mode == "sims":
        run_sims()
