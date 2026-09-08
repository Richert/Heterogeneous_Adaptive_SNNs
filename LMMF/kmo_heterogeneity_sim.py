r"""
Kuramoto heterogeneity control — Lorentzian-mixture fit + transient-h simulation
================================================================================

Companion to the QIF heterogeneity analysis, but for a globally coupled Kuramoto network with an
ARBITRARY (Gaussian-mixture) frequency distribution.  A single heterogeneity knob h scales the
frequency distribution about its mean omega_bar:

    micro:   omega_i(h) = omega_bar + h (omega_i^0 - omega_bar)
    MF:      Omega_m(h)  = omega_bar + h (Omega_m^0 - Omega_bar),   Delta_m(h) = h Delta_m^0

so h=1 is the data fit, h->0 the homogeneous limit (all frequencies -> omega_bar), h>1 more
heterogeneous.  For the symmetric bimodal mixture omega_bar~0, so h ~ an overall frequency scale.

Modes (CLI):
    fit    : sample the Gaussian mixture, fit a Lorentzian mixture, save kmo_het_fit.npz
    scan   : MF steady-state coherence R over an (h, K) grid -> pick a regime (prints a table)
    rate   : transient-h R(t), micro Kuramoto vs ensemble OA mean field -> save kmo_het_rate.npz

Run in the ``pycobi`` conda env (dev PyRates 1.2.2 + scipy):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_heterogeneity_sim.py fit
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from pyrates import NodeTemplate, CircuitTemplate, PopulationTemplate, Connectivity, clear

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import kmo_lorentzian_fit_sweep as KFS          # _op, _var_slice, sample_gaussian_mixture, LM

OUT_DIR = "/home/rgast/data/mpmf_simulations"
FIT_NPZ = os.path.join(OUT_DIR, "kmo_het_fit.npz")
RATE_NPZ = os.path.join(OUT_DIR, "kmo_het_rate.npz")
ENV_NPZ = os.path.join(OUT_DIR, "kmo_het_envelope.npz")
REGIONS_HK_NPZ = os.path.join(OUT_DIR, "kmo_het_regions_hK.npz")
REGIONS_DC_NPZ = os.path.join(OUT_DIR, "kmo_het_regions_DC.npz")

CONFIG = dict(
    # multimodal, asymmetric Gaussian mixture (4 separated components) — a "complex" frequency
    # distribution echoing the multimodal PV L5/6 shape; fits cleanly with M=4 Lorentzians.
    # Rescaled by 0.7 vs the first version so that the data fit (h=1) sits at the previous h=0.7
    # state (closer to the synchronization / limit-cycle regime).
    gmm_means=[-1.40, -0.49, 0.49, 1.40], gmm_stds=[0.21, 0.238, 0.238, 0.21],
    gmm_weights=[0.18, 0.30, 0.30, 0.22],
    N=5000, K=1.0,                       # coupling: at K=1 the data fit (h=1) is incoherent, and
                                         # reducing h synchronizes -> clean heterogeneity control
    # transient h(t): STAIRCASE h0 -> h1 (at t1) -> h2 (at t2): synchronized -> standing wave -> async
    h0=0.4, h1=0.6, h2=1.0, t1=20.0, t2=60.0,
    sigma0=0.8,                          # mildly-coherent initial condition spread (relaxes to baseline)
    T=100.0, dt=1e-2, dts=0.1, rtol=1e-6, atol=1e-8,
    # Lorentzian-mixture fit (single fit, not the full meta-sweep)
    delta_bounds=(1e-4, 1e2), M_max=8, alpha=1e-2, lambda_M=5e-5,
    n_restarts=12, patience=3, loss="cvm", method="slsqp",
    seed=1,
)


def _param(args, keys, suffix):
    """The (mutable) PyRates parameter array whose key ends in `suffix`."""
    return args[keys.index(next(k for k in keys if k.endswith(suffix)))]


# ════════════════════════════════════════════════════════════════════════════
#  fit
# ════════════════════════════════════════════════════════════════════════════
def fit_and_save(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])
    omega = KFS.sample_gaussian_mixture(cfg["gmm_means"], cfg["gmm_stds"], cfg["gmm_weights"],
                                        cfg["N"], rng)
    res = KFS.LM.fit(omega, cfg["delta_bounds"], M_max=cfg["M_max"], alpha=cfg["alpha"],
                     lambda_M=cfg["lambda_M"], patience=cfg["patience"], loss=cfg["loss"],
                     n_restarts=cfg["n_restarts"], seed=cfg["seed"], method=cfg["method"])
    m, M = res["model"], res["M"]
    Ombar = float(m.w @ m.Omega)
    print(f"Gaussian mixture (means={cfg['gmm_means']}, stds={cfg['gmm_stds']}): "
          f"N={cfg['N']}, sample mean={omega.mean():.3f}")
    print(f"  Lorentzian fit: M={M}, p={res['pvalue']:.3f}, Omega_bar={Ombar:.4f}")
    print(f"  w={np.round(m.w,3)}\n  Omega={np.round(m.Omega,3)}\n  Delta={np.round(m.Delta,3)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(FIT_NPZ, weights=m.w, omega=m.Omega, delta=m.Delta, M=np.int64(M),
             samples=omega, omega_bar=Ombar, K=float(cfg["K"]),
             gmm_means=cfg["gmm_means"], gmm_stds=cfg["gmm_stds"], gmm_weights=cfg["gmm_weights"])
    print(f"[saved] {FIT_NPZ}")
    return m, omega


def load_fit():
    d = np.load(FIT_NPZ, allow_pickle=False)
    return (np.asarray(d["weights"], float), np.asarray(d["omega"], float),
            np.asarray(d["delta"], float), float(d["omega_bar"]), np.asarray(d["samples"], float))


# ════════════════════════════════════════════════════════════════════════════
#  ensemble OA mean field with time-varying h  (inject Omega_m(h), Delta_m(h))
# ════════════════════════════════════════════════════════════════════════════
def run_ensemble_h(w, Om0, De0, Ombar, K, z0, cfg, h_of_t, tag="ensh"):
    M = w.size
    node = NodeTemplate(name="ens", operators=[KFS._op("oa_op"), KFS._op("ens_coupling_op")])
    pop = PopulationTemplate(name="ens", node=node, n=M,
                             params={"oa_op/Omega": Om0.copy(), "oa_op/Delta": De0.copy(),
                                     "oa_op/z": np.asarray(z0, complex), "ens_coupling_op/wm": w})
    conn = Connectivity("ens/ens_coupling_op/zc", "ens/oa_op/h", weights=float(K))
    net = CircuitTemplate("kmo_ens", populations={"ens": pop}, connections=[conn])
    func, args, keys, vmap = net.get_run_func(f"{tag}_vf", step_size=cfg["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="complex128")
    y0 = np.asarray(args[1]); extra = tuple(args[2:])
    p_Om = _param(args, keys, "oa_op/Omega"); p_De = _param(args, keys, "oa_op/Delta")

    def f(t, y):
        h = h_of_t(t)
        p_Om[:] = Ombar + h * (Om0 - Ombar)
        p_De[:] = h * De0
        return np.array(func(t, y, *extra))

    t_eval = np.arange(0.0, cfg["T"], cfg["dts"])
    sol = solve_ivp(f, (0.0, cfg["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["dt"])
    z = sol.y[KFS._var_slice(vmap, "oa_op/z")]
    R = np.abs(w @ z)
    clear(net)
    return t_eval, R


# ════════════════════════════════════════════════════════════════════════════
#  microscopic Kuramoto network with time-varying h  (inject omega_i(h))
# ════════════════════════════════════════════════════════════════════════════
def run_micro_h(om0, ombar, K, theta0, cfg, h_of_t, tag="microh"):
    N = om0.size
    node = NodeTemplate(name="osc", operators=[KFS._op("kmo_op")])
    pop = PopulationTemplate(name="osc", node=node, n=N,
                             params={"kmo_op/omega": om0.copy(), "kmo_op/theta": theta0})
    conn = Connectivity("osc/kmo_op/e", "osc/kmo_op/s_in", weights=K / N)
    net = CircuitTemplate("kmo_micro", populations={"osc": pop}, connections=[conn])
    func, args, keys, vmap = net.get_run_func(f"{tag}_vf", step_size=cfg["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="complex128")
    y0 = np.asarray(args[1]); extra = tuple(args[2:])
    p_om = _param(args, keys, "kmo_op/omega")

    def f(t, y):
        h = h_of_t(t)
        p_om[:] = ombar + h * (om0 - ombar)
        return np.array(func(t, y, *extra))

    t_eval = np.arange(0.0, cfg["T"], cfg["dts"])
    sol = solve_ivp(f, (0.0, cfg["T"]), y0, method="RK45", t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["dt"])
    theta = np.real(sol.y[KFS._var_slice(vmap, "kmo_op/theta")])
    R = np.abs(np.exp(1j * theta).mean(axis=0))
    clear(net)
    return t_eval, R


def _hfun(cfg):
    """Transient staircase: h0 for t<t1, h1 for t1<=t<t2, h2 for t>=t2."""
    t1, t2, h0, h1, h2 = cfg["t1"], cfg["t2"], cfg["h0"], cfg["h1"], cfg["h2"]
    return lambda t: (h0 if t < t1 else (h1 if t < t2 else h2))


# ════════════════════════════════════════════════════════════════════════════
#  regime scan: MF steady-state R over an (h, K) grid
# ════════════════════════════════════════════════════════════════════════════
def scan(cfg=CONFIG):
    w, Om0, De0, Ombar, samples = load_fit()
    scfg = dict(cfg, T=60.0)
    print(f"MF steady-state R over (h, K)  [Omega_bar={Ombar:.3f}]")
    Ks = [1.0, 2.0, 3.0, 4.0, 6.0]
    hs = [1.0, 0.7, 0.5, 0.3, 0.1]
    print("     h:  " + "  ".join(f"{h:5.2f}" for h in hs))
    for K in Ks:
        row = []
        for h in hs:
            t, R = run_ensemble_h(w, Om0, De0, Ombar, K, np.full(w.size, 0.3 + 0j), scfg,
                                  (lambda hv: (lambda tt: hv))(h))
            row.append(R[-20:].mean())
        print(f"  K={K:4.1f}:  " + "  ".join(f"{v:5.3f}" for v in row))


# ════════════════════════════════════════════════════════════════════════════
#  transient-h rate simulation: micro vs MF
# ════════════════════════════════════════════════════════════════════════════
def run_rate(cfg=CONFIG):
    w, Om0, De0, Ombar, samples = load_fit()
    rng = np.random.default_rng(cfg["seed"])
    # Mildly coherent start: it relaxes to the incoherent data-fit (h=1) state during the baseline,
    # leaving micro and MF at a common small seed so they synchronize together during the pulse
    # (a fully-incoherent start makes the deterministic MF lag the fluctuation-seeded micro).
    theta0 = rng.normal(0.0, cfg["sigma0"], cfg["N"])
    R0 = float(np.abs(np.exp(1j * theta0).mean()))
    z0 = np.full(w.size, R0 + 0.0j)
    ombar_mic = float(samples.mean())
    hfun = _hfun(cfg)
    print(f"== Kuramoto transient-h: K={cfg['K']}, h: {cfg['h0']} ->{cfg['h1']} (t={cfg['t1']}) "
          f"->{cfg['h2']} (t={cfg['t2']}), R(0)={R0:.3f} ==")
    tf, Rf = run_ensemble_h(w, Om0, De0, Ombar, cfg["K"], z0, cfg, hfun)
    print(f"  MF done: R(end)~{Rf[-20:].mean():.3f}")
    tm, Rm = run_micro_h(samples, ombar_mic, cfg["K"], theta0, cfg, hfun)
    print(f"  micro done: R(end)~{Rm[-20:].mean():.3f}")
    np.savez(RATE_NPZ, t_mf=tf, R_mf=Rf, t_micro=tm, R_micro=Rm, K=float(cfg["K"]),
             t1=float(cfg["t1"]), t2=float(cfg["t2"]), h0=float(cfg["h0"]), h1=float(cfg["h1"]),
             h2=float(cfg["h2"]), N=np.int64(cfg["N"]))
    print(f"[saved] {RATE_NPZ}")


def run_envelope(cfg=CONFIG):
    """Mean-field R-envelope vs h at K=K_fix by a STEPPING sweep (poor-man's continuation): build
    the ensemble net once, then step h while carrying the previous attractor state forward as the
    next initial condition (short re-settle).  Swept UP from a coherent seed and DOWN from an
    asynchronous seed so any hysteresis between the synchronized / standing-wave / asynchronous
    regimes is captured.  Records min/max of R over the settled tail at each step."""
    w, Om0, De0, Ombar, _ = load_fit(); M = w.size
    node = NodeTemplate(name="ens", operators=[KFS._op("oa_op"), KFS._op("ens_coupling_op")])
    pop = PopulationTemplate(name="ens", node=node, n=M,
                             params={"oa_op/Omega": Om0.copy(), "oa_op/Delta": De0.copy(),
                                     "oa_op/z": np.full(M, 0.6 + 0j), "ens_coupling_op/wm": w})
    conn = Connectivity("ens/ens_coupling_op/zc", "ens/oa_op/h", weights=float(cfg["K"]))
    net = CircuitTemplate("kmo_ens_env", populations={"ens": pop}, connections=[conn])
    func, args, keys, vmap = net.get_run_func("env_vf", step_size=cfg["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="complex128")
    extra = tuple(args[2:]); z_sl = KFS._var_slice(vmap, "oa_op/z")
    p_Om = _param(args, keys, "oa_op/Omega"); p_De = _param(args, keys, "oa_op/Delta")
    # longer settle per step + larger post-step transient discount -> cleaner envelope
    T = 500.0; t_eval = np.arange(0.0, T, cfg["dts"]); tail = t_eval > 0.7 * T

    def integrate(h, y0):
        p_Om[:] = Ombar + h * (Om0 - Ombar); p_De[:] = h * De0
        sol = solve_ivp(lambda t, y: np.array(func(t, y, *extra)), (0.0, T), y0, method="RK45",
                        t_eval=t_eval, rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["dt"])
        R = np.abs(w @ sol.y[z_sl])
        return R, sol.y[:, -1]

    hs = np.round(np.arange(0.40, 1.041, 0.01), 3)        # fine raster

    def sweep(hgrid, R_seed):
        y = np.asarray(args[1]).copy(); y[:] = R_seed       # uniform coherent/async seed
        lo, hi = [], []
        for h in hgrid:
            R, y = integrate(float(h), y)
            lo.append(float(R[tail].min())); hi.append(float(R[tail].max()))
        return np.array(lo), np.array(hi)

    print("stepping sweep UP from coherent ...")
    lo_up, hi_up = sweep(hs, 0.8 + 0j)
    print("stepping sweep DOWN from asynchronous ...")
    lo_dn, hi_dn = sweep(hs[::-1], 0.05 + 0j); lo_dn, hi_dn = lo_dn[::-1], hi_dn[::-1]
    clear(net)
    for i, h in enumerate(hs):
        print(f"  h={h:.2f}: up R∈[{lo_up[i]:.3f},{hi_up[i]:.3f}]  down R∈[{lo_dn[i]:.3f},{hi_dn[i]:.3f}]")
    np.savez(ENV_NPZ, h=hs, R_min_up=lo_up, R_max_up=hi_up, R_min_dn=lo_dn, R_max_dn=hi_dn,
             K=float(cfg["K"]))
    print(f"[saved] {ENV_NPZ}")


def _ens_net(K, M, Om0, De0, w, cfg):
    """Build the ensemble OA net at coupling K once; return (net, integrate(Omv,Dev,y0)->(R,yf),
    y_template).  Omega/Delta are injected (mutable params), so a 2-D (h-knob) grid needs no rebuild."""
    node = NodeTemplate(name="ens", operators=[KFS._op("oa_op"), KFS._op("ens_coupling_op")])
    pop = PopulationTemplate(name="ens", node=node, n=M,
                             params={"oa_op/Omega": Om0.copy(), "oa_op/Delta": De0.copy(),
                                     "oa_op/z": np.full(M, 0.6 + 0j), "ens_coupling_op/wm": w})
    conn = Connectivity("ens/ens_coupling_op/zc", "ens/oa_op/h", weights=float(K))
    net = CircuitTemplate("ens_reg", populations={"ens": pop}, connections=[conn])
    func, args, keys, vmap = net.get_run_func("reg_vf", step_size=cfg["dt"], backend="numpy",
                                              vectorize=True, clear=False, float_precision="complex128")
    extra = tuple(args[2:]); z_sl = KFS._var_slice(vmap, "oa_op/z")
    p_Om = _param(args, keys, "oa_op/Omega"); p_De = _param(args, keys, "oa_op/Delta")
    Tg = 220.0; t_eval = np.arange(0.0, Tg, 0.2); tail = t_eval > 0.55 * Tg

    def integrate(Omv, Dev, y0):
        p_Om[:] = Omv; p_De[:] = Dev
        sol = solve_ivp(lambda t, y: np.array(func(t, y, *extra)), (0.0, Tg), y0, method="RK45",
                        t_eval=t_eval, rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["dt"])
        seg = np.abs(w @ sol.y[z_sl])[tail]
        return seg, sol.y[:, -1]
    return net, integrate, np.asarray(args[1])


def _classify(seg, amp_tol=0.02):
    """1 = synchronized FP, -1 = asynchronous FP, 0 = oscillatory (torus)."""
    if seg.max() - seg.min() > amp_tol:
        return 0
    return 1 if seg.mean() > 0.3 else -1


def _region_grid(integrate, ytempl, p1g, p2g, omde):
    """Mono-stable classification on a (p1 x p2) grid via two seeds (coherent + async), each
    carried (stepped) along p1 for fixed p2.  1=mono sync, -1=mono async, 0=multistable/torus."""
    n2, n1 = len(p2g), len(p1g)
    cls = {}
    for seed in (0.8, 0.05):
        c = np.zeros((n2, n1))
        for i2, p2 in enumerate(p2g):
            y = ytempl.copy(); y[:] = seed
            for i1, p1 in enumerate(p1g):
                Omv, Dev = omde(p1, p2)
                seg, y = integrate(Omv, Dev, y)
                c[i2, i1] = _classify(seg)
        cls[seed] = c
    out = np.zeros((n2, n1))
    out[(cls[0.8] == 1) & (cls[0.05] == 1)] = 1
    out[(cls[0.8] == -1) & (cls[0.05] == -1)] = -1
    return out


def run_regions(cfg=CONFIG):
    """Mono-stable sync / async region maps in the (h, K) and (h_Delta, h_ombar) planes."""
    w, Om0, De0, Ombar, _ = load_fit(); M = w.size
    n = 30
    # (h_Delta, h_ombar) plane: K fixed, vary hD (Delta) and hC (Omega) by injection
    net, integ, ytempl = _ens_net(cfg["K"], M, Om0, De0, w, cfg)
    hDg = np.linspace(0.02, 1.3, n); hCg = np.linspace(0.02, 1.3, n)
    DC = _region_grid(integ, ytempl, hDg, hCg, lambda hD, hC: (Ombar + hC * (Om0 - Ombar), hD * De0))
    clear(net)
    np.savez(REGIONS_DC_NPZ, x=hDg, y=hCg, cls=DC)
    print(f"[saved] {REGIONS_DC_NPZ}  (sync {int((DC==1).sum())}, async {int((DC==-1).sum())})")
    # (h, K) plane: rebuild per K, vary lumped h by injection
    Kg = np.linspace(0.2, 3.0, n); hg = np.linspace(0.05, 1.3, n)
    cohK = np.zeros((n, n)); asyK = np.zeros((n, n))
    for iK, K in enumerate(Kg):
        net, integ, ytempl = _ens_net(float(K), M, Om0, De0, w, cfg)
        for ci, seed in ((0, 0.8), (1, 0.05)):
            y = ytempl.copy(); y[:] = seed
            for ih, h in enumerate(hg):
                seg, y = integ(Ombar + h * (Om0 - Ombar), h * De0, y)
                (cohK if ci == 0 else asyK)[iK, ih] = _classify(seg)
        clear(net)
    HK = np.zeros((n, n))
    HK[(cohK == 1) & (asyK == 1)] = 1
    HK[(cohK == -1) & (asyK == -1)] = -1
    np.savez(REGIONS_HK_NPZ, x=hg, y=Kg, cls=HK)
    print(f"[saved] {REGIONS_HK_NPZ}  (sync {int((HK==1).sum())}, async {int((HK==-1).sum())})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    {"fit": fit_and_save, "scan": scan, "rate": run_rate, "envelope": run_envelope,
     "regions": run_regions}[mode]()
