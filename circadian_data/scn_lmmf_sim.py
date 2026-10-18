r"""
Interactive LMMF simulation of one SCN slice: TTX-sensitive + TTX-insensitive coupling, TTX inputs
==================================================================================================

Model (θ = PER2 phase), cell i in lobe L, TTX level u(t) ∈ [0, 1]:

    θ̇_i = ω_i − δω (1 − u(t))  +  K_s(t) Im(e^{−iα_s} Z e^{−iθ_i})  +  K_i Im(e^{−iα_i} Z_• e^{−iθ_i})  [+ Cauchy noise γ u(t)]

  TTX-sensitive coupling (synaptic, VIP/GABA): global mean field Z, phase lag α_s, strength K_s(t) = K_pre / K_ttx /
      K_wash in the three time domains (steps, or first-order relaxations with --rate-ttx / --rate-wash), OR the
      spiking-rate PROFILE (--K-min and/or --K-max given):
          K_s = K_pre                                                   t < t_ttx
          K_s = K_min + (K_ttx − K_min) σ((t − t_ttx − t_rec)/w_rec)    t_ttx ≤ t < t_wash   (drop to K_min, DELAYED
                                                                         S-shaped partial recovery to K_ttx around
                                                                         t_rec h after TTX application; σ = logistic)
          K_s = K_wash + (K_max − K_wash) e^{−(t − t_wash)/τ_wash}      t ≥ t_wash   (rebound to K_max, decay to the
                                                                                      post-washout level K_wash)
  TTX-insensitive coupling (gap junctions, glia): strength K_i, phase lag α_i, mean field Z_• = the cell's own
      lobe (default) or the whole slice (--insens-global).
  Frequency input: the intrinsic frequencies ω_i are MEASURED under TTX. If TTX shifts every cell's frequency by
      δω (--dw-ttx, rad/h, > 0 = TTX speeds the cells up), the cells run at ω_i − δω outside TTX; the input is
      −δω (1 − u(t)) (a global shift).
  TTX noise: Cauchy (Lorentzian) phase noise of scale γ (--gamma-ttx, rad/h) under TTX. For OA ensembles,
      independent Cauchy white noise of scale γ acts exactly like an extra Lorentzian width γ on every component
      (Tönjes & Pikovsky 2020; Tanaka 2020): Δ_m → Δ_m + γ u(t). A proxy for the amplitude/phase noise of
      the weakened single-cell rhythms under TTX.
  TTX quenched disorder: cell-specific, Lorentzian-distributed frequency shifts of scale γ_q (--gamma-q, rad/h)
      caused by TTX, independent of ω_i (at the OA level an extra width γ_q). The frequencies are MEASURED under
      TTX, so the measured distribution contains both the intrinsic and this TTX-induced disorder; outside TTX
      the TTX-induced part is absent and every component is narrower: Δ_m → Δ_m − γ_q (1 − u(t)) (clipped at
      DELTA_MIN; a warning is printed if γ_q exceeds the narrowest component).
  Phase scrambling (--reset-ttx / --reset-wash, rad): at TTX application / washout every cell receives an
      independent phase kick drawn from a wrapped Cauchy distribution of scale σ (zero mean). At the OA level this
      is exact: z_m → e^{−σ} z_m (the kick convolves each component's Poisson kernel). A proxy for a transient,
      cell-specific phase reset by the treatment change (scn_ttx_phase_reset.py; circ SD s ↔ σ ≈ s²/2 for small s).
  u(t): 0 before TTX, 1 under TTX, 0 after washout; steps by default, or first-order relaxations with the
      rates --rate-ttx / --rate-wash (1/h), which also shape K_s(t). Bath-applied TTX blocks spiking within
      minutes; washout is slower (hours) — plausible ≈ 1–2 h⁻¹ (application), 0.1–0.5 h⁻¹ (washout); assumptions.

Data / target (scn_targets): epoch-wise peak phases (validated in scn_pipeline_validation.py), intrinsic
frequencies = peak-phase slopes in the TTX epoch, R(t) of the slice and lobes, the measurement ceiling c(t) and the
mask (valid window without ±MASK_H around TTX application and washout). The model is shown and scored as
c(t)|Z(t)| (--no-ceiling: |Z(t)|), RMSE over the mask.

1. Fits one Lorentzian mixture per lobe to the intrinsic (TTX-epoch) frequencies (rad/h), with or without the
   noise floor (λ, width floor k); weights scaled by the lobe fractions.
2. Simulates the LMMF over the whole recording from the MEASURED state at --t-start (each cell assigned to its
   most likely component; z_m(0) = mean e^{iθ} of its cells):
       ż_m = (i(Ω_m − δω(1 − u)) − (Δ_m + γ u)) z_m + ½ (H_m − H̄_m z_m²),
       H_m = K_s(t) e^{−iα_s} Z + K_i e^{−iα_i} Z_•,       z_m → e^{−σ} z_m at t_ttx / t_wash (phase scrambling)
3. Two-column figure: per-lobe intrinsic distributions + fits | data vs model R(t) (slice and lobes), with the
   inputs (K_s(t), K_i, frequency shift, noise) underneath.

Units: time in hours, frequencies, couplings, δω and γ in rad/h, phase lags in rad.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_lmmf_sim.py --slice 1 --K-pre 0.08 --K-ttx 0.005 \
        --K-wash 0.06 --alpha-s 0.2 --K-i 0.01 --alpha-i 0.0 [--insens-global] [--dw-ttx 0.0] [--gamma-ttx 0.0] \
        [--gamma-q 0.0] [--reset-ttx 0.0] [--reset-wash 0.0] \
        --lam 1e-6 [--no-floor] [--width-k 10] [--rate-ttx 1.0] [--rate-wash 0.2] [--t-start 24] [--no-ceiling] [--show]
(Old option names --alpha and --K-gap are accepted for --alpha-s and --K-i. The pulse-coupling variant is kept
in scn_pulse_fit.py.)
Output: data_paths.MPMF/circadian_data/sim/scn<slice>_<tag>.{png,pdf}
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import argparse
import numpy as np
from scipy.stats import gaussian_kde
import data_paths as dp
import lorentzian_mixture as LM
import scn_common as S
import scn_targets as TG

OUT = dp.ensure(os.path.join(S.OUT, "sim"))
DT = 0.25                     # RK4 step (h)
DELTA_MIN = 1e-5              # lower clip of the component widths outside TTX (rad/h), see --gamma-q


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice", type=int, default=1, choices=S.SLICES)
    p.add_argument("--K-pre", type=float, default=0.08, help="TTX-sensitive coupling before TTX (rad/h)")
    p.add_argument("--K-ttx", type=float, default=0.005, help="TTX-sensitive coupling under TTX (rad/h)")
    p.add_argument("--K-wash", type=float, default=0.06, help="TTX-sensitive coupling after washout (rad/h)")
    p.add_argument("--K-min", type=float, default=None,
                   help="profile: coupling right after TTX application, recovering to K_ttx around --t-rec")
    p.add_argument("--t-rec", type=float, default=72.0, help="profile: recovery time after TTX application (h)")
    p.add_argument("--w-rec", type=float, default=12.0, help="profile: width of the S-shaped recovery (h)")
    p.add_argument("--K-max", type=float, default=None,
                   help="profile: rebound coupling right after washout, decaying to K_wash with --tau-wash")
    p.add_argument("--tau-wash", type=float, default=24.0, help="profile: decay time K_max -> K_wash (h)")
    p.add_argument("--alpha-s", "--alpha", dest="alpha_s", type=float, default=0.0,
                   help="phase lag of the TTX-sensitive coupling (rad)")
    p.add_argument("--K-i", "--K-gap", dest="K_i", type=float, default=0.01, help="TTX-insensitive coupling (rad/h)")
    p.add_argument("--alpha-i", type=float, default=0.0, help="phase lag of the TTX-insensitive coupling (rad)")
    p.add_argument("--insens-global", action="store_true",
                   help="TTX-insensitive coupling to the whole slice instead of the own lobe")
    p.add_argument("--dw-ttx", type=float, default=0.0,
                   help="frequency shift caused by TTX (rad/h, > 0 = faster under TTX); cells run at ω − δω outside TTX")
    p.add_argument("--gamma-ttx", type=float, default=0.0,
                   help="Cauchy phase-noise scale under TTX (rad/h) = extra Lorentzian width")
    p.add_argument("--gamma-q", type=float, default=0.0,
                   help="TTX-induced quenched (Lorentzian) frequency disorder (rad/h), contained in the measured "
                        "distribution: components are narrower by γ_q outside TTX")
    p.add_argument("--reset-ttx", type=float, default=0.0,
                   help="phase scrambling at TTX application: wrapped-Cauchy kick scale σ (rad), z_m → e^{−σ} z_m")
    p.add_argument("--reset-wash", type=float, default=0.0,
                   help="phase scrambling at washout: wrapped-Cauchy kick scale σ (rad)")
    p.add_argument("--lam", type=float, default=1e-6, help="per-component penalty λ of the mixture fits")
    p.add_argument("--no-floor", action="store_true", help="fit without the noise floor")
    p.add_argument("--width-k", type=float, default=10.0, help="width floor k (0 = off)")
    p.add_argument("--rate-ttx", type=float, default=np.inf, help="rate (1/h) of the TTX-application transition")
    p.add_argument("--rate-wash", type=float, default=np.inf, help="rate (1/h) of the washout transition")
    p.add_argument("--t-start", type=float, default=S.EDGE_START_H,
                   help="earliest simulation start (h); the run starts at the first hour from here on at which "
                        "≥ 90 %% of the cells have a phase; initial state = data")
    p.add_argument("--no-ceiling", action="store_true", help="compare |Z| instead of c(t)|Z| with the data")
    p.add_argument("--show", action="store_true", help="open the figure in a window (plt.show) after saving")
    return p.parse_args()


# ─── data and fits ───────────────────────────────────────────────────────────────────────────────
def slice_data(i):
    """Fit target of slice i (scn_targets, cached): ω, lobe, peak phases, R / ceiling of slice and lobes, mask."""
    return TG.load(i)


def fit_mixture(om, lam, floor, width_k, M_max=64):
    mu, sd = om.mean(), om.std()
    x = (om - mu) / sd                                     # fit on standardised frequencies (affine-invariant)
    dmin = 1e-4
    if width_k > 0:
        g = np.linspace(x.min(), x.max(), 2001)
        dmin = max(dmin, width_k / (x.size * float(gaussian_kde(x)(g).max())))
    r = LM.fit(x, (dmin, 1e2), lambda_M=lam, M_max=M_max, floor_c=1.0 if floor else 0.0, patience=3,
               n_restarts=10, seed=1, method="slsqp", large_M=16)
    m = r["model"]
    return dict(w=np.asarray(m.w), Om=mu + sd * np.asarray(m.Omega), De=sd * np.asarray(m.Delta), M=int(m.M))


def build(d, lam, floor, width_k, M_max=64):
    """One mixture per lobe (at most M_max components); global weights = lobe fraction × lobe weights;
    cell -> component assignment."""
    om, lobe = d["om"], d["lobe"]
    Om, De, w, cl, assign = [], [], [], [], np.zeros(om.size, int)
    for k in (0, 1):
        idx = np.where(lobe == k)[0]
        f = fit_mixture(om[idx], lam, floor, width_k, M_max)
        resp = f["w"][None] * f["De"][None] / ((om[idx, None] - f["Om"][None]) ** 2 + f["De"][None] ** 2)
        assign[idx] = len(Om) + np.argmax(resp, axis=1)
        Om += list(f["Om"]); De += list(f["De"]); w += list(idx.size / om.size * f["w"]); cl += [k] * f["M"]
    cl, w = np.array(cl), np.array(w)
    return dict(Om=np.array(Om), De=np.array(De), w=w, lobe=cl, assign=assign,
                p=np.array([w[cl == k].sum() for k in (0, 1)]))


def build_shared(d, f):
    """The SAME mixture f (dict w, Om, De, M; absolute rad/h) for both lobes of the slice: one copy per lobe,
    weights × lobe fraction; each cell assigned to the most likely component of its lobe's copy. Used for
    frequency distributions that are not slice- or lobe-specific (lumped over slices, Gaussian)."""
    om, lobe = d["om"], d["lobe"]
    Om, De, w, cl, assign = [], [], [], [], np.zeros(om.size, int)
    for k in (0, 1):
        idx = np.where(lobe == k)[0]
        resp = f["w"][None] * f["De"][None] / ((om[idx, None] - f["Om"][None]) ** 2 + f["De"][None] ** 2)
        assign[idx] = len(Om) + np.argmax(resp, axis=1)
        Om += list(f["Om"]); De += list(f["De"]); w += list(idx.size / om.size * np.asarray(f["w"]))
        cl += [k] * int(f["M"])
    cl, w = np.array(cl), np.array(w)
    return dict(Om=np.array(Om), De=np.array(De), w=w, lobe=cl, assign=assign,
                p=np.array([w[cl == k].sum() for k in (0, 1)]))


# ─── inputs ──────────────────────────────────────────────────────────────────────────────────────
def coupling_input(K_pre, K_ttx, K_wash, t_ttx, t_wash, rate_ttx, rate_wash):
    """K_syn(t): steps (rate = inf) or first-order relaxations towards the level of each time domain."""
    def K(t):
        if t < t_ttx:
            return K_pre
        if t < t_wash:
            return K_ttx if np.isinf(rate_ttx) else K_ttx + (K_pre - K_ttx) * np.exp(-rate_ttx * (t - t_ttx))
        K0 = K(t_wash - 1e-9)                                  # value reached at washout
        return K_wash if np.isinf(rate_wash) else K_wash + (K0 - K_wash) * np.exp(-rate_wash * (t - t_wash))
    return K


def coupling_profile(K_pre, K_min, K_ttx, t_rec, w_rec, K_max, K_post, tau_wash, t_ttx, t_wash):
    """K_s(t) following the spiking rate: K_pre; at TTX application a drop to K_min with a delayed S-shaped
    recovery to K_ttx (logistic, centre t_rec h after application, width w_rec h); at washout a rebound to K_max
    decaying to K_post (τ_wash)."""
    def K(t):
        if t < t_ttx:
            return K_pre
        if t < t_wash:
            return K_min + (K_ttx - K_min) / (1.0 + np.exp(-(t - t_ttx - t_rec) / w_rec))
        return K_post + (K_max - K_post) * np.exp(-(t - t_wash) / tau_wash)
    return K


def ttx_level(t_ttx, t_wash, rate_ttx, rate_wash):
    """u(t): 0 before TTX, → 1 under TTX, → 0 after washout (steps, or first-order relaxations)."""
    def u(t):
        if t < t_ttx:
            return 0.0
        if t < t_wash:
            return 1.0 if np.isinf(rate_ttx) else 1.0 - np.exp(-rate_ttx * (t - t_ttx))
        u0 = u(t_wash - 1e-9)
        return 0.0 if np.isinf(rate_wash) else u0 * np.exp(-rate_wash * (t - t_wash))
    return u


# ─── LMMF ────────────────────────────────────────────────────────────────────────────────────────
def initial_state(model, PHt):
    """z_m(0) = mean e^{iθ} of the cells assigned to m with a defined phase (else the slice mean field)."""
    ok = np.isfinite(PHt); e = np.exp(1j * np.where(ok, PHt, 0.0)); emean = e[ok].mean()
    return np.array([e[(model["assign"] == m) & ok].mean() if ((model["assign"] == m) & ok).any() else emean
                     for m in range(model["Om"].size)])


def _tabulate(f, tg):
    """f on the grid tg: one vectorised call if f accepts arrays, else point by point."""
    try:
        v = np.asarray(f(tg), float)
        if v.shape == tg.shape:
            return v
    except (ValueError, TypeError):
        pass
    return np.array([f(x) for x in tg], float)


def simulate(model, z0, Kfun, ufun, K_i, alpha_s, alpha_i, insens_global, dw, gamma, t0, t1, gamma_q=0.0,
             resets=(), dt=DT, wfun=None):
    """RK4 (step dt, must divide 1 h) from t0 to t1 (h); hourly times and mean fields (slice, lobe A, lobe B).
    resets: (t_r, σ) pairs; z → e^{−σ} z when t crosses t_r (wrapped-Cauchy phase scrambling).
    wfun: optional extra, component-specific Lorentzian width wfun(t_array) -> (len(t), M) (rad/h), e.g. a
    lobe-specific transient noise.
    K_s(t) and u(t) are tabulated on the half-step grid once (no Python calls inside the integrator)."""
    Om, De = model["Om"], model["De"]; w = model["w"]; cl = model["lobe"]; p = model["p"]; LA = cl == 0
    es, ei = np.exp(-1j * alpha_s), np.exp(-1j * alpha_i)
    sub = int(round(1.0 / dt)); n = int(round(t1 - t0)); ns = n * sub
    tg = t0 + 0.5 * dt * np.arange(2 * ns + 1)
    Kg = _tabulate(Kfun, tg) * es; ug = _tabulate(ufun, tg)
    lin_g = (1j * (Om[None] - dw * (1.0 - ug[:, None]))
             - (np.maximum(De[None] - gamma_q * (1.0 - ug[:, None]), DELTA_MIN) + gamma * ug[:, None]))
    if wfun is not None:
        lin_g = lin_g - np.asarray(wfun(tg), float)
    wA = np.where(LA, w, 0.0) / p[0]; wB = np.where(LA, 0.0, w) / p[1]
    Kie = K_i * ei
    kicks = {}
    for tr, sig in resets:
        if sig:
            k = int(np.ceil((tr - t0) / dt - 1e-9))
            if 1 <= k <= ns:
                kicks[k] = kicks.get(k, 1.0) * np.exp(-sig)

    def F(j, z):
        Z = w @ z
        Zi = Z if insens_global else np.where(LA, wA @ z, wB @ z)
        H = Kg[j] * Z + Kie * Zi
        return lin_g[j] * z + 0.5 * (H - np.conj(H) * z * z)

    z = z0.astype(complex); out = np.empty((n + 1, z.size), complex); out[0] = z
    for s_ in range(ns):
        j = 2 * s_
        k1 = F(j, z); k2 = F(j + 1, z + dt / 2 * k1); k3 = F(j + 1, z + dt / 2 * k2); k4 = F(j + 2, z + dt * k3)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s_ + 1 in kicks:
            z = z * kicks[s_ + 1]
        if (s_ + 1) % sub == 0:
            out[(s_ + 1) // sub] = z
    return t0 + np.arange(n + 1), out @ w, out @ wA, out @ wB


# ─── main ────────────────────────────────────────────────────────────────────────────────────────
def main():
    a = parse()
    d = slice_data(a.slice); T = int(d["T"]); om = d["om"]; floor = not a.no_floor
    model = build(d, a.lam, floor, a.width_k)
    profile = a.K_min is not None or a.K_max is not None
    if profile:
        Kfun = coupling_profile(a.K_pre, a.K_ttx if a.K_min is None else a.K_min, a.K_ttx, a.t_rec, a.w_rec,
                                a.K_wash if a.K_max is None else a.K_max, a.K_wash, a.tau_wash, d["ttx"], d["wash"])
    else:
        Kfun = coupling_input(a.K_pre, a.K_ttx, a.K_wash, d["ttx"], d["wash"], a.rate_ttx, a.rate_wash)
    ufun = ttx_level(d["ttx"], d["wash"], a.rate_ttx, a.rate_wash)
    v0, v1 = S.valid_window(T); t0 = TG.start_hour(d, int(round(a.t_start)))   # first hour with ≥ 90 % phases
    tm, Zg, ZA, ZB = simulate(model, initial_state(model, d["PH"][t0]), Kfun, ufun, a.K_i, a.alpha_s, a.alpha_i,
                              a.insens_global, a.dw_ttx, a.gamma_ttx, t0, v1 - 1, a.gamma_q,
                              resets=((d["ttx"], a.reset_ttx), (d["wash"], a.reset_wash)))
    if a.gamma_q > model["De"].min():
        print(f"[warning] γ_q = {a.gamma_q} exceeds the narrowest component width {model['De'].min():.4f} rad/h: "
              f"widths are clipped at {DELTA_MIN} outside TTX")
    # data coherence (target pipeline, valid window); model as c(t)|Z| unless --no-ceiling; RMSE on the mask
    def vw(x):
        x = np.array(x, float); x[:v0] = np.nan; x[v1:] = np.nan
        return x
    Rd, RdA, RdB = vw(d["R"]), vw(d["RA"]), vw(d["RB"])
    ti = tm.astype(int); one = np.ones(T)
    cg, cA, cB = (one, one, one) if a.no_ceiling else (d["c"], d["cA"], d["cB"])
    Mg, MA, MB = cg[ti] * np.abs(Zg), cA[ti] * np.abs(ZA), cB[ti] * np.abs(ZB)
    wm = d["mask"][ti] > 0
    rmse = lambda Mm, Rdat: np.sqrt(np.nanmean((Mm[wm] - Rdat[ti][wm]) ** 2))
    print(f"[SCN{a.slice}] n = {om.size} cells (lobes {np.bincount(d['lobe']).tolist()}), mixtures M = "
          f"{np.bincount(model['lobe']).tolist()} ({'floor' if floor else 'no floor'}, λ = {a.lam:g}); "
          f"K_s {a.K_pre}/{a.K_ttx}/{a.K_wash} (α_s {a.alpha_s}), K_i {a.K_i} (α_i {a.alpha_i}, "
          f"{'slice' if a.insens_global else 'lobe'}), δω {a.dw_ttx}, γ {a.gamma_ttx}, γ_q {a.gamma_q}, "
          f"reset σ {a.reset_ttx}/{a.reset_wash}, rates "
          f"{a.rate_ttx}/{a.rate_wash} 1/h; RMSE(R, mask) slice {rmse(Mg, Rd):.3f}, lobes {rmse(MA, RdA):.3f}/"
          f"{rmse(MB, RdB):.3f}")

    # figure (two-column)
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")                               # no display needed when only saving
    import matplotlib.pyplot as plt
    try:
        from prl_style import set_prl_style
        set_prl_style("prl")
    except Exception:
        pass
    c_fit = "#eb6834" if floor else "#2a78d6"; CL = ["#1b7837", "#762a83"]
    fig = plt.figure(figsize=(7.0, 3.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.2], height_ratios=[1.0, 0.32])
    ax0 = fig.add_subplot(gs[:, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    cd = 24 / (2 * np.pi)                                   # rad/h -> cycles/day
    g = np.linspace(om.min() - 0.01, om.max() + 0.01, 800)
    for k in (0, 1):
        s = model["lobe"] == k; ww = model["w"][s] / model["w"][s].sum()
        pdf = (ww[:, None] * model["De"][s, None] / np.pi / ((g[None] - model["Om"][s, None]) ** 2 + model["De"][s, None] ** 2)).sum(0)
        ax0.hist(om[d["lobe"] == k] * cd, bins=30, density=True, color=CL[k], alpha=0.3,
                 label=f"lobe {'AB'[k]} (n = {(d['lobe'] == k).sum()})")
        ax0.plot(g * cd, pdf / cd, color=CL[k], lw=1.1, label=f"fit, M = {s.sum()}")
    ax0.set_xlabel("intrinsic frequency (cycles/day)"); ax0.set_ylabel("density"); ax0.set_yticks([])
    ax0.legend(fontsize=5.5, frameon=False, loc="upper left")
    td = np.arange(T) / 24
    for ax in (ax1, ax2):
        ax.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
    for t_c in (d["ttx"], d["wash"]):                         # masked windows around the treatment changes
        ax1.axvspan((t_c - d["mask_h"]) / 24, (t_c + d["mask_h"]) / 24, color="none", hatch="////", ec="0.75",
                    lw=0, zorder=0)
    ax1.plot(td, Rd, color="0.1", lw=1.2, label="data, slice (epoch peak)")
    ax1.plot(tm / 24, Mg, color=c_fit, lw=1.1, label="LMMF, slice" + ("" if a.no_ceiling else r", $c(t)|Z|$"))
    if not a.no_ceiling:
        ax1.plot(td, vw(d["c"]), color="0.6", lw=0.7, ls=":", label="ceiling $c(t)$")
    for k, (Rdat, Mm) in enumerate(((RdA, MA), (RdB, MB))):
        ax1.plot(td, Rdat, color=CL[k], lw=0.6, alpha=0.8)
        ax1.plot(tm / 24, Mm, color=CL[k], lw=0.7, ls="--")
    ax1.set_ylim(0, 1.02); ax1.set_ylabel(r"$R(t)$"); ax1.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax1.set_title(rf"SCN slice {a.slice}: $\alpha_s={a.alpha_s:g}$, $\alpha_i={a.alpha_i:g}$, $\lambda={a.lam:g}$; "
                  rf"RMSE slice {rmse(Mg, Rd):.3f} (lobes: thin = data, dashed = LMMF; hatched = masked)", fontsize=7)
    tt = np.arange(0, T, 0.25)
    ax2.plot(tt / 24, [Kfun(x) for x in tt], color="0.3", lw=1.0, label=r"$K_s(t)$")
    ax2.axhline(a.K_i, color="0.3", lw=0.9, ls=":", label=r"$K_i$")
    uu = np.array([ufun(x) for x in tt])
    if a.dw_ttx:
        ax2.plot(tt / 24, -a.dw_ttx * (1 - uu), color="#b2182b", lw=0.9, label=r"$-\delta\omega(1-u)$")
    if a.gamma_ttx:
        ax2.plot(tt / 24, a.gamma_ttx * uu, color="#2166ac", lw=0.9, label=r"$\gamma u$")
    if a.gamma_q:
        ax2.plot(tt / 24, -a.gamma_q * (1 - uu), color="#5aae61", lw=0.9, label=r"$-\gamma_q(1-u)$")
    ax2.set_ylabel("inputs (rad/h)"); ax2.set_xlabel("time (days)"); ax2.set_xlim(0, T / 24)
    ax2.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=4)
    plt.setp(ax1.get_xticklabels(), visible=False)
    tag = (f"Kpre{a.K_pre:g}_Kttx{a.K_ttx:g}_Kwash{a.K_wash:g}_as{a.alpha_s:g}_Ki{a.K_i:g}_ai{a.alpha_i:g}"
           + (f"_prof{a.K_min if a.K_min is not None else a.K_ttx:g}-{a.t_rec:g}-{a.w_rec:g}-"
              f"{a.K_max if a.K_max is not None else a.K_wash:g}-{a.tau_wash:g}" if profile else "")
           + ("_iglob" if a.insens_global else "")
           + (f"_dw{a.dw_ttx:g}" if a.dw_ttx else "") + (f"_g{a.gamma_ttx:g}" if a.gamma_ttx else "")
           + (f"_gq{a.gamma_q:g}" if a.gamma_q else "")
           + (f"_rs{a.reset_ttx:g}-{a.reset_wash:g}" if a.reset_ttx or a.reset_wash else "")
           + f"_lam{a.lam:g}_{'floor' if floor else 'nofloor'}"
           + ("" if np.isinf(a.rate_ttx) and np.isinf(a.rate_wash) else f"_r{a.rate_ttx:g}-{a.rate_wash:g}")
           + ("_noceil" if a.no_ceiling else ""))
    out = os.path.join(OUT, f"scn{a.slice}_{tag}")
    fig.savefig(out + ".png", dpi=250); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
