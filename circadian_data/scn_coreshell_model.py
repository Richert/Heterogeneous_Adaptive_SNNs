r"""
Two-ensemble (core / shell) Lorentzian-mixture mean field of the SCN slices, fitted to core and shell coherence
==============================================================================================================

Ensembles (scn_coreshell_targets, lobes pooled): CORE = inner half of each lobe, SHELL = outer half. Each ensemble e has
its own noise-floor Lorentzian mixture fitted to its cells' TTX phase slopes (centres and shape from the slopes); the
widths are scaled by a fitted factor s_e, so that the WIDTHS are set by the dephasing under TTX (per-cell slopes carry a
non-accumulating peak-timing error of the same size as the spread; scn_frequency_inference_tests). There is no separate
noise term (it would count the frequency spread twice). Component m of ensemble e(m):

    ż_m = (iΩ_m − s_e Δ_m) z_m + ½ (H_e − H̄_e z_m²)
    H_core  = e^{−iα} ρ(t) K_cc (Z_c + ε Z_s) + K_g Z_c
    H_shell = e^{−iα} ρ(t) K_cs (Z_c + ε Z_s) + K_g Z_s

  synaptic (VIP/GABA, activity-dependent, TTX-SENSITIVE): broadcast by the core (core→core K_cc incl. the other lobe's
  core, core→shell K_cs); the sparsely connected shell sends a fraction ε of it (Abel et al. 2016: hubs in the core,
  shell cells with few links; their SI simulates unconnected cells with weakened mean-field coupling).
  K_g: TTX-INSENSITIVE local coupling within each ensemble (gap junctions).
  AMPLITUDE-LINKED PHASE NOISE (2026-11-18): every component of ensemble e gets the extra width D₀ q_e(t),
  q_e(t) = (A_e,pre / A_e(t))², A_e(t) = the ensemble's median single-cell amplitude (rolling half-range of the
  epoch-wise detrended trace, AMP_WIN h, relative to each cell's pre-TTX level), so phase diffusion grows as the
  amplitudes collapse under TTX (weak, noisy oscillators; Abel et al. 2016, Webb et al. 2009). D₀ is SHARED by all
  slices and FIXED per run (--d0; chosen by a scan), so noise cannot replace the TTX block in a single slice.
  TTX time course (user, 2026-11-18), one multiplier ρ(t) of all TTX-sensitive coupling, relative to the pre-TTX level:
      t < t_ttx                ρ = 1
      t_ttx ≤ t < t_rec        ρ = f_ttx                          (block, instantaneous)
      t_rec ≤ t < t_wash       linear ramp f_ttx → f_wash          (recovery onset t_rec: under TTX or at washout)
      t ≥ t_wash               ρ = f_post + (f_wash − f_post) e^{−(t − t_wash)/τ_post}   (f_wash > f_post: rebound)
  t_rec = t_ttx + 24 h + u_rec (t_wash − t_ttx − 24 h), u_rec ∈ [0, 1].

Models (D₀ fixed):
  sym     no directed structure: K_cc = K_cs = K, ε = 1, s_c = s_s = s  (9: K, K_g, α, s, f_ttx, u_rec, f_wash, f_post,
                                                                          τ_post)
  dir     full                                                       (12: K_cc, K_cs, ε, K_g, α, s_c, s_s, f_ttx, u_rec,
                                                                          f_wash, f_post, τ_post)
Loss over the mask of scn_targets: mean of the squared errors of c_c|Z_c| − R_c, c_s|Z_s| − R_s and c|Z| − R (slice),
+ the core–shell phase difference Δψ (wrapped, where both R > PSI_RMIN, scale SIGMA_PSI) + the collective frequency
term of scn_model_fit (pre-TTX and post-washout stretches). Start from the measured state (cells assigned to the most
likely component of their ensemble). Nelder–Mead in a transformed (log / logit) space from N_STARTS starts, parallel.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_coreshell_model.py time fit figure [--slices 1,2,3,4,5]
        [--models sym,dir] [--d0 5e-4] [--starts 32] [--procs 64]
    SCAN (server): dir for every D₀ in --d0s, then sym at the D₀ with the lowest total dir loss, then the figure:
        python scn_coreshell_model.py scan --d0s 2e-4,5e-4,1e-3 [--starts 32] [--procs 64]
All (slice × start) local searches of one model run in ONE pool (use all cores), then one polish per slice (pool).
Output: data_paths.MPMF/circadian_data/coreshell/{scn<i>_<model>_d<D0>_fit.npy, scn_coreshell_fits_d<D0>.{png,pdf},
        coreshell_summary_d<D0>.json, d0_scan.json}
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import scn_common as S
import scn_targets as TG
import scn_model_fit as F
import scn_lmmf_sim as SIM
import scn_coreshell_targets as CT

OUT = CT.OUT
DT = 0.5
PSI_RMIN, SIGMA_PSI = 0.3, 0.15          # rad (≈ 0.6 h)
AMP_WIN = 24                             # h, rolling window of the single-cell amplitude
Q_MAX = 100.0                            # cap of (A_pre / A)² (amplitudes near the noise floor)
CPD = 24 / (2 * np.pi)
# parameter: (lower, upper, transform)  log: positive with bounds, lin: bounded linear
PAR = dict(K=(1e-3, 5.0, "log"), K_cc=(1e-3, 5.0, "log"), K_cs=(1e-3, 5.0, "log"), eps=(1e-3, 1.0, "log"),
           K_g=(1e-5, 0.3, "log"), alpha=(-0.6, 0.6, "lin"), s=(0.05, 2.0, "log"), s_c=(0.05, 2.0, "log"),
           s_s=(0.05, 2.0, "log"), f_ttx=(1e-4, 1.0, "log"), u_rec=(0.0, 1.0, "lin"), f_wash=(0.02, 10.0, "log"),
           f_post=(0.02, 10.0, "log"), tau_post=(0.5, 200.0, "log"))
TC = ("f_ttx", "u_rec", "f_wash", "f_post", "tau_post")
MODELS = dict(sym=("K", "K_g", "alpha", "s") + TC,
              dir=("K_cc", "K_cs", "eps", "K_g", "alpha", "s_c", "s_s") + TC)
START = dict(K=0.2, K_cc=0.3, K_cs=0.15, eps=0.3, K_g=0.01, alpha=0.0, s=0.6, s_c=0.6, s_s=0.6, f_ttx=0.05,
             u_rec=0.8, f_wash=1.5, f_post=0.8, tau_post=24.0)
D0 = [5e-4]                               # shared, fixed amplitude-noise strength (set by --d0 / scan)
MAXITER_PER_DIM = [300]                   # Nelder–Mead evaluations per free parameter (--maxiter-per-dim)


def d0_tag(d0=None):
    return f"d{(D0[0] if d0 is None else d0):g}"


def to_x(p, keys):
    x = []
    for k in keys:
        lo, hi, tr = PAR[k]; v = np.clip(p[k], lo * (1 + 1e-9), hi * (1 - 1e-9))
        u = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if tr == "log" else (v - lo) / (hi - lo)
        x.append(np.log(u / (1 - u)))
    return np.array(x)


def from_x(x, keys):
    p = {}
    for k, xi in zip(keys, x):
        lo, hi, tr = PAR[k]; u = 1 / (1 + np.exp(-xi))
        p[k] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if tr == "log" else float(lo + u * (hi - lo))
    return p


def full(p):
    """Expand a model's parameters to the general (dir) set."""
    if "K" in p:
        q = dict(K_cc=p["K"], K_cs=p["K"], eps=1.0, K_g=p["K_g"], alpha=p["alpha"], s_c=p["s"], s_s=p["s"])
        q.update({k: p[k] for k in TC})
        return q
    return dict(p)


# ─── data and mixtures ───────────────────────────────────────────────────────────────────────────
def amplitude_q(i, g, ens):
    """q_e(t) = (A_e,pre / A_e(t))² per ensemble (T, 2): A = median over the ensemble's cells of the rolling half-range
    ((p95 − p5)/2 over AMP_WIN h, within each epoch) of the epoch-wise detrended trace, each cell relative to its
    pre-TTX median; the ensemble curve is smoothed over AMP_WIN h. Cached."""
    f = os.path.join(OUT, f"scn{i}_amplitude_q.npy")
    if os.path.exists(f):
        return np.load(f)
    from scipy.ndimage import percentile_filter, uniform_filter1d
    import scn_epoch_processing as EP
    X, _ = S.load(i); X = X[:, g["cells"]]; T = X.shape[0]
    amp = np.empty_like(X)
    for a, b in EP.bounds(T, int(g["ttx"]), int(g["wash"])):
        seg = X[a:b]; y = seg - EP.baseline(seg, TG.PIPE["baseline_kind"])
        hi = percentile_filter(y, 95, size=(AMP_WIN, 1), mode="reflect")
        lo = percentile_filter(y, 5, size=(AMP_WIN, 1), mode="reflect")
        amp[a:b] = 0.5 * (hi - lo)
    v0 = S.valid_window(T)[0]
    rel = amp / np.median(amp[v0:int(g["ttx"])], axis=0)[None]
    q = np.empty((T, 2))
    for e in (0, 1):
        A = uniform_filter1d(np.median(rel[:, ens == e], axis=1), AMP_WIN, mode="nearest")
        q[:, e] = np.clip(1.0 / np.maximum(A, 1e-6) ** 2, 0.0, Q_MAX)
    np.save(f, q)
    return q


_CTX = {}


def context(i):
    if i in _CTX:
        return _CTX[i]
    g = TG.load(i); ct = CT.load(i); ens = ct["ens"]; om = g["om"]
    Om, De, w, ce, assign = [], [], [], [], np.zeros(om.size, int)
    for e in (0, 1):
        idx = np.where(ens == e)[0]
        f = os.path.join(OUT, f"scn{i}_ens{e}_floor_mixture.npy")
        if os.path.exists(f):
            mix = np.load(f, allow_pickle=True).item()
        else:
            mix = SIM.fit_mixture(om[idx], F.LAM, True, F.WIDTH_K); np.save(f, mix, allow_pickle=True)
        resp = mix["w"][None] * mix["De"][None] / ((om[idx, None] - mix["Om"][None]) ** 2 + mix["De"][None] ** 2)
        assign[idx] = len(Om) + np.argmax(resp, axis=1)
        Om += list(mix["Om"]); De += list(mix["De"]); w += list(idx.size / om.size * np.asarray(mix["w"]))
        ce += [e] * int(mix["M"])
    model = dict(Om=np.array(Om), De=np.array(De), w=np.array(w), ens=np.array(ce), assign=assign)
    model["p"] = np.array([model["w"][model["ens"] == e].sum() for e in (0, 1)])
    prep = F.prepare(g)
    t0, t1 = prep["t0"], prep["t1"]
    z0 = SIM.initial_state(model, g["PH"][t0])
    ti = np.arange(t0, t1 + 1)
    model["q"] = amplitude_q(i, g, ens)
    _CTX[i] = (g, ct, model, prep, z0, ti)
    return _CTX[i]


def t_rec(p, ttx, wash):
    return ttx + S.TRANSIENT_H + p["u_rec"] * (wash - ttx - S.TRANSIENT_H)


def rho_t(p, tg, ttx, wash):
    """Coupling multiplier: 1 | f_ttx | linear ramp f_ttx → f_wash from t_rec to t_wash | relaxation to f_post."""
    tr = t_rec(p, ttx, wash)
    r = np.ones_like(tg)
    r[(tg >= ttx) & (tg < tr)] = p["f_ttx"]
    a = (tg >= tr) & (tg < wash)
    r[a] = p["f_ttx"] + (p["f_wash"] - p["f_ttx"]) * (tg[a] - tr) / max(wash - tr, 1e-9)
    b = tg >= wash
    r[b] = p["f_post"] + (p["f_wash"] - p["f_post"]) * np.exp(-(tg[b] - wash) / p["tau_post"])
    return r


def simulate(p, i):
    """Hourly Z_c, Z_s, Z over [t0, t1]."""
    g, ct, model, prep, z0, ti = context(i)
    p = full(p)
    Om, De, w, ens = model["Om"], model["De"], model["w"], model["ens"]
    C = ens == 0; wc = np.where(C, w, 0) / model["p"][0]; ws = np.where(C, 0, w) / model["p"][1]
    lin = 1j * Om - np.where(C, p["s_c"], p["s_s"]) * De
    Ksyn = np.where(C, p["K_cc"], p["K_cs"]) * np.exp(-1j * p["alpha"]); eps = p["eps"]; Kg = p["K_g"]
    t0, t1 = int(ti[0]), int(ti[-1]); n = t1 - t0; sub = int(round(1 / DT))
    tg = t0 + 0.5 * DT * np.arange(2 * n * sub + 1)
    rg = rho_t(p, tg, float(g["ttx"]), float(g["wash"]))
    th = np.arange(model["q"].shape[0], dtype=float)
    qg = np.stack([np.interp(tg, th, model["q"][:, e]) for e in (0, 1)], axis=1)       # (n_half, 2)
    lin_g = lin[None] - D0[0] * np.where(C[None], qg[:, :1], qg[:, 1:])                 # (n_half, M)

    def f(j, z):
        Zc = wc @ z; Zs = ws @ z
        H = rg[j] * Ksyn * (Zc + eps * Zs) + Kg * np.where(C, Zc, Zs)
        return lin_g[j] * z + 0.5 * (H - np.conj(H) * z * z)
    z = z0.astype(complex).copy(); out = np.empty((n + 1, z.size), complex); out[0] = z
    for s_ in range(n * sub):
        j = 2 * s_
        k1 = f(j, z); k2 = f(j + 1, z + 0.5 * DT * k1); k3 = f(j + 1, z + 0.5 * DT * k2); k4 = f(j + 2, z + DT * k3)
        z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (s_ + 1) % sub == 0:
            out[(s_ + 1) // sub] = z
    return out @ wc, out @ ws, out @ w


def loss(x, keys, i, detail=False):
    g, ct, model, prep, z0, ti = context(i)
    p = from_x(x, keys)
    Zc, Zs, Z = simulate(p, i)
    if not (np.all(np.isfinite(Zc)) and np.all(np.isfinite(Zs))):
        return (1e3, None) if detail else 1e3
    msk = g["mask"][ti] > 0
    ec = (ct["cc"][ti] * np.abs(Zc) - ct["Rc"][ti]) ** 2
    es = (ct["cs"][ti] * np.abs(Zs) - ct["Rs"][ti]) ** 2
    eg = (g["c"][ti] * np.abs(Z) - g["R"][ti]) ** 2
    e = np.where(msk, (ec + es + eg) / 3, 0.0)
    n = msk.sum()
    dps_m = np.angle(Zs * np.conj(Zc)); dps_d = ct["dpsi"][ti]
    pm = msk & (ct["Rc"][ti] > PSI_RMIN) & (ct["Rs"][ti] > PSI_RMIN) & np.isfinite(dps_d)
    epsi = np.angle(np.exp(1j * (dps_m - dps_d)))[pm] ** 2 / SIGMA_PSI ** 2
    fterm = sum(((F.collective_frequency(Z[a - ti[0]:b - ti[0]]) - nu) / F.SIGMA_NU) ** 2
                for (a, b), nu in zip(prep["stretches"], prep["nu"]))
    L = (e.sum() + 1e-3 * epsi.sum() + fterm) / n
    if not np.isfinite(L):
        return (1e3, None) if detail else 1e3
    if detail:
        rm = lambda a: float(np.sqrt(np.mean(a[msk])))
        return L, dict(params=p, Zc=Zc, Zs=Zs, Z=Z, ti=ti, rmse_c=rm(ec), rmse_s=rm(es), rmse_g=rm(eg),
                       psi_rms_h=float(np.sqrt(np.mean(np.angle(np.exp(1j * (dps_m - dps_d)))[pm] ** 2)) * CPD)
                       if pm.any() else np.nan)
    return L


def _init(d0):
    D0[0] = d0


def _job(args):
    i, keys, x0, maxiter, xatol, fatol = args
    r = minimize(loss, x0, args=(keys, i), method="Nelder-Mead",
                 options=dict(maxiter=maxiter, maxfev=maxiter, xatol=xatol, fatol=fatol, adaptive=True))
    return i, r.x, float(r.fun)


def fit(slices, models, n_starts, procs, d0):
    """All (slice × start) local searches of a model in one pool, then one polish per slice (also in the pool)."""
    D0[0] = d0
    for i in slices:                                       # build / cache mixtures and amplitudes before forking
        context(i)
    out = {}
    for name in models:
        keys = MODELS[name]; t_ = time.time(); mi = MAXITER_PER_DIM[0] * len(keys)
        rng = np.random.default_rng(3)
        x_start = to_x(START, keys)
        jobs = []
        for i in slices:
            X0 = [x_start] + [x_start + rng.normal(0, 1.2, len(keys)) for _ in range(n_starts - 1)]
            jobs += [(i, keys, x0, mi, 1e-4, 1e-7) for x0 in X0]
        with Pool(procs, initializer=_init, initargs=(d0,)) as pool:
            R = pool.map(_job, jobs, chunksize=1)
            best = {}
            for i, x, L in R:
                if i not in best or L < best[i][1]:
                    best[i] = (x, L)
            P = pool.map(_job, [(i, keys, best[i][0], 2 * mi, 1e-5, 1e-8) for i in slices], chunksize=1)
        for i, xb, _ in P:
            L, det = loss(xb, keys, i, detail=True)
            det.update(slice=i, model=name, keys=keys, x=xb, loss=L, n_free=len(keys), D0=d0,
                       t_rec_h=t_rec(det["params"], float(TG.load(i)["ttx"]), float(TG.load(i)["wash"])),
                       start_losses=sorted(r[2] for r in R if r[0] == i))
            np.save(os.path.join(OUT, f"scn{i}_{name}_{d0_tag(d0)}_fit.npy"), det, allow_pickle=True)
            out[(name, i)] = L
            print(f"[fit D0={d0:g}] SCN{i} {name}: loss {L * 1e3:.3f}e-3 | RMSE core {det['rmse_c']:.3f} shell "
                  f"{det['rmse_s']:.3f} slice {det['rmse_g']:.3f} | Δψ rms {det['psi_rms_h']:.2f} h | t_rec − t_wash "
                  f"{det['t_rec_h'] - float(TG.load(i)['wash']):.0f} h | "
                  + ", ".join(f"{k}={v:.3g}" for k, v in det["params"].items()), flush=True)
        print(f"[fit D0={d0:g}] {name} done in {time.time() - t_:.0f} s", flush=True)
    return out


def scan(slices, d0s, n_starts, procs):
    res = {}
    for d0 in d0s:
        L = fit(slices, ["dir"], n_starts, procs, d0)
        res[f"{d0:g}"] = dict(total=float(sum(L.values())), per_slice={str(i): float(L[("dir", i)]) for i in slices})
        json.dump(res, open(os.path.join(OUT, "d0_scan.json"), "w"), indent=1)
        print(f"[scan] D0 = {d0:g}: total dir loss {res[f'{d0:g}']['total'] * 1e3:.3f}e-3", flush=True)
    best = min(d0s, key=lambda d: res[f"{d:g}"]["total"])
    print(f"[scan] best D0 = {best:g}; fitting sym there", flush=True)
    fit(slices, ["sym"], n_starts, procs, best)
    D0[0] = best
    figure(slices, ["sym", "dir"])


def figure(slices, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, len(slices), figsize=(3.1 * len(slices), 7.2), sharex="col", squeeze=False,
                           gridspec_kw=dict(hspace=0.12, wspace=0.3, left=0.06, right=0.99, top=0.92, bottom=0.07))
    ls = dict(sym=":", dir="-")
    summ = {}
    for col, i in enumerate(slices):
        g, ct, model, prep, z0, ti = context(i)
        T = int(g["T"]); td = np.arange(T) / 24
        for r_ in range(3):
            ax[r_, col].axvspan(g["ttx"] / 24, g["wash"] / 24, color="0.9", lw=0)
        ax[0, col].plot(td, ct["Rc"], color="#08519c", lw=1.2, label="core (data)")
        ax[0, col].plot(td, ct["Rs"], color="#e6550d", lw=1.2, label="shell (data)")
        ax[1, col].plot(td, g["R"], color="k", lw=1.2, label="slice (data)")
        ax[2, col].plot(td, ct["dpsi"] * CPD, color="k", lw=1.0, label="data")
        for name in models:
            f = os.path.join(OUT, f"scn{i}_{name}_{d0_tag()}_fit.npy")
            if not os.path.exists(f):
                continue
            r = np.load(f, allow_pickle=True).item(); tt = r["ti"] / 24
            ax[0, col].plot(tt, ct["cc"][r["ti"]] * np.abs(r["Zc"]), color="#08519c", ls=ls[name], lw=1.6)
            ax[0, col].plot(tt, ct["cs"][r["ti"]] * np.abs(r["Zs"]), color="#e6550d", ls=ls[name], lw=1.6)
            ax[1, col].plot(tt, g["c"][r["ti"]] * np.abs(r["Z"]), color="#c0392b", ls=ls[name], lw=1.6,
                            label=f"model {name}")
            ax[2, col].plot(tt, np.angle(r["Zs"] * np.conj(r["Zc"])) * CPD, color="#c0392b", ls=ls[name], lw=1.4)
            summ.setdefault(str(i), {})[name] = dict(loss=r["loss"], rmse=[r["rmse_c"], r["rmse_s"], r["rmse_g"]],
                                                     psi_rms_h=r["psi_rms_h"], params=r["params"], n_free=r["n_free"])
        ax[0, col].set_title(f"SCN{i}"); ax[0, col].set_ylim(0, 1.02); ax[1, col].set_ylim(0, 1.02)
        ax[2, col].set_ylim(-6, 6); ax[2, col].axhline(0, color="0.6", lw=0.5); ax[2, col].set_xlabel("time (days)")
        if col == 0:
            ax[0, col].set_ylabel("core / shell R"); ax[1, col].set_ylabel("slice R")
            ax[2, col].set_ylabel("shell − core phase (h)")
            ax[0, col].legend(frameon=False, fontsize=7); ax[1, col].legend(frameon=False, fontsize=7)
    fig.suptitle(f"two-ensemble LMMF, D₀ = {D0[0]:g}: data (thin) vs model (dotted: symmetric, solid: directed); "
                 "model shown as c|Z|", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"scn_coreshell_fits_{d0_tag()}.{ext}"), dpi=180)
    json.dump(summ, open(os.path.join(OUT, f"coreshell_summary_{d0_tag()}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")])
    models = arg("--models", ["sym", "dir"], lambda s_: s_.split(","))
    D0[0] = arg("--d0", D0[0], float)
    MAXITER_PER_DIM[0] = arg("--maxiter-per-dim", MAXITER_PER_DIM[0], int)
    procs, starts = arg("--procs", os.cpu_count(), int), arg("--starts", 32, int)
    if "time" in sys.argv:
        for i in slices[:1]:
            context(i); x = to_x(START, MODELS["dir"]); t_ = time.time()
            for _ in range(5):
                L = loss(x, MODELS["dir"], i)
            print(f"[time] SCN{i}: {(time.time() - t_) / 5 * 1e3:.1f} ms per loss evaluation (M = "
                  f"{context(i)[2]['Om'].size}); start loss {L:.4g} at D0 = {D0[0]:g}")
    if "scan" in sys.argv:
        scan(slices, arg("--d0s", [2e-4, 5e-4, 1e-3], lambda s_: [float(v) for v in s_.split(",")]), starts, procs)
    if "fit" in sys.argv:
        fit(slices, models, starts, procs, D0[0])
    if "figure" in sys.argv:
        figure(slices, models)
