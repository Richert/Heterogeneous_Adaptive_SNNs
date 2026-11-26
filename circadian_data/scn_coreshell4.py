r"""
Four-ensemble (core / shell × left / right lobe) Lorentzian-mixture mean field of the SCN slices
================================================================================================

Ensembles e ∈ {core A, shell A, core B, shell B}: the mirror-symmetric planar (or offset-radial) core/shell split of
scn_coreshell_geometry (shell = the faster, sparsely connected group), applied per lobe. Each ensemble has its own FIXED
noise-floor Lorentzian mixture of its cells' TTX frequencies (epoch-wise peaks with cycle-slip handling, naive fit,
scn_deconvolution.fit_mixture) — no width factors, no noise term: the frequency distributions are the primary result
(scn_frequency_figure, scn_core_shell_geometry) and only the COUPLING is fitted here.

    ż_m = (iΩ_m − Δ_m − γ_tr(t)) z_m + ½ (H_e − H̄_e z_m²)
    H_core,L  = ρ(t) e^{−iα} [ K_cc · ½ (Z_core,A + Z_core,B) + ε K_cs Z_shell,L ]
    H_shell,L = ρ(t) e^{−iα} [ K_cs Z_core,L + K_ss Z_shell,L ]
  Literature (SCN_REVIEW.md): dense core → shell projection (K_cs), sparse shell → core (ε K_cs, ε ∈ [0, EPS_MAX]),
  dense within-core coupling incl. the symmetric left–right core–core link (same K_cc), weaker within-shell coupling
  (K_ss); no gap-junction term (weak, and dye coupling is reduced by TTX). All synaptic coupling is TTX-SENSITIVE
  through ONE multiplier ρ(t) (user's time course): 1 before TTX | f_ttx (residual coupling = imperfect block) |
  linear ramp to f_wash from the recovery onset t_rec = t_ttx + 24 h + u_rec (t_wash − t_ttx − 24 h) to washout |
  relaxation from f_wash to f_post with τ_post = TAU_POST after washout.
Models: "base" (9 parameters: K_cc, K_cs, K_ss, ε, α, f_ttx, u_rec, f_wash, f_post);
        "tr" = base + a TTX-onset dephasing transient γ_tr(t) = g_tr e^{−(t − t_ttx)/τ_tr} (t ≥ t_ttx), shared by all
        components (11 parameters) — an explicitly labelled extra TTX effect, used only if base fails.

Targets (cached, coreshell4/scn<i>_target4.npz): R(t) of the four ensembles and of the slice, the core–shell phase
difference per lobe, measurement ceilings c_e(t) per ensemble (perfectly synchronised synthetic traces through the same
pipeline, as scn_targets), the mask of scn_targets. Loss over the mask: mean squared error of c_e|Z_e| − R_e over the
four ensembles and the slice + the sampling-error-weighted core–shell phase term per lobe (as scn_coreshell_model) +
the collective-frequency term of scn_model_fit.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_coreshell4.py targets time fit figure [--slices 1,2,3,4,5]
        [--models base,tr] [--starts 32] [--procs 64]
Output: data_paths.MPMF/circadian_data/coreshell4/
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_epoch_processing as EP
import scn_model_fit as F
import scn_lmmf_sim as SIM

OUT = dp.ensure(os.path.join(S.OUT, "coreshell4"))
ENS = ("core A", "shell A", "core B", "shell B")
DT = 0.5
TAU_POST = 2.0
EPS_MAX = 0.3
SIG_R = 0.05
SIG0 = 0.5 * 2 * np.pi / 24
PSI_RMIN = 0.3
CPD = 24 / (2 * np.pi)
PAR = dict(K_cc=(1e-3, 5.0, "log"), K_cs=(1e-3, 5.0, "log"), K_ss=(1e-4, 5.0, "log"), eps=(0.0, EPS_MAX, "lin"),
           alpha=(-0.6, 0.6, "lin"), f_ttx=(1e-4, 1.0, "log"), u_rec=(0.0, 1.0, "lin"), f_wash=(0.02, 5.0, "log"),
           f_post=(0.02, 10.0, "log"), g_tr=(1e-5, 0.3, "log"), tau_tr=(1.0, 72.0, "log"))
BASE = ("K_cc", "K_cs", "K_ss", "eps", "alpha", "f_ttx", "u_rec", "f_wash", "f_post")
MODELS = dict(base=BASE, tr=BASE + ("g_tr", "tau_tr"))
START = dict(K_cc=0.3, K_cs=0.15, K_ss=0.05, eps=0.05, alpha=0.0, f_ttx=0.05, u_rec=0.8, f_wash=1.5, f_post=0.8,
             g_tr=0.01, tau_tr=12.0)


# ─── ensembles, mixtures, targets ────────────────────────────────────────────────────────────────
def ensembles(i):
    """(cell index into the scn_targets cells, ensemble label 0..3) from the geometry fit (json) of slice i."""
    import scn_coreshell_frequency as CF
    import scn_coreshell_geometry as GE
    import scn_frequency_figure as FF
    geo = json.load(open(os.path.join(S.OUT, "writeup", "scn_core_shell_geometry.json")))["slices"][str(i)]
    r = FF.analyse(i, "epoch"); f, se = r["f"]["ttx"], r["se"]["ttx"]
    idx = np.where(np.isfinite(f) & np.isfinite(se))[0]
    d = CF.slice_data(i); xl, yl, _, _ = GE.frame(d)
    best = geo["best"]; pr = geo[best]["params"]; sgn = geo[best]["sign"]
    coord = (np.cos(pr["phi"]) * xl + np.sin(pr["phi"]) * yl) if best == "Pm" else np.hypot(xl - pr["a"], yl - pr["b"])
    k = int(np.argmin(np.abs(GE.Q_GRID - pr["q"])))
    low = GE.split_by_quantile(coord, d["lobe"])[k]
    core = low if sgn == 1 else ~low
    ens = 2 * d["lobe"] + (~core).astype(int)                # 0 core A, 1 shell A, 2 core B, 3 shell B
    return idx, ens, f[idx], se[idx], best


def build_targets(i):
    import scn_pipeline_validation as V
    g = TG.load(i); idx, ens, f, se, best = ensembles(i)
    PH = g["PH"][:, idx]
    Z = [TG.mean_field(PH[:, ens == e]) for e in range(4)]
    d = S.process(i); sel = np.zeros(d["N"], bool); sel[g["cells"]] = True
    ttx, wash = int(g["ttx"]), int(g["wash"]); C = []
    for k in range(TG.N_CEIL):
        trend, A, noise, kick, Theta = V.ground_truth_inputs(d, sel, 1.0, np.random.default_rng(TG.SEED + 100 * i + k))
        xs = trend + A * np.cos(V.true_phases(d, sel, Theta, "sync", kick))
        P = EP.process(xs + float(g["noise_scale"]) * noise, ttx, wash, **TG.PIPE)["PH"][:, idx]
        C.append([EP.coherence(P[:, ens == e]) for e in range(4)])
    c = np.nanmean(np.array(C), axis=0)
    out = dict(idx=idx, ens=ens, f=f, se=se, Z=np.array(Z), R=np.abs(np.array(Z)), c=c,
               n=np.bincount(ens, minlength=4), split=best)
    np.savez_compressed(os.path.join(OUT, f"scn{i}_target4.npz"), **out)
    T = int(g["T"]); v0, v1 = S.valid_window(T)
    m = lambda x, a, b: float(np.nanmean(x[a:b]))
    print(f"SCN{i} ({best}): n {out['n'].tolist()} | pre R/c " + " ".join(f"{m(out['R'][e], v0, ttx - 24):.2f}/"
          f"{m(c[e], v0, ttx - 24):.2f}" for e in range(4)) + " | TTX c " +
          " ".join(f"{m(c[e], ttx + 24, wash):.2f}" for e in range(4)), flush=True)
    return out


def load_targets(i):
    f = os.path.join(OUT, f"scn{i}_target4.npz")
    if not os.path.exists(f):
        return build_targets(i)
    z = np.load(f, allow_pickle=True)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def mixtures(i, tg):
    """Naive noise-floor mixture per ensemble (rad/h), cached; cell → component assignment."""
    import scn_deconvolution as DC
    f = os.path.join(OUT, f"scn{i}_mixtures4.json")
    if os.path.exists(f):
        mixes = json.load(open(f))
    else:
        mixes = [DC.fit_mixture(tg["f"][tg["ens"] == e], tg["se"][tg["ens"] == e], {}, np.random.default_rng(10 + e),
                                deconv=False) for e in range(4)]
        json.dump(mixes, open(f, "w"), indent=1)
    Om, De, w, ce = [], [], [], []
    assign = np.zeros(tg["f"].size, int); N = tg["f"].size
    for e, mx in enumerate(mixes):
        cells = np.where(tg["ens"] == e)[0]
        mu = np.array(mx["mu"]); ga = np.array(mx["gamma"]); we = np.array(mx["w"])
        resp = we[None] * ga[None] / ((tg["f"][cells, None] - mu[None]) ** 2 + ga[None] ** 2)
        assign[cells] = len(Om) + np.argmax(resp, axis=1)
        Om += list(mu / CPD); De += list(ga / CPD); w += list(cells.size / N * we); ce += [e] * mu.size
    w = np.array(w); ce = np.array(ce)
    return dict(Om=np.array(Om), De=np.array(De), w=w, ens=ce, assign=assign,
                p=np.array([w[ce == e].sum() for e in range(4)]), mixes=mixes)


_CTX = {}


def context(i):
    if i not in _CTX:
        g = TG.load(i); tg = load_targets(i); model = mixtures(i, tg); prep = F.prepare(g)
        z0 = SIM.initial_state(model, g["PH"][prep["t0"]][tg["idx"]])
        _CTX[i] = (g, tg, model, prep, z0, np.arange(prep["t0"], prep["t1"] + 1))
    return _CTX[i]


# ─── parameters, coupling time course, simulation, loss ──────────────────────────────────────────
def to_x(p, keys):
    x = []
    for k in keys:
        lo, hi, tr = PAR[k]
        if tr == "log":
            u = (np.log(np.clip(p[k], lo * 1.000001, hi * 0.999999)) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            u = (np.clip(p[k], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo)) - lo) / (hi - lo)
        x.append(np.log(u / (1 - u)))
    return np.array(x)


def from_x(x, keys):
    p = {}
    for k, xi in zip(keys, x):
        lo, hi, tr = PAR[k]; u = 1 / (1 + np.exp(-xi))
        p[k] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if tr == "log" else float(lo + u * (hi - lo))
    return p


def t_rec(p, ttx, wash):
    return ttx + S.TRANSIENT_H + p["u_rec"] * (wash - ttx - S.TRANSIENT_H)


def rho_t(p, tg, ttx, wash):
    tr = t_rec(p, ttx, wash); r = np.ones_like(tg)
    r[(tg >= ttx) & (tg < tr)] = p["f_ttx"]
    a = (tg >= tr) & (tg < wash)
    r[a] = p["f_ttx"] + (p["f_wash"] - p["f_ttx"]) * (tg[a] - tr) / max(wash - tr, 1e-9)
    b = tg >= wash
    r[b] = p["f_post"] + (p["f_wash"] - p["f_post"]) * np.exp(-(tg[b] - wash) / TAU_POST)
    return r


def coupling_matrix(p):
    """G[target ensemble, source ensemble] (before ρ(t) e^{−iα})."""
    G = np.zeros((4, 4))
    for L in (0, 1):
        c, s = 2 * L, 2 * L + 1
        G[c, 0] += 0.5 * p["K_cc"]; G[c, 2] += 0.5 * p["K_cc"]; G[c, s] = p["eps"] * p["K_cs"]
        G[s, c] = p["K_cs"]; G[s, s] = p["K_ss"]
    return G


def simulate(p, i):
    g, tg, model, prep, z0, ti = context(i)
    Om, De, w, ce, pe = model["Om"], model["De"], model["w"], model["ens"], model["p"]
    W = np.zeros((4, Om.size)); W[ce, np.arange(Om.size)] = w / pe[ce]          # ensemble mean-field weights
    Gm = coupling_matrix(p)[ce] * np.exp(-1j * p["alpha"])                       # (M, 4): input to each component
    t0, t1 = int(ti[0]), int(ti[-1]); n = t1 - t0; sub = int(round(1 / DT))
    tgr = t0 + 0.5 * DT * np.arange(2 * n * sub + 1)
    rg = rho_t(p, tgr, float(g["ttx"]), float(g["wash"]))
    gtr = np.where(tgr >= float(g["ttx"]), p.get("g_tr", 0.0) * np.exp(-np.clip(tgr - float(g["ttx"]), 0, None) /
                                                                       p.get("tau_tr", 1.0)), 0.0)
    lin = 1j * Om - De

    def f(j, z):
        H = rg[j] * (Gm @ (W @ z))
        return (lin - gtr[j]) * z + 0.5 * (H - np.conj(H) * z * z)
    z = z0.astype(complex).copy(); out = np.empty((n + 1, z.size), complex); out[0] = z
    for s_ in range(n * sub):
        j = 2 * s_
        k1 = f(j, z); k2 = f(j + 1, z + 0.5 * DT * k1); k3 = f(j + 1, z + 0.5 * DT * k2); k4 = f(j + 2, z + DT * k3)
        z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (s_ + 1) % sub == 0:
            out[(s_ + 1) // sub] = z
    return out @ W.T, out @ w                                                      # (n+1, 4), (n+1,)


def loss(x, keys, i, detail=False):
    g, tg, model, prep, z0, ti = context(i)
    p = from_x(x, keys)
    Ze, Z = simulate(p, i)
    if not (np.all(np.isfinite(Ze)) and np.all(np.isfinite(Z))):
        return (1e3, None) if detail else 1e3
    msk = g["mask"][ti] > 0; n = msk.sum()
    err = [(tg["c"][e][ti] * np.abs(Ze[:, e]) - tg["R"][e][ti]) ** 2 for e in range(4)]
    err.append((g["c"][ti] * np.abs(Z) - g["R"][ti]) ** 2)
    eR = np.where(msk, np.mean(err, axis=0), 0.0)
    ph = []
    for L in (0, 1):
        c, s = 2 * L, 2 * L + 1
        dm = np.angle(Ze[:, s] * np.conj(Ze[:, c])); dd = np.angle(tg["Z"][s][ti] * np.conj(tg["Z"][c][ti]))
        var = 1 / (2 * tg["n"][c] * np.maximum(tg["R"][c][ti], 1e-3) ** 2) + \
            1 / (2 * tg["n"][s] * np.maximum(tg["R"][s][ti], 1e-3) ** 2) + SIG0 ** 2
        e_ = SIG_R ** 2 * np.angle(np.exp(1j * (dm - dd))) ** 2 / var
        ph.append(np.where(msk & np.isfinite(e_), e_, 0.0))
    fterm = sum(((F.collective_frequency(Z[a - ti[0]:b - ti[0]]) - nu) / F.SIGMA_NU) ** 2
                for (a, b), nu in zip(prep["stretches"], prep["nu"]))
    L = (eR.sum() + 0.5 * (ph[0] + ph[1]).sum() + fterm) / n
    if not np.isfinite(L):
        return (1e3, None) if detail else 1e3
    if detail:
        rm = lambda a: float(np.sqrt(np.mean(a[msk])))
        return L, dict(params=p, Ze=Ze, Z=Z, ti=ti, rmse_e=[rm(a) for a in err[:4]], rmse_g=rm(err[4]),
                       phase_term=float(0.5 * (ph[0] + ph[1])[msk].mean()))
    return L


# ─── fitting ─────────────────────────────────────────────────────────────────────────────────────
def _job(args):
    i, keys, x0, maxiter, xatol, fatol = args
    r = minimize(loss, x0, args=(keys, i), method="Nelder-Mead",
                 options=dict(maxiter=maxiter, maxfev=maxiter, xatol=xatol, fatol=fatol, adaptive=True))
    return i, r.x, float(r.fun)


def fit(slices, models, n_starts, procs):
    for i in slices:
        context(i)
    for name in models:
        keys = MODELS[name]; t_ = time.time(); mi = 300 * len(keys); rng = np.random.default_rng(3)
        x_start = to_x(START, keys); jobs = []
        for i in slices:
            X0 = [x_start] + [x_start + rng.normal(0, 1.2, len(keys)) for _ in range(n_starts - 1)]
            jobs += [(i, keys, x0, mi, 1e-4, 1e-7) for x0 in X0]
        with Pool(procs) as pool:
            R = pool.map(_job, jobs, chunksize=1)
            best = {}
            for i, x, L in R:
                if i not in best or L < best[i][1]:
                    best[i] = (x, L)
            P = pool.map(_job, [(i, keys, best[i][0], 2 * mi, 1e-5, 1e-8) for i in slices], chunksize=1)
        for i, xb, _ in P:
            L, det = loss(xb, keys, i, detail=True); g = TG.load(i)
            det.update(slice=i, model=name, keys=keys, x=xb, loss=L, n_free=len(keys),
                       t_rec_h=t_rec(det["params"], float(g["ttx"]), float(g["wash"])),
                       start_losses=sorted(r[2] for r in R if r[0] == i))
            np.save(os.path.join(OUT, f"scn{i}_{name}_fit.npy"), det, allow_pickle=True)
            print(f"[fit {name}] SCN{i}: loss {L * 1e3:.3f}e-3 | RMSE " +
                  "/".join(f"{v:.3f}" for v in det["rmse_e"]) + f" (cA/sA/cB/sB) slice {det['rmse_g']:.3f} | phase "
                  f"{det['phase_term'] * 1e3:.2f}e-3 | t_rec − t_wash {det['t_rec_h'] - float(g['wash']):.0f} h | " +
                  ", ".join(f"{k}={v:.3g}" for k, v in det["params"].items()), flush=True)
        print(f"[fit {name}] done in {time.time() - t_:.0f} s", flush=True)


def figure(slices, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prl_style import set_prl_style, PNAS_DOUBLE
    set_prl_style("pnas")
    fig, ax = plt.subplots(4, len(slices), figsize=(PNAS_DOUBLE, 6.4), sharex="col", squeeze=False,
                           gridspec_kw=dict(hspace=0.15, wspace=0.3, left=0.07, right=0.99, top=0.95, bottom=0.07))
    col = dict(core="#08519c", shell="#e6550d"); ls = dict(base="-", tr="--")
    for k, i in enumerate(slices):
        g, tg, model, prep, z0, ti = context(i)
        td = np.arange(int(g["T"])) / 24
        for r_ in range(4):
            ax[r_, k].axvspan(g["ttx"] / 24, g["wash"] / 24, color="0.9", lw=0)
        for L in (0, 1):
            for e, nm in ((2 * L, "core"), (2 * L + 1, "shell")):
                ax[L, k].plot(td, tg["R"][e], color=col[nm], lw=0.7, alpha=0.6)
            dd = np.angle(tg["Z"][2 * L + 1] * np.conj(tg["Z"][2 * L])) * CPD
            ax[3, k].plot(td, dd, color=("#1b7837", "#762a83")[L], lw=0.7, alpha=0.6)
        ax[2, k].plot(td, g["R"], color="k", lw=0.7, alpha=0.6)
        for name in models:
            fp = os.path.join(OUT, f"scn{i}_{name}_fit.npy")
            if not os.path.exists(fp):
                continue
            r = np.load(fp, allow_pickle=True).item(); tt = r["ti"] / 24
            for L in (0, 1):
                for e, nm in ((2 * L, "core"), (2 * L + 1, "shell")):
                    ax[L, k].plot(tt, tg["c"][e][r["ti"]] * np.abs(r["Ze"][:, e]), color=col[nm], lw=1.2, ls=ls[name])
                ax[3, k].plot(tt, np.angle(r["Ze"][:, 2 * L + 1] * np.conj(r["Ze"][:, 2 * L])) * CPD,
                              color=("#1b7837", "#762a83")[L], lw=1.2, ls=ls[name])
            ax[2, k].plot(tt, g["c"][r["ti"]] * np.abs(r["Z"]), color="#c0392b", lw=1.2, ls=ls[name], label=name)
        ax[0, k].set_title(f"SCN{i}", fontweight="bold")
        for r_ in range(3):
            ax[r_, k].set_ylim(0, 1.02)
        ax[3, k].set_ylim(-6, 6); ax[3, k].axhline(0, color="0.6", lw=0.5); ax[3, k].set_xlabel("time (days)")
    ax[0, 0].set_ylabel("lobe A R"); ax[1, 0].set_ylabel("lobe B R"); ax[2, 0].set_ylabel("slice R")
    ax[3, 0].set_ylabel("shell − core (h)"); ax[2, 0].legend(frameon=False, fontsize=5.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "scn_coreshell4_fits.png" if ext == "png" else "scn_coreshell4_fits.pdf"),
                    dpi=250)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")])
    models = arg("--models", ["base", "tr"], lambda s_: s_.split(","))
    if "targets" in sys.argv:
        for i in slices:
            build_targets(i)
    if "time" in sys.argv:
        i = slices[0]; context(i); x = to_x(START, MODELS["base"]); t_ = time.time()
        for _ in range(5):
            L = loss(x, MODELS["base"], i)
        print(f"[time] SCN{i}: {(time.time() - t_) / 5 * 1e3:.1f} ms per loss evaluation (M = "
              f"{context(i)[2]['Om'].size}); start loss {L:.4g}")
    if "fit" in sys.argv:
        fit(slices, models, arg("--starts", 32, int), arg("--procs", os.cpu_count(), int))
    if "figure" in sys.argv:
        figure(slices, models)
