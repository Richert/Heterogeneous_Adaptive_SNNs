r"""
Robustness of the SCN mean field: measurement error, residual coupling under TTX, and noise in the bifurcations
===============================================================================================================

Three checks (SCN_REVIEW.md, risks 1–3). Slices SCN1–4 (the dynamical fits); reference dynamics = the winning model C
fitted with the EMPIRICAL distribution (scn_model_fit "hr-noise+transient+local", kind "empirical").

deconv    (risk 1) The measured ω_i carry a per-cell measurement error that is heavy-tailed, with median |error| = h
          (scn_model_fit.empirical_h; for a Cauchy error, h is exactly its half-width). The observed density is then
          the intrinsic one convolved with a Lorentzian of half-width h, so the mixtures are DECONVOLVED by fitting with
          widths ≥ h and subtracting h (kinds "<kind>-dc", scn_lmmf_sim.fit_mixture(h=...)). Reported per slice:
            widths  observed vs deconvolved component widths, and h
            K_c     incoherence threshold of the mixture with one global coupling (α = 0): observed vs deconvolved
            κ_d     coupling-free dephasing rate per lobe from the MEASURED state at t_ttx + 24 h over 48 h:
                    data (ln R/c) vs observed vs deconvolved mixture (no coupling, no noise)
            frozen  RMSE of c|Z| vs the data with the dynamics frozen at the empirical C fit: empirical | floor |
                    floor-dc | single | single-dc
selfcons  (risk 2) Are the TTX-epoch phase slopes the intrinsic frequencies? MICROSCOPIC network of the fitted model C
          (empirical fit): N cells with intrinsic ω_i = the measured ω_i, initial phases = the measured phases at the
          start, global coupling K_s(t) e^{−iα_s} Z and lobe-local K_loc Z_L, Cauchy phase noise of scale γ u(t)
          (+ the lobe-specific transient) — the microscopic counterpart of the OA width terms (Euler, DT_NET; Cauchy
          increments of scale σ dt). Slopes over the TTX window used for ω_i (t_ttx + 24 h to washout). Variants:
            full | coupling only (no noise) | noise only (no coupling)
          Metrics vs the intrinsic ω: regression slope b of slope_i on ω_i (b < 1: compression), Cauchy scale
          (half the interquartile range) of the slopes / of ω, and the median |slope − ω|; N_SEEDS realisations.
lcnoise   (risk 3) Minimal mean field (pooled noise-floor mixture, one global K, α = 0, rotating frame) with every width
          widened by g (Cauchy noise of scale g: Δ_m → Δ_m + g, exact in OA), for observed and deconvolved widths:
          states on a (K, g) grid from a synchronised and a near-incoherent start (RK4, T_LC h, last T_AVG h):
          incoherent | stationary partial synchrony | oscillating R (limit cycle) | bistable. Marks: the fitted noise
          γ (floor fit of C) under TTX and γ u_res after washout.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_robustness.py deconv selfcons lcnoise [--slices 1,2,3,4]
Output: data_paths.MPMF/circadian_data/robustness/ (tables as .json, lcnoise figure)
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from scipy.stats import spearmanr
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_model_fit as F
import scn_lmmf_sim as SIM
import scn_pycobi_bifurcation as B

OUT = dp.ensure(os.path.join(S.OUT, "robustness"))
CPD = 24 / (2 * np.pi)
REF_MODEL = F.ALIASES["C"]
DT_FIT = 0.5
DT_NET = 0.1
N_SEEDS = 3
LC_SLICES = (1, 3)
K_GRID = np.linspace(0.005, 0.08, 151)
G_GRID = np.linspace(0.0, 0.008, 17)
T_LC, T_AVG, DT_LC = 4000.0, 1000.0, 0.5


def ref_params(i, kind="empirical"):
    f = os.path.join(F.OUT, f"scn{i}_{kind}_{REF_MODEL}.npy")
    r = np.load(f, allow_pickle=True).item()
    return r["params"], r["keys"], r["x"]


def K_c(Om, De, w):
    """Incoherence threshold of the mixture with one global coupling (α = 0): smallest K with an unstable mode of
    ż_m = (iΩ_m − Δ_m) z_m + K/2 Σ w_n z_n (bisection on the largest real part)."""
    A0 = np.diag(1j * Om - De)
    grow = lambda K: np.linalg.eigvals(A0 + 0.5 * K * np.outer(np.ones_like(w), w)).real.max()
    lo, hi = 0.0, 1.0
    if grow(hi) < 0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if grow(mid) < 0 else (lo, mid)
    return hi


# ─── risk 1 ─────────────────────────────────────────────────────────────────────────────────────
def deconv(slices):
    res = {}
    print("\n[deconv] h = per-cell measurement half-width; widths in c/d (lobe A | lobe B)")
    for i in slices:
        d = TG.load(i); h = F.empirical_h(i); prep = F.prepare(d)
        r = dict(h_cpd=h * CPD)
        print(f"\nSCN{i}: h = {h * CPD:.4f} c/d")
        for kind in ("floor", "single"):
            mo, md = F.mixture(i, kind, d), F.mixture(i, kind + "-dc", d)
            r[kind] = dict(M_obs=[int((mo["lobe"] == k).sum()) for k in (0, 1)],
                           M_dc=[int((md["lobe"] == k).sum()) for k in (0, 1)],
                           De_obs=(mo["De"] * CPD).round(4).tolist(), De_dc=(md["De"] * CPD).round(4).tolist(),
                           K_c_obs=K_c(mo["Om"] - np.sum(mo["w"] * mo["Om"]), mo["De"], mo["w"]),
                           K_c_dc=K_c(md["Om"] - np.sum(md["w"] * md["Om"]), md["De"], md["w"]))
            print(f"  {kind:6s} M obs {r[kind]['M_obs']} dc {r[kind]['M_dc']}; widths obs {r[kind]['De_obs']}"
                  f" -> dc {r[kind]['De_dc']}; K_c {r[kind]['K_c_obs']:.4f} -> {r[kind]['K_c_dc']:.4f} rad/h")
        # coupling-free dephasing from the measured state at t_ttx + 24 h
        t0 = int(d["ttx"]) + S.TRANSIENT_H; t1 = t0 + 48; tt = np.arange(t0, t1 + 1)
        kd = {}
        for L, (RL, cL) in enumerate((("RA", "cA"), ("RB", "cB"))):
            y = np.log(d[RL][t0:t1 + 1] / d[cL][t0:t1 + 1]); ok = np.isfinite(y)
            kd.setdefault("data", []).append(-np.polyfit(tt[ok] - t0, y[ok], 1)[0] * 24)
        for kind in ("empirical", "floor", "floor-dc", "single", "single-dc"):
            m = F.mixture(i, kind, d); z0 = SIM.initial_state(m, d["PH"][t0])
            zt = z0[None] * np.exp(np.outer(tt - t0, 1j * m["Om"] - m["De"]))
            for L in (0, 1):
                sel = m["lobe"] == L; RL = np.abs(zt[:, sel] @ m["w"][sel]) / m["w"][sel].sum()
                kd.setdefault(kind, []).append(-np.polyfit(tt - t0, np.log(RL), 1)[0] * 24)
        r["kappa_d"] = {k: np.round(v, 3).tolist() for k, v in kd.items()}
        print("  κ_d (1/day, lobe A | B), K = 0 from the measured state: " +
              "; ".join(f"{k} {v[0]:.2f}|{v[1]:.2f}" for k, v in kd.items()))
        # frozen evaluation at the empirical C fit
        p, keys, x = ref_params(i)
        fr = {}
        for kind in ("empirical", "floor", "floor-dc", "single", "single-dc"):
            m = F.mixture(i, kind, d)
            F._GQ_MAX[0] = F.GQ_FRAC * float(m["De"].min())
            L, det = F.loss(x, keys, d, m, prep, DT_FIT, detail=True)
            fr[kind] = dict(loss=float(L), rmse=[float(det["rmse"]), float(det["rmse_A"]), float(det["rmse_B"])])
        r["frozen"] = fr
        print("  frozen at the empirical C fit, RMSE vs data (slice/A/B): " +
              "; ".join(f"{k} {v['rmse'][0]:.3f}/{v['rmse'][1]:.3f}/{v['rmse'][2]:.3f}" for k, v in fr.items()))
        res[str(i)] = r
    json.dump(res, open(os.path.join(OUT, "deconv.json"), "w"), indent=1, default=float)


# ─── risk 2 ─────────────────────────────────────────────────────────────────────────────────────
def network(d, p, om, lobe, ph0, t0, t1, coupling=True, noise=True, seed=0):
    """Euler simulation of the microscopic model C from t0 to t1 (h); hourly phases (n_h, N)."""
    rng = np.random.default_rng(seed)
    N = om.size; LA = lobe == 0
    sub = int(round(1 / DT_NET)); n = int(t1 - t0)
    tg = t0 + DT_NET * np.arange(n * sub + 1)
    Kf, uf = F.Kfun(p, d), F.ufun(p, d)
    Kg = np.array([Kf(t) for t in tg]) * np.exp(-1j * p["alpha_s"]) if coupling else np.zeros(tg.size, complex)
    Ki = (p["K_i"] * np.exp(-1j * p["alpha_i"])) if coupling else 0.0
    ug = np.array([uf(t) for t in tg])
    ttx = float(d["ttx"])
    gl = np.where(LA, p["g_trA"], p["g_trB"])
    th = np.where(np.isfinite(ph0), ph0, rng.uniform(0, 2 * np.pi, N)).astype(float)
    out = np.empty((n + 1, N)); out[0] = th
    for s in range(n * sub):
        t = tg[s]; e = np.exp(1j * th)
        Z = e.mean(); ZA = e[LA].mean(); ZB = e[~LA].mean()
        H = Kg[s] * Z + Ki * np.where(LA, ZA, ZB)
        drift = om + np.imag(H * np.conj(e))
        th = th + DT_NET * drift
        if noise:
            sc = p["gamma"] * ug[s] + (gl * np.exp(-(t - ttx) / p["tau_tr"]) if t >= ttx else 0.0)
            th = th + sc * DT_NET * rng.standard_cauchy(N)
        if (s + 1) % sub == 0:
            out[(s + 1) // sub] = th
    return out


def cauchy_scale(x):
    return 0.5 * np.subtract(*np.percentile(x, [75, 25]))


def selfcons(slices):
    res = {}
    print("\n[selfcons] phase slopes of the microscopic model C (empirical fit) over the TTX window vs intrinsic ω")
    print(f"{'slice':6s}{'variant':16s}{'b (slope~ω)':>12s}{'ρ_S':>10s}{'scale ratio':>12s}{'med|s−ω| c/d':>14s}"
          f"{'h c/d':>8s}")
    for i in slices:
        d = TG.load(i); p, _, _ = ref_params(i); prep = F.prepare(d)
        om, lobe = d["om"], d["lobe"]; t0 = prep["t0"]; t1 = int(d["wash"])
        a = int(d["ttx"]) + S.TRANSIENT_H; tt = np.arange(a, t1)
        h = F.empirical_h(i); r = {}
        for name, cpl, nse in (("full", True, True), ("coupling only", True, False), ("noise only", False, True)):
            bs, sr, md, rho = [], [], [], []
            for sd in range(N_SEEDS if nse else 1):
                PH = network(d, p, om, lobe, d["PH"][t0], t0, t1, cpl, nse, seed=sd)
                X = PH[a - t0:t1 - t0]
                sl = np.polyfit(tt - a, X, 1)[0]
                bs.append(np.polyfit(om - om.mean(), sl - sl.mean(), 1)[0])
                sr.append(cauchy_scale(sl) / cauchy_scale(om)); md.append(np.median(np.abs(sl - om)))
                rho.append(spearmanr(sl, om)[0])
            r[name] = dict(b=float(np.mean(bs)), scale_ratio=float(np.mean(sr)), med_abs_cpd=float(np.mean(md) * CPD),
                           spearman=float(np.mean(rho)))
            print(f"SCN{i:<3d}{name:16s}{r[name]['b']:12.3f}{r[name]['spearman']:10.2f}{r[name]['scale_ratio']:12.3f}"
                  f"{r[name]['med_abs_cpd']:14.4f}{h * CPD:8.4f}")
        r["params"] = dict(K_i=float(p["K_i"]), gamma=float(p["gamma"]), rho_ttx=float(p.get("rho_ttx", np.nan)),
                           K_pre=float(p["K_pre"]))
        res[str(i)] = r
    json.dump(res, open(os.path.join(OUT, "selfcons.json"), "w"), indent=1)


# ─── risk 3 ─────────────────────────────────────────────────────────────────────────────────────
def lc_scan(w, Om, De):
    """States on the (K, g) grid; returns class (nK, ng): 0 incoherent, 1 stationary, 2 limit cycle, 3 bistable,
    and the mean R from the synchronised start."""
    Om = Om - np.sum(w * Om)
    KK, GG = np.meshgrid(K_GRID, G_GRID, indexing="ij")
    Kb, Gb = KK.ravel(), GG.ravel()
    a = (1j * Om)[None] - (De[None] + Gb[:, None])
    rng = np.random.default_rng(0)
    out = []
    for z_init in (np.ones((Kb.size, w.size), complex),
                   1e-3 * np.exp(2j * np.pi * rng.uniform(size=(Kb.size, w.size)))):
        z = z_init.copy()
        f = lambda z: a * z + 0.5 * Kb[:, None] * ((z @ w)[:, None] - np.conj(z @ w)[:, None] * z * z)
        n_tr = int((T_LC - T_AVG) / DT_LC); n_av = int(T_AVG / DT_LC)
        for _ in range(n_tr):
            k1 = f(z); k2 = f(z + DT_LC / 2 * k1); k3 = f(z + DT_LC / 2 * k2); k4 = f(z + DT_LC * k3)
            z = z + DT_LC / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Rmin = np.full(Kb.size, np.inf); Rmax = np.zeros(Kb.size); Rh = [np.zeros(Kb.size), np.zeros(Kb.size)]
        for s in range(n_av):
            k1 = f(z); k2 = f(z + DT_LC / 2 * k1); k3 = f(z + DT_LC / 2 * k2); k4 = f(z + DT_LC * k3)
            z = z + DT_LC / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            R = np.abs(z @ w); Rmin = np.minimum(Rmin, R); Rmax = np.maximum(Rmax, R); Rh[2 * s // n_av] += R
        Rm = (Rh[0] + Rh[1]) / n_av
        conv = np.abs(Rh[1] - Rh[0]) / (n_av / 2) < 0.01        # mean R no longer drifting between the halves
        cls = np.where(Rm < 0.01, 0, np.where(Rmax - Rmin > 0.01, 2, 1))
        out.append((cls, Rm, conv))
    (c1, R1, v1), (c2, R2, v2) = out
    # the synchronised start decides; bistable only where BOTH runs converged to different states (near K_c the
    # near-incoherent start is still growing after T_LC h: critical slowing, not bistability)
    cls = np.where(v1 & v2 & ((c1 != c2) | (np.abs(R1 - R2) > 0.05)), 3, c1)
    return cls.reshape(KK.shape), R1.reshape(KK.shape)


def lcnoise(slices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    sl = [i for i in slices if i in LC_SLICES] or list(LC_SLICES)
    fig, ax = plt.subplots(len(sl), 2, figsize=(9, 3.4 * len(sl)), squeeze=False,
                           gridspec_kw=dict(wspace=0.25, hspace=0.45, left=0.08, right=0.98, top=0.9, bottom=0.1))
    cmap = ListedColormap(["#222222", "#9ecae1", "#e6550d", "#756bb1"])
    res = {}
    for r_, i in enumerate(sl):
        d = TG.load(i); h = F.empirical_h(i)
        mo = B.pooled_mixture(i, "pfloor")
        w = np.asarray(mo["w"], float); w = w / w.sum()
        md = SIM.fit_mixture(d["om"], F.LAM, True, F.WIDTH_K, h=h)
        wd = np.asarray(md["w"], float); wd = wd / wd.sum()
        pf = np.load(os.path.join(F.OUT, f"scn{i}_floor_{REF_MODEL}.npy"), allow_pickle=True).item()
        pp = pf["params"]
        marks = {"γ (TTX)": pp["gamma"], "γ u_res (washout)": pp["gamma"] * pp["u_res"]}
        res[str(i)] = dict(h_cpd=h * CPD, gamma=float(pp["gamma"]), u_res=float(pp["u_res"]))
        for c_, (lab, (ww, Om, De)) in enumerate((("observed widths", (w, np.asarray(mo["Om"]), np.asarray(mo["De"]))),
                                                  ("deconvolved widths", (wd, md["Om"], md["De"])))):
            cls, R = lc_scan(ww, Om, De)
            a = ax[r_, c_]
            a.imshow(cls.T, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest",
                     extent=(K_GRID[0], K_GRID[-1], G_GRID[0] * CPD, G_GRID[-1] * CPD))
            for (mk, g), ls in zip(marks.items(), ("-", "--")):
                a.axhline(g * CPD, color="w", lw=1.2, ls=ls, label=mk)
            a.set_title(f"SCN{i}, {lab} (M = {ww.size}; Δ = {', '.join(f'{x * CPD:.3f}' for x in De)} c/d)",
                        fontsize=9)
            a.set_xlabel("global coupling K (rad/h)"); a.set_ylabel("noise g (c/d)")
            lcK = [K_GRID[cls[:, j] == 2] for j in range(G_GRID.size)]
            res[str(i)][lab] = dict(De_cpd=(np.asarray(De) * CPD).round(4).tolist(),
                                    lc_window=[[float(k.min()), float(k.max())] if k.size else None for k in lcK],
                                    bistable=[[float(k.min()), float(k.max())] if k.size else None
                                              for k in [K_GRID[cls[:, j] == 3] for j in range(G_GRID.size)]],
                                    g_cpd=(G_GRID * CPD).round(4).tolist())
            print(f"[lcnoise] SCN{i} {lab}: LC window (K) per g [c/d]: " +
                  "; ".join(f"{g * CPD:.3f}:{'—' if k.size == 0 else f'{k.min():.3f}–{k.max():.3f}'}"
                            for g, k in zip(G_GRID[::2], lcK[::2])))
        ax[r_, 0].legend(loc="upper right", fontsize=7, framealpha=0.6)
    hs = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(4)]
    fig.legend(hs, ["incoherent", "stationary partial sync", "limit cycle (oscillating R)", "bistable"],
               loc="upper center", ncol=4, frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"scn_lcnoise.{ext}"), dpi=200)
    json.dump(res, open(os.path.join(OUT, "lcnoise.json"), "w"), indent=1)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", [1, 2, 3, 4], lambda s_: [int(v) for v in s_.split(",")])
    for step, fn in (("deconv", deconv), ("selfcons", selfcons), ("lcnoise", lcnoise)):
        if step in sys.argv:
            fn(slices)
