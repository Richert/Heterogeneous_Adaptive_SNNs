r"""
Mixture-only predictions: steady-state coherence vs global coupling, relaxation, and parameter-free data tests
=============================================================================================================

For each slice (SCN1–5) and frequency-distribution variant (no floor | noise floor | single Lorentzian; the cached
scn_model_fit mixtures, λ = 1e-5), ONLY the mixture and one global Kuramoto coupling K (no lag, no noise, no
lobe-local coupling):

    ż_m = (i(Ω_m − Ω̄) − Δ_m) z_m + ½ K (Z − Z̄ z_m²),   Z = Σ_m w_m z_m,   R = |Z|        (rotating frame)

1. STEADY STATES R*(K), K log-spaced in [K_MIN, K_MAX] rad/h: RK4 for T_SS h from a synchronised start (z_m = 1)
   and from a nearly incoherent start (|z_m| = 1e-3, random phases); mean, min and max of R over the last T_AVG h
   (R oscillates for partially synchronised states of multimodal mixtures). The incoherence threshold K_c is the
   smallest K at which the incoherent state is linearly unstable (eigenvalues of the linearised equations).
2. RELAXATION: from in-phase starts z_m = R0 (R0 grid), the time t½ at which |R − R*(K)| has halved, for every
   (K, R0); summarised as t½ vs K at R0 = 0.2 and 0.5.
3. DATA TESTS (no dynamical fit; data R compared after dividing by the measurement ceiling c(t)):
   (i)   effective coupling: K at which R*(K) (synchronised branch, mean) equals the ceiling-corrected data plateau
         before TTX ([start, t_ttx − 24 h)) and after washout ([t_wash + 72 h, end)); and K / K_c
   (ii)  resynchronisation: from the MEASURED mixture state at the data's coherence MINIMUM t_min in
         [t_wash − 96 h, t_wash + 12 h] (cells assigned to components as in the fits; at the washout itself the
         measured coherence has already recovered), coupling K_post^eff: predicted half-time t½ of the rise
         R(t_min) → plateau vs the data's, both counted from t_min (onset-independent)
   (iii) desynchronisation with coupling off: from the measured state at t_ttx + 24 h, K = 0: per-lobe dephasing
         rate κ_d = −d ln R_L/dt over [t_ttx + 24 h, t_ttx + 72 h] vs the data's (fraction of the measured rate
         explained by the frequency distribution alone)

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_bifurcation_mixtures.py [--slices 1,2,3,4,5]
Output: data_paths.MPMF/circadian_data/bifurcation/{scn_bif_mixtures.{png,pdf}, scn<i>_bif.npz,
        scn_bif_tests.json}
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_model_fit as F
import scn_lmmf_sim as SIM

OUT = dp.ensure(os.path.join(S.OUT, "bifurcation"))
VARIANTS = [("nofloor", "no floor", "#2166ac", "-"), ("floor", "noise floor", "#c0392b", "--"),
            ("single", "single Lorentzian", "0.35", ":")]
K_MIN, K_MAX, N_K = 1e-3, 1.0, 80
T_SS, T_AVG, DT = 3000.0, 500.0, 0.5
R0_GRID = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
CPD = 24 / (2 * np.pi)


# ─── mean-field integration, vectorised over a batch of (K, initial state) ───────────────────────
def rhs(z, Om, De, w, K):
    """z: (B, M) complex, K: (B,) — global coupling only."""
    Z = z @ w
    return (1j * Om - De)[None] * z + 0.5 * K[:, None] * (Z[:, None] - np.conj(Z)[:, None] * z * z)


def integrate(z0, Om, De, w, K, T, dt=DT, record=None):
    """RK4 from z0 (B, M) over T h; returns the final state and R(t) on the hourly grid if record."""
    n = int(round(T / dt)); sub = int(round(1 / dt)); z = z0.astype(complex)
    Rs = [] if record else None
    for k in range(n):
        k1 = rhs(z, Om, De, w, K); k2 = rhs(z + dt / 2 * k1, Om, De, w, K)
        k3 = rhs(z + dt / 2 * k2, Om, De, w, K); k4 = rhs(z + dt * k3, Om, De, w, K)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if record and (k + 1) % sub == 0:
            Rs.append(np.abs(z @ w))
    return z, (np.array(Rs) if record else None)


def frame(mod):
    """Mixture in the rotating frame of its mean frequency."""
    Om = np.asarray(mod["Om"], float); w = np.asarray(mod["w"], float); w = w / w.sum()
    return Om - np.sum(w * Om), np.asarray(mod["De"], float), w


def steady_states(mod, Ks, seed=1):
    Om, De, w = frame(mod); M = Om.size; B = Ks.size
    out = {}
    rng = np.random.default_rng(seed)
    for lab, z0 in (("sync", np.ones((B, M), complex)),
                    ("incoh", 1e-3 * np.exp(1j * rng.uniform(0, 2 * np.pi, (1, M))).repeat(B, 0))):
        z, _ = integrate(z0, Om, De, w, Ks, T_SS - T_AVG)
        _, R = integrate(z, Om, De, w, Ks, T_AVG, record=True)
        out[lab] = dict(mean=R.mean(0), min=R.min(0), max=R.max(0))
    return out


def K_crit(mod, K_hi=10.0):
    """Smallest K at which the incoherent state of the global-coupling mixture is linearly unstable."""
    Om, De, w = frame(mod)
    D0 = np.diag(1j * Om - De); G = 0.5 * np.outer(np.ones(Om.size), w)
    growth = lambda K: np.max(np.linalg.eigvals(D0 + K * G).real)
    if growth(K_hi) < 0:
        return np.nan
    lo, hi = 0.0, K_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if growth(mid) < 0 else (lo, mid)
    return hi


def relaxation(mod, Ks, Rstar):
    """t½ (h) for every (K, R0): first time |R − R*(K)| ≤ |R0 − R*(K)| / 2, from in-phase starts z_m = R0."""
    Om, De, w = frame(mod); M = Om.size
    KK, RR = np.meshgrid(Ks, R0_GRID, indexing="ij"); B = KK.size
    z0 = RR.ravel()[:, None] * np.ones((1, M), complex)
    _, R = integrate(z0, Om, De, w, KK.ravel(), 2000.0, record=True)
    tgt = np.repeat(Rstar, R0_GRID.size); d0 = np.abs(RR.ravel() - tgt)
    hit = np.abs(R - tgt[None]) <= 0.5 * d0[None]
    th = np.where(hit.any(0), hit.argmax(0) + 1.0, np.nan); th[d0 < 1e-3] = np.nan
    return th.reshape(KK.shape)


# ─── data tests ──────────────────────────────────────────────────────────────────────────────────
def invert(Ks, Rmean, target):
    """Smallest K with R*(K) ≥ target on the synchronised branch (log-linear interpolation)."""
    above = np.where(Rmean >= target)[0]
    if above.size == 0:
        return np.nan
    k = above[0]
    if k == 0:
        return Ks[0]
    f = (target - Rmean[k - 1]) / max(Rmean[k] - Rmean[k - 1], 1e-12)
    return float(np.exp(np.log(Ks[k - 1]) + f * (np.log(Ks[k]) - np.log(Ks[k - 1]))))


def measured_state(mod, PH_t):
    z = SIM.initial_state(mod, PH_t)
    return z


def half_time(t, R, t0, R_hi):
    sel = t >= t0
    R0 = R[sel][0]; thr = 0.5 * (R0 + R_hi)
    up = np.where(R[sel] >= thr)[0] if R_hi > R0 else np.where(R[sel] <= thr)[0]
    return float(t[sel][up[0]] - t0) if up.size else np.nan


def kappa(t, R, a, b):
    sel = (t >= a) & (t < b) & np.isfinite(R) & (R > 0)
    return float(-np.polyfit(t[sel], np.log(R[sel]), 1)[0] * 24.0)


def data_tests(i, d, mod, Ks, ss):
    T = int(d["T"]); ttx, wash = int(d["ttx"]), int(d["wash"]); v0, v1 = S.valid_window(T); t = np.arange(T, dtype=float)
    Rc = d["R"] / d["c"]                                     # ceiling-corrected data coherence
    pre = float(np.nanmean(Rc[TG.start_hour(d, v0):ttx - 24])); post = float(np.nanmean(Rc[wash + 72:v1]))
    Kc = K_crit(mod); Rm = ss["sync"]["mean"]
    K_pre, K_post = invert(Ks, Rm, min(pre, 0.999)), invert(Ks, Rm, min(post, 0.999))
    out = dict(K_c=Kc, R_pre=pre, R_post=post, K_pre=K_pre, K_post=K_post, K_pre_rel=K_pre / Kc, K_post_rel=K_post / Kc)
    Om, De, w = frame(mod); cl = mod["lobe"]
    # (ii) resynchronisation from the measured state at the coherence minimum before washout, K = K_post^eff
    if np.isfinite(K_post):
        win = np.arange(wash - 96, wash + 12)
        t_min = int(win[np.nanargmin(Rc[win])])
        z0 = measured_state(mod, d["PH"][t_min])[None]
        _, R = integrate(z0, Om, De, w, np.array([K_post]), float(v1 - t_min), record=True)
        tm = np.concatenate([[t_min], t_min + 1 + np.arange(R.shape[0], dtype=float)])
        Rp = np.concatenate([[np.abs(z0[0] @ w)], R[:, 0]])
        out.update(t_min=t_min, t_half_model=half_time(tm, Rp, t_min, post),
                   t_half_data=half_time(t, Rc, t_min, post), resync_t=tm, resync_R=Rp)
    # (iii) desynchronisation with coupling off, per lobe, from the measured state at t_ttx + 24 h
    t0 = ttx + 24
    z0 = measured_state(mod, d["PH"][t0])[None]
    zz = [z0[0]]; z = z0
    for _ in range(72):
        z, _ = integrate(z, Om, De, w, np.zeros(1), 1.0); zz.append(z[0])
    zz = np.array(zz); tt = t0 + np.arange(zz.shape[0], dtype=float)
    p = [w[cl == k].sum() for k in (0, 1)]
    RL = [np.abs(zz[:, cl == k] @ w[cl == k]) / p[k] for k in (0, 1)]
    cL = [d["cA"], d["cB"]]; dL = [d["RA"], d["RB"]]
    out["kd_free"] = [kappa(tt, cL[k][tt.astype(int)] * RL[k], t0, t0 + 48) for k in (0, 1)]
    out["kd_data"] = [kappa(t, dL[k], t0, t0 + 48) for k in (0, 1)]
    return out


# ─── main ────────────────────────────────────────────────────────────────────────────────────────
def figure(res, slices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prl_style import set_prl_style, panel_label, COL_DOUBLE
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center"})
    fig, axs = plt.subplots(3, len(slices), figsize=(COL_DOUBLE, 5.0), squeeze=False,
                            gridspec_kw=dict(hspace=0.55, wspace=0.12, left=0.07, right=0.99, top=0.95, bottom=0.08))
    for j, i in enumerate(slices):
        g = res[i]; Ks = g["Ks"]; d = TG.load(i)
        a = axs[0, j]
        for kind, lab, col, ls in VARIANTS:
            ss = g[kind]["ss"]
            a.fill_between(Ks, ss["sync"]["min"], ss["sync"]["max"], color=col, alpha=0.15, lw=0)
            a.plot(Ks, ss["sync"]["mean"], color=col, ls=ls, lw=1.0, label=lab)
            a.plot(Ks, ss["incoh"]["mean"], color=col, ls=ls, lw=0.5, alpha=0.6)
            tt = g[kind]["tests"]
            a.axvline(tt["K_c"], color=col, lw=0.5, ls=ls, alpha=0.7)
            for key, mk in (("K_pre", "o"), ("K_post", "s")):
                if np.isfinite(tt[key]):
                    a.plot(tt[key], tt["R_" + key[2:]], mk, color=col, ms=3, mec="k", mew=0.3)
        tt = g["floor"]["tests"]
        a.axhline(tt["R_pre"], color="k", lw=0.5, ls=":"); a.axhline(tt["R_post"], color="k", lw=0.5, ls="-.")
        a.set_xscale("log"); a.set_ylim(0, 1.02); a.set_title(f"SCN{i}"); a.set_xlabel(r"$K$ (rad/h)")
        if j == 0:
            a.set_ylabel(r"steady state $R^*$"); a.legend(loc="upper left", fontsize=4.6, handlelength=1.6)
        else:
            a.tick_params(labelleft=False)
        a = axs[1, j]
        for kind, lab, col, ls in VARIANTS:
            th = g[kind]["t_half"]
            for r0, lw in ((0.2, 1.0), (0.5, 0.5)):
                a.plot(Ks, th[:, list(R0_GRID).index(r0)], color=col, ls=ls, lw=lw)
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1, 2000); a.set_xlabel(r"$K$ (rad/h)")
        if j == 0:
            a.set_ylabel(r"relaxation $t_{1/2}$ (h)")
            a.text(0.97, 0.95, r"thick: $R_0$ = 0.2, thin: 0.5", transform=a.transAxes, ha="right", va="top",
                   fontsize=4.6)
        else:
            a.tick_params(labelleft=False)
        a = axs[2, j]; T = int(d["T"]); wash = int(d["wash"]); t = np.arange(T)
        t_min = g["floor"]["tests"].get("t_min", wash)
        sel = (t >= t_min - 24) & (t < min(t_min + 96, T))
        a.plot((t[sel] - t_min) / 24, (d["R"] / d["c"])[sel], color="k", lw=1.0, label="data / c")
        for kind, lab, col, ls in VARIANTS:
            tt = g[kind]["tests"]
            if "resync_t" in tt:
                s_ = tt["resync_t"] < t_min + 96
                a.plot((tt["resync_t"][s_] - t_min) / 24, tt["resync_R"][s_], color=col, ls=ls, lw=0.9)
        a.axvline((wash - t_min) / 24, color="0.5", lw=0.6, ls="--")
        a.set_ylim(0, 1.02); a.set_xlabel(r"days after $R$ minimum")
        if j == 0:
            a.set_ylabel(r"$R$ (resync, $K = K_{\rm post}^{\rm eff}$)"); a.legend(loc="lower right", fontsize=4.6)
            a.text(0.03, 0.04, "dashed: washout", transform=a.transAxes, fontsize=4.6, color="0.4")
        else:
            a.tick_params(labelleft=False)
    for r_, l in zip(range(3), "abc"):
        panel_label(axs[r_, 0], l, dx=-24, dy=3)
    out = os.path.join(OUT, "scn_bif_mixtures")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")


def main():
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")])
    Ks = np.logspace(np.log10(K_MIN), np.log10(K_MAX), N_K)
    res, table = {}, {}
    for i in slices:
        d = TG.load(i); res[i] = dict(Ks=Ks); table[i] = {}
        for kind, lab, *_ in VARIANTS:
            mod = F.mixture(i, kind, d)
            ss = steady_states(mod, Ks)
            th = relaxation(mod, Ks, ss["sync"]["mean"])
            tests = data_tests(i, d, mod, Ks, ss)
            res[i][kind] = dict(ss=ss, t_half=th, tests=tests)
            table[i][kind] = {k: v for k, v in tests.items() if not k.startswith("resync_")}
            np.savez_compressed(os.path.join(OUT, f"scn{i}_{kind}_bif.npz"), Ks=Ks, t_half=th, R0=R0_GRID,
                                **{f"{a}_{b}": ss[a][b] for a in ss for b in ss[a]})
            print(f"SCN{i} {kind:8s} M = {mod['Om'].size:2d} | K_c {tests['K_c']:.4f} | plateaus pre/post "
                  f"{tests['R_pre']:.2f}/{tests['R_post']:.2f} -> K_eff {tests['K_pre']:.4f}/{tests['K_post']:.4f} "
                  f"({tests['K_pre_rel']:.1f}/{tests['K_post_rel']:.1f} K_c) | resync t½ model "
                  f"{tests.get('t_half_model', np.nan):.0f} h vs data {tests.get('t_half_data', np.nan):.0f} h | "
                  f"K=0 κ_d lobes {np.round(tests['kd_free'], 3).tolist()} vs data {np.round(tests['kd_data'], 3).tolist()}",
                  flush=True)
    with open(os.path.join(OUT, "scn_bif_tests.json"), "w") as fh:
        json.dump({str(k): v for k, v in table.items()}, fh, indent=1, default=float)
    figure(res, slices)


if __name__ == "__main__":
    main()
