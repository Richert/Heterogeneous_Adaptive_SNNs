r"""
SCN slices: LMMF with global synaptic (phase-difference, lagged) + lobe-local gap-junction coupling
===================================================================================================

SUPERSEDED (2026-10-16) by scn_model_fit.py: same model family as scn_lmmf_sim (this script's model = its
"base" variant), fitted to the validated epoch-peak target with measurement ceiling and mask (scn_targets).
This script still uses the legacy global-Hilbert phases; kept for the record.

Microscopic model for a cell i in lobe L (θ = PER2 phase):

    θ̇_i = ω_i + K_syn(t) Im(e^{−iα} Z e^{−iθ_i}) + K_gap Im(Z_L e^{−iθ_i})

  synaptic (VIP/GABA): global, TTX-sensitive; Kuramoto coupling with phase lag α. This is the cycle-
      averaged form of pulse coupling (release timing relative to the receivers' phase response plus the
      signalling-cascade delay all enter α); K_syn(t) = K_pre, K_ttx, K_wash in the three epochs.
  gap junctions: lobe-local, TTX-insensitive, diffusive -> Kuramoto coupling without lag, K_gap.

LMMF (one Lorentzian mixture per lobe, weights × lobe fraction; component m in lobe L):
    ż_m = (iΩ_m − Δ_m) z_m + ½ (H_m − H̄_m z_m²),   H_m = K_syn e^{−iα} Z + K_gap Z_L
Fit per slice and mixture (noise floor / no floor): K_pre, K_ttx, K_wash, K_gap ∈ (0, K_MAX) (logistic
transform) and α ∈ (−π/2, π/2) (the synchronising range; tanh transform).
ONE continuous simulation per evaluation, from the MEASURED state at the first valid hour (each cell assigned to
its most likely component) to the end of the valid window, with the time-dependent synaptic input K_syn(t)
(scn_lmmf_sim.coupling_input: steps by default; fixed first-order rates via --rate-ttx / --rate-wash).
Loss over the whole valid window: mean of [(R_A − R̂_A)² + (R_B − R̂_B)² + (R − R̂)²] / 3 + the collective-frequency
mismatch (SIGMA_NU scale) in the coupled stretches (pre-TTX, and post-washout from TRANSIENT_H h after
washout), both taken from the same continuous run. (The pulse-coupling version is kept in
scn_pulse_fit.py; it produced oscillation death / non-synchronising regimes because its effective lag sat
near −π/2 and its tonic drive changed the intrinsic frequencies.)

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_syngap_fit.py fit figure [--procs P] [--slices 1,2]
                                                              [--rate-ttx R] [--rate-wash R]
Results: data_paths.MPMF/circadian_data/syngap_fit/
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import data_paths as dp
import scn_common as S
import scn_lmmf_fit as LF                  # cached slice data, mixture fits, initial states, frequencies
import scn_pulse_fit as PF                 # per-lobe model build, mean fields
import scn_lmmf_sim as SIM                 # time-dependent coupling input K_syn(t)

OUT = dp.ensure(os.path.join(S.OUT, "syngap_fit"))
K_MAX = 0.3                                # rad/h, ≈ the intrinsic frequency (excludes oscillation death)
DT = 0.25
SIGMA_NU = 0.005
KINDS = ("floor", "nofloor")
KEYS = ("K_pre", "K_ttx", "K_wash", "K_gap")
RATES = (np.inf, np.inf)                   # (TTX application, washout) rates of K_syn(t), 1/h; inf = steps
STARTS = [dict(K_pre=0.06, K_ttx=0.006, K_wash=0.05, K_gap=0.01, alpha=0.0),
          dict(K_pre=0.10, K_ttx=0.02, K_wash=0.08, K_gap=0.03, alpha=0.3),
          dict(K_pre=0.10, K_ttx=0.02, K_wash=0.08, K_gap=0.03, alpha=-0.3),
          dict(K_pre=0.04, K_ttx=0.002, K_wash=0.03, K_gap=0.06, alpha=0.0)]


def unpack(x):
    par = {k: K_MAX / (1 + np.exp(-v)) for k, v in zip(KEYS, x[:4])}
    par["alpha"] = 0.5 * np.pi * np.tanh(x[4])
    return par


def pack(p):
    return np.array([np.log(p[k] / (K_MAX - p[k])) for k in KEYS] + [np.arctanh(2 * p["alpha"] / np.pi)])


def simulate(model, z0, Kfun, K_gap, alpha, t0, t1):
    """One continuous RK4 run (step DT) from t0 to t1 (h) with the time-dependent synaptic input Kfun(t);
    returns z at the hourly samples t0 … t1 − 1."""
    lin = 1j * model["Om"] - model["De"]; w = model["w"]; cl = model["lobe"]; p = model["p"]; LA = cl == 0
    ea = np.exp(-1j * alpha)

    def F(t, z):
        Z = w @ z; ZA = (w[LA] @ z[LA]) / p[0]; ZB = (w[~LA] @ z[~LA]) / p[1]
        H = Kfun(t) * ea * Z + K_gap * np.where(LA, ZA, ZB)
        return lin * z + 0.5 * (H - np.conj(H) * z * z)

    n = t1 - t0; sub = int(round(1.0 / DT)); out = np.empty((n, z0.size), complex)
    z = z0.copy(); t = float(t0); out[0] = z
    for k in range(1, n):
        for _ in range(sub):
            k1 = F(t, z); k2 = F(t + DT / 2, z + DT / 2 * k1); k3 = F(t + DT / 2, z + DT / 2 * k2)
            k4 = F(t + DT, z + DT * k3); z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += DT
        out[k] = z
    return out


def loss(x, d, model, detail=False):
    par = unpack(x)
    t0, t1 = S.valid_window(int(d["T"])); ttx, wash = int(d["ttx"]), int(d["wash"])
    Kfun = SIM.coupling_input(par["K_pre"], par["K_ttx"], par["K_wash"], ttx, wash, *RATES)
    z = simulate(model, LF.initial_state(model, d["PH"][t0]), Kfun, par["K_gap"], par["alpha"], t0, t1)
    Zg, ZA, ZB = PF.meanfields(model, z)
    E = np.exp(1j * d["PH"][t0:t1]); Dg = E.mean(1)
    DA, DB = E[:, d["lobe"] == 0].mean(1), E[:, d["lobe"] == 1].mean(1)
    e = ((np.abs(ZA) - np.abs(DA)) ** 2 + (np.abs(ZB) - np.abs(DB)) ** 2 + (np.abs(Zg) - np.abs(Dg)) ** 2) / 3
    fterm = 0.0
    for a, b in ((t0, ttx), (wash + S.TRANSIENT_H, t1)):          # coupled stretches of the same run
        sl = slice(a - t0, b - t0)
        fterm += ((LF.collective_frequency(Zg[sl]) - LF.collective_frequency(Dg[sl])) / SIGMA_NU) ** 2
    L = (e.sum() + fterm) / e.size
    pred = {"all": (t0, Zg, ZA, ZB)}
    return (L, e.mean(), pred, par) if detail else L


def _job(args):
    i, kind = args
    out = os.path.join(OUT, f"scn{i}_{kind}.npy")
    if os.path.exists(out):
        return out, 0.0, None
    t0 = time.time()
    d = LF.slice_data(i); model = PF.build(d, kind)
    best = None
    for s in STARTS:
        r = minimize(loss, pack(s), args=(d, model), method="Nelder-Mead",
                     options=dict(maxiter=2500, xatol=1e-3, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    L, mse, pred, par = loss(best.x, d, model, detail=True)
    np.save(out, dict(slice=i, kind=kind, params=par, x=best.x, loss=L, mse_R=mse, pred=pred, model=model,
                      rates=RATES), allow_pickle=True)
    return out, time.time() - t0, (par, L, mse, model)


def run(slices, procs):
    for i in slices:
        LF.slice_data(i)
    jobs = [(i, k) for i in slices for k in KINDS]
    t0 = time.time()
    with Pool(min(procs, len(jobs))) as p:
        for out, dt, res in p.imap_unordered(_job, jobs, chunksize=1):
            if res is None:
                print(f"[fit] {os.path.basename(out)} exists", flush=True); continue
            par, L, mse, model = res
            print(f"[fit] {os.path.basename(out)}: M = {np.bincount(model['lobe']).tolist()}, K_syn pre/ttx/wash = "
                  f"{par['K_pre']:.4f}/{par['K_ttx']:.4f}/{par['K_wash']:.4f}, K_gap = {par['K_gap']:.4f}, "
                  f"α = {par['alpha']:+.3f}, RMSE(R) = {np.sqrt(mse):.3f}, loss = {L:.4f} "
                  f"({dt / 60:.1f} min; {(time.time() - t0) / 60:.1f} min elapsed)", flush=True)


def figure(slices):
    """Per slice: per-lobe distributions + fits; slice R(t) data vs both fits; lobe R(t) vs the floorless fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C = {"floor": "#eb6834", "nofloor": "#2a78d6"}; CL = ["#1b7837", "#762a83"]
    fig, axs = plt.subplots(4, len(slices), figsize=(3.1 * len(slices), 10.5), layout="constrained", squeeze=False)
    cd = 24 / (2 * np.pi)
    for j, i in enumerate(slices):
        d = LF.slice_data(i); T = int(d["T"])
        R = {k: np.load(os.path.join(OUT, f"scn{i}_{k}.npy"), allow_pickle=True).item()
             for k in KINDS if os.path.exists(os.path.join(OUT, f"scn{i}_{k}.npy"))}
        for row, lob in ((0, 0), (1, 1)):
            a = axs[row, j]; om = d["om"][d["lobe"] == lob]
            g = np.linspace(d["om"].min() - 0.01, d["om"].max() + 0.01, 800)
            a.hist(om * cd, bins=30, density=True, color="0.82")
            for k, r in R.items():
                m = r["model"]; sel = m["lobe"] == lob; ww = m["w"][sel] / m["w"][sel].sum()
                pdf = (ww[:, None] * m["De"][sel, None] / np.pi / ((g[None] - m["Om"][sel, None]) ** 2 + m["De"][sel, None] ** 2)).sum(0)
                a.plot(g * cd, pdf / cd, color=C[k], lw=1.0, label=f"{k}, M = {sel.sum()}")
            a.set_title(f"SCN{i}, lobe {'AB'[lob]} (n = {om.size})", fontsize=8); a.legend(fontsize=6, frameon=False)
            a.set_xlabel("intrinsic frequency (cycles/day)", fontsize=7)
        v0, v1 = S.valid_window(T); t = np.arange(T) / 24; E = np.exp(1j * d["PH"])

        def masked(x):
            x = np.abs(x).astype(float); x[:v0] = np.nan; x[v1:] = np.nan; return x
        c = axs[2, j]; c.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
        c.plot(t, masked(E.mean(1)), color="0.1", lw=1.2)
        for k, r in R.items():
            for name, (t0, Zg, ZA, ZB) in r["pred"].items():
                c.plot((t0 + np.arange(Zg.size)) / 24, np.abs(Zg), color=C[k], lw=0.9)
            p = r["params"]
            c.text(0.02, 0.02 + 0.1 * KINDS.index(k),
                   f"K_syn {p['K_pre']:.3f}/{p['K_ttx']:.3f}/{p['K_wash']:.3f}, K_gap {p['K_gap']:.3f}, "
                   f"α {p['alpha']:+.2f}; RMSE {np.sqrt(r['mse_R']):.3f}", transform=c.transAxes, fontsize=5.2, color=C[k])
        c.set_ylim(0, 1.02); c.set_title("whole slice", fontsize=8); c.set_xlabel("time (days)")
        c = axs[3, j]; c.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
        for lob in (0, 1):
            c.plot(t, masked(E[:, d["lobe"] == lob].mean(1)), color=CL[lob], lw=1.1)
        if "nofloor" in R:
            for name, (t0, Zg, ZA, ZB) in R["nofloor"]["pred"].items():
                for lob, Z in ((0, ZA), (1, ZB)):
                    c.plot((t0 + np.arange(Z.size)) / 24, np.abs(Z), color=CL[lob], lw=0.9, ls="--")
        c.set_ylim(0, 1.02); c.set_title("lobes: data (solid), LMMF no floor (dashed)", fontsize=8)
        c.set_xlabel("time (days)")
        if j == 0:
            axs[2, 0].set_ylabel(r"$R(t)$"); axs[3, 0].set_ylabel(r"lobe $R(t)$")
    h = [Line2D([], [], color="0.1", lw=1.2, label="data"), Line2D([], [], color=C["floor"], label="LMMF, noise floor"),
         Line2D([], [], color=C["nofloor"], label="LMMF, no floor"), Line2D([], [], color=CL[0], label="lobe A"),
         Line2D([], [], color=CL[1], label="lobe B")]
    fig.legend(handles=h, loc="outside lower center", ncol=5, frameon=False)
    out = os.path.join(OUT, "scn_syngap_fit")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", list(S.SLICES), lambda s_: [int(v) for v in s_.split(",")])
    RATES = (arg("--rate-ttx", np.inf, float), arg("--rate-wash", np.inf, float))
    if not all(np.isinf(RATES)):
        OUT = dp.ensure(os.path.join(OUT, f"rates_{RATES[0]:g}-{RATES[1]:g}"))
    if "fit" in sys.argv:
        run(slices, arg("--procs", os.cpu_count(), int))
    if "figure" in sys.argv:
        figure(slices)
