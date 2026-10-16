r"""
LMMF fit of the SCN slices: intrinsic frequency distributions (TTX) + coupling from the coherence dynamics
==========================================================================================================

Per slice (scn_common; cells rhythmic under TTX only):
  (1) intrinsic frequencies ω_i = 2π / period under TTX (rad/h). Lorentzian-mixture fits with and without
      the noise floor (λ = LAM, width floor WIDTH_K, M_max = M_MAX; fitted on standardised frequencies,
      transformed back). Variant "lobes": one mixture per lobe (left/right SCN), weights scaled by the lobe
      fractions.
  (2) the LMMF   dz_m/dt = (iΩ_m − Δ_m) z_m + ½ (H_m − H̄_m z_m²),   H_m = e^{−iα} Σ_n G_mn w_n z_n
      with G = K (variant "global") or K_in / K_out within / between lobes (variant "lobes"), and
      epoch-dependent coupling: K_on before TTX and after washout, K_TTX under TTX; α shared.
      Every epoch starts from the MEASURED state: each cell is assigned to its most likely component and
      z_m(t0) = mean of e^{iθ} over that group (components without cells: the data mean field).
      Loss: mean squared error of R(t) (global R for "global", both lobe R's for "lobes") over the valid
      window, + Σ_coupled epochs ((ν_model − ν_data)/SIGMA_NU)² / n_hours for the collective frequency ν
      (slope of the mean-field phase). Nelder–Mead on (log K…, atanh(2α/π)) from several starts.
  (3) figure: distributions + fits; data vs model R(t) for both fits; parameters.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_lmmf_fit.py fit [--procs P] [--slices 1,2]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_lmmf_fit.py figure
Results: data_paths.MPMF/circadian_data/lmmf_fit/
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
from scipy.stats import gaussian_kde
import data_paths as dp
import lorentzian_mixture as LM
import scn_common as S

OUT = dp.ensure(os.path.join(S.OUT, "lmmf_fit"))
LAM, WIDTH_K, M_MAX = 1e-6, 10, 64
DT = 0.25                       # RK4 step (h); data are hourly
SIGMA_NU = 0.005                # rad/h scale of the collective-frequency mismatch (≈ 0.02 cycles/day)
KINDS = ("floor", "nofloor")
VARIANTS = ("global", "lobes")
STARTS = [(0.05, 0.0), (0.15, -0.4), (0.15, 0.4), (0.4, 0.0)]    # (K_on, α) starts; K_TTX = 0.1 K_on


# ─── data ─────────────────────────────────────────────────────────────────────────────────────────
def slice_data(i):
    """Cached per-slice data of the cells rhythmic under TTX: ω (rad/h), lobe, Hilbert phases, epochs."""
    f = os.path.join(OUT, f"scn{i}_data.npz")
    if not os.path.exists(f):
        d = S.process(i); lab = S.lobes(d["locs"]); r = d["rhythmic"]["ttx"]
        e = d["epochs"]
        np.savez_compressed(f, om=2 * np.pi / d["period"]["ttx"][r], lobe=lab[r], PH=d["PH"][:, r], T=d["T"],
                            ttx=d["ttx"], wash=d["wash"], N_all=d["N"],
                            epochs=np.array([e["pre"], e["ttx"], e["wash"]]))
    z = np.load(f)
    return {k: z[k] for k in z.files}


def segments(d):
    """(name, t0, t1, coupled) of the simulated epochs: each starts from the measured state."""
    v0, v1 = S.valid_window(int(d["T"]))
    return [("pre", v0, int(d["ttx"]), True), ("ttx", int(d["ttx"]), int(d["wash"]), False),
            ("wash", int(d["wash"]), v1, True)]


def collective_frequency(Z):
    ph = np.unwrap(np.angle(Z)); t = np.arange(Z.size)
    return np.polyfit(t, ph, 1)[0]


# ─── mixture fits ─────────────────────────────────────────────────────────────────────────────────
def width_floor(x, k=WIDTH_K):
    g = np.linspace(x.min(), x.max(), 2001)
    return max(1e-4, k / (x.size * float(gaussian_kde(x)(g).max())))


def fit_mixture(om, kind):
    mu, sd = om.mean(), om.std()
    x = (om - mu) / sd
    r = LM.fit(x, (width_floor(x), 1e2), lambda_M=LAM, M_max=M_MAX, floor_c=1.0 if kind == "floor" else 0.0,
               patience=3, n_restarts=10, seed=1, method="slsqp", large_M=16)
    m = r["model"]
    return dict(w=np.asarray(m.w), Om=mu + sd * np.asarray(m.Omega), De=sd * np.asarray(m.Delta), M=int(m.M),
                D=float(r["data_loss"]))


def build_model(d, kind, variant):
    """Components (Om, De, w global weights, lobe of each component) and the cell -> component assignment."""
    om, lobe = d["om"], d["lobe"]
    if variant == "global":
        f = fit_mixture(om, kind)
        parts = [(f, np.arange(om.size), -1, 1.0)]
    else:
        parts = []
        for k in (0, 1):
            idx = np.where(lobe == k)[0]
            parts.append((fit_mixture(om[idx], kind), idx, k, idx.size / om.size))
    Om, De, w, cl, assign = [], [], [], [], np.zeros(om.size, int)
    for f, idx, k, frac in parts:
        resp = f["w"][None] * f["De"][None] / ((om[idx, None] - f["Om"][None]) ** 2 + f["De"][None] ** 2)
        assign[idx] = len(Om) + np.argmax(resp, axis=1)
        Om += list(f["Om"]); De += list(f["De"]); w += list(frac * f["w"]); cl += [k] * f["M"]
    return dict(Om=np.array(Om), De=np.array(De), w=np.array(w), lobe=np.array(cl), assign=assign,
                fits=[p[0] for p in parts])


# ─── LMMF ─────────────────────────────────────────────────────────────────────────────────────────
def coupling(model, K_in, K_out):
    cl = model["lobe"]
    same = (cl[:, None] == cl[None, :]) | (cl[:, None] < 0)
    return np.where(same, K_in, K_out)


def simulate(model, z0, G, alpha, n_hours):
    """RK4 with step DT; returns z (n_hours + 1, M) at hourly samples."""
    a = 1j * model["Om"] - model["De"]
    Gw = np.exp(-1j * alpha) * G * model["w"][None, :]
    F = lambda z: a * z + 0.5 * ((Gw @ z) - np.conj(Gw @ z) * z * z)
    sub = int(round(1.0 / DT)); out = np.empty((n_hours + 1, z0.size), complex); z = z0.copy(); out[0] = z
    for h in range(1, n_hours + 1):
        for _ in range(sub):
            k1 = F(z); k2 = F(z + 0.5 * DT * k1); k3 = F(z + 0.5 * DT * k2); k4 = F(z + DT * k3)
            z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[h] = z
    return out


def initial_state(model, PHt):
    """z_m from the measured phases of the cells assigned to component m."""
    e = np.exp(1j * PHt); z0 = np.full(model["Om"].size, e.mean(), complex)
    for m in range(z0.size):
        sel = model["assign"] == m
        if sel.any():
            z0[m] = e[sel].mean()
    return z0


def unpack(p, variant):
    if variant == "global":
        K_on, K_ttx, al = np.exp(p[0]), np.exp(p[1]), 0.5 * np.pi * np.tanh(p[2])
        return dict(K_on=(K_on, K_on), K_ttx=(K_ttx, K_ttx), alpha=al)
    return dict(K_on=(np.exp(p[0]), np.exp(p[1])), K_ttx=(np.exp(p[2]), np.exp(p[3])), alpha=0.5 * np.pi * np.tanh(p[4]))


def predict(d, model, par):
    """Model mean fields per epoch: dict name -> (t0, Z_global, {lobe: Z_lobe})."""
    res = {}
    for name, t0, t1, coupled in segments(d):
        Kin, Kout = par["K_on"] if coupled else par["K_ttx"]
        z = simulate(model, initial_state(model, d["PH"][t0]), coupling(model, Kin, Kout), par["alpha"], t1 - t0 - 1)
        Zg = z @ model["w"]
        Zl = {}
        for k in (0, 1):
            sel = model["lobe"] == k
            if sel.any():
                Zl[k] = z[:, sel] @ (model["w"][sel] / model["w"][sel].sum())
        res[name] = (t0, Zg, Zl)
    return res


def data_meanfields(d, t0, t1):
    E = np.exp(1j * d["PH"][t0:t1])
    return E.mean(1), {k: E[:, d["lobe"] == k].mean(1) for k in (0, 1)}


def loss(p, d, model, variant, detail=False):
    par = unpack(p, variant)
    pred = predict(d, model, par)
    err, n, fterm, info = 0.0, 0, 0.0, {}
    for name, t0, t1, coupled in segments(d):
        _, Zg, Zl = pred[name]
        Dg, Dl = data_meanfields(d, t0, t1)
        if variant == "global":
            e = (np.abs(Zg) - np.abs(Dg)) ** 2
        else:
            e = 0.5 * sum((np.abs(Zl[k]) - np.abs(Dl[k])) ** 2 for k in (0, 1))
        err += e.sum(); n += e.size
        if coupled:
            nu_m, nu_d = collective_frequency(Zg), collective_frequency(Dg)
            fterm += ((nu_m - nu_d) / SIGMA_NU) ** 2
            info[name] = (nu_m, nu_d)
    L = err / n + fterm / n
    return (L, info, err / n) if detail else L


def fit_dynamics(d, model, variant):
    best = None
    for K0, a0 in STARTS:
        if variant == "global":
            p0 = [np.log(K0), np.log(0.1 * K0), np.arctanh(2 * a0 / np.pi)]
        else:
            p0 = [np.log(K0), np.log(K0), np.log(0.1 * K0), np.log(0.1 * K0), np.arctanh(2 * a0 / np.pi)]
        r = minimize(loss, p0, args=(d, model, variant), method="Nelder-Mead",
                     options=dict(maxiter=400 * len(p0), xatol=1e-3, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    return best


def _job(args):
    i, kind, variant = args
    out = os.path.join(OUT, f"scn{i}_{variant}_{kind}.npy")
    if os.path.exists(out):
        return out, 0.0, None
    t0 = time.time()
    d = slice_data(i)
    model = build_model(d, kind, variant)
    r = fit_dynamics(d, model, variant)
    L, info, mse = loss(r.x, d, model, variant, detail=True)
    par = unpack(r.x, variant)
    pred = predict(d, model, par)
    np.save(out, dict(slice=i, kind=kind, variant=variant, params=par, x=r.x, loss=L, mse_R=mse, nu=info,
                      model={k: v for k, v in model.items() if k != "fits"}, fits=model["fits"],
                      pred={k: (v[0], v[1], v[2]) for k, v in pred.items()}), allow_pickle=True)
    return out, time.time() - t0, (par, L, mse, model)


def run(slices, procs):
    for i in slices:
        slice_data(i)                                           # cache the data once (single process)
    jobs = [(i, k, v) for i in slices for v in VARIANTS for k in KINDS]
    t0 = time.time()
    with Pool(min(procs, len(jobs))) as p:
        for out, dt, res in p.imap_unordered(_job, jobs, chunksize=1):
            if res is None:
                print(f"[fit] {os.path.basename(out)} exists", flush=True); continue
            par, L, mse, model = res
            print(f"[fit] {os.path.basename(out)}: M = {model['Om'].size}, K_on = {np.round(par['K_on'], 4)}, "
                  f"K_TTX = {np.round(par['K_ttx'], 4)}, α = {par['alpha']:+.3f}, RMSE(R) = {np.sqrt(mse):.3f}, "
                  f"loss = {L:.4f} ({dt / 60:.1f} min; {(time.time() - t0) / 60:.1f} min elapsed)", flush=True)


# ─── figure ───────────────────────────────────────────────────────────────────────────────────────
def figure(slices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C = {"floor": "#eb6834", "nofloor": "#2a78d6"}
    fig, axs = plt.subplots(3, len(slices), figsize=(3.1 * len(slices), 8.4), layout="constrained", squeeze=False)
    for j, i in enumerate(slices):
        d = slice_data(i); T = int(d["T"]); om = d["om"]; f_cd = 24 / (2 * np.pi)       # rad/h -> cycles/day
        R = {(v, k): np.load(os.path.join(OUT, f"scn{i}_{v}_{k}.npy"), allow_pickle=True).item()
             for v in VARIANTS for k in KINDS if os.path.exists(os.path.join(OUT, f"scn{i}_{v}_{k}.npy"))}
        # (1) intrinsic distribution and global fits
        a = axs[0, j]; g = np.linspace(om.min() - 0.01, om.max() + 0.01, 800)
        a.hist(om * f_cd, bins=40, density=True, color="0.82")
        for k in KINDS:
            if ("global", k) in R:
                m = R[("global", k)]["model"]
                pdf = (m["w"][:, None] * m["De"][:, None] / np.pi / ((g[None] - m["Om"][:, None]) ** 2 + m["De"][:, None] ** 2)).sum(0)
                a.plot(g * f_cd, pdf / f_cd, color=C[k], lw=1.0, label=f"{k} (M = {m['Om'].size})")
        a.set_xlabel("intrinsic frequency (cycles/day)"); a.set_title(f"SCN slice {i} (n = {om.size})")
        a.legend(fontsize=6, frameon=False)
        # (2)-(3) R(t): global variant, lobes variant
        for row, v in ((1, "global"), (2, "lobes")):
            c = axs[row, j]; t = np.arange(T) / 24
            c.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
            v0, v1 = S.valid_window(T)
            Dg = np.abs(np.exp(1j * d["PH"]).mean(1)); Dg[:v0] = np.nan; Dg[v1:] = np.nan
            c.plot(t, Dg, color="0.1", lw=1.2, label="data")
            for k in KINDS:
                if (v, k) not in R:
                    continue
                res = R[(v, k)]
                for name, (t0, Zg, Zl) in res["pred"].items():
                    c.plot((t0 + np.arange(Zg.size)) / 24, np.abs(Zg), color=C[k], lw=0.9)
                p = res["params"]
                txt = (f"K_on {p['K_on'][0]:.3f}" + (f"/{p['K_on'][1]:.3f}" if v == "lobes" else "")
                       + f", K_TTX {p['K_ttx'][0]:.3f}" + (f"/{p['K_ttx'][1]:.3f}" if v == "lobes" else "")
                       + f", α {p['alpha']:+.2f}; RMSE {np.sqrt(res['mse_R']):.3f}")
                c.text(0.02, 0.02 + 0.09 * KINDS.index(k), txt, transform=c.transAxes, fontsize=5.5, color=C[k])
            c.set_ylim(0, 1.02); c.set_xlim(0, t[-1])
            c.set_xlabel("time (days)")
            c.set_title(f"{v} coupling" + (" (K_in/K_out)" if v == "lobes" else ""), fontsize=8)
            if j == 0:
                c.set_ylabel(r"phase coherence $R(t)$")
    h = [Line2D([], [], color="0.1", lw=1.2, label="data"),
         Line2D([], [], color=C["floor"], lw=1.0, label="LMMF, noise floor"),
         Line2D([], [], color=C["nofloor"], lw=1.0, label="LMMF, no floor")]
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False)
    out = os.path.join(OUT, "scn_lmmf_fit")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", list(S.SLICES), lambda s_: [int(v) for v in s_.split(",")])
    if "fit" in sys.argv:
        run(slices, arg("--procs", os.cpu_count(), int))
    if "figure" in sys.argv:
        figure(slices)
