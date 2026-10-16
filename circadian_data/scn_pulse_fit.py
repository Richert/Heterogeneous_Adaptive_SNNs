r"""
SCN slices: LMMF with global pulse (synaptic) coupling + lobe-local gap-junction coupling
=========================================================================================

Microscopic model for a cell i in lobe L (θ = PER2 phase, θ = 0 at the band-passed PER2 peak):

    θ̇_i = ω_i − h(t) sin θ_i + K_gap Im(Z_L e^{−iθ_i}),     h(t) = K_syn(t) ⟨P(θ)⟩_all cells

  synaptic coupling (global, TTX-sensitive): Winfree / pulse coupling with the phase response
      Q(θ) = −sin θ (Pazó & Montbrió, PRX 2014: OA-exact because Q has a single harmonic); no delay
      or PRC phase lag (synaptic delays are ~10⁻⁵ of a cycle). Pulse (release window) of Duchet et al.
      (2023) shape, centred ψ hours BEFORE the PER2 peak:  P(θ) = c_n (1 + cos(θ − θ_c))^n,
      θ_c = −2π ψ / 24, c_n such that ⟨P⟩ = 1 for uniform phases. n = N_PULSE (default 2: SCN firing is
      broad, high during the subjective day, nearly silent at night). ψ fitted in PSI_RANGE.
      K_syn(t) = K_pre, K_ttx, K_wash in the three epochs.
  gap junctions (lobe-local, TTX-insensitive): Kuramoto coupling to the lobe mean field Z_L, no lag.

LMMF (one Lorentzian mixture per lobe, weights × lobe fraction; component m in lobe L):
    ż_m = (iΩ_m − Δ_m) z_m + ½ (H_m − H̄_m z_m²),   H_m = h + K_gap Z_L,   Z_L = Σ_{n∈L} w_n z_n / p_L
    ⟨P⟩ = Σ_m w_m c_n [a_0 + 2 Σ_{q=1}^{n} a_q Re(z_m^q e^{−iqθ_c})]   (exact on the OA manifold;
    a_q = Fourier coefficients of (1 + cos x)^n).
Fit per slice and per mixture (noise floor / no floor): K_pre, K_ttx, K_wash, K_gap (log scale) and ψ.
Each epoch is simulated from the MEASURED state 24 h after its start (pre: at the first valid hour), so
the zero-phase filter's smearing around the treatment changes does not enter the initial states.
Loss: mean over hours of [(R_A − R̂_A)² + (R_B − R̂_B)² + (R − R̂)²] / 3 (the global R pins the phase
relation between the lobes) + collective-frequency mismatch in the coupled epochs (SIGMA_NU scale).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_pulse_fit.py fit [--procs P] [--slices 1,2] [--n-pulse 2]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_pulse_fit.py figure [--n-pulse 2]
Results: data_paths.MPMF/circadian_data/pulse_fit/
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
import scn_lmmf_fit as LF                  # cached slice data, mixture fits

OUT = dp.ensure(os.path.join(S.OUT, "pulse_fit"))
N_PULSE = 2
PSI_RANGE = (2.0, 10.0)                    # pulse lead before the PER2 peak (h)
DT = 0.25
SIGMA_NU = 0.005
KINDS = ("floor", "nofloor")
STARTS = [dict(K=0.05, gap=0.01, psi=6.0), dict(K=0.15, gap=0.02, psi=6.0),
          dict(K=0.05, gap=0.05, psi=4.0), dict(K=0.15, gap=0.005, psi=8.0)]


def pulse_coeffs(n):
    """Fourier coefficients a_q (q = 0..n) of (1 + cos x)^n and the normalisation c_n."""
    x = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    a = np.array([np.mean((1 + np.cos(x)) ** n * np.cos(q * x)) for q in range(n + 1)])
    return a, 1.0 / a[0]


def build(d, kind):
    om, lobe = d["om"], d["lobe"]
    Om, De, w, cl, assign = [], [], [], [], np.zeros(om.size, int)
    for k in (0, 1):
        idx = np.where(lobe == k)[0]
        f = LF.fit_mixture(om[idx], kind)
        resp = f["w"][None] * f["De"][None] / ((om[idx, None] - f["Om"][None]) ** 2 + f["De"][None] ** 2)
        assign[idx] = len(Om) + np.argmax(resp, axis=1)
        Om += list(f["Om"]); De += list(f["De"]); w += list(idx.size / om.size * f["w"]); cl += [k] * f["M"]
    cl = np.array(cl); w = np.array(w)
    return dict(Om=np.array(Om), De=np.array(De), w=w, lobe=cl, assign=assign,
                p=np.array([w[cl == k].sum() for k in (0, 1)]))


def unpack(x):
    psi = PSI_RANGE[0] + (PSI_RANGE[1] - PSI_RANGE[0]) / (1 + np.exp(-x[4]))
    return dict(K_pre=np.exp(x[0]), K_ttx=np.exp(x[1]), K_wash=np.exp(x[2]), K_gap=np.exp(x[3]), psi=psi)


def pack(K_pre, K_ttx, K_wash, K_gap, psi):
    u = (psi - PSI_RANGE[0]) / (PSI_RANGE[1] - PSI_RANGE[0])
    return np.array([np.log(K_pre), np.log(K_ttx), np.log(K_wash), np.log(K_gap), np.log(u / (1 - u))])


def simulate(model, z0, K_syn, K_gap, psi, n_hours, n_pulse=N_PULSE):
    """RK4; returns z (n_hours + 1, M) hourly."""
    a_q, c_n = pulse_coeffs(n_pulse)
    rot = np.exp(-1j * np.arange(1, n_pulse + 1) * (-2 * np.pi * psi / 24.0))     # e^{−iqθ_c}
    lin = 1j * model["Om"] - model["De"]; w = model["w"]; cl = model["lobe"]; p = model["p"]
    LA = (cl == 0)

    def F(z):
        zq = z[None, :] ** np.arange(1, n_pulse + 1)[:, None]                     # (n, M)
        Pbar = c_n * (a_q[0] + 2 * np.sum(w[None, :] * a_q[1:, None] * np.real(zq * rot[:, None])))
        h = K_syn * Pbar
        ZA = (w[LA] @ z[LA]) / p[0]; ZB = (w[~LA] @ z[~LA]) / p[1]
        H = h + K_gap * np.where(LA, ZA, ZB)
        return lin * z + 0.5 * (H - np.conj(H) * z * z)

    sub = int(round(1.0 / DT)); out = np.empty((n_hours + 1, z0.size), complex); z = z0.copy(); out[0] = z
    for k in range(1, n_hours + 1):
        for _ in range(sub):
            k1 = F(z); k2 = F(z + 0.5 * DT * k1); k3 = F(z + 0.5 * DT * k2); k4 = F(z + DT * k3)
            z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(z)):
            return None
        out[k] = z
    return out


def segments(d):
    v0, v1 = S.valid_window(int(d["T"]))
    t1, t2 = int(d["ttx"]), int(d["wash"])
    return [("pre", v0, t1, "K_pre"), ("ttx", t1 + S.TRANSIENT_H, t2, "K_ttx"), ("wash", t2 + S.TRANSIENT_H, v1, "K_wash")]


def meanfields(model, z):
    w, cl, p = model["w"], model["lobe"], model["p"]
    return z @ w, (z[:, cl == 0] @ w[cl == 0]) / p[0], (z[:, cl == 1] @ w[cl == 1]) / p[1]


def loss(x, d, model, n_pulse=N_PULSE, detail=False):
    par = unpack(x); err, n, fterm, pred = 0.0, 0, 0.0, {}
    for name, t0, t1, key in segments(d):
        z0 = LF.initial_state(model, d["PH"][t0])
        z = simulate(model, z0, par[key], par["K_gap"], par["psi"], t1 - t0 - 1, n_pulse)
        if z is None:
            return 1e3
        Zg, ZA, ZB = meanfields(model, z)
        E = np.exp(1j * d["PH"][t0:t1]); Dg = E.mean(1)
        DA, DB = E[:, d["lobe"] == 0].mean(1), E[:, d["lobe"] == 1].mean(1)
        e = ((np.abs(ZA) - np.abs(DA)) ** 2 + (np.abs(ZB) - np.abs(DB)) ** 2 + (np.abs(Zg) - np.abs(Dg)) ** 2) / 3
        err += e.sum(); n += e.size
        if name != "ttx":
            fterm += ((LF.collective_frequency(Zg) - LF.collective_frequency(Dg)) / SIGMA_NU) ** 2
        pred[name] = (t0, Zg, ZA, ZB)
    L = (err + fterm) / n
    return (L, err / n, pred, par) if detail else L


def tag(n_pulse):
    return "" if n_pulse == N_PULSE else f"_n{n_pulse}"


def _job(args):
    i, kind, n_pulse = args
    out = os.path.join(OUT, f"scn{i}_{kind}{tag(n_pulse)}.npy")
    if os.path.exists(out):
        return out, 0.0, None
    t0 = time.time()
    d = LF.slice_data(i); model = build(d, kind)
    best = None
    for s in STARTS:
        x0 = pack(s["K"], 0.1 * s["K"], s["K"], s["gap"], s["psi"])
        r = minimize(loss, x0, args=(d, model, n_pulse), method="Nelder-Mead",
                     options=dict(maxiter=2500, xatol=1e-3, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    L, mse, pred, par = loss(best.x, d, model, n_pulse, detail=True)
    np.save(out, dict(slice=i, kind=kind, n_pulse=n_pulse, params=par, x=best.x, loss=L, mse_R=mse, pred=pred,
                      model=model), allow_pickle=True)
    return out, time.time() - t0, (par, L, mse, model)


def run(slices, procs, n_pulse):
    for i in slices:
        LF.slice_data(i)
    jobs = [(i, k, n_pulse) for i in slices for k in KINDS]
    t0 = time.time()
    with Pool(min(procs, len(jobs))) as p:
        for out, dt, res in p.imap_unordered(_job, jobs, chunksize=1):
            if res is None:
                print(f"[fit] {os.path.basename(out)} exists", flush=True); continue
            par, L, mse, model = res
            print(f"[fit] {os.path.basename(out)}: M = {np.bincount(model['lobe']).tolist()}, "
                  f"K_syn pre/ttx/wash = {par['K_pre']:.4f}/{par['K_ttx']:.4f}/{par['K_wash']:.4f}, "
                  f"K_gap = {par['K_gap']:.4f}, ψ = {par['psi']:.2f} h, RMSE(R) = {np.sqrt(mse):.3f}, loss = {L:.4f} "
                  f"({dt / 60:.1f} min; {(time.time() - t0) / 60:.1f} min elapsed)", flush=True)


def figure(slices, n_pulse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C = {"floor": "#eb6834", "nofloor": "#2a78d6"}; CL = ["#1b7837", "#762a83"]
    fig, axs = plt.subplots(4, len(slices), figsize=(3.1 * len(slices), 10.5), layout="constrained", squeeze=False)
    cd = 24 / (2 * np.pi)
    for j, i in enumerate(slices):
        d = LF.slice_data(i); T = int(d["T"])
        R = {k: np.load(os.path.join(OUT, f"scn{i}_{k}{tag(n_pulse)}.npy"), allow_pickle=True).item()
             for k in KINDS if os.path.exists(os.path.join(OUT, f"scn{i}_{k}{tag(n_pulse)}.npy"))}
        for row, lob in ((0, 0), (1, 1)):                         # per-lobe distributions + fits
            a = axs[row, j]; om = d["om"][d["lobe"] == lob]
            g = np.linspace(d["om"].min() - 0.01, d["om"].max() + 0.01, 800)
            a.hist(om * cd, bins=30, density=True, color="0.82")
            for k, r in R.items():
                m = r["model"]; sel = m["lobe"] == lob; ww = m["w"][sel] / m["w"][sel].sum()
                pdf = (ww[:, None] * m["De"][sel, None] / np.pi / ((g[None] - m["Om"][sel, None]) ** 2 + m["De"][sel, None] ** 2)).sum(0)
                a.plot(g * cd, pdf / cd, color=C[k], lw=1.0, label=f"{k}, M = {sel.sum()}")
            a.set_title(f"SCN{i}, lobe {'AB'[lob]} (n = {om.size})", fontsize=8); a.legend(fontsize=6, frameon=False)
            a.set_xlabel("intrinsic frequency (cycles/day)", fontsize=7)
        v0, v1 = S.valid_window(T); t = np.arange(T) / 24
        E = np.exp(1j * d["PH"])
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
                   f"ψ {p['psi']:.1f} h; RMSE {np.sqrt(r['mse_R']):.3f}", transform=c.transAxes, fontsize=5.2, color=C[k])
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
    out = os.path.join(OUT, f"scn_pulse_fit{tag(n_pulse)}")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", list(S.SLICES), lambda s_: [int(v) for v in s_.split(",")])
    n_pulse = arg("--n-pulse", N_PULSE, int)
    if "fit" in sys.argv:
        run(slices, arg("--procs", os.cpu_count(), int), n_pulse)
    if "figure" in sys.argv:
        figure(slices, n_pulse)
