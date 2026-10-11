r"""
Double Gaussian (Omel'chenko–Wolfrum, τ = 0.6, δ = 0.1): does a larger M reproduce the
restabilisation of incoherence (α ≈ 1.02–1.16, K ≈ 1–2)?
===================================================================================================

The noise-floor CvM fit picks M = 3, whose incoherence boundary in (K, α) is a flat plateau: the LMMF
then predicts a weakly synchronised state where the network returns to incoherence. Here: plain CvM
fits with FIXED M ∈ M_LIST (warm-started chain, pruned), their incoherence boundaries, and LMMF K-ramps
re-using the network ramps of unimodal_kalpha.py (separate output files; main results untouched).
Figure: rows = M (3 = the main fit, then M_LIST); columns = network classification + LMMF boundary,
spectral RMSE, R along the α = 1.10 ramp. Runtime here: ~2–3 min.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python dgauss_mscan.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from multiprocessing import Pool
import lorentzian_mixture as LM
import unimodal_kalpha as UK
import unimodal_kalpha_figure as UF

NAME = "dgauss"
M_LIST = [5, 7, 9]
A_EX = 1.10                                           # α row for the example ramps


def fits():
    x = UK.sample(NAME); prev = None; out = {}
    for M in range(1, max(M_LIST) + 1):
        m, D = LM.fit_fixed_M(x, M, (1e-4, 1e2), n_restarts=10, seed=1, method="slsqp", init_model=prev)
        m = LM._prune_mixture(m); prev = m
        if M in M_LIST:
            f = dict(w=m.w, Om=m.Omega, De=m.Delta, M=m.M, beta=None, M_nominal=M,
                     D_floor=D / LM.noise_floor(x.size))
            np.save(UK._f(NAME, "fit", f"M{M}") + ".npy", f, allow_pickle=True)
            out[M] = f
            print(f"[fit] M={M}: effective M={m.M}  D/floor={f['D_floor']:.2f}", flush=True)
    return out


def boundaries(fs):
    a_fine = np.linspace(0.0, 1.45, 146)
    for M, f in fs.items():
        kc = np.array([UK.lmmf_Kc(f, a) for a in a_fine])
        np.save(UK._f(NAME, "boundary", f"M{M}") + ".npy", dict(alpha=a_fine, K=kc[:, 0], freq=kc[:, 1]),
                allow_pickle=True)


def _ramp_file(path):
    d = np.load(path)
    direction = "fwd" if "_ramp_fwd_" in path else "bwd"
    alphas = UK.A_GRID[d["ia"]][:, None]; hold = float(d["hold"])
    fs = [UK.load_fit(NAME, f"M{M}") for M in M_LIST]
    zs = [np.full((alphas.size, f["M"]), float(d["R0"]) + 0j) for f in fs]
    n_steps, stride = int(round(hold / UK.DT)), int(round(UK.DTS / UK.DT))
    Rl = np.zeros((len(fs),) + d["R_net"].shape)
    order = range(UK.K_GRID.size) if direction == "fwd" else range(UK.K_GRID.size - 1, -1, -1)
    for ik in order:
        Rl[:, :, ik] = UK._lmmf_step(fs, zs, UK.K_GRID[ik] * np.exp(-1j * alphas), n_steps, stride)
    out = path.replace(".npz", "_Mscan.npz")
    np.savez_compressed(out, R_lmmf=Rl.astype(np.float32), M=np.array(M_LIST), ia=d["ia"])
    return out


def ramps(procs=8):
    files = sorted(os.path.join(UK.OUT, f) for f in os.listdir(UK.OUT)
                   if f.startswith(f"{NAME}_ramp_") and f.endswith(".npz") and "_hold" not in f
                   and "_Mscan" not in f)
    with Pool(procs) as p:
        done = p.map(_ramp_file, files, chunksize=1)
    print(f"[ramps] {len(done)} files", flush=True)


def figure():
    base = UF.analyse(NAME)                            # network + main fits (variant a = M=3)
    net = base["ramps"]
    # assemble the M-scan LMMF traces in the same α order
    Rm = {}
    for d in UK.DIRECTIONS:
        parts = sorted(f for f in os.listdir(UK.OUT) if f.startswith(f"{NAME}_ramp_{d}_c") and f.endswith("_Mscan.npz"))
        P = [np.load(os.path.join(UK.OUT, f)) for f in parts]
        ia = np.concatenate([p["ia"] for p in P]); o = np.argsort(ia)
        Rm[d] = np.concatenate([p["R_lmmf"] for p in P], axis=1)[:, o].astype(float)
    rows = [(3, "a", net["fwd"]["lmmf"][0], net["bwd"]["lmmf"][0])]
    rows += [(M, f"M{M}", Rm["fwd"][j], Rm["bwd"][j]) for j, M in enumerate(M_LIST)]
    ia_ex = int(np.argmin(np.abs(base["alpha"] - A_EX)))
    UF.set_prl_style()
    fig, ax = plt.subplots(len(rows), 3, figsize=(7.0, 1.75 * len(rows)), layout="constrained")
    ext = [UK.K_GRID[0], UK.K_GRID[-1], UK.A_GRID[0], UK.A_GRID[-1]]
    cmap = ListedColormap(UF.CLS_COLORS); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    print(f"{'M':>3s} {'agree':>6s} {'multi net/LMMF':>15s} {'RMSE mean':>10s} {'dip: min α of LMMF boundary on K∈[1.3,1.9]':>45s}")
    ex = np.load(UK._f(NAME, "boundary_exact") + ".npy")
    sel = (ex[:, 0] > 1.3) & (ex[:, 0] < 1.9)
    print(f"{'exact':>3s} {'':>6s} {'':>15s} {'':>10s} {np.abs(ex[sel, 1]).min():45.3f}")
    for r, (M, key, Rf, Rb) in enumerate(rows):
        cls = UF.combine(UK.classify(Rf), UK.classify(Rb))
        rmse = 0.5 * (UF.spectral_rmse(net["fwd"]["net"], Rf) + UF.spectral_rmse(net["bwd"]["net"], Rb))
        bd = np.load(UK._f(NAME, "boundary", key) + ".npy", allow_pickle=True).item()
        if "G" in bd:                                  # restabilisation dip from the full boundary
            G, Kg = bd["G"], bd["K_grid"]; sel_k = (Kg > 1.3) & (Kg < 1.9)
            rows = [i for i in range(G.shape[0]) if (G[i, sel_k] < 0).any() and (G[i, Kg < 1.3] > 0).any()]
            dip = bd["alpha"][min(rows)] if rows else np.nan
        else:
            s = (bd["K"] > 1.3) & (bd["K"] < 1.9)
            dip = bd["alpha"][s].min() if s.any() else np.nan
        print(f"{M:3d} {100 * np.mean(cls == base['cls_net']):5.1f}% {int((base['cls_net'] == 2).sum()):7d}/"
              f"{int((cls == 2).sum()):<7d} {rmse.mean():10.2e} {dip:45.3f}")
        a = ax[r, 0]
        a.imshow(base["cls_net"], origin="lower", aspect="auto", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
        a.contour(UK.K_GRID, base["alpha"], cls, levels=[0.5, 1.5, 2.5], colors="k", linewidths=0.5, linestyles=":")
        a.plot(ex[:, 0], np.abs(ex[:, 1]), "--", color=UF.C_EXACT, lw=1.0)
        (a.contour(bd["K_grid"], bd["alpha"], bd["G"], levels=[0.0], colors=[UF.C_LMMF], linewidths=1.2)
         if "G" in bd else a.plot(bd["K"], bd["alpha"], "-", color=UF.C_LMMF, lw=1.2))
        a.set_xlim(ext[:2]); a.set_ylim(0.8, 1.4); a.set_ylabel(rf"$M={M}$" + "\n" + r"$\alpha$")
        b = ax[r, 1]
        b.imshow(np.log10(rmse + 1e-6), origin="lower", aspect="auto", extent=ext, cmap="Reds",
                 interpolation="nearest", vmin=-3.8, vmax=-1.3)
        b.plot(ex[:, 0], np.abs(ex[:, 1]), "--", color=UF.C_EXACT, lw=1.0)
        (b.contour(bd["K_grid"], bd["alpha"], bd["G"], levels=[0.0], colors=[UF.C_LMMF], linewidths=1.2)
         if "G" in bd else b.plot(bd["K"], bd["alpha"], "-", color=UF.C_LMMF, lw=1.2))
        b.set_xlim(ext[:2]); b.set_ylim(0.8, 1.4)
        c = ax[r, 2]
        K = UK.K_GRID; ns = Rf.shape[-1]
        x = K[:, None] + np.linspace(0, (K[1] - K[0]) * 0.95, ns)[None, :]
        for d, ls, Rl in (("fwd", "-", Rf), ("bwd", "--", Rb)):
            c.plot(x.ravel(), net[d]["net"][ia_ex].ravel(), ls, color=UF.C_NET, lw=0.6)
            c.plot(x.ravel(), Rl[ia_ex].ravel(), ls, color=UF.C_LMMF, lw=0.6)
        c.set_xlim(K[0], K[-1]); c.set_ylim(0, 1); c.set_ylabel("$R$")
        if r == 0:
            a.set_title("network classes (shading), LMMF classes (dotted)", fontsize=6.5)
            b.set_title(r"$\log_{10}$ spectral RMSE", fontsize=6.5)
            c.set_title(rf"ramps at $\alpha={base['alpha'][ia_ex]:.2f}$ (network black, LMMF red)", fontsize=6.5)
        if r == len(rows) - 1:
            for axx in ax[r]:
                axx.set_xlabel(r"coupling $K$")
    out = os.path.join(UK.OUT, "fig_dgauss_Mscan")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg")
    print(f"[saved] {out}.png/.svg")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    fs = fits(); boundaries(fs); ramps(); figure()
    print(f"total {time.time() - t0:.0f}s")
