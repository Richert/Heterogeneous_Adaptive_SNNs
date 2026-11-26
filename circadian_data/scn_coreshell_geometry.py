r"""
Core vs shell geometry: mirror-symmetric planar and offset-radial models of the TTX frequencies (two-column PNAS)
===============================================================================================================

Anatomy (2026-11-25 literature check): each SCN nucleus has its own ventral(-lateral) core (VIP/GRP, retinorecipient,
on the optic chiasm) wrapped dorsally and medially by the AVP shell; the two nuclei are mirror images about the third
ventricle. Recordings are not centred on the core, and the image orientation is unknown. So per slice a MIRROR frame:
e_x = unit vector between the lobe centroids, e_y ⊥ e_x (the shared, sign-unknown "dorsoventral" axis); per lobe L the
local coordinates x' = s_L (p − c_L)·e_x (s_L = ±1, so x' > 0 = LATERAL in both lobes), y' = (p − c_L)·e_y, both in units
of the lobe's RMS radius. Mirror-symmetric models are defined in (x', y') with the SAME parameters for both lobes.

Models (core/shell = a split of each lobe at the same quantile q of a scalar coordinate, q ∈ Q_GRID):
  R0  centred radial: coordinate = distance to the lobe centroid (core = inner)
  Pm  mirror planar: coordinate = projection on (cos φ, sin φ) in (x', y'), φ ∈ PHI_GRID (core = low side, i.e. the
      core lies in the direction −(cos φ, sin φ))
  Rm  mirror offset radial: coordinate = distance to a centre (a, b) in (x', y') anywhere in CENTRE_GRID² (incl. outside
      the recorded area; the quantile split keeps the boundary inside the recorded cells); Pm = centre at infinity
Score: step function (core mean | shell mean) of the lobe-centred TTX frequency (winsorised at ±WINS c/d); 10-fold
cross-validated R² (the parameters are re-selected on every training fold), so model flexibility is not rewarded;
significance: within-lobe permutations of the frequencies (N_PERM), the whole CV fit repeated (calibrates each model's
flexibility); pooled over slices by the sum of the CV R².
LABELLING (2026-11-25): the split itself is symmetric; SHELL := the FASTER group (isolated dorsal SCN has the shorter
period, Noguchi et al. 2004), checked independently against Abel et al.'s MIC degree (shell = sparsely connected).
Checks: angle of the fitted core direction to the ±e_y axis (anatomy: core ventral, possibly ventrolateral); Abel et
al.'s MIC degree in core vs shell (matched cells). Distributions: naive noise-floor fits of core and shell of the best
mirror model (B_BOOT bootstrap).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_coreshell_geometry.py [--slices 1,2,3,4,5] [--perm 200]
        [--boot 30] [--procs 8]
Output: data_paths.MPMF/circadian_data/writeup/scn_core_shell_geometry.{png,pdf,json}
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree
import data_paths as dp
import scn_common as S
import scn_coreshell_frequency as CF
import scn_coreshell_targets as CT
import scn_deconvolution as DC

OUT = dp.ensure(os.path.join(S.OUT, "writeup"))
Q_GRID = np.round(np.arange(0.2, 0.801, 0.05), 2)
PHI_GRID = np.deg2rad(np.arange(0, 360, 5))
CENTRE_GRID = np.linspace(-3, 3, 25)
WINS = 0.1
N_FOLD = 10
N_PERM = 200
B_BOOT = 30
MODELS = ("R0", "Pm", "Rm")
MNAME = dict(R0="centred radial", Pm="mirror planar", Rm="mirror offset radial")
C_CORE, C_SHELL = CF.C_CORE, CF.C_SHELL


def frame(d):
    c0, c1 = d["cen"][0], d["cen"][1]
    ex = (c1 - c0) / np.linalg.norm(c1 - c0); ey = np.array([-ex[1], ex[0]])
    xl = np.zeros(d["f"].size); yl = np.zeros(d["f"].size)
    for L, s in ((0, -1.0), (1, 1.0)):
        m = d["lobe"] == L; v = d["xy"][m] - d["cen"][L]; rms = np.sqrt(np.mean(np.sum(v ** 2, 1)))
        xl[m] = s * (v @ ex) / rms; yl[m] = (v @ ey) / rms
    return xl, yl, ex, ey


def split_by_quantile(coord, lobe):
    """(len(Q_GRID), N) bool: core = coordinate ≤ the lobe's q-quantile."""
    M = np.zeros((Q_GRID.size, coord.size), bool)
    for L in (0, 1):
        m = lobe == L; qs = np.quantile(coord[m], Q_GRID)
        M[:, m] = coord[m][None] <= qs[:, None]
    return M


def partitions(d, model):
    xl, yl = d["xl"], d["yl"]
    if model == "R0":
        coords = [np.hypot(xl, yl)]; params = [dict()]
    elif model == "Pm":
        coords = [np.cos(p) * xl + np.sin(p) * yl for p in PHI_GRID]; params = [dict(phi=float(p)) for p in PHI_GRID]
    else:
        coords, params = [], []
        for a in CENTRE_GRID:
            for b in CENTRE_GRID:
                coords.append(np.hypot(xl - a, yl - b)); params.append(dict(a=float(a), b=float(b)))
    Ms, P = [], []
    for c, pr in zip(coords, params):
        Ms.append(split_by_quantile(c, d["lobe"]))
        P += [dict(pr, q=float(q)) for q in Q_GRID]
    return np.concatenate(Ms).astype(np.float32), P


def sse_select(M, y, idx):
    """Best partition on the cells idx (min SSE of the two-group step function): index, core mean, shell mean."""
    Mi = M[:, idx]; yi = y[idx]
    n_c = Mi.sum(1); n_s = idx.size - n_c
    s_c = Mi @ yi; s_s = yi.sum() - s_c
    ok = (n_c >= 5) & (n_s >= 5)
    sse = np.where(ok, -(s_c ** 2 / np.maximum(n_c, 1) + s_s ** 2 / np.maximum(n_s, 1)), np.inf)
    k = int(np.argmin(sse))
    return k, s_c[k] / n_c[k], s_s[k] / n_s[k]


def cv_r2(M, y, folds):
    sse = sst = 0.0
    for f in range(N_FOLD):
        tr = np.where(folds != f)[0]; te = np.where(folds == f)[0]
        k, mc, ms = sse_select(M, y, tr)
        pred = np.where(M[k, te] > 0.5, mc, ms)
        sse += np.sum((y[te] - pred) ** 2); sst += np.sum((y[te] - y[tr].mean()) ** 2)
    return 1 - sse / sst


def _perm_job(args):
    i, model, seed, n = args
    d = DATA[i]; M, _ = PARTS[(i, model)]
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        y = d["y"].copy()
        for L in (0, 1):
            m = np.where(d["lobe"] == L)[0]; y[m] = y[rng.permutation(m)]
        out.append(cv_r2(M, y, d["folds"]))
    return i, model, out


DATA, PARTS = {}, {}


def prepare(slices):
    for i in slices:
        d = CF.slice_data(i)
        d["xl"], d["yl"], d["ex"], d["ey"] = frame(d)
        d["y"] = np.clip(d["fc"], -WINS, WINS)
        rng = np.random.default_rng(100 + i); d["folds"] = rng.permutation(np.arange(d["f"].size) % N_FOLD)
        DATA[i] = d
        for model in MODELS:
            PARTS[(i, model)] = partitions(d, model)


def mic_degree(i, d):
    raw = os.path.join(S.RAW, "ttx_wash_trajectories"); mic_dir = os.path.join(S.RAW, "raw_mic_scores")
    wl = np.loadtxt(os.path.join(raw, f"scn{i}_TTXwash_locs.csv"), delimiter=",")
    Mx = np.loadtxt(os.path.join(mic_dir, f"mic_indiv_scn{i}.csv"), delimiter=",")
    A = Mx > CT.MIC_THRESH[i]; np.fill_diagonal(A, False); k = A.sum(1)
    dist, idx = cKDTree(wl).query(d["xy"])
    return np.where(dist < CT.MATCH_PX, k[idx], -1)


def _fit_job(args):
    key, b, x, se = args
    rng = np.random.default_rng(abs(hash((key, b))) % (2 ** 31))
    if b > 0:
        ix = rng.integers(0, x.size, x.size); x, se = x[ix], se[ix]
    return key, b, DC.fit_mixture(x, se, {}, rng, deconv=False)


def main(slices, n_perm, n_boot, procs):
    t_ = time.time(); prepare(slices)
    res = dict(slices={})
    for i in slices:
        d = DATA[i]; r = {}
        for model in MODELS:
            M, P = PARTS[(i, model)]
            k, mc, ms = sse_select(M, d["y"], np.arange(d["f"].size))
            sgn = 1 if mc <= ms else -1                           # shell := faster group
            core = (M[k] > 0.5) if sgn == 1 else ~(M[k] > 0.5)
            r[model] = dict(cv_r2=float(cv_r2(M, d["y"], d["folds"])), params=P[k], sign=sgn,
                            core_mean=float(min(mc, ms)), shell_mean=float(max(mc, ms)), core=core)
        res["slices"][str(i)] = r
    print(f"[cv] done in {time.time() - t_:.0f} s", flush=True)
    jobs = [(i, m, 1000 * i + 17 * j + MODELS.index(m), n_perm // procs + (1 if j < n_perm % procs else 0))
            for i in slices for m in MODELS for j in range(procs)]
    with Pool(procs) as pool:
        R = pool.map(_perm_job, jobs, chunksize=1)
    null = {}
    for i, m, out in R:
        null.setdefault((i, m), []).extend(out)
    print(f"[perm] done in {time.time() - t_:.0f} s", flush=True)
    for m in MODELS:
        obs = sum(res["slices"][str(i)][m]["cv_r2"] for i in slices)
        nul = np.sum([np.array(null[(i, m)][:n_perm]) for i in slices], axis=0)
        res[f"pooled_{m}"] = dict(sum_cv_r2=float(obs), p=float((np.sum(nul >= obs) + 1) / (n_perm + 1)))
    for i in slices:
        d = DATA[i]; r = res["slices"][str(i)]
        kdeg = mic_degree(i, d)
        for m in MODELS:
            nl = np.array(null[(i, m)]); r[m]["p"] = float((np.sum(nl >= r[m]["cv_r2"]) + 1) / (nl.size + 1))
            r[m]["null95"] = float(np.percentile(nl, 95))
            c = r[m]["core"]; mk = kdeg >= 0
            r[m]["mic_mean_k"] = [float(kdeg[mk & c].mean()), float(kdeg[mk & ~c].mean())]
        best = max(("Pm", "Rm"), key=lambda m: r[m]["cv_r2"]); r["best"] = best
        pr = r[best]["params"]
        sg = r[best]["sign"]
        if best == "Pm":                                                     # direction towards the core
            u = -sg * np.array([np.cos(pr["phi"]), np.sin(pr["phi"])])
        else:
            u = sg * np.array([pr["a"], pr["b"]]); u = u / max(np.linalg.norm(u), 1e-9)
        r["core_dir"] = u.tolist(); r["angle_to_dv_deg"] = float(np.degrees(np.arccos(min(abs(u[1]), 1.0))))
        print(f"SCN{i}: " + " | ".join(f"{m} CV R² {r[m]['cv_r2']:+.3f} (p {r[m]['p']:.3f})" for m in MODELS) +
              f" | best mirror {best} {pr} | core dir (lat, dv) {np.round(u, 2).tolist()}, angle to DV axis "
              f"{r['angle_to_dv_deg']:.0f}° | core/shell mean {r[best]['core_mean']:+.3f}/{r[best]['shell_mean']:+.3f} | "
              f"MIC k core/shell {np.round(r[best]['mic_mean_k'], 1).tolist()}", flush=True)
    for m in MODELS:
        print(f"[pooled] {m}: Σ CV R² {res['pooled_' + m]['sum_cv_r2']:.3f}, p {res['pooled_' + m]['p']:.3f}")
    jobs = []
    for i in slices:
        d = DATA[i]; c = res["slices"][str(i)][res["slices"][str(i)]["best"]]["core"]
        for g, m in (("core", c), ("shell", ~c)):
            jobs += [((i, g), b, d["f"][m], d["se"][m]) for b in range(n_boot + 1)]
    with Pool(procs) as pool:
        F = pool.map(_fit_job, jobs, chunksize=1)
    fits = {}
    for (i, g), b, f in F:
        fits.setdefault(str(i), {}).setdefault(g, {})[str(b)] = f
    for i in slices:
        for g in ("core", "shell"):
            f0 = fits[str(i)][g]["0"]; Ms = [fits[str(i)][g][str(b)]["M"] for b in range(1, n_boot + 1)]
            res["slices"][str(i)][f"fit_{g}"] = dict(M=f0["M"], mu=f0["mu"], gamma=f0["gamma"], w=f0["w"],
                                                     M_support=float(np.mean(np.array(Ms) == f0["M"])))
            print(f"  SCN{i} {g:5s}: M* = {f0['M']} ({np.mean(np.array(Ms) == f0['M']):.0%}) μ "
                  f"{np.round(f0['mu'], 3).tolist()} γ {np.round(f0['gamma'], 4).tolist()} w {np.round(f0['w'], 2).tolist()}")
    out = json.loads(json.dumps(res, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    for i in slices:
        for m in MODELS:
            out["slices"][str(i)][m].pop("core", None)
    json.dump(out, open(os.path.join(OUT, "scn_core_shell_geometry.json"), "w"), indent=1)
    figure(slices, res, fits, null)
    print(f"[done] {time.time() - t_:.0f} s")


def figure(slices, res, fits, null):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from prl_style import set_prl_style, PNAS_DOUBLE, pnas_panel_label
    set_prl_style("pnas")
    n = len(slices)
    fig = plt.figure(figsize=(PNAS_DOUBLE, 6.6))
    gs = GridSpec(4, 1, figure=fig, height_ratios=(1.15, 0.9, 0.95, 1.0), hspace=0.62, left=0.075, right=0.97,
                  top=0.965, bottom=0.06)
    rA = gs[0].subgridspec(1, n + 2, wspace=0.15, width_ratios=[1] * n + [0.05, 0.12])
    rB = gs[1].subgridspec(1, 2, wspace=0.3, width_ratios=(1.4, 1)); rC = gs[2].subgridspec(1, n, wspace=0.3)
    rD = gs[3].subgridspec(1, n, wspace=0.3)
    half = max(0.5 * np.ptp(DATA[i]["xy"], axis=0).max() for i in slices) * 1.05
    gx = np.linspace(0.75, 1.25, 600)
    lor = lambda f: sum(w * gm / np.pi / ((gx - m) ** 2 + gm ** 2) for w, m, gm in zip(f["w"], f["mu"], f["gamma"]))
    for k, i in enumerate(slices):
        d = DATA[i]; r = res["slices"][str(i)]; best = r["best"]; c = r[best]["core"]
        a = fig.add_subplot(rA[k])
        sc = a.scatter(d["xy"][:, 0], d["xy"][:, 1], c=d["f"] - np.median(d["f"]), cmap="RdBu_r", vmin=-0.08,
                       vmax=0.08, s=3, lw=0)
        a.scatter(d["xy"][c, 0], d["xy"][c, 1], s=9, facecolors="none", edgecolors="k", lw=0.25)
        mid = 0.5 * (d["xy"].max(0) + d["xy"].min(0))
        for L, s_ in ((0, -1.0), (1, 1.0)):                                     # arrow: direction towards the core
            u = np.array(r["core_dir"]); v = s_ * u[0] * d["ex"] + u[1] * d["ey"]
            a.annotate("", xy=d["cen"][L] + 0.45 * half * v, xytext=d["cen"][L],
                       arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
        a.set_xlim(mid[0] - half, mid[0] + half); a.set_ylim(mid[1] + half, mid[1] - half)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"SCN{i} ({MNAME[best].replace('mirror ', '')})", fontsize=6.5, fontweight="bold")
        if k == 0:
            a.text(0.02, 0.02, "○ core (slow)\narrow: towards core", transform=a.transAxes, fontsize=5)
        if k == 0:
            pnas_panel_label(a, "A", dx=-12)
        cpan = fig.add_subplot(rC[k])
        pr = r[best]["params"]
        coord = (np.cos(pr["phi"]) * d["xl"] + np.sin(pr["phi"]) * d["yl"]) if best == "Pm" else \
            np.hypot(d["xl"] - pr["a"], d["yl"] - pr["b"])
        coord = r[best]["sign"] * coord                                     # increases from core to shell
        rk = np.zeros(coord.size)
        for L in (0, 1):
            m = d["lobe"] == L; rk[m] = (np.argsort(np.argsort(coord[m])) / max(m.sum() - 1, 1))
        q_line = pr["q"] if r[best]["sign"] == 1 else 1 - pr["q"]
        cpan.plot(rk[c], d["fc"][c], ".", ms=1.5, color=C_CORE, alpha=0.5)
        cpan.plot(rk[~c], d["fc"][~c], ".", ms=1.5, color=C_SHELL, alpha=0.5)
        qb = np.linspace(0, 1, 9); qm = 0.5 * (qb[1:] + qb[:-1])
        cpan.plot(qm, [np.median(d["fc"][(rk >= lo) & (rk <= hi)]) for lo, hi in zip(qb[:-1], qb[1:])], "k-o", ms=2,
                  lw=0.9)
        cpan.axvline(q_line, color="k", ls="--", lw=0.6); cpan.axhline(0, color="0.5", lw=0.5)
        cpan.set_ylim(-0.12, 0.12)
        cpan.set_xlabel("rank along core→shell axis")
        if k == 0:
            cpan.set_ylabel("f − lobe median (c/d)"); pnas_panel_label(cpan, "C", dx=-24)
        fa = fig.add_subplot(rD[k]); Fd = fits[str(i)]
        for g, col, m in (("core", C_CORE, c), ("shell", C_SHELL, ~c)):
            fa.hist(d["f"][m], bins=np.linspace(0.75, 1.25, 36), density=True, histtype="step", color=col, lw=0.6,
                    alpha=0.7)
            P = np.array([lor(Fd[g][b]) for b in Fd[g] if b != "0"])
            fa.fill_between(gx, np.percentile(P, 2.5, 0), np.percentile(P, 97.5, 0), color=col, alpha=0.2, lw=0)
            fa.plot(gx, lor(Fd[g]["0"]), color=col, lw=1.1, label=f"{g} (M*={Fd[g]['0']['M']})")
        fa.set_xlim(0.75, 1.25); fa.set_yticks([]); fa.set_xlabel("TTX frequency (c/d)")
        fa.legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=1)
        if k == 0:
            fa.set_ylabel("density"); pnas_panel_label(fa, "D", dx=-14)
    cb = fig.colorbar(sc, cax=fig.add_subplot(rA[n]), ticks=[-0.08, 0, 0.08])
    cb.ax.set_yticklabels(["−0.08", "0", "+0.08"]); cb.set_label("f − slice median (c/d)")
    b = fig.add_subplot(rB[0]); xs = np.arange(n); wd = 0.26
    for j, (m, col) in enumerate(zip(MODELS, ("0.6", "#2166ac", "#b2182b"))):
        v = [res["slices"][str(i)][m]["cv_r2"] for i in slices]
        nv = [res["slices"][str(i)][m]["null95"] for i in slices]
        b.bar(xs + (j - 1) * wd, v, width=wd, color=col, label=f"{MNAME[m]} (pooled p = {res['pooled_' + m]['p']:.3f})")
        b.plot(xs + (j - 1) * wd, nv, "_", color="k", ms=6, mew=0.8)
    b.axhline(0, color="0.5", lw=0.5); b.set_xticks(xs); b.set_xticklabels([f"SCN{i}" for i in slices])
    b.set_ylabel("cross-validated R²"); b.legend(frameon=False, fontsize=5.5, loc="upper right")
    pnas_panel_label(b, "B", dx=-24)
    b2 = fig.add_subplot(rB[1])
    ang = [res["slices"][str(i)]["angle_to_dv_deg"] for i in slices]
    kc = [res["slices"][str(i)][res["slices"][str(i)]["best"]]["mic_mean_k"] for i in slices]
    b2.bar(xs - 0.18, [x[0] for x in kc], width=0.34, color=C_CORE, label="core")
    b2.bar(xs + 0.18, [x[1] for x in kc], width=0.34, color=C_SHELL, label="shell")
    b2.set_xticks(xs); b2.set_xticklabels([f"SCN{i}\n{a_:.0f}°" for i, a_ in zip(slices, ang)])
    b2.set_ylabel("mean MIC degree"); b2.set_ylim(0, max(max(x) for x in kc) * 1.25)
    b2.legend(frameon=False, fontsize=5.5, loc="upper left")
    b2.set_xlabel("angle of core→shell axis to the midline axis", fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"scn_core_shell_geometry.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    main(arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")]), arg("--perm", N_PERM, int),
         arg("--boot", B_BOOT, int), arg("--procs", os.cpu_count(), int))
