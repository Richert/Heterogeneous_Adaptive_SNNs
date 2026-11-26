r"""
Core vs shell: intrinsic (TTX) frequencies and their errors by lobe and by position within the lobe (two-column PNAS)
===================================================================================================================

Data (epoch-wise peak method, cycle-slip handling; scn_frequency_figure): per cell the TTX frequency f_i (c/d), its SE
and its position; lobes = left / right SCN (scn_targets). Position within the lobe: radial RANK q_i ∈ [0, 1] of the
distance to the lobe centroid (0 = centre, 1 = edge), per lobe (orientation-free, balanced splits).

Core/shell test (location): for thresholds q* ∈ Q_GRID, core = {q ≤ q*}, shell = {q > q*}; statistic = Mann–Whitney z of
the LOBE-CENTRED frequencies (f_i − lobe median), shell vs core (> 0: shell faster). The peak-timing error is unbiased,
so it cannot create a location shift. Significance: permutation of q within each lobe (N_PERM), max |z| over thresholds
(scan-corrected); per slice and POOLED over slices (sum of the slice z at each threshold). The common threshold q* =
argmax of the pooled |z| defines core and shell in every slice.
Spread: per group the naive noise-floor width γ (upper bound, includes the error) and the median SE; the error-corrected
excess spread s_int = sqrt(max(σ_rob² − ⟨SE²⟩, 0)), σ_rob = 1.4826 MAD, with bootstrap CIs.
Distributions: naive noise-floor Lorentzian-mixture fits (scn_deconvolution.fit_mixture, deconv=False) of core and shell
per slice, B_BOOT bootstrap resamples (95 % bands).

Figure:
  A  cell maps, colour = f_i − slice median; circles: the core boundary (q* quantile radius) of each lobe
  B  f_i − lobe median vs q (binned medians and IQR per lobe; dots: cells); text: lobe A vs B medians (Mann–Whitney)
  C  SE vs q (binned medians per lobe)
  D  threshold scan (z vs q*, per slice and pooled; shaded: 95 % permutation envelope of the pooled max)
  E  excess spread s_int, core vs shell, per slice (bootstrap CI)
  F  core vs shell TTX-frequency distributions per slice: histograms + naive fits (95 % bootstrap bands), M*

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_coreshell_frequency.py [--slices 1,2,3,4,5] [--perm 2000]
        [--boot 30] [--procs 8]
Output: data_paths.MPMF/circadian_data/writeup/scn_core_shell_frequency.{png,pdf,json}
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from multiprocessing import Pool
from scipy.stats import mannwhitneyu, rankdata
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_frequency_figure as FF
import scn_deconvolution as DC

OUT = dp.ensure(os.path.join(S.OUT, "writeup"))
METHOD = "epoch"
Q_GRID = np.round(np.arange(0.3, 0.701, 0.05), 2)
N_PERM = 2000
B_BOOT = 30
C_CORE, C_SHELL = "#08519c", "#e6550d"
C_LOBE = ("#1b7837", "#762a83")


def slice_data(i):
    g = TG.load(i); _, locs = S.load(i)
    r = FF.analyse(i, METHOD); f, se = r["f"]["ttx"], r["se"]["ttx"]
    ok = np.isfinite(f) & np.isfinite(se)
    xy = locs[g["cells"]].astype(float)[ok]; lobe = np.asarray(g["lobe"])[ok]; f, se = f[ok], se[ok]
    q = np.zeros(f.size); rad = np.zeros(f.size); cen = {}
    for L in (0, 1):
        m = lobe == L; c = xy[m].mean(0); d = np.linalg.norm(xy[m] - c, axis=1)
        q[m] = (rankdata(d) - 1) / max(m.sum() - 1, 1); rad[m] = d; cen[L] = c
    fc = f.copy()
    for L in (0, 1):
        fc[lobe == L] -= np.median(f[lobe == L])
    return dict(f=f, se=se, fc=fc, q=q, rad=rad, xy=xy, lobe=lobe, cen=cen)


def zscan(fc, q):
    z = np.zeros(Q_GRID.size)
    for k, qs in enumerate(Q_GRID):
        a, b = fc[q > qs], fc[q <= qs]
        U = mannwhitneyu(a, b, alternative="two-sided").statistic
        n1, n2 = a.size, b.size
        z[k] = (U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return z


def permute_q(q, lobe, rng):
    qp = q.copy()
    for L in (0, 1):
        m = np.where(lobe == L)[0]; qp[m] = q[rng.permutation(m)]
    return qp


def s_int(f, se):
    sig = 1.4826 * np.median(np.abs(f - np.median(f)))
    return float(np.sqrt(max(sig ** 2 - np.mean(se ** 2), 0.0)))


def _fit_job(args):
    key, b, x, se = args
    rng = np.random.default_rng(hash((key, b)) % (2 ** 31))
    if b > 0:
        idx = rng.integers(0, x.size, x.size); x, se = x[idx], se[idx]
    return key, b, DC.fit_mixture(x, se, {}, rng, deconv=False)


def main(slices, n_perm, n_boot, procs):
    D = {i: slice_data(i) for i in slices}
    rng = np.random.default_rng(5)
    Z = {i: zscan(D[i]["fc"], D[i]["q"]) for i in slices}
    zpool = np.sum([Z[i] for i in slices], axis=0)
    Zp = {i: np.empty((n_perm, Q_GRID.size)) for i in slices}
    for p in range(n_perm):
        for i in slices:
            Zp[i][p] = zscan(D[i]["fc"], permute_q(D[i]["q"], D[i]["lobe"], rng))
    pool_null = np.sum([Zp[i] for i in slices], axis=0)
    p_pool = float((np.sum(np.abs(pool_null).max(1) >= np.abs(zpool).max()) + 1) / (n_perm + 1))
    qstar = float(Q_GRID[int(np.argmax(np.abs(zpool)))])
    res = dict(q_grid=Q_GRID.tolist(), z_pooled=zpool.tolist(), p_pooled=p_pool, q_star=qstar, slices={})
    print(f"[scan] pooled: max |z| = {np.abs(zpool).max():.2f} at q* = {qstar:.2f} (sign {np.sign(zpool[np.argmax(np.abs(zpool))]):+.0f}), "
          f"scan-corrected p = {p_pool:.4f}")
    jobs = []
    for i in slices:
        d = D[i]; z = Z[i]
        p_i = float((np.sum(np.abs(Zp[i]).max(1) >= np.abs(z).max()) + 1) / (n_perm + 1))
        k = int(np.argmin(np.abs(Q_GRID - qstar)))
        p_at = float((np.sum(np.abs(Zp[i][:, k]) >= abs(z[k])) + 1) / (n_perm + 1))
        core = d["q"] <= qstar; shell = ~core
        boot = lambda m: [s_int(d["f"][m][ix], d["se"][m][ix]) for ix in
                          (rng.integers(0, m.sum(), m.sum()) for _ in range(500))]
        bc, bs = boot(core), boot(shell)
        lob = mannwhitneyu(d["f"][d["lobe"] == 0], d["f"][d["lobe"] == 1])
        r = dict(n=[int(core.sum()), int(shell.sum())], z=z.tolist(), p_scan=p_i, z_at_qstar=float(z[k]),
                 p_at_qstar=p_at,
                 median=[float(np.median(d["f"][core])), float(np.median(d["f"][shell]))],
                 median_se=[float(np.median(d["se"][core])), float(np.median(d["se"][shell]))],
                 s_int=[s_int(d["f"][core], d["se"][core]), s_int(d["f"][shell], d["se"][shell])],
                 s_int_ci=[np.percentile(bc, [2.5, 97.5]).tolist(), np.percentile(bs, [2.5, 97.5]).tolist()],
                 lobe_median=[float(np.median(d["f"][d["lobe"] == L])) for L in (0, 1)],
                 lobe_median_se=[float(np.median(d["se"][d["lobe"] == L])) for L in (0, 1)],
                 lobe_s_int=[s_int(d["f"][d["lobe"] == L], d["se"][d["lobe"] == L]) for L in (0, 1)],
                 lobe_p=float(lob.pvalue))
        res["slices"][str(i)] = r
        print(f"SCN{i}: n core/shell {r['n']} | z(q*) {r['z_at_qstar']:+.2f} (p {p_at:.3f}; scan p {p_i:.3f}) | median "
              f"core/shell {r['median'][0]:.3f}/{r['median'][1]:.3f} c/d | median SE {r['median_se'][0]:.4f}/"
              f"{r['median_se'][1]:.4f} | excess spread {r['s_int'][0]:.4f}/{r['s_int'][1]:.4f} | lobes A/B median "
              f"{r['lobe_median'][0]:.3f}/{r['lobe_median'][1]:.3f} (p {r['lobe_p']:.3f}), excess spread "
              f"{r['lobe_s_int'][0]:.4f}/{r['lobe_s_int'][1]:.4f}", flush=True)
        for gname, m in (("core", core), ("shell", shell)):
            for b in range(n_boot + 1):
                jobs.append(((i, gname), b, d["f"][m], d["se"][m]))
    with Pool(procs) as pool:
        R = pool.map(_fit_job, jobs, chunksize=1)
    fits = {}
    for (i, gname), b, f in R:
        fits.setdefault(str(i), {}).setdefault(gname, {})[str(b)] = f
    for i in slices:
        for gname in ("core", "shell"):
            F = fits[str(i)][gname]; f0 = F["0"]; Ms = np.array([F[str(b)]["M"] for b in range(1, n_boot + 1)])
            res["slices"][str(i)][f"fit_{gname}"] = dict(M=f0["M"], mu=f0["mu"], gamma=f0["gamma"], w=f0["w"],
                                                          M_support=float(np.mean(Ms == f0["M"])))
            print(f"  SCN{i} {gname:5s}: M* = {f0['M']} ({np.mean(Ms == f0['M']):.0%}) μ {np.round(f0['mu'], 3).tolist()}"
                  f" γ {np.round(f0['gamma'], 4).tolist()} w {np.round(f0['w'], 2).tolist()}", flush=True)
    json.dump(res, open(os.path.join(OUT, "scn_core_shell_frequency.json"), "w"), indent=1)
    figure(slices, D, Z, zpool, pool_null, res, fits)


def figure(slices, D, Z, zpool, pool_null, res, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle
    from prl_style import set_prl_style, PNAS_DOUBLE, pnas_panel_label
    set_prl_style("pnas")
    n = len(slices); qstar = res["q_star"]
    fig = plt.figure(figsize=(PNAS_DOUBLE, 8.6))
    gs = GridSpec(5, 1, figure=fig, height_ratios=(1.1, 0.95, 0.75, 0.95, 1.0), hspace=0.62, left=0.075,
                  right=0.975, top=0.97, bottom=0.05)
    rA = gs[0].subgridspec(1, n + 2, wspace=0.15, width_ratios=[1] * n + [0.05, 0.12])
    rB = gs[1].subgridspec(1, n, wspace=0.3); rC = gs[2].subgridspec(1, n, wspace=0.3)
    rDE = gs[3].subgridspec(1, 2, wspace=0.3, width_ratios=(1.2, 1)); rF = gs[4].subgridspec(1, n, wspace=0.3)
    qb = np.linspace(0, 1, 9); qm = 0.5 * (qb[1:] + qb[:-1])
    half = max(0.5 * np.ptp(D[i]["xy"], axis=0).max() for i in slices) * 1.05     # common map box (same scale)
    gx = np.linspace(0.75, 1.25, 600)
    lor = lambda f: sum(w * gm / np.pi / ((gx - m) ** 2 + gm ** 2) for w, m, gm in zip(f["w"], f["mu"], f["gamma"]))
    for k, i in enumerate(slices):
        d = D[i]; r = res["slices"][str(i)]
        a = fig.add_subplot(rA[k])
        sc = a.scatter(d["xy"][:, 0], d["xy"][:, 1], c=d["f"] - np.median(d["f"]), cmap="RdBu_r", vmin=-0.08,
                       vmax=0.08, s=3, lw=0)
        for L in (0, 1):
            m = d["lobe"] == L; rq = np.quantile(d["rad"][m], qstar)
            a.add_patch(Circle(d["cen"][L], rq, fill=False, ls="--", lw=0.7, color="k"))
        mid = 0.5 * (d["xy"].max(0) + d["xy"].min(0))
        a.set_xlim(mid[0] - half, mid[0] + half); a.set_ylim(mid[1] + half, mid[1] - half)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"SCN{i}", fontweight="bold")
        if k == 0:
            pnas_panel_label(a, "A", dx=-12)
        b = fig.add_subplot(rB[k])
        for L in (0, 1):
            m = d["lobe"] == L
            b.plot(d["q"][m], d["fc"][m], ".", ms=1.2, color=C_LOBE[L], alpha=0.25)
            med = [np.median(d["fc"][m][(d["q"][m] >= lo) & (d["q"][m] <= hi)]) for lo, hi in zip(qb[:-1], qb[1:])]
            iq = [np.percentile(d["fc"][m][(d["q"][m] >= lo) & (d["q"][m] <= hi)], [25, 75]) for lo, hi in
                  zip(qb[:-1], qb[1:])]
            b.fill_between(qm, [x[0] for x in iq], [x[1] for x in iq], color=C_LOBE[L], alpha=0.2, lw=0)
            b.plot(qm, med, "o-", color=C_LOBE[L], ms=2, lw=0.9, label=f"lobe {'AB'[L]}")
        b.axvline(qstar, color="k", ls="--", lw=0.6); b.axhline(0, color="0.5", lw=0.5)
        b.set_ylim(-0.12, 0.12); b.set_xlabel("radial rank in lobe")
        b.text(0.03, 0.97, f"A−B: {r['lobe_median'][0] - r['lobe_median'][1]:+.3f} c/d\np = {r['lobe_p']:.2g}",
               transform=b.transAxes, va="top", fontsize=5.5)
        if k == 0:
            b.set_ylabel("f − lobe median (c/d)"); pnas_panel_label(b, "B", dx=-24)
            b.legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=1)
        c = fig.add_subplot(rC[k])
        for L in (0, 1):
            m = d["lobe"] == L
            med = [np.median(d["se"][m][(d["q"][m] >= lo) & (d["q"][m] <= hi)]) for lo, hi in zip(qb[:-1], qb[1:])]
            c.plot(qm, med, "o-", color=C_LOBE[L], ms=2, lw=0.9)
        c.axvline(qstar, color="k", ls="--", lw=0.6); c.set_ylim(0, None); c.set_xlabel("radial rank in lobe")
        if k == 0:
            c.set_ylabel("median SE (c/d)"); pnas_panel_label(c, "C", dx=-24)
        fa = fig.add_subplot(rF[k]); F = fits[str(i)]
        for gname, col, m in (("core", C_CORE, d["q"] <= qstar), ("shell", C_SHELL, d["q"] > qstar)):
            fa.hist(d["f"][m], bins=np.linspace(0.75, 1.25, 36), density=True, histtype="step", color=col, lw=0.6,
                    alpha=0.7)
            P = np.array([lor(F[gname][b]) for b in F[gname] if b != "0"])
            fa.fill_between(gx, np.percentile(P, 2.5, 0), np.percentile(P, 97.5, 0), color=col, alpha=0.2, lw=0)
            f0 = F[gname]["0"]
            fa.plot(gx, lor(f0), color=col, lw=1.1, label=f"{gname} (M*={f0['M']})")
        fa.set_xlim(0.75, 1.25); fa.set_yticks([]); fa.set_xlabel("TTX frequency (c/d)")
        fa.legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=1)
        if k == 0:
            fa.set_ylabel("density"); pnas_panel_label(fa, "F", dx=-14)
    cb = fig.colorbar(sc, cax=fig.add_subplot(rA[n]), ticks=[-0.08, 0, 0.08])
    cb.ax.set_yticklabels(["−0.08", "0", "+0.08"]); cb.set_label("f − slice median (c/d)")
    dz = fig.add_subplot(rDE[0])
    env = np.percentile(np.abs(pool_null).max(1), 95)
    dz.axhspan(-env, env, color="0.9", lw=0, label="95% null (pooled, max)")
    for i in slices:
        dz.plot(Q_GRID, Z[i], "-", lw=0.8, label=f"SCN{i}")
    dz.plot(Q_GRID, zpool, "k-", lw=1.6, label=f"pooled (p = {res['p_pooled']:.2g})")
    dz.axvline(qstar, color="k", ls="--", lw=0.6); dz.axhline(0, color="0.5", lw=0.5)
    dz.set_xlabel("core/shell threshold q*"); dz.set_ylabel("z (shell faster > 0)")
    dz.set_ylim(-7, 9.5)
    dz.legend(frameon=False, fontsize=5, ncol=4, loc="upper center", columnspacing=0.8, handlelength=1.2)
    pnas_panel_label(dz, "D", dx=-24)
    de = fig.add_subplot(rDE[1]); xs = np.arange(n)
    for dx_, j, col, lab in ((-0.15, 0, C_CORE, "core"), (0.15, 1, C_SHELL, "shell")):
        v = [res["slices"][str(i)]["s_int"][j] for i in slices]
        lo = [v[k_] - res["slices"][str(i)]["s_int_ci"][j][0] for k_, i in enumerate(slices)]
        hi = [res["slices"][str(i)]["s_int_ci"][j][1] - v[k_] for k_, i in enumerate(slices)]
        de.errorbar(xs + dx_, v, yerr=[lo, hi], fmt="o", color=col, ms=3, lw=0.8, capsize=1.5, label=lab)
    de.set_xticks(xs); de.set_xticklabels([f"SCN{i}" for i in slices]); de.set_ylim(0, None)
    de.set_ylabel("error-corrected spread (c/d)"); de.legend(frameon=False, fontsize=5.5)
    pnas_panel_label(de, "E", dx=-24)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"scn_core_shell_frequency.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    main(arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")]), arg("--perm", N_PERM, int),
         arg("--boot", B_BOOT, int), arg("--procs", os.cpu_count(), int))
