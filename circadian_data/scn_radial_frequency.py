r"""
Do intrinsic frequencies differ between the centre and the periphery of each SCN lobe?
======================================================================================

Proxy for core vs shell: the distance of each cell from its lobe centroid (pixel coordinates; the data carry no
VIP / AVP labels and the dorsal–ventral orientation of the images is unknown). Per slice and lobe (fit-target cells,
scn_targets; ω_i = phase slope under TTX):
    ρ_signed   Spearman ρ(ω_i, r_i)                       (> 0: peripheral cells faster)
    ρ_spread   Spearman ρ(|ω_i − median ω_L|, r_i)         (> 0: peripheral cells further from the lobe's typical ω)
    Δ_io       mean ω of the outer tercile − inner tercile (min/day of period-equivalent: reported in cycles/day
               and as a period difference in minutes), permutation p-value (two-sided, N_PERM shuffles of r)
    fast-cluster fraction in the inner vs outer tercile, for slices whose pooled noise-floor mixture has two or more
               components (cells assigned to the component of highest posterior)
Pooled over the lobes of all slices with lobe-wise centring of ω (ω_i − median ω_L) and radius normalised by the
lobe's RMS radius. Optional figure: ω vs normalised radius per slice.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_radial_frequency.py [--slices 1,2,3,4,5] [--figure]
Output: stdout table; data_paths.MPMF/circadian_data/lobe_phase/scn_radial_frequency.{csv,png,pdf}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from scipy.stats import spearmanr
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_pycobi_bifurcation as B

OUT = dp.ensure(os.path.join(S.OUT, "lobe_phase"))
N_PERM = 20000
CPD = 24 / (2 * np.pi)
RNG = np.random.default_rng(7)


def perm_p(x, r, n=N_PERM):
    """Two-sided permutation p-value of the outer − inner tercile mean difference of x (r shuffled)."""
    q1, q2 = np.quantile(r, (1 / 3, 2 / 3))
    stat = lambda rr: x[rr >= q2].mean() - x[rr <= q1].mean()
    d0 = stat(r)
    null = np.array([stat(RNG.permutation(r)) for _ in range(n)])
    return d0, (np.sum(np.abs(null) >= abs(d0)) + 1) / (n + 1)


def cluster_assign(i, om):
    """Index of the highest-posterior component of the pooled noise-floor mixture, components ordered by centre."""
    mod = B.pooled_mixture(i, "pfloor")
    w = np.asarray(mod["w"], float); Om = np.asarray(mod["Om"], float); De = np.asarray(mod["De"], float)
    o = np.argsort(Om); w, Om, De = w[o], Om[o], De[o]
    lik = w[None] * De[None] / np.pi / ((om[:, None] - Om[None]) ** 2 + De[None] ** 2)
    return np.argmax(lik, 1), Om


def main(slices, figure):
    rows, pooled = [], []
    print(f"{'slice':6s}{'lobe':>5s}{'n':>5s}{'ρ_signed':>10s}{'p':>8s}{'ρ_spread':>10s}{'p':>8s}"
          f"{'Δ_io (c/d)':>12s}{'Δ_io (min)':>11s}{'p_perm':>8s}{'fast frac in/out':>18s}")
    for i in slices:
        g = TG.load(i); _, locs = S.load(i)
        xy = locs[g["cells"]].astype(float); om = g["om"]; lobe = g["lobe"]
        lab, Om = cluster_assign(i, om)
        for L in (0, 1):
            m = lobe == L
            c = xy[m].mean(0); r = np.linalg.norm(xy[m] - c, axis=1); rn = r / np.sqrt(np.mean(r ** 2))
            w = om[m]; dev = np.abs(w - np.median(w))
            rs, ps = spearmanr(w, r); rd, pd = spearmanr(dev, r)
            d, pp = perm_p(w, r)
            T = 24 * 60 / (np.mean(w) * CPD)                     # mean period (min); ΔT ≈ −T Δf / f
            dmin = -T * (d * CPD) / (np.mean(w) * CPD)
            q1, q2 = np.quantile(r, (1 / 3, 2 / 3))
            if Om.size >= 2:
                fast = lab[m] == Om.size - 1
                ff = f"{fast[r <= q1].mean():.2f} / {fast[r >= q2].mean():.2f}"
            else:
                ff = "(M = 1)"
            print(f"SCN{i:<3d}{'AB'[L]:>5s}{m.sum():5d}{rs:10.2f}{ps:8.3f}{rd:10.2f}{pd:8.3f}"
                  f"{d * CPD:12.4f}{dmin:11.1f}{pp:8.3f}{ff:>18s}")
            rows.append(dict(slice=i, lobe="AB"[L], n=int(m.sum()), rho_signed=rs, p_signed=ps, rho_spread=rd,
                             p_spread=pd, d_io_cpd=d * CPD, d_io_min=dmin, p_perm=pp))
            pooled.append((w - np.median(w), rn, i, L))
    x = np.concatenate([p[0] for p in pooled]); rr = np.concatenate([p[1] for p in pooled])
    rs, ps = spearmanr(x, rr); rd, pd = spearmanr(np.abs(x), rr); d, pp = perm_p(x, rr)
    print(f"\npooled ({x.size} cells, lobe-centred ω, normalised radius): ρ_signed = {rs:.3f} (p = {ps:.3g}), "
          f"ρ_spread = {rd:.3f} (p = {pd:.3g}), outer − inner = {d * CPD:.4f} c/d (p_perm = {pp:.3g})")
    import csv
    with open(os.path.join(OUT, "scn_radial_frequency.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    if figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(slices), figsize=(2.6 * len(slices), 2.8), sharey=True,
                               gridspec_kw=dict(wspace=0.08, left=0.07, right=0.99, top=0.88, bottom=0.18))
        for a, i in zip(np.atleast_1d(ax), slices):
            for (w, rn, ii, L) in pooled:
                if ii == i:
                    a.plot(rn, w * CPD, "o", ms=2.5, alpha=0.6, color=("#2f7fd8", "#ef6a35")[L], label="lobe " + "AB"[L])
            a.axhline(0, color="0.5", lw=0.6); a.set_title(f"SCN{i}"); a.set_xlabel("radius / RMS radius")
        np.atleast_1d(ax)[0].set_ylabel("ω − lobe median (c/d)"); np.atleast_1d(ax)[0].legend(frameon=False, fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"scn_radial_frequency.{ext}"), dpi=200)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    main(arg("--slices", list(S.SLICES), lambda s_: [int(v) for v in s_.split(",")]), "--figure" in sys.argv)
