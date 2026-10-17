r"""
Do the lobes stay internally coherent and only shift in phase? Inter-lobe phase difference of the SCN slices
=============================================================================================================

Per slice (scn_targets: epoch-peak phases, cells rhythmic under TTX, lobes = left / right SCN):
  Z_A, Z_B   lobe mean fields, R_A = |Z_A|, R_B = |Z_B|, slice R = |p_A Z_A + p_B Z_B| (p = lobe fractions)
  Δψ(t)      inter-lobe phase difference arg(Z_A Z̄_B) (h; > 0: lobe A ahead)
  decomposition   R² = p_A² R_A² + p_B² R_B² + 2 p_A p_B R_A R_B cos Δψ, so
                  R_inphase = p_A R_A + p_B R_B   is the coherence the slice WOULD have with the lobes in phase;
                  R / R_inphase < 1 measures the coherence lost to the inter-lobe phase shift only.
  "lobe-shift hypothesis": the lobes stay coherent (R_A, R_B high) while Δψ grows → R falls although R_inphase
                  stays high. "within-lobe dephasing": R_A, R_B fall with R, Δψ stays small, R ≈ R_inphase.
  NULL             the same quantities for N_NULL random two-group splits of the same cells with the lobe sizes
                   (no anatomical structure): the Δψ and R / R_inphase that finite groups show by chance.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_lobe_phase.py [--slices 1,2,3,4,5]
Output: data_paths.MPMF/circadian_data/lobe_phase/{scn_lobe_phase.png/.pdf}; summary table on stdout
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import data_paths as dp
import scn_common as S
import scn_targets as TG

OUT = dp.ensure(os.path.join(S.OUT, "lobe_phase"))
N_NULL = 200
H = 24 / (2 * np.pi)


def analyse(i, rng):
    d = TG.load(i); T = int(d["T"]); lobe = d["lobe"]; PH = d["PH"]
    if "ZA" not in d:
        d = TG.load(i, rebuild=True)
    pA = (lobe == 0).mean(); pB = 1 - pA
    RA, RB, R = np.abs(d["ZA"]), np.abs(d["ZB"]), d["R"]
    Rin = pA * RA + pB * RB
    v0, v1 = S.valid_window(T); t = np.arange(T); vm = (t >= v0) & (t < v1)
    nd, nr = [], []
    for _ in range(N_NULL):
        g = rng.permutation(lobe)
        za, zb = TG.mean_field(PH[:, g == 0]), TG.mean_field(PH[:, g == 1])
        nd.append(np.abs(np.angle(za * np.conj(zb))))
        nr.append(R / (pA * np.abs(za) + pB * np.abs(zb)))
    nd, nr = np.array(nd), np.array(nr)
    return dict(d=d, T=T, vm=vm, RA=RA, RB=RB, R=R, Rin=Rin, dpsi=d["dpsi"], ratio=R / Rin,
                null_dpsi=np.percentile(nd, [50, 95], axis=0), null_ratio=np.percentile(nr, [5, 50], axis=0))


def summary(i, g):
    d = g["d"]; e = d["epochs"]
    print(f"\nSCN{i}  (epoch means over the valid window; null = random two-group splits, 95th / 5th percentile)")
    print(f"  {'epoch':6s} {'R':>5s} {'R_A':>5s} {'R_B':>5s} {'R_inph':>6s} {'R/R_inph':>8s} {'null 5%':>7s} "
          f"{'|Δψ| (h)':>8s} {'null 95%':>8s} {'max |Δψ|':>8s}")
    for name, (a, b) in zip(("pre", "ttx", "wash"), e):
        sl = slice(int(a), int(b)); m = lambda x: np.nanmean(np.where(g["vm"][sl], x[sl], np.nan))
        ad = np.abs(g["dpsi"]) * H
        print(f"  {name:6s} {m(g['R']):5.2f} {m(g['RA']):5.2f} {m(g['RB']):5.2f} {m(g['Rin']):6.2f} "
              f"{m(g['ratio']):8.2f} {m(g['null_ratio'][0]):7.2f} {m(ad):8.2f} {m(g['null_dpsi'][1] * H):8.2f} "
              f"{np.nanmax(np.where(g['vm'][sl], ad[sl], np.nan)):8.2f}")
    # fraction of the coherence loss under TTX explained by the inter-lobe shift
    a, b = (int(x) for x in e[1]); sl = slice(a, b)
    loss_total = 1 - np.nanmean(g["R"][sl]); loss_shift = np.nanmean(g["Rin"][sl] - g["R"][sl])
    print(f"  under TTX: coherence loss 1 − R = {loss_total:.2f}, of which from the inter-lobe shift "
          f"(R_inph − R) {loss_shift:.2f} ({loss_shift / loss_total:.0%})")


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sl = list(res)
    fig, axs = plt.subplots(3, len(sl), figsize=(3.0 * len(sl), 6.6), layout="constrained", squeeze=False,
                            sharey="row")
    CL = ["#1b7837", "#762a83"]
    for j, i in enumerate(sl):
        g = res[i]; d = g["d"]; T = g["T"]; td = np.arange(T) / 24
        vw = lambda x: np.where(g["vm"], x, np.nan)
        for a in axs[:, j]:
            a.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
        a = axs[0, j]
        a.plot(td, vw(g["R"]), color="k", lw=1.3, label="slice R")
        a.plot(td, vw(g["Rin"]), color="0.5", lw=1.0, ls="--", label=r"$R_{in\,phase} = p_AR_A + p_BR_B$")
        a.plot(td, vw(g["RA"]), color=CL[0], lw=0.9, label="lobe A")
        a.plot(td, vw(g["RB"]), color=CL[1], lw=0.9, label="lobe B")
        a.set_ylim(0, 1.02); a.set_title(f"SCN{i}", fontsize=8)
        a = axs[1, j]
        a.fill_between(td, 0, vw(g["null_dpsi"][1] * H), color="0.85", lw=0, label="null 95% (random split)")
        a.plot(td, vw(np.abs(g["dpsi"]) * H), color="#b2182b", lw=1.1, label=r"$|\Delta\psi|$")
        a.set_ylim(0, 12)
        a = axs[2, j]
        a.fill_between(td, vw(g["null_ratio"][0]), 1, color="0.85", lw=0, label="null 5% (random split)")
        a.plot(td, vw(g["ratio"]), color="#2166ac", lw=1.1, label=r"$R / R_{in\,phase}$")
        a.set_ylim(0, 1.02); a.set_xlabel("time (days)")
    axs[0, 0].set_ylabel("coherence"); axs[1, 0].set_ylabel("inter-lobe phase diff. (h)")
    axs[2, 0].set_ylabel(r"$R / R_{in\,phase}$")
    for r in range(3):
        axs[r, 0].legend(fontsize=5.5, frameon=False, loc="lower left" if r != 1 else "upper left")
    out = os.path.join(OUT, "scn_lobe_phase")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"\n[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", list(S.SLICES), lambda s_: [int(v) for v in s_.split(",")])
    rng = np.random.default_rng(5)
    res = {}
    for i in slices:
        res[i] = analyse(i, rng); summary(i, res[i])
    figure(res)
