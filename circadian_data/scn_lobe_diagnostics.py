r"""
Why do the lobes dephase differently under TTX? Three data-only diagnostics per slice and lobe
==============================================================================================

Target data (scn_targets: epoch-peak phases, cells rhythmic under TTX, lobes = left / right SCN). Under TTX
the windows start after the transient: t0 = t_ttx + TRANSIENT_H.

D1  free-running prediction. From the MEASURED phases at t0, every cell runs at a constant frequency:
        R_pred,L(t) = c_L(t) |⟨exp(i[θ_i(t0) + ω_i (t − t0)])⟩_L|     (c_L: the lobe's measurement ceiling)
    with ω_i = the fit-target frequency (phase slope over the whole TTX epoch) or an EARLY frequency (slope over
    [t0, t0 + EARLY_H]). Compared with the measured R_L over [t0, t0 + PRED_H]:
      measured < predicted  → dephasing beyond constant frequencies (phase noise / diffusion)
      measured > predicted  → phases held together (residual coupling)
    and whether the lobe ORDER (which lobe stays more coherent) is predicted by the frequencies alone.
    Frequency spread per lobe: Cauchy-like scale = half the interquartile range of ω (rad/h).
D2  amplitude collapse per lobe: per cell and epoch, relative amplitude = robust half-amplitude ((p95 − p5)/2)
    of the epoch-wise detrended trace / the epoch's mean baseline; log ratio TTX / pre-TTX. Also the
    per-cell phase jitter under TTX: SD of the residuals of θ_i around its linear fit on the prediction window (h).
D3  position: per cell the mean circular deviation from its lobe's mean phase over the prediction window,
    |θ_i − ψ_L|; Spearman correlation with the coordinate ORTHOGONAL to the lobe axis (dorsal–ventral up to the
    unknown orientation) and with the distance from the lobe centroid (core vs periphery).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_lobe_diagnostics.py [--slices 1,2,3,4]
Output: data_paths.MPMF/circadian_data/lobe_phase/scn_lobe_diagnostics.{png,pdf}; tables on stdout
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from scipy.stats import spearmanr
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_epoch_processing as EP

OUT = dp.ensure(os.path.join(S.OUT, "lobe_phase"))
EARLY_H = 48            # early-frequency window after t0 (h)
PRED_H = 96             # prediction window after t0 (h)
H = 24 / (2 * np.pi)
CL = ["#1b7837", "#762a83"]


def slopes(PH, a, b):
    """Least-squares phase slope (rad/h) per cell over [a, b) (finite samples only)."""
    t = np.arange(PH.shape[0]); out = np.full(PH.shape[1], np.nan)
    for c in range(PH.shape[1]):
        ok = np.isfinite(PH[a:b, c])
        if ok.sum() > 12:
            out[c] = np.polyfit(t[a:b][ok], PH[a:b, c][ok], 1)[0]
    return out


def free_run(PH, om, t0, n):
    ok = np.isfinite(PH[t0]) & np.isfinite(om)
    tt = np.arange(n)[:, None]
    return np.abs(np.exp(1j * (PH[t0, ok][None] + om[ok][None] * tt)).mean(1))


def rel_amplitude(X, a, b):
    seg = X[a:b]; base = EP.baseline(seg, "poly"); dtr = seg - base
    amp = 0.5 * np.subtract(*np.percentile(dtr, [95, 5], axis=0))
    return amp / np.maximum(np.abs(seg.mean(0)), 1e-300)


def analyse(i):
    d = TG.load(i)
    if "ZA" not in d:
        d = TG.load(i, rebuild=True)
    lobe = d["lobe"]; PH = d["PH"]; T = int(d["T"]); ttx, wash = int(d["ttx"]), int(d["wash"])
    t0 = ttx + S.TRANSIENT_H; t1 = min(t0 + PRED_H, wash); n = t1 - t0
    om_full = d["om"]; om_early = slopes(PH, t0, t0 + EARLY_H)
    X, locs = S.load(i); X = X[:, d["cells"]]; locs = np.asarray(locs, float)[d["cells"]]
    amp_pre = rel_amplitude(X, S.EDGE_START_H, ttx); amp_ttx = rel_amplitude(X, t0, wash)
    # phase jitter around the linear trend on the prediction window
    tt = np.arange(t0, t1); jit = np.full(PH.shape[1], np.nan)
    for c in range(PH.shape[1]):
        ok = np.isfinite(PH[t0:t1, c])
        if ok.sum() > 12:
            fit = np.polyval(np.polyfit(tt[ok], PH[t0:t1, c][ok], 1), tt[ok])
            jit[c] = np.std(PH[t0:t1, c][ok] - fit) * H
    # position: lobe axis = line through the two lobe centroids
    cen = np.array([locs[lobe == k].mean(0) for k in (0, 1)]); ax = cen[1] - cen[0]; ax /= np.linalg.norm(ax)
    orth = np.array([-ax[1], ax[0]])
    res = dict(d=d, t0=t0, t1=t1, lobes={})
    for k in (0, 1):
        sel = lobe == k
        PHk = PH[:, sel]
        Rm = d["RA" if k == 0 else "RB"][t0:t1]; c = d["cA" if k == 0 else "cB"][t0:t1]
        Rf = c * free_run(PHk, om_full[sel], t0, n); Re = c * free_run(PHk, om_early[sel], t0, n)
        psi = np.angle(d["ZA" if k == 0 else "ZB"])[t0:t1]
        dev = np.abs(np.angle(np.exp(1j * (PHk[t0:t1] - psi[:, None]))))
        mdev = np.nanmean(dev, 0) * H
        y = (locs[sel] - cen[k]) @ orth; rad = np.linalg.norm(locs[sel] - cen[k], axis=1)
        okd = np.isfinite(mdev)
        res["lobes"][k] = dict(
            n=int(sel.sum()), Rm=Rm, Rf=Rf, Re=Re,
            spread_full=0.5 * np.subtract(*np.nanpercentile(om_full[sel], [75, 25])),
            spread_early=0.5 * np.subtract(*np.nanpercentile(om_early[sel], [75, 25])),
            alr=np.log(amp_ttx[sel] / amp_pre[sel]), jit=jit[sel], mdev=mdev, y=y, rad=rad,
            rho_y=spearmanr(y[okd], mdev[okd]), rho_r=spearmanr(rad[okd], mdev[okd]), locs=locs[sel])
    return res


def summary(i, g):
    print(f"\nSCN{i}: windows t0 = TTX + {S.TRANSIENT_H} h, prediction {g['t1'] - g['t0']} h, early ω over {EARLY_H} h")
    print(f"  {'lobe':4s} {'n':>4s} | {'R meas':>6s} {'R pred(full ω)':>14s} {'R pred(early ω)':>15s} | "
          f"{'spread ω full/early (rad/h)':>27s} | {'amp log-ratio':>13s} {'jitter (h)':>10s} | "
          f"{'ρ(|dev|, orth)':>15s} {'ρ(|dev|, radius)':>16s}")
    for k, L in g["lobes"].items():
        print(f"  {'AB'[k]:4s} {L['n']:4d} | {L['Rm'].mean():6.2f} {L['Rf'].mean():14.2f} {L['Re'].mean():15.2f} | "
              f"{L['spread_full']:13.4f} / {L['spread_early']:.4f} | {np.nanmedian(L['alr']):+13.2f} "
              f"{np.nanmedian(L['jit']):10.2f} | {L['rho_y'][0]:+7.2f} (p={L['rho_y'][1]:.2g}) "
              f"{L['rho_r'][0]:+7.2f} (p={L['rho_r'][1]:.2g})")
    A, B = g["lobes"][0], g["lobes"][1]
    order = lambda a, b: "A" if a > b else "B"
    print(f"  more coherent lobe: measured {order(A['Rm'].mean(), B['Rm'].mean())}, predicted from full ω "
          f"{order(A['Rf'].mean(), B['Rf'].mean())}, from early ω {order(A['Re'].mean(), B['Re'].mean())}; "
          f"larger amplitude loss in lobe {order(-np.nanmedian(A['alr']), -np.nanmedian(B['alr']))}, "
          f"larger jitter in lobe {order(np.nanmedian(A['jit']), np.nanmedian(B['jit']))}")


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sl = list(res)
    fig, axs = plt.subplots(3, len(sl), figsize=(3.1 * len(sl), 7.6), layout="constrained", squeeze=False)
    for j, i in enumerate(sl):
        g = res[i]; t = (np.arange(g["t0"], g["t1"]) - g["t0"]) / 24
        a = axs[0, j]
        for k, L in g["lobes"].items():
            a.plot(t, L["Rm"], color=CL[k], lw=1.4, label=f"lobe {'AB'[k]}, measured")
            a.plot(t, L["Rf"], color=CL[k], lw=0.9, ls="--", label="free run, full-TTX ω")
            a.plot(t, L["Re"], color=CL[k], lw=0.9, ls=":", label=f"free run, early ω ({EARLY_H} h)")
        a.set_ylim(0, 1.02); a.set_xlabel(f"days after TTX + {S.TRANSIENT_H} h"); a.set_title(f"SCN{i}: D1", fontsize=8)
        if j == 0:
            a.set_ylabel("lobe R"); a.legend(fontsize=5, frameon=False, loc="lower left")
        a = axs[1, j]; bins = np.linspace(-3, 1, 33)
        for k, L in g["lobes"].items():
            a.hist(L["alr"][np.isfinite(L["alr"])], bins=bins, histtype="step", color=CL[k], lw=1.3,
                   label=f"lobe {'AB'[k]}: median {np.nanmedian(L['alr']):+.2f}")
        a.set_xlabel("log amplitude ratio TTX / pre"); a.set_title("D2: amplitude collapse", fontsize=8)
        a.legend(fontsize=5.5, frameon=False, loc="upper left")
        a = axs[2, j]
        allm = np.concatenate([L["mdev"] for L in g["lobes"].values()])
        vmax = np.nanpercentile(allm, 95)
        for k, L in g["lobes"].items():
            sc = a.scatter(L["locs"][:, 0], L["locs"][:, 1], c=L["mdev"], s=7, cmap="magma", vmin=0, vmax=vmax,
                           marker="o" if k == 0 else "s", edgecolors=CL[k], linewidths=0.3)
        a.set_aspect("equal"); a.invert_yaxis(); a.set_xticks([]); a.set_yticks([])
        a.set_title("D3: mean |θ_i − ψ_lobe| under TTX (h)", fontsize=8)
        fig.colorbar(sc, ax=a, shrink=0.7)
    out = os.path.join(OUT, "scn_lobe_diagnostics")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"\n[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", [1, 2, 3, 4], lambda s_: [int(v) for v in s_.split(",")])
    res = {}
    for i in slices:
        res[i] = analyse(i); summary(i, res[i])
    figure(res)
