r"""
Lobe-specific transient after TTX application, and measurement confounds of the spatial dephasing pattern
=========================================================================================================

(1) TRANSIENT: hourly lobe coherence R_L(t) and R_L / c_L (measurement ceiling) from TTX − PRE_H to TTX + POST_H,
    with each lobe's single-cell signal-to-noise ratio SNR = Hilbert envelope of the band-passed trace /
    RMS of the out-of-band residual (48 h moving windows; median over the lobe's cells). A real lobe-specific
    transient shows in R / c; a measurement effect shows as an SNR difference between the lobes at the same time.
    Also the coherence drop over the first day, R_L(TTX − 12 h) → R_L(TTX + 24 h), vs the free-running drop
    expected from the lobe's frequencies (mean |e^{iω_i 36 h}| around the lobe mean frequency).
(2) CONFOUNDS of scn_lobe_diagnostics D3 (edge cells deviate more from the lobe phase under TTX): Spearman
    correlations of the cells' radius (distance from the lobe centroid) with their SNR under TTX and with their
    intrinsic frequency deviation |ω_i − median ω_L|, and the PARTIAL correlation of the phase deviation with the
    radius after regressing out SNR and |Δω| (rank-based).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_lobe_transient.py [--slices 1,2,3,4]
Output: data_paths.MPMF/circadian_data/lobe_phase/scn_lobe_transient.{png,pdf}; tables on stdout
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr, rankdata
import data_paths as dp
import scn_common as S
import scn_lobe_diagnostics as LD
import scn_processing_walkthrough as W

OUT = LD.OUT
PRE_H, POST_H = 48, 72
CL = LD.CL


def snr(X):
    D, B, _, A = W.pipeline(X)
    noise = np.sqrt(uniform_filter1d((D - B) ** 2, size=48, axis=0, mode="nearest"))
    return A / np.maximum(noise, 1e-300)


def partial_rank(y, x, covs):
    """Spearman partial correlation of y and x given the covariates (ranks, linear residuals)."""
    ok = np.isfinite(y) & np.isfinite(x) & np.all([np.isfinite(c) for c in covs], axis=0)
    r = lambda v: rankdata(v[ok])
    C = np.column_stack([np.ones(ok.sum())] + [r(c) for c in covs])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return spearmanr(res(r(y)), res(r(x)))


def analyse(i):
    g = LD.analyse(i); d = g["d"]; ttx = int(d["ttx"]); lobe = d["lobe"]
    X, _ = S.load(i); X = X[:, d["cells"]]
    Q = snr(X)
    a, b = ttx - PRE_H, ttx + POST_H; t = np.arange(a, b)
    out = dict(t=t, ttx=ttx, lobes={})
    for k in (0, 1):
        sel = lobe == k; L = g["lobes"][k]
        R = d["RA" if k == 0 else "RB"][a:b]; c = d["cA" if k == 0 else "cB"][a:b]
        q = np.nanmedian(Q[a:b][:, sel], axis=1)
        om = d["om"][sel]; dom = om - np.median(om)
        free = np.abs(np.mean(np.exp(1j * dom * 36.0)))
        R0 = d["RA" if k == 0 else "RB"][ttx - 12]; R1 = d["RA" if k == 0 else "RB"][ttx + 24]
        q_ttx = np.nanmedian(Q[g["t0"]:g["t1"]][:, sel], axis=0)          # per-cell SNR under TTX
        rad = L["rad"]; mdev = L["mdev"]; adom = np.abs(dom)
        okr = np.isfinite(q_ttx)
        out["lobes"][k] = dict(R=R, c=c, Rc=R / c, snr=q, R0=R0, R1=R1, free=free,
                               rho_rad_snr=spearmanr(rad[okr], q_ttx[okr]), rho_rad_dom=spearmanr(rad, adom),
                               rho_dev_snr=spearmanr(mdev[okr], q_ttx[okr]),
                               partial=partial_rank(mdev, rad, [q_ttx, adom]), raw=L["rho_r"])
    return out


def summary(i, g):
    print(f"\nSCN{i}")
    print(f"  {'lobe':4s} | {'R(TTX−12h)':>10s} {'R(TTX+24h)':>10s} {'ratio':>6s} {'free-run 36 h':>13s} | "
          f"{'SNR pre / TTX+12 / TTX+48':>27s} | {'ρ(rad,SNR)':>11s} {'ρ(rad,|Δω|)':>12s} {'ρ(dev,SNR)':>11s} | "
          f"{'ρ(dev,rad) raw':>15s} {'partial':>16s}")
    t = g["t"]; at = lambda x, h: x[np.argmin(np.abs(t - (g["ttx"] + h)))]
    for k, L in g["lobes"].items():
        print(f"  {'AB'[k]:4s} | {L['R0']:10.2f} {L['R1']:10.2f} {L['R1'] / L['R0']:6.2f} {L['free']:13.2f} | "
              f"{at(L['snr'], -24):8.1f} / {at(L['snr'], 12):5.1f} / {at(L['snr'], 48):5.1f} | "
              f"{L['rho_rad_snr'][0]:+11.2f} {L['rho_rad_dom'][0]:+12.2f} {L['rho_dev_snr'][0]:+11.2f} | "
              f"{L['raw'][0]:+7.2f} (p={L['raw'][1]:.1g}) {L['partial'][0]:+7.2f} (p={L['partial'][1]:.1g})")


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sl = list(res)
    fig, axs = plt.subplots(2, len(sl), figsize=(3.1 * len(sl), 5.0), layout="constrained", squeeze=False)
    for j, i in enumerate(sl):
        g = res[i]; th = (g["t"] - g["ttx"]) / 24
        a = axs[0, j]; a.axvspan(0, th[-1], color="0.92", zorder=0)
        for k, L in g["lobes"].items():
            a.plot(th, L["R"], color=CL[k], lw=1.3, label=f"lobe {'AB'[k]}: R")
            a.plot(th, L["Rc"], color=CL[k], lw=0.9, ls="--", label=f"lobe {'AB'[k]}: R / ceiling")
        a.axvspan(-1, 1, color="none", hatch="////", ec="0.75", lw=0, zorder=0)
        a.set_ylim(0, 1.05); a.set_title(f"SCN{i}: lobe coherence around TTX", fontsize=8)
        if j == 0:
            a.set_ylabel("R"); a.legend(fontsize=5.5, frameon=False, loc="lower left")
        a = axs[1, j]; a.axvspan(0, th[-1], color="0.92", zorder=0)
        for k, L in g["lobes"].items():
            a.plot(th, L["snr"], color=CL[k], lw=1.2, label=f"lobe {'AB'[k]}")
        a.set_yscale("log"); a.set_xlabel("days from TTX application"); a.set_title("median single-cell SNR", fontsize=8)
        if j == 0:
            a.set_ylabel("SNR"); a.legend(fontsize=6, frameon=False)
    out = os.path.join(OUT, "scn_lobe_transient")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"\n[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", [1, 2, 3, 4], lambda s_: [int(v) for v in s_.split(",")])
    res = {}
    for i in slices:
        res[i] = analyse(i); summary(i, res[i])
    figure(res)
