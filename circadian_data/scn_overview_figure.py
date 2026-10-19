r"""
SCN slices: populations, intrinsic frequency distributions and phase coherence around TTX
=========================================================================================

One column per slice (each slice = one population: both lobes, i.e. left and right SCN, are phase-locked
before TTX and after washout, see the printed table; slices are NOT pooled — different animals, mean
periods and TTX timing). Rows (CURRENT pipeline, scn_targets: epoch-wise peak phases, validated in scn_pipeline_validation.py):
  (1) cell positions coloured by lobe (left/right SCN, scn_common.lobes); cells outside the fit target (not
      rhythmic under TTX, or TTX period outside 18–32 h) in grey
  (2) frequency distributions in cycles/day = least-squares slope of the epoch-peak phase per epoch (the fit
      target's frequencies under TTX): TTX epoch (≈ intrinsic, coupling blocked), stacked by lobe; pre-TTX and
      post-washout distributions as outlines. Epochs as in scn_common.epochs (first TRANSIENT_H h after a change
      and the recording edges excluded)
  (3) phase coherence R(t) of the slice and of each lobe (scn_targets), measurement ceiling c(t) dotted, TTX
      shaded; the ±MASK_H windows around TTX application and washout (excluded from the fits) hatched; recording
      edges (first 24 h, last 72 h) omitted

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_overview_figure.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scn_common as S
import scn_targets as TG

C_LOBE = ["#2a78d6", "#eb6834"]          # lobe A / lobe B
C_ALL = "0.15"
F_RANGE = (24 / 32, 24 / 18)             # cycles/day shown (periods 18-32 h)


def slopes(PH, a, b):
    """Least-squares phase slope (rad/h) per cell over [a, b) (finite samples only; ≥ 24 h)."""
    t = np.arange(PH.shape[0]); out = np.full(PH.shape[1], np.nan)
    for c in range(PH.shape[1]):
        ok = np.isfinite(PH[a:b, c])
        if ok.sum() > 24:
            out[c] = np.polyfit(t[a:b][ok], PH[a:b, c][ok], 1)[0]
    return out


def main():
    fig, axs = plt.subplots(3, len(S.SLICES), figsize=(15, 8.2), layout="constrained",
                            gridspec_kw=dict(height_ratios=[1.0, 0.8, 0.8]))
    # common position span: every row-1 panel shows the same width × height (equal aspect), centred on its slice
    ext = [np.ptp(np.asarray(S.load(i)[1], float), axis=0) for i in S.SLICES]
    span = 1.08 * np.max(ext, axis=0)                     # (width, height) in pixels
    print(f"{'slice':5s} {'N':>4s} {'lobes':>9s} | {'TTX freq (c/d) mean ± sd':>26s} | R pre / TTX end / wash end | lobe PLV pre/TTX/wash")
    for j, i in enumerate(S.SLICES):
        g = TG.load(i); X, locs = S.load(i); locs = np.asarray(locs, float)
        lab = g["lobe"]; cells = g["cells"]; T = int(g["T"]); x, y = locs.T
        ep = {n: tuple(int(v) for v in g["epochs"][k]) for k, n in enumerate(("pre", "ttx", "wash"))}
        # (1) positions
        a = axs[0, j]
        out_ = np.setdiff1d(np.arange(locs.shape[0]), cells)
        a.scatter(x[out_], y[out_], s=6, color="0.8")
        for k in (0, 1):
            a.scatter(x[cells[lab == k]], y[cells[lab == k]], s=8, color=C_LOBE[k])
        cx, cy = 0.5 * (locs.min(0) + locs.max(0))
        a.set_xlim(cx - span[0] / 2, cx + span[0] / 2); a.set_ylim(cy + span[1] / 2, cy - span[1] / 2)  # y down
        a.set_aspect("equal", adjustable="box"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"SCN slice {i} (N = {cells.size} of {locs.shape[0]})")
        # (2) frequency distributions (epoch-peak phase slopes; TTX = the fit target's ω)
        b = axs[1, j]; bins = np.linspace(*F_RANGE, 41)
        f = {"ttx": g["om"] * 24 / (2 * np.pi)}
        for e in ("pre", "wash"):
            f[e] = slopes(g["PH"], *ep[e]) * 24 / (2 * np.pi)
        b.hist([f["ttx"][lab == k] for k in (0, 1)], bins=bins, stacked=True, color=C_LOBE, alpha=0.75)
        for e, ls in (("pre", "-"), ("wash", "--")):
            b.hist(f[e][np.isfinite(f[e])], bins=bins, histtype="step", color=C_ALL, ls=ls, lw=0.9)
        b.set_xlim(*F_RANGE); b.set_xlabel("frequency (cycles/day)")
        b.set_ylim(0, 0.45 * cells.size)                          # TTX histogram is the focus
        # (3) R(t)
        c = axs[2, j]; t = np.arange(T) / 24.0; ttx, wash, mh = int(g["ttx"]), int(g["wash"]), int(g["mask_h"])
        c.axvspan(ttx / 24, wash / 24, color="0.9", zorder=0)
        for t0 in (ttx, wash):
            c.axvspan((t0 - mh) / 24, (t0 + mh) / 24, facecolor="none", edgecolor="0.6", hatch="////", lw=0, zorder=0)
        v0, v1 = S.valid_window(T); vm = (np.arange(T) >= v0) & (np.arange(T) < v1)
        mask = lambda r: np.where(vm, r, np.nan)                  # recording edges excluded
        R = mask(g["R"]); c.plot(t, R, color=C_ALL, lw=1.1, zorder=3)
        for k, key in ((0, "RA"), (1, "RB")):
            c.plot(t, mask(g[key]), color=C_LOBE[k], lw=0.7, zorder=2)
        c.plot(t, mask(g["c"]), color="0.5", lw=0.7, ls=":", zorder=1)
        c.set_ylim(0, 1.02); c.set_xlim(0, t[-1]); c.set_xlabel("time (days)")
        if j == 0:
            b.set_ylabel("cells"); c.set_ylabel(r"phase coherence $R(t)$")
        # printed summary
        plv = []
        for e_ in ("pre", "ttx", "wash"):
            s0, s1 = ep[e_]; dpsi = g["dpsi"][s0:s1]
            plv.append(np.abs(np.nanmean(np.exp(1j * dpsi))))
        ft = f["ttx"]
        print(f"SCN{i:<2d} {cells.size:4d} {str(np.bincount(lab).tolist()):>9s} | {ft.mean():.3f} ± {ft.std():.3f} "
              f"({cells.size / locs.shape[0]:.0%} of cells) | {np.nanmean(R[slice(*ep['pre'])]):.2f} / "
              f"{np.nanmean(R[ep['ttx'][1] - 48:ep['ttx'][1]]):.2f} / {np.nanmean(R[v1 - 72:v1]):.2f} | "
              + "/".join(f"{v:.2f}" for v in plv))
    h = [Patch(fc=C_LOBE[0], alpha=0.75, label="lobe A (TTX, ≈ intrinsic)"),
         Patch(fc=C_LOBE[1], alpha=0.75, label="lobe B (TTX, ≈ intrinsic)"),
         Line2D([], [], color=C_ALL, lw=0.9, label="pre-TTX (coupled)"),
         Line2D([], [], color=C_ALL, lw=0.9, ls="--", label="after washout (coupled)"),
         Line2D([], [], color=C_ALL, lw=1.1, label=r"$R(t)$ whole slice"),
         Line2D([], [], color="0.5", lw=0.7, ls=":", label="measurement ceiling $c(t)$"),
         Patch(fc="0.9", label="TTX"), Patch(fc="none", ec="0.6", hatch="////", label="±24 h excluded from fits")]
    fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False)
    out = os.path.join(S.OUT, "scn_overview")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    main()
