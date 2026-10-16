r"""
SCN slices: populations, intrinsic frequency distributions and phase coherence around TTX
=========================================================================================

One column per slice (each slice = one population: both lobes, i.e. left and right SCN, are phase-locked
before TTX and after washout, see the printed table; slices are NOT pooled — different animals, mean
periods and TTX timing). Rows:
  (1) cell positions coloured by lobe (left/right SCN, scn_common.lobes)
  (2) frequency distributions in cycles/day (24 h / period from the Hilbert-phase slope per epoch):
      TTX epoch (≈ intrinsic, coupling blocked; rhythmic cells only), stacked by lobe; pre-TTX and
      post-washout distributions as outlines
  (3) phase coherence R(t) of the whole slice and of each lobe (Hilbert phase), TTX epoch shaded; the
      first TRANSIENT_H hours after a change are hatched (excluded from the period estimates); the first
      24 h and last 72 h of each recording are omitted (filter edge effects, also excluded from the periods)

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

C_LOBE = ["#2a78d6", "#eb6834"]          # lobe A / lobe B
C_ALL = "0.15"
F_RANGE = (24 / 32, 24 / 18)             # cycles/day shown (periods 18-32 h)


def main():
    fig, axs = plt.subplots(3, len(S.SLICES), figsize=(15, 8.2), layout="constrained",
                            gridspec_kw=dict(height_ratios=[1.0, 0.8, 0.8]))
    print(f"{'slice':5s} {'N':>4s} {'lobes':>9s} | {'TTX freq (c/d) mean ± sd':>26s} | R pre / TTX end / wash end | lobe PLV pre/TTX/wash")
    for j, i in enumerate(S.SLICES):
        d = S.process(i); lab = S.lobes(d["locs"]); x, y = d["locs"].T
        # (1) positions
        a = axs[0, j]
        for k in (0, 1):
            a.scatter(x[lab == k], y[lab == k], s=8, color=C_LOBE[k])
        a.set_aspect("equal"); a.invert_yaxis(); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"SCN slice {i} (N = {d['N']})")
        # (2) frequency distributions
        b = axs[1, j]; bins = np.linspace(*F_RANGE, 41)
        f = {e: 24.0 / d["period"][e] for e in d["epochs"]}
        r = d["rhythmic"]["ttx"]
        b.hist([f["ttx"][r & (lab == k)] for k in (0, 1)], bins=bins, stacked=True, color=C_LOBE, alpha=0.75)
        for e, ls in (("pre", "-"), ("wash", "--")):
            b.hist(f[e], bins=bins, histtype="step", color=C_ALL, ls=ls, lw=0.9)
        b.set_xlim(*F_RANGE); b.set_xlabel("frequency (cycles/day)")
        b.set_ylim(0, 0.45 * d["N"])                              # TTX histogram is the focus
        # (3) R(t)
        c = axs[2, j]; t = np.arange(d["T"]) / 24.0
        c.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.9", zorder=0)
        for t0 in (d["ttx"], d["wash"]):
            c.axvspan(t0 / 24, (t0 + S.TRANSIENT_H) / 24, facecolor="none", edgecolor="0.6", hatch="////", lw=0, zorder=0)
        v0, v1 = S.valid_window(d["T"]); vm = (np.arange(d["T"]) >= v0) & (np.arange(d["T"]) < v1)
        mask = lambda r: np.where(vm, r, np.nan)                  # recording edges excluded (filter edges)
        R = mask(S.order_parameter(d["PH"])[0]); c.plot(t, R, color=C_ALL, lw=1.1, zorder=3)
        for k in (0, 1):
            c.plot(t, mask(S.order_parameter(d["PH"], lab == k)[0]), color=C_LOBE[k], lw=0.7, zorder=2)
        c.set_ylim(0, 1.02); c.set_xlim(0, t[-1]); c.set_xlabel("time (days)")
        if j == 0:
            b.set_ylabel("cells"); c.set_ylabel(r"phase coherence $R(t)$")
        # printed summary
        e = d["epochs"]; Z = np.exp(1j * d["PH"])
        plv = []
        for ep in ("pre", "ttx", "wash"):
            s0, s1 = e[ep]; za = Z[s0:s1, lab == 0].mean(1); zb = Z[s0:s1, lab == 1].mean(1)
            plv.append(np.abs(np.mean(np.exp(1j * (np.angle(za) - np.angle(zb))))))
        ft = f["ttx"][r]
        print(f"SCN{i:<2d} {d['N']:4d} {str(np.bincount(lab).tolist()):>9s} | {ft.mean():.3f} ± {ft.std():.3f} "
              f"({r.mean():.0%} rhythmic) | {np.nanmean(R[slice(*e['pre'])]):.2f} / "
              f"{np.nanmean(R[e['ttx'][1] - 48:e['ttx'][1]]):.2f} / {np.nanmean(R[v1 - 72:v1]):.2f} | "
              + "/".join(f"{v:.2f}" for v in plv))
    h = [Patch(fc=C_LOBE[0], alpha=0.75, label="lobe A (TTX, ≈ intrinsic)"),
         Patch(fc=C_LOBE[1], alpha=0.75, label="lobe B (TTX, ≈ intrinsic)"),
         Line2D([], [], color=C_ALL, lw=0.9, label="pre-TTX (coupled)"),
         Line2D([], [], color=C_ALL, lw=0.9, ls="--", label="after washout (coupled)"),
         Line2D([], [], color=C_ALL, lw=1.1, label=r"$R(t)$ whole slice"),
         Patch(fc="0.9", label="TTX"), Patch(fc="none", ec="0.6", hatch="////", label="first 24 h after a change")]
    fig.legend(handles=h, loc="outside lower center", ncol=7, frameon=False)
    out = os.path.join(S.OUT, "scn_overview")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    main()
