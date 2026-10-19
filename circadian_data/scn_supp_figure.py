r"""
Supplementary figure: raw results of the SCN LMMF fits at three levels of detail of the frequency distribution
=============================================================================================================

Same dynamics in all three models (setup C: homeostatic TTX-sensitive coupling, TTX noise γ u(t), lobe-specific
transient noise after TTX application, TTX-insensitive lobe-local coupling; scn_model_fit "hr-noise+transient+local"),
frequency distribution per slice and lobe captured by
    no floor           Lorentzian mixture without the noise floor (λ = 1e-5)         solid
    noise floor        Lorentzian mixture with the noise floor (λ = 1e-5)            dashed
    single Lorentzian  one Lorentzian (M_max = 1)                                    dotted
Two-column PRL figure, columns = SCN1–4, rows:
  (a) lobe A: intrinsic-frequency histogram (grey) and the three model distributions (normalised per lobe);
      N = cells, M = components (no floor / noise floor / single)
  (b) lobe B: the same
  (c) slice coherence R(t): data (black) and c(t)|Z(t)| of the three models; TTX shaded, masked ±24 h hatched
  (d) lobe difference R_A(t) − R_B(t): data and models (c_A|Z_A| − c_B|Z_B|)
  (e) fitted TTX-sensitive coupling K_s(t) (rad/h, log scale) of the three models; thin horizontal lines: the
      TTX-insensitive lobe-local coupling K_loc
Missing fits are skipped (preview); fits of the older "hf-" parametrisation are used if the "hr-" fit is absent.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_supp_figure.py
Output: data_paths.MPMF/circadian_data/writeup/scn_lmmf_fits_supp.{pdf,png}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_model_fit as F

OUT = dp.ensure(os.path.join(S.OUT, "writeup"))
FIT = os.path.join(S.OUT, "model_fit", "shapes")
SLICES = (1, 2, 3, 4)
NAMES = ("hr-noise+transient+local", "hf-noise+transient+local")      # preferred, fallback
VARIANTS = [("nofloor", "no floor", "#2166ac", "-"), ("floor", "noise floor", "#c0392b", "--"),
            ("single", "single Lorentzian", "0.35", ":")]
CPD = 24 / (2 * np.pi)


def load(i, kind):
    for name in NAMES:
        f = os.path.join(FIT, f"scn{i}_{kind}_{name}.npy")
        if os.path.exists(f):
            return np.load(f, allow_pickle=True).item()
    return None


def lobe_pdf(mod, k, g):
    sel = mod["lobe"] == k; w = mod["w"][sel] / mod["w"][sel].sum()
    return (w[:, None] * mod["De"][sel, None] / np.pi
            / ((g[None] - mod["Om"][sel, None]) ** 2 + mod["De"][sel, None] ** 2)).sum(0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from prl_style import set_prl_style, panel_label, COL_DOUBLE
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center"})
    # all five rows have the same height ROW_H; blocks (a–b | c–e) separated by a larger gap for the x labels
    ROW_H, HS_IN, HS_OUT, TOP, BOT = 1.05, 0.12, 0.17, 0.965, 0.06
    b1, b2 = 2 + HS_IN, 3 + 2 * HS_IN                       # block heights in units of ROW_H
    fig = plt.figure(figsize=(COL_DOUBLE, ROW_H * (b1 + b2 + HS_OUT * (b1 + b2) / 2) / (TOP - BOT)))
    outer = fig.add_gridspec(2, 1, height_ratios=[b1, b2], hspace=HS_OUT, left=0.075, right=0.985, top=TOP, bottom=BOT)
    g1 = outer[0].subgridspec(2, len(SLICES), hspace=HS_IN, wspace=0.12)       # frequency block (a, b)
    g2 = outer[1].subgridspec(3, len(SLICES), hspace=HS_IN, wspace=0.12)       # time block (c, d, e)
    axs = np.empty((5, len(SLICES)), object)
    for j in range(len(SLICES)):                            # rows a–b and c–e share their frequency / time axes
        axs[1, j] = fig.add_subplot(g1[1, j]); axs[0, j] = fig.add_subplot(g1[0, j], sharex=axs[1, j])
        axs[4, j] = fig.add_subplot(g2[2, j])
        axs[2, j] = fig.add_subplot(g2[0, j], sharex=axs[4, j]); axs[3, j] = fig.add_subplot(g2[1, j], sharex=axs[4, j])
    missing = []
    for j, i in enumerate(SLICES):
        d = TG.load(i); T = int(d["T"]); v0, v1 = S.valid_window(T); td = np.arange(T) / 24
        vw = lambda x: np.where((np.arange(T) >= v0) & (np.arange(T) < v1), x, np.nan)
        fits = {kind: load(i, kind) for kind, *_ in VARIANTS}
        missing += [f"SCN{i} {k}" for k, r in fits.items() if r is None]
        # (a), (b) lobe frequency distributions
        f_all = d["om"] * CPD
        g = np.linspace(f_all.min() - 0.03, f_all.max() + 0.03, 800)
        for row, k in ((0, 0), (1, 1)):
            a = axs[row, j]; fl = f_all[d["lobe"] == k]
            a.hist(fl, bins=np.linspace(g[0], g[-1], 30), density=True, color="0.82", lw=0)
            Ms = []
            for kind, lab, col, ls in VARIANTS:
                mod = fits[kind]["model"] if fits[kind] is not None else F.mixture(i, kind, d)
                a.plot(g, lobe_pdf(mod, k, g / CPD) / CPD, color=col, ls=ls, lw=1.0, label=lab)
                Ms.append(str(int((mod["lobe"] == k).sum())))
            a.set_yticks([])
            if row == 0:
                a.set_title(f"SCN{i}"); a.tick_params(labelbottom=False)
            if row == 1:
                a.set_xlabel("intrinsic frequency (cycles/day)")
            a.text(0.03, 0.95, f"lobe {'AB'[k]}, N = {fl.size}\nM = {' / '.join(Ms)}", transform=a.transAxes,
                   va="top", fontsize=5.2, linespacing=1.3)
            if j == 0:
                a.set_ylabel("density")
        # (c) slice coherence, (d) lobe difference, (e) coupling
        for row in (2, 3, 4):
            a = axs[row, j]
            a.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", lw=0, zorder=0)
            if row in (2, 3):
                for tc in (d["ttx"], d["wash"]):
                    a.axvspan((tc - 24) / 24, (tc + 24) / 24, color="none", hatch="////", ec="0.82", lw=0, zorder=0)
        axs[2, j].plot(td, vw(d["R"]), color="k", lw=1.1, zorder=3, label="data")
        axs[3, j].plot(td, vw(d["RA"] - d["RB"]), color="k", lw=1.1, zorder=3, label="data")
        axs[3, j].axhline(0, color="0.6", lw=0.5, zorder=1)
        tt = np.arange(0, T, 0.25)
        for kind, lab, col, ls in VARIANTS:
            r = fits[kind]
            if r is None:
                continue
            ti = r["tm"].astype(int)
            axs[2, j].plot(r["tm"] / 24, d["c"][ti] * np.abs(r["Zg"]), color=col, ls=ls, lw=0.9, zorder=2, label=lab)
            axs[3, j].plot(r["tm"] / 24, d["cA"][ti] * np.abs(r["ZA"]) - d["cB"][ti] * np.abs(r["ZB"]), color=col,
                           ls=ls, lw=0.9, zorder=2, label=lab)
            p = r["params"]
            axs[4, j].plot(tt / 24, np.maximum(F.Kfun(p, d)(tt), 1e-6), color=col, ls=ls, lw=0.9, label=lab)
            if p.get("K_i", 0) > 1e-4:
                axs[4, j].axhline(p["K_i"], color=col, lw=0.6, alpha=0.6)
        axs[2, j].set_ylim(0, 1.02); axs[3, j].set_ylim(-0.5, 0.5)
        axs[4, j].set_yscale("log"); axs[4, j].set_ylim(1e-4, 3); axs[4, j].set_xlabel("time (days)")
        for row in (2, 3, 4):
            axs[row, j].set_xlim(0, T / 24)
            if j > 0:
                axs[row, j].tick_params(labelleft=False)
        for row in (2, 3):
            axs[row, j].tick_params(labelbottom=False)
    axs[2, 0].set_ylabel(r"coherence $R(t)$"); axs[3, 0].set_ylabel(r"$R_A - R_B$")
    axs[4, 0].set_ylabel(r"$K_s(t)$ (rad/h)")
    for row, letter in enumerate("abcde"):
        panel_label(axs[row, 0], letter, dx=-26, dy=2)
    axs[2, 0].legend(handles=[Line2D([], [], color="k", lw=1.1, label="data")] +
                     [Line2D([], [], color=c, ls=ls, lw=0.9, label=l) for _, l, c, ls in VARIANTS],
                     loc="lower left", fontsize=4.8, handlelength=1.8)
    out = os.path.join(OUT, "scn_lmmf_fits_supp")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png" + (f"  (missing fits: {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
