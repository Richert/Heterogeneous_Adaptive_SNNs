r"""
Adaptive-coupling Kuramoto: K-ramp / hysteresis figure
======================================================

Loads the ramp simulations written by ``weight_variance_ramp.py`` and renders a single-column PRL
figure (analytic branches are recomputed here from the saved parameters, so plotting can be tuned
without re-running the simulations):
  * row 1 — V_A/Ā² vs K: analytic bifurcation branches (one panel per μ);
  * row 2 — R vs K: analytic branches + microscopic forward/backward K-ramp measurements (per μ);
  * rows 3–4 — R(t) ramp dynamics per μ: micro vs. mean field (Eqs. 8–10) by colour, forward vs.
    backward ramp by line style, with the mean-field V_A/Ā²(t) on a secondary y-axis (separate colour).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_ramp_figure.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import weight_variance_bifurcation as B                            # analytic fixed-point branches

NPZ = "/home/rgast/data/kmo_adaptive/weight_variance_ramp.npz"
OUT = "/home/rgast/data/kmo_adaptive/weight_variance_ramp"

C_MICRO, C_MF, C_ASYNC, C_SYNC, C_RATIO = "0.2", "#c1121f", "0.55", "#1f77b4", "#2a9d8f"
LS = {0: "-", 1: "--"}                                            # 0=forward, 1=backward


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 5.2, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def _panel_label(ax, letter, dx=-24, dy=3):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


def plot_branches(ax, mu, g, delta, k_min, k_max, quant):
    """Analytic fixed-point branches of `quant` ('R' or 'ratio'=V_A/Ā²) vs K, recomputed at plot time."""
    Ktc = 2.0 * delta
    Ks = np.linspace(k_min, k_max, 800)
    up, lo = np.full_like(Ks, np.nan), np.full_like(Ks, np.nan)
    for i, K in enumerate(Ks):
        A1, A2 = B.sync_abar(K, mu, g)
        for A, arr in ((A1, up), (A2, lo)):
            v = B.branch_values(K, A, mu, g, delta)
            if v:
                arr[i] = v["R"] if quant == "R" else v["VA"] / v["Abar"] ** 2
    ax.plot(Ks[Ks < Ktc], np.zeros((Ks < Ktc).sum()), color=C_ASYNC, lw=1.1, ls="-", zorder=2)
    ax.plot(Ks[Ks >= Ktc], np.zeros((Ks >= Ktc).sum()), color=C_ASYNC, lw=1.0, ls="--", zorder=2)
    ax.plot(Ks, up, color=C_SYNC, lw=1.2, ls="-", zorder=3)
    ax.plot(Ks, lo, color=C_SYNC, lw=1.0, ls="--", zorder=3)
    ax.set_xlim(k_min, k_max)


def main():
    d = np.load(NPZ)
    mus, Ks = d["mus"], d["Ks"]
    Tm, Tf, Rm, Rf, ratio, Rp_m = d["Tm"], d["Tf"], d["Rm"], d["Rf"], d["ratio"], d["Rp_m"]
    g, delta = float(d["gamma"]), float(d["delta"])
    k_min, k_max = float(d["k_min"]), float(d["k_max"])
    Ttot = float(d["n_ramp"]) * float(d["tau_d"])

    set_prl_style()
    fig = plt.figure(figsize=(3.4, 5.6), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.06)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.0, 1.05])

    # ── rows 1–2: analytic bifurcation panels (V_A/Ā² and R) per μ (columns) ───
    gst = sf_top.add_gridspec(2, len(mus))
    lab = iter("abcd")
    for r, quant in enumerate(("ratio", "R")):
        for c, mu in enumerate(mus):
            ax = sf_top.add_subplot(gst[r, c])
            plot_branches(ax, mu, g, delta, k_min, k_max, quant)
            if quant == "R":
                for di, mk in ((0, "o"), (1, "s")):                # 0=fwd, 1=bwd (ascending-K order)
                    ax.plot(Ks, Rp_m[c, di], mk, ms=3.0, mfc="none", mec=C_MICRO, mew=0.8,
                            ls="none", zorder=5)
                ax.set_ylim(-0.03, 1.03)
                ax.set_xlabel(r"global coupling $K$", labelpad=1)
            ax.set_ylabel(r"$V_A/\bar A^2$" if quant == "ratio" else r"$R$", labelpad=2)
            if r == 0:
                ax.set_title(rf"$\mu={mu:g}$", fontsize=6.8, pad=2)
            _panel_label(ax, next(lab))
    sf_top.axes[3].legend(handles=[Line2D([0], [0], color=C_SYNC, lw=1.2, label="stable"),
                                   Line2D([0], [0], color=C_SYNC, lw=1.0, ls="--", label="unstable"),
                                   Line2D([0], [0], marker="o", ls="none", mfc="none", mec=C_MICRO,
                                          mew=0.8, ms=3, label="ramp fwd"),
                                   Line2D([0], [0], marker="s", ls="none", mfc="none", mec=C_MICRO,
                                          mew=0.8, ms=3, label="ramp bwd")],
                          loc="lower right", fontsize=4.6, handlelength=1.3, labelspacing=0.16)

    # ── rows 3–4: R(t) ramp dynamics per μ (left axis) + MF V_A/Ā²(t) (right axis) ─────
    gsb = sf_bot.add_gridspec(len(mus), 1)
    for row, mu in enumerate(mus):
        dyn = sf_bot.add_subplot(gsb[row, 0])
        tw = dyn.twinx()
        for di in (0, 1):                                          # 0=fwd, 1=bwd
            dyn.plot(Tm, Rm[row, di], color=C_MICRO, lw=0.9, ls=LS[di], zorder=3)
            dyn.plot(Tf, Rf[row, di], color=C_MF, lw=1.1, ls=LS[di], zorder=4)
            tw.plot(Tf, ratio[row, di], color=C_RATIO, lw=0.9, ls=LS[di], zorder=2)
        dyn.set_xlim(0, Ttot); dyn.set_ylim(-0.03, 1.03)
        dyn.set_ylabel(r"$R(t)$", labelpad=2)
        dyn.set_title(rf"$\mu={mu:g}$", fontsize=6.8, pad=2)
        tw.set_ylabel(r"$V_A/\bar A^2$", color=C_RATIO, labelpad=2)
        tw.tick_params(axis="y", colors=C_RATIO); tw.spines["right"].set_color(C_RATIO)
        tw.set_ylim(bottom=0.0)
        if row == len(mus) - 1:
            dyn.set_xlabel(r"time $t$", labelpad=1)
        _panel_label(dyn, "ef"[row])
        if row == 0:
            dyn.legend(handles=[Line2D([0], [0], color=C_MICRO, lw=0.9, label="micro $R$"),
                                Line2D([0], [0], color=C_MF, lw=1.1, label="mean-field $R$"),
                                Line2D([0], [0], color=C_RATIO, lw=0.9, label=r"MF $V_A/\bar A^2$"),
                                Line2D([0], [0], color="0.4", lw=1.0, ls="-", label="forward"),
                                Line2D([0], [0], color="0.4", lw=1.0, ls="--", label="backward")],
                       loc="center right", fontsize=4.6, handlelength=1.5, labelspacing=0.16,
                       ncol=2, columnspacing=0.7)

    fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
