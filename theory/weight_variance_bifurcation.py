r"""
Adaptive-coupling Kuramoto: bifurcation diagrams of the mean-field fixed points (Section D)
===========================================================================================

Draws the bifurcation structure of the steady states of the reduced mean-field system
(PRL_2026 "Weight Variance", Section D), at fixed Δ and γ.  Fixed points:

  * asynchronous branch:  R̃₀ = 0,  Ā₀ = 1,  V_A = 0  — stable for K < 2Δ  (Eqs. 76, 77 with R=0);
  * synchronous branch:   Ā_{1/2} = ½(1+μ/γ) ± √[¼(1+μ/γ)² − 2μΔ/(γK)]        (Eq. 81)
                          R̃₁ = √(1 − 2Δ/(KĀ))                                 (Eq. 76)
                          C_A = (μ/2γ)(S² − R̃₁⁴),  V_A = (μ²/2γ²)(S² − R̃₁⁴)   (Eqs. 82, 83)
    with S = ⟨|c|²⟩ the locked+drifting order parameter (Eq. 75; evaluated on-manifold via
    weight_variance_meanfield.order_parameter_S).  Ā₁ (upper) is a stable node, Ā₂ (lower) an
    unstable saddle.
  * saddle-node (fold):  Δ = K(γ+μ)²/(8γμ)  ⇒  K_SN = 8γμΔ/(γ+μ)²;
    transcritical (R̃₁ meets R̃₀) at K = 2Δ;  cusp (organizing centre) at (K=2Δ, μ=γ).
    ⇒ bistable for μ > γ (fold at physical R over K_SN < K < 2Δ), monostable for μ < γ.

Two-column PRL figure, 2×4 grid:
  * col 1, panel (a): 2-D bifurcation diagram in the K–μ plane (saddle-node + transcritical curves,
    cusp, bistable region; dotted lines mark the two μ used for the 1-D cuts);
  * cols 2–4: 1-D bifurcation diagrams (R, Ā, V_A vs K) for a bistable μ (top) and a monostable μ
    (bottom); stability by line style (solid=stable, dashed=unstable), the critical bifurcation by a
    marker, a dashed vertical line at the critical K, and its condition labelled in the R column.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_bifurcation.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from weight_variance_meanfield import order_parameter_S    # on-manifold S = ⟨|c|²⟩ (Eq. 75)

CONFIG = dict(
    gamma=0.001, delta=1.0,
    mu_bi=0.01,      # bistable cut (μ > γ)
    mu_mono=0.0003,  # monostable cut (μ < γ)
    k_min=0.3, k_max=3.0, n_k=1200,
    mu_min=1e-4, mu_max=3e-2,          # μ-axis range of panel (a)
    out="/home/rgast/data/kmo_adaptive/weight_variance_bifurcation",
)

C_ASYNC, C_SYNC, C_CRIT = "0.55", "#1f77b4", "#e63946"
QUANTS = ["R", "Abar", "VA"]
Q_LABEL = {"R": r"$\tilde R$", "Abar": r"$\bar A$", "VA": r"$V_A$"}


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 5.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
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


# ════════════════════════════════════════════════════════════════════════════
#  fixed-point branches (Section D)
# ════════════════════════════════════════════════════════════════════════════
def sync_abar(K, mu, g):
    """Synchronous-branch coupling roots Ā₁ (upper/stable), Ā₂ (lower/saddle) — Eq. 81.
    Returns (nan, nan) where the roots are complex."""
    b = 1.0 + mu / g
    disc = 0.25 * b ** 2 - 2.0 * mu * CONFIG["delta"] / (g * K)
    if disc < 0:
        return np.nan, np.nan
    s = np.sqrt(disc)
    return 0.5 * b + s, 0.5 * b - s


def branch_values(K, A, mu, g, delta):
    """(R, Ā, V_A) on the synchronous branch for coupling root Ā at coupling K.
    Returns None if unphysical (R̃₁² < 0, i.e. KĀ < 2Δ)."""
    if not np.isfinite(A):
        return None
    J = K * A
    R2 = 1.0 - 2.0 * delta / J
    if R2 < 0:
        return None
    R = np.sqrt(R2)
    S = float(order_parameter_S(R, A, K, delta))
    dV = S ** 2 - R ** 4
    return dict(R=R, Abar=A, VA=mu ** 2 / (2.0 * g ** 2) * dV, CA=mu / (2.0 * g) * dV)


def async_value(quant):
    return {"R": 0.0, "Abar": 1.0, "VA": 0.0}[quant]


def critical_K_and_condition(mu, g, delta):
    """Critical coupling + its condition label for a given μ.
    μ>γ → saddle-node fold; μ<γ (or =γ) → transcritical."""
    if mu > g:
        Ksn = 8.0 * g * mu * delta / (g + mu) ** 2
        return Ksn, "saddle–node", r"$\Delta=\dfrac{K(\gamma+\mu)^2}{8\gamma\mu}$"
    return 2.0 * delta, "transcritical", r"$K=2\Delta$"


# ════════════════════════════════════════════════════════════════════════════
#  panel (a): 2-D bifurcation diagram in the K–μ plane
# ════════════════════════════════════════════════════════════════════════════
def plot_2d(ax, cfg):
    g, delta = cfg["gamma"], cfg["delta"]
    mu = np.logspace(np.log10(cfg["mu_min"]), np.log10(cfg["mu_max"]), 600)
    Ksn = 8.0 * g * mu * delta / (g + mu) ** 2               # saddle-node curve K_SN(μ)
    Ktc = 2.0 * delta                                        # transcritical line
    phys = mu > g                                            # physical fold only for μ>γ

    # bistable region: μ>γ, K_SN(μ) < K < 2Δ
    ax.fill_betweenx(mu[phys], Ksn[phys], Ktc, color=C_SYNC, alpha=0.13, lw=0, zorder=0)
    ax.plot(Ksn[phys], mu[phys], color=C_CRIT, lw=1.3, zorder=3)          # physical saddle-node
    ax.plot(Ksn[~phys], mu[~phys], color=C_CRIT, lw=0.9, ls=":", zorder=3)  # unphysical continuation
    ax.axvline(Ktc, color="0.35", lw=1.0, ls="--", zorder=2)             # transcritical
    ax.plot([Ktc], [g], marker="*", ms=7, mfc="k", mec="k", zorder=5)    # cusp

    for muv, lab in ((cfg["mu_bi"], "(b–d)"), (cfg["mu_mono"], "(e–g)")):
        ax.axhline(muv, color="0.5", lw=0.7, ls=":", zorder=1)
        ax.annotate(lab, xy=(cfg["k_max"], muv), xytext=(-2, 1), textcoords="offset points",
                    ha="right", va="bottom", fontsize=5.4, color="0.35")

    ax.set_xscale("linear"); ax.set_yscale("log")
    ax.set_xlim(cfg["k_min"], cfg["k_max"]); ax.set_ylim(cfg["mu_min"], cfg["mu_max"])
    ax.set_xlabel(r"global coupling $K$", labelpad=1)
    ax.set_ylabel(r"adaptation rate $\mu$", labelpad=2)
    ax.annotate("bistable", xy=(0.5 * (Ktc + 0.66), 0.02), fontsize=5.6, color=C_SYNC, ha="center")
    ax.annotate(r"cusp $(2\Delta,\gamma)$", xy=(Ktc, g), xytext=(4, 4), textcoords="offset points",
                fontsize=5.4)
    ax.set_title(r"$K$–$\mu$ bifurcations", fontsize=6.8, pad=3)
    _panel_label(ax, "a", dx=-30)


# ════════════════════════════════════════════════════════════════════════════
#  1-D bifurcation diagram of one quantity vs K, at one μ
# ════════════════════════════════════════════════════════════════════════════
def plot_1d(ax, quant, mu, cfg, show_label):
    g, delta = cfg["gamma"], cfg["delta"]
    Ks = np.linspace(cfg["k_min"], cfg["k_max"], cfg["n_k"])
    Ktc = 2.0 * delta
    up, lo = np.full_like(Ks, np.nan), np.full_like(Ks, np.nan)
    for i, K in enumerate(Ks):
        A1, A2 = sync_abar(K, mu, g)
        v1, v2 = branch_values(K, A1, mu, g, delta), branch_values(K, A2, mu, g, delta)
        if v1:
            up[i] = v1[quant]
        if v2:
            lo[i] = v2[quant]

    # asynchronous branch (R=0, Ā=1, V_A=0): stable for K<2Δ, unstable above
    va = async_value(quant)
    ax.plot(Ks[Ks < Ktc], np.full((Ks < Ktc).sum(), va), color=C_ASYNC, lw=1.2, ls="-", zorder=2)
    ax.plot(Ks[Ks >= Ktc], np.full((Ks >= Ktc).sum(), va), color=C_ASYNC, lw=1.0, ls="--", zorder=2)
    # synchronous branch: upper root stable (solid), lower root saddle (dashed)
    ax.plot(Ks, up, color=C_SYNC, lw=1.2, ls="-", zorder=3)
    ax.plot(Ks, lo, color=C_SYNC, lw=1.0, ls="--", zorder=3)

    # critical point: marker on the branch + dashed vertical line (+ condition label in R column)
    Kc, _, cond = critical_K_and_condition(mu, g, delta)
    if mu > g:                                           # saddle-node fold value
        A_sn = 0.5 * (1.0 + mu / g)
        vc = branch_values(Kc, A_sn, mu, g, delta)
        yc = vc[quant] if vc else va
    else:                                                # transcritical (meets async)
        yc = va
    ax.axvline(Kc, color=C_CRIT, lw=0.8, ls="--", zorder=1)
    ax.plot([Kc], [yc], marker="o", ms=3.6, mfc=C_CRIT, mec="k", mew=0.4, zorder=6)
    if show_label:
        ax.annotate(cond, xy=(Kc, 0.1), xycoords=("data", "axes fraction"), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=5.6, color=C_CRIT)

    ax.set_xlim(cfg["k_min"], cfg["k_max"])
    ax.set_ylabel(Q_LABEL[quant], labelpad=2)


# ════════════════════════════════════════════════════════════════════════════
#  assemble the 2×4 figure
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    set_prl_style()
    fig = plt.figure(figsize=(7.0, 2.5), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.06)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.0, 1.0, 1.0])

    plot_2d(fig.add_subplot(gs[:, 0]), cfg)

    rows = [(cfg["mu_bi"], "bistable"), (cfg["mu_mono"], "monostable")]
    letters = iter("bcdefg")
    for r, (mu, regime) in enumerate(rows):
        for c, q in enumerate(QUANTS):
            ax = fig.add_subplot(gs[r, c + 1])
            plot_1d(ax, q, mu, cfg, show_label=(c == 0))
            if r == 0:
                ax.set_title(Q_LABEL[q] + r"$(K)$", fontsize=6.8, pad=3)
            if r == len(rows) - 1:
                ax.set_xlabel(r"global coupling $K$", labelpad=1)
            _panel_label(ax, next(letters))

    # stability / branch legend (in panel (b))
    axb = fig.axes[1]
    axb.legend(handles=[Line2D([0], [0], color=C_SYNC, lw=1.2, label="stable"),
                        Line2D([0], [0], color=C_SYNC, lw=1.0, ls="--", label="unstable"),
                        Line2D([0], [0], color=C_ASYNC, lw=1.2, label="async."),
                        Line2D([0], [0], marker="o", ls="none", mfc=C_CRIT, mec="k", mew=0.4,
                               ms=3.6, label="bifurcation")],
               loc="center right", fontsize=5.0, handlelength=1.4, labelspacing=0.2, borderaxespad=0.3)

    fig.suptitle(rf"Mean-field bifurcations of adaptively-coupled Kuramoto oscillators "
                 rf"($\Delta={cfg['delta']:g}$, $\gamma={cfg['gamma']:g}$)", fontsize=7.5)
    fig.savefig(cfg["out"] + ".svg"); fig.savefig(cfg["out"] + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {cfg['out']}.svg / .png")


if __name__ == "__main__":
    main()
