r"""
Bifurcation structure of the phase coherence R for adaptively coupled Kuramoto oscillators
==========================================================================================

Steady states of the reduced mean-field system (Eqs. 8 & 10 of the PRL_2026 "Weight Variance"
manuscript):
    Ṙ   = -Δ R + (K Ā / 2) R (1 - R²)          -> sync fixed point  R² = 1 - 2Δ/(K Ā),  or  R = 0
    Ā̇   = μ R² + γ (1 - Ā)                       -> slow nullcline    Ā = 1 + (μ/γ) R²

Eliminating R gives a quadratic in Ā with roots
    Ā_± = ½(1 + μ/γ) ± sqrt[ ¼(1 + μ/γ)² - 2μΔ/(γK) ],
each carrying the synchronized coherence R² = 1 - 2Δ/(K Ā). A useful identity is
    R_±² = 1 - (γ/μ) Ā_∓ ,
so the physicality (R² ≥ 0) of the '-' root (the saddle) is governed by μ vs γ:

  * μ < γ  (continuous):  the '-' root has R² < 0 everywhere (unphysical). The stable '+' root
    collides with the asynchronous branch (R=0) at Δ_c = K/2 in a transcritical bifurcation.
  * μ > γ  (hysteretic):  the '-' root is unphysical for Δ < Δ_c but becomes a physical saddle
    (small R > 0) for Δ_c < Δ < Δ_SN. It emerges from the async branch at Δ_c (transcritical)
    and annihilates with the stable '+' root at the saddle-node
        Δ_SN = K (γ + μ)² / (8 μ γ).

The figure draws R vs Δ for both regimes: stable sync ('+' root), saddle ('-' root, physical
part only), and the async branch (R = 0), with line style encoding stability.

    python bifurcation_R.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── configuration ───────────────────────────────────────────────────────────
K = 3.0
REGIMES = [                                   # (mu, gamma, title, x-limit)
    dict(mu=0.01,  g=0.001, title=r"hysteretic  $\mu{>}\gamma$",  xmax=6.0),
    dict(mu=0.001, g=0.01,  title=r"continuous  $\mu{<}\gamma$",  xmax=2.2),
]
OUT = "bifurcation_R"                          # output basename (.pdf and .png)

C_STABLE, C_SADDLE = "#c1121f", "#1f77b4"     # stable sync (red), saddle (blue)


# ── steady-state branches ────────────────────────────────────────────────────
def roots_Abar(mu, g, delta, K):
    """The two roots Ā_± of the steady-state quadratic (Eq. 82); (nan, nan) past the fold."""
    a = 1.0 + mu / g
    disc = 0.25 * a * a - 2.0 * mu * delta / (g * K)
    if disc < 0:
        return np.nan, np.nan
    s = np.sqrt(disc)
    return 0.5 * a + s, 0.5 * a - s            # Ā_+, Ā_-


def coherence_branches(mu, g, K, deltas):
    """R on the '+' and '-' roots as functions of Δ (nan where R² < 0, i.e. unphysical)."""
    Ap = np.array([roots_Abar(mu, g, d, K)[0] for d in deltas])
    Am = np.array([roots_Abar(mu, g, d, K)[1] for d in deltas])
    Rp2 = 1.0 - 2.0 * deltas / (K * Ap)
    Rm2 = 1.0 - 2.0 * deltas / (K * Am)
    Rp = np.sqrt(np.where(Rp2 > 0, Rp2, np.nan))
    Rm = np.sqrt(np.where(Rm2 > 0, Rm2, np.nan))
    return Rp, Rm


def thresholds(mu, g, K):
    dc = K / 2.0                               # transcritical / async stability boundary
    dSN = K * (g + mu) ** 2 / (8.0 * mu * g)   # saddle-node (fold)
    return dc, dSN


# ── plotting ─────────────────────────────────────────────────────────────────
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def make_figure():
    set_prl_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained")
    d = np.linspace(1e-3, max(r["xmax"] for r in REGIMES), 2000)

    for ax, reg in zip(axes, REGIMES):
        mu, g = reg["mu"], reg["g"]
        dc, dSN = thresholds(mu, g, K)
        Rp, Rm = coherence_branches(mu, g, K, d)

        # async branch R = 0: stable (solid) for Δ > Δ_c, unstable (dashed) for Δ < Δ_c
        ax.plot(d[d >= dc], 0.0 * d[d >= dc], color="0.35", lw=1.3)
        ax.plot(d[d < dc],  0.0 * d[d < dc],  color="0.35", lw=1.0, ls=(0, (4, 3)))

        # stable synchronized branch ('+' root)
        ax.plot(d, Rp, color=C_STABLE, lw=1.6)

        # saddle ('-' root): only physical for μ > γ, in (Δ_c, Δ_SN)
        if np.any(np.isfinite(Rm)):
            ax.plot(d, Rm, color=C_SADDLE, lw=1.3, ls=(0, (4, 3)))

        # threshold markers
        ax.axvline(dc, color="0.6", lw=0.6)
        ax.text(dc, 1.02, r"$\Delta_c$", fontsize=6.5, ha="center")
        if mu > g:
            ax.axvline(dSN, color="0.6", lw=0.6)
            ax.text(dSN, 1.02, r"$\Delta_{\rm SN}$", fontsize=6.5, ha="center")
            ax.axvspan(dc, dSN, color="0.94", zorder=0)   # bistable window

        ax.set_xlabel(r"heterogeneity $\Delta$")
        ax.set_ylabel(r"phase coherence $R$")
        ax.set_title(reg["title"], fontsize=7.5)
        ax.set_xlim(0, reg["xmax"])
        ax.set_ylim(-0.03, 1.05)

    axes[0].legend(handles=[
        Line2D([0], [0], color=C_STABLE, lw=1.6, label="stable sync (+ root)"),
        Line2D([0], [0], color=C_SADDLE, lw=1.3, ls=(0, (4, 3)), label="saddle (- root)"),
        Line2D([0], [0], color="0.35", lw=1.3, label="async, stable"),
        Line2D([0], [0], color="0.35", lw=1.0, ls=(0, (4, 3)), label="async, unstable"),
    ], fontsize=5.4, loc="center right")

    fig.suptitle(r"Bifurcation structure of the coherence $R$: "
                 r"the saddle is physical only for $\mu>\gamma$", fontsize=7.4)
    fig.savefig(OUT + ".svg")
    fig.savefig(OUT + ".png", dpi=200)
    plt.close(fig)
    print(f"[saved] {OUT}.svg / {OUT}.png")


if __name__ == "__main__":
    make_figure()