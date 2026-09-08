r"""
Adaptive Kuramoto — microscopic coupling structure & coherence across the three rules
=====================================================================================

Contrasts the three adaptation rules G_A ∈ {cos, sin, |sin|} of the microscopic
adaptively-coupled Kuramoto network at a single adaptation rate μ, showing:

  (a) the average phase-coherence dynamics R(t) of all three rules overlaid, and
  (b–d) the final (frequency-sorted, block-averaged) microscopic coupling matrix A_ij
        for each rule.

The microscopic model is the all-to-all adaptive network of the weight-variance study
(``grid_search/kmo_lmmf_variance_bound_sweep.py``), reused verbatim via its CONFIG and
``simulate_micro`` (uniform natural frequencies ω ~ U(±a), Ȧ_ij = μ G_A(θ_j−θ_i) + γ(1−A_ij)).
Only the microscopic model is shown here, and it does not depend on the LMMF variance
budget, so the three runs are computed directly (and cached) rather than read from the
(slow) full sweep. Change `MU` / `TRIAL` to pick the adaptation rate / random realization.

Run in the ``pycobi`` conda env (needs PyRates for ``simulate_micro``):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_adaptive_rules_figure.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# reuse the sweep's config + microscopic simulator (weight-variance convention)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grid_search")))
import kmo_lmmf_variance_bound_sweep as SW
from kmo_adaptive_single_sweep import simulate_micro, block_average

MU = 0.001                       # adaptation rate to plot
TRIAL = 0                        # random realization (matches the sweep's per-trial RNG)
RULES = ["cos", "sin", "|sin|"]
OUT = "/home/rgast/data/mpmf_simulations/kmo_adaptive_rules_mu{mu:g}"
FORCE = False                    # True => recompute even if the cache exists

# rule -> colour (coherence overlay) and filename-safe tag
C_RULE = {"cos": "#1f77b4", "sin": "#e63946", "|sin|": "#2a9d8f"}
LBL = {"cos": r"\cos", "sin": r"\sin", "|sin|": r"|\sin|"}   # math body (no $ delimiters)
MATRIX_CMAP = "magma"


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


def _panel_label(ax, letter, dx=-6, dy=3):
    ax.annotate(rf"$\bf({letter})$", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


# ════════════════════════════════════════════════════════════════════════════
#  microscopic runs (μ, three rules) — generated once, cached to .npz
# ════════════════════════════════════════════════════════════════════════════
def get_data(cfg, mu, trial):
    cache = OUT.format(mu=mu) + f"_trial{trial}_data.npz"
    if os.path.exists(cache) and not FORCE:
        d = np.load(cache, allow_pickle=True)
        print(f"[cache] {cache}")
        return (d["t"], {r: d[f"R_{i}"] for i, r in enumerate(RULES)},
                {r: d[f"A_{i}"] for i, r in enumerate(RULES)})

    N, K, gamma, a = cfg["N"], cfg["K"], cfg["gamma"], cfg["omega_halfwidth"]
    A0, res = cfg["A0"], cfg["save_res"]
    # same per-trial RNG / frequencies / IC as SW.main (uniform, sorted ⇒ freq-ordered A_ij)
    rng = np.random.default_rng(cfg["seed"] + trial)
    omega = np.sort(rng.uniform(-a, a, N))
    theta0 = rng.normal(0.0, cfg["sigma0"], N)
    print(f"microscopic adaptive Kuramoto — μ={mu}, N={N}, K={K}, ω~U(±{a}), trial={trial}")

    t, R, A = None, {}, {}
    for rule in RULES:
        tt, Rm, Ab, VA, A_fin = simulate_micro(theta0, A0, omega, K, mu, gamma, rule, cfg)
        t = tt
        R[rule] = Rm
        A[rule] = block_average(A_fin, res)
        print(f"  G_A={rule:6s} -> R(end)={Rm[-1]:.3f}  Ā={Ab[-1]:.3f}  V_A/Ā²={VA[-1]/Ab[-1]**2:.4f}")

    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, t=t, **{f"R_{i}": R[r] for i, r in enumerate(RULES)},
             **{f"A_{i}": A[r] for i, r in enumerate(RULES)})
    print(f"[saved] {cache}")
    return t, R, A


# ════════════════════════════════════════════════════════════════════════════
#  figure — (a) R(t) overlay | (b–d) final coupling matrices per rule
# ════════════════════════════════════════════════════════════════════════════
def make_figure(cfg, mu, trial):
    t, R, A = get_data(cfg, mu, trial)

    fig = plt.figure(figsize=(7.0, 2.0), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.05)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.9, 1.0, 1.0, 1.0])

    # (a) coherence dynamics, all rules overlaid
    axR = fig.add_subplot(gs[0, 0])
    for rule in RULES:
        axR.plot(t, R[rule], color=C_RULE[rule], lw=1.1, label=rf"${LBL[rule]}$")
    axR.set_xlim(t[0], t[-1]); axR.set_ylim(-0.02, 1.02)
    axR.set_yticks([0, 0.5, 1.0])
    axR.set_xlabel(r"time $t$", labelpad=1)
    axR.set_ylabel(r"phase coherence $R(t)$", labelpad=2)
    axR.legend(title=r"$G_A$", loc="center right", ncol=1, handlelength=1.3,
               labelspacing=0.3, frameon=True, framealpha=0.9, edgecolor="0.8")
    _panel_label(axR, "a", dx=-30)

    # (b–d) final coupling matrix per rule (own colour scale — ranges differ by rule)
    for k, rule in enumerate(RULES):
        ax = fig.add_subplot(gs[0, k + 1])
        M = A[rule]
        im = ax.imshow(M, origin="lower", aspect="equal", cmap=MATRIX_CMAP,
                       vmin=np.nanmin(M), vmax=np.nanmax(M), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$G_A={LBL[rule]}$", fontsize=7, pad=2)
        if k == 0:
            ax.set_ylabel(r"osc. $i$ (by $\omega_i$)", labelpad=2)
        ax.set_xlabel(r"osc. $j$", labelpad=1)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=5, pad=0.6)
        cb.set_label(r"$A_{ij}$", fontsize=6, labelpad=1)
        _panel_label(ax, "bcd"[k], dx=-4)

    fig.suptitle(rf"microscopic adaptive Kuramoto, $\mu={mu:g}$ ($\gamma={cfg['gamma']:g}$, "
                 rf"$K={cfg['K']:g}$)", fontsize=8, x=0.01, ha="left")

    out = OUT.format(mu=mu)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {out}.pdf / .png")


if __name__ == "__main__":
    set_prl_style()
    make_figure(SW.CONFIG, MU, TRIAL)
