r"""
Adaptive-coupling Kuramoto: relative weight variance V_A/Ā² vs Δ — mean field vs micro, across μ
================================================================================================

Loads the microscopic (rule × μ × Δ) sweep written by ``weight_variance_rule_micro.py`` and plots
the steady-state relative weight variance V_A/Ā² in a single-column 2×2 grid: columns = adaptation
rule G ∈ {cos, sin}; row 1 = V_A/Ā² vs heterogeneity Δ, row 2 = V_A/Ā² vs steady-state phase
coherence R. In every panel:

  * mean field — TWO models, one colour per μ, both evaluated at the steady-state (R, Ā, S):
      – SOLID  : full closure  C_A = C_S + C_F  →  V_A = μ²/[2γ²(γ+2Δ)] (γ(1−R⁴) + 2Δ(S²−R⁴)),
      – DASHED : static closure C_A = C_S       →  V_A = μ²/(2γ²) (S²−R⁴).
    For the Δ view each is the coherent-IC-selected stable branch (synchronized where physical, else
    asynchronous R=0, Ā=1); for the R view the synchronized (+saddle) branch spanning R∈[0,1].
  * microscopic (MARKERS, same colour per μ) — V_A/Ā² (and, for the R view, the steady-state R)
    measured directly from the simulation.

Steady-state weight-variance closures (S = ⟨|c|²⟩ clipped to [0,1], b = KĀR):

    V_A(full)   = μ² / [2γ²(γ + 2Δ)] · ( γ(1 − R⁴) + 2Δ(S² − R⁴) ),   [C_A = C_S + C_F]
    V_A(static) = μ² / (2γ²) · ( S² − R⁴ ).                            [C_A = C_S]

(The full per-μ bifurcation structure — stable/unstable branches, folds, transcritical points — is
drawn in ``weight_variance_bifurcation_rules.py``.)

    python weight_variance_rule_meanfield.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from weight_variance_analysis import S_order, branches, set_prl_style, _panel_label  # noqa: E402

CONFIG = dict(
    micro_npz=dp.mpmf("weight_variance_rule_micro_oainit.npz"),
    n_curve=800,                 # dense Δ grid for the mean-field curves
    cmap="viridis",              # colour map over μ
    out=dp.mpmf("weight_variance_rule_vratio"),
)


# ════════════════════════════════════════════════════════════════════════════
#  closed forms
# ════════════════════════════════════════════════════════════════════════════
def va_steady(R2, A, delta, K, mu, gamma, closure="full"):
    """Steady-state weight variance V_A; R2 = R², A = Ā. `closure` selects the model:
      "full"   → C_A = C_S + C_F :  μ²/[2γ²(γ+2Δ)] (γ(1−R⁴) + 2Δ(S²−R⁴)),
      "static" → C_A = C_S       :  μ²/(2γ²) (S²−R⁴).
    S = ⟨|c|²⟩ is clipped to its physical range [0, 1]."""
    R2 = np.clip(R2, 0.0, None)
    S = np.clip(S_order(R2, A, delta, K), 0.0, 1.0)
    dV = S ** 2 - R2 ** 2
    if closure == "static":
        return mu ** 2 / (2 * gamma ** 2) * dV
    return mu ** 2 / (2 * gamma ** 2 * (gamma + 2 * delta)) * (gamma * (1.0 - R2 ** 2) + 2 * delta * dV)


def _rel(R2, A, delta, K, mu, gamma, closure="full"):
    return va_steady(R2, A, delta, K, mu, gamma, closure) / np.asarray(A) ** 2


def mf_selected(rule, d, mu, gamma, K, closure="full"):
    """Coherent-IC-selected stable V_A/Ā²(Δ): the synchronized branch where physical, else the
    asynchronous branch (R=0, Ā=1) — i.e. the state a coherent-IC microscopic run settles onto."""
    ones = np.ones_like(d)
    rel_async = _rel(np.zeros_like(d), ones, d, K, mu, gamma, closure)
    _, rel_sync = mf_sync_RV(rule, d, mu, gamma, K, closure)
    return np.where(np.isfinite(rel_sync), rel_sync, rel_async)


def mf_sync_RV(rule, d, mu, gamma, K, closure="full"):
    """(R, V_A/Ā²) along the SYNCHRONIZED branch, parametrized by Δ (nan where unphysical)."""
    ones = np.ones_like(d)
    if rule == "cos":
        br = branches(d, mu, K, gamma)["sync"]
        R, rel = br["R"], _rel(br["R"] ** 2, br["A"], d, K, mu, gamma, closure)
    else:                                                               # sin: Ā ≡ 1
        R2 = 1.0 - 2.0 * d / K
        R = np.where(R2 > 0, np.sqrt(np.clip(R2, 0, None)), np.nan)
        rel = np.where(R2 > 0, _rel(np.clip(R2, 0, None), ones, d, K, mu, gamma, closure), np.nan)
    return R, rel


def mf_full_RV(rule, d, mu, gamma, K, closure="full"):
    """(R, V_A/Ā²) covering the FULL R∈[0,1] range for the V_A/Ā²-vs-R view. For cos the synchronized
    (stable, R∈[R_SN,1]) AND saddle (unstable, R∈[0,R_SN]) branches together span all R and meet at
    the fold; for sin the single branch already spans R∈[0,1]. Sorted by R."""
    ones = np.ones_like(d)
    if rule == "cos":
        br = branches(d, mu, K, gamma)
        Rl, rl = [], []
        for name in ("sync", "saddle"):
            R = br[name]["R"]
            rel = _rel(R ** 2, br[name]["A"], d, K, mu, gamma, closure)
            fin = np.isfinite(R) & np.isfinite(rel)
            Rl.append(R[fin]); rl.append(rel[fin])
        R, rel = np.concatenate(Rl), np.concatenate(rl)
    else:
        R2 = 1.0 - 2.0 * d / K
        fin = R2 > 0
        R = np.sqrt(R2[fin]); rel = _rel(R2[fin], ones[fin], d[fin], K, mu, gamma, closure)
    order = np.argsort(R)
    return R[order], rel[order]


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main(cfg=CONFIG):
    dat = np.load(cfg["micro_npz"], allow_pickle=True)
    rules = [str(r) for r in dat["rules"]]
    mus, deltas, t = dat["mus"], dat["deltas"], dat["t"]
    R, Abar, VA = dat["R"], dat["Abar"], dat["VA"]           # (rule, μ, Δ, time)
    K, gamma = float(dat["K"]), float(dat["gamma"])
    T, trans_frac = float(dat["T"]), float(dat["trans_frac"])
    tail = t >= trans_frac * T
    d_curve = np.linspace(1e-3, deltas.max() * 1.02, cfg["n_curve"])
    cmap = plt.get_cmap(cfg["cmap"])
    colors = [cmap(x) for x in np.linspace(0.12, 0.88, len(mus))]

    set_prl_style()
    ncol = len(rules)                                        # columns = rule; rows = (vs Δ, vs R)
    fig, axes = plt.subplots(2, ncol, figsize=(3.4, 3.0), squeeze=False)   # shorter panels

    for i, rule in enumerate(rules):
        ax0, ax1 = axes[0][i], axes[1][i]                    # vs Δ (top) / vs R (bottom)
        ymax = 0.0
        for m, mu in enumerate(mus):
            col = colors[m]
            # steady-state microscopic quantities
            R_ss = np.nanmean(R[i, m][:, tail], axis=1)
            A_ss = np.nanmean(Abar[i, m][:, tail], axis=1)
            rel_micro = np.nanmean(VA[i, m][:, tail], axis=1) / A_ss ** 2
            # both mean-field models — full C_A=C_S+C_F (solid), static C_A=C_S (dashed)
            # row 0: V_A/Ā² vs Δ ;  row 1: V_A/Ā² vs steady-state R
            for cl, ls in (("full", "-"), ("static", "--")):
                rel_dcurve = mf_selected(rule, d_curve, float(mu), gamma, K, cl)
                R_fcurve, rel_fcurve = mf_full_RV(rule, d_curve, float(mu), gamma, K, cl)  # full R∈[0,1]
                ax0.plot(d_curve, rel_dcurve, ls, color=col, lw=1.3, zorder=2)
                ax1.plot(R_fcurve, rel_fcurve, ls, color=col, lw=1.3, zorder=2)
                if cl == "full":
                    ymax = max(ymax, np.nanmax(rel_dcurve[np.isfinite(rel_dcurve)]))
            ax0.plot(deltas, rel_micro, "o", ms=3.4, mfc=col, mec="k", mew=0.3, zorder=4)
            ax1.plot(R_ss, rel_micro, "o", ms=3.4, mfc=col, mec="k", mew=0.3, zorder=4)
            ymax = max(ymax, np.nanmax(rel_micro))

        ymax = 1.12 * (ymax or 1.0)
        ax0.set_xlim(0, deltas.max() * 1.02); ax0.set_ylim(0, ymax)
        ax1.set_xlim(0, 1.02); ax1.set_ylim(0, ymax)
        ax0.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
        ax1.set_xlabel(r"phase coherence $R$", labelpad=1)
        if i == 0:
            ax0.set_ylabel(r"rel. weight variance $V_A/\bar A^2$", labelpad=2)
            ax1.set_ylabel(r"rel. weight variance $V_A/\bar A^2$", labelpad=2)
        rule_tex = {"cos": r"\cos", "sin": r"\sin"}.get(rule, rule)
        ax0.set_title(rf"$G(x)={rule_tex}(x)$", fontsize=8, pad=3)
        _panel_label(ax0, "abcd"[i]); _panel_label(ax1, "abcd"[ncol + i])

    # μ-colour legend (first panel, upper-left) + line/marker legend (top-right panel)
    mu_handles = [Line2D([0], [0], color=colors[m], lw=1.3, marker="o", mfc=colors[m], mec="k",
                         mew=0.3, ms=4, label=rf"$\mu={mu:g}$") for m, mu in enumerate(mus)]
    axes[0][0].legend(handles=mu_handles, loc="upper left", fontsize=5.6, handlelength=1.5,
                      labelspacing=0.25, title=r"$\mu$", title_fontsize=6.0)
    axes[0][-1].legend(handles=[Line2D([0], [0], color="0.3", lw=1.3, ls="-", label=r"MF $C_S{+}C_F$"),
                                Line2D([0], [0], color="0.3", lw=1.3, ls="--", label=r"MF $C_S$"),
                                Line2D([0], [0], color="0.3", ls="none", marker="o", mfc="0.3",
                                       mec="k", mew=0.3, ms=4, label="micro.")],
                       loc="upper right", fontsize=5.4, handlelength=1.7, labelspacing=0.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    fig.savefig(cfg["out"] + ".svg"); fig.savefig(cfg["out"] + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {cfg['out']}.svg / .png")


if __name__ == "__main__":
    main()
