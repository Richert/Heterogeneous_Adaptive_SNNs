r"""
Adaptive-coupling Kuramoto: weight-statistics bifurcation analysis (PRL_2026 Weight_Variance)
=============================================================================================

Closed-form analysis of the mean-field equations (manuscript Eqs. 8 & 10) for the average coupling
weight Ā and phase coherence R of a Kuramoto network with adaptive coupling
    θ̇_i = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i),   Ȧ_ij = μ cos(θ_j − θ_i) + γ(1 − A_ij),
with Lorentzian ω (centre 0, HWHM Δ).  Adiabatic elimination of the fast R (R slaves to Eq. 8)
reduces the joint steady state to a single quadratic in Ā:

    γ Ā² − (γ + μ) Ā + 2μΔ/K = 0,   Ā_± = [ (γ+μ) ± √D ] / (2γ),   D = (γ+μ)² − 8γμΔ/K,

with the synchronized state R² = 1 − 2Δ/(KĀ) (Eq. 33).  The weight variance follows from the
manuscript's current closure (Eqs. 32 & 37), using the squared order parameter S = ⟨|c|²⟩ from the
locked/drifting decomposition (b = KĀR):

    S = (2/π)arctan(b/Δ)(2 + 2Δ²/b²) − 4Δ/(πb) − R²   (Eq. 32),
    V_A = μ²(S² − R⁴) / (2γ²)                          (Eq. 37).

This matches the microscopic sims on the SYNCHRONIZED branch, unlike the earlier weak-coupling form
V_A = μ²(1 − R⁴)/[2γ(γ + 2Δ)] (old Eq. 42), which held only at low R.

FIGURE 1 — 1-D bifurcation diagrams R(Δ): synchronized (Ā_+, stable), saddle (Ā_−, unstable) and
           asynchronous (R=0, Ā=1) branches; solid = stable, dashed = unstable; saddle-node (fold)
           and transcritical points marked.  One panel per μ.
FIGURE 2 — V_A, Ā and Ā/V_A vs Δ from the closed forms (synchronized branch), verified against
           direct simulations of the N-oscillator adaptive network at equidistant Δ.  One panel/μ.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_analysis.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    K=1.0, gamma=0.001,
    mus=[0.1, 0.01, 0.001],          # μ values (one panel each); spans the μ≷γ transition
    # microscopic verification network
    N=500, T=1000.0, dt=0.05, trans_frac=0.6, n_sim=6, seed=1,
    out_bif="/home/rgast/data/mpmf_simulations/weight_variance_bifurcation",
    out_stats="/home/rgast/data/mpmf_simulations/weight_variance_statistics",
)

C_SYNC, C_ASYNC = "#1f77b4", "0.25"
C_ABAR, C_VAR, C_RATIO = "#1f77b4", "#c1121f", "#2a9d8f"


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


def _panel_label(ax, letter, dx=-22, dy=4):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


# ════════════════════════════════════════════════════════════════════════════
#  closed-form steady-state branches
# ════════════════════════════════════════════════════════════════════════════
def S_order(R2, A, delta, K):
    """Squared order parameter S = ⟨|c|²⟩ (manuscript Eq. 32) on the OA manifold, with
    locked/drifting decomposition and b = KĀR.  Rewritten to avoid the catastrophic
    cancellation of the two ∝1/b terms as R→0 (b→0): with x = b/Δ,
        (2/π)arctan(x)(2+2/x²) − 4/(πx) = (4/π)arctan(x) + (4/π)[arctan(x)/x² − 1/x],
    and the bracket → −x/3 smoothly as x→0."""
    b = K * A * np.sqrt(np.clip(R2, 0.0, None))
    x = b / delta
    xs = np.where(x > 1e-3, x, 1.0)                        # safe x for the direct branch
    bracket = np.where(x > 1e-3, np.arctan(xs) / xs ** 2 - 1.0 / xs, -x / 3.0 + x ** 3 / 5.0)
    return (4.0 / np.pi) * np.arctan(x) + (4.0 / np.pi) * bracket - R2


def branches(delta, mu, K, g):
    """Return dict of analytic steady-state branches over the Δ-array `delta`.
    Ā_± roots of γĀ²−(γ+μ)Ā+2μΔ/K; R²=1−2Δ/(KĀ); physical where R²≥0 (and D≥0).
    Weight variance from the manuscript's current closure (Eq. 37, via S of Eq. 32):
    V_A = μ²(S²−R⁴)/(2γ²)  — this matches the microscopic sims on the SYNCHRONIZED
    branch (the earlier weak-coupling Eq. 42 only matched at low R)."""
    D = (g + mu) ** 2 - 8 * g * mu * delta / K
    real = D >= 0
    sqrtD = np.sqrt(np.where(real, D, np.nan))
    out = {}
    for name, sign in (("sync", +1.0), ("saddle", -1.0)):
        A = ((g + mu) + sign * sqrtD) / (2 * g)
        J = K * A
        R2 = 1.0 - 2.0 * delta / J
        phys = real & (R2 >= 0) & (A > 0)
        R2p = np.where(phys, R2, np.nan)
        R = np.sqrt(R2p)
        S = S_order(R2p, np.where(phys, A, np.nan), delta, K)
        VA = mu ** 2 * (S ** 2 - R2p ** 2) / (2 * g ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(phys, A, np.nan) / VA
        out[name] = dict(A=np.where(phys, A, np.nan), R=R, VA=VA, ratio=ratio, phys=phys)
    # asynchronous branch: R=0, Ā=1 (stable for Δ>K/2) ⇒ S=0 and V_A=0 (Eq. 37)
    out["async"] = dict(A=np.ones_like(delta), R=np.zeros_like(delta),
                        VA=np.zeros_like(delta), ratio=np.full_like(delta, np.inf),
                        stable=delta > K / 2)
    return out


def special_points(mu, K, g):
    """Saddle-node (fold) and transcritical bifurcation points."""
    dD = K * (g + mu) ** 2 / (8 * g * mu)             # discriminant → 0
    pts = {"transcritical": (K / 2.0, 0.0)}           # nonzero branch meets R=0, Ā=1
    if mu > g:                                        # fold is physical only for μ>γ
        R_sn = np.sqrt((mu - g) / (2 * mu))
        pts["fold"] = (dD, R_sn)
    return pts


def sync_delta_end(mu, K, g):
    """Largest Δ at which the synchronized (upper) branch is physical."""
    return K * (g + mu) ** 2 / (8 * g * mu) if mu > g else K / 2.0


# ════════════════════════════════════════════════════════════════════════════
#  microscopic verification: full N×N adaptive Kuramoto network (Euler)
# ════════════════════════════════════════════════════════════════════════════
def simulate(delta, mu, cfg):
    """Integrate the adaptive network from a coherent IC; return steady (R, Ā, V_A).
    Statistics taken over OFF-diagonal weights (self-coupling is spurious in the MF limit)."""
    N, K, g, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["dt"]
    nsteps = int(cfg["T"] / dt)
    rec = int(cfg["trans_frac"] * nsteps)
    rng = np.random.default_rng(cfg["seed"])
    p = (np.arange(N) + 0.5) / N
    omega = delta * np.tan(np.pi * (p - 0.5))         # deterministic Lorentzian quantiles (ω̄=0)
    theta = rng.normal(0.0, 0.3, N)                   # coherent IC → synchronized branch
    A = np.ones((N, N))
    KinvN = K / N
    n_off = N * N - N
    Rs, Abars, VAs = [], [], []
    for k in range(nsteps):
        e = np.exp(1j * theta)
        field = A @ e
        theta = theta + dt * (omega + KinvN * np.imag(np.conj(e) * field))
        C = np.real(np.conj(e)[:, None] * e[None, :])  # cos(θ_j − θ_i)
        A = A + dt * (mu * C + g * (1.0 - A))
        if k >= rec:
            d = np.diagonal(A)
            s1 = A.sum() - d.sum()
            s2 = (A * A).sum() - (d * d).sum()
            Abar = s1 / n_off
            Rs.append(np.abs(e.mean())); Abars.append(Abar); VAs.append(s2 / n_off - Abar ** 2)
    return float(np.mean(Rs)), float(np.mean(Abars)), float(np.mean(VAs))


def run_sims(mu, cfg):
    """Simulate at equidistant Δ within the synchronized-branch range."""
    g, K = cfg["gamma"], cfg["K"]
    d_end = sync_delta_end(mu, K, g)
    deltas = np.linspace(0.05, 0.97 * d_end, cfg["n_sim"])   # include near-threshold (low-R) points
    res = []
    for d in deltas:
        R, A, V = simulate(d, mu, cfg)
        res.append((d, R, A, V))
        print(f"    μ={mu:<6} Δ={d:6.3f} -> R={R:.3f}  Ā={A:.3f}  V_A={V:.4f}  Ā/V_A={A/V:.2f}")
    return np.array(res)                               # columns: Δ, R, Ā, V_A


# ════════════════════════════════════════════════════════════════════════════
#  figure 1 — bifurcation diagrams
# ════════════════════════════════════════════════════════════════════════════
def make_bifurcation(cfg, sims):
    g, K, mus = cfg["gamma"], cfg["K"], cfg["mus"]
    fig, axes = plt.subplots(1, len(mus), figsize=(7.0, 2.2), layout="constrained")
    axes = np.atleast_1d(axes)
    for j, mu in enumerate(mus):
        ax = axes[j]
        d_end = sync_delta_end(mu, K, g)
        dmax = 1.35 * d_end
        dlt = np.linspace(1e-3, dmax, 1000)
        br = branches(dlt, mu, K, g)
        ax.plot(dlt, br["sync"]["R"], color=C_SYNC, ls="-", lw=1.1, zorder=4)      # stable sync
        ax.plot(dlt, br["saddle"]["R"], color=C_SYNC, ls="--", lw=1.0, zorder=3)   # unstable saddle
        st = br["async"]["stable"]
        ax.plot(dlt[~st], np.zeros_like(dlt[~st]), color=C_ASYNC, ls="--", lw=1.0)  # async unstable
        ax.plot(dlt[st], np.zeros_like(dlt[st]), color=C_ASYNC, ls="-", lw=1.0)     # async stable
        for name, (xd, yr) in special_points(mu, K, g).items():
            mk = "o" if name == "fold" else "s"
            ax.plot(xd, yr, mk, ms=4, mfc="#e63946", mec="k", mew=0.4, zorder=6, clip_on=False)
        if sims[mu].size:                                                          # sim R markers
            ax.plot(sims[mu][:, 0], sims[mu][:, 1], "o", ms=3.2, mfc="none",
                    mec="#e63946", mew=0.8, zorder=7, label="simulation")
        ax.set_xlim(0, dmax); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"phase coherence $R$", labelpad=2)
        ax.set_title(rf"$\mu={mu:g}$  ($\gamma={g:g}$)", fontsize=6.8, pad=2)
        _panel_label(ax, "abc"[j])
    handles = [Line2D([0], [0], color=C_SYNC, ls="-", lw=1.1, label="synchronized (stable)"),
               Line2D([0], [0], color=C_SYNC, ls="--", lw=1.0, label="saddle (unstable)"),
               Line2D([0], [0], color=C_ASYNC, ls="-", lw=1.0, label="async. (stable)"),
               Line2D([0], [0], color=C_ASYNC, ls="--", lw=1.0, label="async. (unstable)"),
               Line2D([0], [0], marker="o", ls="none", mfc="#e63946", mec="k", mew=0.4, ms=4, label="fold"),
               Line2D([0], [0], marker="s", ls="none", mfc="#e63946", mec="k", mew=0.4, ms=4, label="transcritical"),
               Line2D([0], [0], marker="o", ls="none", mfc="none", mec="#e63946", mew=0.8, ms=3.2, label="simulation")]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=5.0, handlelength=1.4,
                    labelspacing=0.25, borderaxespad=0.3)
    fig.savefig(cfg["out_bif"] + ".pdf"); fig.savefig(cfg["out_bif"] + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {cfg['out_bif']}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
#  figure 2 — weight statistics V_A, Ā, Ā/V_A vs Δ (analytic + simulation)
# ════════════════════════════════════════════════════════════════════════════
def make_statistics(cfg, sims):
    g, K, mus = cfg["gamma"], cfg["K"], cfg["mus"]
    fig, axes = plt.subplots(1, len(mus), figsize=(7.0, 2.3), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for j, mu in enumerate(mus):
        ax = axes[j]
        d_end = sync_delta_end(mu, K, g)
        dlt = np.linspace(0.02, 0.999 * d_end, 800)
        br = branches(dlt, mu, K, g)
        ax.plot(dlt, br["sync"]["A"], color=C_ABAR, lw=1.1, label=r"$\bar A$")
        ax.plot(dlt, br["sync"]["VA"], color=C_VAR, lw=1.1, label=r"$V_A$")
        ax.plot(dlt, br["sync"]["ratio"], color=C_RATIO, lw=1.1, label=r"$\bar A/V_A$")
        ax.axhline(1.0, color="0.6", lw=0.6, ls="--", zorder=0)
        if sims[mu].size:                                          # simulation markers
            sd, _, sA, sV = sims[mu].T
            ax.plot(sd, sA, "o", ms=3.0, mfc="none", mec=C_ABAR, mew=0.9)
            ax.plot(sd, sV, "s", ms=3.0, mfc="none", mec=C_VAR, mew=0.9)
            ax.plot(sd, sA / sV, "^", ms=3.0, mfc="none", mec=C_RATIO, mew=0.9)
        ax.set_yscale("log")
        ax.set_xlim(0, d_end)
        ax.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"$\bar A$,  $V_A$,  $\bar A/V_A$", labelpad=2)
        ax.set_title(rf"$\mu={mu:g}$  ($\gamma={g:g}$)", fontsize=6.8, pad=2)
        _panel_label(ax, "abc"[j])
        if j == len(mus) - 1:
            ax.legend(loc="best", fontsize=5.2, handlelength=1.4, labelspacing=0.25)
    fig.savefig(cfg["out_stats"] + ".pdf"); fig.savefig(cfg["out_stats"] + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {cfg['out_stats']}.pdf / .png")


def main(cfg=CONFIG):
    print(f"adaptive-coupling weight-variance analysis — K={cfg['K']}, γ={cfg['gamma']}, μ={cfg['mus']}")
    sims = {}
    for mu in cfg["mus"]:
        print(f"  simulating network for μ={mu} ...")
        sims[mu] = run_sims(mu, cfg)
    set_prl_style()
    make_bifurcation(cfg, sims)
    make_statistics(cfg, sims)


if __name__ == "__main__":
    main()
