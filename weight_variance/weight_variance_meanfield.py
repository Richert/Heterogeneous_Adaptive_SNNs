r"""
Adaptive-coupling Kuramoto: improved mean field (Eqs. 10, 71–76) vs. microscopic dynamics
=========================================================================================

Time-domain comparison of the coupling-weight statistics of a Kuramoto network with adaptive coupling
against the improved closed mean-field system from the PRL_2026 "Weight Variance" manuscript
(~/Downloads/weight_variance.pdf).

Microscopic model (Eqs. 1, 2, 5 with G_A(x)=cos x; Lorentzian ω, centre 0, HWHM Δ):
    θ̇_i = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i),   Ȧ_ij = μ cos(θ_j − θ_i) + γ(1 − A_ij).
Observables (off-diagonal pairs): Ā = ⟨A_ij⟩, V_A = ⟨A_ij²⟩ − Ā², C_A = ⟨A_ij G_ij⟩ − Ā Ḡ,
with G_ij = cos(θ_j − θ_i), Ḡ = ⟨G_ij⟩.

Mean field — the phase coherence R(t) is PULLED FROM THE MICRO SIMULATION (single MF version); it
then drives the weight-statistics closure:
    Ā̇   = μ R² + γ(1 − Ā)                                   (Eq. 10)
    Ċ_S = −γ C_S + μ σ_S²                                   (Eq. 71)   static  part
    Ċ_F = −(γ + 2Δ) C_F + μ σ_F²                            (Eq. 72)   fluctuating part
    V̇_A = 2μ(C_S + C_F) − 2γ V_A                            (Eq. 73)
    σ_S² = ½(S² − R⁴),   σ_F² = ½(1 − S²)                   (Eqs. 74, 75)
    S    = S_L + S_D,     b = K Ā R                          (see below)
    C_A  = C_S + C_F.

S — order parameter, FIXED.  The closed form printed as Eq. 76,
    S = (2/π) arctan(b/Δ)(2 + 2Δ²/b²) − 4Δ/(πb) − R²,
was obtained by substituting the *self-consistent* OA relation for R² to remove the drifting
integral; it is only valid when (R, b) lie on the OA manifold.  Here R is pulled from the micro
sim while b = K Ā R uses the MF Ā, so (R, b) is OFF the manifold: the −R² term no longer cancels
the locked contribution and S → 2 − R² > 1 as b → ∞, inflating σ_S² ∝ S² and hence C_S, C_A, V_A
by an Ā-dependent factor.  We instead evaluate S = ⟨|c|²⟩ directly from its definition,
    S_L = (2/π) arctan(b/Δ),                              (locked set, |c|² = 1)   Eq. 42
    S_D = ∫_{|ω|>b} q(ω)² ρ(ω) dω,  q = (ω − sgn(ω)√(ω²−b²))/b,                     Eq. 45
which guarantees 0 ≤ S_L ≤ S ≤ 1 by construction.  S depends only on x = b/Δ, so it is
tabulated once and interpolated (no runtime cost).

Single figure (single-column PRL): rows = Ā(t), V_A(t), C_A(t); micro vs. MF; the C_A row also shows
the two MF components C_S and C_F.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_meanfield.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp, quad

CONFIG = dict(
    K=3.0, gamma=0.001, mu=0.003, delta=1.0,     # single parameter set
    N=500, T=5000.0, dt=0.05, dts=1.0,            # micro: Euler step dt, record every dts
    sigma0=0.3, seed=1,                          # coherent IC θ_i(0) ~ N(0, sigma0)
    out="/home/rgast/data/mpmf_simulations/weight_variance_meanfield",
)

C_MICRO, C_MF, C_S, C_F = "0.2", "#c1121f", "#2a9d8f", "#1f77b4"


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
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


def _panel_label(ax, letter, dx=-26, dy=3):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


# ════════════════════════════════════════════════════════════════════════════
#  microscopic simulation (full N×N adaptive network, Euler) — records time series
# ════════════════════════════════════════════════════════════════════════════
def simulate_micro(cfg):
    N, K, g, mu, dt, delta = cfg["N"], cfg["K"], cfg["gamma"], cfg["mu"], cfg["dt"], cfg["delta"]
    nsteps = int(cfg["T"] / dt)
    rec_every = max(1, int(round(cfg["dts"] / dt)))
    rng = np.random.default_rng(cfg["seed"])
    p = (np.arange(N) + 0.5) / N
    omega = delta * np.tan(np.pi * (p - 0.5))            # deterministic Lorentzian quantiles (ω̄=0)
    theta = rng.normal(0.0, cfg["sigma0"], N)            # coherent IC
    A = np.ones((N, N))
    KinvN = K / N
    n_off = N * N - N
    tr, Rt, At, Vt, Ct = [], [], [], [], []
    for k in range(nsteps + 1):
        e = np.exp(1j * theta)
        G = np.real(np.conj(e)[:, None] * e[None, :])    # G_ij = cos(θ_j − θ_i)
        if k % rec_every == 0:
            dgA = np.diagonal(A)
            Abar = (A.sum() - dgA.sum()) / n_off
            VA = ((A * A).sum() - (dgA * dgA).sum()) / n_off - Abar ** 2
            Gbar = (G.sum() - N) / n_off                 # diag(G)=1
            AGbar = ((A * G).sum() - dgA.sum()) / n_off  # diag(A*G)=A_ii*1
            tr.append(k * dt); Rt.append(np.abs(e.mean()))
            At.append(Abar); Vt.append(VA); Ct.append(AGbar - Abar * Gbar)
        if k == nsteps:
            break
        theta = theta + dt * (omega + KinvN * np.imag(np.conj(e) * (A @ e)))
        A = A + dt * (mu * G + g * (1.0 - A))
    return dict(t=np.array(tr), R=np.array(Rt), Abar=np.array(At), VA=np.array(Vt), CA=np.array(Ct))


# ════════════════════════════════════════════════════════════════════════════
#  mean field (Eqs. 10, 71–75) + fixed order parameter S, driven by R(t) from micro
# ════════════════════════════════════════════════════════════════════════════
def _SD_of_x(x):
    """Drifting contribution S_D as a function of x = b/Δ (Eq. 45)."""
    if x <= 0.0:
        return 0.0
    a = 1.0 / x                                          # α = Δ/b
    f = lambda u: (2.0 / np.pi) * (u - np.sqrt(u * u - 1.0)) ** 2 * a / (u * u + a * a)
    val, _ = quad(f, 1.0, np.inf, limit=200)
    return val


# one-time table of S(x), x = b/Δ  (S depends only on this ratio)
_S_X = np.concatenate(([0.0], np.logspace(-3, 4, 500)))
_S_TAB = np.clip((2.0 / np.pi) * np.arctan(_S_X)
                 + np.array([_SD_of_x(x) for x in _S_X]), 0.0, 1.0)


def order_parameter_S(R, A, K, delta):
    """Locked+drifting squared order parameter S = ⟨|c|²⟩ = S_L + S_D, via direct
    integration (Eqs. 42, 45). Bounded 0 ≤ S ≤ 1 by construction — unlike the printed
    Eq. 76, which is only valid on the self-consistent OA manifold and diverges above 1
    when R is supplied off-manifold (see module docstring)."""
    b = np.maximum(K * A * R, 1e-12)                     # locking bandwidth b = K Ā R
    return np.interp(b / delta, _S_X, _S_TAB)


def simulate_meanfield(cfg, t_micro, R_micro):
    K, g, mu, delta = cfg["K"], cfg["gamma"], cfg["mu"], cfg["delta"]

    def rhs(t, y):
        A, CS, CF, V = y
        R = np.interp(t, t_micro, R_micro)
        S = order_parameter_S(R, A, K, delta)
        sS2 = 0.5 * (S ** 2 - R ** 4)                    # Eq. 74
        sF2 = 0.5 * (1.0 - S ** 2)                       # Eq. 75
        dA = mu * R ** 2 + g * (1.0 - A)                 # Eq. 10
        dCS = -g * CS + mu * sS2                         # Eq. 71
        dCF = -(g + 2.0 * delta) * CF + mu * sF2         # Eq. 72
        dV = 2.0 * mu * (CS + CF) - 2.0 * g * V          # Eq. 73
        return [dA, dCS, dCF, dV]

    sol = solve_ivp(rhs, (t_micro[0], t_micro[-1]), [1.0, 0.0, 0.0, 0.0], t_eval=t_micro,
                    method="RK45", rtol=1e-7, atol=1e-9, max_step=cfg["dt"])
    A, CS, CF, V = sol.y
    return dict(t=t_micro, Abar=A, VA=V, CS=CS, CF=CF, CA=CS + CF, R=R_micro)


# ════════════════════════════════════════════════════════════════════════════
#  figure — single column, rows = Ā, V_A, C_A
# ════════════════════════════════════════════════════════════════════════════
def make_figure(cfg, mic, mf):
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.2), sharex=True, layout="constrained")
    t = mic["t"]

    # (a) Ā
    ax = axes[0]
    ax.plot(t, mic["Abar"], color=C_MICRO, lw=1.0)
    ax.plot(mf["t"], mf["Abar"], color=C_MF, lw=1.0, ls="--")
    ax.set_ylabel(r"$\bar A(t)$", labelpad=2)
    ax.legend(handles=[Line2D([0], [0], color=C_MICRO, lw=1.0, label="microscopic"),
                       Line2D([0], [0], color=C_MF, lw=1.0, ls="--", label="mean field")],
              loc="best", fontsize=5.8, handlelength=1.6)
    _panel_label(ax, "a")

    # (b) V_A
    ax = axes[1]
    ax.plot(t, mic["VA"], color=C_MICRO, lw=1.0)
    ax.plot(mf["t"], mf["VA"], color=C_MF, lw=1.0, ls="--")
    ax.set_ylabel(r"$V_A(t)$", labelpad=2)
    _panel_label(ax, "b")

    # (c) C_A with components C_S, C_F
    ax = axes[2]
    ax.plot(t, mic["CA"], color=C_MICRO, lw=1.0)
    ax.plot(mf["t"], mf["CA"], color=C_MF, lw=1.0, ls="--")
    ax.plot(mf["t"], mf["CS"], color=C_S, lw=0.9, ls=":")
    ax.plot(mf["t"], mf["CF"], color=C_F, lw=0.9, ls="-.")
    ax.set_ylabel(r"$C_A(t)$", labelpad=2)
    ax.set_xlabel(r"time $t$", labelpad=1)
    ax.legend(handles=[Line2D([0], [0], color=C_MF, lw=1.0, ls="--", label=r"MF $C_A{=}C_S{+}C_F$"),
                       Line2D([0], [0], color=C_S, lw=0.9, ls=":", label=r"$C_S$"),
                       Line2D([0], [0], color=C_F, lw=0.9, ls="-.", label=r"$C_F$")],
              loc="best", fontsize=5.8, handlelength=1.8)
    _panel_label(ax, "c")

    for ax in axes:
        ax.set_xlim(t[0], t[-1])
    fig.suptitle(rf"mean field (Eqs. 10, 71–75; fixed $S$) vs. microscopic  "
                 rf"($K{{=}}{cfg['K']:g}$, $\mu{{=}}{cfg['mu']:g}$, $\gamma{{=}}{cfg['gamma']:g}$, "
                 rf"$\Delta{{=}}{cfg['delta']:g}$)", fontsize=6.8)
    fig.savefig(cfg["out"] + ".pdf"); fig.savefig(cfg["out"] + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {cfg['out']}.pdf / .png")


def main(cfg=CONFIG):
    print(f"MF (Eqs.10,71-76) vs micro — K={cfg['K']}, μ={cfg['mu']}, γ={cfg['gamma']}, Δ={cfg['delta']}")
    mic = simulate_micro(cfg)
    mf = simulate_meanfield(cfg, mic["t"], mic["R"])
    print(f"  steady [micro/MF]: Ā {mic['Abar'][-1]:.3f}/{mf['Abar'][-1]:.3f}  "
          f"V_A {mic['VA'][-1]:.4f}/{mf['VA'][-1]:.4f}  "
          f"C_A {mic['CA'][-1]:.4f}/{mf['CA'][-1]:.4f}  (C_S={mf['CS'][-1]:.4f}, C_F={mf['CF'][-1]:.4f})")
    set_prl_style()
    make_figure(cfg, mic, mf)


if __name__ == "__main__":
    main()