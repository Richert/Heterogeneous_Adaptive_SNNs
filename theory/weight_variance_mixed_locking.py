r"""
Adaptive-coupling Kuramoto: weight-variance closures with a STATIC (DC) drift contribution
==========================================================================================

Single Delta below the transition.  The microscopic network is simulated ONCE.  Three
mean-field models are compared, each run self-consistently and with R(t) taken from micro:

  manuscript : single drift channel, C' = -(g+2D) C + mu/2 (1-R^4)              [Eq. 36]
  twopop     : locked + corrected-drift blend (NO drift-static term)
                 C_L' = -g C_L + mu sigma_L^2
                 C_D' = -(g+lambda_D) C_D + mu sigma_D^2,   sigma_D^2 = 1/2 (1+rho2D^2)
                 C_A  = q_sc^2 C_L + (1-q_sc^2) C_D
  static     : static + fluctuating decomposition (THIS turn's fix)
                 C_stat' = -g            C_stat + mu sigma_stat^2     (DC, rate g)
                 C_fluc' = -(g+lambda_D) C_fluc + mu sigma_fluc^2
                 C_A     = C_stat + C_fluc

Static decomposition (c = p + i q the per-oscillator first Daido moment;
locked: p=cos phi*, q=sin phi*;  drift: p=0, q=a(w)=sgn(w)(|w|-sqrt(w^2-b^2))/b):
    Phi(inf) = Var_pairs(p_i p_j + q_i q_j) = P2^2 + Q2^2 - R^4   (DC plateau, exact)
    Phi(0)   = 1/2 (1 + (P2-Q2)^2) - R^4   (=> sigma_fluc^2 = 1/2 (1-(P2+Q2)^2))
with P2 = <p^2>, Q2 = <q^2> over the whole population.  Both reduce to the manuscript
(sigma_stat^2 -> 0, sigma_fluc^2 -> 1/2, lambda_D -> 2 Delta) as b = K Abar R -> 0.

    python weight_variance_q_sweep.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

CONFIG = dict(
    K=1.0, gamma=0.01, mu=0.02,
    delta=0.7,                          # single Delta (below transition)
    N=500, T=1000.0, dt=0.05, dts=1.0,
    sigma0=0.3, seed=1,
    out="weight_variance_static",
)

C_MICRO = "0.12"
C_MF    = "#1f6fc1"
VARS = [("R", r"$R$"), ("Abar", r"$\bar A$"), ("VA", r"$V_A$"), ("CA", r"$C_A$")]
MODELS = [
    ("manuscript", "manuscript\n" + r"$\frac{\mu}{2}(1-R^4)$, rate $2\Delta$"),
    ("twopop",     "two-population\n(locked + drift, no static)"),
    ("static",     "two-pop + static\n(static + fluctuating)"),
]


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


def _panel_label(ax, letter, dx=-24, dy=4):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


# ════════════════════════════════════════════════════════════════════════════
#  precomputed b-tables for all closure source terms / rates
# ════════════════════════════════════════════════════════════════════════════
def build_tables(delta, gamma, K, b_max, nb=80, npts=1500, n_tau=2000, span=120.0):
    """Tabulate, vs b = K Abar R: q_sc, sigma_L^2, sigma_D^2, sigma_stat^2, sigma_fluc^2,
    lambda_D, R_sc.  Locked integrals via omega=b sin(psi); drift tail via dense u^2 grid."""
    bs = np.linspace(0.0, b_max, nb)
    out = {k: np.zeros(nb) for k in
           ("q_sc", "sL2", "sD2", "sStat2", "sFluc2", "lamD", "R_sc")}
    out["b"] = bs
    out["sD2"][:] = 0.5; out["sFluc2"][:] = 0.5; out["lamD"][:] = 2.0 * delta
    taus = np.linspace(0.0, 40.0 / delta, n_tau)
    for i, b in enumerate(bs):
        if b <= 1e-9:
            continue
        # --- locked |w|<b : w = b sin(psi) ---
        psi = np.linspace(0.0, np.pi / 2, npts)
        sp, cp = np.sin(psi), np.cos(psi)
        rl = (delta / np.pi) / ((b * sp) ** 2 + delta ** 2)
        Lp2 = 2.0 * _trapz(cp ** 2 * rl * b * cp, psi)   # <(1-(w/b)^2)> mass
        Lq2 = 2.0 * _trapz(sp ** 2 * rl * b * cp, psi)
        Lp  = 2.0 * _trapz(cp * rl * b * cp, psi)
        qlf = 2.0 * _trapz(rl * b * cp, psi)
        # --- drift |w|>b ---
        u = np.linspace(0.0, 1.0, npts)
        om = b + (span * delta) * u ** 2
        rd = (delta / np.pi) / (om ** 2 + delta ** 2)
        w = np.empty(npts)
        w[1:-1] = 0.5 * (om[2:] - om[:-2]); w[0] = 0.5 * (om[1] - om[0]); w[-1] = 0.5 * (om[-1] - om[-2])
        rdw = rd * w
        root = np.sqrt(np.clip(om ** 2 - b ** 2, 0.0, None))   # = Omega (w>0)
        m = (om - root) / b
        Dq2 = 2.0 * np.sum(m ** 2 * rdw)
        dmass = 2.0 * np.sum(rdw)
        # --- assemble static / fluctuating amplitudes (whole-population p,q) ---
        R_sc = Lp
        P2 = Lp2
        Q2 = Lq2 + Dq2
        sStat = max(P2 ** 2 + Q2 ** 2 - R_sc ** 4, 0.0)
        sFluc = max(0.5 * (1.0 - (P2 + Q2) ** 2), 0.0)
        rho2D = -Dq2 / dmass if dmass > 1e-12 else 0.0
        sD2 = 0.5 * (1.0 + rho2D ** 2)
        if qlf > 1e-9:
            s2 = Lq2 / qlf; cc = Lp2 / qlf; k1 = Lp / qlf
            sL2 = max(cc ** 2 + s2 ** 2 - k1 ** 4, 0.0)
        else:
            sL2 = 0.0
        # --- lambda_D : half-life of chi1(tau)=<cos(Omega tau)>_drift ---
        chi = (np.cos(np.outer(taus, root)) * rdw).sum(axis=1) / np.sum(rdw)
        below = np.where(chi <= 0.5)[0]
        if below.size == 0:
            thalf = taus[-1]
        elif below[0] == 0:
            thalf = taus[1]
        else:
            k = below[0]; t0, t1, c0, c1 = taus[k - 1], taus[k], chi[k - 1], chi[k]
            thalf = t0 + (0.5 - c0) * (t1 - t0) / (c1 - c0)
        out["q_sc"][i] = qlf; out["R_sc"][i] = R_sc; out["sStat2"][i] = sStat
        out["sFluc2"][i] = sFluc; out["sD2"][i] = sD2; out["sL2"][i] = sL2
        out["lamD"][i] = 2.0 * np.log(2.0) / max(thalf, 1e-9)
    return out


def _ip(tables, key, b):
    return float(np.interp(b, tables["b"], tables[key]))


# ════════════════════════════════════════════════════════════════════════════
#  microscopic simulation
# ════════════════════════════════════════════════════════════════════════════
def simulate_micro(delta, cfg):
    N, K, g, mu, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["mu"], cfg["dt"]
    nsteps = int(cfg["T"] / dt)
    rec_every = max(1, int(round(cfg["dts"] / dt)))
    rng = np.random.default_rng(cfg["seed"])
    p = (np.arange(N) + 0.5) / N
    omega = delta * np.tan(np.pi * (p - 0.5))
    theta = rng.normal(0.0, cfg["sigma0"], N)
    A = np.ones((N, N))
    KinvN = K / N
    n_off = N * N - N
    tr, Rt, At, Vt, Ct = [], [], [], [], []
    for k in range(nsteps + 1):
        e = np.exp(1j * theta)
        G = np.real(np.conj(e)[:, None] * e[None, :])
        if k % rec_every == 0:
            dgA = np.diagonal(A)
            Abar = (A.sum() - dgA.sum()) / n_off
            VA = ((A * A).sum() - (dgA * dgA).sum()) / n_off - Abar ** 2
            Gbar = (G.sum() - N) / n_off
            AGbar = ((A * G).sum() - dgA.sum()) / n_off
            tr.append(k * dt); Rt.append(np.abs(e.mean()))
            At.append(Abar); Vt.append(VA); Ct.append(AGbar - Abar * Gbar)
        if k == nsteps:
            break
        theta = theta + dt * (omega + KinvN * np.imag(np.conj(e) * (A @ e)))
        A = A + dt * (mu * G + g * (1.0 - A))
    return dict(t=np.array(tr), R=np.array(Rt), Abar=np.array(At),
                VA=np.array(Vt), CA=np.array(Ct), R0=Rt[0])


# ════════════════════════════════════════════════════════════════════════════
#  mean-field: model in {manuscript, twopop, static}, mode in {dynamic, micro_R}
# ════════════════════════════════════════════════════════════════════════════
NSUB = {"manuscript": 1, "twopop": 2, "static": 2}


def simulate_meanfield(delta, model, cfg, t_eval, mode, tables,
                       R0=None, t_micro=None, R_micro=None):
    K, g, mu = cfg["K"], cfg["gamma"], cfg["mu"]

    def cov_module(R, A, sub):
        b = K * A * max(R, 0.0)
        if model == "manuscript":
            (C,) = sub
            return C, [-(g + 2.0 * delta) * C + mu * 0.5 * (1.0 - max(R, 0.0) ** 4)]
        if model == "twopop":
            CL, CD = sub
            qsc = _ip(tables, "q_sc", b); sL2 = _ip(tables, "sL2", b)
            sD2 = _ip(tables, "sD2", b); lam = _ip(tables, "lamD", b)
            CA = qsc ** 2 * CL + (1.0 - qsc ** 2) * CD
            return CA, [-g * CL + mu * sL2, -(g + lam) * CD + mu * sD2]
        # static
        Cs, Cf = sub
        sS = _ip(tables, "sStat2", b); sF = _ip(tables, "sFluc2", b); lam = _ip(tables, "lamD", b)
        return Cs + Cf, [-g * Cs + mu * sS, -(g + lam) * Cf + mu * sF]

    nsub = NSUB[model]
    if mode == "dynamic":
        def rhs(t, y):
            R, A = y[0], y[1]; sub = y[2:2 + nsub]; V = y[2 + nsub]
            CA, dsub = cov_module(R, A, sub)
            return [-delta * R + (K * A / 2.0) * R * (1.0 - R ** 2),
                    mu * R ** 2 + g * (1.0 - A), *dsub, 2.0 * mu * CA - 2.0 * g * V]
        y0 = [R0, 1.0] + [0.0] * nsub + [0.0]
        sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=cfg["dt"])
        R, A = sol.y[0], sol.y[1]; subarr = sol.y[2:2 + nsub]
        V = sol.y[2 + nsub]
    else:  # micro_R
        def rhs(t, y):
            A = y[0]; sub = y[1:1 + nsub]; V = y[1 + nsub]
            R = np.interp(t, t_micro, R_micro)
            CA, dsub = cov_module(R, A, sub)
            return [mu * R ** 2 + g * (1.0 - A), *dsub, 2.0 * mu * CA - 2.0 * g * V]
        y0 = [1.0] + [0.0] * nsub + [0.0]
        sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=cfg["dt"])
        A = sol.y[0]; subarr = sol.y[1:1 + nsub]; V = sol.y[1 + nsub]
        R = np.interp(t_eval, t_micro, R_micro)

    CA = np.array([cov_module(R[k], A[k], [s[k] for s in subarr])[0] for k in range(len(R))])
    return dict(t=t_eval, R=R, Abar=A, VA=V, CA=CA)


# ════════════════════════════════════════════════════════════════════════════
#  figure: rows = (R, Abar, V_A, C_A), columns = models;  micro + 2 MF curves
# ════════════════════════════════════════════════════════════════════════════
def make_figure(cfg, mic, results):
    delta = cfg["delta"]
    nrow, ncol = len(VARS), len(results)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 1.3 * nrow + 0.5),
                             sharex=True, squeeze=False, layout="constrained")
    letters = "abcdefghijklmnopqrstuvwx"
    for cj, (title, dyn, mR) in enumerate(results):
        for ri, (key, lab) in enumerate(VARS):
            ax = axes[ri][cj]
            ax.plot(mic["t"], mic[key], color=C_MICRO, lw=1.1)
            ax.plot(dyn["t"], dyn[key], color=C_MF, lw=1.0, ls="--")
            ax.plot(mR["t"], mR[key], color=C_MF, lw=1.0, ls=":")
            ax.set_xlim(mic["t"][0], mic["t"][-1])
            if ri == 0:
                ax.set_title(title, fontsize=6.4, pad=3)
            if cj == 0:
                ax.set_ylabel(lab, labelpad=2)
            if ri == nrow - 1:
                ax.set_xlabel(r"time $t$", labelpad=1)
            _panel_label(ax, letters[(ri * ncol + cj) % 24])
    handles = [Line2D([0], [0], color=C_MICRO, lw=1.1, label="microscopic"),
               Line2D([0], [0], color=C_MF, lw=1.0, ls="--", label=r"mean field, self-consistent $R$"),
               Line2D([0], [0], color=C_MF, lw=1.0, ls=":", label=r"mean field, $R$ from micro")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2,
               handlelength=2.0, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(rf"weight-variance closures with a static drift term, $\Delta={delta:g}$  "
                 rf"($K={cfg['K']:g}$, $\mu={cfg['mu']:g}$, $\gamma={cfg['gamma']:g}$)", fontsize=7.4)
    fig.savefig(cfg["out"] + ".pdf", bbox_inches="tight")
    fig.savefig(cfg["out"] + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {cfg['out']}.pdf / .png")


def main(cfg=CONFIG):
    cfg = dict(cfg)
    delta = cfg["delta"]
    print(f"static-drift closures — Delta={delta}, K={cfg['K']}, mu={cfg['mu']}, gamma={cfg['gamma']}")
    mic = simulate_micro(delta, cfg)
    b_max = max(2.0, 1.3 * cfg["K"] * float(np.max(mic["Abar"] * mic["R"])))
    tables = build_tables(delta, cfg["gamma"], cfg["K"], b_max)
    b_ss = cfg["K"] * mic["Abar"][-1] * mic["R"][-1]
    print(f"  micro steady: R={mic['R'][-1]:.3f} Abar={mic['Abar'][-1]:.3f} "
          f"V_A={mic['VA'][-1]:.4f} C_A={mic['CA'][-1]:.4f}  b={b_ss:.3f}")
    print(f"  @b={b_ss:.3f}: sigma_stat^2={_ip(tables,'sStat2',b_ss):.4f}  "
          f"sigma_fluc^2={_ip(tables,'sFluc2',b_ss):.4f}  lambda_D={_ip(tables,'lamD',b_ss):.3f}  "
          f"q_sc={_ip(tables,'q_sc',b_ss):.3f}")
    results = []
    for model, title in MODELS:
        dyn = simulate_meanfield(delta, model, cfg, mic["t"], "dynamic", tables, R0=mic["R0"])
        mR = simulate_meanfield(delta, model, cfg, mic["t"], "micro_R", tables,
                                t_micro=mic["t"], R_micro=mic["R"])
        results.append((title, dyn, mR))
        print(f"    {model:11s}:  C_A dyn={dyn['CA'][-1]:.4f} microR={mR['CA'][-1]:.4f}   "
              f"V_A dyn={dyn['VA'][-1]:.4f} microR={mR['VA'][-1]:.4f}   (micro C_A={mic['CA'][-1]:.4f})")
    set_prl_style()
    make_figure(cfg, mic, results)


if __name__ == "__main__":
    main()