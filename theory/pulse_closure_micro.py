r"""
Pulse-driven weight adaptation: microscopic theta-neuron sweep + moment hierarchy
=================================================================================

Model — theta neurons with pulse coupling and pulse-driven, bidirectional plasticity
of the Duchet et al. (Neural Computation 2023) / Fennelly et al. (Chaos 2025) type:

    dθ_k/dt  = (1 − cos θ_k) + (1 + cos θ_k)[ η_k + (J/N) Σ_l A_kl s(n_s, θ_l) ]
    dA_ij/dt = μ_p G_p(θ_i,θ_j) (A_m − A_ij) − μ_d G_d(θ_i,θ_j) A_ij

with separable, non-negative pulse kernels (see ``pulse_kernels.py``); the default
``duchet`` choice is coincidence potentiation with purely pre-synaptic depression,

    G_p = s(n_p, θ_i) s(n_p, θ_j),      G_d = s(n_d, θ_j).

Unlike the logistic rule of ``logistic_closure_micro.py`` this rule is LINEAR in
A_ij: writing it as dA/dt = F_ij − Γ_ij A_ij with

    F_ij = μ_p G_p A_m,     Γ_ij = μ_p G_p + μ_d G_d ≥ 0,

every pair relaxes to its own target A*_ij = F_ij/Γ_ij = A_m/(1+r_ij), r = μ_dG_d/(μ_pG_p),
at its own rate Γ_ij.  Γ̄ therefore appears as an explicit damping in every moment
equation, which is what makes a second-order truncation viable here.

What is recorded
----------------
Everything in the exact hierarchy (K_x = Cov(G_x,A), Γ̄ = μ_pḠ_p + μ_dḠ_d, a = A−Ā):

    Ā'   = μ_pḠ_p(A_m−Ā) − μ_dḠ_dĀ − (μ_pK_p + μ_dK_d)
    V_A' = −2Γ̄V_A + 2[μ_pK_p(A_m−Ā) − μ_dK_dĀ] − 2[μ_p⟨δG_p a²⟩ + μ_d⟨δG_d a²⟩]
    K_p' = Cov(Ġ_p,A) − Γ̄K_p + μ_p(A_m−Ā)Var(G_p) − μ_dĀCov(G_p,G_d) − (3rd order)
    K_d' = Cov(Ġ_d,A) − Γ̄K_d + μ_p(A_m−Ā)Cov(G_p,G_d) − μ_dĀVar(G_d) − (3rd order)

plus the Daido order parameters z_q (so the drive-moment reduction can be checked
against the measured drive moments), and snapshots of the per-pair LATENT RATIO
r̄_ij = μ_d⟨G_d⟩_t/(μ_p⟨G_p⟩_t) together with A_ij (the adiabatic prediction is
A_ij = A_m/(1+r̄_ij)).

All pair averages are computed as bilinear forms u·W·v, so the cost is a handful of
matrix-vector products rather than an N² python loop — no numba needed.

Usage (``pycobi`` env is enough; numpy/scipy only)
-------------------------------------------------
    python pulse_closure_micro.py                    # all regimes
    python pulse_closure_micro.py partial
    python pulse_closure_micro.py partial --kernel symmetric
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import numpy as np
import pulse_kernels as PK

# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    N=400,
    Am=1.0,
    A0=0.5,                      # uniform initial weight -> V_A(0)=K(0)=0
    n_s=2,                       # pulse order of the SYNAPTIC coupling
    kernel="duchet",             # plasticity kernels; see pulse_kernels.KERNELS
    dt=0.005,
    T=600.0,
    rec_every=100,
    eta_cutoff=20.0,
    tau_static=None,            # EMA timescale isolating the SLOW part of each kernel.
                                # None => adaptive tau_w = 1/Gammabar, i.e. the weight's
                                # OWN low-pass cutoff (a number overrides it).  This is
                                # the principled choice: A_ij is a linear filter of the
                                # drive with cutoff Gamma_ij, so "slow" must be defined
                                # relative to 1/Gammabar, which varies by ~7x across the
                                # regimes here.
    tau_gamma=50.0,             # smoothing of the running Gammabar estimate itself
    n_snap=3,
    snap_window=40.0,            # trailing window for the latent ratio r̄_ij
    n_sub=150,
    seed=0,
    out_dir=dp.KMO_ADAPTIVE,
)

#: (eta0, Delta, J, mu_p, mu_d) per regime.  Chosen from a (eta0, Delta, J) scan to
#: span the coherence range AND the drive-modulation ratio std(Γ)/Γ̄ that controls
#: whether the second-order truncation is expected to hold.
REGIMES = dict(
    #  <R>~0.10, std(Γ)/Γ̄~1.2 — asynchronous firing
    async_=dict(eta0=1.0, Delta=0.05, J=0.5, mu_p=0.02, mu_d=0.01),
    #  <R>~0.38, largest weight variance (s~0.10), std(Γ)/Γ̄~1.25
    partial=dict(eta0=0.0, Delta=1.0, J=0.5, mu_p=0.02, mu_d=0.01),
    #  <R>~0.97, weak modulation std(Γ)/Γ̄~0.44 — the friendly limit
    sync=dict(eta0=-1.0, Delta=0.05, J=0.5, mu_p=0.02, mu_d=0.01),
    #  std(Γ)/Γ̄~2.5 — deliberately outside the truncation's expected validity
    stress=dict(eta0=0.0, Delta=0.05, J=0.5, mu_p=0.02, mu_d=0.01),
)


# ════════════════════════════════════════════════════════════════════════════
#  pair averages as bilinear forms
# ════════════════════════════════════════════════════════════════════════════
def _bil(u, v, W, npair):
    """⟨u_i v_j W_ij⟩ over i≠j."""
    return (u @ (W @ v) - np.sum(u * v * np.diag(W))) / npair


def _bil0(u, v, npair):
    """⟨u_i v_j⟩ over i≠j (W ≡ 1)."""
    return (u.sum() * v.sum() - np.sum(u * v)) / npair


def _static_moments(Ep, Ed, A, npair):
    """Pair moments of the SLOW (time-averaged) kernels gbar_p, gbar_d.

    The weight is a low-pass filter of the drive with cutoff Gamma ~ mu << 1, so it
    couples to the slow part of each kernel rather than the instantaneous one.  This
    is the weight-level analogue of the C_S/C_F (static/fluctuating) split of the
    adaptive-coupling-statistics manuscript, and it is what removes Cov(Gdot, A) from
    the K equations: the slow kernels are frozen on the weight timescale, so their
    covariance with A obeys an autonomous equation.
    """
    dg = np.diag_indices_from(A)
    def m(X):
        return (X.sum() - X[dg].sum()) / npair
    gp, gd, Ab = m(Ep), m(Ed), m(A)
    return dict(sGp=gp, sGd=gd,
                sVarGp=m(Ep * Ep) - gp ** 2,
                sVarGd=m(Ed * Ed) - gd ** 2,
                sCovPD=m(Ep * Ed) - gp * gd,
                sKp=m(Ep * A) - gp * Ab,
                sKd=m(Ed * A) - gd * Ab)


def _moments(theta, A, A2, kernel, npair, qmax):
    """All pair moments of the exact hierarchy, plus the Daido parameters."""
    (ap, bp), (ad, bd) = kernel["p"], kernel["d"]
    p, P = PK.s_pulse(ap, theta), PK.s_pulse(bp, theta)     # G_p = p_i P_j
    q, Q = PK.s_pulse(ad, theta), PK.s_pulse(bd, theta)     # G_d = q_i Q_j

    one = np.ones_like(theta)
    Ab = _bil(one, one, A, npair)
    A2b = _bil(one, one, A2, npair)
    VA = A2b - Ab ** 2

    Gp, Gd = _bil0(p, P, npair), _bil0(q, Q, npair)
    GpA, GdA = _bil(p, P, A, npair), _bil(q, Q, A, npair)
    Kp, Kd = GpA - Gp * Ab, GdA - Gd * Ab

    Gpp = _bil0(p * p, P * P, npair)
    Gdd = _bil0(q * q, Q * Q, npair)
    Gpd = _bil0(p * q, P * Q, npair)
    VarGp, VarGd, CovPD = Gpp - Gp ** 2, Gdd - Gd ** 2, Gpd - Gp * Gd

    # third-order terms (measured so the truncation can be quantified)
    GpA2, GdA2 = _bil(p, P, A2, npair), _bil(q, Q, A2, npair)
    Wp = GpA2 - 2 * Ab * GpA + Ab ** 2 * Gp - Gp * VA
    Wd = GdA2 - 2 * Ab * GdA + Ab ** 2 * Gd - Gd * VA
    GppA = _bil(p * p, P * P, A, npair)
    GddA = _bil(q * q, Q * Q, A, npair)
    GpdA = _bil(p * q, P * Q, A, npair)
    Tpp = GppA - Ab * Gpp - 2 * Gp * GpA + 2 * Ab * Gp ** 2
    Tdd = GddA - Ab * Gdd - 2 * Gd * GdA + 2 * Ab * Gd ** 2
    Tpd = GpdA - Ab * Gpd - Gp * GdA - Gd * GpA + 2 * Ab * Gp * Gd

    z = PK.daido_from_phases(theta, qmax)
    return dict(Abar=Ab, VA=VA, Kp=Kp, Kd=Kd, Gp=Gp, Gd=Gd,
                VarGp=VarGp, VarGd=VarGd, CovPD=CovPD,
                Wp=Wp, Wd=Wd, Tpp=Tpp, Tpd=Tpd, Tdd=Tdd), z


# ════════════════════════════════════════════════════════════════════════════
#  integrator
# ════════════════════════════════════════════════════════════════════════════
def simulate(eta0, Delta, J, mu_p, mu_d, cfg=CONFIG, progress=False):
    N, Am, dt, n_s = cfg["N"], cfg["Am"], cfg["dt"], cfg["n_s"]
    kernel = PK.KERNELS[cfg["kernel"]]
    (ap, bp), (ad, bd) = kernel["p"], kernel["d"]
    qmax = PK.max_harmonic(kernel)
    npair = N * (N - 1)

    rng = np.random.default_rng(cfg["seed"])
    qq = (np.arange(1, N + 1) - 0.5) / N                    # Lorentzian quantiles
    eta = np.clip(eta0 + Delta * np.tan(np.pi * (qq - 0.5)), -cfg["eta_cutoff"],
                  cfg["eta_cutoff"])
    rng.shuffle(eta)
    theta = 2 * np.pi * rng.random(N)
    A = np.full((N, N), cfg["A0"] * Am)

    nsteps = int(cfg["T"] / dt)
    snap_steps = np.linspace(nsteps // 2, nsteps, cfg["n_snap"]).astype(int)
    snap_win = int(cfg["snap_window"] / dt)
    ns = cfg["n_sub"]
    acc_p, acc_d, nacc = np.zeros((ns, ns)), np.zeros((ns, ns)), 0
    r_snap, A_snap = np.zeros((cfg["n_snap"], ns, ns)), np.zeros((cfg["n_snap"], ns, ns))
    si = 0

    p0, P0 = PK.s_pulse(ap, theta), PK.s_pulse(bp, theta)
    q0, Q0 = PK.s_pulse(ad, theta), PK.s_pulse(bd, theta)
    Ep, Ed = np.outer(p0, P0), np.outer(q0, Q0)

    def _pairmean(u, v):
        """<u_i v_j> over i != j, in O(N) using separability."""
        return (u.sum() * v.sum() - np.sum(u * v)) / npair

    # running estimate of Gammabar = mu_p<G_p> + mu_d<G_d>; sets the EMA timescale
    Gam_ema = mu_p * _pairmean(p0, P0) + mu_d * _pairmean(q0, Q0)
    tau_gam = cfg["tau_gamma"]
    tau_fixed = cfg["tau_static"]
    tau_w = tau_fixed if tau_fixed else 1.0 / max(Gam_ema, 1e-9)
    tau_hist = []

    rec, zrec, trec = [], [], []

    def dtheta(th, Amat):
        sv = PK.s_pulse(n_s, th)
        return (1 - np.cos(th)) + (1 + np.cos(th)) * (eta + (J / N) * (Amat @ sv))

    for n in range(nsteps + 1):
        if n % cfg["rec_every"] == 0:
            m, z = _moments(theta, A, A * A, kernel, npair, qmax)
            m.update(_static_moments(Ep, Ed, A, npair))
            rec.append(m); zrec.append(z); trec.append(n * dt); tau_hist.append(tau_w)

        # ---- latent-ratio accumulation inside the trailing window
        if si < len(snap_steps) and n > snap_steps[si] - snap_win:
            p_, P_ = PK.s_pulse(ap, theta[:ns]), PK.s_pulse(bp, theta[:ns])
            q_, Q_ = PK.s_pulse(ad, theta[:ns]), PK.s_pulse(bd, theta[:ns])
            acc_p += np.outer(p_, P_)
            acc_d += np.outer(q_, Q_)
            nacc += 1
        if si < len(snap_steps) and n == snap_steps[si]:
            gp, gd = acc_p / nacc, acc_d / nacc
            r_snap[si] = (mu_d * gd) / np.maximum(mu_p * gp, 1e-30)
            A_snap[si] = A[:ns, :ns]
            acc_p[:] = 0.0; acc_d[:] = 0.0; nacc = 0
            si += 1

        # ---- phases: Heun (RK2); weights: Euler (slow)
        k1 = dtheta(theta, A)
        k2 = dtheta(theta + dt * k1, A)
        pv, Pv = PK.s_pulse(ap, theta), PK.s_pulse(bp, theta)
        qv, Qv = PK.s_pulse(ad, theta), PK.s_pulse(bd, theta)
        Gp_m, Gd_m = np.outer(pv, Pv), np.outer(qv, Qv)
        if tau_fixed is None:                           # tau_w = 1/Gammabar, adaptive
            Gam_ema += dt * (mu_p * _pairmean(pv, Pv)
                             + mu_d * _pairmean(qv, Qv) - Gam_ema) / tau_gam
            tau_w = 1.0 / max(Gam_ema, 1e-9)
        Ep += dt * (Gp_m - Ep) / tau_w                  # slow part of each kernel
        Ed += dt * (Gd_m - Ed) / tau_w
        A += dt * (mu_p * Gp_m * (Am - A) - mu_d * Gd_m * A)
        np.clip(A, 0.0, Am, out=A)
        theta += 0.5 * dt * (k1 + k2)
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        if progress and n % (nsteps // 10 or 1) == 0:
            print(f"    {100*n/nsteps:3.0f}%", end="", flush=True)

    out = {k: np.array([r[k] for r in rec]) for k in rec[0]}
    out["t"] = np.array(trec)
    out["z"] = np.array(zrec)                                # (nrec, qmax+1) complex
    out["R"] = np.abs(out["z"][:, 1])
    out["tau_w"] = np.array(tau_hist)
    out.update(tau_static=tau_w, r_snap=r_snap, A_snap=A_snap, snap_t=snap_steps * dt, eta=eta,
               eta0=eta0, Delta=Delta, J=J, mu_p=mu_p, mu_d=mu_d, Am=Am,
               A0=cfg["A0"], N=N, n_s=n_s, qmax=qmax,
               kernel=cfg["kernel"],
               kern_p=np.array(kernel["p"]), kern_d=np.array(kernel["d"]))
    return out


# ════════════════════════════════════════════════════════════════════════════
#  entry point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--kernel" in sys.argv:
        CONFIG["kernel"] = sys.argv[sys.argv.index("--kernel") + 1]
    which = args or list(REGIMES)
    tag = "" if CONFIG["kernel"] == "duchet" else f"_{CONFIG['kernel']}"
    dp.ensure(CONFIG["out_dir"])
    for name in which:
        if name not in REGIMES:
            raise SystemExit(f"unknown regime {name!r}; choose from {list(REGIMES)}")
        p = REGIMES[name]
        print(f"[{name}] {p} kernel={CONFIG['kernel']}", flush=True)
        res = simulate(**p, progress=True)
        f = os.path.join(CONFIG["out_dir"], f"pulse_closure{tag}_{name}.npz")
        np.savez_compressed(f, **res)
        Ab, VA, Am = res["Abar"][-1], res["VA"][-1], res["Am"]
        print(f"\n   R={res['R'][-1]:.3f} Ā={Ab:.3f} V_A={VA:.4f} "
              f"s={VA/(Ab*(Am-Ab)):.3f}  Γ̄={p['mu_p']*res['Gp'][-1]+p['mu_d']*res['Gd'][-1]:.4f}"
              f"\n   -> {f}", flush=True)
