r"""
Numerical checks of the continuity / convergence analysis for the LMMF reduction
(referee report lu21625, point 1 & 2). See continuity_notes.md for the derivations.

Everything is written in the *local-order-parameter (Riccati) form*, which covers all
three models with one right-hand side:

    dz_j/dt = i nu_j z_j + (K/2) (Z - conj(Z) z_j^2),     Z = sum_j q_j z_j

  * continuum OA reference for the true rho :  nu_j = quantile nodes of rho, q_j = 1/Nq,
                                               z_j(0) = R0  (OA manifold, Poisson-kernel IC)
  * LMMF (Lorentzian mixture)               :  nu_m = Omega_m + i Delta_m, q_m = w_m, z_m(0)=R0
  * N-oscillator Kuramoto network           :  nu_i = omega_i, q_i = 1/N, z_i(0) = exp(i theta_i(0))
                                               (|z_i| = 1 is invariant; this IS the Kuramoto eq.)

Experiments (each writes an npz into this directory):
  A  distribution distance vs. dynamical error for M = 1..M_hi fits to rho  (Prop. 1/Lipschitz)
  B  linear response around incoherence: Laplace/Fourier spectrum of Z vs. the closed form
     z0 Phi/(1 - K Phi/2),  Phi(lambda) = int rho(w)/(lambda - i w) dw
  C  finite N and off-manifold initial condition: micro (wrapped Gaussian / wrapped Cauchy IC)
     vs. continuum reference, N in {500, 5000, 50000}
  D  sampling variability of the selected M (manuscript selection rule) across seeds and N,
     plus the sample-free approximation curve a(M)

Run in the pycobi env:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python continuity_check.py [A|B|C|D ...]
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "..", "shared")]
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
from scipy.stats import norm
import lorentzian_mixture as LM

# ── true frequency distribution: Fig. 1 Gaussian mixture ─────────────────────
MU, SD, CW = np.array([-2.0, 2.0]), np.array([0.6, 0.6]), np.array([0.5, 0.5])
K_FIG1, SIGMA0 = 3.0, 0.5
R0 = float(np.exp(-SIGMA0 ** 2 / 2))          # |<e^{i theta}>| of the wrapped Gaussian IC
DB = (1e-4, 1e2)                               # Delta bounds used in the manuscript


def F_true(x):
    return (CW * norm.cdf((np.asarray(x)[:, None] - MU) / SD)).sum(1)


def pdf_true(x):
    return (CW * norm.pdf((np.asarray(x)[:, None] - MU) / SD) / SD).sum(1)


def phi_true(s):
    s = np.asarray(s)[:, None]
    return (CW * np.exp(1j * MU * s - 0.5 * SD ** 2 * s ** 2)).sum(1)


def phi_mix(s, m):
    s = np.abs(np.asarray(s))[:, None]
    return (m.w * np.exp((1j * m.Omega - m.Delta) * s)).sum(1)


def quantile_nodes(n):
    """Deterministic 'perfect sample': omega_j = F^{-1}((j-1/2)/n)."""
    g = np.linspace(-12, 12, 400001)
    return np.interp((np.arange(n) + 0.5) / n, F_true(g), g)


def sample_rho(n, rng):
    c = rng.choice(len(MU), size=n, p=CW)
    return rng.normal(MU[c], SD[c])


# ── one Riccati integrator for all three model classes ──────────────────────
def riccati(nu, q, z0, K, T, dts=0.1, rtol=1e-8, atol=1e-10, phase_form=False):
    """Returns t, Z(t). nu complex (Omega + i Delta) or real; q weights summing to 1."""
    nu = np.asarray(nu, complex); q = np.asarray(q, float)
    t_eval = np.arange(0.0, T + 1e-9, dts)
    if phase_form:                               # unit-modulus nodes: integrate theta (exact |z|=1)
        om = nu.real
        def f(t, th):
            Z = q @ np.exp(1j * th)
            return om + K * np.imag(Z * np.exp(-1j * th))
        y0 = np.angle(z0)
        sol = solve_ivp(f, (0, T), y0, t_eval=t_eval, rtol=rtol, atol=atol, method="RK45")
        return sol.t, q @ np.exp(1j * sol.y)
    a = 1j * nu
    def f(t, z):
        Z = q @ z
        return a * z + 0.5 * K * (Z - np.conj(Z) * z * z)
    sol = solve_ivp(f, (0, T), np.asarray(z0, complex) * np.ones(nu.size), t_eval=t_eval,
                    rtol=rtol, atol=atol, method="RK45")
    return sol.t, q @ sol.y


def lmmf(m, z0, K, T, **kw):
    return riccati(m.Omega + 1j * m.Delta, m.w, z0, K, T, **kw)


# ── distances between rho and a Lorentzian mixture ──────────────────────────
XG = np.linspace(-14, 14, 2801)                 # grid for KS / CvM / BL


def d_ks(m):
    return float(np.max(np.abs(F_true(XG) - m.cdf(XG))))


def d_cvm(m):
    """Population CvM: int (F - F_M)^2 dF_M."""
    x = np.linspace(-60, 60, 120001)
    return float(np.trapezoid((F_true(x) - m.cdf(x)) ** 2 * m.pdf(x), x))


def d_bl(m):
    """Bounded-Lipschitz (Fortet-Mourier) distance sup{|int f d(mu-nu)| : |f|<=1, Lip f<=1},
    by LP on a grid; tail mass outside the grid is lumped onto the end nodes."""
    x = XG; h = x[1] - x[0]
    edges = np.concatenate([[-np.inf], 0.5 * (x[1:] + x[:-1]), [np.inf]])
    def cell_mass(F):
        Fe = np.concatenate([[0.0], F(edges[1:-1]), [1.0]])
        return np.diff(Fe)
    dm = cell_mass(F_true) - cell_mass(m.cdf)
    n = x.size
    # maximise dm.f  <=>  minimise -dm.f ; |f_{j+1} - f_j| <= h ; -1 <= f <= 1
    D = np.zeros((n - 1, n)); D[np.arange(n - 1), np.arange(n - 1)] = -1; D[np.arange(n - 1), np.arange(1, n)] = 1
    A = np.vstack([D, -D]); b = np.full(2 * (n - 1), h)
    from scipy.sparse import csr_matrix
    res = linprog(-dm, A_ub=csr_matrix(A), b_ub=b, bounds=[(-1, 1)] * n, method="highs")
    return float(-res.fun)


def d_char(m, T):
    s = np.linspace(0, T, 4001)
    return float(np.max(np.abs(phi_true(s) - phi_mix(s, m))))


def spectral_rmse(R_ref, R):                    # identical to kmo_lorentzian_fit_figure.py
    n = min(len(R_ref), len(R))
    Fa = np.abs(np.fft.rfft(R_ref[:n])) / n
    Fb = np.abs(np.fft.rfft(R[:n])) / n
    return float(np.sqrt(np.mean((Fb - Fa) ** 2)))


# ════════════════════════════════════════════════════════════════════════════
#  A: distance vs. dynamical error (finite-time Lipschitz continuity)
# ════════════════════════════════════════════════════════════════════════════
def exp_A(M_hi=10, Nq=20000, Nfit=4000, T_long=200.0):
    K = K_FIG1
    nodes = quantile_nodes(Nq)
    t, Z_ref = riccati(nodes, np.full(Nq, 1 / Nq), R0, K, T_long)
    fit_x = quantile_nodes(Nfit)
    out = dict(t=t, Z_ref=Z_ref, M=[], ks=[], cvm=[], bl=[], ch10=[], ch30=[],
               Esup=[], Tsup=np.array([1, 2, 5, 10, 20, 30.0]), srmse=[], Z=[], w=[], Om=[], De=[])
    for M in range(1, M_hi + 1):
        m, _ = LM.fit_fixed_M(fit_x, M, DB, n_restarts=10, seed=M, method="slsqp")
        m = LM._prune_mixture(m)
        _, Z = lmmf(m, R0, K, T_long)
        E = np.abs(Z - Z_ref)
        Esup = [float(E[t <= T].max()) for T in out["Tsup"]]
        n30 = int(np.searchsorted(t, 30.0))
        sr = spectral_rmse(np.abs(Z_ref[:n30]), np.abs(Z[:n30]))
        rec = dict(M=m.M, ks=d_ks(m), cvm=d_cvm(m), bl=d_bl(m), ch10=d_char(m, 10), ch30=d_char(m, 30))
        for k, v in rec.items():
            out[k].append(v)
        out["Esup"].append(Esup); out["srmse"].append(sr); out["Z"].append(Z)
        out["w"].append(m.w); out["Om"].append(m.Omega); out["De"].append(m.Delta)
        print(f"[A] M={M:2d} (eff {m.M:2d})  KS={rec['ks']:.2e} CvM={rec['cvm']:.2e} BL={rec['bl']:.2e} "
              f"chi10={rec['ch10']:.2e} | sup|dZ| T=5:{Esup[2]:.2e} T=30:{Esup[5]:.2e} specRMSE={sr:.2e}",
              flush=True)
    np.savez(os.path.join(_HERE, "expA.npz"),
             **{k: np.array(v, dtype=object) if k in ("w", "Om", "De") else np.asarray(v)
                for k, v in out.items()})


# ════════════════════════════════════════════════════════════════════════════
#  B: linear response around incoherence -> spectrum of Z
# ════════════════════════════════════════════════════════════════════════════
def Phi_true(lam):
    w = np.linspace(-12, 12, 240001)
    r = pdf_true(w)
    return np.array([np.trapezoid(r / (l - 1j * w), w) for l in np.atleast_1d(lam)])


def Phi_mix(lam, m):
    lam = np.atleast_1d(lam)[:, None]
    return (m.w / (lam + m.Delta - 1j * m.Omega)).sum(1)


def exp_B(Ks=(1.0, 1.6), z0=0.02, eps=0.1, T=200.0, Ms=(2, 4, 8), Nq=20000, Nfit=4000):
    nodes = quantile_nodes(Nq); fit_x = quantile_nodes(Nfit)
    nu = np.linspace(-5, 5, 401); lam = eps + 1j * nu
    PhiT = Phi_true(lam)
    out = dict(nu=nu, eps=eps, Ks=np.array(Ks), Ms=np.array(Ms), PhiT=PhiT)
    mixes = {}
    for M in Ms:
        m, _ = LM.fit_fixed_M(fit_x, M, DB, n_restarts=10, seed=M, method="slsqp")
        mixes[M] = LM._prune_mixture(m)
        out[f"PhiM{M}"] = Phi_mix(lam, mixes[M])
    def laplace(t, Z):
        return np.array([np.trapezoid(Z * np.exp(-l * t), t) for l in lam])
    for K in Ks:
        t, Zr = riccati(nodes, np.full(Nq, 1 / Nq), z0, K, T, dts=0.02)
        out[f"Lnum_ref_K{K}"] = laplace(t, Zr)
        out[f"Lth_ref_K{K}"] = z0 * PhiT / (1 - 0.5 * K * PhiT)
        for M in Ms:
            t, Zm = lmmf(mixes[M], z0, K, T, dts=0.02)
            PM = out[f"PhiM{M}"]
            out[f"Lnum_M{M}_K{K}"] = laplace(t, Zm)
            out[f"Lth_M{M}_K{K}"] = z0 * PM / (1 - 0.5 * K * PM)
        rel = np.max(np.abs(out[f"Lnum_ref_K{K}"] - out[f"Lth_ref_K{K}"])) / np.max(np.abs(out[f"Lth_ref_K{K}"]))
        print(f"[B] K={K}: linear-response formula vs simulation, max rel. err = {rel:.2e}", flush=True)
    np.savez(os.path.join(_HERE, "expB.npz"), **out)


# ════════════════════════════════════════════════════════════════════════════
#  C: finite N + off-manifold IC vs continuum reference
# ════════════════════════════════════════════════════════════════════════════
def wrapped_cauchy(n, r, rng):
    # theta = 2 arctan( (1-r)/(1+r) tan(pi(U-1/2)) )  has <e^{i theta}> = r
    u = rng.random(n)
    return 2 * np.arctan((1 - r) / (1 + r) * np.tan(np.pi * (u - 0.5)))


def exp_C(Ns=(500, 5000, 50000), seeds=5, T=30.0, Nq=20000):
    K = K_FIG1
    t, Z_ref = riccati(quantile_nodes(Nq), np.full(Nq, 1 / Nq), R0, K, T)
    out = dict(t=t, Z_ref=Z_ref, Ns=np.array(Ns))
    for N in Ns:
        for ic in ("gauss", "cauchy"):
            E5, E30, SR = [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(1000 + s)
                om = sample_rho(N, rng)
                th0 = rng.normal(0, SIGMA0, N) if ic == "gauss" else wrapped_cauchy(N, R0, rng)
                _, Z = riccati(om, np.full(N, 1 / N), np.exp(1j * th0), K, T, phase_form=True,
                               rtol=1e-7, atol=1e-9)
                E = np.abs(Z - Z_ref)
                E5.append(E[t <= 5].max()); E30.append(E.max())
                SR.append(spectral_rmse(np.abs(Z_ref), np.abs(Z)))
                if s == 0:
                    out[f"Z_{ic}_N{N}"] = Z
            out[f"E5_{ic}_N{N}"] = np.array(E5); out[f"E30_{ic}_N{N}"] = np.array(E30)
            out[f"SR_{ic}_N{N}"] = np.array(SR)
            print(f"[C] N={N:6d} IC={ic:6s}  sup|dZ|_[0,5]={np.mean(E5):.3e}  "
                  f"sup|dZ|_[0,30]={np.mean(E30):.3e}  specRMSE={np.mean(SR):.2e}", flush=True)
    np.savez(os.path.join(_HERE, "expC.npz"), **out)


# ════════════════════════════════════════════════════════════════════════════
#  D: sampling variability of the selected M
# ════════════════════════════════════════════════════════════════════════════
def _fit_job(args):
    N, seed, lam = args
    om = sample_rho(N, np.random.default_rng(seed))
    r = LM.fit(om, DB, M_max=16, alpha=1e-2, lambda_M=lam, patience=3, n_restarts=10,
               seed=seed, method="slsqp")
    m = r["model"]
    return dict(N=N, seed=seed, lam=lam, M=r["M"], ks=d_ks(m), bl=d_bl(m), data_loss=r["data_loss"])


def exp_D(Ns=(500, 2000, 5000, 20000), seeds=20, lams=(1e-5, 1e-4), M_hi=12, Nq=20000):
    from multiprocessing import Pool
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(N, s, lam) for N in Ns for lam in lams for s in range(seeds)]
    with Pool(min(12, os.cpu_count())) as p:
        res = p.map(_fit_job, jobs, chunksize=1)
    # sample-free approximation curve a(M) (fit to the quantile nodes of rho)
    xq = quantile_nodes(Nq)
    aM = []
    for M in range(1, M_hi + 1):
        m, D = LM.fit_fixed_M(xq, M, DB, n_restarts=10, seed=M, method="slsqp")
        m = LM._prune_mixture(m)
        aM.append((M, m.M, D, d_ks(m), d_bl(m)))
        print(f"[D] a(M): M={M:2d} eff={m.M:2d} CvM={D:.2e} KS={aM[-1][3]:.2e} BL={aM[-1][4]:.2e}", flush=True)
    np.savez(os.path.join(_HERE, "expD.npz"),
             N=np.array([r["N"] for r in res]), seed=np.array([r["seed"] for r in res]),
             lam=np.array([r["lam"] for r in res]), M=np.array([r["M"] for r in res]),
             ks=np.array([r["ks"] for r in res]), bl=np.array([r["bl"] for r in res]),
             data_loss=np.array([r["data_loss"] for r in res]), aM=np.array(aM))
    for N in Ns:
        for lam in lams:
            Ms = [r["M"] for r in res if r["N"] == N and r["lam"] == lam]
            print(f"[D] N={N:6d} lam={lam:g}: M* counts {dict(zip(*np.unique(Ms, return_counts=True)))}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C", "D"]
    for w in which:
        t0 = time.time()
        dict(A=exp_A, B=exp_B, C=exp_C, D=exp_D)[w]()
        print(f"[{w}] done in {time.time() - t0:.0f}s", flush=True)
