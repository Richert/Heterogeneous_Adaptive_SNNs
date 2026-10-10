r"""
RESULT (2026-10-08): no consistent improvement over the CvM fit -- outcome depends erratically on
eps (see continuity_notes.md, Sec. 7). Kept for reference.

Prototype: a dynamics-matched loss for the Lorentzian-mixture fit, compared with the CvM fit.

Spectral (Cauchy-transform) loss at resolution eps > 0:

    L_eps(mu, nu) = int |Phi_mu(eps+i nu) - Phi_nu(eps+i nu)|^2 d nu
                  = 2 pi int_0^inf |phi_mu(s) - phi_nu(s)|^2 e^{-2 eps s} ds          (Plancherel)

where Phi is the Cauchy transform and phi the characteristic function. This is the quantity that
enters the linear-response error dZ^ = z0 dPhi / ((1-kPhi)(1-kPhi_M)) (continuity_notes.md, Sec. 2).
For Lorentzian/Dirac atoms with poles p_k = Delta_k - i Omega_k (Delta=0 for samples):

    <k|l> = 2 pi / (2 eps + p_k + conj(p_l))

so L_eps = w^T G_MM w - 2 Re(w^T G_MN 1/N) + const  — closed form, O(N M), quadratic in w.

Usage (pycobi env):  python spectral_loss_prototype.py
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "..", "shared")]
import numpy as np
from scipy.optimize import minimize
import lorentzian_mixture as LM
import continuity_check as C


def gram(pa, pb, eps):
    return 2 * np.pi / (2 * eps + pa[:, None] + np.conj(pb)[None, :])


def spectral_const(omega, eps, block=2000):
    """Parameter-independent part of L_eps (the sample-sample term), computed once.
    O(N^2) work, done in blocks to avoid an N x N matrix in memory."""
    pn = -1j * np.asarray(omega, float)
    tot = 0.0
    for i in range(0, pn.size, block):
        tot += gram(pn[i:i + block], pn, eps).real.sum()
    return float(tot / pn.size ** 2)


def spectral_loss(w, Om, De, omega, eps, const=None):
    """L_eps(rho_N, rho_M). Pass `const` (spectral_const(omega, eps)) when calling repeatedly --
    otherwise the O(N^2) sample-sample term is recomputed on every call."""
    pm = De - 1j * Om
    pn = -1j * omega
    G_mm = gram(pm, pm, eps).real
    cross = (gram(pm, pn, eps).real.sum(1)) / omega.size
    if const is None:
        const = spectral_const(omega, eps)
    return float(w @ G_mm @ w - 2 * w @ cross + const)


def spectral_loss_grad(w, Om, De, omega, eps):
    """Analytic gradient of L_eps w.r.t. (w, Omega, Delta). With h(s) = 2π/(2ε + s), p = Δ − iΩ:
    L = Σ_jk w_j w_k Re h(p_j + p̄_k) − 2 Σ_j w_j <Re h(p_j + iω_i)>_i + const, so
    ∂L/∂w_j = 2 Σ_k w_k Re h(p_j+p̄_k) − 2 <Re h(p_j+iω_i)>,
    ∂L/∂Δ_j = 2 w_j [Σ_k w_k Re h'(p_j+p̄_k) − <Re h'(p_j+iω_i)>],
    ∂L/∂Ω_j = 2 w_j [Σ_k w_k Im h'(p_j+p̄_k) − <Im h'(p_j+iω_i)>],  h'(s) = −2π/(2ε+s)²."""
    p = De - 1j * Om
    S_mm = 2 * eps + p[:, None] + np.conj(p)[None, :]
    S_mn = 2 * eps + p[:, None] + 1j * np.asarray(omega)[None, :]
    h_mm, h_mn = 2 * np.pi / S_mm, 2 * np.pi / S_mn
    dh_mm, dh_mn = -h_mm / S_mm, -h_mn / S_mn
    g_w = 2 * (h_mm.real @ w) - 2 * h_mn.real.mean(1)
    g_De = 2 * w * ((dh_mm.real @ w) - dh_mn.real.mean(1))
    g_Om = 2 * w * ((dh_mm.imag @ w) - dh_mn.imag.mean(1))
    return np.concatenate([g_w, g_Om, g_De])


def fit_spectral(omega, M, eps, init=None, n_restarts=6, seed=0, dmin=1e-3):
    """Fixed-M fit minimising L_eps. Natural params + SLSQP (simplex + Delta >= dmin)."""
    rng = np.random.default_rng(seed)
    xs = np.sort(omega)
    def obj(p):
        return spectral_loss(p[:M], p[M:2 * M], p[2 * M:], omega, eps)
    inits = []
    if init is not None and init.M == M:                # pruned inits may have fewer components
        inits.append(np.concatenate([init.w, init.Omega, np.maximum(init.Delta, dmin)]))
    for _ in range(n_restarts):
        q = np.sort(rng.uniform(0.02, 0.98, M))
        Om0 = np.quantile(xs, q)
        inits.append(np.concatenate([np.full(M, 1 / M), Om0,
                                     np.full(M, np.std(xs) / M) * rng.uniform(0.5, 1.5, M)]))
    bounds = [(0, 1)] * M + [(None, None)] * M + [(dmin, 1e2)] * M
    eq = dict(type="eq", fun=lambda p: p[:M].sum() - 1)
    best = None
    for p0 in inits:
        r = minimize(obj, p0, method="SLSQP", bounds=bounds, constraints=[eq],
                     options=dict(maxiter=500, ftol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    w = np.clip(p[:M], 0, None); w /= w.sum()
    o = np.argsort(p[M:2 * M])
    return LM._prune_mixture(LM.LorentzianMixture(w[o], p[M:2 * M][o], p[2 * M:][o])), best.fun


def noise_floor(omega, eps, n_boot=20, seed=0):
    """Bootstrap estimate of E L_eps(rho_N, rho) via L_eps between two resamples / 2."""
    rng = np.random.default_rng(seed)
    N = omega.size; v = []
    for _ in range(n_boot):
        a = rng.choice(omega, N); b = rng.choice(omega, N)
        pa, pb = -1j * a, -1j * b
        Laa = gram(pa, pa, eps).real.sum() / N ** 2
        Lbb = gram(pb, pb, eps).real.sum() / N ** 2
        Lab = gram(pa, pb, eps).real.sum() / N ** 2
        v.append((Laa + Lbb - 2 * Lab) / 2)
    return float(np.mean(v))


def main(N=5000, seeds=(1, 2, 3), Ms=(2, 3, 4, 6, 8), eps_list=(0.05, 0.2), K=3.0, T=30.0):
    t, Z_ref = C.riccati(C.quantile_nodes(20000), np.full(20000, 1 / 20000), C.R0, K, T)
    n30 = t.size
    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        om = C.sample_rho(N, rng)
        th0 = C.wrapped_cauchy(N, C.R0, rng)
        _, Z_net = C.riccati(om, np.full(N, 1 / N), np.exp(1j * th0), K, T, phase_form=True,
                             rtol=1e-7, atol=1e-9)
        floors = {eps: noise_floor(om, eps, seed=sd) for eps in eps_list}
        for M in Ms:
            m_cvm, _ = LM.fit_fixed_M(om, M, C.DB, n_restarts=10, seed=sd, method="slsqp")
            m_cvm = LM._prune_mixture(m_cvm)
            fits = {"cvm": m_cvm}
            for eps in eps_list:
                fits[f"spec{eps}"], _ = fit_spectral(om, M, eps, init=m_cvm, seed=sd)
            for name, m in fits.items():
                _, Z = C.lmmf(m, C.R0, K, T)
                rec = dict(seed=sd, M=M, Meff=m.M, fit=name, bl=C.d_bl(m), ks=C.d_ks(m),
                           Lspec=[spectral_loss(m.w, m.Omega, m.Delta, om, e) for e in eps_list],
                           e5_ref=np.abs(Z - Z_ref)[t <= 5].max(),
                           s_ref=C.spectral_rmse(np.abs(Z_ref), np.abs(Z)),
                           e5_net=np.abs(Z - Z_net)[t <= 5].max(),
                           s_net=C.spectral_rmse(np.abs(Z_net), np.abs(Z)))
                rows.append(rec)
                print(f"seed={sd} M={M} ({m.M}) {name:9s} BL={rec['bl']:.3f} "
                      f"| vs MF: sup5={rec['e5_ref']:.3f} spec={rec['s_ref']:.2e} "
                      f"| vs net: sup5={rec['e5_net']:.3f} spec={rec['s_net']:.2e} "
                      f"| L/floor={[round(l / floors[e], 1) for l, e in zip(rec['Lspec'], eps_list)]}",
                      flush=True)
        net_ref = (np.abs(Z_net - Z_ref)[t <= 5].max(), C.spectral_rmse(np.abs(Z_ref), np.abs(Z_net)))
        print(f"seed={sd}: network vs MF(rho): sup5={net_ref[0]:.3f} spec={net_ref[1]:.2e}; "
              f"noise floors {floors}", flush=True)
    np.save(os.path.join(_HERE, "spectral_loss_prototype.npy"), rows, allow_pickle=True)


if __name__ == "__main__":
    main()
