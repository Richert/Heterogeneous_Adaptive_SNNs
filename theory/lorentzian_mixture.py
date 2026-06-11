# -*- coding: utf-8 -*-
r"""
Fit an arbitrary 1-D parameter distribution by a weighted sum of M Lorentzians
(Eq. 5 of the manuscript):

    rho(w) ~ (1/pi) * sum_m w_m * Delta_m / ((w - Omega_m)^2 + Delta_m^2),
    sum_m w_m = 1,  w_m >= 0.

The number of ensembles M is a free parameter, selected by minimizing

    total_loss(M) = D(empirical, mixture_M) + lambda_M * M,

with D a distribution-mismatch term, lambda_M an explicit per-ensemble penalty,
and hard bounds Delta_min <= Delta_m <= Delta_max.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


# ---------------------------------------------------------------- mixture ----
def _comp_pdf(x, Om, De):                      # (n, M) per-component densities
    x = np.asarray(x)[:, None]
    return (De[None, :] / np.pi) / ((x - Om[None, :]) ** 2 + De[None, :] ** 2)


def _comp_cdf(x, Om, De):                      # (n, M) per-component CDFs
    x = np.asarray(x)[:, None]
    return 0.5 + np.arctan((x - Om[None, :]) / De[None, :]) / np.pi


class LorentzianMixture:
    def __init__(self, w, Omega, Delta):
        self.w = np.asarray(w, float)
        self.Omega = np.asarray(Omega, float)
        self.Delta = np.asarray(Delta, float)

    @property
    def M(self):
        return len(self.w)

    def pdf(self, x):
        return _comp_pdf(np.atleast_1d(x), self.Omega, self.Delta) @ self.w

    def cdf(self, x):
        return _comp_cdf(np.atleast_1d(x), self.Omega, self.Delta) @ self.w

    def params(self):
        return dict(w=self.w, Omega=self.Omega, Delta=self.Delta)


# ------------------------------------------------ (un)constrained transform --
def _softmax(a):
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def _unpack(theta, M, dmin, dmax):
    w = _softmax(theta[:M])
    Om = theta[M:2 * M]
    De = dmin + (dmax - dmin) * expit(theta[2 * M:3 * M])           # in (dmin,dmax)
    return w, Om, De


def _pack(w, Om, De, dmin, dmax):
    a = np.log(np.clip(w, 1e-12, None)); a -= a.mean()
    t = np.clip((De - dmin) / (dmax - dmin), 1e-6, 1 - 1e-6)
    b = np.log(t / (1.0 - t))
    return np.concatenate([a, Om, b])


# ------------------------------------------------------------- data losses ---
def _cvm_obj(theta, M, xs, n, dmin, dmax, u):
    """Cramer-von Mises loss and its analytic gradient (jac=True)."""
    w, Om, De = _unpack(theta, M, dmin, dmax)
    dx = xs[:, None] - Om[None, :]                  # (n, M)
    denom = dx * dx + De[None, :] ** 2
    Phi = 0.5 + np.arctan(dx / De[None, :]) / np.pi
    pdf = (De[None, :] / np.pi) / denom             # d Phi / d Om = -pdf
    F = Phi @ w
    r = F - u
    D = np.mean(r * r)
    g = (2.0 / n) * r                               # dD/dF_i
    # weights (through softmax): dF/da_k = w_k (Phi_k - F)
    grad_a = w * (Phi.T @ g - g @ F)
    grad_Om = -w * (pdf.T @ g)
    dPhi_dDe = -(dx / np.pi) / denom                # d Phi / d Delta
    grad_De = w * (dPhi_dDe.T @ g)
    sig = (De - dmin) / (dmax - dmin)               # = sigmoid(b)
    grad_b = grad_De * (dmax - dmin) * sig * (1.0 - sig)
    return D, np.concatenate([grad_a, grad_Om, grad_b])


def _nll_obj(theta, M, xs, n, dmin, dmax, u):
    w, Om, De = _unpack(theta, M, dmin, dmax)
    p = _comp_pdf(xs, Om, De) @ w
    return -np.mean(np.log(p + 1e-300))


def _init(xs, M, dmin, dmax, rng, jitter=0.0):
    qs = (np.arange(M) + 0.5) / M
    Om = np.quantile(xs, qs)
    edges = np.quantile(xs, np.linspace(0.0, 1.0, M + 1))
    De = np.clip(0.5 * np.diff(edges), dmin, dmax) if M > 1 else \
        np.clip(np.array([0.5 * (np.percentile(xs, 75) - np.percentile(xs, 25))]),
                dmin, dmax)
    w = np.full(M, 1.0 / M)
    th = _pack(w, Om, De, dmin, dmax)
    if jitter:
        th = th + rng.normal(0.0, jitter, th.size)
        th[M:2 * M] = rng.choice(xs, M) + rng.normal(0, 0.1 * np.std(xs) + 1e-9, M)
    return th


def fit_fixed_M(samples, M, delta_bounds, loss="cvm", n_restarts=6, seed=0):
    """Best continuous fit of an M-Lorentzian mixture to `samples`."""
    xs = np.sort(np.asarray(samples, float)); n = xs.size
    dmin, dmax = delta_bounds
    u = (np.arange(n) + 0.5) / n
    rng = np.random.default_rng(seed)
    inits = [_init(xs, M, dmin, dmax, rng, 0.0)]
    inits += [_init(xs, M, dmin, dmax, rng, 0.75) for _ in range(n_restarts)]
    use_grad = (loss == "cvm")
    obj = _cvm_obj if use_grad else _nll_obj
    best = None
    for th0 in inits:
        res = minimize(obj, th0, args=(M, xs, n, dmin, dmax, u),
                       method="L-BFGS-B", jac=use_grad,
                       options=dict(maxiter=1000, ftol=1e-13, gtol=1e-10))
        if best is None or res.fun < best.fun:
            best = res
    w, Om, De = _unpack(best.x, M, dmin, dmax)
    order = np.argsort(Om)
    return LorentzianMixture(w[order], Om[order], De[order]), float(best.fun)


def fit(samples, delta_bounds, M_max=8, lambda_M=1e-3, loss="cvm",
        penalty=None, n_restarts=6, seed=0, verbose=False):
    """Fit with M selected by total_loss = D(M) + penalty(M).

    delta_bounds : (Delta_min, Delta_max) hard bounds on every ensemble width.
    lambda_M     : per-ensemble penalty (used as penalty(M)=lambda_M*M unless a
                   callable `penalty` is supplied; e.g. BIC -> see make_bic_penalty).
    loss         : 'cvm' (Cramer-von Mises, default) or 'nll'.
    Returns a dict with the chosen model, M, and the full per-M trace.
    """
    pen = (lambda M: lambda_M * M) if penalty is None else penalty
    trace = []
    for M in range(1, M_max + 1):
        model, D = fit_fixed_M(samples, M, delta_bounds, loss=loss,
                               n_restarts=n_restarts, seed=seed)
        total = D + pen(M)
        trace.append(dict(M=M, data_loss=D, penalty=pen(M), total_loss=total,
                          model=model))
        if verbose:
            print("  M=%2d  D=%.6e  pen=%.6e  total=%.6e"
                  % (M, D, pen(M), total), flush=True)
    best = min(trace, key=lambda r: r["total_loss"])
    return dict(model=best["model"], M=best["M"], data_loss=best["data_loss"],
                total_loss=best["total_loss"], trace=trace)


def make_bic_penalty(n):
    """BIC-style penalty for use with loss='nll' (which returns *mean* NLL):
    total ~ 2*n*meanNLL + k*ln(n), k = 3M-1 free params.  Returns penalty(M)."""
    return lambda M: (3 * M - 1) * np.log(n) / (2.0 * n)
