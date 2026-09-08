# -*- coding: utf-8 -*-
r"""
Fit an arbitrary 1-D parameter distribution by a weighted sum of M Lorentzians
(Eq. 5 of the manuscript):

    rho(w) ~ (1/pi) * sum_m w_m * Delta_m / ((w - Omega_m)^2 + Delta_m^2),
    sum_m w_m = 1,  w_m >= 0.

The number of ensembles M is a free parameter, selected (``fit``) by a GREEDY penalized
goodness-of-fit search. Every fixed-M fit is first PRUNED to its non-degenerate form (drop
zero-weight components, merge coincident centres), so the result never contains degenerate
ensembles and the penalty acts on the EFFECTIVE order m. Loop M=1..M_max:
  (alpha)  if the Cramer-von Mises fit is accepted, 1 - p < alpha (p the GoF p-value of the
           statistic T = N W^2 = N*D + 1/(12N)), stop and keep this smallest adequate M; or
  (lambda) otherwise track total(M) = D(M) + lambda_M * m and, once it has not improved for
           `patience` consecutive M, return argmin_M [D(M) + lambda_M m]. LARGER lambda_M =>
           fewer components. (Greedy argmin => robust to M_max.)
D is the mean-squared CDF discrepancy and Delta is hard-bounded, Delta_min <= Delta_m <= Delta_max.

The fixed-M fit handles the constraints (w_m >= 0, sum_m w_m = 1, Delta bounds) by one
of two equivalent routes, selected with ``method``:
  * "softmax" (default): unconstrained reparametrisation -- softmax for the simplex
    weights, scaled logistic for the bounded widths -- minimised with L-BFGS-B. The
    simplex equality is imposed implicitly (its Lagrange multiplier appears as the
    weighted-mean term in the softmax gradient).
  * "slsqp": optimise the *natural* parameters (w, Omega, Delta) directly with SLSQP,
    using box bounds (0<=w<=1, Delta_min<=Delta<=Delta_max) and the explicit linear
    equality sum_m w_m = 1 (SLSQP carries the multiplier internally). Uses the same
    analytic Cramer-von Mises gradient, before the softmax/logistic chain rule.
Both minimise the identical CvM loss D, so their results are directly comparable.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import cramervonmises


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


# ----------------------------------------- objectives in NATURAL params (SLSQP) --
def _cvm_obj_natural(p, M, xs, n, u):
    """CvM loss and analytic gradient w.r.t. the natural params p=[w, Omega, Delta]
    (i.e. Eqs. 10-12 directly, without the softmax/logistic chain rule). The simplex
    equality and the box bounds are imposed by SLSQP, not by this objective."""
    w, Om, De = p[:M], p[M:2 * M], p[2 * M:3 * M]
    dx = xs[:, None] - Om[None, :]
    denom = dx * dx + De[None, :] ** 2
    Phi = 0.5 + np.arctan(dx / De[None, :]) / np.pi
    pdf = (De[None, :] / np.pi) / denom
    F = Phi @ w
    r = F - u
    D = np.mean(r * r)
    g = (2.0 / n) * r                                   # dD/dF_i
    grad_w = Phi.T @ g                                  # dD/dw_k = sum_i g_i Phi_ik
    grad_Om = -w * (pdf.T @ g)                          # dD/dOmega_k
    dPhi_dDe = -(dx / np.pi) / denom
    grad_De = w * (dPhi_dDe.T @ g)                      # dD/dDelta_k
    return D, np.concatenate([grad_w, grad_Om, grad_De])


def _nll_obj_natural(p, M, xs, n, u):
    w, Om, De = p[:M], p[M:2 * M], p[2 * M:3 * M]
    return -np.mean(np.log(_comp_pdf(xs, Om, De) @ w + 1e-300))


def _init_raw(xs, M, dmin, dmax, rng, jitter=0.0):
    """Quantile-based initial (w, Omega, Delta) in natural parameters (valid simplex
    weights, in-bounds widths). Shared by both optimisers so restarts are comparable."""
    qs = (np.arange(M) + 0.5) / M
    Om = np.quantile(xs, qs)
    edges = np.quantile(xs, np.linspace(0.0, 1.0, M + 1))
    De = np.clip(0.5 * np.diff(edges), dmin, dmax) if M > 1 else \
        np.clip(np.array([0.5 * (np.percentile(xs, 75) - np.percentile(xs, 25))]),
                dmin, dmax)
    w = np.full(M, 1.0 / M)
    if jitter:
        Om = rng.choice(xs, M) + rng.normal(0, 0.1 * np.std(xs) + 1e-9, M)
        De = np.clip(De * np.exp(rng.normal(0.0, jitter, M)), dmin, dmax)
        w = _softmax(np.log(w) + rng.normal(0.0, jitter, M))
    return w, Om, De


def _init(xs, M, dmin, dmax, rng, jitter=0.0):                 # softmax-packed init
    w, Om, De = _init_raw(xs, M, dmin, dmax, rng, 0.0)
    th = _pack(w, Om, De, dmin, dmax)
    if jitter:                                                 # jitter in packed space
        th = th + rng.normal(0.0, jitter, th.size)
        th[M:2 * M] = rng.choice(xs, M) + rng.normal(0, 0.1 * np.std(xs) + 1e-9, M)
    return th


def fit_fixed_M(samples, M, delta_bounds, loss="cvm", n_restarts=6, seed=0,
                method="softmax"):
    """Best continuous fit of an M-Lorentzian mixture to `samples`.

    method : "softmax" (unconstrained reparametrisation + L-BFGS-B, default) or
             "slsqp" (natural params with box bounds + sum_m w_m = 1 via SLSQP).
    """
    xs = np.sort(np.asarray(samples, float)); n = xs.size
    dmin, dmax = delta_bounds
    u = (np.arange(n) + 0.5) / n
    rng = np.random.default_rng(seed)
    use_grad = (loss == "cvm")

    best = None
    if method == "softmax":
        inits = [_init(xs, M, dmin, dmax, rng, 0.0)]
        inits += [_init(xs, M, dmin, dmax, rng, 0.75) for _ in range(n_restarts)]
        obj = _cvm_obj if use_grad else _nll_obj
        for th0 in inits:
            res = minimize(obj, th0, args=(M, xs, n, dmin, dmax, u),
                           method="L-BFGS-B", jac=use_grad,
                           options=dict(maxiter=1000, ftol=1e-13, gtol=1e-10))
            if best is None or res.fun < best.fun:
                best = res
        w, Om, De = _unpack(best.x, M, dmin, dmax)
    elif method == "slsqp":
        raws = [_init_raw(xs, M, dmin, dmax, rng, 0.0)]
        raws += [_init_raw(xs, M, dmin, dmax, rng, 0.75) for _ in range(n_restarts)]
        obj = _cvm_obj_natural if use_grad else _nll_obj_natural
        bounds = [(0.0, 1.0)] * M + [(None, None)] * M + [(dmin, dmax)] * M
        eq = dict(type="eq",
                  fun=lambda p: np.sum(p[:M]) - 1.0,
                  jac=lambda p: np.concatenate([np.ones(M), np.zeros(2 * M)]))
        for w0, Om0, De0 in raws:
            p0 = np.concatenate([w0, Om0, De0])
            res = minimize(obj, p0, args=(M, xs, n, u), method="SLSQP", jac=use_grad,
                           bounds=bounds, constraints=[eq],
                           options=dict(maxiter=1000, ftol=1e-12))
            if res.success and (best is None or res.fun < best.fun):
                best = res
        if best is None:                               # all restarts flagged failure
            best = res
        p = best.x
        w, Om, De = p[:M], p[M:2 * M], p[2 * M:3 * M]
        w = np.clip(w, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(M, 1.0 / M)
    else:
        raise ValueError(f"unknown method {method!r} (use 'softmax' or 'slsqp')")

    order = np.argsort(Om)
    return LorentzianMixture(w[order], Om[order], De[order]), float(best.fun)


def _prune_mixture(model, w_min=1e-3, merge_frac=0.5):
    """Remove DEGENERATE components from a fitted mixture so the model order equals the number
    of components that are actually visible/usable: (i) drop components with weight < w_min,
    (ii) greedily merge components whose centres are closer than merge_frac*(Delta_i+Delta_j)/2
    (a weight-averaged centre & width), (iii) renormalise the weights. A non-degenerate fit is
    returned unchanged. merge_frac=0 disables merging."""
    w = np.asarray(model.w, float).copy()
    Om = np.asarray(model.Omega, float).copy()
    De = np.asarray(model.Delta, float).copy()
    keep = w >= w_min
    if not keep.any():
        keep = np.zeros_like(w, bool); keep[int(np.argmax(w))] = True
    w, Om, De = w[keep], Om[keep], De[keep]
    if merge_frac > 0:
        merged = True
        while merged and len(w) > 1:
            merged = False
            ci, cj, cd = -1, -1, np.inf
            for i in range(len(w)):
                for j in range(i + 1, len(w)):
                    tol = merge_frac * 0.5 * (De[i] + De[j])
                    d = abs(Om[i] - Om[j])
                    if d <= tol and d < cd:
                        ci, cj, cd = i, j, d
            if ci >= 0:
                s = w[ci] + w[cj]
                Om[ci] = (w[ci] * Om[ci] + w[cj] * Om[cj]) / s
                De[ci] = (w[ci] * De[ci] + w[cj] * De[cj]) / s
                w[ci] = s
                w = np.delete(w, cj); Om = np.delete(Om, cj); De = np.delete(De, cj)
                merged = True
    order = np.argsort(Om)
    return LorentzianMixture((w / w.sum())[order], Om[order], De[order])


def fit(samples, delta_bounds, M_max=8, alpha=0.05, lambda_M=1e-3, patience=2, loss="cvm",
        n_restarts=6, seed=0, method="softmax", w_min=1e-3, merge_frac=0.5, verbose=False):
    """Fit a Lorentzian mixture, choosing M by a GREEDY penalized goodness-of-fit search.

    Each fixed-M fit is PRUNED to its non-degenerate form (drop weight<w_min, merge coincident
    centres -- see _prune_mixture), so the returned model can never contain zero-weight or
    overlapping components, and the penalty acts on the EFFECTIVE number of ensembles m.

    Greedy loop over M=1..M_max with the penalized total loss  total(M) = D(M) + lambda_M * m:
      (1) GoF ACCEPTANCE -- if the fit is good enough, 1 - p < alpha (GoF p-value p > 1 - alpha,
          statistic T = N W^2), stop immediately and keep this (smallest adequate) M.
      (2) PENALIZED MINIMISATION -- otherwise track the running minimum of total(M); if it has
          not improved for `patience` consecutive M, stop and return the M with the LOWEST total
          (argmin_M [D(M) + lambda_M m]). Because total is found greedily, this is robust to
          M_max: the argmin is returned whether the minimum is interior or at the cap.

    alpha    : GoF acceptance level on (1 - p) (CvM test). NOTE: a target on the p-value, not a
               classical significance level -- a GoF test can only fail to reject. With fitted
               params the asymptotic CvM null is approximate; for large N, p saturates (so alpha
               may never trigger, and lambda_M/patience then choose M).
    lambda_M : per-(effective-)ensemble complexity penalty (replaces the old beta). LARGER
               lambda_M => fewer components (the marginal D drop must exceed lambda_M to help).
    patience : consecutive non-improving steps in total(M) before stopping (default 2).
    delta_bounds : (Delta_min, Delta_max) hard bounds on every ensemble width.
    loss     : 'cvm' (Cramer-von Mises, default) or 'nll'.
    method   : 'softmax' (default) or 'slsqp' constrained fit (see fit_fixed_M).
    w_min, merge_frac : pruning thresholds (internal; not tuning meta-parameters).
    Returns a dict with the chosen model, M, data_loss, T, pvalue, total_loss, alpha, lambda_M,
    patience and the per-M trace.
    """
    xs = np.sort(np.asarray(samples, float)); n = xs.size
    trace = []
    best = None                                          # entry with the lowest penalized total
    accepted = None
    stall = 0
    for M in range(1, M_max + 1):
        model, _ = fit_fixed_M(samples, M, delta_bounds, loss=loss,
                               n_restarts=n_restarts, seed=seed, method=method)
        model = _prune_mixture(model, w_min, merge_frac)  # -> non-degenerate, effective order m
        m = model.M
        gof = cramervonmises(xs, model.cdf)               # statistic T = N W^2, asymptotic p-value
        T, pval = float(gof.statistic), float(gof.pvalue)
        D = (T - 1.0 / (12.0 * n)) / n                    # CvM loss (mean-squared CDF discrepancy)
        total = D + lambda_M * m
        trace.append(dict(M=m, M_nominal=M, data_loss=D, T=T, pvalue=pval,
                          total_loss=total, model=model))
        if verbose:
            print("  M=%2d (eff %2d)  D=%.4e  1-p=%.4f  total=D+lambda*m=%.4e"
                  % (M, m, D, 1.0 - pval, total), flush=True)
        if (1.0 - pval) < alpha:                          # (1) GoF acceptance -> smallest adequate M
            accepted = trace[-1]
            if verbose:
                print("    -> GoF accepted (1-p < alpha=%.3g): M=%d" % (alpha, m), flush=True)
            break
        if best is None or total < best["total_loss"]:    # (2) penalized minimisation w/ patience
            best, stall = trace[-1], 0
        else:
            stall += 1
            if verbose:
                print("    total not improved (stall %d/%d)" % (stall, patience), flush=True)
            if stall >= patience:
                if verbose:
                    print("    -> no improvement for %d steps: keep argmin M=%d"
                          % (patience, best["M"]), flush=True)
                break
    chosen = accepted if accepted is not None else best
    return dict(model=chosen["model"], M=chosen["M"], data_loss=chosen["data_loss"],
                T=chosen["T"], pvalue=chosen["pvalue"], total_loss=chosen["total_loss"],
                alpha=alpha, lambda_M=lambda_M, patience=patience, trace=trace)
