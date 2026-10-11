r"""
Three fitting scenarios for the unimodal distributions: best fit to ρ, and sample fits with / without
the noise floor.
====================================================================================================

  (1) "true ρ, no floor"   : quantile nodes of ρ (no sampling noise), floor_c = 0 (no acceptance rule),
                             M chosen by a small λ alone (penalized argmin, patience, M_max) — the best
                             possible Lorentzian-mixture approximation of ρ;
  (2) "sample, floor"      : N-sample, noise-floor cap (floor_c = 1, floor at the sample size), same λ —
                             the fit that accounts for sampling noise, i.e. approximates ρ from the data;
  (3) "sample, no floor"   : same samples, floor_c = 0, same λ — the best fit to the SAMPLE distribution.
For every fit: the full LMMF incoherence boundary in (K, α) (zero contour of the largest eigenvalue real
part at z = 0, unimodal_kalpha.lmmf_stability_map), compared with the exact boundary of the true ρ.
Samples: N_SAMPLES independent draws of size N (the same draws for (2) and (3)).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python fit_scenarios.py [--procs P] [--N 1000] [--dist a,b]
                                                                [--mmax 100] [--restarts-large R]
                                                                [--width-k K]
For M_max > 16 the population fit runs ONE greedy search at λ = LAM_SMALL up to M_max and keeps the trace;
λ is then raised (×LAM_FACTOR) until the replayed selection stays below M_max, and the sample fits of
that distribution use the same λ.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import time
import numpy as np
from multiprocessing import Pool
import lorentzian_mixture as LM
import unimodal_kalpha as UK

LAM_SMALL, PATIENCE = 1e-7, 3
M_MAX = 16                     # set via --mmax; M_MAX = 16 keeps the original file names
N_RESTARTS, N_RESTARTS_LARGE, LARGE_M = 10, None, 16   # random restarts (for M > LARGE_M: N_RESTARTS_LARGE)
LAM_FACTOR = 2.0               # λ escalation step for the population fit (see choose_lambda)
WIDTH_K = 0                    # width floor for SAMPLE fits: Δ_min = WIDTH_K / (N ρ̂_max), i.e. every
                               # component spans >= WIDTH_K oscillator spacings at the density peak
                               # (ρ̂ = Gaussian KDE of the sample); 0 = off (only the 1e-4 hard bound)
DELTA_MIN_HARD = 1e-4
N_NODES, N_SAMPLES = 5000, 5
OUT = UK.dp.ensure(os.path.join(UK.OUT, "fit_scenarios"))
SCEN = {"pop": "true ρ, no floor", "floor": "sample, floor", "nofloor": "sample, no floor"}


def quantile_nodes(name, n):
    g = np.linspace(-12, 12, 1_200_001)
    p = np.asarray(UK.dists()[name]["pdf"](g), float).ravel()
    F = np.cumsum(p); F /= F[-1]
    return np.interp((np.arange(n) + 0.5) / n, F, g)


def samples(name, N):
    rng = np.random.default_rng(2026)
    return [np.asarray(UK.dists()[name]["sample"](N, rng), float) for _ in range(N_SAMPLES)]


def fpath(name, scen, i, N, mmax=None, width_k=None):
    mmax = M_MAX if mmax is None else mmax
    width_k = WIDTH_K if width_k is None else width_k
    tag = "" if mmax == 16 else f"_mmax{mmax}"
    if width_k and scen != "pop":                    # the width floor only applies to sample fits
        tag += f"_wk{width_k:g}"
    return os.path.join(OUT, f"{name}_{scen}_{i}_N{N}{tag}.npy")


def width_floor(x, k=None):
    """Δ_min = k / (N ρ̂_max): the half-width of every component must span at least k oscillator
    spacings at the peak of the sample density (Gaussian KDE). Never below DELTA_MIN_HARD."""
    k = WIDTH_K if k is None else k
    if not k:
        return DELTA_MIN_HARD
    from scipy.stats import gaussian_kde
    g = np.linspace(x.min(), x.max(), 2001)
    return max(DELTA_MIN_HARD, k / (x.size * float(gaussian_kde(x)(g).max())))


def replay(trace, lam, patience=PATIENCE):
    """LM.fit's selection with floor_c = 0 (no acceptance rule) replayed on a λ-independent trace:
    penalized argmin of D(M) + λ M with patience. Returns the selected trace entry."""
    best, stall = None, 0
    for e in trace:
        total = e["data_loss"] + lam * e["M"]
        if best is None or total < best[1]:
            best, stall = (e, total), 0
        else:
            stall += 1
            if stall >= patience:
                break
    return best[0]


def choose_lambda(trace, mmax):
    """Smallest λ = LAM_SMALL · LAM_FACTOR^k for which the replayed selection stays below M_max."""
    lam = LAM_SMALL
    while replay(trace, lam)["M_nominal"] >= mmax:
        lam *= LAM_FACTOR
    return lam


def _save(out, m, scen, i, N, lam, extra):
    f = dict(w=m.w, Om=m.Omega, De=m.Delta, M=m.M)
    np.save(out, dict(**f, scen=scen, i=i, N=N, lam=lam, M_max=M_MAX, hit_cap=bool(m.M >= M_MAX),
                      K_c0=UK.lmmf_Kc(f, 0.0)[0], G=UK.lmmf_stability_map(f), **extra), allow_pickle=True)


def _job(args):
    name, scen, i, N = args
    out = fpath(name, scen, i, N)
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    common = dict(M_max=M_MAX, patience=PATIENCE, n_restarts=N_RESTARTS, seed=1, method="slsqp",
                  n_restarts_large=N_RESTARTS_LARGE, large_M=LARGE_M)
    if scen == "pop":
        # one λ-independent trace (smallest λ), then the smallest λ whose selection stays below M_max
        r = LM.fit(quantile_nodes(name, N_NODES), (1e-4, 1e2), lambda_M=LAM_SMALL, floor_c=0.0, **common)
        lam = choose_lambda(r["trace"], M_MAX)
        e = replay(r["trace"], lam)
        _save(out, e["model"], scen, i, N, lam,
              dict(data_loss=e["data_loss"], trace_D=np.array([t["data_loss"] for t in r["trace"]]),
                   trace_M=np.array([t["M"] for t in r["trace"]]),
                   trace_time=np.array([t["time"] for t in r["trace"]])))
    else:
        # samples: the same λ as the population fit of this distribution
        lam = float(np.load(fpath(name, "pop", 0, N), allow_pickle=True).item()["lam"])
        x = samples(name, N)[i]
        dmin = width_floor(x)
        r = LM.fit(x, (dmin, 1e2), lambda_M=lam, floor_c=(1.0 if scen == "floor" else 0.0), **common)
        _save(out, r["model"], scen, i, N, lam,
              dict(data_loss=r["data_loss"], floor_reached=r.get("accepted", False), delta_min=dmin,
                   width_k=WIDTH_K))
    return out, time.time() - t0


def _stage(jobs, procs, label):
    t0 = time.time()
    with Pool(procs) as p:
        for k, (out, dt) in enumerate(p.imap_unordered(_job, jobs, chunksize=1), 1):
            el = time.time() - t0
            print(f"[{label}] {k}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; {el / 60:.1f} min elapsed, "
                  f"~{el / k * (len(jobs) - k) / 60:.1f} min left)", flush=True)


def run(procs, N, names=None):
    names = names or list(UK.dists())
    _stage([(n, "pop", 0, N) for n in names], procs, "pop")             # sets λ per distribution
    _stage([(n, s, i, N) for n in names for s in ("floor", "nofloor") for i in range(N_SAMPLES)], procs, "samples")


def figure(N, names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    UK_F = __import__("unimodal_kalpha_figure")
    UK_F.set_prl_style()
    col = {"pop": "#2a78d6", "floor": "#eb6834", "nofloor": "#7b2cbf"}
    names = names or list(UK.dists())
    fig, axs = plt.subplots(1, len(names), figsize=(1.75 * len(names), 2.4), sharey=True, layout="constrained")
    axs = np.atleast_1d(axs)
    print(f"{'dist':8s} {'scenario':18s} {'M':>14s} {'K_c(α=0) err':>16s}   (exact K_c)")
    for ax, name in zip(axs, names):
        ex = np.load(UK._f(name, "boundary_exact") + ".npy"); ex = ex[np.isfinite(ex[:, 0])]
        Kc_ex = ex[np.argmin(np.abs(ex[:, 1])), 0]
        ax.plot(ex[:, 0], np.abs(ex[:, 1]), "--", color="0.1", lw=1.2)
        for scen, n_i, lw, al in (("nofloor", N_SAMPLES, 0.7, 0.7), ("floor", N_SAMPLES, 0.7, 0.8), ("pop", 1, 1.6, 1.0)):
            rs = [np.load(fpath(name, scen, i, N), allow_pickle=True).item() for i in range(n_i)]
            for r in rs:
                ax.contour(UK.K_FINE, UK.A_FINE, r["G"], levels=[0.0], colors=[col[scen]], linewidths=lw, alpha=al)
            Ms = [r["M"] for r in rs]; err = np.array([r["K_c0"] / Kc_ex - 1 for r in rs]) * 100
            cap = " (cap)" if any(r["hit_cap"] for r in rs) else ""
            lam = rs[0].get("lam", LAM_SMALL)
            print(f"{name:8s} {SCEN[scen]:18s} λ={lam:<9.3g} {str(Ms) + cap:>22s} {err.mean():+7.1f}% ± {err.std():4.1f}%   ({Kc_ex:.3f})")
        ax.set_xlim(0, 4); ax.set_ylim(0, 1.45); ax.set_xlabel(r"$K$"); ax.set_title(name, fontsize=7)
    axs[0].set_ylabel(r"phase lag $\alpha$")
    h = [Line2D([], [], color="0.1", ls="--", label=r"exact (true $\rho$)")]
    h += [Line2D([], [], color=col[s], label=SCEN[s] + (f" (N={N})" if s != "pop" else "")) for s in ("pop", "floor", "nofloor")]
    axs[-1].legend(handles=h, fontsize=5, loc="lower right")
    out = os.path.join(UK.OUT, f"fig_fit_scenarios_N{N}" + ("" if M_MAX == 16 else f"_mmax{M_MAX}")
                       + (f"_wk{WIDTH_K:g}" if WIDTH_K else ""))
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg")
    print(f"[saved] {out}.png/.svg")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    N = arg("--N", 1000, int)
    names = arg("--dist", None, lambda s_: s_.split(","))
    M_MAX = arg("--mmax", M_MAX, int)
    WIDTH_K = arg("--width-k", WIDTH_K, float)
    N_RESTARTS_LARGE = arg("--restarts-large", None, int)
    if "--figure" not in sys.argv:
        run(arg("--procs", os.cpu_count(), int), N, names)
    figure(N, names)
