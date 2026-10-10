r"""
Unimodal threshold bias: width floor + Cauchy-transform loss term (population fits).
====================================================================================

Diagnosis (unimodal_threshold_test.py): CvM-fitted Lorentzian mixtures are rippled — every component
creates a local density maximum above the true density — and the incoherent state destabilises
early (K_c too small by 7–24 %). The Cauchy-transform term L_ε = ∫|Φ_N(ε+iν) − Φ_M(ε+iν)|² dν compares
the ε-smoothed densities (and their Hilbert transforms), so it penalises ripples on scales ≳ ε.

Grid: Δ_min ∈ DMINS × ε ∈ EPS × β ∈ BETAS; each refit starts from the CvM fit with the same Δ_min and
keeps its M (kmo_bifurcation_cauchy.fit_cauchy, analytic gradients).
Metrics: K_c error, mean |r_M − r| on [1.05, 3] K_c, D/floor, highest density bump / true maximum.

Two stages, resumable (one result file per job in OUT; existing files are skipped):
  1. base CvM fits, one per (distribution, Δ_min)
  2. Cauchy refits, one per (distribution, Δ_min, ε, β)
then a summary table. Results live on the shared drive (data_paths.MPMF/unimodal_threshold).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_cauchy_test.py [--procs N]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_cauchy_test.py --summary
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import time
import numpy as np
from multiprocessing import Pool
import data_paths as dp
import lorentzian_mixture as LM
import unimodal_threshold_test as U

OUT = dp.ensure(dp.mpmf("unimodal_threshold"))
DMINS = [1e-4, 0.1]
EPS = [0.05, 0.1, 0.2]
BETAS = [1.0, 10.0, 100.0]


def _tag(name, dmin, eps=None, beta=None):
    base = f"{name}_dmin{dmin:g}"
    return base if eps is None else f"{base}_eps{eps:g}_beta{beta:g}"


def _refs():
    """Exact K_c and r(K) on [1.05, 3] K_c, once per distribution."""
    out = {}
    for name, d in U.make_dists().items():
        Kc, _, _ = U.exact_branch(d, np.array([1.0]))
        Ks = Kc * np.linspace(1.05, 3.0, 14)
        out[name] = (Kc, Ks, U.exact_branch(d, Ks)[1])
    return out


def _metrics(d, ref, w, Om, De, x):
    Kc_true, Ks, r_true = ref
    m = LM.LorentzianMixture(w, Om, De)
    xs = np.linspace(-3, 3, 6001)
    xs_s = np.sort(x); u = (np.arange(xs_s.size) + 0.5) / xs_s.size
    D = float(np.mean((m.cdf(xs_s) - u) ** 2))
    return dict(Kc_err=U.lmmf_Kc(w, Om, De) / Kc_true - 1,
                branch_err=float(np.mean(np.abs(U.lmmf_r(w, Om, De, Ks) - r_true))),
                D_floor=D / LM.noise_floor(xs_s.size), bump=float(m.pdf(xs).max() / np.max(d["pdf"](xs))))


def _job(args):
    import kmo_bifurcation_cauchy as BC
    stage, name, dmin, eps, beta, ref = args
    out = os.path.join(OUT, _tag(name, dmin, eps, beta) + ".npy")
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    d = U.make_dists()[name]
    x = U.quantile_nodes(d, U.N)
    if stage == 1:
        r = LM.fit(x, (dmin, 1e2), M_max=16, lambda_M=U.LAM, patience=3, n_restarts=10, seed=1,
                   method="slsqp", floor_c=1.0)
        w, Om, De = r["model"].w, r["model"].Omega, r["model"].Delta
    else:
        b = np.load(os.path.join(OUT, _tag(name, dmin) + ".npy"), allow_pickle=True).item()
        c = BC.fit_cauchy(x, dict(M=b["M"], w=b["w"], Om=b["Om"], De=b["De"]), eps, beta,
                          dbounds=(dmin, 1e2))
        w, Om, De = c["w"], c["Om"], c["De"]
    res = dict(dist=name, dmin=dmin, eps=eps, beta=beta, M=w.size, w=w, Om=Om, De=De,
               **_metrics(d, ref, w, Om, De, x))
    np.save(out, res, allow_pickle=True)
    return out, time.time() - t0


def _run_stage(jobs, procs, label):
    t0 = time.time()
    with Pool(procs) as p:
        for i, (out, dt) in enumerate(p.imap_unordered(_job, jobs, chunksize=1), 1):
            el = time.time() - t0
            print(f"[{label}] {i}/{len(jobs)}  {os.path.basename(out)}  ({dt:.0f}s; {el / 60:.1f} min elapsed, "
                  f"~{el / i * (len(jobs) - i) / 60:.1f} min left)", flush=True)


def summary():
    for name in U.make_dists():
        print(f"\n== {name}")
        for dmin in DMINS:
            for eps, beta in [(None, None)] + [(e, b) for e in EPS for b in BETAS]:
                f = os.path.join(OUT, _tag(name, dmin, eps, beta) + ".npy")
                if not os.path.exists(f):
                    continue
                r = np.load(f, allow_pickle=True).item()
                tag = "CvM only" if eps is None else f"+Cauchy ε={eps:g} β={beta:g}"
                print(f"  Δmin={dmin:<6g} M={r['M']:2d} {tag:24s} K_c err={100 * r['Kc_err']:+6.1f}%  "
                      f"mean|Δr|={r['branch_err']:.4f}  D/floor={r['D_floor']:5.2f}  "
                      f"max bump/true max={r['bump']:.2f}")


def main(procs):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    refs = _refs()
    names = list(U.make_dists())
    _run_stage([(1, n, dm, None, None, refs[n]) for n in names for dm in DMINS], procs, "base")
    _run_stage([(2, n, dm, e, b, refs[n]) for n in names for dm in DMINS for e in EPS for b in BETAS],
               procs, "cauchy")
    summary()


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        main(int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else os.cpu_count())
