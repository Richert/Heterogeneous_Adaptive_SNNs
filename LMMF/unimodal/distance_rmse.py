r"""
LMMF error as a function of the distance between the Lorentzian mixture and the true distribution.
====================================================================================================

Along the greedy fit trace of the TRUE ρ (quantile nodes, the "true ρ, no floor" population fit of
fit_scenarios.py, M = 1, 2, ..., until patience) every intermediate mixture is an LMMF at a known
CvM distance D(M) from ρ. Each is compared with the infinite-N mean field of ρ itself (the OA equation
on N_CONT quantile nodes), both started from the same partially synchronous state z ≡ R0 (identical
initial condition on the OA manifold: a constant z(ω) continues to z_m = R0 for every component), at
α = ALPHA and three equidistant couplings K = (1 − 2ε, 1 − ε, 1)·K_c(α), K_c(α) the exact threshold of ρ.
Error (--metric): "spectral" (default; the |FFT| amplitude RMSE of R(t) on [0, T] used in the main
unimodal figure, unimodal_kalpha_figure.spectral_rmse) or "time" (RMSE of R(t) over [0, T]).

Markers in the figure: the M that the λ-only rule (floor_c = 0, penalized argmin with patience) would
select on this trace for several λ, and the noise floor D = 1/(6N) (N = 1000), at which the floor rule
stops. (The noise-floor / no-floor sample fits of fit_scenarios.py are simulated too, R_floor / R_nofloor
in the sim files, but no longer drawn.)

SAMPLE MODE (stages sfit, ssim, sfigure): the same procedure for N_TRIALS independent N = 1000 samples
of ρ instead of ρ itself. Every sample gets a full greedy trace (λ = 1e-7, no floor stop, width floor
k = 10, M_max = 64; the settings of the main figure's sample fits), each trace model is simulated against
the true-ρ mean field, and the figure averages over the trials at fixed M: x = mean CvM loss D to the
SAMPLE (the quantity the noise-floor rule tests), y = mean error; only M reached by every trial is shown.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python distance_rmse.py [fit] [sim] [figure] [--dist a,b]
                                                                [--metric spectral|time] [--ncont 5000]
    ... python distance_rmse.py sfit ssim sfigure [--procs P] [--trials 10]
--ncont sets the number of quantile nodes of the true-ρ reference (resolution check; files tagged _n{N}).
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker: the
    os.environ.setdefault(_v, "1")                                         # stages parallelise over jobs
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import time
import numpy as np
from multiprocessing import Pool
from scipy.stats import cramervonmises
import lorentzian_mixture as LM
import unimodal_kalpha as UK
import fit_scenarios as FS

NAMES = ("gauss", "g8", "dgauss")
ALPHA = 0.7                                  # phase lag
KFACS = (0.7, 0.85, 1.0)                     # K / K_c(α); set via --kfacs (files tagged when not the default)
R0, T_END = 0.5, 100.0                       # initial coherence, simulated time
N_CONT, N_NET = 5000, 1000                   # continuum nodes; sample size of the floor / sample fits
MMAX_SAMPLE, WIDTH_K_SAMPLE = 64, 10         # sample-fit set of the main unimodal figure
LAMBDAS = (1e-7, 1e-6, 1e-5, 1e-4)           # λ-only selections marked on the trace
N_TRIALS, SEED_TRIALS = 10, 7                # sample mode: independent N_NET-samples per distribution
OUT = UK.dp.ensure(os.path.join(UK.OUT, "distance_rmse"))


def kc_alpha(name, alpha=None):
    """Exact incoherence threshold of ρ at phase lag α (smallest K crossing on the exact boundary);
    α = 0: K_c = 2 / (π ρ(0)) (Kuramoto, exact for the symmetric unimodal densities used here)."""
    alpha = ALPHA if alpha is None else alpha
    if alpha == 0:
        return float(2.0 / (np.pi * np.asarray(UK.dists()[name]["pdf"](np.array([0.0])), float).ravel()[0]))
    ex = np.load(UK._f(name, "boundary_exact") + ".npy"); ex = ex[np.isfinite(ex[:, 0])]
    K, a = ex[:, 0], np.abs(ex[:, 1]) - alpha
    i = np.where(np.sign(a[:-1]) != np.sign(a[1:]))[0]
    return float(min(K[j] - a[j] * (K[j + 1] - K[j]) / (a[j + 1] - a[j]) for j in i))


def couplings(name):
    return kc_alpha(name) * np.array(KFACS)


def cvm_distance(m, nodes):
    """CvM distance of a mixture to ρ, evaluated on quantile nodes as in the greedy trace."""
    n = nodes.size
    return (float(cramervonmises(nodes, m.cdf).statistic) - 1.0 / (12.0 * n)) / n


# ─── fit: the population trace with every intermediate model ──────────────────────────────────────
def _fit(name):
    out = os.path.join(OUT, f"{name}_trace.npy")
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    nodes = FS.quantile_nodes(name, FS.N_NODES)
    # identical settings to the "pop" scenario of fit_scenarios.py (M_max = 64)
    r = LM.fit(nodes, (1e-4, 1e2), lambda_M=FS.LAM_SMALL, floor_c=0.0, M_max=MMAX_SAMPLE,
               patience=FS.PATIENCE, n_restarts=FS.N_RESTARTS, seed=1, method="slsqp", large_M=FS.LARGE_M)
    tr = [dict(M=e["M"], M_nominal=e["M_nominal"], D=e["data_loss"], w=e["model"].w, Om=e["model"].Omega,
               De=e["model"].Delta) for e in r["trace"]]
    # the sample fits of the main figure, at their distance from ρ
    smp = {}
    for s in ("floor", "nofloor"):
        smp[s] = []
        for i in range(FS.N_SAMPLES):
            f = np.load(FS.fpath(name, s, i, N_NET, MMAX_SAMPLE, WIDTH_K_SAMPLE), allow_pickle=True).item()
            m = LM.LorentzianMixture(f["w"], f["Om"], f["De"])
            smp[s].append(dict(M=f["M"], D=cvm_distance(m, nodes), w=f["w"], Om=f["Om"], De=f["De"]))
    np.save(out, dict(trace=tr, samples=smp), allow_pickle=True)
    return out, time.time() - t0


def run_fits(names, procs):
    with Pool(min(procs, len(names))) as p:
        for out, dt in p.imap_unordered(_fit, names):
            print(f"[fit] {os.path.basename(out)} ({dt:.0f} s)", flush=True)


# ─── sim: true-ρ mean field and every LMMF from z ≡ R0 ────────────────────────────────────────────
def _simulate(f, Kc):
    n_steps, stride = int(round(T_END / UK.DT)), int(round(UK.DTS / UK.DT))
    z0 = np.full((Kc.size, f["M"]), R0 + 0j)
    return UK._rk4_carry(UK._lmmf_rhs(f, Kc[:, None]), z0, n_steps, stride, lambda y: np.abs(y @ f["w"]))


def sim_file(name, n_cont=None):
    n_cont = N_CONT if n_cont is None else n_cont
    return os.path.join(OUT, f"{name}_sim" + ("" if n_cont == 5000 else f"_n{n_cont}") + ".npz")


def _sim(name):
    out = sim_file(name)
    t0 = time.time()
    d = np.load(os.path.join(OUT, f"{name}_trace.npy"), allow_pickle=True).item()
    K = couplings(name); Kc = K * np.exp(-1j * ALPHA)
    nodes = FS.quantile_nodes(name, N_CONT)
    R_ref = _simulate(dict(Om=nodes, De=np.zeros(N_CONT), w=np.full(N_CONT, 1.0 / N_CONT), M=N_CONT), Kc)
    R_tr = np.array([_simulate(dict(e, M=len(e["w"])), Kc) for e in d["trace"]])
    res = dict(K=K, K_c=kc_alpha(name), R_ref=R_ref, R_trace=R_tr, n_cont=N_CONT,
               D_trace=np.array([e["D"] for e in d["trace"]]), M_trace=np.array([e["M"] for e in d["trace"]]))
    for s, fs in d["samples"].items():
        res[f"R_{s}"] = np.array([_simulate(dict(e, M=len(e["w"])), Kc) for e in fs])
        res[f"D_{s}"] = np.array([e["D"] for e in fs])
    np.savez_compressed(out, **res)
    return out, time.time() - t0


def run_sims(names, procs):
    with Pool(min(procs, len(names))) as p:
        for out, dt in p.imap_unordered(_sim, names):
            print(f"[sim] {os.path.basename(out)} ({dt:.0f} s)", flush=True)


# ─── figure ───────────────────────────────────────────────────────────────────────────────────────
def lambda_selection(D, M, lam, patience=FS.PATIENCE):
    """Index of the trace entry the λ-only rule selects (FS.replay on the stored D(M))."""
    tr = [dict(data_loss=d_, M=m_, i=i) for i, (d_, m_) in enumerate(zip(D, M))]
    return FS.replay(tr, lam, patience)["i"]


def error(R, R_ref, metric):
    """Error of LMMF traces R (..., nK, ns) against the reference R_ref (nK, ns) -> (..., nK)."""
    if metric == "time":
        return np.sqrt(((R - R_ref) ** 2).mean(-1))
    import unimodal_kalpha_figure as UF
    return UF.spectral_rmse(R, np.broadcast_to(R_ref, R.shape))


def figure(names, metric="spectral", n_cont=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from prl_style import set_prl_style, panel_label
    import fig_unimodal_combined as FC
    set_prl_style("prl")
    C_K = ["#9ec5ef", "#4a8fd9", "#0d3d7a"]                               # K / K_c = 0.7, 0.85, 1 (light -> dark)
    fig, axs = plt.subplots(len(names), 1, figsize=(3.4, 1.35 * len(names) + 0.55), sharex=True,
                            layout="constrained")
    axs = np.atleast_1d(axs)
    floor = LM.noise_floor(N_NET)
    for r, (ax, name) in enumerate(zip(axs, names)):
        d = np.load(sim_file(name, n_cont))
        D, M, R_ref = d["D_trace"], d["M_trace"], d["R_ref"]
        E = error(d["R_trace"], R_ref, metric)
        for k in range(3):
            ax.loglog(D, E[:, k], "-o", color=C_K[k], lw=0.9, ms=1.8, zorder=3)
        ax.axvline(floor, color="0.3", ls="--", lw=0.8, zorder=1)
        # λ-only selections on the trace: lines with the λ exponents on top (λ that select the same M share one)
        sel = {}
        for lam in LAMBDAS:
            sel.setdefault(lambda_selection(D, M, lam), []).append(int(np.log10(lam)))
        for i, ex in sel.items():
            ax.axvline(D[i], color="0.55", ls=(0, (1, 1.5)), lw=0.6, zorder=1)
            txt = ", ".join(rf"$10^{{{e}}}$" for e in sorted(ex))
            ax.text(D[i], 1.0, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5,
                    color="0.35")
        ax.set_ylabel(FC.label(name) + "\n" + ("spectral RMSE" if metric == "spectral" else "RMSE of $R(t)$"),
                      labelpad=2)
        panel_label(ax, "abc"[r], dx=-28, dy=9, fontsize=8)
    axs[-1].set_xlabel(r"CvM distance $D$ to true $\rho$")
    axs[-1].invert_xaxis()                                               # better fits to the right
    h = [Line2D([], [], color=C_K[k], lw=0.9, label=rf"$K = {f:g}\,K_c$") for k, f in enumerate(KFACS)]
    h += [Line2D([], [], color="0.3", ls="--", lw=0.8, label=rf"noise floor $1/(6N)$, $N={N_NET}$"),
          Line2D([], [], color="0.55", ls=(0, (1, 1.5)), lw=0.6, label=r"no-floor selection ($\lambda$)")]
    fig.legend(handles=h, loc="outside lower center", ncol=2, fontsize=5.5, frameon=False, handlelength=1.6)
    n_cont = N_CONT if n_cont is None else n_cont
    out = os.path.join(UK.OUT, f"fig_distance_rmse_{metric}" + ("" if n_cont == 5000 else f"_n{n_cont}"))
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


# ─── sample mode ──────────────────────────────────────────────────────────────────────────────────
def trial_samples(name, n_trials=None):
    rng = np.random.default_rng(SEED_TRIALS)
    return [np.asarray(UK.dists()[name]["sample"](N_NET, rng), float) for _ in range(n_trials or N_TRIALS)]


def _sfit(args):
    name, i = args
    out = os.path.join(OUT, f"{name}_strial{i}.npy")
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    x = trial_samples(name, i + 1)[i]
    dmin = FS.width_floor(x, WIDTH_K_SAMPLE)
    r = LM.fit(x, (dmin, 1e2), lambda_M=FS.LAM_SMALL, floor_c=0.0, M_max=MMAX_SAMPLE, patience=FS.PATIENCE,
               n_restarts=FS.N_RESTARTS, seed=1, method="slsqp", large_M=FS.LARGE_M)
    nodes = FS.quantile_nodes(name, FS.N_NODES)
    tr = [dict(M=e["M"], M_nominal=e["M_nominal"], D=e["data_loss"], D_true=cvm_distance(e["model"], nodes),
               w=e["model"].w, Om=e["model"].Omega, De=e["model"].Delta, time=e["time"]) for e in r["trace"]]
    np.save(out, dict(trace=tr, delta_min=dmin, i=i), allow_pickle=True)
    return out, time.time() - t0


def run_sfits(names, procs, n_trials):
    jobs = [(n, i) for i in range(n_trials) for n in names]
    t0 = time.time()
    with Pool(min(procs, len(jobs))) as p:
        for k, (out, dt) in enumerate(p.imap_unordered(_sfit, jobs), 1):
            el = time.time() - t0
            print(f"[sfit] {k}/{len(jobs)} {os.path.basename(out)} ({dt / 60:.1f} min; {el / 60:.1f} min elapsed, "
                  f"~{el / k * (len(jobs) - k) / 60:.1f} min left)", flush=True)


def _ssim(args):
    name, n_trials = args
    out = ssim_file(name)
    t0 = time.time()
    K = couplings(name); Kc = K * np.exp(-1j * ALPHA)
    nodes = FS.quantile_nodes(name, N_CONT)
    R_ref = _simulate(dict(Om=nodes, De=np.zeros(N_CONT), w=np.full(N_CONT, 1.0 / N_CONT), M=N_CONT), Kc)
    trials = []
    for i in range(n_trials):
        tr = np.load(os.path.join(OUT, f"{name}_strial{i}.npy"), allow_pickle=True).item()["trace"]
        trials.append(dict(D=np.array([e["D"] for e in tr]), D_true=np.array([e["D_true"] for e in tr]),
                           M=np.array([e["M"] for e in tr]), M_nominal=np.array([e["M_nominal"] for e in tr]),
                           R=np.array([_simulate(dict(e, M=len(e["w"])), Kc) for e in tr])))
    np.save(out, dict(K=K, K_c=kc_alpha(name), R_ref=R_ref, trials=trials), allow_pickle=True)
    return out, time.time() - t0


def _tag():
    return (("" if ALPHA == 0.7 else f"_a{ALPHA:g}")
            + ("" if tuple(KFACS) == (0.7, 0.85, 1.0) else "_k" + "-".join(f"{f:g}" for f in KFACS)))


def ssim_file(name):
    return os.path.join(OUT, f"{name}_ssim{_tag()}.npy")


def run_ssims(names, procs, n_trials):
    with Pool(min(procs, len(names))) as p:
        for out, dt in p.imap_unordered(_ssim, [(n, n_trials) for n in names]):
            print(f"[ssim] {os.path.basename(out)} ({dt:.0f} s)", flush=True)


def sample_figure(names, metric="spectral"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from prl_style import set_prl_style, panel_label
    import fig_unimodal_combined as FC
    set_prl_style("prl")
    C_K = ["#9ec5ef", "#4a8fd9", "#0d3d7a"]
    fig, axs = plt.subplots(len(names), 1, figsize=(3.4, 1.35 * len(names) + 0.45), sharex=True,
                            layout="constrained")
    axs = np.atleast_1d(axs)
    floor = LM.noise_floor(N_NET)
    for r, (ax, name) in enumerate(zip(axs, names)):
        d = np.load(ssim_file(name), allow_pickle=True).item()
        tr, R_ref = d["trials"], d["R_ref"]
        L = min(len(t["D"]) for t in tr)                                  # M reached by every trial
        D = np.exp(np.mean([np.log(t["D"][:L]) for t in tr], 0))          # geometric mean (log axis)
        E = np.mean([error(t["R"][:L], R_ref, metric) for t in tr], 0)    # (L, nK)
        print(f"{name}: {len(tr)} trials, M = 1..{L} shown (trace lengths {[len(t['D']) for t in tr]})")
        for k in range(3):
            ax.loglog(D, E[:, k], "-o", color=C_K[k], lw=0.9, ms=1.8, zorder=3)
        ax.axvline(floor, color="0.3", ls="--", lw=0.8, zorder=1)
        sel = {}
        for lam in LAMBDAS:                                               # median selection over the trials
            i = int(np.median([lambda_selection(t["D"], t["M"], lam) for t in tr]))
            if i < L:
                sel.setdefault(i, []).append(int(np.log10(lam)))
        for i, ex in sel.items():
            ax.axvline(D[i], color="0.55", ls=(0, (1, 1.5)), lw=0.6, zorder=1)
            ax.text(D[i], 1.0, ", ".join(rf"$10^{{{e}}}$" for e in sorted(ex)), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=5, color="0.35")
        ax.set_ylabel(FC.label(name) + "\n" + ("spectral RMSE" if metric == "spectral" else "RMSE of $R(t)$"),
                      labelpad=2)
        panel_label(ax, "abc"[r], dx=-28, dy=9, fontsize=8)
    axs[-1].set_xlabel(rf"CvM loss $D$ to the sample ($N={N_NET}$, mean of {len(tr)} samples)")
    axs[-1].invert_xaxis()
    h = [Line2D([], [], color=C_K[k], lw=0.9, label=rf"$K = {f:g}\,K_c$") for k, f in enumerate(KFACS)]
    h += [Line2D([], [], color="0.3", ls="--", lw=0.8, label=rf"noise floor $1/(6N)$"),
          Line2D([], [], color="0.55", ls=(0, (1, 1.5)), lw=0.6, label=r"no-floor selection ($\lambda$)")]
    fig.legend(handles=h, loc="outside lower center", ncol=3, fontsize=5.5, frameon=False, handlelength=1.6)
    out = os.path.join(UK.OUT, f"fig_distance_rmse_sample_{metric}{_tag()}")
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


def sample_figure_kc(names, metric="spectral"):
    """Single panel: K = K_c(α) only, the distributions colour-coded (sample mode, trial means)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from prl_style import set_prl_style
    import fig_unimodal_combined as FC
    set_prl_style("prl")
    C_D = {"gauss": "#1f5fa8", "g8": "#c8443a", "dgauss": "#2a9d6f"}
    fig, ax = plt.subplots(figsize=(3.4, 2.5), layout="constrained")
    floor = LM.noise_floor(N_NET)
    h = []
    for name in names:
        d = np.load(ssim_file(name), allow_pickle=True).item()
        tr, R_ref = d["trials"], d["R_ref"]
        L = min(len(t["D"]) for t in tr)
        D = np.exp(np.mean([np.log(t["D"][:L]) for t in tr], 0))
        kc = KFACS.index(1.0)                                                            # K = K_c
        E = np.mean([error(t["R"][:L, kc:kc + 1], R_ref[kc:kc + 1], metric)[:, 0] for t in tr], 0)
        fsel = int(np.median([np.argmax(t["D"] <= floor) for t in tr]))
        print(f"{name}: K_c(α={ALPHA:g}) = {d['K_c']:.4f}; M = 1..{L}; error at M = 1, floor stop (M = {fsel + 1}), "
              f"min (M = {E.argmin() + 1}), last: {E[0]:.4f}, {E[fsel]:.4f}, {E.min():.4f}, {E[-1]:.4f}")
        ax.loglog(D, E, "-o", color=C_D[name], lw=0.9, ms=1.8, zorder=3)
        ax.plot(D[fsel], E[fsel], "o", ms=4.5, mfc="none", mec=C_D[name], mew=0.9, zorder=4)
        h.append(Line2D([], [], color=C_D[name], lw=0.9, label=FC.label(name)))
    ax.axvline(floor, color="0.3", ls="--", lw=0.8, zorder=1)
    ax.invert_xaxis()
    ax.set_xlabel(rf"CvM loss $D$ to the sample ($N={N_NET}$, mean of {len(tr)} samples)")
    ax.set_ylabel("spectral RMSE" if metric == "spectral" else "RMSE of $R(t)$")
    ax.set_title(rf"$K = K_c$, $\alpha = {ALPHA:g}$", pad=3)
    h += [Line2D([], [], color="0.3", ls="--", lw=0.8, label=r"noise floor $1/(6N)$"),
          Line2D([], [], ls="none", marker="o", ms=4.5, mfc="none", mec="0.3", label="floor-rule selection")]
    ax.legend(handles=h, fontsize=5.5, frameon=False, loc="best")
    out = os.path.join(UK.OUT, f"fig_distance_rmse_sample_kc_{metric}_a{ALPHA:g}")
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


def sample_figure_bydist(names, metric="spectral"):
    """One panel per K (stacked), the distributions colour-coded (sample mode, trial means)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from prl_style import set_prl_style, panel_label
    import fig_unimodal_combined as FC
    set_prl_style("prl")
    C_D = {"gauss": "#1f5fa8", "g8": "#c8443a", "dgauss": "#2a9d6f"}
    fig, axs = plt.subplots(len(KFACS), 1, figsize=(3.4, 1.3 * len(KFACS) + 0.45), sharex=True,
                            layout="constrained")
    floor = LM.noise_floor(N_NET)
    data = {n: np.load(ssim_file(n), allow_pickle=True).item() for n in names}
    for k, (ax, fk) in enumerate(zip(axs, KFACS)):
        for name in names:
            tr, R_ref = data[name]["trials"], data[name]["R_ref"]
            L = min(len(t["D"]) for t in tr)
            D = np.exp(np.mean([np.log(t["D"][:L]) for t in tr], 0))
            E = np.mean([error(t["R"][:L, k:k + 1], R_ref[k:k + 1], metric)[:, 0] for t in tr], 0)
            fsel = int(np.median([np.argmax(t["D"] <= floor) for t in tr]))
            print(f"K = {fk:g} K_c  {name:7s} ref R(T) = {R_ref[k, -20:].mean():.3f}; error at M = 1, floor stop "
                  f"(M = {fsel + 1}), min (M = {E.argmin() + 1}), last (M = {L}): "
                  f"{E[0]:.4f}, {E[fsel]:.4f}, {E.min():.4f}, {E[-1]:.4f}")
            ax.loglog(D, E, "-o", color=C_D[name], lw=0.9, ms=1.8, zorder=3)
            ax.plot(D[fsel], E[fsel], "o", ms=4.5, mfc="none", mec=C_D[name], mew=0.9, zorder=4)
        ax.axvline(floor, color="0.3", ls="--", lw=0.8, zorder=1)
        ax.set_ylabel(("spectral RMSE" if metric == "spectral" else "RMSE of $R(t)$"), labelpad=2)
        ax.text(0.03, 0.06, rf"$K = {fk:g}\,K_c$", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
        panel_label(ax, "abcdef"[k], dx=-28, dy=3, fontsize=8)
    axs[-1].invert_xaxis()
    axs[-1].set_xlabel(rf"CvM loss $D$ to the sample ($N={N_NET}$, mean of {len(tr)} samples)")
    h = [Line2D([], [], color=C_D[n], lw=0.9, label=FC.label(n)) for n in names]
    h += [Line2D([], [], color="0.3", ls="--", lw=0.8, label=r"noise floor $1/(6N)$"),
          Line2D([], [], ls="none", marker="o", ms=4.5, mfc="none", mec="0.3", label="floor-rule selection")]
    fig.legend(handles=h, loc="outside lower center", ncol=3, fontsize=5.5, frameon=False, handlelength=1.6)
    out = os.path.join(UK.OUT, f"fig_distance_rmse_sample_bydist_{metric}{_tag()}")
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.svg/.pdf")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    names = arg("--dist", list(NAMES), lambda s_: s_.split(","))
    procs = arg("--procs", os.cpu_count(), int)
    N_CONT = arg("--ncont", N_CONT, int)
    ALPHA = arg("--alpha", ALPHA, float)
    KFACS = arg("--kfacs", KFACS, lambda s_: tuple(float(v) for v in s_.split(",")))
    n_trials = arg("--trials", N_TRIALS, int)
    for n in names:
        print(f"{n}: K_c(α={ALPHA:g}) = {kc_alpha(n):.4f}, K = {np.round(couplings(n), 4)}")
    sample_stages = [s for s in ("sfit", "ssim", "sfigure", "sfigure_kc", "sfigure_bydist") if s in sys.argv]
    if sample_stages:
        if "sfit" in sample_stages:
            run_sfits(names, procs, n_trials)
        if "ssim" in sample_stages:
            run_ssims(names, procs, n_trials)
        if "sfigure" in sample_stages:
            sample_figure(names, arg("--metric", "spectral"))
        if "sfigure_bydist" in sample_stages:
            sample_figure_bydist(names, arg("--metric", "spectral"))
        if "sfigure_kc" in sample_stages:
            sample_figure_kc(names, arg("--metric", "spectral"))
        sys.exit(0)
    stages = [s for s in ("fit", "sim", "figure") if s in sys.argv] or ["fit", "sim", "figure"]
    if "fit" in stages:
        run_fits(names, procs)
    if "sim" in stages:
        run_sims(names, procs)
    if "figure" in stages:
        figure(names, arg("--metric", "spectral"), N_CONT)
