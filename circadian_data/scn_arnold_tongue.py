r"""
Arnold tongue of an SCN LMMF model by brute-force (parallel) simulation
======================================================================

Choose (1) the LMMF model, (2) the coupling K, (3) the slice, (4) the grid of driving frequencies and light
amplitudes, (5) the nonlinearity of the driver. Every grid point is simulated; the tongue shows the coherence between
the driver and the model at the driving frequency. The figure also shows three example time series (non-coherent,
transition, coherent).

Model (minimal mean field, global coupling, α = 0; frequency mixture fitted to the slice's POOLED intrinsic
frequencies, scn_pycobi_bifurcation.pooled_mixture):
    ż_m = (iΩ_m − Δ_m) z_m + ½ (H − H̄ z_m²),   H = K Z + I(t),   Z = Σ w_m z_m
    --model floor    noise-floor Lorentzian mixture (free M)
    --model single   one Lorentzian (M = 1)
Driver (Hopf oscillator on its limit cycle, y_d = sin ω_f t) and light (non-negative):
    I(t) = F · ½ [1 + tanh(β y_d) / tanh β]          β → 0: sinusoid F/2 (1 + sin ω_f t); large β: square wave
    --beta B          nonlinearity directly
    --transition-h H  set β such that I rises from 10 % to 90 % in H hours at a 24-h period (overrides --beta)
Coherence at the driving frequency (magnitude-squared coherence, Welch estimate with one-period segments): the
population rhythm x(t) = Re Z(t) = R cos Φ is split into the N_meas forcing periods after the transient;
X_n = Fourier coefficient of x at ω_f over period n, and (the driver being exactly periodic)
    C = |⟨X_n⟩_n|² / ⟨|X_n|²⟩_n ∈ [0, 1]
C = 1 if the rhythm repeats every forcing period (1:1 entrainment), C → 0 if it drifts against the driver. Unlike a
phase-locking value of arg Z this is valid for strong forcing too, where the entrained orbit of Z need not wind
around the origin. Frequency resolution ~ 1 / (N_meas T_f): small detunings look coherent over short windows.
Initial state: the unforced steady state at K (synchronised if K > K_c). Close to the tongue edges the approach to the
entrained state can be very slow (hundreds of periods at weak K): increase --transient if the edge looks ragged.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_arnold_tongue.py --slice 3 --model floor --K 0.05 \
        [--freq 0.8 1.2 --n-freq 81] [--F 0 0.3 --n-F 61] [--transition-h 2 | --beta 4.2] \
        [--transient 300 --measure 50 --steps-per-period 96] [--jobs N] [--example-F 0.1] \
        [--examples "f1,F1;f2,F2;f3,F3"] [--out NAME] [--show]
Frequencies in cycles/day (1/T_f · 24 h), F and K in rad/h.
Output: data_paths.MPMF/circadian_data/arnold_tongue/<name>.{pdf,png,npz}
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import argparse
import multiprocessing as mp
import time
import numpy as np
from scipy.optimize import brentq
import data_paths as dp

OUT = dp.ensure(dp.mpmf("circadian_data", "arnold_tongue"))
KINDS = {"floor": "pfloor", "single": "psingle"}


# ─── model ───────────────────────────────────────────────────────────────────────────────────────
def mixture(slice_id, model):
    import scn_pycobi_bifurcation as B
    mod = B.pooled_mixture(slice_id, KINDS[model])
    w = np.asarray(mod["w"], float)
    return w / w.sum(), np.asarray(mod["Om"], float), np.asarray(mod["De"], float)


def light(F, y, beta):
    if beta < 1e-6:
        return 0.5 * F * (1.0 + y)
    return 0.5 * F * (1.0 + np.tanh(beta * y) / np.tanh(beta))


def beta_from_transition(hours, period=24.0):
    """β such that I rises from 10 % to 90 % of F within `hours` at the given period."""
    def rise(beta):
        s = np.arcsin(np.clip(np.arctanh(0.8 * np.tanh(beta)) / beta, -1, 1)) if beta > 1e-6 else np.arcsin(0.8)
        return 2 * s / (2 * np.pi / period)
    if hours >= rise(1e-7):
        return 0.0                                         # slower than a sinusoid: use the sinusoid
    return brentq(lambda b: rise(b) - hours, 1e-6, 500.0)


def rhs(z, K, I, w, a):
    H = K * (z @ w) + I
    return a * z + 0.5 * (H[..., None] - np.conj(H)[..., None] * z ** 2)


def rk4(z, t, dt, K, F, wf, beta, w, a):
    I0, Ih, I1 = (light(F, np.sin(wf * s), beta) for s in (t, t + 0.5 * dt, t + dt))
    h = np.asarray(dt)[..., None] if np.ndim(dt) else dt
    k1 = rhs(z, K, I0, w, a); k2 = rhs(z + 0.5 * h * k1, K, Ih, w, a)
    k3 = rhs(z + 0.5 * h * k2, K, Ih, w, a); k4 = rhs(z + h * k3, K, I1, w, a)
    return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def unforced(w, Om, De, K, T=4000.0, dt=0.2, T_meas=1000.0):
    """Settled unforced state z (M,) and its collective frequency (rad/h; NaN if incoherent)."""
    a = 1j * Om - De
    z = 0.99 * np.exp(1j * 1e-3 * np.random.default_rng(0).standard_normal(w.size))
    for _ in range(int(T / dt)):
        z = rk4(z, 0.0, dt, K, 0.0, 0.0, 0.0, w, a)
    ph, z0 = 0.0, z
    for _ in range(int(T_meas / dt)):
        z1 = rk4(z0, 0.0, dt, K, 0.0, 0.0, 0.0, w, a); ph += np.angle((z1 @ w) / (z0 @ w)); z0 = z1
    return z, (ph / T_meas if abs(z @ w) > 1e-3 else np.nan)


# ─── brute-force grid (one worker = one chunk of grid points, vectorised) ────────────────────────
def simulate_chunk(args):
    """Coherence C, mean R and mean |Z − Z_strobe drift| for a chunk of (ω_f, F) points."""
    w, Om, De, K, beta, z0, wf, F, n_tr, n_meas, n_per = args
    a = 1j * Om - De; dt = 2 * np.pi / wf / n_per
    z = np.repeat(z0[None], wf.size, axis=0); t = np.zeros(wf.size)
    for _ in range(n_tr * n_per):
        z = rk4(z, t, dt, K, F, wf, beta, w, a); t = t + dt
    ker = np.exp(-2j * np.pi * (np.arange(n_per) + 1) / n_per)       # e^{−iω_f t} at the step ends (t mod T_f)
    X = np.zeros((n_meas, wf.size), complex); Rm = np.zeros(wf.size)
    for n in range(n_meas):
        for k in range(n_per):
            z = rk4(z, t, dt, K, F, wf, beta, w, a); t = t + dt
            Z = z @ w; X[n] += Z.real * ker[k]; Rm += np.abs(Z)
    C = np.abs(X.mean(0)) ** 2 / np.maximum(np.mean(np.abs(X) ** 2, 0), 1e-300)
    return C, Rm / (n_meas * n_per)


def tongue(w, Om, De, K, beta, z0, freqs, Fs, n_tr, n_meas, n_per, jobs):
    FF, ff = np.meshgrid(Fs, freqs, indexing="ij")                  # (n_F, n_freq)
    wf_all, F_all = (2 * np.pi * ff / 24.0).ravel(), FF.ravel()     # cycles/day -> rad/h
    chunks = np.array_split(np.arange(wf_all.size), max(1, jobs * 4))
    tasks = [(w, Om, De, K, beta, z0, wf_all[c], F_all[c], n_tr, n_meas, n_per) for c in chunks if c.size]
    if jobs > 1:
        with mp.get_context("fork").Pool(jobs) as pool:
            res = pool.map(simulate_chunk, tasks)
    else:
        res = list(map(simulate_chunk, tasks))
    C = np.concatenate([r[0] for r in res]).reshape(FF.shape)
    R = np.concatenate([r[1] for r in res]).reshape(FF.shape)
    return C, R


def time_series(w, Om, De, K, beta, z0, f, F, n_tr, n_show, n_per):
    """Trajectory over the last n_show forcing periods after n_tr transient periods."""
    a = 1j * Om - De; wf = 2 * np.pi * f / 24.0; dt = 2 * np.pi / wf / n_per
    z = z0.copy(); t = 0.0
    for _ in range(n_tr * n_per):
        z = rk4(z, t, dt, K, F, wf, beta, w, a); t += dt
    T, Z = [t], [z @ w]
    for _ in range(n_show * n_per):
        z = rk4(z, t, dt, K, F, wf, beta, w, a); t += dt; T.append(t); Z.append(z @ w)
    T = np.array(T)
    return T - T[0], np.array(Z), light(F, np.sin(wf * T), beta)


# ─── examples ────────────────────────────────────────────────────────────────────────────────────
def pick_examples(C, freqs, Fs, f_c, F_ex=None):
    """Three points in one row F: coherent (highest C, closest to the tongue centre), transition (C closest to 0.5
    between the coherent and the non-coherent point), non-coherent (lowest C). Default row: the one closest to the
    middle of the F range that spans the full contrast (min C < 0.2, max C > 0.95)."""
    if F_ex is None:
        ok = np.flatnonzero((C.min(1) < 0.2) & (C.max(1) > 0.95))
        rows = ok if ok.size else np.arange(Fs.size)
        r = int(rows[np.argmin(np.abs(Fs[rows] - 0.5 * (Fs[0] + Fs[-1])))])
    else:
        r = int(np.argmin(np.abs(Fs - F_ex)))
    c = C[r]; centre = f_c if np.isfinite(f_c) else freqs[np.argmax(c)]
    hi = np.flatnonzero(c >= c.max() - 1e-3); j_coh = int(hi[np.argmin(np.abs(freqs[hi] - centre))])
    j_inc = int(np.argmin(c))
    step = 1 if j_inc > j_coh else -1
    seg = np.arange(j_coh + step, j_inc, step)              # strictly between: three distinct points
    j_tr = int(seg[np.argmin(np.abs(c[seg] - 0.5))]) if seg.size else j_inc
    return [("non-coherent", freqs[j_inc], Fs[r]), ("transition", freqs[j_tr], Fs[r]), ("coherent", freqs[j_coh], Fs[r])]


# ─── main ────────────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice", type=int, required=True, help="SCN slice ID (1-5)")
    p.add_argument("--model", choices=tuple(KINDS), default="floor", help="LMMF model")
    p.add_argument("--K", type=float, required=True, help="global coupling (rad/h)")
    p.add_argument("--freq", type=float, nargs=2, default=(0.8, 1.2), help="driving-frequency range (cycles/day)")
    p.add_argument("--n-freq", type=int, default=81)
    p.add_argument("--F", type=float, nargs=2, default=(0.0, 0.3), help="light-amplitude range (rad/h)")
    p.add_argument("--n-F", type=int, default=61)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--beta", type=float, default=4.2, help="driver nonlinearity β (0 = sinusoid)")
    g.add_argument("--transition-h", type=float, help="10-90 %% light transition time at 24 h (sets β)")
    p.add_argument("--transient", type=int, default=300, help="transient forcing periods")
    p.add_argument("--measure", type=int, default=50, help="forcing periods for the coherence")
    p.add_argument("--steps-per-period", type=int, default=96, help="RK4 steps per forcing period")
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel processes")
    p.add_argument("--example-F", type=float,
                   help="light amplitude of the example row (default: mid-range row spanning the full contrast)")
    p.add_argument("--examples", help='three examples "f1,F1;f2,F2;f3,F3" (cycles/day, rad/h) instead of automatic')
    p.add_argument("--show-periods", type=int, default=10, help="forcing periods shown per example")
    p.add_argument("--out", help="output name (default from the parameters)")
    p.add_argument("--show", action="store_true", help="open the figure in a window after saving it")
    a = p.parse_args()

    beta = beta_from_transition(a.transition_h) if a.transition_h is not None else a.beta
    w, Om, De = mixture(a.slice, a.model)
    z0, w_c = unforced(w, Om, De, a.K)
    f_c = w_c * 24 / (2 * np.pi)
    freqs = np.linspace(*a.freq, a.n_freq); Fs = np.linspace(*a.F, a.n_F)
    name = a.out or (f"scn{a.slice}_{a.model}_K{a.K:g}_beta{beta:.2f}")
    print(f"SCN{a.slice} {a.model} (M = {w.size}), K = {a.K}, β = {beta:.2f}; unforced R = {abs(z0 @ w):.3f}, "
          f"f_c = {f_c:.4f} c/d; grid {a.n_F} × {a.n_freq} on {a.jobs} processes ...", flush=True)
    t0 = time.time()
    C, R = tongue(w, Om, De, a.K, beta, z0, freqs, Fs, a.transient, a.measure, a.steps_per_period, a.jobs)
    print(f"  tongue: {time.time() - t0:.0f} s, coherent (C > 0.9) {np.mean(C > 0.9):.0%} of the grid", flush=True)

    if a.examples:
        ex = [(lab, *map(float, s.split(","))) for lab, s in zip(("example 1", "example 2", "example 3"),
                                                                 a.examples.split(";"))]
    else:
        ex = pick_examples(C, freqs, Fs, f_c, a.example_F)
    traj = [time_series(w, Om, De, a.K, beta, z0, f, F, a.transient, a.show_periods, a.steps_per_period)
            for _, f, F in ex]
    np.savez(os.path.join(OUT, name + ".npz"), freqs=freqs, F=Fs, C=C, R=R, K=a.K, beta=beta, slice=a.slice,
             model=a.model, f_c=f_c, w=w, Om=Om, De=De, examples=np.array([(f, F) for _, f, F in ex]))
    figure(a, beta, w, freqs, Fs, C, f_c, ex, traj, name)


def figure(a, beta, w, freqs, Fs, C, f_c, ex, traj, name):
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from prl_style import set_prl_style
    FS = 11                                                # base font size (pt); everything else relative to it
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center", "font.size": FS,
                            "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS - 1,
                            "ytick.labelsize": FS - 1, "legend.fontsize": FS - 1, "lines.linewidth": 1.5,
                            "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
                            "xtick.major.size": 4, "ytick.major.size": 4})
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.35], hspace=0.6, wspace=0.22, left=0.06, right=0.99,
                          top=0.86, bottom=0.17)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.pcolormesh(freqs, Fs, C, cmap="magma", vmin=0, vmax=1, shading="nearest", rasterized=True)
    ax.contour(freqs, Fs, C, levels=[0.5], colors="w", linewidths=1.2, linestyles="--")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"coherence with driver at $f_d$")
    if np.isfinite(f_c):
        ax.plot(f_c, Fs[0], marker="v", color="c", mec="k", mew=0.6, ms=10, clip_on=False, zorder=5)
    cols = ["#4dac26", "#f1a340", "#2c7bb6"]
    for k, (lab, f, F) in enumerate(ex):
        ax.plot(f, F, "o", ms=9, mfc=cols[k], mec="k", mew=1.0, zorder=6)
        ax.annotate("i" * (k + 1), (f, F), xytext=(0, 8), textcoords="offset points", ha="center", color="k",
                    fontsize=FS + 1, weight="bold", zorder=7, path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
    ax.set_xlabel(r"driving frequency $f_d$ (cycles/day)"); ax.set_ylabel(r"light amplitude $F$ (rad/h)")
    top = ax.secondary_xaxis("top", functions=(lambda f: 24 / np.maximum(f, 1e-9), lambda T: 24 / np.maximum(T, 1e-9)))
    top.set_xlabel(r"period $T_d$ (h)")
    model_lab = "noise floor" if a.model == "floor" else "single Lorentzian"
    fig.suptitle(f"SCN{a.slice}, {model_lab} (M = {w.size}),  $K$ = {a.K:g} rad/h,  $\\beta$ = {beta:.2f}",
                 fontsize=FS + 2, weight="bold", x=0.5, y=0.985)
    ax.text(-0.13, 1.1, "a", transform=ax.transAxes, fontsize=FS + 4, weight="bold", va="bottom")
    for k, ((lab, f, F), (T, Z, I)) in enumerate(zip(ex, traj)):
        b = fig.add_subplot(gs[k, 1]); td = T / 24
        on = I > 0.5 * F if F > 0 else np.zeros(T.size, bool)
        b.fill_between(td, 0, 1, where=on, color="#fff2b3", lw=0, transform=b.get_xaxis_transform(), zorder=0)
        b.plot(td, Z.real, color=cols[k], lw=1.8)
        b.plot(td, np.abs(Z), color="0.25", lw=1.3, ls="--")
        b.plot(td, I / max(F, 1e-12) * 0.9 - 1.0 if F > 0 else np.full(T.size, np.nan), color="0.55", lw=1.1)
        jf = np.argmin(np.abs(freqs - f)); jF = np.argmin(np.abs(Fs - F))
        b.set_title(f"{'i' * (k + 1)}  {lab}:  $f_d$ = {f:.3f} c/d ($T_d$ = {24 / f:.2f} h),  $F$ = {F:.3f},  "
                    f"coherence {C[jF, jf]:.2f}", loc="left", fontsize=FS)
        b.set_ylim(-1.1, 1.1); b.set_xlim(0, td[-1])
        b.set_ylabel("rhythm")
        if k == 2:
            b.set_xlabel(f"time (days, after {a.transient} forcing periods)")
            b.legend(handles=[Line2D([], [], color="0.2", lw=1.8, label=r"rhythm $\mathrm{Re}\,Z = R\cos\Phi$"),
                              Line2D([], [], color="0.25", lw=1.3, ls="--", label=r"coherence $R$"),
                              Line2D([], [], color="0.55", lw=1.1, label="light (scaled)"),
                              Patch(fc="#fff2b3", label="light on")],
                     loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=4, frameon=False)
        else:
            b.tick_params(labelbottom=False)
        if k == 0:
            b.text(-0.08, 1.12, "b", transform=b.transAxes, fontsize=FS + 4, weight="bold", va="bottom")
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(OUT, name + ext), dpi=300, bbox_inches="tight")
    print(f"[saved] {os.path.join(OUT, name)}.pdf/.png/.npz")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
