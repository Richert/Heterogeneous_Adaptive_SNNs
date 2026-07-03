r"""
Skardal benchmark — SUPPLEMENTARY figure: seed-to-seed finite-size fidelity
===========================================================================

For the near-uniform case (large n, where the finite-N frequency sample departs most from the
ideal g_n), show that the data-driven Lorentzian-mixture ("best fit") reduction follows the
*specific realised* microscopic network, seed by seed, whereas the Skardal reduction tracks the
ideal-g_n / N→∞ limit (and is therefore seed-independent).

Layout (two-column PRL, 3 rows = 3 random seeds; one subfigure per row with a centred (a)/(b)/(c)
title): LEFT = detailed view of the frequency distribution — true g_n(ω), the microscopic
realisation (histogram), and the Lorentzian-mixture fit to that realisation; RIGHT = average
phase-coherence dynamics R(t) (micro vs. Skardal vs. best fit), integrated to T=60.

Run in the ``pycobi`` conda env:
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python skardal_benchmark_supplement.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "grid_search"))
import kmo_lorentzian_fit_sweep as KFS
import skardal_benchmark_simulate as SK
import skardal_benchmark_figure as FIG          # reuse PRL style, colours, fit hyper-parameters

DATA_DIR = "/home/rgast/data/mpmf_simulations"
OUT = os.path.join(DATA_DIR, "skardal_benchmark_supplement")

CONFIG = dict(
    n_values=[1, 100],             # exponents of g_n to render (one separate figure each)
    seeds=[4, 5, 6],               # three independent microscopic realisations (rows = trials)
    Delta=1.0, K=1.2, N=5000, sigma0=0.5,
    T=60.0, dt=1e-2, dts=0.1, rtol=1e-6, atol=1e-8,
)


def xview_for(n, Delta):
    """Half-range of the detailed distribution view: wide for the heavy-tailed small-n (Cauchy at
    n=1), tight for the near-box large-n."""
    return (5.0 if n <= 2 else 1.6) * Delta


def run_one(n, cfg):
    """Generate one supplementary figure (3 trials = seeds) for a given exponent n."""
    Delta, K, N, sigma0 = cfg["Delta"], cfg["K"], cfg["N"], cfg["sigma0"]
    sim = {k: cfg[k] for k in ("T", "dt", "dts", "rtol", "atol")}
    Kc = 2.0 / (np.pi * SK.gn_density(np.array([0.0]), n, Delta)[0])
    print(f"n={n}: K={K}, K_c={Kc:.3f} ({'below' if K < Kc else 'above'} threshold), T={cfg['T']}")

    rows = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        omega = SK.sample_gn(n, Delta, N, rng)
        theta0 = rng.normal(0.0, sigma0, N)
        R0 = float(np.abs(np.exp(1j * theta0).mean()))
        # microscopic + Skardal (ideal g_n) + Lorentzian-mixture ensemble (fit to THIS realisation)
        t_m, R_m = KFS.simulate_micro(omega, K, theta0, sim, tag=f"sup{n}_mic{seed}")
        t_sk, R_sk = SK.simulate_skardal(n, Delta, K, R0, sim, tag=f"sup{n}_sk{seed}")
        res = KFS.LM.fit(omega, FIG.DELTA_BOUNDS, M_max=FIG.M_MAX, alpha=FIG.ALPHA,
                         lambda_M=FIG.LAMBDA_M, patience=FIG.PATIENCE, loss="cvm",
                         n_restarts=FIG.N_RESTARTS, seed=FIG.SEED, method=FIG.FIT_METHOD)
        m, M = res["model"], res["M"]
        t_e, R_e = KFS.simulate_ensemble(m.w, m.Omega, m.Delta, K, R0, sim, tag=f"sup{n}_ens{seed}")
        print(f"  seed={seed}: R0={R0:.3f}, M={M} -> "
              f"micro R(end)={R_m[-1]:.3f}, Skardal={R_sk[-1]:.3f}, best fit={R_e[-1]:.3f}")
        rows.append(dict(seed=seed, omega=omega, m=m, M=M, R0=R0,
                         t_m=t_m, R_m=R_m, t_sk=t_sk, R_sk=R_sk, t_e=t_e, R_e=R_e))

    # ── figure (two-column PRL width) ────────────────────────────────────────
    FIG.set_prl_style()
    xv = xview_for(n, Delta)
    gx = np.linspace(-xv, xv, 2000)
    with np.errstate(over="ignore", invalid="ignore"):
        g_true = SK.gn_density(gx, n, Delta)              # near-box (large n) / Cauchy (n=1) density
    g_true = np.nan_to_num(g_true, nan=0.0, posinf=0.0)
    bw = 2 * xv / 120                                     # bin width; normalise hist to TRUE density
                                                          # (so heavy n=1 tails outside the view don't rescale it)

    fig = plt.figure(figsize=(7.0, 1.65 * len(rows) + 0.5), layout="constrained")
    subfigs = fig.subfigures(len(rows), 1, squeeze=False)[:, 0]
    for i, (sf, r) in enumerate(zip(subfigs, rows)):
        m, M = r["m"], r["M"]; last = (i == len(rows) - 1)
        axL, axR = sf.subplots(1, 2, width_ratios=[1.15, 1.0])
        # LEFT: detailed distribution — realisation (hist), best-fit mixture, true g_n
        comps = m.w[None, :] * (m.Delta[None, :] / np.pi) / ((gx[:, None] - m.Omega[None, :]) ** 2 + m.Delta[None, :] ** 2)
        axL.hist(r["omega"], bins=120, range=(-xv, xv), weights=np.full(r["omega"].size, 1.0 / (r["omega"].size * bw)),
                 color="0.82", edgecolor="none", zorder=0, label="realisation")
        for k in range(M):
            axL.plot(gx, comps[:, k], color=FIG.C_COMP, lw=0.4, alpha=0.45, zorder=1)
        axL.plot(gx, comps.sum(axis=1), color=FIG.C_ENS, lw=1.3, zorder=2, label="best fit")
        axL.plot(gx, g_true, color=FIG.C_SKARDAL, lw=1.1, zorder=3, label=r"$g_n(\omega)$ (true)")
        axL.set_xlim(-xv, xv); axL.set_ylim(0, None); axL.set_ylabel(r"$\rho(\omega)$")
        if last:
            axL.set_xlabel(r"natural frequency $\omega$", labelpad=1)
        if i == 0:
            axL.legend(loc="upper right", fontsize=5.5)
        # RIGHT: phase-coherence dynamics R(t)
        axR.plot(r["t_m"], r["R_m"], color=FIG.C_MICRO, lw=1.0, label="microscopic")
        axR.plot(r["t_sk"], r["R_sk"], color=FIG.C_SKARDAL, lw=1.1, label="Skardal")
        axR.plot(r["t_e"], r["R_e"], color=FIG.C_ENS, lw=1.1, ls="--", label="best fit")
        axR.set_xlim(0, cfg["T"]); axR.set_ylim(-0.02, 1.02); axR.set_ylabel(r"$R(t)$")
        if last:
            axR.set_xlabel(r"time $t$", labelpad=1)
        if i == 0:
            axR.legend(loc="upper right", fontsize=5.5)
        sf.suptitle(rf"({'abc'[i]})  trial {i + 1}:  best fit $M={M}$  (Skardal {n} eqs)", fontsize=8)

    fig.suptitle(rf"Finite size bias for $n={n}$", fontsize=8.5)
    out = f"{OUT}_n{n}"
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    np.savez(out + ".npz", n=np.int64(n), seeds=np.array(cfg["seeds"]),
             **{f"omega_{r['seed']}": r["omega"] for r in rows},
             **{f"R_micro_{r['seed']}": r["R_m"] for r in rows},
             **{f"R_skardal_{r['seed']}": r["R_sk"] for r in rows},
             **{f"R_bestfit_{r['seed']}": r["R_e"] for r in rows},
             **{f"M_{r['seed']}": np.int64(r["M"]) for r in rows},
             t=rows[0]["t_m"], K=float(K), Delta=float(Delta), N=np.int64(N), T=float(cfg["T"]))
    plt.close(fig)
    print(f"[saved] {os.path.basename(out)}.{{png,pdf,npz}}")


def main():
    for n in CONFIG["n_values"]:
        run_one(n, CONFIG)


if __name__ == "__main__":
    main()
