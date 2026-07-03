r"""
2D bifurcation diagram of the adaptive Kuramoto–OA ensemble model in (γ, Δ)
===========================================================================

Companion to ``kmo_adaptive_1d.py``. For the multistable M=5 regime the 1D
diagram (in the weight decay γ at fixed Δ) revealed one connected limit-cycle
branch with three organizing codim-1 bifurcations:

  * SNIC               — the stable node + saddle collide on the invariant
                         circle (saddle-node of equilibria); the STABLE limit
                         cycle is born here with infinite period.   (γ_c≈0.170)
  * fold-of-cycles     — the stable LC and an unstable LC annihilate
                         (saddle-node of periodic orbits); beyond it only a
                         2-torus remains.                            (γ≈0.220)
  * saddle-loop        — the UNSTABLE LC terminates in a homoclinic orbit to the
    (homoclinic)         regular saddle (infinite period).            (γ≈0.099)

This script continues all three loci in the (γ, Δ) plane and overlays them.

How each curve is computed
--------------------------
  * SNIC          : 2-parameter continuation of the equilibrium fold (LP) —
                    ``IPS=1, ISW=2, ICP=[g, Δ]``. Because the cycle is born on
                    this fold, the saddle-node-of-equilibria curve *is* the SNIC
                    locus.
  * fold-of-cycles: 2-parameter continuation of the limit-cycle fold (LP) —
                    ``IPS=2, ISW=2, ICP=[g, Δ, 11]`` (PAR(11)=period carried).
  * saddle-loop   : CONSTANT-(large-)PERIOD continuation. A true homoclinic
                    (HomCont, IPS=9) is intractable here: at reduced dimension
                    2M−1+M²=34 the saddle's stable manifold is 33-dimensional and
                    the HomCont projection boundary conditions will not march
                    (it works at M=3 / NSTAB=13, not at M=5 / NSTAB=33). Instead
                    we pin a near-homoclinic orbit's period at a large value
                    (``SL_PERIOD``) and continue it in (γ, Δ) with PAR(11) held
                    fixed (``ICP=[g, Δ]``, period NOT in ICP). At large period
                    this curve traces the homoclinic locus; ``SL_PERIOD`` can be
                    raised to confirm convergence.

The ASYNCHRONOUS fixed point (z_i = 0, A_ij = 0 — weights decay, populations
decohere) is NOT continued here, and on purpose: it is a singularity of the
co-rotating reduced coordinates (Ω = Im F_0/x_0 → 0/0 at x_0 = r_0 = 0), AND it
turns out to be linearly stable for ALL parameters. Its Jacobian (computed in the
full, non-co-rotating OA model by ``async_fp_eigenvalues`` below) is block
diagonal with eigenvalues −Δ_i ± iω_i (order-parameter modes) and −γ (weight
modes): the coupling term linearizes to zero at z=A=0, so there is no
stability-loss bifurcation of the async state — it COEXISTS with the synchronous
FP / limit cycle / torus everywhere (the model is multistable). The curves below
therefore bound the *existence* regions of the other attractors on top of an
always-stable asynchronous background, rather than partitioning the plane.

Run inside the ``pycobi`` conda env with meson/ninja on PATH:
    PATH="$CONDA_PREFIX/bin:$PATH" python kmo_adaptive_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pycobi import ODESystem
from pycobi.utility import write_auto_dat

# reuse the model, helpers and analytical-Jacobian stability from the 1D script
import kmo_adaptive_1d as K
from kmo_adaptive_1d import (
    AUTO_DIR, PAR_G, PAR_DELTA, PAR_PERIOD,
    build_circuit_cartesian, make_jacobian_eval, initialize,
    settle_and_classify, recompute_equilibrium_stability,
    _pcol, _fold_gammas,
)

# colours
C_SNIC = "#1F77B4"     # saddle-node of equilibria (= SNIC)
C_FOC = "#2CA02C"      # fold-of-cycles (saddle-node of periodic orbits)
C_SL = "#D62728"       # saddle-loop homoclinic

CONFIG = dict(
    M=5, K=1.0, mu=0.1, delta=0.02,
    gamma_locked=0.05, gamma_cycle=0.20,
    omega_mean=0.40, omega_std=0.20, r0_mean=0.90, r0_std=0.10, A0_scale=0.50,
    seed=42,
    SL_PERIOD=500.0,        # fixed period for the saddle-loop (homoclinic) proxy
    delta_min=1e-3, delta_max=0.12,   # Δ window for the 2D curves
)


def async_fp_eigenvalues(M, K, mu, delta, omega, gamma):
    """Jacobian eigenvalues of the ASYNCHRONOUS fixed point (z_i=0, A_ij=0) in the
    full, non-co-rotating OA model. Returns the eigenvalue array.

    This FP cannot be represented in the co-rotating reduced model (Ω=Im F_0/x_0
    is singular at x_0=0), so we evaluate the full-model Jacobian directly. The
    result is always {−Δ_i ± iω_i (order-parameter modes), −γ (weight modes)} —
    all with negative real part for Δ, γ > 0 — i.e. the async state is linearly
    stable for every parameter value and undergoes no stability-loss bifurcation."""
    n = 2 * M + M * M

    def rhs(s):
        x = s[0:2 * M:2]; y = s[1:2 * M:2]; A = s[2 * M:].reshape(M, M)
        z = x + 1j * y
        H = (K / M) * (A @ z)
        F = (-delta + 1j * omega) * z + 0.5 * (H - z ** 2 * np.conj(H))
        dA = mu * (np.outer(x, x) + np.outer(y, y)) - gamma * A
        out = np.empty_like(s)
        out[0:2 * M:2] = F.real; out[1:2 * M:2] = F.imag; out[2 * M:] = dA.ravel()
        return out

    s0 = np.zeros(n)
    f0 = rhs(s0)
    J = np.empty((n, n))
    for k in range(n):
        sp = s0.copy(); sp[k] += 1e-7
        J[:, k] = (rhs(sp) - f0) / 1e-7
    return np.linalg.eigvals(J)


def main():
    cfg = CONFIG
    M = cfg["M"]
    Kc, mu, delta = cfg["K"], cfg["mu"], cfg["delta"]
    DIM = 2 * M - 1 + M * M
    NPAR = max(11, 2 * M)
    dmin, dmax = cfg["delta_min"], cfg["delta_max"]

    omega, r0, phi0, A0 = initialize(
        M, cfg["omega_mean"], cfg["omega_std"], cfg["r0_mean"], cfg["r0_std"],
        cfg["A0_scale"], cfg["seed"])
    print(f"KMO adaptive — 2D (γ, Δ) bifurcation diagram — M={M} (dim {DIM})")

    # Asynchronous fixed point (z=0, A=0): always linearly stable, no bifurcation.
    print("\n[0] asynchronous fixed point (z=0, A=0) stability check (full model):")
    for gtest in (0.05, 0.20, 0.60):
        ev = async_fp_eigenvalues(M, Kc, mu, delta, omega, gtest)
        print(f"    γ={gtest:4.2f}: max Re(eig) = {ev.real.max():+.4f}  "
              f"(modes at −Δ={-delta:+.3f} and −γ={-gtest:+.3f}) → "
              f"{'STABLE' if ev.real.max() < 0 else 'UNSTABLE'}")
    print("    ⇒ asynchronous state is stable for ALL γ, Δ — it coexists with the "
          "other attractors (no stability-loss locus to draw).")

    jac = make_jacobian_eval(M, Kc, mu, delta, omega, r0, phi0, A0)
    circuit = build_circuit_cartesian(M, Kc, mu, cfg["gamma_locked"], delta,
                                      omega, r0, phi0, A0)
    ode = ODESystem.from_template(circuit, auto_dir=AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True,
                                  auto_constants=("ivp", "eq", "lc"))

    # ── IVP → equilibrium continuation in γ → fold (SNIC) ───────────────────
    print("\n[1] IVP + equilibrium continuation in γ")
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=200000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: 3000.0}, STOP={"UZ1"})
    eq_sols, _ = ode.run(
        origin="time", starting_point="UZ1", name="eq", c="eq", ICP="g",
        bidirectional=True, RL0=0.0, RL1=1.0, IPS=1, ILP=1, ISP=2, ISW=1,
        NMX=8000, NPR=200, DS=1e-3, DSMIN=1e-10, DSMAX=2e-2,
        EPSL=1e-7, EPSU=1e-7, EPSS=1e-5)
    gamma_c = _fold_gammas(eq_sols)[0]
    print(f"  fold/SNIC at γ_c = {gamma_c:.4f}")

    # ── Curve 1: SNIC = saddle-node of equilibria in (γ, Δ) ─────────────────
    print("\n[2] SNIC curve: 2-parameter continuation of the equilibrium fold")
    snic_curve = None
    try:
        ode.run(origin="eq", starting_point="LP1", name="snic", c="eq",
                ICP=["g", "delta"], bidirectional=True, IPS=1, ISW=2, ISP=2, ILP=0,
                RL0=0.0, RL1=1.5, NMX=6000, NPR=200,
                DS=1e-3, DSMIN=1e-9, DSMAX=2e-2, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
                UZSTOP={"delta": [dmin, dmax + 0.05]})
        snic_curve = "snic"
        print("  SNIC curve traced")
    except Exception as e:
        print(f"  SNIC curve failed: {type(e).__name__}: {e}")

    # ── Limit cycle: from γ=0.20 onto the UNSTABLE arm (DS>0 rounds the
    #    fold-of-cycles at ~0.22, then heads to the saddle-loop near γ≈0.099).
    #    One run yields BOTH the fold-of-cycles LP and a large-period UZ. ─────
    print("\n[3] limit-cycle continuation onto the unstable arm")
    _, _, orbit_df, _, _ = settle_and_classify(
        M, Kc, mu, cfg["gamma_cycle"], delta, omega, r0, phi0, A0)
    write_auto_dat(orbit_df, "kmo2d_seed.dat", normalize_time=False)
    lc_sols, _ = ode.run(
        name="lc_un", dat="kmo2d_seed", c="lc", NDIM=DIM, NPAR=NPAR,
        PAR={PAR_G: cfg["gamma_cycle"]}, IPS=2, ISP=2, ILP=1, ISW=1,
        ICP=[PAR_G, PAR_PERIOD], NTST=300, NCOL=4, RL0=0.0, RL1=0.6,
        NMX=40000, NPR=300, DS=1e-3, DSMIN=1e-12, DSMAX=3e-3,
        EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, THL={PAR_PERIOD: 0.0},
        UZR={PAR_PERIOD: [cfg["SL_PERIOD"]]}, UZSTOP={PAR_PERIOD: 1.4 * cfg["SL_PERIOD"]},
        get_period=True)
    foc_gammas = _fold_gammas(lc_sols)
    print(f"  fold-of-cycles (LP) at γ = {[round(x, 4) for x in foc_gammas]}")
    n_uz = int((lc_sols[("bifurcation", "")] == "UZ").sum())
    print(f"  planted {n_uz} large-period UZ on the unstable arm")

    # ── Curve 2: fold-of-cycles in (γ, Δ) ───────────────────────────────────
    print("\n[4] fold-of-cycles curve: 2-parameter continuation of the LC fold")
    foc_curves = []
    n_lp = int((lc_sols[("bifurcation", "")] == "LP").sum())
    for i in range(1, n_lp + 1):
        sp = f"LP{i}"
        try:
            # ISP=1 (no codim-2 detection — too costly at this dim); period
            # carried as the 3rd ICP entry.
            ode.run(origin="lc_un", starting_point=sp, name=f"foc_{sp}", c="lc",
                    NDIM=DIM, NPAR=NPAR, IPS=2, ISW=2, ISP=1, ILP=0,
                    ICP=[PAR_G, PAR_DELTA, PAR_PERIOD], NTST=100, NCOL=4,
                    RL0=0.0, RL1=1.0, NMX=3000, NPR=100,
                    DS=1e-3, DSMIN=1e-9, DSMAX=1e-2, THL={PAR_PERIOD: 0.0},
                    EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, bidirectional=True,
                    UZSTOP={PAR_DELTA: [dmin, dmax + 0.05]})
            foc_curves.append(f"foc_{sp}")
            print(f"  fold-of-cycles curve from {sp} traced")
        except Exception as e:
            print(f"  fold-of-cycles from {sp} skipped: {type(e).__name__}: {str(e)[:60]}")

    # ── Curve 3: saddle-loop locus via constant-(large-)period continuation ─
    print("\n[5] saddle-loop curve: constant-period continuation (period fixed)")
    sl_curve = None
    if n_uz:
        try:
            # PAR(11) is NOT in ICP, so the period stays pinned at SL_PERIOD; the
            # near-homoclinic orbit is continued in (γ, Δ) → homoclinic locus.
            sl, _ = ode.run(
                origin="lc_un", starting_point="UZ1", name="saddleloop", c="lc",
                NDIM=DIM, NPAR=NPAR, IPS=2, ISP=1, ILP=0, ISW=1,
                ICP=[PAR_G, PAR_DELTA], NTST=300, NCOL=4, RL0=0.0, RL1=0.6,
                NMX=6000, NPR=100, DS=1e-3, DSMIN=1e-10, DSMAX=5e-3,
                EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, THL={PAR_PERIOD: 0.0},
                bidirectional=True, UZSTOP={PAR_DELTA: [dmin, dmax + 0.05]},
                get_period=True)
            sl_curve = "saddleloop"
            gv = np.asarray(sl[_pcol(sl, "g")], float)
            dv = np.asarray(sl[_pcol(sl, "delta")], float)
            print(f"  saddle-loop curve (period {cfg['SL_PERIOD']:.0f}): "
                  f"γ∈[{gv.min():.4f},{gv.max():.4f}] Δ∈[{dv.min():.4f},{dv.max():.4f}]")
        except Exception as e:
            print(f"  saddle-loop curve failed: {type(e).__name__}: {str(e)[:60]}")

    # ── Plot the 2D bifurcation diagram ─────────────────────────────────────
    print("\n[plot] assembling 2D (γ, Δ) diagram")
    fig, ax = plt.subplots(figsize=(8.5, 6))

    def _plot(curve, color, label):
        if curve is None:
            return
        try:
            ode.plot_continuation("g", "delta", cont=curve, ax=ax,
                                  line_color_stable=color, line_color_unstable=color,
                                  line_style_stable="solid", line_style_unstable="solid",
                                  get_stability=False, bifurcation_legend=False,
                                  ignore=["UZ", "EP", "RG", "BP", "LP", "HB"])
        except Exception as e:
            print(f"  plot {curve} partial: {e}")

    _plot(snic_curve, C_SNIC, "SNIC")
    for c in foc_curves:
        _plot(c, C_FOC, "fold-of-cycles")
    _plot(sl_curve, C_SL, "saddle-loop")

    # reference markers at the working slice Δ = delta (the 1D diagram values)
    ax.axhline(delta, color="0.7", lw=0.8, ls=":", zorder=0)
    ax.scatter([gamma_c], [delta], marker="o", s=70, facecolors="none",
               edgecolors=C_SNIC, linewidths=1.8, zorder=12)
    if foc_gammas:
        ax.scatter([max(foc_gammas)], [delta], marker="^", s=80, facecolors="none",
                   edgecolors=C_FOC, linewidths=1.8, zorder=12)

    # region labels — each "+ async" because the asynchronous FP is stable
    # throughout (see [0]); these mark where the OTHER attractor additionally
    # exists, on top of the always-stable asynchronous background.
    tprops = dict(fontsize=8, ha="center", va="center", style="italic", color="0.25")
    ax.text(0.085, 0.085, "synchronous FP\n(+ async)", **tprops)
    ax.text(0.245, 0.055, "torus\n(+ async)", **tprops)
    ax.annotate("stable limit cycle\n(+ async)", xy=(0.195, 0.012), xytext=(0.265, 0.018),
                fontsize=8, style="italic", color="0.25", ha="center",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.text(0.5 * (0.05 + 0.29), dmax * 0.96,
            "asynchronous FP (z=0, A=0): linearly stable everywhere  (eigenvalues −Δ, −γ)",
            fontsize=8, ha="center", va="top", color=C_SL,
            bbox=dict(boxstyle="round", fc="white", ec=C_SL, alpha=0.8))

    ax.set_xlabel(r"weight decay $\gamma$")
    ax.set_ylabel(r"OA half-width $\Delta$")
    ax.set_ylim(0.0, dmax)
    ax.set_title(f"2D bifurcation diagram in $(\\gamma, \\Delta)$  (M={M})")
    legend = [
        Line2D([0], [0], color=C_SNIC, lw=2, label="SNIC (saddle-node of equilibria)"),
        Line2D([0], [0], color=C_FOC, lw=2, label="fold-of-cycles (saddle-node of LCs)"),
        Line2D([0], [0], color=C_SL, lw=2,
               label=f"saddle-loop (homoclinic, period≈{cfg['SL_PERIOD']:.0f})"),
        Line2D([0], [0], color="0.7", lw=0.8, ls=":", label=f"Δ={delta} (1D slice)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("kmo_adaptive_2d.png", dpi=140, bbox_inches="tight")
    print("  saved kmo_adaptive_2d.png")

    ode.close_session(clear_files=True)
    print("Done.")


if __name__ == "__main__":
    main()
    plt.show()
