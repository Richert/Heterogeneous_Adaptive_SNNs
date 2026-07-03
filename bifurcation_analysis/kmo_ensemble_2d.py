r"""
2D fixed-point bifurcation diagram of the fitted ensemble adaptive-Kuramoto model
=================================================================================

The heterogeneous (weighted-Lorentzian) ensemble model is FIXED-POINT based: as
the weight decay γ increases, the synchronized relative equilibrium loses
stability at a saddle-node (fold) and the system drops to a *different*, lower-
coherence equilibrium (the branches are NOT saddle-node-connected — they are
separate branches), in a cascade that ends with a collapse to the asynchronous
state (all weights → 0, r → 0). There is no limit cycle / fold-of-cycles here
(verified by simulation), so the whole 2D diagram is built from equilibrium
continuation.

Pipeline
--------
  1. Multi-branch 1D tracer (in γ at K = K_anchor): settle onto a stable
     equilibrium by simulation, continue it in γ, locate the fold (LP)/Hopf (HB)
     where it loses stability, then NUMERICALLY find the next stable equilibrium
     just past that bifurcation (a single simulation) and restart the
     continuation from there. Repeat until that single confirmation simulation
     lands on the asynchronous state instead of another equilibrium.
  2. For every 1D bifurcation point found, continue it in two parameters (γ, K)
     → the fold/Hopf LOCI.
  3. The FINAL fold — the one past which the confirmation simulation goes async —
     IS the collapse-to-asynchronous boundary (the last coherent equilibrium
     disappears there, leaving only the always-stable z=0, A=0 state). Its (γ, K)
     locus is therefore the async boundary; no separate parameter scan is needed,
     just the one confirmation simulation per step.
  4. Labelled 2D (γ, K) diagram: intermediate fold loci + the terminal
     (collapse-to-async) fold locus + any Hopf loci.

Run inside the ``pycobi`` conda env (loader is standalone np.load):
    PATH="$CONDA_PREFIX/bin:$PATH" python kmo_ensemble_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from pyrates import clear
from pycobi import ODESystem

import kmo_ensemble_bifurcation as B   # model builders, Jacobian, stability, loader

_EPS = 1e-12
C_FOLD = "#1F77B4"
C_HOPF = "#9467BD"
C_ASYNC = "#D62728"


def set_prl_style():
    """Physical Review Letters figure rcParams: serif/STIX text, ~8 pt, thin
    axes, ticks in, vector (Type-42) fonts for the PDF."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 140,
    })


CONFIG = dict(
    params_npz="../kuramoto/oa_params_M5.npz",
    K_anchor=2.0,                      # K at which the 1D γ-cascade is traced
    gamma_min=0.0, gamma_max=0.06,     # γ-range for the 1D continuations
    K_min=0.01, K_max=3.0,              # K-range for the 2D loci
    MAX_BRANCHES=5,
    reseed_offset=0.004,               # how far past a bifurcation to look for the next FP
    r0_start=0.9, T_settle=4000.0,
    async_tol=0.10,                    # mean r below this ⇒ asynchronous
    coh_tol=0.30,                      # r_i above this ⇒ that ensemble counts as coherent
)


# ═════════════════════════════════════════════════════════════════════════════
#  numpy lab-frame weighted-OA RHS  →  settle onto a stable equilibrium
# ═════════════════════════════════════════════════════════════════════════════
def make_oa_rhs(M, mu, omega, delta, weights, plasticity):
    hebbian = (plasticity == "hebbian")

    def rhs(t, y, gamma, Kc):
        r = np.clip(y[:M], _EPS, 1.0 - _EPS)
        psi = y[M:2 * M]
        A = y[2 * M:].reshape(M, M)
        dp = psi[None, :] - psi[:, None]
        Ar = A * r[None, :]
        dr = -delta * r + 0.5 * (1.0 - r ** 2) * Kc * ((Ar * np.cos(dp)) @ weights)
        dps = omega + 0.5 * (1.0 + r ** 2) / r * Kc * ((Ar * np.sin(dp)) @ weights)
        drive = np.cos(dp) if hebbian else np.sin(dp)
        dA = mu * (r[:, None] * r[None, :]) * drive - gamma * A
        return np.concatenate([dr, dps, dA.ravel()])

    return rhs


def settle_fp(rhs, M, gamma, Kc, r0_start, T_settle, async_tol):
    """Integrate from a synchronized IC to a steady state. Returns
    (status, r, psi, A); status in {'fp', 'async', 'oscillatory'}."""
    y0 = np.concatenate([np.full(M, r0_start), np.zeros(M), np.ones(M * M)])
    sol = solve_ivp(rhs, (0.0, T_settle), y0, args=(gamma, Kc), method="RK45",
                    rtol=1e-9, atol=1e-11, dense_output=True)
    tt = np.linspace(0.85 * T_settle, T_settle, 4000)
    Y = sol.sol(tt)
    r = Y[:M]
    if r.mean() < async_tol:
        return "async", None, None, None
    if float((r.max(axis=1) - r.min(axis=1)).max()) > 1e-3:
        return "oscillatory", None, None, None
    return "fp", Y[:M, -1], Y[M:2 * M, -1], Y[2 * M:, -1].reshape(M, M)


# ═════════════════════════════════════════════════════════════════════════════
#  small helpers
# ═════════════════════════════════════════════════════════════════════════════
def _curve_arrays(sols, px, py):
    return (np.asarray(sols[B._pcol(sols, px)], float),
            np.asarray(sols[B._pcol(sols, py)], float))


def _state_arr(sols, name):
    """1D array of a state variable (or parameter) column, by short name."""
    return np.asarray(sols[B._pcol(sols, name)], float).ravel()


def _bif_xy(sols, label, px, py):
    """(px, py) coordinates of every row flagged with the given bifurcation
    label (e.g. 'LP', 'HB')."""
    px_col, py_col = B._pcol(sols, px), B._pcol(sols, py)
    rows = sols[sols[("bifurcation", "")] == label]
    return [(float(a), float(b)) for a, b in zip(rows[px_col], rows[py_col])]


def _dedupe_xy(points, tol=2e-3):
    """Collapse near-coincident (x, y) points. Used to discard the spurious,
    overlapping BP branch-switch continuations Auto emits (they duplicate the
    primary branch in (γ, x_0)), and to avoid double bifurcation markers from the
    bidirectional run."""
    out = []
    for x, y in points:
        if not any(abs(x - a) < tol and abs(y - b) < tol for a, b in out):
            out.append((x, y))
    return out


def _clean_path(g, x, stab, tol=2e-3):
    """Drop points that coincide (in (γ, x_0)) with an earlier, non-adjacent
    point — i.e. the spurious BP branch-switch overlay — while preserving the
    natural continuation (path) order of the primary branch. A genuine fold
    revisits a γ at a *different* x_0, so it survives."""
    g = np.asarray(g, float); x = np.asarray(x, float)
    stab = np.asarray(stab, bool)
    keep = []
    for i in range(len(g)):
        dup = False
        for j in keep:
            if abs(i - j) > 2 and abs(g[i] - g[j]) < tol and abs(x[i] - x[j]) < tol:
                dup = True
                break
        if not dup:
            keep.append(i)
    keep = np.array(keep, int)
    return g[keep], x[keep], stab[keep]


def _plot_stab(ax, g, x, stab, color, lw=1.2, zorder=3, jump=0.25):
    """Plot a continuation branch IN PATH (continuation) ORDER — never sorted by
    γ, which would connect the lower and upper fold branches with a spurious
    vertical segment. Solid = stable, dashed = unstable. The polyline is broken
    wherever consecutive points jump (the bidirectional seam / a disconnected
    spurious segment), measured in axis-normalised distance."""
    g, x, stab = _clean_path(g, x, stab)
    n = len(g)
    if n < 2:
        return
    gr = max(g.max() - g.min(), 1e-9)
    xr = max(x.max() - x.min(), 1e-9)
    d = np.sqrt((np.diff(g) / gr) ** 2 + (np.diff(x) / xr) ** 2)
    pieces = np.split(np.arange(n), np.where(d > jump)[0] + 1)
    for piece in pieces:
        if len(piece) < 2:
            continue
        s = stab[piece]
        cut = np.where(np.diff(s.astype(int)) != 0)[0] + 1
        for seg in np.split(np.arange(len(piece)), cut):
            if len(seg) < 2:
                continue
            idx = piece[seg]
            ls = "-" if s[seg[0]] else "--"
            ax.plot(g[idx], x[idx], ls=ls, color=color, lw=lw, zorder=zorder)


def _interp_gamma_of_K(fold, Kgrid):
    """Interpolate a fold/Hopf locus γ(K) onto a common K grid; NaN outside the
    locus' own K-range so fill_betweenx leaves undefined K's blank."""
    g, K = np.asarray(fold["g"], float), np.asarray(fold["K"], float)
    o = np.argsort(K)
    K, g = K[o], g[o]
    K, idx = np.unique(K, return_index=True)
    g = g[idx]
    out = np.interp(Kgrid, K, g, left=np.nan, right=np.nan)
    out[(Kgrid < K[0]) | (Kgrid > K[-1])] = np.nan
    return out


def _dedupe(vals, tol=2e-3):
    """Collapse near-duplicate bifurcation γ's (the bidirectional run can flag the
    same fold twice)."""
    out = []
    for v in sorted(vals):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    p = B.load_meanfield_params(cfg["params_npz"])
    M = p["M"]
    K0, mu = cfg["K_anchor"], p["mu"]
    gamma = p["gamma"]
    omega, delta, weights = p["omega"], p["delta"], p["weights"]
    plasticity = p["plasticity"]
    DIM = 2 * M - 1 + M * M

    # Reorder ensembles so the co-rotating reference (index 0) is the HIGHEST-WEIGHT
    # ensemble. In the heterogeneous model the off-centre, low-weight ensembles
    # desynchronise first (r_i → 0); pinning the frame (Ω = Im F_0/x_0) to one of
    # those makes it singular on the lower branches. The highest-weight, most
    # central ensemble stays coherent longest, keeping x_0 > 0 along the cascade.
    # (Relabelling is exact — the model is equivariant under ensemble permutation.)
    order = np.argsort(weights)[::-1]
    omega, delta, weights = omega[order], delta[order], weights[order]
    print(f"Ensemble 2D bifurcation — M={M} (dim {DIM}), {plasticity}, "
          f"K_anchor={K0}, mu={mu}")
    print(f"  reordered (reference = highest-weight ensemble): "
          f"w0={weights[0]:.3f}, ω0={omega[0]:+.3f}")

    rhs = make_oa_rhs(M, mu, omega, delta, weights, plasticity)
    jac = B.make_jacobian_eval(M, K0, mu, omega, delta, weights, plasticity)

    # fold_loci entries: dict(g, K, gamma0, terminal)
    fold_loci = []
    hopf_loci = []
    branches = []                 # per-branch 1D curve + bif points + coherent count
    prev_rightmost_idx = None     # index in fold_loci of the current branch's right-most fold
    async_confirm = None
    gamma_seed = gamma

    for b in range(cfg["MAX_BRANCHES"]):
        status, r, psi, A = settle_fp(rhs, M, gamma_seed, K0,
                                      cfg["r0_start"], cfg["T_settle"], cfg["async_tol"])
        n_coh = int((r > cfg["coh_tol"]).sum()) if status == "fp" else 0
        tag = f"<r>={r.mean():.3f}, n_coherent={n_coh}" if status == "fp" else ""
        print(f"\n branch {b}: confirmation sim at γ={gamma_seed:.4f}, K={K0} → {status}  {tag}")
        if status != "fp":
            # the single confirmation simulation past the previous fold landed on
            # the asynchronous state ⇒ that fold IS the collapse-to-async boundary
            if status == "async" and prev_rightmost_idx is not None:
                fold_loci[prev_rightmost_idx]["terminal"] = True
                async_confirm = (gamma_seed, K0)
                print("   → asynchronous (r→0): previous fold is the collapse boundary.")
            else:
                print(f"   → {status}; stopping the cascade.")
            break

        # fresh circuit seeded at this equilibrium; continue it in γ
        phi = psi - psi[0]
        phi[0] = 0.0
        circ = B.build_circuit(M, K0, mu, gamma_seed, omega, delta, weights,
                               r, phi, A, plasticity, name=f"ens_b{b}")
        ode = ODESystem.from_template(circ, auto_dir=B.AUTO_DIR, init_cont=False,
                                      analytical_jacobian=True, auto_constants=("ivp", "eq"))
        eq, _ = ode.run(c="eq", name="eq", ICP="g", bidirectional=True,
                        RL0=cfg["gamma_min"], RL1=cfg["gamma_max"],
                        IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=50,
                        DS=1e-3, DSMIN=1e-10, DSMAX=5e-3,
                        EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, get_stability=True,
                        STOP=["BP2", "HB2"])
        B.recompute_equilibrium_stability(eq, M, jac, "g", {"K": K0, "mu": mu, "g": gamma_seed})
        # drop bifurcations flagged right at the γ-domain edges — those are
        # continuation boundary artifacts (RG/EP mislabels), not genuine folds.
        gtol = 1e-5
        gmn, gmx = cfg["gamma_min"], cfg["gamma_max"]
        keep_g = lambda v: (gmn + gtol) < v < (gmx - gtol)
        lp = [v for v in _dedupe(B._bif_param(eq, "LP", "g")) if keep_g(v)]
        hb = [v for v in _dedupe(B._bif_param(eq, "HB", "g")) if keep_g(v)]
        print(f"   eq branch: LP at γ={[round(x,4) for x in lp]}  HB at γ={[round(x,4) for x in hb]}")

        # store the 1D continuation (x_0 vs γ at K_anchor) for the left panel
        branch_rec = dict(
            n_coh=n_coh,
            gamma_birth=gamma_seed,
            g1d=_state_arr(eq, "g"),
            x1d=_state_arr(eq, "x_0"),
            stab1d=np.asarray(eq[("stability", "")], bool),
            lp_pts=_dedupe_xy([(g, x) for g, x in _bif_xy(eq, "LP", "g", "x_0") if keep_g(g)]),
            hb_pts=_dedupe_xy([(g, x) for g, x in _bif_xy(eq, "HB", "g", "x_0") if keep_g(g)]),
            rightmost_fold_idx=None,
        )

        # 2D loci: continue each LP/HB in (γ, K); track this branch's right-most fold
        branch_fold = []   # (gamma, fold_loci_index)
        for k, glp in enumerate(lp):
            try:
                c2, _ = ode.run(origin="eq", starting_point=f"LP{k+1}", name=f"fold_b{b}_{k}",
                                c="eq", ICP=["g", "K"], bidirectional=True,
                                IPS=1, ISW=2, ISP=2, ILP=0, NMX=6000, NPR=100,
                                DS=1e-3, DSMIN=1e-9, DSMAX=1e-2,
                                EPSL=1e-7, EPSU=1e-7, EPSS=1e-6,
                                UZSTOP={"K": [cfg["K_min"], cfg["K_max"]]})
                g_arr, K_arr = _curve_arrays(c2, "g", "K")
                fold_loci.append(dict(g=g_arr, K=K_arr, gamma0=glp, terminal=False))
                branch_fold.append((glp, len(fold_loci) - 1))
            except Exception as e:
                print(f"   fold locus LP{k+1} skipped: {type(e).__name__}: {str(e)[:60]}")
        for k, ghb in enumerate(hb):
            try:
                c2, _ = ode.run(origin="eq", starting_point=f"HB{k+1}", name=f"hopf_b{b}_{k}",
                                c="eq", ICP=["g", "K"], bidirectional=True,
                                IPS=1, ISW=2, ISP=2, ILP=0, NMX=6000, NPR=100,
                                DS=1e-3, DSMIN=1e-9, DSMAX=1e-2,
                                EPSL=1e-7, EPSU=1e-7, EPSS=1e-6,
                                UZSTOP={"K": [cfg["K_min"], cfg["K_max"]]})
                g_arr, K_arr = _curve_arrays(c2, "g", "K")
                hopf_loci.append(dict(g=g_arr, K=K_arr, gamma0=ghb))
            except Exception as e:
                print(f"   hopf locus HB{k+1} skipped: {type(e).__name__}: {str(e)[:60]}")

        ode.close_session(clear_files=True)
        clear(circ)

        # right-most fold of this branch = the destabilising fold that bounds the
        # coherence region to its right
        prev_rightmost_idx = max(branch_fold, key=lambda t: t[0])[1] if branch_fold else None
        branch_rec["rightmost_fold_idx"] = prev_rightmost_idx
        branches.append(branch_rec)

        # reseed just past the right-most destabilising bifurcation (going up in γ)
        bif_gs = [g for g in (lp + hb) if g > gamma_seed]
        if not bif_gs:
            print("   no destabilising bifurcation above the seed; cascade ends.")
            break
        gamma_seed = max(bif_gs) + cfg["reseed_offset"]
        if gamma_seed > cfg["gamma_max"]:
            print(f"   next seed γ={gamma_seed:.4f} exceeds γ_max; stop.")
            break

    # ── PRL-style two-panel figure ──────────────────────────────────────────
    print("\n[plot] kmo_ensemble_2d.{pdf,png}")
    set_prl_style()
    from matplotlib import cm
    from matplotlib.colors import Normalize

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.0, 2.9),
        gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.30))

    # colour each coherence level by its number of coherent ensembles
    cmap = plt.get_cmap("YlGnBu")
    cmax = max([rec["n_coh"] for rec in branches], default=M)
    norm = Normalize(vmin=0, vmax=cmax)

    # ----- branch / region summary (survives Auto's stdout flood) -------------
    print("\n[summary] cascade branches (ordered by birth γ):")
    for i, rec in enumerate(branches):
        ridx = rec["rightmost_fold_idx"]
        g0 = fold_loci[ridx]["gamma0"] if ridx is not None else None
        term = fold_loci[ridx]["terminal"] if ridx is not None else False
        print(f"   branch {i}: n_coherent={rec['n_coh']}, γ_birth={rec['gamma_birth']:.4f}, "
              f"rightmost fold γ0={g0}{' (terminal)' if term else ''}")
    print("[summary] fold loci K-spans:")
    for j, fl in enumerate(fold_loci):
        K = np.asarray(fl["K"], float)
        print(f"   fold {j}: γ0={fl['gamma0']:.4f}, terminal={fl['terminal']}, "
              f"K∈[{K.min():.3f},{K.max():.3f}] ({len(K)} pts)")

    # ----- panel (a): 1D continuations x_0(γ) at K = K_anchor -----------------
    for rec in branches:
        col = cmap(norm(rec["n_coh"]))
        _plot_stab(axL, rec["g1d"], rec["x1d"], rec["stab1d"], col, lw=1.3, zorder=3)
        for gx, xx in rec["lp_pts"]:
            axL.scatter([gx], [xx], marker="o", s=22, zorder=6,
                        facecolors="white", edgecolors=C_FOLD, linewidths=1.0)
        for gx, xx in rec["hb_pts"]:
            axL.scatter([gx], [xx], marker="s", s=22, zorder=6,
                        facecolors="white", edgecolors=C_HOPF, linewidths=1.0)
    axL.set_xlabel(r"weight decay $\gamma$")
    axL.set_ylabel(r"reference coherence $x_0$")
    axL.set_xlim(cfg["gamma_min"], cfg["gamma_max"])
    axL.set_ylim(bottom=0.0)
    axL.set_title(rf"(a)  1D cascade at $K={K0:g}$", loc="left")
    leg1 = [
        Line2D([0], [0], color="0.35", lw=1.3, ls="-", label="stable"),
        Line2D([0], [0], color="0.35", lw=1.3, ls="--", label="unstable"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=C_FOLD, label="fold (LP)", markersize=5),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=C_HOPF, label="Hopf (HB)", markersize=5),
    ]
    axL.legend(handles=leg1, loc="upper right", handlelength=1.6)

    # ----- panel (b): 2D (γ, K) loci with coherent-count shading --------------
    Kgrid = np.linspace(cfg["K_min"], cfg["K_max"], 400)
    # ordered region boundaries = each branch's right-most fold (the cascade steps)
    bnds = [(branches[i]["rightmost_fold_idx"], branches[i]["n_coh"])
            for i in range(len(branches))
            if branches[i]["rightmost_fold_idx"] is not None]
    bnds.sort(key=lambda t: fold_loci[t[0]]["gamma0"])
    boundary_curves = [_interp_gamma_of_K(fold_loci[idx], Kgrid) for idx, _ in bnds]
    region_counts = [n for _, n in bnds] + [0]   # left→right; last region = async

    # Shade by painting progressively rightward: start with the highest-coherence
    # background, then for each fold boundary paint everything to its RIGHT with the
    # next-lower coherent count. Where a fold curve is undefined (NaN, outside its
    # own K-range) nothing is painted there, so the underlying region shows through.
    gmin, gmax = cfg["gamma_min"], cfg["gamma_max"]
    axR.fill_betweenx(Kgrid, gmin, gmax, color=cmap(norm(region_counts[0])),
                      lw=0, zorder=1)
    for bi, curve in enumerate(boundary_curves):
        cnt = region_counts[bi + 1]
        le = np.where(np.isnan(curve), gmax, curve)   # NaN ⇒ zero-width ⇒ no paint
        axR.fill_betweenx(Kgrid, le, gmax, where=(gmax > le),
                          color=cmap(norm(cnt)), lw=0, zorder=2 + bi)

    for hl in hopf_loci:
        axR.plot(hl["g"], hl["K"], color=C_HOPF, lw=1.2, zorder=4)
    for fl in fold_loci:
        if not fl["terminal"]:
            axR.plot(fl["g"], fl["K"], color=C_FOLD, lw=1.3, zorder=6)
    for fl in fold_loci:
        if fl["terminal"]:
            axR.plot(fl["g"], fl["K"], color=C_ASYNC, lw=1.6, ls="--", zorder=8)
    axR.axhline(K0, color="0.4", lw=0.6, ls=":", zorder=2)
    if async_confirm is not None:
        axR.scatter([async_confirm[0]], [async_confirm[1]], marker="x", s=28,
                    color=C_ASYNC, zorder=13)

    axR.set_xlabel(r"weight decay $\gamma$")
    axR.set_ylabel(r"coupling $K$")
    axR.set_xlim(cfg["gamma_min"], cfg["gamma_max"])
    axR.set_ylim(cfg["K_min"], cfg["K_max"])
    axR.set_title(rf"(b)  $(\gamma, K)$ bifurcations, $M={M}$", loc="left")

    # discrete colourbar = number of coherent ensembles
    sm = cm.ScalarMappable(norm=Normalize(vmin=-0.5, vmax=cmax + 0.5), cmap=cmap)
    cbar = fig.colorbar(sm, ax=axR, ticks=range(0, cmax + 1), pad=0.02, fraction=0.05)
    cbar.set_label("coherent ensembles", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    leg2 = [Line2D([0], [0], color=C_FOLD, lw=1.3, label="fold of equilibria"),
            Line2D([0], [0], color=C_ASYNC, lw=1.6, ls="--",
                   label="collapse to async")]
    if hopf_loci:
        leg2.insert(1, Line2D([0], [0], color=C_HOPF, lw=1.2, label="Hopf"))
    axR.legend(handles=leg2, loc="lower right", handlelength=1.6)

    fig.savefig("kmo_ensemble_2d.pdf", bbox_inches="tight")
    fig.savefig("kmo_ensemble_2d.png", dpi=300, bbox_inches="tight")
    print("  saved kmo_ensemble_2d.pdf / .png")
    print("Done.")


if __name__ == "__main__":
    main()
    plt.show()
