r"""
SCN mixtures: frequency distribution, 1-D bifurcation in K and a backward K staircase (noise floor vs single Lorentzian)
=====================================================================================================================

Minimal mean field (global coupling only, α = 0, h = 1; scn_pycobi_bifurcation):
    ż_m = (iΩ_m − Δ_m) z_m + ½ K (Z − Z̄ z_m²),   Z = Σ w_m z_m,   R = |Z|
The minimal model has no lobe structure, so the mixtures are fitted to the slice's POOLED intrinsic frequencies
(scn_pycobi_bifurcation kinds "pfloor" = noise-floor Lorentzian mixture, free M; "psingle" = one Lorentzian, M = 1;
λ = 1e-5). For SCN4 the noise-floor fit selects M = 1 (identical curves).
Rows = SCN1 (bimodal), SCN3 (broad + side peak), SCN4 (narrow). Columns:
  (1) intrinsic frequencies under TTX (all fit cells) and the two mixture densities
  (2) 1-D bifurcation diagram R*(K) from PyCoBi (scn_pycobi_bifurcation.py run --no-loci --cases i:pfloor,i:psingle):
      stable / unstable equilibria, folds, Hopf points, limit-cycle R min/max (trimmed where the amplitude collapses);
      incoherent state z = 0 (stable below K_c, analytic); staircase levels marked on the axis
  (3) staircase simulation: settle at K = K_STEPS[0], then hold each K in K_STEPS for HOLD_DAYS (0.06 → 0 in steps
      of 0.01, 30 days each); R(t) for both mixtures, thin grey = stable steady state R*(K) of the noise-floor
      mixture. Deterministic RK4 on the complex z_m (no pinned frame), tiny symmetry-breaking initial perturbation.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_bif_ramp_figure.py [--resim]
Output: data_paths.MPMF/circadian_data/pycobi_bif/scn_bif_steps{.pdf,.png}, traces in scn_steps.npz
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import scn_pycobi_bifurcation as B

SLICES = (1, 2, 3, 4, 5)                                   # figure rows: --slices 1,3,4 for a subset
KINDS = (("pfloor", "#c0392b", "-", "noise floor"), ("psingle", "0.3", "--", "single Lorentzian"))
K_STEPS = np.round(np.arange(0.06, -0.005, -0.01), 3)      # 0.06, 0.05, ..., 0.00
HOLD_DAYS = 30.0
T_SETTLE = 3000.0                                          # h at K_STEPS[0] before the staircase
DT = 0.25                                                  # h
CPD = 24 / (2 * np.pi)                                     # rad/h -> cycles/day
C_LC = "#1a9850"


# ─── staircase simulation ────────────────────────────────────────────────────────────────────────
def rhs(z, K, w, a):
    Z = np.dot(w, z)
    return a * z + 0.5 * K * (Z - np.conj(Z) * z ** 2)


def _rk4(z, K, w, a):
    k1 = rhs(z, K, w, a); k2 = rhs(z + 0.5 * DT * k1, K, w, a)
    k3 = rhs(z + 0.5 * DT * k2, K, w, a); k4 = rhs(z + DT * k3, K, w, a)
    return z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def staircase(w, Om, De):
    """Settle at K_STEPS[0], then hold each level HOLD_DAYS; returns t (h), K(t), R(t)."""
    a = 1j * Om - De
    z = 0.99 * np.exp(1j * 1e-3 * np.random.default_rng(0).standard_normal(w.size))
    for _ in range(int(T_SETTLE / DT)):
        z = _rk4(z, K_STEPS[0], w, a)
    n = int(HOLD_DAYS * 24 / DT)
    K = np.repeat(K_STEPS, n); R = np.empty(K.size + 1); R[0] = abs(np.dot(w, z))
    for k in range(K.size):
        z = _rk4(z, K[k], w, a); R[k + 1] = abs(np.dot(w, z))
    return np.arange(K.size + 1) * DT, np.r_[K[0], K], R


def simulate():
    out = {}
    for i in SLICES:
        for k, *_ in KINDS:
            t, K, R = staircase(*B.mixture(i, k))
            out[f"s{i}_{k}_t"], out[f"s{i}_{k}_K"], out[f"s{i}_{k}_R"] = t, K, R
            print(f"[steps] SCN{i} {k}: R at the end of each level "
                  + " ".join(f"{R[int((j + 1) * HOLD_DAYS * 24 / DT)]:.2f}" for j in range(K_STEPS.size)), flush=True)
    out["K_steps"], out["hold_days"] = K_STEPS, HOLD_DAYS
    return out


def stable_Rstar(res, K):
    """Stable equilibrium R* of a PyCoBi result at coupling K (NaN if none, e.g. in the limit-cycle window)."""
    if K <= res["kc"]["K_c"]:
        return 0.0
    eq = res["eq"]; Rs = []
    for j in range(eq["K"].size - 1):
        k0, k1 = eq["K"][j], eq["K"][j + 1]
        if min(k0, k1) <= K <= max(k0, k1) and eq["stable"][j] and eq["stable"][j + 1] and k0 != k1:
            Rs.append(eq["R"][j] + (eq["R"][j + 1] - eq["R"][j]) * (K - k0) / (k1 - k0))
    return max(Rs) if Rs else np.nan


# ─── figure ──────────────────────────────────────────────────────────────────────────────────────
def main():
    fz = os.path.join(B.OUT, "scn_steps.npz")
    if "--resim" in sys.argv or not os.path.exists(fz):
        np.savez(fz, **simulate())
    S = np.load(fz)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from prl_style import set_prl_style, panel_label, COL_DOUBLE
    import scn_targets as TG
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center"})
    rows = (tuple(int(s_) for s_ in sys.argv[sys.argv.index("--slices") + 1].split(","))
            if "--slices" in sys.argv else SLICES)
    H = 1.55 * len(rows) + 0.45
    fig, axs = plt.subplots(len(rows), 3, figsize=(COL_DOUBLE, H), squeeze=False,
                            gridspec_kw=dict(hspace=0.75, wspace=0.3, left=0.07, right=0.99, top=1 - 0.31 / H,
                                             bottom=0.58 / H,
                                             width_ratios=[1, 1, 1.35]))
    Ks, hold = S["K_steps"], float(S["hold_days"])
    for r, i in enumerate(rows):
        f_cells = TG.load(i)["om"] * CPD
        res = {k: np.load(os.path.join(B.OUT, f"scn{i}_{k}.npy"), allow_pickle=True).item() for k, *_ in KINDS}
        # (1) frequency distribution
        a = axs[r, 0]
        g = np.linspace(f_cells.min() - 0.03, f_cells.max() + 0.03, 800)
        a.hist(f_cells, bins=np.linspace(g[0], g[-1], 36), density=True, color="0.85", lw=0)
        for k, col, ls, lab in KINDS:
            mod = B.pooled_mixture(i, k); w = mod["w"] / mod["w"].sum(); gg = g / CPD
            pdf = (w[:, None] * mod["De"][:, None] / np.pi / ((gg[None] - mod["Om"][:, None]) ** 2
                                                              + mod["De"][:, None] ** 2)).sum(0) / CPD
            a.plot(g, pdf, color=col, ls=ls, lw=1.0)
        a.set_yticks([]); a.set_xlabel("intrinsic frequency (cycles/day)")
        a.set_title(f"SCN{i}  (N = {f_cells.size})", fontsize=6.5)
        a.text(0.97, 0.95, "\n".join(f"{lab}: M = {res[k]['M']}" for k, _, _, lab in KINDS), transform=a.transAxes,
               ha="right", va="top", fontsize=5.2, linespacing=1.4, zorder=5,
               bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        if r == 0:
            a.set_ylabel("density")
        # (2) bifurcation diagram (linear K over the staircase range)
        a = axs[r, 1]
        for k, col, ls, lab in KINDS:
            gq = res[k]; eq = gq["eq"]
            a.plot(eq["K"], np.where(eq["stable"], eq["R"], np.nan), color=col, ls=ls, lw=1.1)
            a.plot(eq["K"], np.where(~eq["stable"], eq["R"], np.nan), color=col, ls=":", lw=0.9)
            a.plot([0, gq["kc"]["K_c"]], [0, 0], color=col, ls=ls, lw=1.1)
            a.plot([gq["kc"]["K_c"], 0.07], [0, 0], color=col, ls=":", lw=0.9)
            for b_, mk in (("LP", "o"), ("HB", "s")):
                s_ = eq["bif"] == b_
                a.plot(eq["K"][s_], eq["R"][s_], mk, ms=3, mfc="white", mec="k", mew=0.7, zorder=4)
            for lc in filter(None, map(B.trim_lc, gq["lc"])):
                for key in ("Rmax", "Rmin"):
                    a.plot(lc["K"], np.where(lc["stable"], lc[key], np.nan), color=C_LC, lw=0.9)
                    a.plot(lc["K"], np.where(~lc["stable"], lc[key], np.nan), color=C_LC, lw=0.7, ls=":")
        a.plot(Ks, np.full(Ks.size, -0.02), "|", color="0.4", ms=4, mew=0.8, clip_on=False)
        a.set_xlim(-0.001, 0.063); a.set_ylim(-0.02, 1.02)
        a.set_xlabel(r"coupling $K$ (rad/h)"); a.set_ylabel(r"$R^*$")
        a.set_title(f"SCN{i}: bifurcation in $K$", fontsize=6.5)
        # (3) staircase
        a = axs[r, 2]
        for j, Kj in enumerate(Ks):
            if j % 2 == 0:
                a.axvspan(j * hold, (j + 1) * hold, color="0.95", lw=0, zorder=0)
            a.text((j + 0.5) * hold, 1.04, f"{abs(Kj):.2f}", ha="center", va="bottom", fontsize=5, color="0.3")
            Rs = stable_Rstar(res["pfloor"], Kj)
            if np.isfinite(Rs):
                a.plot([j * hold, (j + 1) * hold], [Rs, Rs], color="0.55", lw=0.6, zorder=1)
        for k, col, ls, lab in KINDS:
            a.plot(S[f"s{i}_{k}_t"] / 24, S[f"s{i}_{k}_R"], color=col, ls=ls, lw=0.9, zorder=2)
        a.text(0.5, 1.16, r"$K$ (rad/h) per 30-day hold", transform=a.transAxes, ha="center", fontsize=5.5)
        a.set_xlim(0, Ks.size * hold); a.set_ylim(-0.02, 1.02)
        a.set_xticks(np.arange(0, Ks.size * hold + 1, hold))
        a.set_xlabel("time (days)"); a.set_ylabel(r"$R(t)$")
    for r in range(len(rows)):
        panel_label(axs[r, 0], "abcdefgh"[r], dx=-14, dy=3)
    hd = ([Patch(fc="0.85", label="cells (TTX)")] +
          [Line2D([], [], color=c, ls=ls, lw=1.0, label=lab) for _, c, ls, lab in KINDS] +
          [Line2D([], [], color="0.3", ls=":", lw=0.9, label="unstable equilibrium"),
           Line2D([], [], color=C_LC, lw=0.9, label="limit cycle (R min/max)"),
           Line2D([], [], ls="none", marker="o", mfc="white", mec="k", ms=3, label="fold"),
           Line2D([], [], ls="none", marker="s", mfc="white", mec="k", ms=3, label="Hopf"),
           Line2D([], [], color="0.55", lw=0.6, label=r"stable $R^*$ (noise floor)")])
    fig.legend(handles=hd, loc="lower center", ncol=8, frameon=False, fontsize=5.3, bbox_to_anchor=(0.5, 0.0))
    out = os.path.join(B.OUT, "scn_bif_steps" + ("" if rows == SLICES else "_scn" + "".join(map(str, rows))))
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")


if __name__ == "__main__":
    main()
