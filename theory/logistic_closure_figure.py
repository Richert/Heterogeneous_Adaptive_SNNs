r"""
Saturating (logistic) adaptive coupling: moment-closure summary figure
======================================================================

Loads the scoring written by ``logistic_closure_meanfield.py`` and assembles one
two-column PRL figure, 2 rows x 4 columns:

  row 1 — the mechanism (one representative regime)
    (a) microscopic R(t), Ā(t), V_A(t) against the Bhatia–Davis bound Ā(A_m−Ā):
        the weight variance rises towards the bound, i.e. the weight law saturates
        towards a two-atom (binary) distribution.
    (b) the three terms of the EXACT V_A equation.  The source 2Ā(A_m−Ā)C_A is
        balanced almost entirely by −2Q: the third-order term is what bounds V_A.
    (c) validation of the latent-drive picture: A_ij against the per-pair latent
        u_ij = ⟨G_ij⟩_t collapses onto A_m σ(λu + b), the exact single-pair solution.
    (d) the latent distribution p(u) with the locked+drifting mixture fit.

  row 2 — closure performance
    (e) V_A(t): microscopic vs. each closure (log scale; P=Q=0 diverges past the bound)
    (f) Ā(t): same comparison
    (g) saturation s = V_A/[Ā(A_m−Ā)]: admissible closures must stay below 1
    (h) relative error in V_A for every regime x closure

Usage (``pycobi`` env)
----------------------
    python logistic_closure_figure.py                 # default representative regime
    python logistic_closure_figure.py sync
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
from prl_style import set_prl_style, panel_label, COL_DOUBLE
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

_panel_label = functools.partial(panel_label, dx=-22, dy=3)

CONFIG = dict(
    scored=dp.kmo_adaptive("logistic_closure_scored.npz"),
    out=dp.kmo_adaptive("logistic_closure"),
    show_regime="partial",
    closures=("zero", "gauss", "mix", "mix_bd"),
)

C_MICRO = "0.15"
C_CLOS = {"zero": "#c1121f", "gauss": "#f08c00", "mix": "#1f6fb2", "mix_bd": "#0b7285"}
L_CLOS = {"zero": r"$P=Q=0$", "gauss": "Gaussian latent", "mix": "locked+drifting latent",
          "mix_bd": "locked+drifting, admissible"}
C_BOUND = "#2ca02c"
NICE = {"async_": "asynchronous", "partial": "partially locked",
        "slow": "slow adaptation", "sync": "strongly locked"}


def _get(d, regime, field):
    return d[f"{regime}/{field}"]


def main(show=None):
    d = np.load(CONFIG["scored"], allow_pickle=False)
    regimes = [str(r) for r in d["regimes"]]
    show = show or CONFIG["show_regime"]
    if show not in regimes:
        show = regimes[0]
    closures = [c for c in CONFIG["closures"] if f"{show}/VA_{c}" in d]

    t = _get(d, show, "t")
    R, Ab, VA, CA, P, Q = (_get(d, show, k) for k in "R Abar VA CA P Q".split())
    Gb, bound, sat = (_get(d, show, k) for k in ("Gbar", "bound", "sat"))
    Am = 1.0

    set_prl_style()
    fig, axes = plt.subplots(2, 4, figsize=(COL_DOUBLE, 3.5), layout="constrained")

    # ── (a) micro traces + Bhatia–Davis bound ──────────────────────────────
    ax = axes[0, 0]
    ax.plot(t, R, color="#7b2cbf", lw=0.9, label=r"$R$")
    ax.plot(t, Ab, color=C_MICRO, lw=0.9, label=r"$\bar A$")
    ax.plot(t, VA, color="#0b7285", lw=0.9, label=r"$V_A$")
    ax.plot(t, bound, color=C_BOUND, lw=0.8, ls=":", label=r"$\bar A(A_m-\bar A)$")
    ax.set_xlabel(r"time $t$"); ax.set_ylabel("order parameters")
    ax.legend(loc="center right", handlelength=1.2, borderpad=0.2, labelspacing=0.25)
    _panel_label(ax, "a")

    # ── (b) the three terms of the exact V_A equation ───────────────────────
    ax = axes[0, 1]
    ax.plot(t, 2 * bound * CA, color="#0b7285", lw=0.9, label=r"$2\bar A(A_m{-}\bar A)C_A$")
    ax.plot(t, 2 * (Am - 2 * Ab) * P, color="#f08c00", lw=0.9, label=r"$2(A_m{-}2\bar A)P$")
    ax.plot(t, -2 * Q, color="#c1121f", lw=0.9, label=r"$-2Q$")
    ax.plot(t, 2 * bound * CA + 2 * (Am - 2 * Ab) * P - 2 * Q, color=C_MICRO,
            lw=0.9, ls="--", label=r"$V_A'$")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"terms of $V_A'$")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.7)          # headroom for the legend
    ax.legend(loc="upper left", handlelength=1.0, borderpad=0.2, labelspacing=0.2,
              fontsize=5.0)
    _panel_label(ax, "b")

    # ── (c) latent-drive validation ─────────────────────────────────────────
    ax = axes[0, 2]
    u, A = d[f"lat_{show}/u"], d[f"lat_{show}/A"]
    lam, beff, r2 = (float(d[f"lat_{show}/{k}"]) for k in ("lam", "b_eff", "r2"))
    sub = np.random.default_rng(0).choice(len(u), size=min(4000, len(u)), replace=False)
    ax.plot(u[sub], A[sub], ".", ms=0.7, color="0.55", alpha=0.5, rasterized=True)
    ug = np.linspace(u.min(), u.max(), 300)
    ax.plot(ug, Am / (1 + np.exp(-(lam * ug + beff))), color="#c1121f", lw=1.1)
    ax.set_xlabel(r"latent drive $u_{ij}=\langle G_{ij}\rangle_t$")
    ax.set_ylabel(r"$A_{ij}$")
    ax.text(0.04, 0.92, rf"$A_m\sigma({lam:.0f}u{beff:+.2f})$" + "\n" + rf"$R^2={r2:.3f}$",
            transform=ax.transAxes, va="top", fontsize=5.6)
    _panel_label(ax, "c")

    # ── (d) latent distribution + locked/drifting mixture ───────────────────
    ax = axes[0, 3]
    nL, uL, sD = (float(d[f"lat_{show}/{k}"]) for k in ("nL", "uL", "sD"))
    _, edges, _ = ax.hist(u, bins=70, density=True, color="0.8", label=r"micro $p(u)$")
    ug = np.linspace(u.min(), u.max(), 400)
    drift = (1 - nL) * np.exp(-ug ** 2 / (2 * sD ** 2)) / (sD * np.sqrt(2 * np.pi))
    ax.plot(ug, drift, color="#1f6fb2", lw=1.0, label=f"drifting ({1-nL:.2f})")
    # stem height chosen so the atom's AREA equals its weight n_L (one bin wide)
    ax.vlines(uL, 0, nL / (edges[1] - edges[0]), color="#c1121f", lw=1.2,
              label=f"locked ({nL:.2f})")
    ax.set_xlabel(r"latent drive $u$"); ax.set_ylabel(r"density")
    ax.legend(loc="upper left", handlelength=1.0, borderpad=0.2, labelspacing=0.25,
              fontsize=5.4)
    _panel_label(ax, "d")

    # ── (e) V_A(t): micro vs closures ───────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t, VA, color=C_MICRO, lw=1.1, label="micro")
    ax.plot(t, bound, color=C_BOUND, lw=0.8, ls=":", label="bound")
    for c in closures:
        ax.plot(t, _get(d, show, f"VA_{c}"), color=C_CLOS[c], lw=0.9, label=L_CLOS[c])
    ax.set_yscale("log"); ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"$V_A$")
    ax.set_ylim(1e-4, max(3 * np.nanmax(bound), 1.0))
    ax.legend(loc="lower right", handlelength=1.2, borderpad=0.2, labelspacing=0.25,
              fontsize=5.2)
    _panel_label(ax, "e")

    # ── (f) Ā(t) ────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t, Ab, color=C_MICRO, lw=1.1)
    for c in closures:
        ax.plot(t, _get(d, show, f"Abar_{c}"), color=C_CLOS[c], lw=0.9)
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"$\bar A$")
    _panel_label(ax, "f")

    # ── (g) saturation s, with the admissibility ceiling ────────────────────
    ax = axes[1, 2]
    ax.axhspan(1.0, 3.0, color="#c1121f", alpha=0.10, lw=0)
    ax.axhline(1.0, color="#c1121f", lw=0.7, ls="--")
    ax.plot(t, sat, color=C_MICRO, lw=1.1)
    for c in closures:
        A_, V_ = _get(d, show, f"Abar_{c}"), _get(d, show, f"VA_{c}")
        ax.plot(t, V_ / (A_ * (Am - A_)), color=C_CLOS[c], lw=0.9)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$s=V_A/[\bar A(A_m-\bar A)]$")
    ax.text(0.5, 1.05, "inadmissible", color="#c1121f", fontsize=5.4,
            ha="center", transform=ax.get_yaxis_transform() if False else ax.transAxes)
    _panel_label(ax, "g")

    # ── (h) error summary across regimes ────────────────────────────────────
    ax = axes[1, 3]
    x = np.arange(len(regimes)); w = 0.8 / max(len(closures), 1)
    for k, c in enumerate(closures):
        vals = [float(_get(d, r, f"errVA_{c}")) for r in regimes]
        ax.bar(x + (k - (len(closures) - 1) / 2) * w, vals, width=w,
               color=C_CLOS[c], label=L_CLOS[c])
    ax.set_yscale("log"); ax.axhline(1.0, color="0.6", lw=0.5, ls=":")
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(r, r) for r in regimes], rotation=30, ha="right")
    ax.set_ylabel(r"rel. error in $V_A$")
    ax.legend(loc="upper right", handlelength=0.9, borderpad=0.2, labelspacing=0.25,
              fontsize=5.0)
    _panel_label(ax, "h")

    out = CONFIG["out"]
    fig.savefig(out + ".svg"); fig.savefig(out + ".png", dpi=300)
    print(f"[saved] {out}.svg / .png   (regime shown: {show})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
