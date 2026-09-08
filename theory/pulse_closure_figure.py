r"""
Pulse-driven weight adaptation: moment-closure summary figure
=============================================================

Loads the scoring written by ``pulse_closure_meanfield.py`` and assembles one
two-column PRL figure, 2 rows x 4 columns:

  row 1 — the mechanism
    (a) microscopic R(t), Ā(t), V_A(t) against the Bhatia–Davis bound Ā(A_m−Ā).
        Because the rule is LINEAR in A, the −2Γ̄V_A damping holds V_A far below the
        bound — the opposite of the saturating rule, which drifts onto it.
    (b) drive moments reconstructed from the Daido order parameters z_q vs. measured,
        with and without the OA ansatz z_q = Z^q.  Tests the reduction of Sec. 4.
    (c) validation of the latent-target picture: A_ij against the adiabatic
        prediction A_m/(1+r̄_ij), r̄_ij = μ_d⟨G_d⟩_t/(μ_p⟨G_p⟩_t).
    (d) the terms of the exact V_A equation: the damping −2Γ̄V_A balances the
        covariance source, and the third-order term is a small correction.

  row 2 — closure performance
    (e) Ā(t): micro vs. naive (K=0) vs. slow-kernel truncation vs. quasi-static
    (f) V_A(t): micro vs. each closure
    (g) how much of the drive variance is SLOW: Var(ḡ_x) vs Var(G_x), and K vs K_slow
    (h) relative error for every regime x model

Usage (``pycobi`` env)
----------------------
    python pulse_closure_figure.py                 # default regime in panels a–d
    python pulse_closure_figure.py sync
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

_panel_label = functools.partial(panel_label, dx=-24, dy=3)

CONFIG = dict(
    scored=dp.kmo_adaptive("pulse_closure_scored_measured.npz"),
    out=dp.kmo_adaptive("pulse_closure"),
    show_regime="partial",
    models=("naive", "driven", "static"),
)

C_MICRO = "0.15"
C_MDL = {"naive": "#c1121f", "driven": "#f08c00", "static": "#1f6fb2", "qs": "#0b7285"}
L_MDL = {"naive": "naive ($K=0$)", "driven": r"$V_A$ closure ($K$ from micro)",
         "static": "slow-kernel truncation", "qs": "quasi-static"}
C_BOUND = "#2ca02c"
NICE = {"async_": "asynchronous", "partial": "partially locked",
        "sync": "strongly locked", "stress": r"strong modulation"}


def main(show=None):
    d = np.load(CONFIG["scored"], allow_pickle=False)
    regimes = [str(r) for r in d["regimes"]]
    show = show if show in regimes else CONFIG["show_regime"]
    if show not in regimes:
        show = regimes[0]
    models = [m for m in CONFIG["models"] if f"{show}/Abar_{m}" in d]
    g = lambda r, k: d[f"{r}/{k}"]

    t = g(show, "t")
    R, Ab, VA, Kp, Kd = (g(show, k) for k in ("R", "Abar", "VA", "Kp", "Kd"))
    bound, sat, Gam = g(show, "bound"), g(show, "sat"), g(show, "Gbar")
    dm, dms, third = g(show, "dm"), g(show, "dm_slow"), g(show, "third")
    mu_p, mu_d, Am = float(g(show, "mu_p")), float(g(show, "mu_d")), 1.0

    set_prl_style()
    fig, axes = plt.subplots(2, 4, figsize=(COL_DOUBLE, 3.6), layout="constrained")

    # ── (a) micro traces vs the Bhatia-Davis bound ─────────────────────────
    ax = axes[0, 0]
    ax.plot(t, R, color="#7b2cbf", lw=0.7, label=r"$R$")
    ax.plot(t, Ab, color=C_MICRO, lw=1.0, label=r"$\bar A$")
    ax.plot(t, bound, color=C_BOUND, lw=0.8, ls=":", label=r"$\bar A(A_m{-}\bar A)$")
    ax.plot(t, VA, color="#0b7285", lw=1.0, label=r"$V_A$")
    ax.set_xlabel(r"time $t$"); ax.set_ylabel("order parameters")
    ax.legend(loc="center right", handlelength=1.1, borderpad=0.2, labelspacing=0.22,
              fontsize=5.2)
    ax.text(0.03, 0.04, rf"$s_{{\max}}={np.nanmax(sat):.3f}$", transform=ax.transAxes,
            fontsize=5.4)
    _panel_label(ax, "a")

    # ── (b) Daido reconstruction of the drive moments ───────────────────────
    ax = axes[0, 1]
    for src, col, lbl in (("daido", "#1f6fb2", r"$z_q$ measured"),
                          ("oa", "#c1121f", r"OA: $z_q=Z^q$")):
        xs, ys = [], []
        for r in regimes:
            key = f"chk_{r}/{src}"
            if key not in d:
                continue
            xs.append(np.abs(g(r, "dm")).mean(0)); ys.append(d[key])
        if not xs:
            continue
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        ax.semilogy(np.arange(len(ys)) % 5 + (0.15 if src == "oa" else -0.15),
                    np.maximum(ys, 1.2e-6), "o", ms=2.6, color=col, label=lbl, alpha=0.8)
    ax.set_ylim(1e-6, 1.0)
    ax.text(0.03, 0.97, "floor = exact to machine precision", transform=ax.transAxes,
            fontsize=4.2, ha="left", va="top", color="0.4")
    ax.set_xticks(range(5))
    ax.set_xticklabels([r"$\bar G_p$", r"$\bar G_d$", r"Var$G_p$", r"Var$G_d$",
                        r"Cov"], fontsize=5.4)
    ax.set_ylabel("rel. error of reconstruction")
    ax.legend(loc="lower right", handlelength=0.9, borderpad=0.2, fontsize=5.2)
    _panel_label(ax, "b")

    # ── (c) latent-target validation ────────────────────────────────────────
    ax = axes[0, 2]
    r_lat, A_lat = g(show, "r_snap"), g(show, "A_snap")
    ns = r_lat.shape[-1]
    off = ~np.eye(ns, dtype=bool)
    rr, AA = r_lat[-1][off], A_lat[-1][off]
    pred = Am / (1.0 + rr)
    sub = np.random.default_rng(0).choice(len(rr), size=min(4000, len(rr)), replace=False)
    ax.plot(pred[sub], AA[sub], ".", ms=0.7, color="0.55", alpha=0.5, rasterized=True)
    lo, hi = min(pred.min(), AA.min()), max(pred.max(), AA.max())
    ax.plot([lo, hi], [lo, hi], color="#c1121f", lw=0.9)
    ss = np.sum((AA - AA.mean()) ** 2)
    r2 = 1.0 - np.sum((AA - pred) ** 2) / ss
    ax.set_xlabel(r"adiabatic target $A_m/(1+\bar r_{ij})$")
    ax.set_ylabel(r"$A_{ij}$")
    ax.text(0.05, 0.92, rf"$R^2={r2:.3f}$", transform=ax.transAxes, va="top", fontsize=5.8)
    _panel_label(ax, "c")

    # ── (d) terms of the exact V_A equation ─────────────────────────────────
    ax = axes[0, 3]
    src = 2 * (mu_p * Kp * (Am - Ab) - mu_d * Kd * Ab)
    damp = -2 * Gam * VA
    Wp, Wd = third[:, 0], third[:, 1]
    thi = -2 * (mu_p * Wp + mu_d * Wd)
    ax.plot(t, src, color="#0b7285", lw=0.9, label=r"$2[\mu_pK_p(A_m{-}\bar A){-}\mu_dK_d\bar A]$")
    ax.plot(t, damp, color="#c1121f", lw=0.9, label=r"$-2\bar\Gamma V_A$")
    ax.plot(t, thi, color="#f08c00", lw=0.9, label=r"3rd order")
    ax.plot(t, src + damp + thi, color=C_MICRO, lw=0.9, ls="--", label=r"$V_A'$")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"terms of $\dot V_A$")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.9)
    ax.legend(loc="upper left", handlelength=0.9, borderpad=0.2, labelspacing=0.2,
              fontsize=4.6)
    _panel_label(ax, "d")

    # ── (e) Abar(t) ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t, Ab, color=C_MICRO, lw=1.2, label="micro")
    for mdl in models:
        ax.plot(t, g(show, f"Abar_{mdl}"), color=C_MDL[mdl], lw=0.9, label=L_MDL[mdl])
    ax.plot(t, g(show, "Abar_qs"), color=C_MDL["qs"], lw=0.9, ls="--", label=L_MDL["qs"])
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"$\bar A$")
    lo, hi = np.nanmin(Ab), np.nanmax(Ab); pad = 0.6 * (hi - lo) + 1e-3
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(loc="best", handlelength=1.0, borderpad=0.2, labelspacing=0.2, fontsize=4.8)
    _panel_label(ax, "e")

    # ── (f) V_A(t) ──────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t, VA, color=C_MICRO, lw=1.2)
    for mdl in models:
        if mdl == "naive":
            continue
        ax.plot(t, g(show, f"VA_{mdl}"), color=C_MDL[mdl], lw=0.9)
    ax.plot(t, g(show, "VA_qs"), color=C_MDL["qs"], lw=0.9, ls="--")
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"$V_A$")
    ax.set_ylim(0, 2.2 * np.nanmax(VA))
    _panel_label(ax, "f")

    # ── (g) how much of the drive variance is slow ──────────────────────────
    ax = axes[1, 2]
    ax.plot(t, dm[:, 2], color="#1f6fb2", lw=0.9, label=r"Var$(G_p)$")
    ax.plot(t, dms[:, 2], color="#1f6fb2", lw=0.9, ls="--", label=r"Var$(\bar g_p)$")
    ax.plot(t, dm[:, 3], color="#c1121f", lw=0.9, label=r"Var$(G_d)$")
    ax.plot(t, dms[:, 3], color="#c1121f", lw=0.9, ls="--", label=r"Var$(\bar g_d)$")
    ax.set_yscale("log")
    ax.set_xlabel(r"time $t$"); ax.set_ylabel("drive variance")
    ax.legend(loc="best", handlelength=1.1, borderpad=0.2, labelspacing=0.2, fontsize=4.8)
    _panel_label(ax, "g")

    # ── (h) error summary ───────────────────────────────────────────────────
    ax = axes[1, 3]
    keys = [("naive", "errAbar_naive", r"$\bar A$, naive"),
            ("static", "errAbar_static", r"$\bar A$, slow-kernel"),
            ("driven", "errVA_driven", r"$V_A$, $K$ from micro"),
            ("static", "errVA_static", r"$V_A$, slow-kernel")]
    x = np.arange(len(regimes)); w = 0.8 / len(keys)
    cols = ["#c1121f", "#1f6fb2", "#f08c00", "#0b7285"]
    for k, ((_, key, lbl), col) in enumerate(zip(keys, cols)):
        vals = [float(g(r, key)) if f"{r}/{key}" in d else np.nan for r in regimes]
        ax.bar(x + (k - (len(keys) - 1) / 2) * w, vals, width=w, color=col, label=lbl)
    ax.set_yscale("log"); ax.axhline(1.0, color="0.6", lw=0.5, ls=":")
    ax.set_ylim(top=ax.get_ylim()[1] * 12)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(r, r) for r in regimes], rotation=30, ha="right")
    ax.set_ylabel("relative error")
    ax.legend(loc="upper right", handlelength=0.8, borderpad=0.2, labelspacing=0.2,
              fontsize=4.4)
    _panel_label(ax, "h")

    out = CONFIG["out"]
    fig.savefig(out + ".svg"); fig.savefig(out + ".png", dpi=300)
    print(f"[saved] {out}.svg / .png   (regime shown: {show})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
