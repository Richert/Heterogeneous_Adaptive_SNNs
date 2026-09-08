r"""
LMMF vs. adaptive Kuramoto — discrepancy vs. the weight-variance budget: summary figure
=======================================================================================

Loads the sweep written by ``kmo_lmmf_variance_bound_sweep.py`` and renders, SEPARATELY FOR
EACH μ, a single-column PRL-style 2×2 figure of the LMMF-vs-microscopic discrepancy as a
function of the upper bound on the predicted relative weight variance V_A/Ā² (x-axis, log).
In every panel the line COLOUR encodes the adaptation rule G_A ∈ {cos, sin, |sin|}; each line
is the trial mean ± s.d. (errorbars) over the sweep's random realizations. One figure is
written per μ (``..._mu{μ}.pdf/.png``).

  (a) spectral RMSE of the average phase coherence R(t): RMSE between the (sample-normalized)
      amplitude spectra |rfft(R)| of the microscopic and LMMF traces.
  (b) number of LMMF ensembles M (from the mixture fit; rule-independent ⇒ one line per μ).
  (c) RMSE between the mean-field and microscopic mean-coupling traces Ā(t) (time domain).
  (d) spatial-spectral RMSE of the coupling matrix A_ij: RMSE between the (normalized) 2-D
      amplitude spectra |fft2(A)| of the microscopic matrix and the LMMF matrix, the latter
      EXTRAPOLATED from the M×M ensemble mean-coupling Ā_{ml}(T) to the microscopic (block-
      averaged) resolution via soft ensemble responsibilities P(m|ω_i): Ã = P Ā_{ml} Pᵀ. The
      (0,0) spatial-frequency (mean) bin is excluded so the metric measures STRUCTURE, not the
      mean coupling (that is panel c).

The LMMF final matrix Ā_{ml}(T) is not stored in the sweep, so it is recomputed from the
stored mixture parameters via ``kmo_lmmf_variance_bound_sweep.simulate_lmmf`` (cheap, pure
numpy); the coherent-IC coherence R(0) is reconstructed by replaying each trial's RNG. All
per-(rule, μ, budget) metrics are cached to an .npz so re-styling is instant (set FORCE=True
to recompute).

Run in the ``pycobi`` conda env (imports the sweep module):
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_lmmf_variance_bound_figure.py
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import functools, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
from prl_style import set_prl_style as _set_prl_style
from prl_style import panel_label
# ---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import kmo_lmmf_variance_bound_sweep as SW

CSV = dp.mpmf("kmo_lmmf_variance_bound_sweep.csv")
OUT = dp.mpmf("kmo_lmmf_variance_bound_summary")
CACHE = OUT + "_metrics.npz"
FORCE = False                                  # True => recompute metrics (ignore cache)

RULES = ["cos", "sin", "|sin|"]
C_RULE = {"cos": "#1f77b4", "sin": "#e63946", "|sin|": "#2a9d8f"}
LBL = {"cos": r"\cos", "sin": r"\sin", "|sin|": r"|\sin|"}
RES = 100                                      # micro coupling-matrix resolution (block-averaged)


set_prl_style = functools.partial(_set_prl_style, "prl",
                                 **{"legend.fontsize": 5.6})


_panel_label = functools.partial(panel_label, dx=-24, dy=3)


# ════════════════════════════════════════════════════════════════════════════
#  metrics
# ════════════════════════════════════════════════════════════════════════════
def _spectral_rmse(a, b):
    """RMSE between sample-normalized single-sided amplitude spectra of two equal-length signals
    (DC included: the mean coherence is part of R(t))."""
    fa = np.abs(np.fft.rfft(a)) / a.size
    fb = np.abs(np.fft.rfft(b)) / b.size
    return float(np.sqrt(np.mean((fa - fb) ** 2)))


def _spatial_spectral_rmse(A, B):
    """RMSE between normalized 2-D amplitude spectra of two matrices, excluding the (0,0)
    (mean) bin ⇒ measures spatial STRUCTURE, not the mean coupling."""
    d = (np.abs(np.fft.fft2(A)) - np.abs(np.fft.fft2(B))).ravel()[1:] / A.size
    return float(np.sqrt(np.mean(d ** 2)))


def _responsibilities(omega, w, Om, De):
    """Soft ensemble membership P(m|ω_i) = w_m ρ_m(ω_i) / Σ_l w_l ρ_l(ω_i)  (n_osc × M)."""
    comp = w[None, :] * (De[None, :] / np.pi) / ((omega[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    return comp / comp.sum(axis=1, keepdims=True)


def _R0(cfg, trial):
    """Reconstruct the trial's coherent-IC coherence R(0) by replaying the sweep RNG order
    (uniform frequencies drawn first, then the phase IC)."""
    rng = np.random.default_rng(cfg["seed"] + trial)
    _ = np.sort(rng.uniform(-cfg["omega_halfwidth"], cfg["omega_halfwidth"], cfg["N"]))
    theta0 = rng.normal(0.0, cfg["sigma0"], cfg["N"])
    return float(np.abs(np.exp(1j * theta0).mean()))


def compute_metrics(cfg=SW.CONFIG):
    df = pd.read_csv(CSV)
    rules = [r for r in RULES if r in df["G_A"].dropna().unique().tolist()]
    mus = sorted(df["mu"].dropna().unique())
    rvs = sorted(df["rv_max"].dropna().unique())
    trials = sorted(int(t) for t in df["trial"].dropna().unique())

    # pre-group everything once (boolean filtering on the ~1M-row frame is otherwise the bottleneck)
    def _arr(g):
        return g.sort_values("time")["value"].to_numpy()
    R_mic = {k: _arr(g) for k, g in df[df.quantity == "R_micro"].groupby(["trial", "G_A", "mu"])}
    Ab_mic = {k: _arr(g) for k, g in df[df.quantity == "Abar_micro"].groupby(["trial", "G_A", "mu"])}
    R_mf = {k: _arr(g) for k, g in df[df.quantity == "R_mf"].groupby(["trial", "G_A", "mu", "rv_max"])}
    Ab_mf = {k: _arr(g) for k, g in df[df.quantity == "Abar_mf"].groupby(["trial", "G_A", "mu", "rv_max"])}
    omega = {t: g.sort_values("idx")["value"].to_numpy() for t, g in df[df.quantity == "omega"].groupby("trial")}
    mix = {k: g.sort_values("idx") for k, g in df[df.quantity == "mixture"].groupby(["trial", "mu", "rv_max"])}
    Amic = {}
    for k, g in df[df.quantity == "A_final"].groupby(["trial", "G_A", "mu"]):
        M = np.zeros((RES, RES))
        M[g.row.astype(int), g.col.astype(int)] = g["value"].to_numpy()
        Amic[k] = M

    shape = (len(rules), len(mus), len(rvs))
    specR = np.full(shape + (len(trials),), np.nan)
    AbarR = np.full(shape + (len(trials),), np.nan)
    spatR = np.full(shape + (len(trials),), np.nan)
    Mens = np.full((len(mus), len(rvs), len(trials)), np.nan)

    for it, tr in enumerate(trials):
        for jm, mu in enumerate(mus):
            for kv, rv in enumerate(rvs):
                g = mix[(tr, mu, rv)]
                w, Om, De = g.w.to_numpy(), g.Omega.to_numpy(), g.Delta.to_numpy()
                Mens[jm, kv, it] = w.size
                P = _responsibilities(omega[tr], w, Om, De)
                for ir, rule in enumerate(rules):
                    specR[ir, jm, kv, it] = _spectral_rmse(R_mic[(tr, rule, mu)], R_mf[(tr, rule, mu, rv)])
                    AbarR[ir, jm, kv, it] = np.sqrt(np.mean((Ab_mic[(tr, rule, mu)] - Ab_mf[(tr, rule, mu, rv)]) ** 2))
                    _, _, _, _, Aml = SW.simulate_lmmf(w, Om, De, cfg["K"], mu, cfg["gamma"],
                                                       rule, _R0(cfg, tr), cfg["A0"], cfg,
                                                       return_final_A=True)
                    spatR[ir, jm, kv, it] = _spatial_spectral_rmse(Amic[(tr, rule, mu)], P @ Aml @ P.T)
        print(f"  metrics: trial {it + 1}/{len(trials)} done")

    out = dict(rvs=np.array(rvs), rules=np.array(rules), mus=np.array(mus),
               specR_m=np.nanmean(specR, -1), specR_s=np.nanstd(specR, -1),
               AbarR_m=np.nanmean(AbarR, -1), AbarR_s=np.nanstd(AbarR, -1),
               spatR_m=np.nanmean(spatR, -1), spatR_s=np.nanstd(spatR, -1),
               Mens_m=np.nanmean(Mens, -1), Mens_s=np.nanstd(Mens, -1))
    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    np.savez(CACHE, **out)
    print(f"[cached] {CACHE}")
    return out


def load_metrics():
    if os.path.exists(CACHE) and not FORCE:
        print(f"[cache] {CACHE}")
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    return compute_metrics()


# ════════════════════════════════════════════════════════════════════════════
#  figure
# ════════════════════════════════════════════════════════════════════════════
def make_figure(m, mu):
    """One 2×2 summary figure for a single μ (lines coloured by rule)."""
    rvs = m["rvs"]; rules = [str(r) for r in m["rules"]]; mus = [float(x) for x in m["mus"]]
    jm = mus.index(mu)

    fig = plt.figure(figsize=(3.4, 3.1), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.05)
    axes = fig.subplots(2, 2).ravel()
    (axA, axB, axC, axD) = axes

    def plot_rules(ax, mean, std):
        for ir, rule in enumerate(rules):
            ax.errorbar(rvs, mean[ir, jm], yerr=std[ir, jm], color=C_RULE[rule],
                        ls="-", marker="o", ms=2.2, lw=0.9, capsize=1.4, elinewidth=0.5)

    # (a) spectral RMSE of R(t)
    plot_rules(axA, m["specR_m"], m["specR_s"])
    axA.set_ylabel(r"spectral RMSE $R(t)$", labelpad=2)
    _panel_label(axA, "a")

    # (b) number of ensembles M (rule-independent ⇒ a single line at this μ)
    axB.errorbar(rvs, m["Mens_m"][jm], yerr=m["Mens_s"][jm], color="0.2",
                 ls="-", marker="o", ms=2.2, lw=0.9, capsize=1.4, elinewidth=0.5)
    axB.set_ylabel(r"ensembles $M$", labelpad=2)
    _panel_label(axB, "b")

    # (c) RMSE of Ā(t)
    plot_rules(axC, m["AbarR_m"], m["AbarR_s"])
    axC.set_ylabel(r"RMSE $\bar A(t)$", labelpad=2)
    _panel_label(axC, "c")

    # (d) spatial-spectral RMSE of A_ij
    plot_rules(axD, m["spatR_m"], m["spatR_s"])
    axD.set_ylabel(r"spatial-spec. RMSE $A_{ij}$", labelpad=2)
    _panel_label(axD, "d")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(min(rvs) * 0.8, max(rvs) * 1.25)
    for ax in (axC, axD):
        ax.set_xlabel(r"bound on $V_A/\bar A^2$", labelpad=1)
    for ax in (axA, axB):
        ax.set_xticklabels([])

    rule_handles = [Line2D([0], [0], color=C_RULE[r], lw=1.1, label=rf"$G_A={LBL[r]}$") for r in rules]
    axA.legend(handles=rule_handles, loc="best", handlelength=1.6, labelspacing=0.25)
    fig.suptitle(rf"$\mu={mu:g}$", fontsize=8, x=0.01, ha="left")

    out = f"{OUT}_mu{mu:g}"
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {out}.pdf / .png")


if __name__ == "__main__":
    set_prl_style()
    metrics = load_metrics()
    for mu in [float(x) for x in metrics["mus"]]:
        make_figure(metrics, mu)
