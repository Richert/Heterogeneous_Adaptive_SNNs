#!/usr/bin/env python3
r"""
Fit a Lorentzian mixture to an Allen Cell-Types excitability distribution.
=========================================================================

Companion to ``allen_spike_thresholds.py``. For a chosen cell class and cortical
layer, this script

  1. loads the Allen Cell-Types spike thresholds v_θ (``threshold_v_long_square``)
     and resting potentials v_r (``vrest``),
  2. builds the empirical distribution of the excitability gap v_θ − v_r (mV)
     (the depolarisation needed to fire; smaller ⇒ more excitable),
  3. fits a weighted sum of M Lorentzians to it with the Cramér–von Mises (CvM)
     algorithm in ``theory/lorentzian_mixture.py`` (M selected by D + λ·M),
  4. saves the fitted mixture parameters (weights w_m, centres Ω_m, half-widths
     Δ_m) to ``<stem>.npz`` (+ a human-readable ``.txt``) so a downstream script
     can use them directly as the ensembles of an Ott–Antonsen mean-field network,
  5. writes a figure (empirical pdf + fitted mixture, and the empirical vs fitted
     CDF that the CvM criterion matches).

Run in the ``allen`` conda env:
    PATH="$HOME/conda/envs/allen/bin:$PATH" python allen_lorentzian_fit.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi

# CvM Lorentzian-mixture fitter (theory/lorentzian_mixture.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "theory"))
import lorentzian_mixture as LM


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
SPECIES = CellTypesApi.MOUSE          # or CellTypesApi.HUMAN
CELL_CLASS = "Pyramidal"              # "Pyramidal" | "PV+ interneuron" | "SOM interneuron"
LAYER = "L5/6"                        # "L2/3" | "L5/6"
THRESH_FEATURE = "threshold_v_long_square"

# Lorentzian-mixture fit
M_MAX = 10
ALPHA = 0.001                          # GoF acceptance level for greedy model selection: stop at
                                      # the smallest M with 1−p < ALPHA (CvM p-value p > 1−ALPHA,
                                      # T=N·W²). SMALLER ALPHA ⇒ better fit ⇒ more ensembles.
LAMBDA_M = 1e-5                       # per-ensemble complexity penalty: M* = argmin[D(M)+λ·M]
                                      # (D ~ 1e-4…1e-3 here). LARGER ⇒ fewer ensembles.
PATIENCE = 2                          # stop once the penalized loss D+λM has not improved for
                                      # PATIENCE consecutive M, then return the argmin (lowest total).
DELTA_BOUNDS = None                   # (Δ_min, Δ_max) in mV; None → derived from the data spread
N_RESTARTS = 10
SEED = 1
FIT_METHOD = "slsqp"                  # constraint handling: "slsqp" (natural params, box bounds
                                      # + Σw=1 equality) or "softmax" (reparam + L-BFGS-B)
EMP_BINS = 20                         # bins for the saved empirical (normalized-histogram) distribution

MANIFEST = os.path.join(_HERE, "cell_types", "manifest.json")
OUT_DIR = _HERE
C_HIST, C_COMP, C_MIX = "0.8", "#2e6f95", "#c1121f"


# ════════════════════════════════════════════════════════════════════════════
#  load the Allen excitability gap v_θ − v_r for a cell class / layer
# ════════════════════════════════════════════════════════════════════════════
def _layer_group(layer):
    layer = str(layer)
    if layer == "2/3":
        return "L2/3"
    if layer in ("5", "6", "6a", "6b"):
        return "L5/6"
    return None


def _cell_class(row, line_cols):
    line = " ".join(str(row[c]) for c in line_cols if isinstance(row.get(c), str))
    if "Pvalb" in line:
        return "PV+ interneuron"
    if "Sst" in line:
        return "SOM interneuron"
    if row.get("dendrite_type") == "spiny":
        return "Pyramidal"
    return None


def load_excitability_gap(cell_class, layer):
    """Return the (1-D) array of v_θ − v_r (mV) for the given class & layer."""
    ctc = CellTypesCache(manifest_file=MANIFEST)
    cells = pd.DataFrame(ctc.get_cells(species=[SPECIES]))
    ephys = pd.DataFrame(ctc.get_ephys_features())
    df = cells.merge(ephys, left_on="id", right_on="specimen_id", how="inner")

    df["layer_group"] = df["structure_layer_name"].apply(_layer_group)
    line_cols = [c for c in ("transgenic_line", "line_name", "transgenic_line_name")
                 if c in df.columns]
    df["cell_class"] = df.apply(lambda r: _cell_class(r, line_cols), axis=1)
    df["dist_mV"] = df[THRESH_FEATURE] - df["vrest"]

    sel = (df["cell_class"] == cell_class) & (df["layer_group"] == layer)
    samples = df.loc[sel, "dist_mV"].dropna().to_numpy(dtype=float)
    return samples


# ════════════════════════════════════════════════════════════════════════════
#  figure
# ════════════════════════════════════════════════════════════════════════════
def make_figure(samples, model, cell_class, layer, data_loss, out_png):
    gx = np.linspace(np.percentile(samples, 0.5), np.percentile(samples, 99.5), 800)
    comps = (model.w[None, :] * (model.Delta[None, :] / np.pi)
             / ((gx[:, None] - model.Omega[None, :]) ** 2 + model.Delta[None, :] ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")

    # (a) empirical pdf + fitted mixture
    ax1.hist(samples, bins=30, range=(gx[0], gx[-1]), density=True, color=C_HIST,
             edgecolor="none", label=f"data (n={samples.size})", zorder=0)
    for k in range(model.M):
        ax1.plot(gx, comps[:, k], lw=0.8, color=C_COMP, alpha=0.7, zorder=2,
                 label="components" if k == 0 else None)
    ax1.plot(gx, comps.sum(axis=1), lw=1.8, color=C_MIX, zorder=3,
             label=f"Lorentzian mixture (M={model.M})")
    ax1.set_xlabel(r"$v_\theta - v_r$ (mV)")
    ax1.set_ylabel("density")
    ax1.set_title(f"(a) {cell_class}, {layer}")
    ax1.legend(fontsize=8, frameon=False)

    # (b) empirical vs fitted CDF (the CvM-matched quantity)
    xs = np.sort(samples)
    ecdf = (np.arange(xs.size) + 0.5) / xs.size
    ax2.step(xs, ecdf, where="post", color="0.4", lw=1.2, label="empirical CDF")
    ax2.plot(gx, model.cdf(gx), color=C_MIX, lw=1.8, label="mixture CDF")
    ax2.set_xlabel(r"$v_\theta - v_r$ (mV)")
    ax2.set_ylabel("cumulative probability")
    ax2.set_title(f"(b) CvM fit  (D = {data_loss:.2e})")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png} (+ .pdf)")


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def _tag(cell_class, layer):
    c = cell_class.split("+")[0].split()[0].lower()       # Pyramidal→pyramidal, PV+ int→pv
    return f"{c}_{layer.replace('/', '').replace(' ', '')}"


def main():
    samples = load_excitability_gap(CELL_CLASS, LAYER)
    if samples.size < 5:
        raise SystemExit(f"too few cells (n={samples.size}) for {CELL_CLASS} {LAYER}")
    std = float(np.std(samples))
    dbounds = DELTA_BOUNDS or (0.05 * std, std)
    print(f"{CELL_CLASS} {LAYER}: n={samples.size}, "
          f"v_θ−v_r mean={samples.mean():.2f} mV, std={std:.2f} mV, "
          f"Δ-bounds={tuple(round(b, 3) for b in dbounds)}")

    res = LM.fit(samples, dbounds, M_max=M_MAX, alpha=ALPHA, lambda_M=LAMBDA_M, patience=PATIENCE,
                 loss="cvm", n_restarts=N_RESTARTS, seed=SEED, method=FIT_METHOD, verbose=True)
    m = res["model"]
    print(f"selected M={m.M} (GoF α={ALPHA}, λ={LAMBDA_M}, patience={PATIENCE}): "
          f"CvM D={res['data_loss']:.4e}, total={res['total_loss']:.4e}, "
          f"T=N·W²={res['T']:.4f}, p-value={res['pvalue']:.3f}")
    for k in range(m.M):
        print(f"  comp {k}: w={m.w[k]:.3f}  Ω={m.Omega[k]:+.3f} mV  Δ={m.Delta[k]:.3f} mV")

    # empirical distribution of v_θ−v_r as a NORMALIZED histogram (density, ∫=1), so a
    # downstream script can interpolate the counts into an empirical PDF and sample from it.
    emp_pdf, emp_edges = np.histogram(samples, bins=EMP_BINS, density=True)
    emp_centers = 0.5 * (emp_edges[:-1] + emp_edges[1:])

    stem = os.path.join(OUT_DIR, f"allen_lorentzian_{_tag(CELL_CLASS, LAYER)}")
    # parameters for a downstream ensemble mean-field network: weights w_m,
    # ensemble centres Ω_m (= mean v_θ−v_r per ensemble, mV) and half-widths Δ_m;
    # plus the empirical distribution (samples + normalized histogram).
    np.savez(stem + ".npz",
             weights=m.w, omega=m.Omega, delta=m.Delta, M=np.int64(m.M),
             samples=samples, emp_centers=emp_centers, emp_pdf=emp_pdf, emp_edges=emp_edges,
             cell_class=CELL_CLASS, layer=LAYER,
             n_samples=np.int64(samples.size), data_loss=float(res["data_loss"]),
             delta_min=float(dbounds[0]), delta_max=float(dbounds[1]),
             alpha=float(ALPHA), lambda_M=float(LAMBDA_M), patience=np.int64(PATIENCE),
             T=float(res["T"]), pvalue=float(res["pvalue"]),
             total_loss=float(res["total_loss"]), feature=THRESH_FEATURE)
    with open(stem + ".txt", "w") as f:
        f.write(f"# Lorentzian-mixture fit of v_theta - v_r  ({CELL_CLASS}, {LAYER})\n")
        f.write(f"# n_samples={samples.size}  M={m.M}  CvM_loss={res['data_loss']:.6e}\n")
        f.write(f"# GoF: alpha={ALPHA}  lambda_M={LAMBDA_M}  patience={PATIENCE}  "
                f"T=N*W^2={res['T']:.6f}  p-value={res['pvalue']:.6f}\n")
        f.write(f"# delta_bounds=({dbounds[0]:.4f},{dbounds[1]:.4f}) mV\n")
        f.write("# columns: weight  Omega_mV  Delta_mV\n")
        for k in range(m.M):
            f.write(f"{m.w[k]:.6f}  {m.Omega[k]:+.6f}  {m.Delta[k]:.6f}\n")
    print(f"[saved] {stem}.npz / .txt")

    make_figure(samples, m, CELL_CLASS, LAYER, res["data_loss"], stem + ".png")


if __name__ == "__main__":
    main()
