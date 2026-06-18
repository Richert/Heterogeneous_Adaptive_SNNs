r"""
PV interneuron — journal figure 1: the heterogeneity re-parameterization principle
==================================================================================

Single-column PRL-style 2x2 figure (demonstrated on the L5/6 PV+ interneuron Lorentzian fit):

  (a) the fitted Lorentzian mixture (data histogram + components + mixture density),
  (b) its re-parameterization into per-component {weight w_m, centre eta_m, width Delta_m},
      with the two heterogeneity knobs marked:
          widths        Delta_m -> h_Delta * Delta_m
          centre spread eta_m   -> etabar + h_etabar * (eta_m - etabar),   etabar = sum_m w_m eta_m
  (c) distorting the mixture by rescaling the WIDTHS   (h_Delta sweep, centres fixed),
  (d) distorting the mixture by rescaling the CENTRES  (h_etabar sweep, widths fixed).

Pure numpy/matplotlib; reads only the Lorentzian fit .npz -> runs in the `allen` env.

    python pv_heterogeneity_principle.py            # PV+ Interneuron, L5/6
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIT_DIR = os.path.join(os.path.dirname(_HERE), "data_fitting")

CELL_CLASS = sys.argv[1] if len(sys.argv) > 1 else "PV+ Interneuron"
LAYER = sys.argv[2] if len(sys.argv) > 2 else "L5/6"
C_LAYER = "#e8702a"                       # L5/6 colour (matches the summary figure)
C_MEAN = "#333333"
CMAP = plt.cm.managua                     # heterogeneity-knob colour scale (diverging, centred at h=1)
HMAX = 1.5                                # h>1 = MORE heterogeneity than the data fit
NORM = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=HMAX)   # data fit (h=1) at the colour-map centre
HD_SWEEP = [1.5, 1.0, 0.5, 0.2]           # width-rescaling values (1.5 = broader than data)
HC_SWEEP = [1.5, 1.0, 0.5, 0.0]           # centre-spread-rescaling values (1.5 = wider spread)
ETA_VIEW = (12.0, 51.0)                   # eta range (widened so the h=1.5 centre spread fits)


def _tag(cell_class, layer):
    c = cell_class.split("+")[0].split()[0].lower()
    return f"{c}_{layer.replace('/', '').replace(' ', '')}"


def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "legend.fontsize": 5.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    })


def mixture(eta, w, Om, De):
    """Lorentzian-mixture density rho(eta) = sum_m w_m (De_m/pi)/((eta-Om_m)^2 + De_m^2)."""
    comps = w[None, :] * (De[None, :] / np.pi) / ((eta[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    return comps


def reparam(Om, De, etabar, hD=1.0, hC=1.0):
    """Apply the two heterogeneity knobs: widths *= hD, centre spread about etabar *= hC."""
    return etabar + hC * (Om - etabar), hD * De


def _panel_label(ax, letter):
    """PRL convention: bold panel label OUTSIDE the axis box, above its top-left corner."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-16, 7),
                textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="bottom")


def main():
    set_prl_style()
    tag = _tag(CELL_CLASS, LAYER)
    d = np.load(os.path.join(_FIT_DIR, f"allen_lorentzian_{tag}.npz"), allow_pickle=False)
    w = d["weights"].astype(float); w = w / w.sum()
    Om = d["omega"].astype(float); De = d["delta"].astype(float)
    xs = d["samples"].astype(float); M = len(w)
    etabar = float(w @ Om)
    eta = np.linspace(*ETA_VIEW, 800)

    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.5), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.06)

    # ── (a) the Lorentzian-mixture fit ──────────────────────────────────────
    ax = axs[0][0]
    comps = mixture(eta, w, Om, De)
    ax.hist(xs, bins=22, range=ETA_VIEW, density=True, color="0.85", edgecolor="none", zorder=0)
    for k in range(M):
        ax.plot(eta, comps[:, k], color=C_LAYER, lw=0.5, alpha=0.45, zorder=2)
    ax.plot(eta, comps.sum(1), color=C_LAYER, lw=1.5, zorder=3)
    ax.set_xlim(*ETA_VIEW); ax.set_yticks([])
    ax.set_ylabel(r"$\rho(\eta)$", labelpad=2)
    ax.set_title("Lorentzian-mixture fit", pad=2)

    # ── (b) re-parameterization into {w_m, eta_m, Delta_m} ──────────────────
    ax = axs[0][1]
    cb = mixture(eta, w, Om, De)                                # component PDFs
    ymax = float(cb.max())
    for k in range(M):                                          # component PDFs (grey) in background
        ax.plot(eta, cb[:, k], color="0.6", lw=0.8, alpha=0.7, zorder=1)
    # mean centre eta-bar (dashed); label to the RIGHT of the line
    ax.axvline(etabar, color=C_MEAN, ls="--", lw=0.9, zorder=3)
    ax.text(etabar + 0.7, ymax * 1.18, r"$\bar\eta=\sum_m w_m \eta_m$", color=C_MEAN,
            ha="left", va="center", fontsize=5.5)
    # Delta*_m: double-headed arrow at the half-width of a (prominent) component left of eta-bar
    left = np.where(Om < etabar)[0]
    kd = int(left[np.argmax(De[left])])                         # widest centre left of the mean
    yhalf = 0.5 * w[kd] / (np.pi * De[kd])
    ax.annotate("", xy=(Om[kd] - 1.5*De[kd], yhalf), xytext=(Om[kd] + 1.5*De[kd], yhalf),
                arrowprops=dict(arrowstyle="<->", color=C_LAYER, lw=0.5))
    ax.text(Om[kd] - 8.0, yhalf + 0.06 * ymax, r"$\Delta^*_m = h_\Delta\,\Delta_m$",
            color=C_LAYER, ha="left", va="bottom", fontsize=5.5)        # above arrow, shifted left of centre
    # eta*_m: dotted vertical line at the LEFT-MOST component's centre + its rescaling rule
    k0 = int(np.argmin(Om))
    ax.plot([Om[k0], Om[k0]], [0, ymax * 1.12], color=C_LAYER, ls=":", lw=1.0, zorder=3)
    ax.text(ETA_VIEW[0] + 1.0, ymax * 1.30, r"$\eta^*_m = h_\eta\,\eta_m + (1-h_\eta)\,\bar\eta$",
            color=C_LAYER, ha="left", va="center", fontsize=5.5)
    ax.set_xlim(*ETA_VIEW); ax.set_ylim(0, ymax * 1.62); ax.set_yticks([])
    ax.set_title("re-parameterization", pad=2)

    # ── (c) width rescaling h_Delta / (d) centre rescaling h_etabar ─────────
    def sweep(ax, values, which, title):
        for h in values:
            Omh, Deh = reparam(Om, De, etabar, hD=h, hC=1.0) if which == "hD" \
                else reparam(Om, De, etabar, hD=1.0, hC=h)
            ax.plot(eta, mixture(eta, w, Omh, Deh).sum(1), color=CMAP(NORM(h)),
                    lw=(1.8 if h == 1.0 else 1.0), zorder=int(20 * abs(1 - h)) + 2)
        ax.set_xlim(*ETA_VIEW); ax.set_yticks([])
        ax.set_xlabel(r"$\eta$ (mV)", labelpad=1); ax.set_title(title, pad=2)

    sweep(axs[1][0], HD_SWEEP, "hD", r"width rescaling $h_\Delta$")
    axs[1][0].set_ylabel(r"$\rho(\eta)$", labelpad=2)
    sweep(axs[1][1], HC_SWEEP, "hC", r"centre rescaling $h_\eta$")

    # shared slim colourbar for the knob value
    sm = ScalarMappable(norm=NORM, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=axs[1, :], location="bottom", fraction=0.08, pad=0.02, aspect=40)
    cb.set_label(r"global heterogeneity parameter $h$", labelpad=1); cb.set_ticks([0, 0.5, 1.0, 1.5])

    for ax, lt in zip([axs[0][0], axs[0][1], axs[1][0], axs[1][1]], "abcd"):
        _panel_label(ax, lt)

    out = os.path.join(_HERE, f"pv_heterogeneity_principle_{tag}")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"[saved] {os.path.basename(out)}.{{png,svg}}")


if __name__ == "__main__":
    main()
