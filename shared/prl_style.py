r"""
Shared PRL figure style for the PRL_2026 manuscripts
=====================================================

Single home for the matplotlib style and the panel-label convention that used to be
copy-pasted as ``set_prl_style()`` / ``_panel_label()`` into every figure script
(26 and 18 copies respectively before this module existed).

Conventions (see the manuscripts in ~/OneDrive/manuscripts/PRL_2026):
  * STIX serif text + math, tick marks pointing inwards, no legend frame,
    Type-42 fonts so the SVG/PDF stays editable in Inkscape.
  * Panel labels are BOLD and sit OUTSIDE the axis box, above its top-left corner.
  * Column widths: ``COL_SINGLE`` = 3.4 in, ``COL_DOUBLE`` = 7.0 in.

Usage
-----
    from prl_style import set_prl_style, panel_label, COL_SINGLE
    set_prl_style()                                  # canonical tight PRL style
    set_prl_style(**{"legend.fontsize": 5.6})        # per-figure tweak
    set_prl_style("diagnostic")                      # net-vs-mean-field diagnostics
    panel_label(ax, "a", dx=-22, dy=4)

Presets
-------
``"prl"``         canonical single/two-column style (the style of most figures).
``"prl_wide"``    slightly larger type, matplotlib's default tick marks; used by the
                  Skardal benchmark figures.
``"diagnostic"``  larger type / thicker lines for the QIF-network-vs-mean-field
                  comparison scripts, which are read on screen rather than in print.
"""

import matplotlib.pyplot as plt

# PRL column widths in inches
COL_SINGLE = 3.4
COL_DOUBLE = 7.0

# shared by every preset
_BASE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
}

PRESETS = {
    # canonical tight style: 21 of the 26 original copies were exactly this
    "prl": {
        **_BASE,
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    },
    # Skardal benchmark figures: larger type, matplotlib's default tick geometry
    "prl_wide": {
        **_BASE,
        "font.serif": ["STIXGeneral", "Times", "DejaVu Serif"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5,
    },
    # PNAS: sans-serif (Arial / Helvetica; Liberation Sans is metric-compatible with Arial), 6-8 pt text,
    # columns 8.7 cm (PNAS_SINGLE) / 17.8 cm (PNAS_DOUBLE); panel labels: bold capitals (pnas_panel_label)
    "pnas": {
        **_BASE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    },
    # on-screen diagnostics (spiking net vs. mean field)
    "diagnostic": {
        **_BASE,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "figure.dpi": 140,
    },
}


def set_prl_style(preset="prl", **overrides):
    """Apply a PRL style preset, then any per-figure rcParam overrides.

    ``overrides`` are plain rcParam keys, e.g. ``set_prl_style(**{"font.size": 8})``.
    """
    try:
        params = dict(PRESETS[preset])
    except KeyError:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}") from None
    params.update(overrides)
    plt.rcParams.update(params)


def panel_label(ax, letter, dx=-16, dy=4, fontsize=8, weight="bold"):
    """Bold panel label OUTSIDE the axis box, above its top-left corner (PRL convention).

    ``dx``/``dy`` are offsets in points from the axis' top-left corner; they need
    per-figure tuning because the required clearance depends on the tick labels.
    """
    return ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                       xytext=(dx, dy), textcoords="offset points",
                       fontsize=fontsize, fontweight=weight, ha="left", va="bottom")


# PNAS column widths in inches (8.7 cm / 17.8 cm)
PNAS_SINGLE = 3.42
PNAS_DOUBLE = 7.0


def pnas_panel_label(ax, letter, dx=-16, dy=4, fontsize=9):
    """PNAS panel label: bold capital letter without parentheses, above the axis' top-left corner."""
    return ax.annotate(letter.upper(), xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                       textcoords="offset points", fontsize=fontsize, fontweight="bold", ha="left", va="bottom")
