#!/usr/bin/env python3
"""
Plot intrinsic-excitability distributions from the Allen Cell Types Database.

Compares Layer 2/3 vs Layer 5/6 for three cell classes:
    - Pyramidal (excitatory)  -> dendrite_type == 'spiny'
    - PV+ interneurons        -> transgenic (Cre) line contains 'Pvalb'
    - SOM interneurons        -> transgenic (Cre) line contains 'Sst'

Three metrics are plotted as a 3 x 3 grid (rows = metric, cols = cell class):
    1. Spike threshold      = threshold_v_long_square (mV)
    2. Resting potential    = vrest (mV)
    3. Threshold - rest     = depolarisation needed to reach threshold (mV);
                              smaller values indicate higher excitability.

The first two come from Allen's precomputed feature table, so this script
downloads only small metadata/CSV files, NOT the multi-GB raw NWB sweep files.

Requirements:
    pip install allensdk pandas numpy matplotlib seaborn
(allensdk is happiest on Python 3.8-3.11.)

First run downloads metadata into ./cell_types/ and caches it for reuse.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SPECIES = CellTypesApi.MOUSE          # switch to CellTypesApi.HUMAN to compare human cells
THRESH_FEATURE = "threshold_v_long_square"   # AP threshold voltage (mV), long-square protocol
OUT_PNG = "allen_spike_threshold_distributions.png"

CLASSES = ["Pyramidal", "PV+ interneuron", "SOM interneuron"]
PALETTE = {"L2/3": "#4C72B0", "L5/6": "#C44E52"}

# ----------------------------------------------------------------------
# 1. Load metadata + precomputed electrophysiology features
# ----------------------------------------------------------------------
ctc = CellTypesCache(manifest_file="cell_types/manifest.json")

cells = pd.DataFrame(ctc.get_cells(species=[SPECIES]))
ephys = pd.DataFrame(ctc.get_ephys_features())

# Helpful when adapting to a different allensdk version: see what's available.
print("cells columns:", list(cells.columns))
print("ephys columns includes threshold:", THRESH_FEATURE in ephys.columns)

# get_cells uses 'id' for the specimen; ephys features use 'specimen_id'.
df = cells.merge(ephys, left_on="id", right_on="specimen_id", how="inner")

# ----------------------------------------------------------------------
# 2. Classify layer group  (L2/3 vs L5/6; L1 and L4 are dropped)
# ----------------------------------------------------------------------
def layer_group(layer):
    layer = str(layer)
    if layer == "2/3":
        return "L2/3"
    if layer in ("5", "6", "6a", "6b"):
        return "L5/6"
    return None

df["layer_group"] = df["structure_layer_name"].apply(layer_group)

# ----------------------------------------------------------------------
# 3. Classify cell class
#    PV / SOM via Cre line; everything spiny that isn't PV/SOM -> Pyramidal.
# ----------------------------------------------------------------------
# Gather any column that may carry the transgenic line text (version-robust).
LINE_COLS = [c for c in ("transgenic_line", "line_name", "transgenic_line_name")
             if c in df.columns]

def line_text(row):
    parts = [str(row[c]) for c in LINE_COLS if isinstance(row.get(c), str)]
    return " ".join(parts)

def cell_class(row):
    line = line_text(row)
    if "Pvalb" in line:
        return "PV+ interneuron"
    if "Sst" in line:
        return "SOM interneuron"
    if row.get("dendrite_type") == "spiny":
        return "Pyramidal"
    return None

df["cell_class"] = df.apply(cell_class, axis=1)

# ----------------------------------------------------------------------
# 4. Filter to usable rows and build the three metrics
#    - threshold_mV : AP threshold voltage
#    - vrest_mV     : resting membrane potential (Allen feature 'vrest')
#    - dist_mV      : threshold - rest = depolarisation needed to fire
#                     (smaller = more excitable)
# ----------------------------------------------------------------------
keep = df["layer_group"].notna() & df["cell_class"].notna()
d = (df.loc[keep, ["cell_class", "layer_group", THRESH_FEATURE, "vrest"]]
       .rename(columns={THRESH_FEATURE: "threshold_mV", "vrest": "vrest_mV"})
       .copy())
d["dist_mV"] = d["threshold_mV"] - d["vrest_mV"]

# (column, axis label) for each row of the figure
METRICS = [
    ("threshold_mV", "Spike threshold (mV)"),
    ("vrest_mV",     "Resting potential (mV)"),
    ("dist_mV",      "Threshold \u2212 rest (mV)"),
]

# ----------------------------------------------------------------------
# 5. Report sample sizes and summary stats (the undersampling check)
# ----------------------------------------------------------------------
for col, lbl in METRICS:
    sub = d.dropna(subset=[col])
    counts = (sub.groupby(["cell_class", "layer_group"]).size()
                 .unstack(fill_value=0)
                 .reindex(CLASSES))
    summary = (sub.groupby(["cell_class", "layer_group"])[col]
                  .agg(["count", "mean", "median", "std"]))
    print(f"\n=== {lbl}: cell counts per group ===")
    print(counts)
    print(f"\n=== {lbl}: summary ===")
    print(summary)

# ----------------------------------------------------------------------
# 6. Plot: rows = metrics, columns = cell classes; L2/3 vs L5/6 overlaid.
#    x-axis is shared within each row (same metric -> same scale).
# ----------------------------------------------------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(len(METRICS), len(CLASSES),
                         figsize=(15, 12), sharex="row")

for i, (col, xlabel) in enumerate(METRICS):
    for j, cls in enumerate(CLASSES):
        ax = axes[i, j]
        sub = d[d["cell_class"] == cls]

        # Shared bin edges across both layer groups in this panel.
        panel_vals = sub[col].dropna()
        bins = np.histogram_bin_edges(panel_vals, bins=20) if len(panel_vals) else 20

        for lg in ("L2/3", "L5/6"):
            vals = sub.loc[sub["layer_group"] == lg, col].dropna()
            label = f"{lg} (n={len(vals)})"
            if len(vals) >= 2:
                # Histogram (density-normalised) ...
                sns.histplot(vals, ax=ax, bins=bins, stat="density",
                             color=PALETTE[lg], alpha=0.30, edgecolor="none",
                             label=label)
                # ... with the estimated PDF (KDE) line on top.
                sns.kdeplot(vals, ax=ax, color=PALETTE[lg], lw=2, cut=0)
            elif len(vals) == 1:
                ax.axvline(vals.iloc[0], color=PALETTE[lg], ls="--", label=label)
            else:
                ax.plot([], [], color=PALETTE[lg], label=label)  # keep legend entry

        if i == 0:
            ax.set_title(cls)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if j == 0 else "")
        ax.legend(fontsize=8)

fig.suptitle("Allen Cell Types — intrinsic excitability measures by layer",
             y=1.01, fontsize=14)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved figure to {OUT_PNG}")
plt.show()