"""
Plot the result of the QIF / OA / WC sweep produced by
qif_stdp_mean_field_evaluation.py.

This script exposes three top-level plotting entry points:

    plot_heatmaps_only(sweep_path, examples, ...)
        Standalone 2 x 2 heatmap summary (correlational distance and RMSE,
        for OA and WC vs QIF). Designed for the top-right poster slot.

    plot_examples_only(sweep_path, examples, ...)
        Standalone 4 x 6 grid of example panels: <s>(t) trace + three
        final-A matrices for each of the user-specified (tau_s, M) cells.
        Designed to fill the slot currently occupied by the combined
        sweep summary on the poster.

    plot_sweep_summary(sweep_path, examples, ...)
        Original combined 4 x 10 figure with both heatmaps (left block) and
        example panels (right block) in one figure. Retained for non-poster
        use cases.

Each function takes the same kwargs (sweep_path, examples, savepath, show,
dpi, fontsize); the standalone functions additionally take figsize.

Metrics:
    correlational distance : 1 - Pearson correlation of flattened A matrices
    RMSE                   : sqrt(mean((s_qif(t) - s_other(t))^2))
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def matrix_corr_dist(A1, A2):
    """
    Correlational distance between two matrices: d = 1 - r, where r is the
    Pearson correlation of the flattened matrices. For real-valued matrices
    that are positively related (as the plasticity-driven structures here),
    d sits in [0, 1] with 0 = perfect agreement, 1 = uncorrelated. Negative
    correlations would push d above 1, but this doesn't happen for our
    plasticity matrices.
    """
    a, b = A1.ravel(), A2.ravel()
    if a.std() < 1e-15 or b.std() < 1e-15:
        return np.nan
    r = float(np.corrcoef(a, b)[0, 1])
    return 1.0 - r


def trace_rmse(s1, s2):
    """RMSE between two time series of equal length."""
    s1, s2 = np.asarray(s1), np.asarray(s2)
    n = min(s1.size, s2.size)
    diff = s1[:n] - s2[:n]
    return float(np.sqrt((diff * diff).mean()))


def collect_sweep_metrics(cells, tau_values, M_values):
    """
    Build 4 metric grids of shape (len(tau_values), len(M_values)):
        dist_mpr[i, j] = 1 - corr(A_qif, A_mpr) at (tau_values[i], M_values[j])
        dist_wc [i, j] = 1 - corr(A_qif, A_wc )
        rmse_mpr[i, j] = RMSE(s_qif, s_mpr)
        rmse_wc [i, j] = RMSE(s_qif, s_wc )
    """
    nt, nm = len(tau_values), len(M_values)
    dist_mpr = np.full((nt, nm), np.nan)
    dist_wc  = np.full((nt, nm), np.nan)
    rmse_mpr = np.full((nt, nm), np.nan)
    rmse_wc  = np.full((nt, nm), np.nan)

    for i, tau in enumerate(tau_values):
        for j, M in enumerate(M_values):
            key = (tau, M)
            if key not in cells:
                continue
            c = cells[key]
            dist_mpr[i, j] = matrix_corr_dist(c["A_final_qif_cg"],
                                                c["A_final_mpr"])
            dist_wc [i, j] = matrix_corr_dist(c["A_final_qif_cg"],
                                                c["A_final_wc"])
            rmse_mpr[i, j] = trace_rmse(c["s_mean_qif"], c["s_mean_mpr"])
            rmse_wc [i, j] = trace_rmse(c["s_mean_qif"], c["s_mean_wc"])

    return dict(dist_mpr=dist_mpr, dist_wc=dist_wc,
                rmse_mpr=rmse_mpr, rmse_wc=rmse_wc)


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap helper
# ─────────────────────────────────────────────────────────────────────────────
def _draw_heatmap(ax, data, tau_values, M_values, title,
                   cmap, vmin, vmax, example_marks=None):
    """
    Render a (len(tau_values) x len(M_values)) heatmap with M on the
    x-axis (columns) and tau_s on the y-axis (rows).

    Parameters
    ----------
    example_marks : dict or None
        Optional mapping {(tau_s, M): "a"|"b"|"c"|"d"} indicating which cell
        corresponds to which example. The letter is overlaid on the cell with
        a high-contrast color chosen from the cell's colormap value.
    """
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect="auto", interpolation="nearest", origin="upper")
    ax.set_xticks(np.arange(len(M_values)))
    ax.set_xticklabels([str(m) for m in M_values])
    ax.set_yticks(np.arange(len(tau_values)))
    ax.set_yticklabels([str(t) for t in tau_values])
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$\tau_s$")
    ax.set_title(title)

    if example_marks:
        for (tau, M), letter in example_marks.items():
            if tau not in tau_values or M not in M_values:
                continue
            i = tau_values.index(tau)
            j = M_values.index(M)
            v = data[i, j]
            # Text color: pick black/white based on the cell's brightness
            if not np.isnan(v):
                norm = (v - vmin) / (vmax - vmin + 1e-12)
                color = "white" if (norm < 0.3 or norm > 0.7) else "black"
            else:
                color = "0.2"
            ax.text(j, i, letter, ha="center", va="center",
                     fontsize=13, fontweight="bold", color=color)

    return im


# ─────────────────────────────────────────────────────────────────────────────
# Example panel (single cell): wide <s>(t) on top, 3 matrices below
# ─────────────────────────────────────────────────────────────────────────────
def _draw_example_panel(fig, gs_trace, gs_matrices, cell, c_qif, c_oa, c_wc,
                          letter=None):
    """
    Draw one example pair: trace overlay on `gs_trace` (a single SubplotSpec
    spanning the trace row over 3 columns) and three final-A matrices on
    `gs_matrices` (a SubplotSpec covering the matrix row over 3 columns,
    which we subdivide into 3 equal sub-cells).
    """
    # Trace axis
    ax_s = fig.add_subplot(gs_trace)
    t = cell["t_rec"]
    ax_s.plot(t, cell["s_mean_qif"], color=c_qif, lw=1.3, label="QIF")
    ax_s.plot(t, cell["s_mean_mpr"], color=c_oa,  lw=1.3, ls="--", label="OA")
    ax_s.plot(t, cell["s_mean_wc"],  color=c_wc,  lw=1.3, ls=":",  label="WC")
    ax_s.set_xlabel(r"time $t$")
    ax_s.set_ylabel(r"$\langle s\rangle$")
    ax_s.grid(alpha=0.25)
    ax_s.legend(loc="best", frameon=False, ncol=3)
    prefix = f"({letter}) " if letter else ""
    ax_s.set_title(fr"{prefix}$\tau_s={cell['tau_s']}$, $M={cell['M']}$")

    # Matrices: subdivide the matrix SubplotSpec into 3 cells
    sub = gs_matrices.subgridspec(nrows=1, ncols=3, wspace=0.10)
    A_qif = cell["A_final_qif_cg"]
    A_oa  = cell["A_final_mpr"]
    A_wc  = cell["A_final_wc"]
    A_all = np.stack([A_qif, A_oa, A_wc])
    vmin, vmax = float(A_all.min()), float(A_all.max())
    if vmin < 0.5 < vmax:
        half = max(0.5 - vmin, vmax - 0.5)
        vmin, vmax = 0.5 - half, 0.5 + half
        cmap = "RdBu_r"
    else:
        cmap = "viridis"

    titles = ["QIF (block-avg)", "OA", "WC"]
    mats   = [A_qif, A_oa, A_wc]
    last_im = None
    last_ax = None
    M_ = A_qif.shape[0]
    tick_step = max(1, M_ // 5)
    ticks = np.arange(0, M_, tick_step)
    for k, (T_, A_) in enumerate(zip(titles, mats)):
        ax = fig.add_subplot(sub[0, k])
        last_im = ax.imshow(A_, cmap=cmap, vmin=vmin, vmax=vmax,
                             interpolation="nearest", aspect="equal",
                             origin="upper")
        last_ax = ax
        ax.set_title(T_)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        if k == 0:
            ax.set_ylabel(r"$m$ (post)")
        ax.set_xlabel(r"$n$ (pre)")

    # Colorbar attached to the rightmost matrix; constrained layout will
    # space it correctly.
    if last_im is not None:
        fig.colorbar(last_im, ax=last_ax, fraction=0.046, pad=0.04)


# ─────────────────────────────────────────────────────────────────────────────
# Main: load + render
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (used by the standalone plotting entry points below)
# ─────────────────────────────────────────────────────────────────────────────
def _load_sweep(sweep_path):
    with open(sweep_path, "rb") as fh:
        out = pickle.load(fh)
    cells = out["cells"]
    cfg = out["config"]
    tau_values = list(cfg["tau_values"])
    M_values   = list(cfg["M_values"])
    return cells, tau_values, M_values


def _validate_examples(examples, cells):
    if len(examples) != 4:
        raise ValueError(f"Need exactly 4 example (tau_s, M) tuples, "
                          f"got {len(examples)}")
    for ex in examples:
        if tuple(ex) not in cells:
            avail = sorted(cells.keys())
            raise KeyError(
                f"Example {ex} not in sweep cells. Available: {avail}"
            )


def _rc_for_fontsize(fontsize):
    return {
        "font.size":       fontsize,
        "axes.labelsize":  fontsize,
        "axes.titlesize":  fontsize,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "legend.fontsize": fontsize * 0.9,
        "figure.titlesize": fontsize * 1.2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone: heatmap summary (for the top-right poster slot)
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmaps_only(sweep_path, examples,
                        savepath=None, show=True, dpi=150,
                        fontsize=10, figsize=(13, 6)):
    """
    Build a standalone figure containing only the four summary heatmaps
    (correlational distance and RMSE for OA and WC vs QIF), in a 2x2
    arrangement with shared colorbars per row.

    Designed to fit into the empty top-right "Sequence Encoding" slot of
    the poster. The default figsize is landscape; if the poster slot has
    a different aspect, override via the `figsize` kwarg.

    Heatmap cells corresponding to the four examples (a, b, c, d) are
    annotated so they can be cross-referenced with the example figure.

    Parameters
    ----------
    examples : sequence of 4 (tau_s, M) tuples
        Same as for plot_examples_only / plot_sweep_summary: used to label
        cells a-d on each heatmap.
    figsize : tuple
        Default (13, 6) is landscape.
    """
    cells, tau_values, M_values = _load_sweep(sweep_path)
    _validate_examples(examples, cells)
    metrics = collect_sweep_metrics(cells, tau_values, M_values)

    letters = ["a", "b", "c", "d"]
    example_marks = {tuple(ex): letters[i] for i, ex in enumerate(examples)}

    with plt.rc_context(_rc_for_fontsize(fontsize)):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2)

        ax_dist_oa = fig.add_subplot(gs[0, 0])
        ax_dist_wc = fig.add_subplot(gs[0, 1])
        ax_rmse_oa = fig.add_subplot(gs[1, 0])
        ax_rmse_wc = fig.add_subplot(gs[1, 1])

        dist_all = np.concatenate([metrics["dist_mpr"].ravel(),
                                     metrics["dist_wc"].ravel()])
        dist_all = dist_all[~np.isnan(dist_all)]
        dist_vmin = 0.0
        dist_vmax = float(max(dist_all.max(), 1e-6)) if dist_all.size else 1.0

        rmse_all = np.concatenate([metrics["rmse_mpr"].ravel(),
                                     metrics["rmse_wc"].ravel()])
        rmse_all = rmse_all[~np.isnan(rmse_all)]
        rmse_vmin = 0.0
        rmse_vmax = float(rmse_all.max()) if rmse_all.size else 1.0

        im_dist = _draw_heatmap(
            ax_dist_oa, metrics["dist_mpr"], tau_values, M_values,
            r"$1-\mathrm{corr}(A_\mathrm{QIF}, A_\mathrm{OA})$",
            cmap="magma_r", vmin=dist_vmin, vmax=dist_vmax,
            example_marks=example_marks,
        )
        _draw_heatmap(
            ax_dist_wc, metrics["dist_wc"], tau_values, M_values,
            r"$1-\mathrm{corr}(A_\mathrm{QIF}, A_\mathrm{WC})$",
            cmap="magma_r", vmin=dist_vmin, vmax=dist_vmax,
            example_marks=example_marks,
        )
        im_rmse = _draw_heatmap(
            ax_rmse_oa, metrics["rmse_mpr"], tau_values, M_values,
            r"RMSE$(s_\mathrm{QIF}, s_\mathrm{OA})$",
            cmap="magma_r", vmin=rmse_vmin, vmax=rmse_vmax,
            example_marks=example_marks,
        )
        _draw_heatmap(
            ax_rmse_wc, metrics["rmse_wc"], tau_values, M_values,
            r"RMSE$(s_\mathrm{QIF}, s_\mathrm{WC})$",
            cmap="magma_r", vmin=rmse_vmin, vmax=rmse_vmax,
            example_marks=example_marks,
        )

        fig.colorbar(im_dist, ax=ax_dist_wc, fraction=0.046, pad=0.04)
        fig.colorbar(im_rmse, ax=ax_rmse_wc, fraction=0.046, pad=0.04)

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi)
            print(f"Figure saved -> {savepath}")
        if show:
            plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Standalone: example traces + connectivity matrices
# ─────────────────────────────────────────────────────────────────────────────
def plot_examples_only(sweep_path, examples,
                        savepath=None, show=True, dpi=150,
                        fontsize=10, figsize=(22, 11)):
    """
    Build a standalone figure containing the four example panels (raw
    synaptic-activation traces + final-A matrices) for the user-specified
    (tau_s, M) cells. Designed to fill the slot currently occupied by the
    full sweep summary on the poster.

    Layout: 4 rows x 6 cols, where each example occupies a 2-row x 3-col
    block (top sub-row = wide <s>(t) trace, bottom sub-row = three matrices).

        ┌──────────────┬──────────────┐
        │  Example a   │  Example b   │
        │  trace       │  trace       │
        │  matrices    │  matrices    │
        ├──────────────┼──────────────┤
        │  Example c   │  Example d   │
        │  trace       │  trace       │
        │  matrices    │  matrices    │
        └──────────────┴──────────────┘

    Parameters
    ----------
    figsize : tuple
        Default (22, 11) is landscape; matches the slot that the original
        combined-summary figure occupied on the poster.
    """
    cells, _tau_values, _M_values = _load_sweep(sweep_path)
    _validate_examples(examples, cells)

    letters = ["a", "b", "c", "d"]

    with plt.rc_context(_rc_for_fontsize(fontsize)):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(
            nrows=4, ncols=6,
            height_ratios=[1.0, 2.0, 1.0, 2.0],
        )

        tab10 = plt.get_cmap("tab10")
        c_qif, c_oa, c_wc = tab10(0), tab10(1), tab10(2)

        # Each example: top row gets the trace, the row below gets the matrices
        slots = [
            (gs[0, 0:3], gs[1, 0:3]),   # (a) top-left
            (gs[0, 3:6], gs[1, 3:6]),   # (b) top-right
            (gs[2, 0:3], gs[3, 0:3]),   # (c) bottom-left
            (gs[2, 3:6], gs[3, 3:6]),   # (d) bottom-right
        ]

        for letter, ex, (gs_tr, gs_mat) in zip(letters, examples, slots):
            cell = cells[tuple(ex)]
            _draw_example_panel(fig, gs_tr, gs_mat, cell,
                                  c_qif, c_oa, c_wc, letter=letter)

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi)
            print(f"Figure saved -> {savepath}")
        if show:
            plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Combined: original 4x10 summary (retained for non-poster use)
# ─────────────────────────────────────────────────────────────────────────────
def plot_sweep_summary(sweep_path, examples,
                        savepath=None, show=True, dpi=150, fontsize=10):
    """
    Build the 4 x 10 summary figure described in the script docstring.

    Parameters
    ----------
    sweep_path : str or Path
        Path to the .pkl produced by run_sweep.
    examples : sequence of 4 (tau_s, M) tuples
        Parameter combinations whose raw traces + matrices are plotted in
        the right six columns.
    savepath : str or None
        If given, save the figure to this path.
    show : bool
        Call plt.show() after rendering (default True).
    fontsize : float
        Base font size used for labels, ticks, legend, and title. Applied
        via a local plt.rc_context so it does not leak to other figures.
    """
    with open(sweep_path, "rb") as fh:
        out = pickle.load(fh)

    cells = out["cells"]
    cfg = out["config"]
    tau_values = list(cfg["tau_values"])
    M_values   = list(cfg["M_values"])

    if len(examples) != 4:
        raise ValueError(f"Need exactly 4 example (tau_s, M) tuples, "
                          f"got {len(examples)}")
    for ex in examples:
        if tuple(ex) not in cells:
            avail = sorted(cells.keys())
            raise KeyError(
                f"Example {ex} not in sweep cells. Available: {avail}"
            )

    metrics = collect_sweep_metrics(cells, tau_values, M_values)

    # Map each example to a letter for the heatmap overlay
    letters = ["a", "b", "c", "d"]
    example_marks = {tuple(ex): letters[i] for i, ex in enumerate(examples)}

    # Local rc context so the fontsize change doesn't leak to other figures
    rc = {
        "font.size":       fontsize,
        "axes.labelsize":  fontsize,
        "axes.titlesize":  fontsize,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "legend.fontsize": fontsize * 0.9,
        "figure.titlesize": fontsize * 1.2,
    }

    with plt.rc_context(rc):
        # Constrained layout: lets matplotlib handle spacing automatically;
        # avoids leaking labels into neighbouring axes.
        fig = plt.figure(figsize=(22, 11), constrained_layout=True)
        gs = fig.add_gridspec(
            nrows=4, ncols=10,
            width_ratios=[1.0] * 10,
            height_ratios=[1.0, 1.1, 1.0, 1.1],
        )

        # ── Cols 1-4: summary heatmaps (each spans 2 cols x 2 rows) ──────────
        ax_dist_oa  = fig.add_subplot(gs[0:2, 0:2])
        ax_dist_wc  = fig.add_subplot(gs[0:2, 2:4])
        ax_rmse_oa  = fig.add_subplot(gs[2:4, 0:2])
        ax_rmse_wc  = fig.add_subplot(gs[2:4, 2:4])

        # Shared color scales: one for distance, one for RMSE
        dist_all = np.concatenate([metrics["dist_mpr"].ravel(),
                                     metrics["dist_wc"].ravel()])
        dist_all = dist_all[~np.isnan(dist_all)]
        dist_vmin = 0.0
        dist_vmax = float(max(dist_all.max(), 1e-6)) if dist_all.size else 1.0

        rmse_all = np.concatenate([metrics["rmse_mpr"].ravel(),
                                     metrics["rmse_wc"].ravel()])
        rmse_all = rmse_all[~np.isnan(rmse_all)]
        rmse_vmin = 0.0
        rmse_vmax = float(rmse_all.max()) if rmse_all.size else 1.0

        im_dist = _draw_heatmap(
            ax_dist_oa, metrics["dist_mpr"], tau_values, M_values,
            r"$1-\mathrm{corr}(A_\mathrm{QIF}, A_\mathrm{OA})$",
            cmap="magma_r", vmin=dist_vmin, vmax=dist_vmax,
            example_marks=example_marks,
        )
        _draw_heatmap(
            ax_dist_wc, metrics["dist_wc"], tau_values, M_values,
            r"$1-\mathrm{corr}(A_\mathrm{QIF}, A_\mathrm{WC})$",
            cmap="magma_r", vmin=dist_vmin, vmax=dist_vmax,
            example_marks=example_marks,
        )

        im_rmse = _draw_heatmap(
            ax_rmse_oa, metrics["rmse_mpr"], tau_values, M_values,
            r"RMSE$(s_\mathrm{QIF}, s_\mathrm{OA})$",
            cmap="magma_r", vmin=rmse_vmin, vmax=rmse_vmax,
            example_marks=example_marks,
        )
        _draw_heatmap(
            ax_rmse_wc, metrics["rmse_wc"], tau_values, M_values,
            r"RMSE$(s_\mathrm{QIF}, s_\mathrm{WC})$",
            cmap="magma_r", vmin=rmse_vmin, vmax=rmse_vmax,
            example_marks=example_marks,
        )

        # Shared colorbars: one per metric pair. Place each next to the WC
        # (right-side) heatmap of its row. Using fig.colorbar(ax=...) lets
        # constrained layout reserve space automatically.
        fig.colorbar(im_dist, ax=ax_dist_wc, fraction=0.046, pad=0.04)
        fig.colorbar(im_rmse, ax=ax_rmse_wc, fraction=0.046, pad=0.04)

        # ── Right six columns: 4 example panels ─────────────────────────────
        tab10 = plt.get_cmap("tab10")
        c_qif, c_oa, c_wc = tab10(0), tab10(1), tab10(2)

        slots = [
            (gs[0, 4:7], gs[1, 4:7]),
            (gs[0, 7:10], gs[1, 7:10]),
            (gs[2, 4:7], gs[3, 4:7]),
            (gs[2, 7:10], gs[3, 7:10]),
        ]

        for letter, ex, (gs_tr, gs_mat) in zip(letters, examples, slots):
            cell = cells[tuple(ex)]
            _draw_example_panel(fig, gs_tr, gs_mat, cell, c_qif, c_oa, c_wc,
                                  letter=letter)

        fig.suptitle("QIF / OA / WC sweep summary")

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi)
            print(f"Figure saved -> {savepath}")
        if show:
            plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sweep_path = "/home/rgast/data/qif_plasticity/qif_mf_sweep.pkl"

    # User-configurable example parameter sets.
    # Pick 4 representative (tau_s, M) combinations. The defaults below sweep
    # the corners: (small tau_s, small M), (small tau_s, large M),
    # (large tau_s, small M), (large tau_s, large M).
    examples = [
        (0.1, 6),
        (0.1, 50),
        (0.8, 6),
        (0.8, 50),
    ]

    # Heatmap-only figure: goes into the top-right "Sequence Encoding" slot
    plot_heatmaps_only(
        sweep_path=sweep_path,
        examples=examples,
        savepath="/home/rgast/data/qif_plasticity/qif_mf_heatmaps.png",
        show=True,
        dpi=200,
        fontsize=16,
        figsize=(13, 10),
    )

    # Examples-only figure: fills the slot the combined-summary figure was in
    plot_examples_only(
        sweep_path=sweep_path,
        examples=examples,
        savepath="/home/rgast/data/qif_plasticity/qif_mf_examples.png",
        show=True,
        dpi=200,
        fontsize=16,
        figsize=(22, 12),
    )