"""
Plot the result of the QIF / OA / WC sweep produced by
qif_stdp_mean_field_evaluation.py.

The figure has 4 rows x 10 columns:

  ┌─────┬─────┬──────────────┬──────────────┐
  │ C1  │ C2  │  Example 1   │  Example 2   │
  │     │     │  cols 5-7    │  cols 8-10   │
  │QIF  │QIF  │  row 1: <s>  │  row 1: <s>  │
  │vs   │vs   │  row 2: A's  │  row 2: A's  │
  │OA   │WC   ├──────────────┼──────────────┤
  │DIST │DIST │  Example 3   │  Example 4   │
  ├─────┼─────┤  cols 5-7    │  cols 8-10   │
  │QIF  │QIF  │  row 3: <s>  │  row 3: <s>  │
  │vs   │vs   │  row 4: A's  │  row 4: A's  │
  │OA   │WC   │              │              │
  │RMSE │RMSE │              │              │
  └─────┴─────┴──────────────┴──────────────┘

Each heatmap occupies a 2-col x 2-row block (~square).
The user supplies the .pkl path and a list of 4 (tau_s, M) tuples for the
example panels.

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
                    aspect="equal", interpolation="nearest", origin="upper")
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
    sub = gs_matrices.subgridspec(nrows=1, ncols=3, wspace=0.02)
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

    titles = ["QIF", "OA", "WC"]
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
    # if last_im is not None:
    #     fig.colorbar(last_im, ax=last_ax, fraction=0.046, pad=0.04)


# ─────────────────────────────────────────────────────────────────────────────
# Main: load + render
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
        fig = plt.figure(figsize=(22, 12), constrained_layout=True)
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

        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.05, hspace=0.05, wspace=0.02)

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

    plot_sweep_summary(
        sweep_path=sweep_path,
        examples=examples,
        savepath="/home/rgast/data/qif_plasticity/qif_mf_sweep_summary.png",
        show=True,                 # set True for interactive viewing
        dpi=200,
        fontsize=16,                # base font size; tweak to taste
    )