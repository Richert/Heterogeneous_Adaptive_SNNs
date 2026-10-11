r"""
Figures + summary for unimodal_kalpha.py (K-ramps of network and LMMF in the (K, α) plane).

Per distribution × loss variant (a/b/c), one figure:
  (a) network classification map: monostable async / monostable sync / multistable (forward and
      backward ramp disagree) / oscillatory; overlaid: exact incoherence boundary (true ρ, dashed),
      LMMF incoherence boundary (solid), LMMF fold curves (if computed), and the LMMF's own
      classification boundaries (thin contours) for direct comparison;
  (b) spectral RMSE (LMMF vs network, per holding window, mean over both ramp directions), same lines;
  (c,d) R along the K-ramps (forward / backward; network vs LMMF) for α = 0 and for the α row with
      the largest multistable range of the network.
Plus a summary table (classification agreement, mean RMSE) over all distributions and variants.
Reads only saved results; runtime ~1–2 min.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python unimodal_kalpha_figure.py [--hold H]
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import functools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from prl_style import set_prl_style as _set_prl_style, panel_label
import unimodal_kalpha as UK

set_prl_style = functools.partial(_set_prl_style, "prl")
_lab = functools.partial(panel_label, dx=-16, dy=4)
CLS_NAMES = ["monostable async", "monostable sync", "multistable", "oscillatory"]
CLS_COLORS = ["0.82", "#9ec5f4", "#eda100", "#e87ba4"]
C_NET, C_LMMF, C_EXACT = "0.15", "#c1121f", "0.1"
VNAME = {"a": "CvM", "b": r"CvM + density ($\beta=1$)", "c": r"CvM + density ($\beta=10$)"}


def spectral_rmse(a, b):
    n = a.shape[-1]
    return np.sqrt(np.mean((np.abs(np.fft.rfft(a, axis=-1)) - np.abs(np.fft.rfft(b, axis=-1))) ** 2, -1)) / n


def load_ramps(name, hold=UK.HOLD):
    """Assemble all α chunks -> per direction: R_net (nα, nK, ns), R_lmmf (nVar, nα, nK, ns)."""
    tag = "" if hold == UK.HOLD else f"_hold{hold:g}"
    out = {}
    for d in UK.DIRECTIONS:
        files = sorted(f for f in os.listdir(UK.OUT)
                       if f.startswith(f"{name}_ramp_{d}_c") and f.endswith(f"{tag}.npz")
                       and (tag or "_hold" not in f) and "_Mscan" not in f)
        parts = [np.load(os.path.join(UK.OUT, f)) for f in files]
        if not parts:
            return None
        ia = np.concatenate([p["ia"] for p in parts]); o = np.argsort(ia)
        out[d] = dict(net=np.concatenate([p["R_net"] for p in parts])[o].astype(float),
                      lmmf=np.concatenate([p["R_lmmf"] for p in parts], axis=1)[:, o].astype(float),
                      alpha=np.concatenate([p["alpha"] for p in parts])[o])
    return out


def combine(cf, cb):
    """Per-direction classes (0 async, 1 sync, 2 osc) -> 0 mono async, 1 mono sync, 2 multistable, 3 osc."""
    c = np.full(cf.shape, 2)
    c[(cf == 0) & (cb == 0)] = 0
    c[(cf == 1) & (cb == 1)] = 1
    c[(cf == 2) & (cb == 2)] = 3
    return c


def analyse(name, hold=UK.HOLD):
    r = load_ramps(name, hold)
    if r is None:
        return None
    f, b = r["fwd"], r["bwd"]
    cls_net = combine(UK.classify(f["net"]), UK.classify(b["net"]))
    nv = f["lmmf"].shape[0]
    cls_l = [combine(UK.classify(f["lmmf"][v]), UK.classify(b["lmmf"][v])) for v in range(nv)]
    rmse = [0.5 * (spectral_rmse(f["net"], f["lmmf"][v]) + spectral_rmse(b["net"], b["lmmf"][v])) for v in range(nv)]
    return dict(alpha=f["alpha"], cls_net=cls_net, cls_lmmf=cls_l, rmse=rmse, ramps=r)


def _lines(ax, name, v):
    ex = os.path.join(UK.OUT, f"{name}_boundary_exact.npy")
    if os.path.exists(ex):
        e = np.load(ex); ok = np.isfinite(e[:, 0]) & (e[:, 0] <= UK.K_GRID[-1] * 1.05)
        ax.plot(e[ok, 0], np.abs(e[ok, 1]), "--", color=C_EXACT, lw=1.0)
    bd = np.load(os.path.join(UK.OUT, f"{name}_boundary_{v}.npy"), allow_pickle=True).item()
    if "G" in bd:                                     # full boundary incl. restabilisation
        ax.contour(bd["K_grid"], bd["alpha"], bd["G"], levels=[0.0], colors=[C_LMMF], linewidths=1.2)
    else:
        ax.plot(bd["K"], bd["alpha"], "-", color=C_LMMF, lw=1.2)
    fo = os.path.join(UK.OUT, f"{name}_folds_{v}.npy")
    if os.path.exists(fo):
        for c in UK._dedupe(list(np.load(fo, allow_pickle=True))):
            ax.plot(c[:, 0], c[:, 1], "-", color="#7b2cbf", lw=1.0)
    ax.set_xlim(UK.K_GRID[0], UK.K_GRID[-1]); ax.set_ylim(UK.A_GRID[0], UK.A_GRID[-1])


def _ramp_panel(ax, res, v, ia, title):
    K = UK.K_GRID; ns = res["ramps"]["fwd"]["net"].shape[-1]
    x = K[:, None] + np.linspace(0, (K[1] - K[0]) * 0.95, ns)[None, :]          # spread each window over its K step
    for d, ls in (("fwd", "-"), ("bwd", "--")):
        rr = res["ramps"][d]
        ax.plot(x.ravel(), rr["net"][ia].ravel(), ls, color=C_NET, lw=0.6)
        ax.plot(x.ravel(), rr["lmmf"][v][ia].ravel(), ls, color=C_LMMF, lw=0.6)
    ax.set_xlim(K[0], K[-1] + K[1] - K[0]); ax.set_ylim(0, 1)
    ax.set_xlabel(r"coupling $K$ (ramp)"); ax.set_ylabel(r"$R$")
    ax.set_title(title, fontsize=6.5, pad=2)


def make_figure(name, res, vi, v, hold=UK.HOLD):
    set_prl_style()
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained")
    ext = [UK.K_GRID[0], UK.K_GRID[-1], UK.A_GRID[0], UK.A_GRID[-1]]
    cmap = ListedColormap(CLS_COLORS); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    a = ax[0, 0]
    a.imshow(res["cls_net"], origin="lower", aspect="auto", extent=ext, cmap=cmap, norm=norm,
             interpolation="nearest")
    a.contour(UK.K_GRID, res["alpha"], res["cls_lmmf"][vi], levels=[0.5, 1.5, 2.5], colors="k",
              linewidths=0.5, linestyles=":")
    _lines(a, name, v)
    a.set_xlabel(r"coupling $K$"); a.set_ylabel(r"phase lag $\alpha$")
    a.set_title(f"network classification — {name}", fontsize=7, pad=2)
    h = [Patch(fc=c, label=l) for c, l in zip(CLS_COLORS, CLS_NAMES)]
    h += [Line2D([], [], color=C_EXACT, ls="--", label=r"exact boundary (true $\rho$)"),
          Line2D([], [], color=C_LMMF, label="LMMF boundary"),
          Line2D([], [], color="#7b2cbf", label="LMMF folds"),
          Line2D([], [], color="k", ls=":", lw=0.6, label="LMMF classes")]
    a.legend(handles=h, fontsize=5, loc="upper left", framealpha=0.85)
    _lab(a, "a")
    b = ax[0, 1]
    im = b.imshow(np.log10(res["rmse"][vi] + 1e-6), origin="lower", aspect="auto", extent=ext,
                  cmap="Reds", interpolation="nearest")
    _lines(b, name, v)
    fig.colorbar(im, ax=b, label=r"$\log_{10}$ spectral RMSE")
    b.set_xlabel(r"coupling $K$"); b.set_title(f"LMMF vs network — {VNAME[v]}", fontsize=7, pad=2)
    _lab(b, "b")
    # example ramps: α = 0 and the row with the largest network multistable range
    multi = (res["cls_net"] == 2).sum(1)
    ia2 = int(np.argmax(multi)) if multi.max() > 0 else int(np.argmin(np.abs(res["alpha"] - 1.0)))
    _ramp_panel(ax[1, 0], res, vi, 0, r"$\alpha = 0$ (network black, LMMF red; fwd solid, bwd dashed)")
    _ramp_panel(ax[1, 1], res, vi, ia2, rf"$\alpha = {res['alpha'][ia2]:.2f}$")
    _lab(ax[1, 0], "c"); _lab(ax[1, 1], "d")
    tag = "" if hold == UK.HOLD else f"_hold{hold:g}"
    out = os.path.join(UK.OUT, f"fig_{name}_{v}{tag}")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg")
    plt.close(fig)
    return out


def main(hold=UK.HOLD):
    print(f"{'dist':8s} {'var':3s} {'agree':>6s} {'multi net/LMMF':>15s} {'RMSE mean':>10s} {'RMSE median':>11s}")
    for name in UK.dists():
        res = analyse(name, hold)
        if res is None:
            print(f"{name:8s} -- no ramp data"); continue
        for vi, v in enumerate(UK.VARIANTS):
            agree = float(np.mean(res["cls_lmmf"][vi] == res["cls_net"]))
            print(f"{name:8s} {v:3s} {100 * agree:5.1f}% {int((res['cls_net'] == 2).sum()):7d}/"
                  f"{int((res['cls_lmmf'][vi] == 2).sum()):<7d} {np.mean(res['rmse'][vi]):10.2e} "
                  f"{np.median(res['rmse'][vi]):11.2e}")
            make_figure(name, res, vi, v, hold)
    print(f"[saved] figures fig_<dist>_<variant>.png/.svg in {UK.OUT}")


if __name__ == "__main__":
    main(float(sys.argv[sys.argv.index("--hold") + 1]) if "--hold" in sys.argv else UK.HOLD)
