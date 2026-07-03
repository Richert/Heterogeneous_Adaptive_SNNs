"""Quick scan for a genuine Hopf bifurcation in the Allen-QIF mean field.

The 1-D continuation in I (allen_qif_bifurcation.py) finds NO Hopf for the PV fits at the
default J0 / tau_s: the synaptic mode sits at Re(lambda) ~ -1/tau_s and never crosses zero.
This script maps, over a (J, tau_s) grid, the largest real part of the Jacobian eigenvalues
of the equilibrium MAXIMISED over a sweep of the external input I. Where that quantity > 0
there is an unstable focus for some I, i.e. a Hopf bifurcation lies on the I-branch -> the
oscillatory (ING-type) regime. The zero contour is the Hopf locus.

Pure numpy/scipy/matplotlib -> runs in any env (e.g. `allen`).

    python allen_qif_hopf_scan.py "PV+ Interneuron" "L2/3"
"""
import os
import sys
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIT_DIR = os.path.join(os.path.dirname(HERE), "data_fitting")

CONFIG = dict(cell_class="PV+ Interneuron", layer="L2/3", v_r=-70.0,
              J_grid=(-400.0, -5.0, 60), tau_grid=(1.0, 40.0, 60),
              I_sweep=(-50.0, 2500.0, 120))
if len(sys.argv) > 2:
    CONFIG["cell_class"], CONFIG["layer"] = sys.argv[1], sys.argv[2]


def _tag(cell_class, layer):
    c = cell_class.split("+")[0].split()[0].lower()
    return f"{c}_{layer.replace('/', '').replace(' ', '')}"


def load_fit(path):
    d = np.load(path, allow_pickle=False)
    w = np.asarray(d["weights"], float)
    return w / w.sum(), np.asarray(d["omega"], float), np.asarray(d["delta"], float), int(d["M"])


def make_rhs(M, vthbar, delta, weights, v_r, tau_s, J):
    PI = np.pi

    def rhs(y, I):
        r, v, a, s = y[:M], y[M:2 * M], y[2 * M], y[2 * M + 1]
        dr = delta / PI * (v - v_r) + r * (2 * v - v_r - vthbar)
        dv = (v - v_r) * (v - vthbar) - (PI * r) ** 2 - PI * delta * r + I + J * s
        return np.concatenate([dr, dv, [(weights @ r - a) / tau_s, (a - s) / tau_s]])
    return rhs


def jac(rhs, y, I):
    f0 = rhs(y, I); n = y.size; Jm = np.empty((n, n))
    for j in range(n):
        h = 1e-6 * (1.0 + abs(y[j])); yp = y.copy(); yp[j] += h
        Jm[:, j] = (rhs(yp, I) - f0) / h
    return Jm


def max_reig_over_I(M, vthbar, delta, weights, v_r, tau_s, J, I_vals):
    """Largest Re(eig) of the equilibrium, maximised over I (continuation guess in I)."""
    rhs = make_rhs(M, vthbar, delta, weights, v_r, tau_s, J)
    y = np.concatenate([np.full(M, 1e-3), np.full(M, v_r), [1e-3, 1e-3]])  # low-rate guess
    best = -np.inf
    for I in I_vals:
        sol, info, ier, _ = fsolve(rhs, y, args=(I,), full_output=True, xtol=1e-11)
        if ier != 1 or np.max(np.abs(rhs(sol, I))) > 1e-6:
            continue
        y = sol  # continuation
        best = max(best, float(np.max(np.linalg.eigvals(jac(rhs, sol, I)).real)))
    return best


def main():
    cfg = CONFIG
    tag = _tag(cfg["cell_class"], cfg["layer"])
    w, Omega, Delta, M = load_fit(os.path.join(FIT_DIR, f"allen_lorentzian_{tag}.npz"))
    vthbar = cfg["v_r"] + Omega
    print(f"{cfg['cell_class']} {cfg['layer']}  M={M}  Om={np.round(Omega,2)}  De={np.round(Delta,2)}  w={np.round(w,3)}")

    Js = np.linspace(*[cfg["J_grid"][i] for i in (0, 1)], int(cfg["J_grid"][2]))
    Ts = np.linspace(*[cfg["tau_grid"][i] for i in (0, 1)], int(cfg["tau_grid"][2]))
    Iv = np.linspace(*[cfg["I_sweep"][i] for i in (0, 1)], int(cfg["I_sweep"][2]))

    Z = np.full((Ts.size, Js.size), np.nan)
    for i, ts in enumerate(Ts):
        for j, J in enumerate(Js):
            Z[i, j] = max_reig_over_I(M, vthbar, Delta, w, cfg["v_r"], ts, J, Iv)
        print(f"  tau_s={ts:5.1f}  max Re over J: {np.nanmax(Z[i]):+.4f}")

    unstable = Z > 0
    if unstable.any():
        ii, jj = np.where(unstable)
        print(f"\nHOPF REGION FOUND: {unstable.sum()} grid cells with max Re(eig)>0")
        print(f"  tau_s in [{Ts[ii].min():.1f}, {Ts[ii].max():.1f}] ms,  J in [{Js[jj].min():.0f}, {Js[jj].max():.0f}]")
        k = np.nanargmax(Z); ki, kj = np.unravel_index(k, Z.shape)
        print(f"  strongest instability: tau_s={Ts[ki]:.1f}, J={Js[kj]:.0f}  ->  max Re={Z[ki,kj]:+.4f}")
    else:
        print(f"\nNO HOPF anywhere in the scanned box: global max Re(eig) = {np.nanmax(Z):+.4f}")
        print(f"  (J in [{Js.min():.0f},{Js.max():.0f}], tau_s in [{Ts.min():.1f},{Ts.max():.1f}], I in [{Iv.min():.0f},{Iv.max():.0f}])")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    vmax = np.nanmax(np.abs(Z))
    pc = ax.pcolormesh(Js, Ts, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    if (Z > 0).any() and (Z < 0).any():
        ax.contour(Js, Ts, Z, levels=[0.0], colors="k", linewidths=2)
    ax.axhline(8.0, color="0.3", ls=":", lw=1)   # default tau_s
    ax.axvline(-100.0, color="0.3", ls=":", lw=1)  # default J0
    ax.set_xlabel(r"coupling $J$"); ax.set_ylabel(r"synaptic time constant $\tau_s$ (ms)")
    ax.set_title(f"{cfg['cell_class']} {cfg['layer']}: $\\max_I\\,\\mathrm{{Re}}(\\lambda)$ of equilibrium")
    fig.colorbar(pc, ax=ax, label=r"$\max_I\,\mathrm{Re}(\lambda)$  (>0 $\Rightarrow$ Hopf)")
    fig.tight_layout()
    out = os.path.join(HERE, f"allen_qif_hopf_scan_{tag}")
    fig.savefig(out + ".png", dpi=160)
    np.savez(out + ".npz", J=Js, tau_s=Ts, max_reig=Z, I_sweep=Iv)
    print(f"[saved] {os.path.basename(out)}.png / .npz")


if __name__ == "__main__":
    main()
