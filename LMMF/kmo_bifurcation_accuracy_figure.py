r"""
Figure + table for kmo_bifurcation_accuracy.py (LMMF accuracy near bifurcations, Fig. 1 example).
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python kmo_bifurcation_accuracy_figure.py

Panels: (a) bifurcation diagram in K (exact ρ branch, network attractor range, LMMF locked branches),
(b) relative error of K_H and K_SN vs M (sample / population fits), (c) K_H error vs the peak-density
error ρ_M(ν*) − ρ(ν*) and vs d_BL(ρ, ρ_M), (d) spectral RMSE (vs network) as a function of K.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared"), os.path.join(_HERE, "continuity")]
import functools
import numpy as np
import matplotlib.pyplot as plt
from prl_style import set_prl_style as _set_prl_style, panel_label
import kmo_bifurcation_accuracy as BA
import continuity_check as CC

set_prl_style = functools.partial(_set_prl_style, "prl")
_lab = functools.partial(panel_label, dx=-16, dy=4)
C_M = {1: "#9e9e9e", 2: "#eda100", 4: "#1baf7a", 6: "#2a78d6", 7: "#7b2cbf"}
C_NET, C_TRUE = "0.15", "#c1121f"
XPK = np.linspace(0.5, 3.5, 30001)                         # search range for the cluster peak


def spectral_rmse(a, b):
    n = min(a.size, b.size)
    return float(np.sqrt(np.mean((np.abs(np.fft.rfft(a[:n])) - np.abs(np.fft.rfft(b[:n]))) ** 2)) / n)


def main():
    fits = [f for f in BA.load_fits() if not f["dup"]]
    tr = np.load(BA.TRUTH_NPZ)
    sims = np.load(BA.SIMS_NPZ)
    K_H, K_SN, nu = float(tr["K_H"]), float(tr["K_SN"]), float(tr["nu_star"])
    rho_star = float(BA.rho_true(XPK).max())               # true peak density
    Ks, t = sims["K"], sims["t"]
    tail = t > 0.5 * t[-1]
    rows = []
    for j, f in enumerate(fits):
        inc = np.load(os.path.join(BA.OUT, f"auto_{f['idx']:02d}_incoh.npy"), allow_pickle=True).item()
        syn = np.load(os.path.join(BA.OUT, f"auto_{f['idx']:02d}_sync.npy"), allow_pickle=True).item()
        m = CC.LM.LorentzianMixture(f["w"], f["Om"], f["De"])
        kh = float(inc["K_hb"][0]) if inc["K_hb"].size else np.nan
        # peak-density overshoot: max of the fitted density near the clusters vs the true maximum.
        # (K_H = 2/(π ρ_M(ν*_M)) exactly at the mixture's own Hilbert zero ν*_M, which sits at or
        # near the highest bump; Lorentzian tails force taller cores to match the CDF.)
        rho_M_star = float(m.pdf(XPK).max())
        srm = np.array([spectral_rmse(sims["R_net"][i], sims["R_fit"][j, i]) for i in range(Ks.size)])
        rows.append(dict(f=f, M=f["M"], src=f["source"], K_H=kh, K_SN=float(syn["K_SN"]),
                         dBL=CC.d_bl(m), drho=rho_M_star - rho_star, srm=srm, syn=syn, j=j))
    srm_cont = np.array([spectral_rmse(sims["R_net"][i], sims["R_cont"][i]) for i in range(Ks.size)])

    print(f"truth: K_H = {K_H:.4f}, K_SN = {K_SN:.4f}, ρ(ν*) = {rho_star:.4f}")
    print(f"{'source':10s} {'M':>2s} {'K_H':>7s} {'err%':>6s} {'K_SN':>7s} {'err%':>6s} {'d_BL':>6s} "
          f"{'Δmaxρ':>7s} {'specRMSE: mean':>15s} {'near K_H':>9s} {'far':>7s}")
    nearH = np.abs(Ks - K_H) <= 0.3
    far = (np.abs(Ks - K_H) > 0.6) & (np.abs(Ks - K_SN) > 0.6)
    for r in sorted(rows, key=lambda r: (r["src"], r["M"])):
        print(f"{r['src']:10s} {r['M']:2d} {r['K_H']:7.4f} {100 * (r['K_H'] / K_H - 1):6.1f} "
              f"{r['K_SN']:7.4f} {100 * (r['K_SN'] / K_SN - 1):6.1f} {r['dBL']:6.3f} {r['drho']:7.4f} "
              f"{r['srm'].mean():15.2e} {r['srm'][nearH].mean():9.2e} {r['srm'][far].mean():7.2e}")
    print(f"{'true ρ (continuum) vs network':28s}  mean {srm_cont.mean():.2e}  near K_H "
          f"{srm_cont[nearH].mean():.2e}  far {srm_cont[far].mean():.2e}")

    set_prl_style()
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained")
    # (a) bifurcation diagram
    a = ax[0, 0]
    bK, br = tr["branch_K"], tr["branch_r"]
    up = br >= tr["r_SN"]
    a.plot(bK[up], br[up], color=C_TRUE, lw=1.6, label=r"true $\rho$, locked (stable)")
    a.plot(bK[~up], br[~up], color=C_TRUE, lw=1.0, ls=":", label=r"true $\rho$, locked (unstable)")
    a.plot([0, K_H], [0, 0], color=C_TRUE, lw=1.6)
    a.plot([K_H, 5.0], [0, 0], color=C_TRUE, lw=1.0, ls=":")
    Rn = sims["R_net"][:, tail]
    a.vlines(Ks, Rn.min(1), Rn.max(1), color=C_NET, lw=2.2, alpha=0.35, label="network (range, $t>T/2$)")
    for r in rows:
        if r["src"] != "sample" or r["M"] == 1:
            continue
        s = r["syn"]; ok = s["stab"] & (s["R"] > 0.5)
        a.plot(s["K"][ok], s["R"][ok], ".", ms=1.5, color=C_M[r["M"]])
        a.axvline(r["K_H"], color=C_M[r["M"]], lw=0.7, ls="--")
    a.axvline(K_H, color=C_TRUE, lw=0.9); a.axvline(K_SN, color=C_TRUE, lw=0.9)
    a.set_xlim(0.5, 5.0); a.set_ylim(-0.02, 1.0)
    a.set_xlabel(r"coupling $K$"); a.set_ylabel(r"$R$")
    a.legend(fontsize=5.5, loc="upper left")
    _lab(a, "a")
    # (b) bifurcation-point errors vs M
    b = ax[0, 1]
    for src, mk in (("sample", "o"), ("population", "s")):
        rr = sorted([r for r in rows if r["src"] == src], key=lambda r: r["M"])
        M = [r["M"] for r in rr]
        b.plot(M, [100 * (r["K_H"] / K_H - 1) for r in rr], mk + "-", color="#2a78d6", ms=3.5,
               mfc="white" if src == "population" else None, label=f"$K_H$, {src}")
        b.plot(M, [100 * (r["K_SN"] / K_SN - 1) for r in rr], mk + "-", color="#eb6834", ms=3.5,
               mfc="white" if src == "population" else None, label=f"$K_{{SN}}$, {src}")
    b.axhline(0, color="0.5", lw=0.6)
    b.set_xlabel(r"number of Lorentzians $M$"); b.set_ylabel("relative error [%]")
    b.legend(fontsize=5.5)
    _lab(b, "b")
    # (c) K_H error vs peak-density error and vs d_BL
    c = ax[1, 0]
    for r in rows:
        c.plot(r["drho"], 100 * (r["K_H"] / K_H - 1), "o" if r["src"] == "sample" else "s",
               color=C_M[r["M"]], mfc="white" if r["src"] == "population" else C_M[r["M"]], ms=4)
    x = np.linspace(min(r["drho"] for r in rows) - 0.02, max(r["drho"] for r in rows) + 0.02, 200)
    c.plot(x, 100 * (rho_star / (rho_star + x) - 1), color="0.4", lw=0.8, ls="--",
           label=r"$K_H \propto 1/\max\rho$")
    c.set_xlabel(r"peak-density overshoot $\max\rho_M-\max\rho$"); c.set_ylabel(r"$K_H$ error [%]")
    ci = c.inset_axes([0.62, 0.58, 0.35, 0.37])
    for r in rows:
        ci.plot(r["dBL"], 100 * (r["K_H"] / K_H - 1), "o" if r["src"] == "sample" else "s",
                color=C_M[r["M"]], mfc="white" if r["src"] == "population" else C_M[r["M"]], ms=2.5)
    ci.set_xscale("log"); ci.set_xlabel(r"$d_{BL}(\rho,\rho_M)$", fontsize=5, labelpad=0)
    ci.tick_params(labelsize=4.5)
    c.legend(fontsize=5.5, loc="lower left")
    _lab(c, "c")
    # (d) spectral RMSE vs K
    d = ax[1, 1]
    for r in rows:
        if r["src"] != "sample":
            continue
        d.semilogy(Ks, r["srm"], color=C_M[r["M"]], lw=1.0, label=f"LMMF $M={r['M']}$")
    d.semilogy(Ks, srm_cont, color=C_TRUE, lw=1.2, ls="--", label=r"true $\rho$ (continuum)")
    d.axvline(K_H, color=C_TRUE, lw=0.8); d.axvline(K_SN, color=C_TRUE, lw=0.8)
    d.set_xlabel(r"coupling $K$"); d.set_ylabel("spectral RMSE vs network")
    d.legend(fontsize=5.5, ncol=2)
    _lab(d, "d")
    out = os.path.join(BA.OUT, "bif_accuracy_figure")
    fig.savefig(out + ".png", dpi=250); fig.savefig(out + ".svg")
    print(f"[saved] {out}.png/.svg")


if __name__ == "__main__":
    main()
