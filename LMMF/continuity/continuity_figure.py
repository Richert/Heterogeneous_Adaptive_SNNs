r"""Diagnostic figure for continuity_check.py (experiments A-D). Not a manuscript figure.
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python continuity_figure.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#888888"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "lines.linewidth": 1.5})

A = np.load(os.path.join(H, "expA.npz"), allow_pickle=True)
B = np.load(os.path.join(H, "expB.npz"))
D = np.load(os.path.join(H, "expD.npz"))
SAMPLE_BL = {500: 0.081, 5000: 0.018, 50000: 0.0048}     # d_BL(rho, rho_N), 5 seeds (see notes)

fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2), layout="constrained")

# (a) dynamical error vs distribution distance
a = ax[0, 0]
bl = A["bl"][1:]
for i, (T, c) in enumerate(zip((1, 2, 5), (C1, C2, C3))):
    a.loglog(bl, A["Esup"][1:, i], "o-", color=c, ms=4, label=fr"$\sup_{{t\leq{T}}}|Z_\rho-Z_M|$")
a.loglog(bl, A["srmse"][1:] * 20, "s--", color=C4, ms=4, label=r"spectral RMSE of $R$ ($\times 20$)")
x = np.array([bl.min(), bl.max()])
a.loglog(x, 0.6 * x, color=MUTED, lw=0.8, ls=":", label=r"slope 1")
for N, v in SAMPLE_BL.items():
    a.axvline(v, color=MUTED, lw=0.6, ls="--")
    a.text(v, 0.97, f"N={N} ", rotation=90, va="top", ha="right", color=MUTED, fontsize=6,
           transform=a.get_xaxis_transform())
for Mi, (xx, yy) in zip(A["M"][1:], zip(bl, A["Esup"][1:, 2])):
    a.annotate(f"{Mi}", (xx, yy), textcoords="offset points", xytext=(3, 3), fontsize=6, color=INK)
a.set_xlabel(r"$d_{BL}(\rho,\rho_M)$"); a.set_ylabel("error")
a.set_title(r"(a) finite-time error vs. $d_{BL}$ (labels: effective $M$)", fontsize=8, loc="left")
a.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0.08, 0.45))

# (b) linear response spectrum near threshold
b = ax[0, 1]
K = B["Ks"][-1]; nu = B["nu"]
b.plot(nu, np.abs(B[f"Lth_ref_K{K}"]), color=INK, lw=2, label=r"$\rho$ (Gaussian mixture)")
for M, c in zip(B["Ms"], (C1, C2, C3)):
    b.plot(nu, np.abs(B[f"Lth_M{M}_K{K}"]), color=c, label=f"LMMF M={M}")
b.plot(nu[::8], np.abs(B[f"Lnum_ref_K{K}"])[::8], "o", ms=3, mfc="none", color=INK, label="simulation")
b.set_xlabel(r"$\nu$"); b.set_ylabel(r"$|\hat Z(\epsilon+i\nu)|$")
b.set_title(fr"(b) linear response, K={K} (near threshold), $\epsilon$={B['eps']}", fontsize=8, loc="left")
b.legend(frameon=False, fontsize=6)

# (c) sampling distribution of the selected M
c = ax[1, 0]
Ns = np.unique(D["N"]); lams = np.unique(D["lam"])
for j, (lam, col) in enumerate(zip(lams, (C1, C2))):
    for i, N in enumerate(Ns):
        Ms = D["M"][(D["N"] == N) & (D["lam"] == lam)]
        vals, cnt = np.unique(Ms, return_counts=True)
        c.scatter(np.full(vals.size, i + (j - 0.5) * 0.25), vals, s=12 * cnt, color=col,
                  edgecolor="white", lw=0.5, label=fr"$\lambda$={lam:g}" if i == 0 else None)
c.set_xticks(range(len(Ns))); c.set_xticklabels([str(n) for n in Ns])
c.set_xlabel("N (20 samples each)"); c.set_ylabel(r"selected $M$")
c.set_title("(c) sampling variability of selected M (marker area ~ count)", fontsize=8, loc="left")
c.legend(frameon=False, fontsize=6)

# (d) sample-free approximation curve a(M)
d = ax[1, 1]
aM = D["aM"]
d.semilogy(aM[:, 1], aM[:, 4], "o-", color=C1, ms=4, label=r"$d_{BL}$")
d.semilogy(aM[:, 1], aM[:, 3], "s-", color=C2, ms=4, label=r"$d_{KS}$")
d.semilogy(aM[:, 1], np.sqrt(aM[:, 2]), "^-", color=C3, ms=4, label=r"$\sqrt{\mathrm{CvM}}$")
for N, v in SAMPLE_BL.items():
    d.axhline(v, color=MUTED, lw=0.6, ls="--")
    d.text(aM[-1, 1], v, f"N={N}", va="bottom", ha="right", color=MUTED, fontsize=6)
d.set_xlabel(r"effective $M$"); d.set_ylabel(r"distance to $\rho$")
d.set_title(r"(d) approximation error $a(M)$ (dashed: $d_{BL}(\rho,\rho_N)$)", fontsize=8, loc="left")
d.legend(frameon=False, fontsize=6)

fig.savefig(os.path.join(H, "continuity_figure.png"), dpi=200)
print("saved")
