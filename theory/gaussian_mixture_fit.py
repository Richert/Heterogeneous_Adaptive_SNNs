# -*- coding: utf-8 -*-
"""Demo: fit empirical samples from a Gaussian mixture with Lorentzian mixtures,
for two choices of the width bounds (Delta_min, Delta_max) and two choices of the
ensemble-count penalty lambda.  Same plotting style as lorentzian_mixture_demo.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lorentzian_mixture as LM

# ---- empirical samples from a 3-component Gaussian mixture -----------------
rng = np.random.default_rng(3)
means, stds, wts = [-4.0, 0.0, 5.0], [0.8, 1.2, 0.6], [0.30, 0.45, 0.25]
N = 5000
comp = rng.choice(3, size=N, p=wts)
samples = rng.normal(np.array(means)[comp], np.array(stds)[comp])

# ---- the four (Delta-bounds, lambda) settings ------------------------------
delta_choices = [(0.01, 0.1), (0.01, 1.0)]        # rows: narrow vs forced-broad
lambda_choices = [1e-8, 1e-5]                    # cols: permissive vs strict

# ---- figure (same style as the previous demo) ------------------------------
plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "mathtext.fontset": "cm"})
fig, ax = plt.subplots(2, 2, figsize=(7, 5), layout="constrained")
gx = np.linspace(np.percentile(samples, 0.5), np.percentile(samples, 99.5), 700)

def lam_str(l):
    e = int(round(np.log10(l))); m = l / 10.0 ** e
    return (r"10^{%d}" % e) if abs(m - 1) < 1e-9 else (r"%g\times10^{%d}" % (m, e))

for i, dbounds in enumerate(delta_choices):
    for j, lam in enumerate(lambda_choices):
        a = ax[i, j]
        res = LM.fit(samples, dbounds, M_max=15, lambda_M=lam, loss="cvm",
                     n_restarts=4)
        m = res["model"]
        a.hist(samples, bins=80, range=(gx[0], gx[-1]), density=True,
               color="0.8", label="samples")
        for k in range(m.M):
            a.plot(gx, m.w[k] * (m.Delta[k] / np.pi)
                   / ((gx - m.Omega[k]) ** 2 + m.Delta[k] ** 2),
                   lw=0.8, color="#2e6f95", alpha=0.7)
        a.plot(gx, m.pdf(gx), lw=1.8, color="#c1121f", label="mixture")
        a.set_title(r"$\Delta_m\in[%g,%g]$, $\lambda=%s$  $\to$ $M^*=%d$"
                    % (dbounds[0], dbounds[1], lam_str(lam), res["M"]),
                    fontsize=8.5)
        a.set_yticks([])
        a.legend(fontsize=7, frameon=False)

fig.suptitle(r"Gaussian-mixture samples fit by Lorentzian mixtures (Eq. 5)",
             fontsize=10)
fig.savefig("/home/rgast/data/mpmf_simulations/gaussian_mixture_demo.png", dpi=200)
print("saved figure")
