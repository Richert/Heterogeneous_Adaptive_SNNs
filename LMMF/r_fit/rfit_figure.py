r"""Diagnostic figure + table for rfit_lmmf.py (scenarios S1–S3 vs. baselines). Not a manuscript figure.
    $HOME/conda/envs/sbi/bin/python rfit_figure.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import rfit_lmmf as RF

C = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "rho_fit": "#eda100", "true_rho": "0.45"}
NAMES = {"S1": "S1: ICs, no input", "S2": "S2: forcing periods", "S3": "S3: periods × ICs",
         "rho_fit": "ρ-fit LMMF (true K)", "true_rho": "true ρ, continuum"}
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.2})


def spectral_rmse(a, b):                       # identical to the Fig. 1 metric
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((np.abs(np.fft.rfft(a[:n])) / n - np.abs(np.fft.rfft(b[:n])) / n) ** 2)))


def mixture_pdf(x, w, Om, De):
    return (w * (De / np.pi) / ((x[:, None] - Om) ** 2 + De ** 2)).sum(1)


def main():
    data = RF.load_data()
    res = {s: np.load(os.path.join(_HERE, f"rfit_{s}.npy"), allow_pickle=True).item()
           for s in ("S1", "S2", "S3") if os.path.exists(os.path.join(_HERE, f"rfit_{s}.npy"))}
    base = np.load(os.path.join(_HERE, "rfit_baselines.npy"), allow_pickle=True).item()
    Robs = np.stack([data[RF._key(c)] for c in RF.TEST])

    # table
    print(f"{'model':28s} {'M*':>3s} {'K':>6s} {'test RMSE':>10s} {'test spec. RMSE':>16s}  per-trace spec")
    rows = [(s, r["best"]["M"], r["best"]["K"], r["R_test"]) for s, r in res.items()]
    rows += [("rho_fit", base["rho_fit"]["M"], RF.K_TRUE, base["rho_fit"]["R_test"]),
             ("true_rho", "-", RF.K_TRUE, base["true_rho"]["R_test"])]
    for name, M, K, R in rows:
        sp = [spectral_rmse(Robs[i], R[i]) for i in range(len(RF.TEST))]
        print(f"{NAMES[name]:28s} {M!s:>3s} {K:6.3f} {np.sqrt(np.mean((R - Robs) ** 2)):10.4f} "
              f"{np.mean(sp):16.2e}  {[f'{v:.1e}' for v in sp]}")

    fig, ax = plt.subplots(2, 4, figsize=(10, 4.6), layout="constrained")
    # (a) train/val RMSE vs M
    a = ax[0, 0]
    for s, r in res.items():
        Ms = [e["M"] for e in r["trace"]]
        a.plot(Ms, [e["train_rmse"] for e in r["trace"]], "o-", color=C[s], ms=3, label=f"{s} train")
        a.plot(Ms, [e["val_rmse"] for e in r["trace"]], "s--", color=C[s], ms=3, label=f"{s} val")
        a.plot(r["best"]["M"], r["best"]["val_rmse"], "*", color=C[s], ms=9)
    a.axhline(base["true_rho"]["test_rmse"], color=C["true_rho"], lw=0.8, ls=":")
    a.set_xlabel("M"); a.set_ylabel("RMSE of R(t)"); a.set_title("(a) greedy search", loc="left")
    a.legend(fontsize=5, ncol=2, frameon=False)
    # (b) K estimates
    b = ax[0, 1]
    for s, r in res.items():
        b.plot([e["M"] for e in r["trace"]], [e["K"] for e in r["trace"]], "o-", color=C[s], ms=3, label=s)
    b.axhline(RF.K_TRUE, color="0.2", lw=0.8, ls="--", label="true K")
    b.set_xlabel("M"); b.set_ylabel("fitted K"); b.set_title("(b) coupling strength", loc="left")
    b.legend(fontsize=6, frameon=False)
    # (c) recovered densities
    c = ax[0, 2]
    x = np.linspace(-5, 5, 1001)
    c.hist(data["omega"], bins=80, range=(-5, 5), density=True, color="0.85", label="network ω")
    true = sum(wk * norm.pdf(x, mk, sk) for mk, sk, wk in zip(RF.GMM["means"], RF.GMM["stds"], RF.GMM["weights"]))
    c.plot(x, true, color="0.2", lw=1.0, label="true ρ")
    for s, r in res.items():
        e = r["best"]
        c.plot(x, mixture_pdf(x, e["w"], e["Om"], e["De"]), color=C[s], label=f"{s} (M={e['M']})")
    f = base["rho_fit"]
    c.plot(x, mixture_pdf(x, f["w"], f["Om"], f["De"]), color=C["rho_fit"], ls="--", label="ρ-fit")
    c.set_xlabel("ω"); c.set_yticks([]); c.set_title("(c) recovered ρ_M", loc="left")
    c.legend(fontsize=5, frameon=False)
    ax[0, 3].axis("off")
    # (d-f) test traces
    for i, (cond, a) in enumerate(zip(RF.TEST, ax[1, :3])):
        a.plot(RF.TS, Robs[i], color="0.1", lw=1.6, label="network (test)")
        for s, r in res.items():
            a.plot(RF.TS, r["R_test"][i], color=C[s], lw=0.9, label=s)
        a.plot(RF.TS, base["rho_fit"]["R_test"][i], color=C["rho_fit"], lw=0.9, ls="--", label="ρ-fit")
        ttl = "no input" if cond[0] is None else f"Ω_f={cond[0]:g}"
        a.set_title(f"({'def'[i]}) test: {ttl}, R0={cond[1]:g}", loc="left")
        a.set_xlabel("t"); a.set_ylabel("R"); a.set_ylim(0, 1)
    ax[1, 0].legend(fontsize=5, frameon=False, ncol=2)
    ax[1, 3].axis("off")
    fig.savefig(os.path.join(_HERE, "rfit_figure.png"), dpi=200)
    print("saved rfit_figure.png")


if __name__ == "__main__":
    main()
