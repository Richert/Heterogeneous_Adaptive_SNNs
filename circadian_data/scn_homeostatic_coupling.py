r"""
Homeostatic model of the TTX-sensitive coupling strength K_s(t)
===============================================================

Idea: the TTX-sensitive coupling (VIP/GABA release) scales with the network's spike rate r; TTX lowers r by raising
the effective spike threshold (Na⁺ channel block); a slow homeostatic controller raises the cells' excitability g
while r is below its set point r*, and lowers it while r is above r*. Equations (time in h):

    τ_u du/dt = u_ext(t) − u,  τ_u = τ_on (u < u_ext) | τ_off (u > u_ext)    TTX level in the tissue (solved
                                                                           exactly: tissue_level)
    r(t)      = σ(g − θ_0 − Δθ u),   σ(x) = 1 / (1 + e^{−x}),   θ_0 = −logit(r*)      normalised spike rate
    τ_g dg/dt = (r* − r) − λ g                                  homeostatic excitability (leaky integral control)
    K_s(t)    = K_pre r / r*                                    coupling ∝ spike rate (release ∝ firing)

  u_ext(t)  extrinsic driver: 0 before TTX, 1 under TTX, u_res after washout (u_res > 0: incomplete washout /
            residual TTX bound in the tissue).
  τ_on      application kinetics (fixed, TAU_ON ≈ 0.5 h): bath-applied TTX blocks within minutes.
  τ_off     WASHOUT kinetics (free, hours): TTX unbinds from the Na⁺ channels and diffuses out of the slice only
            gradually, so u relaxes continuously from 1 to u_res after the medium change — the threshold, the rate
            rebound, K_s and the TTX-dependent noise all follow this gradual decline instead of jumping.
  g         excitability in units of the slope of the rate function (dimensionless; e.g. Na⁺/K⁺ conductance balance,
            intrinsic excitability, synaptic scaling). g = 0 is the untreated steady state.
  θ_0       threshold chosen such that r = r* at g = 0, u = 0: the untreated network sits at its set point.
  Δθ        TTX potency: the threshold shift by full block. Sets how deep the rate (and K) drops,
            K_min / K_pre ≈ σ(logit r* − Δθ) / r*.
  r*        homeostatic set point of the rate (fraction of the maximal rate). Since r ∈ (0, 1), the largest
            possible coupling is K_pre / r*: the rebound ceiling.
  τ_g       homeostatic time constant. With r ≈ 0 under TTX, g climbs at ≈ r*/τ_g and reaches the new threshold
            after t_rec ≈ τ_g Δθ / r* → DELAYED, S-SHAPED recovery (the sigmoid of a linear ramp).
            After washout r ≈ 1 > r*, g falls at ≈ (1 − r*)/τ_g: the rebound decays (1 − r*)/r* times faster than
            the recovery under TTX built up — e.g. r* = 0.2: recovery over days, rebound over hours. One time
            constant produces both, the asymmetry follows from the set point.
  λ         leak of the controller (imperfect integral control; λ = 0: perfect homeostasis). Under sustained block
            the controller settles where λ g = r* − r: partial recovery, K_ttx < K_pre; recovery needs
            λ < r*/Δθ at all (otherwise g saturates at r*/λ below the new threshold). With u_res > 0 the same
            leak leaves the post-washout rate below r*: K_post < K_pre.
  K_pre     coupling of the untreated network (as before).

Free parameters: K_pre, r*, Δθ, τ_g, λ, u_res, τ_off (7; the sigmoid profile had 8: K_pre, f_ttx, f_min, t_rec,
w_rec, K_post, f_max, τ_wash).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_homeostatic_coupling.py      # sample traces
Output: data_paths.MPMF/circadian_data/homeostatic/scn_homeostatic_coupling.{png,pdf}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import math
import numpy as np

TAU_ON = 0.5
BASE = dict(K_pre=0.15, r_star=0.2, d_theta=6.0, tau_g=2.4, lam=0.015, u_res=0.1, tau_off=12.0)


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def tissue_level(t, t_ttx, t_wash, u_res, tau_off, tau_on=TAU_ON):
    """TTX level in the tissue (exact solution of the driver kinetics): 0 before TTX, fast rise (τ_on) to 1,
    gradual decline (τ_off) to u_res after washout. Works on scalars and arrays."""
    t = np.asarray(t, float)
    u_w = 1.0 - np.exp(-(t_wash - t_ttx) / tau_on)
    return np.where(t < t_ttx, 0.0,
                    np.where(t < t_wash, 1.0 - np.exp(-np.clip(t - t_ttx, 0, None) / tau_on),
                             u_res + (u_w - u_res) * np.exp(-np.clip(t - t_wash, 0, None) / tau_off)))


def homeostatic_coupling(K_pre, r_star, d_theta, tau_g, lam, u_res, t_ttx, t_wash, T, tau_off=12.0, dt=0.05,
                         tau_on=TAU_ON):
    """Controller from the untreated steady state (g = 0) over [0, T] h: u(t) exact (tissue_level), g by Heun's
    method (second order; dt ≤ 0.25 h is accurate for τ_g ≥ 0.2 h). Returns t, K, r, g, u on the grid dt —
    use np.interp(t_query, t, K) as K_s(t)."""
    n = int(round(T / dt)) + 1; t = np.arange(n) * dt
    u = tissue_level(t, t_ttx, t_wash, u_res, tau_off, tau_on)
    th = -math.log(r_star / (1 - r_star)) + d_theta * u        # effective threshold θ_0 + Δθ u
    g = np.zeros(n); gk = 0.0
    f = lambda gg, k: ((r_star - 1.0 / (1.0 + math.exp(-(gg - th[k])))) - lam * gg) / tau_g
    for k in range(n - 1):
        k1 = f(gk, k); k2 = f(gk + dt * k1, k + 1)
        gk = gk + 0.5 * dt * (k1 + k2); g[k + 1] = gk
    r = sigma(g - th)
    return t, K_pre * r / r_star, r, g, u


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import data_paths as dp
    import scn_common as S
    out_dir = dp.ensure(os.path.join(S.OUT, "homeostatic"))
    t_ttx, t_wash, T = 90.0, 234.0, 400.0                  # SCN1 timing
    sweeps = [("tau_g", "τ_g (h)", [1.2, 2.4, 4.0]),
              ("lam", "λ", [0.0, 0.015, 0.03]),
              ("r_star", "r*", [0.1, 0.2, 0.4]),
              ("d_theta", "Δθ", [3.0, 6.0, 9.0]),
              ("u_res", "u_res", [0.0, 0.1, 0.3]),
              ("tau_off", "τ_off (h)", [1.0, 12.0, 36.0])]
    fig = plt.figure(figsize=(10.5, 9.2), layout="constrained")
    gs = fig.add_gridspec(4, 3)
    shade = lambda a: (a.axvspan(t_ttx / 24, t_wash / 24, color="0.92", zorder=0))
    # (a) anatomy of the baseline trace: driver, threshold, excitability, rate, K
    t, K, r, g, u = homeostatic_coupling(**BASE, t_ttx=t_ttx, t_wash=t_wash, T=T)
    th0 = -np.log(BASE["r_star"] / (1 - BASE["r_star"]))
    a = fig.add_subplot(gs[0, 0]); shade(a)
    a.plot(t / 24, th0 + BASE["d_theta"] * u, color="#b2182b", lw=1.1, label=r"threshold $\theta_0+\Delta\theta\,u$")
    a.plot(t / 24, g, color="#2166ac", lw=1.1, label="excitability $g$")
    a.set_ylabel("g, threshold"); a.set_title("(a) controller: g climbs to the raised threshold", fontsize=8)
    a.legend(fontsize=6, frameon=False, loc="upper left")
    a = fig.add_subplot(gs[0, 1]); shade(a)
    a.plot(t / 24, r, color="0.2", lw=1.1, label="rate $r$"); a.axhline(BASE["r_star"], color="0.5", ls=":", lw=0.9,
                                                                       label="set point $r^*$")
    a.plot(t / 24, u, color="#e08214", lw=0.9, label="TTX level $u$")
    a.set_title("(b) spike rate and driver", fontsize=8); a.legend(fontsize=6, frameon=False, loc="upper left")
    a = fig.add_subplot(gs[0, 2]); shade(a)
    a.plot(t / 24, K, color="k", lw=1.3); a.set_ylabel(r"$K_s(t)$ (rad/h)")
    a.set_title(f"(c) coupling, baseline: " + ", ".join(f"{k}={v:g}" for k, v in BASE.items() if k != "K_pre"),
                fontsize=6.5)
    cols = ["#2166ac", "k", "#b2182b"]
    for n_, (key, lab, vals) in enumerate(sweeps):
        a = fig.add_subplot(gs[1 + n_ // 3, n_ % 3]); shade(a)
        for v, c in zip(vals, cols):
            p = dict(BASE); p[key] = v
            t, K, *_ = homeostatic_coupling(**p, t_ttx=t_ttx, t_wash=t_wash, T=T)
            a.plot(t / 24, K, color=c, lw=1.1, label=f"{lab} = {v:g}")
        a.set_yscale("log"); a.set_ylim(1e-4, 2.0)
        a.set_title(f"({'defghi'[n_]}) vary {lab}", fontsize=8)
        a.legend(fontsize=6, frameon=False, loc="lower left")
        if n_ % 3 == 0:
            a.set_ylabel(r"$K_s(t)$ (rad/h, log)")
        if n_ >= 3:
            a.set_xlabel("time (days)")
    # (i) the asymmetry: recovery time vs rebound decay time as functions of r*
    a = fig.add_subplot(gs[3, 0])
    rs = np.linspace(0.05, 0.6, 12); trec, treb = [], []
    for rv in rs:
        p = dict(BASE); p["r_star"] = rv; p["lam"] = 0.0
        t, K, *_ = homeostatic_coupling(**p, t_ttx=t_ttx, t_wash=t_wash, T=T)
        Kp = p["K_pre"]; m1 = (t >= t_ttx) & (t < t_wash); m2 = t >= t_wash
        i_min = np.where(m1)[0][np.argmin(K[m1])]; i_max = np.where(m2)[0][np.argmax(K[m2])]
        up = np.where(m1 & (np.arange(t.size) > i_min) & (K >= 0.5 * Kp))[0]   # after the TTX dip
        dn = np.where(m2 & (np.arange(t.size) > i_max) & (K <= 1.5 * Kp))[0]   # after the rebound peak
        trec.append(t[up[0]] - t_ttx if up.size else np.nan); treb.append(t[dn[0]] - t_wash if dn.size else np.nan)
    a.plot(rs, trec, "o-", color="#2166ac", ms=3, label="recovery under TTX (to K_pre/2)")
    a.plot(rs, treb, "s-", color="#b2182b", ms=3, label="rebound decay (to 1.5 K_pre)")
    a.plot(rs, BASE["tau_g"] * BASE["d_theta"] / rs, color="#2166ac", lw=0.7, ls=":", label=r"$\tau_g\Delta\theta/r^*$")
    a.set_yscale("log"); a.set_xlabel("set point $r^*$"); a.set_ylabel("time (h)")
    a.set_title("(j) one τ_g, asymmetric time scales (λ = 0)", fontsize=8); a.legend(fontsize=6, frameon=False)
    # (k) the gradual washout: TTX level and rate for the τ_off values
    a = fig.add_subplot(gs[3, 1:]); shade(a)
    for v, c in zip([1.0, 12.0, 36.0], cols):
        p = dict(BASE); p["tau_off"] = v
        t, K, r, g, u = homeostatic_coupling(**p, t_ttx=t_ttx, t_wash=t_wash, T=T)
        a.plot(t / 24, u, color=c, lw=1.0, ls="--", label=f"u, τ_off = {v:g} h")
        a.plot(t / 24, r, color=c, lw=1.1, label=f"r, τ_off = {v:g} h")
    a.set_xlim(t_wash / 24 - 1, t_wash / 24 + 4); a.set_xlabel("time (days)"); a.set_ylabel("u (dashed), r")
    a.set_title("(k) gradual washout: TTX level u and spike rate r", fontsize=8)
    a.legend(fontsize=6, frameon=False, ncol=3, loc="upper right")
    out = os.path.join(out_dir, "scn_homeostatic_coupling")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")
    # numbers of the baseline trace
    t, K, r, g, u = homeostatic_coupling(**BASE, t_ttx=t_ttx, t_wash=t_wash, T=T)
    at = lambda x: K[np.argmin(np.abs(t - x))]
    print(f"baseline: K_pre {at(80):.3f}, K just after TTX {at(t_ttx + 3):.4f}, K at +1/+3/+5 d "
          f"{at(t_ttx + 24):.4f}/{at(t_ttx + 72):.4f}/{at(t_ttx + 120):.4f}, end of TTX {at(t_wash - 1):.4f}, "
          f"max after washout {K[t >= t_wash].max():.3f}, K at wash + 1/+3 d {at(t_wash + 24):.3f}/{at(t_wash + 72):.3f}, "
          f"end {K[-1]:.3f}")


if __name__ == "__main__":
    figure()
