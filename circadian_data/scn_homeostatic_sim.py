r"""
Interactive LMMF simulation of one SCN slice with HOMEOSTATIC TTX-sensitive coupling and TTX-dependent noise
============================================================================================================

Cell i (θ = PER2 phase), one global harmonic coupling term:

    θ̇_i = ω_i + K_s(t) Im(e^{−iα_s} Z e^{−iθ_i})  +  Cauchy phase noise of scale γ u(t)

Coupling (scn_homeostatic_coupling; time in h):
    τ_u du/dt = u_ext(t) − u                  TTX level in the tissue; u_ext = 0 | 1 | u_res (pre | TTX | washout);
                                              τ_u = τ_on (fast application) | τ_off (gradual washout)
    r         = σ(g − θ_0 − Δθ u)             spike rate, θ_0 = −logit(r*) (r = r* untreated)
    τ_g dg/dt = (r* − r) − λ g                homeostatic excitability (leaky integral control)
    K_s       = K_pre r / r*                  coupling ∝ spike rate
Noise: Cauchy (Lorentzian) phase noise with scale γ u(t), i.e. an extra Lorentzian width γ u(t) of every mixture
component (exact for OA ensembles; Tönjes & Pikovsky 2020). It follows the SAME tissue TTX level u as the block:
none before TTX, γ under TTX, γ u_res after an incomplete washout — the residual noise lowers the post-washout
coherence without an extra parameter.

LMMF: one Lorentzian mixture per lobe fitted to the TTX-epoch frequencies of the fit target (scn_targets: epoch-peak
phases), weights × lobe fractions; started from the measured state (scn_targets.start_hour) and run to the end of
the valid window with scn_lmmf_sim.simulate:
    ż_m = (iΩ_m − (Δ_m + γ u)) z_m + ½ (H − H̄ z_m²),   H = K_s(t) e^{−iα_s} Z
The model is compared with the data as c(t)|Z| (measurement ceiling; --no-ceiling: |Z|), RMSE over the mask.

Figure: lobe frequency distributions + mixtures | data vs model R(t) (slice, lobes) | inter-lobe phase
difference | inputs K_s(t) (log), spike rate r, TTX level u and noise γu.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_homeostatic_sim.py --slice 1 --K-pre 0.15 --r-star 0.2 \
        --d-theta 6 --tau-g 2.4 --lam 0.015 --u-res 0.1 --tau-off 12 --alpha-s 0.0 --gamma 0.01 [--tau-on 0.5] \
        [--no-floor] [--lam-mix 1e-5] \
        [--width-k 10] [--no-ceiling] [--show]
Units: rates, couplings and γ in rad/h; τ in h; Δθ, λ, r*, u_res dimensionless.
(The figure's input panel shows u(t): fast rise at application, gradual decline over ~τ_off after washout.)
Output: data_paths.MPMF/circadian_data/homeostatic/sim/scn<slice>_<tag>.{png,pdf}
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import argparse
import numpy as np
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_lmmf_sim as SIM
import scn_homeostatic_coupling as HC

OUT = dp.ensure(os.path.join(S.OUT, "homeostatic", "sim"))


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice", type=int, default=1, choices=S.SLICES)
    p.add_argument("--K-pre", type=float, default=HC.BASE["K_pre"], help="untreated coupling (rad/h)")
    p.add_argument("--r-star", type=float, default=HC.BASE["r_star"], help="rate set point r* (0–1)")
    p.add_argument("--d-theta", type=float, default=HC.BASE["d_theta"], help="TTX threshold shift Δθ")
    p.add_argument("--tau-g", type=float, default=HC.BASE["tau_g"], help="homeostatic time constant τ_g (h)")
    p.add_argument("--lam", type=float, default=HC.BASE["lam"], help="controller leak λ (0 = perfect homeostasis)")
    p.add_argument("--u-res", type=float, default=HC.BASE["u_res"], help="residual TTX level after washout (0–1)")
    p.add_argument("--tau-off", type=float, default=HC.BASE["tau_off"], help="washout time constant τ_off (h)")
    p.add_argument("--tau-on", type=float, default=HC.TAU_ON, help="application time constant τ_on (h)")
    p.add_argument("--alpha-s", type=float, default=0.0, help="phase lag of the coupling (rad)")
    p.add_argument("--gamma", type=float, default=0.0, help="Cauchy noise scale at full TTX (rad/h); noise = γ u(t)")
    p.add_argument("--lam-mix", type=float, default=1e-5, help="per-component penalty λ of the mixture fits")
    p.add_argument("--no-floor", action="store_true", help="mixture fits without the noise floor")
    p.add_argument("--width-k", type=float, default=10.0, help="width floor k (0 = off)")
    p.add_argument("--no-ceiling", action="store_true", help="compare |Z| instead of c(t)|Z| with the data")
    p.add_argument("--show", action="store_true", help="open the figure in a window after saving")
    return p.parse_args()


def run(a, d=None, model=None):
    """Simulate with the arguments `a`; returns the target d, the mixture, coupling and LMMF traces."""
    d = TG.load(a.slice) if d is None else d
    T = int(d["T"]); floor = not a.no_floor
    model = SIM.build(d, a.lam_mix, floor, a.width_k) if model is None else model
    tc, Kc, rc, gc, uc = HC.homeostatic_coupling(a.K_pre, a.r_star, a.d_theta, a.tau_g, a.lam, a.u_res,
                                                 float(d["ttx"]), float(d["wash"]), float(T), tau_off=a.tau_off,
                                                 tau_on=a.tau_on)
    Kfun = lambda t: np.interp(t, tc, Kc); ufun = lambda t: np.interp(t, tc, uc)
    v0, v1 = S.valid_window(T); t0 = TG.start_hour(d, v0)
    tm, Zg, ZA, ZB = SIM.simulate(model, SIM.initial_state(model, d["PH"][t0]), Kfun, ufun, 0.0, a.alpha_s, 0.0,
                                  False, 0.0, a.gamma, t0, v1 - 1)
    return dict(d=d, model=model, tc=tc, Kc=Kc, rc=rc, gc=gc, uc=uc, tm=tm, Zg=Zg, ZA=ZA, ZB=ZB, v=(v0, v1))


def main():
    a = parse()
    res = run(a); d = res["d"]; model = res["model"]; tm = res["tm"]; T = int(d["T"]); v0, v1 = res["v"]
    om = d["om"]; floor = not a.no_floor
    ti = tm.astype(int); one = np.ones(T)
    cg, cA, cB = (one, one, one) if a.no_ceiling else (d["c"], d["cA"], d["cB"])
    Mg, MA, MB = cg[ti] * np.abs(res["Zg"]), cA[ti] * np.abs(res["ZA"]), cB[ti] * np.abs(res["ZB"])
    wm = d["mask"][ti] > 0
    vw = lambda x: np.where((np.arange(T) >= v0) & (np.arange(T) < v1), x, np.nan)
    Rd, RdA, RdB = vw(d["R"]), vw(d["RA"]), vw(d["RB"])
    rmse = lambda M, R: np.sqrt(np.nanmean((M[wm] - R[ti][wm]) ** 2))
    at = lambda x: np.interp(x, res["tc"], res["Kc"])
    print(f"[SCN{a.slice}] n = {om.size} cells, mixtures M = {np.bincount(model['lobe']).tolist()} "
          f"({'floor' if floor else 'no floor'}); K_pre {a.K_pre}, r* {a.r_star}, Δθ {a.d_theta}, τ_g {a.tau_g} h, "
          f"λ {a.lam}, u_res {a.u_res}, τ_on/τ_off {a.tau_on}/{a.tau_off} h, α_s {a.alpha_s}, γ {a.gamma}")
    print(f"  K_s: min under TTX {res['Kc'][(res['tc'] >= d['ttx']) & (res['tc'] < d['wash'])].min():.4f}, "
          f"end of TTX {at(d['wash'] - 1):.4f}, rebound peak {res['Kc'][res['tc'] >= d['wash']].max():.3f}, "
          f"end {res['Kc'][-1]:.4f} (K_pre {a.K_pre}); noise after washout γ u_res = {a.gamma * a.u_res:.4g} rad/h")
    print(f"  RMSE(R, mask) slice {rmse(Mg, Rd):.3f}, lobes {rmse(MA, RdA):.3f}/{rmse(MB, RdB):.3f}")

    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from prl_style import set_prl_style
        set_prl_style("prl")
    except Exception:
        pass
    c_fit = "#eb6834" if floor else "#2a78d6"; CL = ["#1b7837", "#762a83"]
    fig = plt.figure(figsize=(7.0, 4.6), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 2.2], height_ratios=[1.0, 0.4, 0.55])
    ax0 = fig.add_subplot(gs[:, 0]); ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1); ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    cd = 24 / (2 * np.pi)
    g = np.linspace(om.min() - 0.01, om.max() + 0.01, 800)
    for k in (0, 1):
        s = model["lobe"] == k; ww = model["w"][s] / model["w"][s].sum()
        pdf = (ww[:, None] * model["De"][s, None] / np.pi / ((g[None] - model["Om"][s, None]) ** 2
                                                             + model["De"][s, None] ** 2)).sum(0)
        ax0.hist(om[d["lobe"] == k] * cd, bins=30, density=True, color=CL[k], alpha=0.3,
                 label=f"lobe {'AB'[k]} (n = {(d['lobe'] == k).sum()})")
        ax0.plot(g * cd, pdf / cd, color=CL[k], lw=1.1, label=f"fit, M = {s.sum()}")
    ax0.set_xlabel("intrinsic frequency (cycles/day)"); ax0.set_ylabel("density"); ax0.set_yticks([])
    ax0.legend(fontsize=5.5, frameon=False, loc="upper left")
    td = np.arange(T) / 24
    for ax in (ax1, ax2, ax3):
        ax.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
    for t_c in (d["ttx"], d["wash"]):
        ax1.axvspan((t_c - d["mask_h"]) / 24, (t_c + d["mask_h"]) / 24, color="none", hatch="////", ec="0.75",
                    lw=0, zorder=0)
    ax1.plot(td, Rd, color="0.1", lw=1.2, label="data, slice")
    ax1.plot(tm / 24, Mg, color=c_fit, lw=1.1, label="LMMF, slice" + ("" if a.no_ceiling else r", $c(t)|Z|$"))
    if not a.no_ceiling:
        ax1.plot(td, vw(d["c"]), color="0.6", lw=0.7, ls=":", label="ceiling $c(t)$")
    for k, (Rdat, Mm) in enumerate(((RdA, MA), (RdB, MB))):
        ax1.plot(td, Rdat, color=CL[k], lw=0.6, alpha=0.8)
        ax1.plot(tm / 24, Mm, color=CL[k], lw=0.7, ls="--")
    ax1.set_ylim(0, 1.02); ax1.set_ylabel(r"$R(t)$"); ax1.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax1.set_title(rf"SCN slice {a.slice}: homeostatic $K_s$, noise $\gamma u(t)$; RMSE slice {rmse(Mg, Rd):.3f} "
                  "(lobes: thin = data, dashed = LMMF; hatched = masked)", fontsize=7)
    H = 24 / (2 * np.pi)
    ax2.plot(td, vw(np.angle(d["ZA"] * np.conj(d["ZB"]))) * H, color="0.1", lw=1.0, label="data")
    ax2.plot(tm / 24, np.angle(res["ZA"] * np.conj(res["ZB"])) * H, color=c_fit, lw=0.9, label="LMMF")
    ax2.axhline(0, color="0.6", lw=0.5); ax2.set_ylabel(r"$\Delta\psi_{AB}$ (h)")
    ax2.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2)
    tc = res["tc"]
    ax3.plot(tc / 24, res["Kc"], color="k", lw=1.1, label=r"$K_s(t)$")
    ax3.axhline(a.K_pre, color="0.5", lw=0.6, ls=":")
    ax3.set_yscale("log"); ax3.set_ylim(max(res["Kc"].min() * 0.5, 1e-5), res["Kc"].max() * 2)
    ax3.set_ylabel(r"$K_s$ (rad/h)"); ax3.set_xlabel("time (days)"); ax3.set_xlim(0, T / 24)
    b3 = ax3.twinx()
    b3.plot(tc / 24, res["rc"], color="#2166ac", lw=0.8, label="rate $r$")
    b3.plot(tc / 24, res["uc"], color="#e08214", lw=0.8, label="TTX level $u$ (noise $\\gamma u$)")
    b3.set_ylim(0, 1.05); b3.set_ylabel("r, u", fontsize=7); b3.tick_params(labelsize=6)
    h1, l1 = ax3.get_legend_handles_labels(); h2, l2 = b3.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=5.5, frameon=False, loc="upper left", ncol=3)
    plt.setp(ax1.get_xticklabels(), visible=False); plt.setp(ax2.get_xticklabels(), visible=False)
    tag = (f"Kpre{a.K_pre:g}_rs{a.r_star:g}_dth{a.d_theta:g}_tg{a.tau_g:g}_lam{a.lam:g}_ures{a.u_res:g}_toff{a.tau_off:g}"
           f"_as{a.alpha_s:g}_g{a.gamma:g}_{'floor' if floor else 'nofloor'}" + ("_noceil" if a.no_ceiling else ""))
    out = os.path.join(OUT, f"scn{a.slice}_{tag}")
    fig.savefig(out + ".png", dpi=250); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
