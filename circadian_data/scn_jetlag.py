r"""
Jet lag: re-entrainment of the SCN mean field after an instantaneous shift of the light-dark cycle
================================================================================================

Minimal mean field (global coupling, α = 0, lab frame), pooled-lobe mixtures "pfloor" (noise floor) and "psingle"
(one Lorentzian), driven by the smoothed square-wave light of scn_entrainment / scn_arnold_tongue:
    ż_m = (iΩ_m − Δ_m) z_m + ½ (H − H̄ z_m²),   H = K Z + I(t),   I(t) = F · ½ [1 + tanh(β sin θ_d) / tanh β]
    θ_d(t) = ω_d t + Δ(t),   ω_d = 2π / 24 h,   F = 0.06 rad/h,   β = 4.2 (12:12, ~2-h transitions)
Jet-lag protocol (user, 2026-10-21): the model is entrained (300 days of forcing; periodicity checked); at t₀ the
driver phase jumps instantaneously, Δ = ω_d · X h. X > 0: eastward flight (light comes X h earlier = phase advance,
the day of the jump is shortened); X < 0: westward (phase delay). X ∈ SHIFTS (4-h grid; ±12 h are the same shift).
Couplings: N_K values geometrically spaced in [1.05 K_c, K_pre] of the pooled noise-floor mixture (K_pre: R*(K) equals
the measured pre-TTX coherence), the SAME for both mixtures; K_c, K_mid, K_post (scn_entrainment) marked.

Readouts after the jump (target = the model's own entrained orbit, shifted by X h: Z_tgt(t) = Z_ent(t + X h)):
    phase error         ψ(t) = arg(Z(t) Z̄_tgt(t)) in hours (× 24 / 2π), unwrapped from ψ(t₀) = −X... i.e. the
                        rhythm starts X h behind (east) / ahead (west) of the new target
    re-entrainment time T_1h = last time with |ψ| > 1 h (censored at T_POST if not reached)
    route               "direct" if the unwrapped ψ goes to 0 without wrapping (the rhythm shifts in the direction of
                        the jump), "around" if it re-entrains via ψ → ±24 h (the long way round)
    coherence dip       ΔR = max_t [R_tgt(t) − R(t)] (largest drop below the entrained coherence) and min_t R(t)
A grid point whose 24-h cycle is NOT entrained at F (outside the tongue, or no 1:1 locking) is flagged and skipped.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_jetlag.py [--slices 1,2] [--F 0.06] [--resim] [--show]
Output: data_paths.MPMF/circadian_data/jetlag/  (jetlag_scn<i>_<kind>_F<F>.npz, scn_jetlag_F<F>.{pdf,png})
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import json
import time
import numpy as np
import data_paths as dp
import scn_pycobi_bifurcation as B
import scn_entrainment as E

OUT = dp.ensure(dp.mpmf("circadian_data", "jetlag"))
KINDS = (("pfloor", "#c0392b", "-", "noise floor"), ("psingle", "0.3", "--", "single Lorentzian"))
SHIFTS = (-8, -4, 4, 8, 12)                 # h; > 0 east (advance), < 0 west (delay)
F_DEF, BETA, T_D = 0.06, 4.2, 24.0
N_K = 30
N_PER = 96                                  # RK4 steps per 24 h (dt = 0.25 h)
T_PRE, T_POST = 300, 120                    # days of entrainment before / follow-up after the jump
ENTRAIN_TOL = 1e-6                          # max |Z(t + 24 h) − Z(t)| over the last days before the jump
PHASE_TOL = 1.0                             # h


def K_grid(i):
    g = np.load(os.path.join(B.OUT, f"scn{i}_pfloor.npy"), allow_pickle=True).item()
    eq, Kc = g["eq"], float(g["kc"]["K_c"])
    R_pre = json.load(open(dp.mpmf("circadian_data", "bifurcation", "scn_bif_tests.json")))[str(i)]["floor"]["R_pre"]
    up = eq["stable"] & (eq["K"] > 1.5 * Kc); o = np.argsort(eq["K"][up])
    K_pre = float(np.interp(R_pre, eq["R"][up][o], eq["K"][up][o]))
    return np.geomspace(1.05 * Kc, K_pre, N_K), Kc, K_pre


def light(F, theta):
    return 0.5 * F * (1.0 + np.tanh(BETA * np.sin(theta)) / np.tanh(BETA))


def rhs(z, K, I, w, a):
    H = K * (z @ w) + I
    return a * z + 0.5 * (H[..., None] - np.conj(H)[..., None] * z ** 2)


def step(z, th, dth, K, F, w, a, dt):
    """RK4 step; driver phase th (G,) advances by dth = ω_d dt."""
    I0, Ih, I1 = light(F, th), light(F, th + 0.5 * dth), light(F, th + dth)
    k1 = rhs(z, K, I0, w, a); k2 = rhs(z + 0.5 * dt * k1, K, Ih, w, a)
    k3 = rhs(z + 0.5 * dt * k2, K, Ih, w, a); k4 = rhs(z + dt * k3, K, I1, w, a)
    return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run_model(i, kind, F, Ks=None, shifts=SHIFTS):
    """All (K, X) of one slice and mixture in one vectorised integration (default: the K grid of K_grid and SHIFTS).
    Also records the phase error of every mixture component (cluster) against its own shifted entrained orbit."""
    w, Om, De = E.mixture_abs(i, kind); a = 1j * Om - De; M = w.size
    Kg, Kc, K_pre = K_grid(i)
    Ks = Kg if Ks is None else np.atleast_1d(np.asarray(Ks, float))
    G = Ks.size; dt = T_D / N_PER; wd = 2 * np.pi / T_D; dth = wd * dt
    Kv = Ks                                                              # (G,): H = K (z @ w) is (G,) / (S, G)
    # 1. entrain: from a synchronised state, 300 days of forcing
    z = np.repeat((0.9 * np.ones(M, complex))[None], G, axis=0); th = 0.0
    strobe = []
    for n in range(T_PRE * N_PER):
        z = step(z, th, dth, Kv, F, w, a, dt); th += dth
        if (n + 1) % N_PER == 0 and n >= (T_PRE - 6) * N_PER:
            strobe.append(z @ w)
    strobe = np.array(strobe)
    drift = np.abs(np.diff(strobe, axis=0)).max(0)
    entrained = drift < ENTRAIN_TOL
    # one period of the entrained orbit (driver phase 0 at the start, i.e. th ≡ 0 mod 2π)
    orbit_z = [z]; zz = z.copy(); tt = th
    for _ in range(N_PER):
        zz = step(zz, tt, dth, Kv, F, w, a, dt); tt += dth; orbit_z.append(zz)
    orbit_z = np.array(orbit_z[:-1])                                     # (N_PER, G, M): components
    orbit = orbit_z @ w                                                  # (N_PER, G): Z_ent at τ = k dt
    # 2. jump: all shifts at once, (S, G) batch
    S = len(shifts); shifts = np.array(shifts, float)
    zS = np.repeat(z[None], S, axis=0)                                   # (S, G, M)
    thS = th + wd * shifts[:, None] * np.ones((1, G))                    # new driver phase (S, G)
    KvS = Ks
    n_post = T_POST * N_PER
    psi = np.zeros((S, G)); last_out = np.zeros((S, G)); dR = np.zeros((S, G)); Rmin = np.full((S, G), np.inf)
    rec_every = 4; rec_t, rec_psi, rec_R, rec_cm = [], [], [], []
    k0 = 0                                                               # index of τ in the entrained orbit
    sh_steps = np.rint(shifts / dt).astype(int)                          # target: Z_ent(τ + X h)
    Z = zS @ w
    tgt = orbit[(k0 + sh_steps[:, None]) % N_PER, np.arange(G)[None]]
    psi = np.angle(Z * np.conj(tgt)) * T_D / (2 * np.pi)                 # (S, G), ≈ −X mod 24 at the jump
    for n in range(1, n_post + 1):
        zS = step(zS, thS, dth, KvS, F, w, a, dt); thS = thS + dth
        Z = zS @ w
        tgt = orbit[(n + sh_steps[:, None]) % N_PER, np.arange(G)[None]]
        dpsi = np.angle(Z * np.conj(tgt)) * T_D / (2 * np.pi) - np.mod(psi + 12, 24) + 12
        psi = psi + (np.mod(dpsi + 12, 24) - 12)                         # unwrapped phase error (h)
        R, Rt = np.abs(Z), np.abs(tgt)
        dR = np.maximum(dR, Rt - R); Rmin = np.minimum(Rmin, R)
        off = np.abs(np.mod(psi + 12, 24) - 12) > PHASE_TOL
        last_out = np.where(off, n * dt, last_out)
        if n % rec_every == 0:
            rec_t.append(n * dt); rec_psi.append(psi.copy()); rec_R.append(R.copy())
            tgt_m = orbit_z[(n + sh_steps[:, None]) % N_PER, np.arange(G)[None]]          # (S, G, M)
            rec_cm.append(np.angle(zS * np.conj(tgt_m)))                                # wrapped (rad)
    psi_end = psi
    T1h = np.where(last_out >= T_POST * 24 - dt, np.nan, last_out) / 24.0     # days (NaN: not re-entrained)
    route = np.where(np.abs(psi_end) < 6, "direct", "around")
    T1h[:, ~entrained] = np.nan; dR[:, ~entrained] = np.nan; Rmin[:, ~entrained] = np.nan
    return dict(K=Ks, K_c=Kc, K_pre=K_pre, shifts=shifts, entrained=entrained, drift=drift, T1h=T1h, dR=dR,
                Rmin=Rmin, route=route, psi_end=psi_end, R_ent=np.abs(orbit).mean(0), t_rec=np.array(rec_t) / 24.0,
                psi_rec=np.array(rec_psi), R_rec=np.array(rec_R), F=F, w=w, Om=Om, De=De,
                psi_comp_rec=np.unwrap(np.array(rec_cm), axis=0) * T_D / (2 * np.pi))   # (T, S, G, M), h


def main():
    rows = (tuple(int(s) for s in sys.argv[sys.argv.index("--slices") + 1].split(","))
            if "--slices" in sys.argv else (1, 2))
    F = float(sys.argv[sys.argv.index("--F") + 1]) if "--F" in sys.argv else F_DEF
    show = "--show" in sys.argv
    res = {}
    for i in rows:
        for kind, *_ in KINDS:
            f = os.path.join(OUT, f"jetlag_scn{i}_{kind}_F{F:g}.npz")
            if "--resim" in sys.argv or not os.path.exists(f):
                t0 = time.time(); r = run_model(i, kind, F); np.savez(f, **r)
                print(f"[jetlag] SCN{i} {kind}: {r['entrained'].sum()}/{r['K'].size} K entrained at F = {F}; "
                      f"T_1h (d) median per shift " + ", ".join(
                          f"{int(x):+d}h: {np.nanmedian(r['T1h'][s]):.1f}" for s, x in enumerate(r["shifts"]))
                      + f" ({time.time() - t0:.0f} s)", flush=True)
            res[(i, kind)] = dict(np.load(f, allow_pickle=True))
    figure(res, rows, F, show)


def figure(res, rows, F, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from prl_style import set_prl_style
    FS = 11
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center", "font.size": FS,
                            "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS - 1,
                            "ytick.labelsize": FS - 1, "legend.fontsize": FS - 1})
    Kmeta = E.load_K()
    cmap = plt.get_cmap("coolwarm")
    shifts = res[(rows[0], KINDS[0][0])]["shifts"]
    scol = {x: cmap(0.5 + 0.5 * x / 12) for x in shifts}
    fig, axs = plt.subplots(len(rows), 4, figsize=(18, 4.2 * len(rows) + 0.8), squeeze=False,
                            gridspec_kw=dict(hspace=0.45, wspace=0.3, left=0.05, right=0.99, top=0.9, bottom=0.14))
    for r, i in enumerate(rows):
        Kc_, Km, Kp = Kmeta[str(i)]["K"]
        for c, (key, ylab) in enumerate((("T1h", "re-entrainment time (days)"), ("dR", r"coherence dip $\Delta R$"))):
            a = axs[r, c]
            for kind, col, ls, lab in KINDS:
                d = res[(i, kind)]
                for s, x in enumerate(d["shifts"]):
                    y = d[key][s]
                    a.plot(d["K"], y, color=scol[x], ls=ls, lw=1.8 if kind == "pfloor" else 1.3, marker="o" if kind ==
                           "pfloor" else None, ms=3)
                    if key == "T1h":
                        around = (d["route"][s] == "around") & np.isfinite(y)
                        a.plot(d["K"][around], y[around], "x", color=scol[x], ms=7, mew=1.5)
                nonent = ~d["entrained"].astype(bool)
                if nonent.any():
                    a.plot(d["K"][nonent], np.zeros(nonent.sum()), "|", color=col, ms=12, mew=2, clip_on=False)
            for Kv_, nm in ((Kc_, r"$K_c$"), (Km, r"$K_{mid}$"), (Kp, r"$K_{post}$")):
                a.axvline(Kv_, color="0.6", lw=0.8, ls=":")
                a.text(Kv_, 1.01, nm, transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS - 1)
            a.set_xscale("log"); a.set_xlabel(r"coupling $K$ (rad/h)"); a.set_ylabel(ylab)
            a.set_ylim(bottom=0)
            a.set_title(f"SCN{i}", loc="left", fontsize=FS + 1, pad=16)
        # examples: phase error and R after the +8 h (east) and −8 h (west) jumps at K_post, both models
        Kex = Kp
        for c, x in ((2, 8.0), (3, -8.0)):
            a = axs[r, c]; b = a.twinx()
            for kind, col, ls, lab in KINDS:
                d = res[(i, kind)]; k = int(np.argmin(np.abs(np.log(d["K"] / Kex)))); s = list(d["shifts"]).index(x)
                a.plot(d["t_rec"], d["psi_rec"][:, s, k], color=col, ls=ls, lw=1.6)
                b.plot(d["t_rec"], d["R_rec"][:, s, k], color=col, ls=ls, lw=0.9, alpha=0.5)
            a.axhspan(-PHASE_TOL, PHASE_TOL, color="0.9", zorder=0)
            a.set_xlim(0, 30); a.set_xlabel("days after the jump")
            a.set_ylabel(r"phase error $\psi$ (h)"); b.set_ylabel(r"$R(t)$ (thin)", color="0.4"); b.set_ylim(0, 1.05)
            a.set_title(f"{'east' if x > 0 else 'west'} {abs(int(x))} h, $K$ ≈ $K_{{post}}$ = {Kp:.3f}",
                        fontsize=FS, pad=16)
    hd = ([Line2D([], [], color=scol[x], lw=1.8, label=f"{int(x):+d} h ({'east' if x > 0 else 'west'})")
           for x in shifts] +
          [Line2D([], [], color="0.3", lw=1.8, marker="o", ms=3, label="noise floor"),
           Line2D([], [], color="0.3", lw=1.3, ls="--", label="single Lorentzian"),
           Line2D([], [], ls="none", marker="x", color="0.3", ms=7, mew=1.5, label="re-entrains the long way round"),
           Line2D([], [], ls="none", marker="|", color="0.3", ms=12, mew=2, label="not entrained before the jump")])
    fig.legend(handles=hd, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Jet lag: instantaneous shift of the 24-h light cycle (F = {F:g} rad/h, 12:12, β = {BETA})",
                 fontsize=FS + 2, weight="bold")
    out = os.path.join(OUT, f"scn_jetlag_F{F:g}_scn" + "".join(map(str, rows)))
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
