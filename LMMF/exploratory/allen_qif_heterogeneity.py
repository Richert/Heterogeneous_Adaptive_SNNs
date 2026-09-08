"""Heterogeneity-controlled oscillations in the Allen-QIF inhibitory mean field.

Inhibitory recurrent coupling through an alpha-synapse can produce (fast/ING-type) oscillations
when the excitatory drive I is strong enough, BUT heterogeneity desynchronises and suppresses
them. We make the system's heterogeneity tunable around the data fit with two GLOBAL knobs
(both = 1 at the fit, see build_equations in allen_qif_bifurcation.py):

    h_Delta : width scaling      Delta_m = h_Delta * Delta_m^0
    h_eta   : centre-spread      Omega_m = Ombar + h_eta*(Omega_m^0 - Ombar),  Ombar = sum_m w_m Omega_m^0

h->0 is the homogeneous (single-QIF) limit. The Hopf bifurcation is the locus where the leading
complex-conjugate eigenvalue pair of the equilibrium crosses the imaginary axis (Re lambda = 0);
we map it directly (robust; validated against Auto's HB detection at the same I). The emergent
limit cycle is obtained by direct integration of the mean field.

Pure numpy/scipy/matplotlib -> runs in `allen`.

    python allen_qif_heterogeneity.py "PV+ Interneuron" "L2/3"
"""
import os
import sys
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIT_DIR = os.path.join(os.path.dirname(HERE), "data_fitting")

CONFIG = dict(
    cell_class="PV+ Interneuron", layer="L2/3", v_r=-70.0,
    J=-200.0, tau_s=2.0,                  # inhibitory regime that oscillates (fast GABA synapse)
    # PROTOCOL (per user): reduce centre-spread h_eta FIRST, then width h_Delta as the extra knob.
    hC_grid=(0.0, 1.0, 60),               # centre-spread axis of the heterogeneity plane (linear)
    hD_log=(5e-3, 1.0, 60),               # width axis of the heterogeneity plane (LOG; incl. data h=1)
    I_grid=(0.0, 1100.0, 140),            # drive sweep (max over I -> "can it oscillate at all")
    hC_for_drive=0.0,                     # centres collapsed when mapping the (h_Delta, I) tongue
    hD_drive=(1e-3, 0.20, 70),            # width range for the drive tongue / 1-D branch
    I_fix=480.0,                          # fixed drive for the 1-D branch + limit cycle (h_eta=0)
    hD_branch=(1e-3, 0.20, 160),          # h_Delta sweep for the 1-D branch (at h_eta=0)
)
if len(sys.argv) > 2:
    CONFIG["cell_class"], CONFIG["layer"] = sys.argv[1], sys.argv[2]


def _tag(cell_class, layer):
    c = cell_class.split("+")[0].split()[0].lower()
    return f"{c}_{layer.replace('/', '').replace(' ', '')}"


def load_fit(path):
    d = np.load(path, allow_pickle=False)
    w = np.asarray(d["weights"], float)
    return w / w.sum(), np.asarray(d["omega"], float), np.asarray(d["delta"], float), int(d["M"])


class Model:
    """Allen-QIF threshold-heterogeneity mean field with tunable heterogeneity (h_Delta, h_eta)."""
    def __init__(self, w, omega, delta, v_r, tau_s, J):
        self.w, self.om0, self.de0 = w, omega, delta
        self.v_r, self.tau_s, self.J = v_r, tau_s, J
        self.M = len(w)
        self.ombar = float(w @ omega)

    def rhs(self, y, I, hD, hC):
        M, PI, v_r = self.M, np.pi, self.v_r
        vth = v_r + self.ombar + hC * (self.om0 - self.ombar)
        De = hD * self.de0
        r, v, a, s = y[:M], y[M:2 * M], y[2 * M], y[2 * M + 1]
        dr = De / PI * (v - v_r) + r * (2 * v - v_r - vth)
        dv = (v - v_r) * (v - vth) - (PI * r) ** 2 - PI * De * r + I + self.J * s
        return np.concatenate([dr, dv, [(self.w @ r - a) / self.tau_s, (a - s) / self.tau_s]])

    def jac(self, y, I, hD, hC):
        f0 = self.rhs(y, I, hD, hC); n = y.size; Jm = np.empty((n, n))
        for j in range(n):
            h = 1e-6 * (1 + abs(y[j])); yp = y.copy(); yp[j] += h
            Jm[:, j] = (self.rhs(yp, I, hD, hC) - f0) / h
        return Jm

    def y0(self):
        return np.concatenate([np.full(self.M, 1e-3), np.full(self.M, self.v_r), [1e-3, 1e-3]])

    def equilibrium(self, I, hD, hC, guess):
        sol, _, ier, _ = fsolve(lambda y: self.rhs(y, I, hD, hC), guess, full_output=True, xtol=1e-12)
        ok = ier == 1 and np.max(np.abs(self.rhs(sol, I, hD, hC))) < 1e-7
        return sol, ok

    def max_reig(self, y, I, hD, hC):
        return float(np.max(np.linalg.eigvals(self.jac(y, I, hD, hC)).real))


def _trace(mdl, pts, jump_tol=0.25):
    """Predictor-corrector continuation along the (I,hD,hC) sequence `pts`. A secant predictor
    + a continuity guard (reject steps where the synaptic state s jumps by > jump_tol) keep the
    tracker on the single physical equilibrium branch (cold fsolve jumps branches at high I).
    Returns the solutions (n×dim, NaN where lost) and max Re(eig) per point."""
    n = len(pts); sols = np.full((n, 2 * mdl.M + 2), np.nan); mr = np.full(n, np.nan)
    p1 = p2 = None
    for k, (I, hD, hC) in enumerate(pts):
        guess = mdl.y0() if p1 is None else (p1 if p2 is None else 2 * p1 - p2)
        sol, ok = mdl.equilibrium(I, hD, hC, guess)
        if ok and p1 is not None and abs(sol[-1] - p1[-1]) > jump_tol:
            ok = False                                    # branch jump -> drop point, keep predictor
        if not ok:
            p2 = None                                     # reset predictor after a gap
            continue
        sols[k] = sol; mr[k] = mdl.max_reig(sol, I, hD, hC); p2, p1 = p1, sol
    return sols, mr


def maxre_over_I(mdl, hD, hC, Ivals):
    """Largest Re(eig) of the equilibrium over a drive sweep -> 'can this heterogeneity oscillate
    at all (for some I)'."""
    _, mr = _trace(mdl, [(I, hD, hC) for I in Ivals])
    return np.nanmax(mr) if np.isfinite(mr).any() else np.nan


def phase_grid(mdl, hCvals, hDvals, Ivals):
    """Heterogeneity phase plane: max-over-I Re(eig) on the (h_eta, h_Delta) grid (rows h_Delta)."""
    Z = np.full((hDvals.size, hCvals.size), np.nan)
    for i, hD in enumerate(hDvals):
        for j, hC in enumerate(hCvals):
            Z[i, j] = maxre_over_I(mdl, hD, hC, Ivals)
    return Z


def maxre_grid(mdl, hvals, Ivals, which, hD_fix, hC_fix):
    """max Re(eig) of the equilibrium over the (heterogeneity, I) grid (robust continuation in I
    per heterogeneity row). `which` in {'hD','hC'}; the other knob is held at hD_fix / hC_fix."""
    Z = np.full((Ivals.size, hvals.size), np.nan)
    for j, h in enumerate(hvals):
        hD, hC = (h, hC_fix) if which == "hD" else (hD_fix, h)
        _, mr = _trace(mdl, [(I, hD, hC) for I in Ivals])
        Z[:, j] = mr
    return Z


def branch_in_hD(mdl, hDvals, I, hC=1.0):
    """Equilibrium s(h_Delta) + stability at fixed I (continuation from high h_Delta downward)."""
    order = np.argsort(hDvals)[::-1]                       # start at the (stable) large-hD end
    sols, mr = _trace(mdl, [(I, hDvals[k], hC) for k in order])
    s_eq = np.full(hDvals.size, np.nan); stab = np.zeros(hDvals.size, bool); mro = np.full(hDvals.size, np.nan)
    s_eq[order] = sols[:, -1]; mro[order] = mr; stab[order] = mr < 0
    return s_eq, stab, mro


def limit_cycle(mdl, hD, I, hC=1.0, T=400.0, t_drop=250.0):
    """Direct integration -> steady-state min/max of s and dominant frequency (Hz, t in ms)."""
    y0 = mdl.equilibrium(I, hD, hC, mdl.y0())[0] + 1e-2
    sol = solve_ivp(lambda t, y: mdl.rhs(y, I, hD, hC), (0, T), y0, method="RK45",
                    rtol=1e-8, atol=1e-10, max_step=0.05, dense_output=False)
    t, s = sol.t, sol.y[-1]
    m = t > t_drop
    if m.sum() < 20:
        return np.nan, np.nan, np.nan
    smin, smax = float(s[m].min()), float(s[m].max())
    tg = np.linspace(t[m][0], t[m][-1], 4000); sg = np.interp(tg, t[m], s[m])
    sg -= sg.mean(); F = np.fft.rfft(sg); freq = np.fft.rfftfreq(tg.size, (tg[1] - tg[0]) / 1000.0)
    fdom = float(freq[1 + np.argmax(np.abs(F)[1:])]) if smax - smin > 1e-6 else np.nan
    return smin, smax, fdom


def zero_contour_band(ax, hvals, Ivals, Z, label):
    """Shade the unstable (Hopf) region and draw the Re=0 boundary. Returns the mesh handle."""
    vmax = np.nanmax(np.abs(Z))
    pcm = ax.pcolormesh(hvals, Ivals, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    if (Z > 0).any() and (Z < 0).any():
        ax.contour(hvals, Ivals, Z, levels=[0.0], colors="k", linewidths=2)
    ax.set_xlabel(label); ax.set_ylabel(r"external drive $I$")
    return pcm


def boundary_threshold(hgrid, mr_line):
    """Largest width h_Delta that still oscillates (upper edge of the Hopf region): the highest
    grid value with max Re(eig) > 0. Heterogeneity below this -> oscillations possible."""
    f = np.isfinite(mr_line) & (mr_line > 0)
    return float(hgrid[f].max()) if f.any() else np.nan


def main():
    cfg = CONFIG; tag = _tag(cfg["cell_class"], cfg["layer"])
    w, Om, De, M = load_fit(os.path.join(FIT_DIR, f"allen_lorentzian_{tag}.npz"))
    mdl = Model(w, Om, De, cfg["v_r"], cfg["tau_s"], cfg["J"])
    print(f"{cfg['cell_class']} {cfg['layer']}  M={M}  J={cfg['J']}  tau_s={cfg['tau_s']}")
    print(f"  Om={np.round(Om,2)}  De={np.round(De,2)}  w={np.round(w,3)}  Ombar={mdl.ombar:.2f}")

    hC = np.linspace(*cfg["hC_grid"][:2], int(cfg["hC_grid"][2]))
    hD = np.geomspace(*cfg["hD_log"][:2], int(cfg["hD_log"][2]))           # log width axis (incl. 1.0)
    hDd = np.linspace(*cfg["hD_drive"][:2], int(cfg["hD_drive"][2]))       # focused width range
    Iv = np.linspace(*cfg["I_grid"][:2], int(cfg["I_grid"][2]))

    # ── (a) heterogeneity phase plane (h_eta, h_Delta): can it oscillate for SOME drive? ──
    print("[a] heterogeneity phase plane (h_eta, h_Delta), max over I ...")
    Zphase = phase_grid(mdl, hC, hD, Iv)
    hD_thr_full = boundary_threshold(hD, Zphase[:, -1])    # reduce widths at full centres (h_eta=1)
    hD_thr_coll = boundary_threshold(hD, Zphase[:, 0])     # reduce widths at collapsed centres (h_eta=0)
    print(f"    max h_Delta that oscillates:  {hD_thr_full:.3f} at h_eta=1  vs  {hD_thr_coll:.3f} at h_eta=0")

    # ── (b) drive tongue (h_Delta, I) at collapsed centres (h_eta=0) ──
    print(f"[b] drive tongue (h_Delta, I) at h_eta={cfg['hC_for_drive']} ...")
    ZD = maxre_grid(mdl, hDd, Iv, "hD", 0.0, cfg["hC_for_drive"])
    # drive for panel (c): the I with the widest oscillation window in h_Delta (per-layer adaptive)
    unst_per_I = (ZD > 0).sum(axis=1)
    I_fix = float(Iv[int(np.argmax(unst_per_I))]) if unst_per_I.max() > 0 else cfg["I_fix"]

    # ── (c) 1-D branch + limit cycle along the protocol (h_eta=0, reduce h_Delta) at fixed I ──
    print(f"[c] 1-D branch + limit cycle at I={I_fix:.0f}, h_eta={cfg['hC_for_drive']} ...")
    hDb = np.linspace(*cfg["hD_branch"][:2], int(cfg["hD_branch"][2]))
    s_eq, stab, mr = branch_in_hD(mdl, hDb, I_fix, hC=cfg["hC_for_drive"])
    smin = np.full(hDb.size, np.nan); smax = np.full(hDb.size, np.nan); fdom = np.full(hDb.size, np.nan)
    unstable_hD = hDb[~stab & np.isfinite(mr)]
    if unstable_hD.size:
        lo, hi = unstable_hD.min(), unstable_hD.max()
        for k in np.where((hDb >= 0.5 * lo) & (hDb <= 1.5 * hi))[0]:
            smin[k], smax[k], fdom[k] = limit_cycle(mdl, hDb[k], I_fix, hC=cfg["hC_for_drive"])
        print(f"    equilibrium unstable for h_Delta in [{lo:.3f},{hi:.3f}];  osc freq ~ {np.nanmedian(fdom):.0f} Hz")
    else:
        print("    no unstable equilibrium along the branch at this I")

    # ============================ figure ============================
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.8))
    C_DATA = "k"; C_PATH = "limegreen"

    # (a) heterogeneity phase plane + protocol path (centres first, then widths). Reduction of
    #     heterogeneity = rightward (h_eta) and downward (h_Delta); data fit sits at top-left (1,1).
    ax = axs[0]; vmax = np.nanmax(np.abs(Zphase))
    pcm = ax.pcolormesh(hC, hD, Zphase, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    if (Zphase > 0).any() and (Zphase < 0).any():
        ax.contour(hC, hD, Zphase, levels=[0.0], colors="k", linewidths=2)
    top = hD.max(); ystop = hD_thr_coll if np.isfinite(hD_thr_coll) else hD.min()
    ax.annotate("", xy=(0.03, top), xytext=(1.0, top),           # (1) reduce centres (top edge)
                arrowprops=dict(arrowstyle="-|>", color=C_PATH, lw=3))
    ax.annotate("", xy=(0.03, ystop), xytext=(0.03, top),        # (2) reduce widths (right edge)
                arrowprops=dict(arrowstyle="-|>", color=C_PATH, lw=3))
    ax.plot(1.0, top, "o", color=C_DATA, ms=9, zorder=6)
    ax.text(0.95, top * 0.93, "data fit\n(1, 1)", color=C_DATA, va="top", ha="right", fontsize=8.5)
    ax.text(0.5, top * 1.04, "① reduce centres", color="darkgreen", ha="center", va="bottom", fontsize=8.5, weight="bold")
    ax.text(0.065, np.sqrt(top * ystop), "② reduce widths", color="darkgreen", ha="left", va="center",
            fontsize=8.5, weight="bold", rotation=90)
    ax.text(hC.min() + 0.5, ystop * 0.5, "oscillations", color="white", ha="center", fontsize=9, weight="bold")
    ax.set_yscale("log"); ax.set_ylim(hD.min(), hD.max())
    ax.set_xlim(hC.max(), hC.min())                        # reverse: heterogeneity decreases rightward
    ax.set_xlabel(r"centre-spread  $h_\eta$  (decreasing $\rightarrow$)")
    ax.set_ylabel(r"width  $h_\Delta$  (decreasing $\downarrow$)")
    ax.set_title(r"(a) heterogeneity plane:  $\max_I\,\mathrm{Re}\,\lambda$")
    fig.colorbar(pcm, ax=ax, label=r"$\max_I\,\mathrm{Re}\,\lambda$", fraction=0.046, pad=0.03)

    # (b) drive tongue at collapsed centres
    pcm = zero_contour_band(axs[1], hDd, Iv, ZD, r"width heterogeneity  $h_\Delta$")
    axs[1].set_title(rf"(b) Hopf tongue in $(h_\Delta, I)$ at $h_\eta={cfg['hC_for_drive']:.0f}$")
    axs[1].axhline(I_fix, color="0.25", ls=":", lw=1.2)
    axs[1].text(hDd.max() * 0.97, I_fix, " panel (c)", color="0.25", va="bottom", ha="right", fontsize=8)
    fig.colorbar(pcm, ax=axs[1], label=r"$\mathrm{Re}\,\lambda$", fraction=0.046, pad=0.03)

    # (c) branch + limit cycle along the protocol (h_eta=0), reducing h_Delta at fixed I
    ax = axs[2]; fin = np.isfinite(s_eq)
    ax.plot(np.where(fin & stab, hDb, np.nan), np.where(fin & stab, s_eq, np.nan), "-", color="C0", lw=2.2, label="stable focus")
    ax.plot(np.where(fin & ~stab, hDb, np.nan), np.where(fin & ~stab, s_eq, np.nan), "--", color="C0", lw=2.2, label="unstable focus")
    ax.fill_between(hDb, smin, smax, color="C3", alpha=0.25, label="limit cycle")
    ax.plot(hDb, smin, color="C3", lw=1.2); ax.plot(hDb, smax, color="C3", lw=1.2)
    if np.isfinite(hD_thr_coll):
        ax.axvline(hD_thr_coll, color="0.5", ls=":", lw=1)
        ax.text(hD_thr_coll, ax.get_ylim()[1], f"  Hopf\n  $h_\\Delta$={hD_thr_coll:.3f}", color="0.4", va="top", fontsize=8)
    ax.set_xlabel(r"width heterogeneity  $h_\Delta$"); ax.set_ylabel(r"synaptic activation  $s$")
    ax.set_title(rf"(c) branch + cycle at $I={I_fix:.0f}$, $h_\eta={cfg['hC_for_drive']:.0f}$")
    axf = ax.twinx(); axf.plot(hDb, fdom, ".", color="0.45", ms=4)
    axf.set_ylabel("osc. frequency (Hz)", color="0.45"); axf.tick_params(axis="y", labelcolor="0.45")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(f"{cfg['cell_class']} {cfg['layer']}  —  heterogeneity control of inhibition-based oscillations "
                 f"($J={cfg['J']:.0f}$, $\\tau_s={cfg['tau_s']:.0f}$ ms);  protocol: reduce centres, then widths",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, f"allen_qif_heterogeneity_{tag}")
    fig.savefig(out + ".png", dpi=160); fig.savefig(out + ".pdf")
    np.savez(out + ".npz", hC=hC, hD=hD, hD_drive=hDd, I=Iv, maxre_phase=Zphase, maxre_drive=ZD,
             hD_thr_full=hD_thr_full, hD_thr_coll=hD_thr_coll, hD_branch=hDb, s_eq=s_eq, stab=stab,
             max_reig_branch=mr, lc_smin=smin, lc_smax=smax, lc_freq=fdom,
             I_fix=I_fix, J=cfg["J"], tau_s=cfg["tau_s"], hC_for_drive=cfg["hC_for_drive"])
    print(f"[saved] {os.path.basename(out)}.png / .pdf / .npz")


if __name__ == "__main__":
    main()
