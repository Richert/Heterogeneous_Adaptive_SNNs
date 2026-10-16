r"""
Is there a phase reset at TTX application? Single-cell phase jumps across treatment changes
===========================================================================================

Peak-based phases only (no band-pass filter, so nothing is smeared across the transition): each trace is detrended
(24 h moving average), smoothed (Gaussian σ = SMOOTH_H) and its circadian peaks are located with sub-hour
precision (parabola through the three samples around each maximum; scn_ttx_frequency_drift.cycles).

For an event at time t_e (TTX application, washout, or a PSEUDO event as a control for the method's noise floor:
mid-TTX = same low amplitude / SNR as right after TTX, post-wash = stationary resynchronised state), per cell; peaks before and after t_e are detected SEPARATELY, so the
prominence threshold follows the local amplitude (which collapses under TTX):
  before train   the last N_BEFORE peaks before t_e (peak time vs cycle number, linear fit)
  after train    peaks from t_e + SETTLE_H on, up to N_AFTER of them (linear fit, the new period)
  phase jump     Δφ = 2π [t_before(t_e) − t_after(t_e)] / P_after — the difference of the two trains' phases,
                 both extrapolated to t_e (wrapped to (−π, π]); Δφ = 0 means a pure frequency change without reset.
                 Δφ > 0: the cell's phase ADVANCED.
  phase at t_e   φ_e = 2π (t_e − last peak before t_e) / P_before (0 = at the PER2 peak), for a PRC-like plot.
  amplitude      mean relative cycle amplitude after / before (log ratio).
Population: circular mean and resultant length of Δφ (a common shift does not dephase; a broad Δφ distribution
does), circular SD, and the coherence predicted at t_e + SETTLE_H with and without the measured jumps:
  R_noreset = |⟨exp(i[φ_before(t_e) + ω_after (t − t_e)])⟩|,   R_reset = the same + Δφ_i,
compared with the data (peak-based phase R at the same time).

A real instantaneous reset gives a Δφ spread that does not depend on SETTLE_H; a fast transient (frequency
excursion in the first hours) gives a spread that grows with SETTLE_H until the transient is over.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_ttx_phase_reset.py [--slices 1,2,3,4,5] [--settle 24]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_ttx_phase_reset.py --settle-scan 12,18,24,36
Output: data_paths.MPMF/circadian_data/ttx_reset/{scn_ttx_reset[_settle<h>].png/.pdf, scn<i>_ttx_reset.npz,
        scn_ttx_reset_settle_scan.png/.pdf}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import data_paths as dp
import scn_common as S
import scn_ttx_frequency_drift as TD       # cycles(): sub-hour peaks and cycle amplitudes

OUT = dp.ensure(os.path.join(S.OUT, "ttx_reset"))
N_BEFORE, N_AFTER, SETTLE_H = 3, 4, 24
TAG = ""
PSEUDO_AFTER_WASH_H = 72                    # control: pseudo event this long after the post-washout transient,
                                            # in the stationary resynchronised state (method noise floor for Δφ)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def local_cycles(x, a0, a1):
    """cycles() on the segment [a0, a1) with 24 h margins, so the peak-prominence threshold follows the
    LOCAL amplitude (the amplitude collapses under TTX); peak times returned in absolute hours."""
    m0, m1 = max(a0 - 24, 0), min(a1 + 24, x.size)
    c = TD.cycles(x[m0:m1])
    if c is None:
        return np.array([]), np.array([])
    tp = c["tp"] + m0; keep = (tp >= a0) & (tp < a1)
    return tp[keep], c["amp"][keep]


def event_jumps(X, t_e, t_end, t_begin):
    """Per cell: Δφ, φ_e, periods before/after, amplitude log ratio, before-phase at t_e, after frequency.
    Peaks before t_e are detected in [t_begin, t_e), peaks after in [t_e + SETTLE_H, t_end), separately."""
    out = []
    for c in range(X.shape[1]):
        tb, ab_all = local_cycles(X[:, c], t_begin, t_e)
        ta, aa_all = local_cycles(X[:, c], t_e + SETTLE_H, t_end)
        if tb.size < 2 or ta.size < 2:
            out.append(None); continue
        tp = np.concatenate([tb[-N_BEFORE:], ta[:N_AFTER]]); amp = np.concatenate([ab_all[-N_BEFORE:], aa_all[:N_AFTER]])
        nb = min(N_BEFORE, tb.size); b = np.arange(nb); a = nb + np.arange(min(N_AFTER, ta.size))
        fb = np.polyfit(np.arange(b.size), tp[b], 1); fa = np.polyfit(np.arange(a.size), tp[a], 1)
        Pb, Pa = fb[0], fa[0]
        if not (16 < Pb < 34 and 16 < Pa < 34):
            out.append(None); continue
        # phases of both trains at t_e (2π per cycle, 0 at a peak)
        ph_b = 2 * np.pi * (t_e - np.polyval(fb, b.size - 1)) / Pb
        ph_a = 2 * np.pi * (t_e - np.polyval(fa, 0)) / Pa
        dphi = wrap(ph_a - ph_b)
        ab, aa = np.nanmean(amp[b]), np.nanmean(amp[a])
        out.append(dict(dphi=dphi, phi_e=wrap(ph_b), Pb=Pb, Pa=Pa, ph_b=ph_b, om_a=2 * np.pi / Pa,
                        amp_lr=np.log(aa / ab) if ab > 0 and aa > 0 else np.nan))
    return out


def peak_phase_R(X, t):
    """Data coherence at hour t from peak-interpolated phases (no band-pass)."""
    ph = []
    for c in range(X.shape[1]):
        y = S.detrend_smooth(X[:, c:c + 1])[:, 0]
        p = S.phase(y)
        if np.isfinite(p[t]):
            ph.append(p[t])
    return np.abs(np.mean(np.exp(1j * np.array(ph)))) if ph else np.nan


def circ(x):
    z = np.mean(np.exp(1j * x))
    return np.angle(z), np.abs(z), np.sqrt(-2 * np.log(max(np.abs(z), 1e-12)))


def analyse(i):
    d = S.process(i); X = d["X"][:, d["rhythmic"]["ttx"]]; T = d["T"]
    t_ttx, t_wash = d["ttx"], d["wash"]
    v0, v1 = S.valid_window(T)
    t_ps = int(t_wash + S.TRANSIENT_H + PSEUDO_AFTER_WASH_H)
    t_mid = int(0.5 * (t_ttx + S.TRANSIENT_H + t_wash - SETTLE_H))      # equal before / after spans inside TTX
    events = {"TTX application": (t_ttx, t_wash, v0), "washout": (t_wash, v1, t_ttx + S.TRANSIENT_H),
              "pseudo (mid-TTX)": (t_mid, t_wash, t_ttx + S.TRANSIENT_H),
              "pseudo (post-wash)": (t_ps, v1, t_wash + S.TRANSIENT_H)}
    res = {}
    for name, (te, tend, tbeg) in events.items():
        J = [j for j in event_jumps(X, te, tend, tbeg) if j is not None]
        if not J:
            continue
        dphi = np.array([j["dphi"] for j in J]); phb = np.array([j["ph_b"] for j in J])
        oma = np.array([j["om_a"] for j in J])
        tq = te + SETTLE_H
        R_nr = np.abs(np.mean(np.exp(1j * (phb + oma * (tq - te)))))
        R_r = np.abs(np.mean(np.exp(1j * (phb + dphi + oma * (tq - te)))))
        res[name] = dict(dphi=dphi, phi_e=np.array([j["phi_e"] for j in J]), Pb=np.array([j["Pb"] for j in J]),
                         Pa=np.array([j["Pa"] for j in J]), amp_lr=np.array([j["amp_lr"] for j in J]),
                         R_before=np.abs(np.mean(np.exp(1j * phb))), R_noreset=R_nr, R_reset=R_r,
                         R_data=peak_phase_R(X, int(tq)), t_e=te, n=len(J))
    np.savez_compressed(os.path.join(OUT, f"scn{i}_ttx_reset{TAG}.npz"),
                        **{f"{k.split()[0]}_{f}": v[f] for k, v in res.items() for f in v})
    return res, X.shape[1]


def summary(i, res, n_cells):
    print(f"\nSCN{i} ({n_cells} cells rhythmic under TTX)")
    print(f"  {'event':18s} {'n':>4s} {'mean Δφ (h)':>11s} {'|⟨e^iΔφ⟩|':>10s} {'circ SD (h)':>11s} "
          f"{'R(t_e)':>7s} {'R +1d: no reset / reset / data':>32s} {'amp log-ratio':>13s}")
    for name, r in res.items():
        mu, L, sd = circ(r["dphi"])
        h = 24 / (2 * np.pi)
        print(f"  {name:18s} {r['n']:4d} {mu * h:+11.2f} {L:10.2f} {sd * h:11.2f} {r['R_before']:7.2f} "
              f"{r['R_noreset']:10.2f} / {r['R_reset']:5.2f} / {r['R_data']:5.2f} {np.nanmedian(r['amp_lr']):+13.2f}")


def figure(slices, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"TTX application": "#b2182b", "pseudo (mid-TTX)": "#e08214", "pseudo (post-wash)": "0.55",
         "washout": "#2166ac"}
    h = 24 / (2 * np.pi)
    fig, axs = plt.subplots(3, len(slices), figsize=(3.0 * len(slices), 7.6), layout="constrained", squeeze=False)
    for j, i in enumerate(slices):
        res = results[i]
        a = axs[0, j]; bins = np.linspace(-12, 12, 37)
        for name, r in res.items():
            a.hist(r["dphi"] * h, bins=bins, density=True, histtype="step", lw=1.3, color=C[name], label=name)
        a.set_title(f"SCN{i}: phase jump Δφ", fontsize=8); a.set_xlabel("Δφ (h; > 0 = advance)")
        if j == 0:
            a.legend(fontsize=6, frameon=False)
        a = axs[1, j]
        for name in ("TTX application", "pseudo (mid-TTX)", "pseudo (post-wash)"):
            if name in res:
                r = res[name]
                a.scatter(r["phi_e"] * h, r["dphi"] * h, s=5, alpha=0.5, color=C[name], label=name)
        a.axhline(0, color="0.3", lw=0.6); a.set_xlim(-12, 12); a.set_ylim(-12, 12)
        a.set_xlabel("phase at event (h after PER2 peak)"); a.set_ylabel("Δφ (h)"); a.set_title("PRC-like plot", fontsize=8)
        a = axs[2, j]
        names = list(res); x = np.arange(len(names))
        for k, (lab, key, mk) in enumerate((("before", "R_before", "o"), ("+1 d, no reset", "R_noreset", "s"),
                                            ("+1 d, with reset", "R_reset", "^"), ("+1 d, data", "R_data", "D"))):
            a.plot(x + 0.15 * (k - 1.5), [res[n][key] for n in names], mk, ms=5, label=lab, ls="none")
        a.set_xticks(x); a.set_xticklabels([n.replace("pseudo (", "").rstrip(")").split()[0] for n in names],
                                         fontsize=7); a.set_ylim(0, 1.02)
        a.set_ylabel("coherence R"); a.set_title("coherence decomposition", fontsize=8)
        if j == 0:
            a.legend(fontsize=6, frameon=False)
    out = os.path.join(OUT, "scn_ttx_reset" + TAG)
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"\n[saved] {out}.png/.pdf")


def settle_scan(slices, settles):
    """Circular SD and resultant length of Δφ vs SETTLE_H, per slice and event."""
    global SETTLE_H
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = 24 / (2 * np.pi)
    tab = {}                                   # (slice, event) -> list of (settle, circ SD, L, n)
    for st in settles:
        SETTLE_H = st
        print(f"\n── SETTLE_H = {st} h ──")
        for i in slices:
            res, n = analyse(i); summary(i, res, n)
            for name, r in res.items():
                _, L, sd = circ(r["dphi"])
                tab.setdefault((i, name), []).append((st, sd * h, L, r["n"]))
    C = {"TTX application": "#b2182b", "pseudo (mid-TTX)": "#e08214", "pseudo (post-wash)": "0.55",
         "washout": "#2166ac"}
    fig, axs = plt.subplots(1, len(slices), figsize=(2.6 * len(slices), 2.6), layout="constrained", squeeze=False)
    for j, i in enumerate(slices):
        a = axs[0, j]
        for name, c in C.items():
            if (i, name) in tab:
                v = np.array(tab[(i, name)])
                a.plot(v[:, 0], v[:, 1], "o-", color=c, ms=4, lw=1.2, label=name)
        a.set_title(f"SCN{i}", fontsize=8); a.set_xlabel("settle window (h)"); a.set_ylim(0, 9)
        if j == 0:
            a.set_ylabel("circular SD of Δφ (h)"); a.legend(fontsize=6, frameon=False)
    out = os.path.join(OUT, "scn_ttx_reset_settle_scan")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"\n[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", list(S.SLICES), lambda s_: [int(v) for v in s_.split(",")])
    if "--settle-scan" in sys.argv:
        TAG = "_scan"
        settle_scan(slices, arg("--settle-scan", None, lambda s_: [int(v) for v in s_.split(",")]))
        sys.exit()
    SETTLE_H = arg("--settle", SETTLE_H, int)
    TAG = "" if SETTLE_H == 24 else f"_settle{SETTLE_H}"
    results = {}
    for i in slices:
        results[i], n = analyse(i)
        summary(i, results[i], n)
    figure(slices, results)
