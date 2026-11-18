r"""
Slide figure: example time series of the forced SCN mean field (no / full / partial entrainment)
================================================================================================

Same model, mixture and light signal as scn_entrainment.py (pooled noise-floor mixture, lab frame, α = 0,
I(t) = F·½[1 + tanh(β y_d)/tanh β], β = 4.2), one slice at one coupling (default SCN1 at K_mid).
A coarse candidate grid in (f_d, F) is simulated (vectorised RK4, dt = 0.25 h, N_TR transient days, N_SHOW shown
days) and every mixture component is classified as LOCKED to the driver if its mean frequency over the shown
window differs from ω_f by less than half a cycle over that window. Examples:
    entrained          all components locked     (default point: f_d = unforced collective frequency, F = F_ENT)
    not entrained      no component locked       (default point: F_NONE / f_NONE)
    partial            exactly one component locked; the partial candidate with the most partial neighbours
                       on the grid (the most robust one) is chosen
Shown: light (yellow bands), Re z_m of each cluster (colour by its centre frequency) and |Z| (black).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_entrainment_examples.py [--slice 1] [--k 1] [--out DIR]
Output: data_paths.MPMF/circadian_data/entrainment/scn<i>_entrainment_examples_K<k>.{png,pdf,npz}
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scn_entrainment as E
import scn_arnold_tongue as AT

DT = 0.25
N_TR, N_SHOW = 80, 15                        # days
F_ENT = 0.06
F_NONE, f_NONE = 0.02, 1.30
CAND_F = np.linspace(0.02, 0.14, 13)
CAND_fd = np.concatenate([np.linspace(0.62, 0.88, 14), np.linspace(0.98, 1.38, 21)])
CPD = 24 / (2 * np.pi)
CLUSTER_COLS = ("#f4a582", "#67001f", "#2166ac", "#92c5de")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})


def simulate(w, Om, De, K, fd, F, z0, keep=False):
    """Vectorised over (fd, F) arrays; returns the recorded z (n_rec, n, M) over the shown window if keep, and the
    mean component frequencies over the shown window (n, M)."""
    a = 1j * Om - De
    wf = np.asarray(fd, float) / CPD; F = np.asarray(F, float)
    z = np.broadcast_to(z0, (wf.size, w.size)).astype(complex).copy()
    n_tr, n_sh = int(N_TR * 24 / DT), int(N_SHOW * 24 / DT)
    t = 0.0
    for _ in range(n_tr):
        z = AT.rk4(z, t, DT, K, F, wf, E.BETA, w, a); t += DT
    ph = np.angle(z); dph = np.zeros_like(ph); rec = [z] if keep else None; t_rec = [t]
    for s in range(n_sh):
        z1 = AT.rk4(z, t, DT, K, F, wf, E.BETA, w, a); t += DT
        dph += np.angle(z1 / z); z = z1
        if keep and (s + 1) % 2 == 0:
            rec.append(z); t_rec.append(t)
    nu = dph / (n_sh * DT)
    return (np.array(rec) if keep else None), np.array(t_rec), nu, wf


def classify(nu, wf):
    return np.abs(nu - wf[:, None]) * N_SHOW * 24 < np.pi          # < half a cycle slip over the window


def main(i, kj, out):
    meta = E.load_K(); K = meta[str(i)]["K"][kj]
    w, Om, De = E.mixture_abs(i, "pfloor")
    z0, wc = AT.unforced(w, Om, De, K)
    f_c = (wc if np.isfinite(wc) else np.sum(w * Om)) * CPD
    # candidate scan for partial entrainment
    FF, ff = np.meshgrid(CAND_F, CAND_fd, indexing="ij")
    _, _, nu, wf = simulate(w, Om, De, K, ff.ravel(), FF.ravel(), z0)
    L = classify(nu, wf).reshape(*FF.shape, -1)
    nl = L.sum(-1)
    part = (nl >= 1) & (nl < w.size)
    print(f"[scan] SCN{i} K = {K:.4f}: partial on {part.sum()} / {part.size} candidates, "
          f"full on {(nl == w.size).sum()}, none on {(nl == 0).sum()}")
    if not part.any():
        raise SystemExit("no partial entrainment among the candidates")
    pad = np.pad(part, 1).astype(int)
    nb = sum(pad[1 + di:1 + di + part.shape[0], 1 + dj:1 + dj + part.shape[1]]
             for di in (-1, 0, 1) for dj in (-1, 0, 1))
    score = np.where(part, nb, -1)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    cases = [("(1) not entrained", f_NONE, F_NONE), ("(2) entrained", round(f_c, 3), F_ENT),
             ("(3) partially entrained", float(CAND_fd[c]), float(CAND_F[r]))]
    rec, t_rec, nu, wf = simulate(w, Om, De, K, [x[1] for x in cases], [x[2] for x in cases], z0, keep=True)
    lock = classify(nu, wf)
    np.savez(os.path.join(out, f"scn{i}_entrainment_examples_K{kj}.npz"), K=K, w=w, Om=Om, De=De,
             fd=[x[1] for x in cases], F=[x[2] for x in cases], z=rec, t=t_rec, nu=nu, lock=lock,
             cand_F=CAND_F, cand_fd=CAND_fd, cand_lock=L)

    order = np.argsort(Om)
    fig, ax = plt.subplots(3, 1, figsize=(7.4, 6.8), sharex=True,
                           gridspec_kw=dict(hspace=0.75, left=0.1, right=0.98, top=0.84, bottom=0.08))
    td = (t_rec - t_rec[0]) / 24
    for n, (name, fd, F) in enumerate(cases):
        a_ = ax[n]
        wfn = fd / CPD
        yd = np.sin(wfn * t_rec)
        I = AT.light(F, yd, E.BETA) / max(F, 1e-12)
        a_.fill_between(td, -1.05, 1.05, where=I > 0.5, color="#fde58a", lw=0, zorder=0, step="mid")
        for k, m in enumerate(order):
            a_.plot(td, rec[:, n, m].real, color=CLUSTER_COLS[k % 4], lw=1.2,
                    label=f"Re $z_m$, cluster {Om[m] * CPD:.2f} c/d")
        a_.plot(td, np.abs(rec[:, n] @ w), color="k", lw=1.6, label=r"$|Z|$ (slice coherence)")
        a_.set_ylim(-1.05, 1.05); a_.set_ylabel(r"Re $z_m$, $|Z|$"); a_.set_yticks([-1, 0, 1])
        st = ", ".join(f"{Om[m] * CPD:.2f}: " + ("locked" if lock[n, m] else f"drifts ({nu[n, m] * CPD:.3f} c/d)")
                       for m in order)
        a_.set_title(f"{name}: $f_d$ = {fd:.2f} c/d, $F$ = {F:.3f} rad/h\nclusters {st}", loc="left",
                     fontsize=11)
        if n == 0:
            h, l = a_.get_legend_handles_labels()
            h.append(plt.Rectangle((0, 0), 1, 1, color="#fde58a")); l.append("light on")
            fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.54, 1.0))
    ax[-1].set_xlabel(f"time (days, after {N_TR} transient days)"); ax[-1].set_xlim(0, N_SHOW)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"scn{i}_entrainment_examples_K{kj}.{ext}"), dpi=200)
    plt.close(fig)
    for (name, fd, F), lk, v in zip(cases, lock, nu):
        print(f"[example] {name}: f_d = {fd:.3f}, F = {F:.3f}; locked {lk.tolist()}, "
              f"component freq {np.round(v * CPD, 3).tolist()} c/d")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    out = arg("--out", E.OUT)
    main(arg("--slice", 1, int), arg("--k", 1, int), out)
