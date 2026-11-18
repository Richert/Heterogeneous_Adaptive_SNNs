r"""
Core/shell fit targets of the SCN slices (two ensembles per slice, lobes pooled) and a functional check of the split
==================================================================================================================

Assignment (option 1 of the core/shell discussion, 2026-11-17): per lobe, the distance r of each fit-target cell
(scn_targets) from its lobe centroid; inner half (r ≤ lobe median) = CORE (ens = 0), outer half = SHELL (ens = 1). The
two lobes are pooled within each ensemble (they are phase-locked before TTX and after washout).

targets   per ensemble: R_e(t) = |Z_e|, Z_e = ⟨e^{iθ}⟩ over the ensemble's cells (epoch-peak phases of scn_targets),
          the core–shell phase difference Δψ(t) = arg(Z_shell Z̄_core) (rad; > 0: shell ahead), and the measurement
          ceilings c_e(t): R of PERFECTLY synchronised synthetic traces (each cell's measured trend, envelope and noise;
          scn_pipeline_validation "sync") through the same pipeline, with the slice's calibrated noise scale and the
          same random seeds as scn_targets (N_CEIL realisations). Mask: that of scn_targets.
mic       functional check against Abel et al.'s MIC networks (raw_mic_scores, cells of the post-washout set): cells
          matched by position (nearest neighbour < MATCH_PX px), edges = MIC above the authors' per-slice thresholds;
          node degree k vs normalised radius (Spearman), mean k and the fraction of unconnected cells (k = 0) in core
          vs shell.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_coreshell_targets.py targets mic [--slices 1,2,3,4,5]
Output: data_paths.MPMF/circadian_data/coreshell/scn<i>_coreshell_target.npz, mic_check.json
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
import data_paths as dp
import scn_common as S
import scn_targets as TG
import scn_epoch_processing as EP

OUT = dp.ensure(os.path.join(S.OUT, "coreshell"))
MIC_THRESH = {1: 0.949, 2: 0.935, 3: 0.9895, 4: 0.968, 5: 0.969}       # README of the data repository
MATCH_PX = 3.0


def assign(g, locs):
    """(ens, r_norm): ens 0 = core (inner half of its lobe), 1 = shell; radius / lobe RMS radius."""
    xy = locs[g["cells"]].astype(float); lobe = g["lobe"]
    ens = np.zeros(lobe.size, int); rn = np.zeros(lobe.size)
    for L in (0, 1):
        m = lobe == L; r = np.linalg.norm(xy[m] - xy[m].mean(0), axis=1)
        rn[m] = r / np.sqrt(np.mean(r ** 2)); ens[m] = (r > np.median(r)).astype(int)
    return ens, rn


def mean_fields(PH, ens):
    return [TG.mean_field(PH[:, ens == e]) for e in (0, 1)]


def build(i):
    import scn_pipeline_validation as V
    g = TG.load(i); _, locs = S.load(i); ens, rn = assign(g, locs)
    Zc, Zs = mean_fields(g["PH"], ens)
    d = S.process(i); sel = np.zeros(d["N"], bool); sel[g["cells"]] = True
    ttx, wash = int(g["ttx"]), int(g["wash"])
    C = []
    for k in range(TG.N_CEIL):
        trend, A, noise, kick, Theta = V.ground_truth_inputs(d, sel, 1.0, np.random.default_rng(TG.SEED + 100 * i + k))
        xs = trend + A * np.cos(V.true_phases(d, sel, Theta, "sync", kick))
        PH = EP.process(xs + float(g["noise_scale"]) * noise, ttx, wash, **TG.PIPE)["PH"]
        C.append([EP.coherence(PH[:, ens == e]) for e in (0, 1)])
    cc, cs = np.nanmean(np.array(C), axis=0)
    out = dict(slice=i, ens=ens, r_norm=rn, Zc=Zc, Zs=Zs, Rc=np.abs(Zc), Rs=np.abs(Zs),
               dpsi=np.angle(Zs * np.conj(Zc)), cc=cc, cs=cs, mask=g["mask"], n=np.bincount(ens, minlength=2))
    np.savez_compressed(os.path.join(OUT, f"scn{i}_coreshell_target.npz"), **out)
    T = int(g["T"]); v0, v1 = S.valid_window(T)
    m = lambda x, a, b: float(np.nanmean(x[a:b]))
    print(f"SCN{i}: core/shell n = {out['n'].tolist()}; R/c pre {m(out['Rc'], v0, ttx - 24):.2f}/{m(cc, v0, ttx - 24):.2f}"
          f" | {m(out['Rs'], v0, ttx - 24):.2f}/{m(cs, v0, ttx - 24):.2f};  TTX mean c core/shell "
          f"{m(cc, ttx + 24, wash):.2f}/{m(cs, ttx + 24, wash):.2f};  post R/c {m(out['Rc'], wash + 72, v1):.2f}/"
          f"{m(cc, wash + 72, v1):.2f} | {m(out['Rs'], wash + 72, v1):.2f}/{m(cs, wash + 72, v1):.2f}")
    return out


def load(i, rebuild=False):
    f = os.path.join(OUT, f"scn{i}_coreshell_target.npz")
    if rebuild or not os.path.exists(f):
        return build(i)
    z = np.load(f)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def mic_check(slices):
    raw = os.path.join(S.RAW, "ttx_wash_trajectories"); mic_dir = os.path.join(S.RAW, "raw_mic_scores")
    res = {}
    print(f"\n{'slice':6s}{'matched':>9s}{'ρ(k, r)':>9s}{'p':>9s}{'mean k core/shell':>19s}{'k=0 core/shell':>16s}")
    for i in slices:
        g = TG.load(i); _, locs = S.load(i); ens, rn = assign(g, locs)
        wl = np.loadtxt(os.path.join(raw, f"scn{i}_TTXwash_locs.csv"), delimiter=",")
        M = np.loadtxt(os.path.join(mic_dir, f"mic_indiv_scn{i}.csv"), delimiter=",")
        A = (M > MIC_THRESH[i]); np.fill_diagonal(A, False); k_all = A.sum(1)
        dist, idx = cKDTree(wl).query(locs[g["cells"]].astype(float))
        ok = dist < MATCH_PX; k = k_all[idx[ok]]; e = ens[ok]; r = rn[ok]
        rho, p = spearmanr(k, r)
        res[str(i)] = dict(matched=int(ok.sum()), n=int(ok.size), rho_k_r=float(rho), p=float(p),
                           mean_k=[float(k[e == 0].mean()), float(k[e == 1].mean())],
                           frac_k0=[float(np.mean(k[e == 0] == 0)), float(np.mean(k[e == 1] == 0))])
        q = res[str(i)]
        print(f"SCN{i:<3d}{q['matched']:5d}/{q['n']:<4d}{rho:9.2f}{p:9.1g}{q['mean_k'][0]:9.1f} / {q['mean_k'][1]:<7.1f}"
              f"{q['frac_k0'][0]:8.2f} / {q['frac_k0'][1]:.2f}")
    json.dump(res, open(os.path.join(OUT, "mic_check.json"), "w"), indent=1)


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    slices = arg("--slices", [1, 2, 3, 4, 5], lambda s_: [int(v) for v in s_.split(",")])
    if "targets" in sys.argv:
        for i in slices:
            build(i)
    if "mic" in sys.argv:
        mic_check(slices)
