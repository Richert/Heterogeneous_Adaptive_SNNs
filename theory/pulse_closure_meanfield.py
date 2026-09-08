r"""
Pulse-driven weight adaptation: moment closure for the weight statistics
========================================================================

Companion to ``pulse_closure_micro.py``.  Writing the rule in canonical form,

    dA_ij/dt = F_ij − Γ_ij A_ij,   F_ij = μ_p G_p A_m,  Γ_ij = μ_p G_p + μ_d G_d ≥ 0,

every pair relaxes LINEARLY to its own target A*_ij = A_m/(1 + r_ij) with
r_ij = μ_d G_d,ij/(μ_p G_p,ij), at its own rate Γ_ij.  [0, A_m] is invariant because
the pulse kernels are non-negative.  The exact hierarchy needs only two covariances,
K_p = Cov(G_p, A) and K_d = Cov(G_d, A), and every equation carries an explicit −Γ̄
damping (Γ̄ = μ_pḠ_p + μ_dḠ_d):

  Ā'   = μ_pḠ_p(A_m−Ā) − μ_dḠ_dĀ − (μ_pK_p + μ_dK_d)
  V_A' = −2Γ̄V_A + 2[μ_pK_p(A_m−Ā) − μ_dK_dĀ] − 2[μ_p⟨δG_p a²⟩ + μ_d⟨δG_d a²⟩]
  K_p' = Cov(Ġ_p,A) − Γ̄K_p + μ_p(A_m−Ā)Var(G_p) − μ_dĀ Cov(G_p,G_d)
                            − μ_p⟨δG_p² a⟩ − μ_d⟨δG_pδG_d a⟩
  K_d' = Cov(Ġ_d,A) − Γ̄K_d + μ_p(A_m−Ā)Cov(G_p,G_d) − μ_dĀ Var(G_d)
                            − μ_p⟨δG_pδG_d a⟩ − μ_d⟨δG_d² a⟩

That damping is what the logistic rule lacked, and it is why simply DROPPING the
three-fluctuation terms (the last line of each equation) is a viable closure here —
contrast ``logistic_closure_meanfield.py``, where the third-order term ⟨Ga³⟩ was the
only thing bounding V_A.

Closed forms of the truncated system (verified symbolically).  With
X_ij ≡ μ_p(A_m−Ā)G_p,ij − μ_dĀG_d,ij, i.e. the drift evaluated at the mean weight:

    K_p* = Cov(G_p,X)/Γ̄,   K_d* = Cov(G_d,X)/Γ̄,   V_A* = Var(X)/Γ̄²
    Ā*   = A_m μ_pĜ_p/(μ_pĜ_p + μ_dĜ_d),   Ĝ_x ≡ Ḡ_x − Cov(G_x,Γ)/Γ̄

V_A* = Var(X)/Γ̄² is the direct analogue of Eq. 37 of the adaptive-coupling-statistics
manuscript (V_A = μ²σ_S²/γ²).  Note Ĝ_p, Ĝ_d and Γ̄ carry no Ā, so Ā* is explicit.

Models compared
---------------
``naive``   K_p = K_d = 0: pure mean-field, Ā only.  Isolates how much the
            weight/drive covariance matters.
``driven``  (Ā, V_A) integrated with the MEASURED K_p(t), K_d(t).  Isolates the
            variance closure, i.e. whether dropping ⟨δG_x a²⟩ is acceptable.
``static``  the full 4-D truncation, closed on the SLOW parts of the kernels.
``qs``      the quasi-static closed forms Ā*, V_A* on the slow drive moments.

On Cov(Ġ_x, A).  In the exact K equations this term dominates, because Ġ_x lives on
the phase timescale while every other term is O(μ).  It is removed — not neglected —
by splitting each kernel into a slow (pair-specific, time-averaged) part ḡ_x,ij and a
fast zero-mean part.  The weight is a low-pass filter with cutoff Γ ~ μ, so it tracks
only ḡ_x, which is frozen on the weight timescale; hence Cov(dḡ_x/dt, A) ≈ 0 and the
K equations become autonomous, driven by the SLOW drive moments Var(ḡ_x), Cov(ḡ_p,ḡ_d).
This is the weight-level analogue of the manuscript's C_S/C_F split.  The micro script
measures the slow kernels as an EMA with timescale ``tau_static``.

Drive moments come either from the MEASURED drive statistics, or — the point of the
exercise — reconstructed from the Daido order parameters z_q via ``pulse_kernels``
(``--daido``), optionally under the OA ansatz z_q = Z^q (``--oa``).

Usage (``pycobi`` env)
----------------------
    python pulse_closure_meanfield.py
    python pulse_closure_meanfield.py partial --oa
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import glob
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pulse_kernels as PK

CONFIG = dict(
    in_dir=dp.KMO_ADAPTIVE,
    out=dp.kmo_adaptive("pulse_closure_scored"),
    models=("naive", "driven", "static"),
    skip_frac=0.05,
)


# ════════════════════════════════════════════════════════════════════════════
#  the hierarchy
# ════════════════════════════════════════════════════════════════════════════
def hierarchy_rhs(y, dm, mu_p, mu_d, Am, third=(0.0, 0.0, 0.0, 0.0, 0.0)):
    """d/dt (Ā, V_A, K_p, K_d).

    ``dm`` = (Ḡ_p, Ḡ_d, Var G_p, Var G_d, Cov(G_p,G_d));
    ``third`` = (⟨δG_p a²⟩, ⟨δG_d a²⟩, ⟨δG_p² a⟩, ⟨δG_pδG_d a⟩, ⟨δG_d² a⟩).
    The Cov(Ġ_x, A) terms are NOT included here (see ``dK_extra``).
    """
    Ab, VA, Kp, Kd = y
    Gp, Gd, Vp, Vd, Cpd = dm
    Wp, Wd, Tpp, Tpd, Tdd = third
    Gam = mu_p * Gp + mu_d * Gd
    dAb = mu_p * Gp * (Am - Ab) - mu_d * Gd * Ab - (mu_p * Kp + mu_d * Kd)
    dVA = (-2 * Gam * VA + 2 * (mu_p * Kp * (Am - Ab) - mu_d * Kd * Ab)
           - 2 * (mu_p * Wp + mu_d * Wd))
    dKp = (-Gam * Kp + mu_p * (Am - Ab) * Vp - mu_d * Ab * Cpd
           - mu_p * Tpp - mu_d * Tpd)
    dKd = (-Gam * Kd + mu_p * (Am - Ab) * Cpd - mu_d * Ab * Vd
           - mu_p * Tpd - mu_d * Tdd)
    return np.array([dAb, dVA, dKp, dKd])


def quasi_static(dm, mu_p, mu_d, Am):
    """Closed-form fixed point of the truncated system: (Ā*, V_A*, K_p*, K_d*)."""
    Gp, Gd, Vp, Vd, Cpd = dm
    Gam = mu_p * Gp + mu_d * Gd
    Ghp = Gp - (mu_p * Vp + mu_d * Cpd) / Gam            # Ḡ_p − Cov(G_p,Γ)/Γ̄
    Ghd = Gd - (mu_p * Cpd + mu_d * Vd) / Gam
    Ab = Am * mu_p * Ghp / (mu_p * Ghp + mu_d * Ghd)
    Kp = (mu_p * (Am - Ab) * Vp - mu_d * Ab * Cpd) / Gam
    Kd = (mu_p * (Am - Ab) * Cpd - mu_d * Ab * Vd) / Gam
    VA = (mu_p ** 2 * (Am - Ab) ** 2 * Vp
          - 2 * mu_p * mu_d * Ab * (Am - Ab) * Cpd
          + mu_d ** 2 * Ab ** 2 * Vd) / Gam ** 2
    return np.array([Ab, VA, Kp, Kd])


# ════════════════════════════════════════════════════════════════════════════
#  scoring
# ════════════════════════════════════════════════════════════════════════════
def drive_series(d, source="measured"):
    """(nrec, 5) array of (Ḡ_p, Ḡ_d, VarG_p, VarG_d, Cov) from the chosen source."""
    if source == "measured":
        return np.column_stack([d[k] for k in ("Gp", "Gd", "VarGp", "VarGd", "CovPD")])
    if source == "slow":            # slow (time-averaged) kernels: what the weight sees
        return np.column_stack([d[k] for k in
                                ("sGp", "sGd", "sVarGp", "sVarGd", "sCovPD")])
    kernel = dict(p=tuple(d["kern_p"]), d=tuple(d["kern_d"]))
    z = d["z"]
    if source == "oa":                                  # z_q -> Z^q
        Z = z[:, 1]
        z = np.stack([Z ** q for q in range(z.shape[1])], axis=1)
    return np.array([PK.drive_moments(kernel, zz) for zz in z])


def score_regime(d, models=CONFIG["models"], source="measured",
                 skip_frac=CONFIG["skip_frac"]):
    t = d["t"]
    Ab, VA, Kp, Kd = (d[k] for k in ("Abar", "VA", "Kp", "Kd"))
    mu_p, mu_d, Am = float(d["mu_p"]), float(d["mu_d"]), float(d["Am"])
    dm = drive_series(d, source)
    dm_slow = drive_series(d, "slow")
    Kp_s, Kd_s = d["sKp"], d["sKd"]
    third = np.column_stack([d[k] for k in ("Wp", "Wd", "Tpp", "Tpd", "Tdd")])

    dm_f = interp1d(t, dm, axis=0, bounds_error=False, fill_value=(dm[0], dm[-1]))
    dms_f = interp1d(t, dm_slow, axis=0, bounds_error=False,
                     fill_value=(dm_slow[0], dm_slow[-1]))
    Kp_f = interp1d(t, Kp, bounds_error=False, fill_value=(Kp[0], Kp[-1]))
    Kd_f = interp1d(t, Kd, bounds_error=False, fill_value=(Kd[0], Kd[-1]))
    th_f = interp1d(t, third, axis=0, bounds_error=False, fill_value=(third[0], third[-1]))
    m = t > t[-1] * skip_frac
    Gam = mu_p * dm[:, 0] + mu_d * dm[:, 1]
    sdG = np.sqrt(np.maximum(mu_p ** 2 * dm[:, 2] + mu_d ** 2 * dm[:, 3]
                             + 2 * mu_p * mu_d * dm[:, 4], 0.0))

    out = {"t": t, "R": d["R"], "Abar": Ab, "VA": VA, "Kp": Kp, "Kd": Kd,
           "Gbar": Gam, "modulation": sdG / Gam, "dm": dm, "dm_slow": dm_slow,
           "third": third, "Kp_slow": Kp_s, "Kd_slow": Kd_s,
           "mu_p": mu_p, "mu_d": mu_d, "Am": Am,
           "r_snap": d["r_snap"], "A_snap": d["A_snap"],
           "bound": Ab * (Am - Ab), "sat": VA / (Ab * (Am - Ab))}

    # --- quasi-static closed forms, evaluated pointwise on the drive moments
    qs = np.array([quasi_static(dm_slow[i], mu_p, mu_d, Am) for i in range(len(t))])
    out["qs"] = qs
    for k, nm in enumerate(("Abar", "VA", "Kp", "Kd")):
        out[f"{nm}_qs"] = qs[:, k]

    # --- integrate each dynamical model
    for name in models:
        def rhs(tt, y, name=name):
            if name == "naive":                 # mean only, no covariance feedback
                z = np.array([y[0], 0.0, 0.0, 0.0])
                return np.array([hierarchy_rhs(z, dm_f(tt), mu_p, mu_d, Am)[0],
                                 0.0, 0.0, 0.0])
            if name == "driven":                # K from micro; tests the V_A closure
                z = np.array([y[0], y[1], float(Kp_f(tt)), float(Kd_f(tt))])
                dy = hierarchy_rhs(z, dm_f(tt), mu_p, mu_d, Am)
                return np.array([dy[0], dy[1], 0.0, 0.0])
            return hierarchy_rhs(y, dms_f(tt), mu_p, mu_d, Am)   # "static": full 4-D

        y0 = np.array([Ab[0], max(VA[0], 0.0),
                       Kp_s[0] if name == "static" else Kp[0],
                       Kd_s[0] if name == "static" else Kd[0]])
        sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method="LSODA",
                        rtol=1e-9, atol=1e-13)
        Y = sol.y if sol.y.shape[1] == len(t) else np.full((4, len(t)), np.nan)
        for k, nm in enumerate(("Abar", "VA", "Kp", "Kd")):
            out[f"{nm}_{name}"] = Y[k]
        rel = lambda h, x: (np.sqrt(np.nanmean((h - x)[m] ** 2))
                            / np.sqrt(np.mean(x[m] ** 2)))
        out[f"errAbar_{name}"] = rel(Y[0], Ab)
        out[f"errVA_{name}"] = rel(Y[1], VA) if name != "naive" else np.nan
    out["errAbar_qs"] = np.sqrt(np.nanmean((qs[:, 0] - Ab)[m] ** 2)) / np.sqrt(np.mean(Ab[m] ** 2))
    out["errVA_qs"] = np.sqrt(np.nanmean((qs[:, 1] - VA)[m] ** 2)) / np.sqrt(np.mean(VA[m] ** 2))
    return out


def daido_check(d):
    """Relative error of the Daido-reconstructed drive moments vs. the measured ones."""
    meas = drive_series(d, "measured")
    res = {}
    for src in ("daido", "oa"):
        rec = drive_series(d, src)
        res[src] = np.abs(rec - meas).mean(0) / np.maximum(np.abs(meas).mean(0), 1e-12)
    return res


# ════════════════════════════════════════════════════════════════════════════
#  entry point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    source = "oa" if "--oa" in sys.argv else ("daido" if "--daido" in sys.argv else "measured")
    stem = "pulse_closure_"
    found = sorted(os.path.basename(f)[len(stem):-len(".npz")]
                   for f in glob.glob(os.path.join(CONFIG["in_dir"], stem + "*.npz"))
                   if "scored" not in f)
    which = args or found
    if not which:
        raise SystemExit(f"no micro output in {CONFIG['in_dir']} — run pulse_closure_micro.py")

    LBL = {"naive": "naive (K=0)", "driven": "V_A closure (K from micro)",
           "static": "slow-kernel 4-D truncation"}
    res, flat = {}, {}
    for name in which:
        f = os.path.join(CONFIG["in_dir"], f"{stem}{name}.npz")
        if not os.path.exists(f):
            print(f"[{name}] missing {f}"); continue
        d = np.load(f, allow_pickle=False)
        r = score_regime(d, source=source)
        res[name] = r
        dchk = daido_check(d)

        print(f"\n=== {name}:  <R>={r['R'].mean():.3f}  Ā={r['Abar'][-1]:.3f}  "
              f"V_A={r['VA'][-1]:.5f}  s={r['sat'][-1]:.3f}  "
              f"std(Γ)/Γ̄={r['modulation'].mean():.2f}   [drive source: {source}]")
        print("    drive-moment reconstruction, mean rel. err "
              "(Ḡp, Ḡd, VarGp, VarGd, Cov):")
        for k, v in dchk.items():
            tag = "z_q measured (exact)" if k == "daido" else "OA ansatz z_q = Z^q"
            print(f"      {tag:<22} " + "  ".join(f"{x:.1e}" for x in v))
        print(f"    {'model':<32} {'err Ā':>9} {'err V_A':>9}")
        for mdl in CONFIG["models"]:
            print(f"    {LBL[mdl]:<32} {r[f'errAbar_{mdl}']:>9.4f} "
                  f"{r[f'errVA_{mdl}']:>9.4f}")
        print(f"    {'quasi-static closed form':<32} {r['errAbar_qs']:>9.4f} "
              f"{r['errVA_qs']:>9.4f}")
        for k, v in dchk.items():
            flat[f"chk_{name}/{k}"] = v
        flat.update({f"{name}/{kk}": vv for kk, vv in r.items()})

    np.savez_compressed(CONFIG["out"] + f"_{source}.npz",
                        regimes=np.array(list(res)), source=source, **flat)
    print(f"\n[saved] {CONFIG['out']}_{source}.npz")
