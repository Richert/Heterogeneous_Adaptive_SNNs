r"""
Fit nested LMMF model variants to the SCN coherence targets (epoch-peak pipeline, ceiling, mask)
================================================================================================

Model = scn_lmmf_sim (identical build and integrator), for a cell i in lobe L, TTX level u(t):
    θ̇_i = ω_i − δω (1 − u) + K_s(t) Im(e^{−iα_s} Z e^{−iθ_i}) + K_i Im(e^{−iα_i} Z_L e^{−iθ_i})  [+ Cauchy noise γ u]
    phase scrambling z_m → e^{−σ} z_m at TTX application (σ_ttx) / washout (σ_wash);
    TTX-induced quenched disorder γ_q (contained in the measured distribution, absent outside TTX).
TTX level in the tissue u(t) (scn_homeostatic_coupling.tissue_level), SHARED by all shapes: 0 before TTX, fast
rise (τ_on = 0.5 h) to 1, gradual decline (τ_off) to the residual u_res after washout. The noise is γ u(t) in every
shape (an extra Lorentzian width; some noise remains after an incomplete washout). In the step and profile shapes
u(t) enters only through the noise, so u_res and τ_off are free only in their noise versions.
Three shapes of the TTX-sensitive coupling K_s(t):
    step     K_pre | K_ttx | K_wash                                       (K_min = K_ttx, K_max = K_wash)
    profile  K_pre | K_min → K_ttx, delayed S-shaped recovery (logistic centred t_rec h after TTX application,
             width w_rec h) | K_max → K_post (τ_wash):  K_s follows the spiking rate — a sudden drop at TTX
             application, a delayed partial recovery under TTX, a rebound at washout that decays to a (possibly
             lower) post-washout level. K_post is stored as K_wash. Parametrised as K_ttx = f_ttx K_pre,
             K_min = f_min K_ttx (so K_min < K_ttx < K_pre), K_max = K_post + f_max (K_REB_MAX − K_post)
             (K_max > K_post), f ∈ (0, 1), t_rec ∈ (0, T_REC_MAX) h, w_rec ∈ (W_MIN, W_MAX) h and
             τ_wash ∈ (TAU_MIN, TAU_MAX) h (log-uniform); K_post vs K_pre is free. (The exponential recovery
             K_min → K_ttx of 2026-10-16 is dropped.)
    homeo    homeostatic controller (scn_homeostatic_coupling): r = σ(g − θ_0 − Δθ u), τ_g ġ = (r* − r) − λ g,
             K_s = K_pre r / r*, driven by the same tissue TTX level u(t). Parameters K_pre, r* ∈ (0.02, 0.8),
             Δθ ∈ (1, 12), τ_g ∈ (0.2, 24) h (log), λ = f_λ r*/Δθ with f_λ ∈ (0, 1) (recovery under TTX stays
             possible), u_res ∈ (0, 0.8), τ_off ∈ (0.5, 48) h (log), α_s. Not nested with the other shapes.
LMMF from one Lorentzian mixture per lobe (scn_lmmf_sim.build; noise floor or not), started from the measured
state at the first hour with ≥ 90 % defined phases and run continuously to the end of the valid window.

Target (scn_targets): data R of the slice and both lobes vs the model c(t)|Z(t)| (measurement ceiling), over the
mask (valid window without ±MASK_H h around the treatment changes):
    loss = Σ_t mask_t [(c|Z| − R)² + (c_A|Z_A| − R_A)² + (c_B|Z_B| − R_B)²] / (3 Σ mask)
           + Σ_coupled stretches ((ν_model − ν_data) / SIGMA_NU)² / Σ mask
(ν = collective frequency in the pre-TTX and post-washout stretches outside the mask).

Models (MODELS; all parameters not listed fixed at 0; ONE harmonic coupling term unless +Ki):
    base         step: K_pre, K_ttx, K_wash, α_s       prof-base    K_pre, f_ttx, f_min, t_rec, w_rec, K_wash, f_max,
    noise        base + γ, u_res, τ_off                             τ_wash, α_s
    reset        base + σ_ttx                          homeo-base   K_pre, r*, Δθ, τ_g, f_λ, u_res, τ_off, α_s
                                                       <shape>-noise  + γ (and u_res, τ_off where not yet free)
    quenched     base + γ_q  (γ_q < GQ_FRAC × the narrowest component width, so no width is clipped)
    reset+noise  base + σ_ttx + γ
    shift, full, <name>+Ki   (not in the defaults) base + δω | σ_ttx, σ_wash, γ, δω | + lobe-local K_i
NESTED optimisation in STAGES over all (slice, mixture kind, model family) units at once; every start of a stage
is a separate task of one process pool (scales to many cores):
  1 base           --n-base generic starts (the hand-picked STARTS, then uniform random points in the
                   transformed parameter space)
  2 single mech.   reset / noise / quenched: the base optimum with the mechanism near 0 + --n-ext generic starts
  3 reset+noise    the reset, noise and base optima (missing mechanism near 0) + --n-ext generic starts
  4 base polish    base from every extended optimum with its mechanism switched off
  5 refits         every model that ended above a model nested in it, from that model's optimum; the step
                   coupling is nested in the profile (f_min → 1, f_max → 0), so profile models are also checked
                   against (and restarted from) the same-mechanism step optimum (stage 4 also seeds the profile
                   base from the step base)
Each start is a Nelder–Mead run; the best run of each search is polished once (tighter tolerances). Every model
is then at least as good as the models nested in it.
--optimizer basinhopping: each search runs scipy basinhopping from its first --bh-starts starts (the warm starts
come first; --bh-niter hops, --bh-T temperature relative to the start's loss, --bh-step step in the transformed
space, --bh-maxiter-per-dim for the local Nelder–Mead); results in model_fit/nested_bh/, to check the 'nm' optima.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_model_fit.py fit figure [--slices 1,2,3,4]
        [--models base,reset,...,prof-base,...] [--kinds floor] [--procs 8] [--dt 0.5] [--maxiter-per-dim 300]
        [--n-base 16] [--n-ext 8] [--optimizer nm|basinhopping] [--bh-niter 20] [--bh-T 0.02] [--bh-step 0.75]
        [--bh-maxiter-per-dim 150] [--bh-starts 8]
    e.g. the optimiser check:  ... fit figure --optimizer basinhopping --models prof-base,prof-reset+noise
    (time a single loss evaluation: python scn_model_fit.py time --slices 1)
SETUPS A–C (--models A,B,C, the defaults), all homeostatic K_s with r* fixed at R_STAR_FIX and α_s free, in the
IDENTIFIABLE parametrisation "hr-" (block depth ρ_min, plateau fraction f_rec → ρ_ttx, recovery time t_hom; one-to-
one with Δθ, λ, τ_g; earlier runs: "h0-" with α_s = 0 and r* free, "hf-" with Δθ, τ_g, f_λ) + noise γ u(t):
    A  hr-noise+anchor            8 params: K_pre, ρ_min, f_rec, t_hom, u_res, τ_off, α_s, γ; the mixture state is
                                  re-anchored to the measured phases at t_ttx + 24 h (the lobe-specific first-day
                                  transient is taken from the data, not modelled)
    B  hr-noise+transient        11 params: no anchoring; + a lobe-specific transient noise after TTX application,
                                  extra width γ_tr,L e^{−(t − t_ttx)/τ_tr} on the components of lobe L (γ_tr,A, γ_tr,B,
                                  τ_tr ∈ (1, 72) h)
    C  hr-noise+transient+local  12 params: B + a TTX-insensitive lobe-local coupling K_loc (= K_i, no lag)
  Their warm-start ancestors (hr-base, hr-noise) are fitted along. Nesting: hr-noise ⊂ B ⊂ C; A is not nested.
Earlier defaults: base, noise, prof-base, prof-noise, homeo-base, homeo-noise (the phase reset is dropped: its
discontinuous jump in R disagrees with the data).
FREQUENCY-DISTRIBUTION VARIANTS (--kinds; mixture()), all with the noise floor unless nofloor: floor = SCN- and
lobe-specific mixtures; single = one Lorentzian per slice and lobe (M_max = 1); lobelump = one mixture per slice with the lobes pooled, the same for both lobes; lumped = one
mixture of the frequencies pooled over SCN1–4 (and lobes), the same for every slice and lobe; nofloor = floor without
the noise floor.
Results: data_paths.MPMF/circadian_data/model_fit/shapes/{scn<i>_<kind>_<model>.npy, scn_model_fit.{png,pdf},
         scn_model_fit_summary.csv}   (mixtures cached in model_fit/)
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import csv
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize, basinhopping
import data_paths as dp
import scn_common as S
import scn_lmmf_sim as SIM
import scn_targets as TG
import scn_homeostatic_coupling as HC

MIX = dp.ensure(os.path.join(S.OUT, "model_fit"))
OUT = dp.ensure(os.path.join(MIX, "shapes"))               # step / sigmoid profile / homeostatic (old: nested/)
K_MAX, DW_MAX, G_MAX, S_MAX = 0.3, 0.03, 0.05, 3.0
K_REB_MAX = 1.0                # upper bound of the washout rebound K_max (phase model: large K = fast resync only)
TAU_MIN, TAU_MAX = 1.0, 240.0
T_REC_MAX = 150.0              # recovery centre after TTX application (h; TTX epochs are 129–145 h)
W_MIN, W_MAX = 1.0, 48.0       # width of the S-shaped recovery (h)
BOUNDS_HOMEO = dict(r_star=(0.005, 0.8), d_theta=(1.0, 20.0), tau_g=(0.05, 24.0), f_lam=(0.0, 1.0),
                    u_res=(0.0, 0.8), tau_off=(0.1, 48.0),    # τ_g, τ_off, τ_tr log-uniform; widened 2026-10-16
                    g_trA=(0.0, 0.2), g_trB=(0.0, 0.2), tau_tr=(1.0, 72.0),   # after hits of the first bounds
                    rho_min=(1e-10, 0.5), f_rec=(0.0, 1.0), t_hom=(6.0, 1e4))   # complete block, no recovery
LOG_KEYS = ("tau_wash", "w_rec", "tau_g", "tau_off", "tau_tr", "rho_min", "t_hom")
DT_HOMEO = 0.1                 # controller grid (h)
GQ_FRAC = 0.95
SIGMA_NU = 0.005
LAM, WIDTH_K = 1e-6, 10.0
DT_FIT = 0.5
MAXITER_PER_DIM = 300
OPT = dict(optimizer="nm", bh_niter=20, bh_T=0.02, bh_step=0.75, bh_mpd=150, bh_starts=8, seed=1)  # --optimizer
N_BASE, N_EXT = 16, 8          # generic starts per base / extended model (+ the nested warm starts); see --n-base
REFIT_ROUNDS = 3               # stage-5 passes (stop early when no model is above a nested one)
X_SPREAD = 2.5                 # random starts: uniform in the transformed space, |x| < X_SPREAD
STEP = ("K_pre", "K_ttx", "K_wash", "alpha_s")
PROF = ("K_pre", "f_ttx", "f_min", "t_rec", "w_rec", "K_wash", "f_max", "tau_wash", "alpha_s")
HOMEO = ("K_pre", "r_star", "d_theta", "tau_g", "f_lam", "u_res", "tau_off", "alpha_s")
H0 = HOMEO[:-1]                                             # homeostatic with α_s = 0 (run of 2026-10-16: the
                                                            # lag IS needed for the collective frequency)
HF = tuple(k for k in HOMEO if k != "r_star")              # homeostatic with r* fixed at R_STAR_FIX, α_s free
HR = ("K_pre", "rho_min", "f_rec", "t_hom", "u_res", "tau_off", "alpha_s")   # the same controller, IDENTIFIABLE
# parameters (r* = R_STAR_FIX): block depth ρ_min = K_min/K_pre, plateau under sustained TTX
# ρ_ttx = ρ_min + f_rec (1 − ρ_min) = K_ttx/K_pre, recovery time t_hom (h) = τ_g g_ss / r*; mapped one-to-one onto
# Δθ, λ, τ_g in homeo_from_identifiable
R_STAR_FIX = 0.05                                           # (r* alone is not identifiable: it ran to its bound)
SHAPES = {"": STEP, "prof-": PROF, "homeo-": HOMEO, "h0-": H0, "hf-": HF, "hr-": HR}
PREFIXES = ("prof-", "homeo-", "h0-", "hf-", "hr-")
SHAPE_NAME = {"": "step", "prof-": "sigmoid profile", "homeo-": "homeostatic", "h0-": "homeostatic, α_s = 0",
              "hf-": f"homeostatic, r* = {R_STAR_FIX:g}",
              "hr-": f"homeostatic (identifiable), r* = {R_STAR_FIX:g}"}
NOISE = ("gamma", "u_res", "tau_off")                      # noise γ u(t) with the tissue TTX level u(t)
TRANSIENT = ("g_trA", "g_trB", "tau_tr")                 # lobe-specific transient noise after TTX application
MECH = {"base": (), "reset": ("reset_ttx",), "noise": NOISE, "quenched": ("gamma_q",),
        "reset+noise": ("reset_ttx",) + NOISE, "shift": ("dw",), "full": ("reset_ttx", "reset_wash", "dw") + NOISE,
        "noise+anchor": NOISE + ("@anchor",), "noise+transient": NOISE + TRANSIENT,
        "noise+transient+local": NOISE + TRANSIENT + ("K_i",)}
# warm starts (from these optima, missing parameters near 0) and strict nesting (refit if worse than these)
WARM = {"reset+noise": ["reset", "noise", "base"], "noise+anchor": ["noise", "base"],
        "noise+transient": ["noise", "base"], "noise+transient+local": ["noise+transient", "noise"]}
NESTED = {"reset+noise": ["reset", "noise", "base"], "noise+anchor": [],          # anchoring changes the
          "noise+transient": ["noise", "base"],                                   # dynamics: not nested
          "noise+transient+local": ["noise+transient", "noise", "base"]}
ALIASES = {"A": "hr-noise+anchor", "B": "hr-noise+transient", "C": "hr-noise+transient+local"}


def ancestors(m):
    out = []
    for q in WARM.get(m, ["base"] if m != "base" else []):
        out += [q] + ancestors(q)
    return out


def depth(m):
    return 0 if m == "base" else 1 + max(depth(q) for q in WARM.get(m, ["base"]))


def free(keys):
    """Fitted parameters (keys starting with '@' are model flags, e.g. '@anchor')."""
    return [k for k in keys if not k.startswith("@")]
MODELS = {}
for _m, _x in MECH.items():
    for _pre, _keys in SHAPES.items():
        MODELS[_pre + _m] = _keys + tuple(k for k in _x if k not in _keys)
    MODELS[_m + "+Ki"] = STEP + ("K_i",) + _x + (("alpha_i",) if _m == "full" else ())
DEFAULT_MODELS = ("A", "B", "C")                          # 2026-10-16: setups A–C (ALIASES); earlier: base,
                                                            # noise, prof-base, prof-noise, homeo-base, homeo-noise
DEFAULT_SLICES = (1, 2, 3, 4)                              # SCN5 set aside (outlier) for now
DEFAULTS = dict(K_pre=0.0, K_ttx=0.0, K_wash=0.0, K_i=0.0, alpha_s=0.0, alpha_i=0.0, dw=0.0, gamma=0.0,
                gamma_q=0.0, reset_ttx=0.0, reset_wash=0.0, f_ttx=1.0, f_min=1.0, t_rec=72.0, w_rec=12.0, f_max=0.0,
                tau_wash=24.0, r_star=0.2, d_theta=6.0, tau_g=2.4, f_lam=0.3, u_res=0.0, tau_off=HC.TAU_ON,
                g_trA=0.0, g_trB=0.0, tau_tr=12.0, rho_min=1e-3, f_rec=0.3, t_hom=72.0)
SMALL = dict(reset_ttx=1e-3, reset_wash=1e-3, gamma=1e-4, gamma_q=1e-5, dw=0.0, K_i=1e-4, u_res=1e-3, tau_off=1.0,
             g_trA=1e-4, g_trB=1e-4, tau_tr=12.0)
STARTS = [dict(K_pre=0.10, K_ttx=0.006, K_wash=0.08, alpha_s=0.0, f_ttx=0.1, f_min=0.2, t_rec=72.0, w_rec=12.0, f_max=0.3,
               tau_wash=12.0, reset_ttx=0.3, gamma=0.003, gamma_q=0.3),
          dict(K_pre=0.25, K_ttx=0.02, K_wash=0.15, alpha_s=0.2, f_ttx=0.3, f_min=0.1, t_rec=96.0, w_rec=24.0, f_max=0.5,
               tau_wash=24.0, reset_ttx=0.7, gamma=0.01, gamma_q=0.6),
          dict(K_pre=0.06, K_ttx=0.002, K_wash=0.05, alpha_s=-0.2, f_ttx=0.05, f_min=0.5, t_rec=36.0, w_rec=6.0, f_max=0.1,
               tau_wash=48.0, reset_ttx=0.1, gamma=0.001, gamma_q=0.1),
          dict(K_pre=0.28, K_ttx=0.05, K_wash=0.2, alpha_s=0.0, f_ttx=0.5, f_min=0.05, t_rec=120.0, w_rec=36.0, f_max=0.7,
               tau_wash=6.0, reset_ttx=0.5, gamma=0.02, gamma_q=0.8)]
# τ_g of the starts chosen so that t_rec ≈ τ_g Δθ / r* ≈ 2–6 days for r* = R_STAR_FIX
_HSTARTS = [dict(r_star=0.2, d_theta=6.0, tau_g=0.6, f_lam=0.3, u_res=0.1, tau_off=6.0, g_trA=0.03, g_trB=0.01,
                 tau_tr=12.0, rho_min=1e-3, f_rec=0.3, t_hom=72.0),
            dict(r_star=0.1, d_theta=4.0, tau_g=0.6, f_lam=0.1, u_res=0.3, tau_off=12.0, g_trA=0.01, g_trB=0.03,
                 tau_tr=24.0, rho_min=1e-2, f_rec=0.6, t_hom=48.0),
            dict(r_star=0.4, d_theta=8.0, tau_g=0.8, f_lam=0.6, u_res=0.05, tau_off=2.0, g_trA=0.05, g_trB=0.05,
                 tau_tr=6.0, rho_min=1e-4, f_rec=0.1, t_hom=120.0),
            dict(r_star=0.03, d_theta=12.0, tau_g=0.2, f_lam=0.8, u_res=0.5, tau_off=0.5, g_trA=0.002, g_trB=0.002,
                 tau_tr=36.0, rho_min=0.05, f_rec=0.8, t_hom=24.0)]
for _s, _h in zip(STARTS, _HSTARTS):                       # common defaults of the remaining parameters
    _s.update(_h)
    for _k, _v in dict(K_i=0.01, alpha_i=0.0, dw=0.0, reset_wash=0.1).items():
        _s.setdefault(_k, _v)
KINDS = ("floor", "nofloor", "lobelump", "lumped", "single")           # frequency-distribution variants, see mixture()
LUMP_SLICES = (1, 2, 3, 4)
C_MODEL = {"base": "0.35", "reset": "#b2182b", "noise": "#2166ac", "shift": "#5aae61", "quenched": "#e08214",
           "reset+noise": "#762a83", "full": "#01665e", "noise+anchor": "#d6604d", "noise+transient": "#4393c3",
           "noise+transient+local": "#1b7837"}
_GQ_MAX = [G_MAX]                                          # set per job from the mixture (γ_q < narrowest width)


def prefix_of(name):
    return next((p for p in PREFIXES if name.startswith(p)), "")


def mech_of(name):
    return name[len(prefix_of(name)):].replace("+Ki", "")


def family_of(name):
    return prefix_of(name) + ("|+Ki" if name.endswith("+Ki") else "")


# ─── parameter transforms ────────────────────────────────────────────────────────────────────────
def _bounds(k):
    if k in BOUNDS_HOMEO:
        lo, hi = BOUNDS_HOMEO[k]
        return (np.log(lo), np.log(hi)) if k in LOG_KEYS else (lo, hi)
    if k.startswith("K_"):
        return 0.0, K_MAX
    if k.startswith("alpha"):
        return -0.5 * np.pi, 0.5 * np.pi
    if k == "dw":
        return -DW_MAX, DW_MAX
    if k == "gamma":
        return 0.0, G_MAX
    if k == "gamma_q":
        return 0.0, _GQ_MAX[0]
    if k.startswith("f_"):
        return 0.0, 1.0
    if k.startswith("tau"):
        return np.log(TAU_MIN), np.log(TAU_MAX)
    if k == "w_rec":
        return np.log(W_MIN), np.log(W_MAX)
    if k == "t_rec":
        return 0.0, T_REC_MAX
    return 0.0, S_MAX


def to_x(p, keys):
    x = []
    for k in free(keys):
        lo, hi = _bounds(k); v = np.log(p[k]) if k in LOG_KEYS else p[k]
        q = np.clip((v - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        x.append(np.log(q / (1 - q)))
    return np.array(x)


def from_x(x, keys):
    """Raw parameters (the fitted ones) and the derived couplings K_min, K_ttx (profile), K_max."""
    p = dict(DEFAULTS)
    p["anchor"] = "@anchor" in keys
    for k, v in zip(free(keys), x):
        lo, hi = _bounds(k); q = lo + (hi - lo) / (1 + np.exp(-v))
        p[k] = np.exp(q) if k in LOG_KEYS else q
    p["lam"] = 0.0
    if "rho_min" in keys:                                   # identifiable homeostatic parameters
        p["r_star"] = R_STAR_FIX
        p["d_theta"], p["lam"], p["tau_g"], p["rho_ttx"] = homeo_from_identifiable(
            p["rho_min"], p["f_rec"], p["t_hom"], R_STAR_FIX)
        p["f_lam"] = p["lam"] * p["d_theta"] / p["r_star"]
    if "d_theta" in keys or "rho_min" in keys:              # homeostatic shape: K_s from the controller
        p["shape"] = "homeo"
        if "r_star" not in keys:
            p["r_star"] = R_STAR_FIX
    elif "f_ttx" in keys:
        p["K_ttx"] = p["f_ttx"] * p["K_pre"]
        p["K_min"] = p["f_min"] * p["K_ttx"]
        p["K_max"] = p["K_wash"] + p["f_max"] * (K_REB_MAX - p["K_wash"])
    if "d_theta" in keys:
        p["lam"] = p["f_lam"] * p["r_star"] / p["d_theta"]
    if p.get("shape") != "homeo" and "f_ttx" not in keys:
        p["K_min"], p["K_max"] = p["K_ttx"], p["K_wash"]
    return p


def _logit(x):
    return np.log(x / (1 - x))


def homeo_from_identifiable(rho_min, f_rec, t_hom, r_star):
    """(Δθ, λ, τ_g, ρ_ttx) of the controller from block depth ρ_min = K_min/K_pre, plateau fraction
    ρ_ttx = ρ_min + f_rec (1 − ρ_min) = K_ttx/K_pre (steady state under sustained TTX) and recovery time
    t_hom = τ_g g_ss / r* (time to climb to the steady-state excitability at the initial rate r*/τ_g):
        Δθ = logit r* − logit(ρ_min r*),   g_ss = logit(ρ_ttx r*) − logit(ρ_min r*),
        λ = r* (1 − ρ_ttx) / g_ss,         τ_g = t_hom r* / g_ss."""
    rho_ttx = rho_min + f_rec * (1 - rho_min)
    d_theta = _logit(r_star) - _logit(rho_min * r_star)
    g_ss = max(_logit(min(rho_ttx, 1 - 1e-9) * r_star) - _logit(rho_min * r_star), 1e-9)
    return d_theta, r_star * (1 - rho_ttx) / g_ss, t_hom * r_star / g_ss, rho_ttx


def identifiable(p, d=None):
    """Identifiable summary of any homeostatic fit p: block depth ρ_min, plateau ρ_ttx under sustained TTX,
    recovery time t_hom (h); with the target d also the rebound peak K_max / K_pre and its 1/e decay time (h)."""
    from scipy.optimize import brentq
    r, dth, lam, tg = p["r_star"], p["d_theta"], p["lam"], p["tau_g"]
    th = -_logit(r) + dth                                   # threshold under full block
    f = lambda g: (r - 1 / (1 + np.exp(-(g - th)))) - lam * g
    g_hi = min(r / lam, th + 60.0) if lam > 0 else th + 60.0   # σ saturates far above the threshold
    g_ss = brentq(f, 0.0, g_hi) if f(g_hi) < 0 else g_hi        # (λ ≈ 0: full recovery, ρ_ttx → 1)
    out = dict(rho_min=1 / (1 + np.exp(-(_logit(r) - dth))) / r,
               rho_ttx=1 / (1 + np.exp(-(g_ss - th))) / r, t_hom=tg * g_ss / r)
    if d is not None:
        tc, Kc, *_ = HC.homeostatic_coupling(p["K_pre"], r, dth, tg, lam, p["u_res"], float(d["ttx"]),
                                             float(d["wash"]), float(d["T"]), tau_off=p["tau_off"], dt=0.05)
        m = tc >= float(d["wash"]); k = np.argmax(np.where(m, Kc, -np.inf)); Kpost = Kc[-1]
        thr = Kpost + (Kc[k] - Kpost) / np.e
        dn = np.where((np.arange(tc.size) > k) & (Kc <= thr))[0]
        out.update(K_max_rel=Kc[k] / p["K_pre"], t_rebound=(tc[dn[0]] - tc[k]) if dn.size else np.nan,
                   K_post_rel=Kpost / p["K_pre"], t_peak=tc[k] - float(d["wash"]))
    return out


def start_params(s):
    """A generic start as raw parameters (γ_q given as a fraction of its bound)."""
    p = dict(DEFAULTS); p.update(s); p["gamma_q"] = s["gamma_q"] * _GQ_MAX[0]
    return p


# ─── data, model, loss ───────────────────────────────────────────────────────────────────────────
def _cached(f, make):
    if os.path.exists(f):
        return np.load(f, allow_pickle=True).item()
    m = make(); np.save(f, m, allow_pickle=True)
    return m


def lumped_fit():
    """One noise-floor Lorentzian mixture of the POOLED intrinsic frequencies of LUMP_SLICES (absolute rad/h)."""
    om = np.concatenate([TG.load(j)["om"] for j in LUMP_SLICES])
    return _cached(os.path.join(MIX, "lumped_floor_fit.npy"), lambda: SIM.fit_mixture(om, LAM, True, WIDTH_K))


def mixture(i, kind, d):
    """Mixture model of slice i for the frequency-distribution variant `kind` (all noise-floor fits except nofloor):
      floor / nofloor  SCN- and lobe-specific Lorentzian mixtures (with / without the noise floor)
      single           ONE Lorentzian per slice and lobe (M_max = 1; with the noise floor)
      lobelump         one mixture of the slice's frequencies with the lobes POOLED, the same for both lobes
      lumped           one mixture of the frequencies pooled over the slices LUMP_SLICES (and lobes), the same for
                       every slice and lobe"""
    f = os.path.join(MIX, f"scn{i}_{kind}_mixture.npy")
    if kind in ("floor", "nofloor"):
        return _cached(f, lambda: SIM.build(d, LAM, kind == "floor", WIDTH_K))
    if kind == "lumped":
        return _cached(f, lambda: SIM.build_shared(d, lumped_fit()))
    if kind == "single":
        return _cached(f, lambda: SIM.build(d, LAM, True, WIDTH_K, M_max=1))
    if kind == "lobelump":
        return _cached(f, lambda: SIM.build_shared(d, SIM.fit_mixture(d["om"], LAM, True, WIDTH_K)))
    raise ValueError(kind)


def collective_frequency(Z):
    ph = np.unwrap(np.angle(Z)); t = np.arange(Z.size)
    return np.polyfit(t, ph, 1)[0]


def prepare(d):
    T = int(d["T"]); v0, v1 = S.valid_window(T); ttx, wash, mh = int(d["ttx"]), int(d["wash"]), int(d["mask_h"])
    ok = np.isfinite(d["PH"]); E = np.where(ok, np.exp(1j * np.nan_to_num(d["PH"])), 0)
    Dg = E.sum(1) / np.maximum(ok.sum(1), 1)
    t0 = TG.start_hour(d, v0)                               # first hour with ≥ 90 % defined phases
    stretches = [(max(v0, t0), ttx - mh), (wash + mh, v1)]
    nu = [collective_frequency(Dg[a:b]) for a, b in stretches]
    return dict(t0=t0, t1=v1 - 1, stretches=stretches, nu=nu)


def Kfun(p, d):
    """K_s(t) of the parameters p (step, sigmoid profile or homeostatic controller)."""
    if p.get("shape") == "homeo":
        tc, Kc, *_ = HC.homeostatic_coupling(p["K_pre"], p["r_star"], p["d_theta"], p["tau_g"], p["lam"],
                                             p["u_res"], float(d["ttx"]), float(d["wash"]), float(d["T"]),
                                             tau_off=p["tau_off"], dt=DT_HOMEO)
        return lambda t: np.interp(t, tc, Kc)
    return SIM.coupling_profile(p["K_pre"], p["K_min"], p["K_ttx"], p["t_rec"], p["w_rec"], p["K_max"], p["K_wash"],
                                p["tau_wash"], int(d["ttx"]), int(d["wash"]))


def ufun(p, d):
    """Tissue TTX level u(t) (noise γ u): fast application, gradual washout (τ_off) to u_res."""
    ttx, wash, ur, toff = float(d["ttx"]), float(d["wash"]), p["u_res"], p["tau_off"]
    return lambda t: HC.tissue_level(t, ttx, wash, ur, toff)


def wfun(p, d, model):
    """Lobe-specific transient noise after TTX application: extra width γ_tr,L e^{−(t − t_ttx)/τ_tr} (t ≥ t_ttx)."""
    if not (p["g_trA"] or p["g_trB"]):
        return None
    ttx = float(d["ttx"]); gl = np.where(model["lobe"] == 0, p["g_trA"], p["g_trB"])
    def w(t):
        t = np.asarray(t, float)[:, None]
        return np.where(t >= ttx, np.exp(-np.clip(t - ttx, 0, None) / p["tau_tr"]), 0.0) * gl[None]
    return w


def predict(p, d, model, prep, dt):
    """One continuous run from the measured state; with p['anchor'] the mixture state is RE-ANCHORED to the
    measured phases at t_ttx + TRANSIENT_H (the lobe-specific first-day transient is taken from the data)."""
    ttx, wash = int(d["ttx"]), int(d["wash"])
    uf = ufun(p, d); Kf = Kfun(p, d); wf = wfun(p, d, model)
    t0, t1 = prep["t0"], prep["t1"]
    run = lambda z0, a, b: SIM.simulate(model, z0, Kf, uf, p["K_i"], p["alpha_s"], p["alpha_i"], False, p["dw"],
                                        p["gamma"], a, b, p["gamma_q"],
                                        resets=((ttx, p["reset_ttx"]), (wash, p["reset_wash"])), dt=dt, wfun=wf)
    if not p.get("anchor"):
        return run(SIM.initial_state(model, d["PH"][t0]), t0, t1)
    ta = ttx + S.TRANSIENT_H
    s1 = run(SIM.initial_state(model, d["PH"][t0]), t0, ta)
    s2 = run(SIM.initial_state(model, d["PH"][ta]), ta, t1)
    return tuple(np.concatenate([a[:-1], b]) for a, b in zip(s1, s2))


def loss(x, keys, d, model, prep, dt, detail=False):
    p = from_x(x, keys)
    tm, Zg, ZA, ZB = predict(p, d, model, prep, dt)
    ti = tm.astype(int); w = d["mask"][ti]
    e = ((d["c"][ti] * np.abs(Zg) - d["R"][ti]) ** 2 + (d["cA"][ti] * np.abs(ZA) - d["RA"][ti]) ** 2
         + (d["cB"][ti] * np.abs(ZB) - d["RB"][ti]) ** 2) / 3
    e = np.where(w > 0, e, 0.0); n = w.sum()
    if not np.all(np.isfinite(e)):
        return (1e3, None) if detail else 1e3
    fterm = sum(((collective_frequency(Zg[a - prep["t0"]:b - prep["t0"]]) - nu) / SIGMA_NU) ** 2
                for (a, b), nu in zip(prep["stretches"], prep["nu"]))
    L = (e.sum() + fterm) / n
    if detail:
        rm = lambda M, R, c: np.sqrt(np.sum(np.where(w > 0, (c[ti] * np.abs(M) - R[ti]) ** 2, 0)) / n)
        return L, dict(params=p, tm=tm, Zg=Zg, ZA=ZA, ZB=ZB, rmse=rm(Zg, d["R"], d["c"]),
                       rmse_A=rm(ZA, d["RA"], d["cA"]), rmse_B=rm(ZB, d["RB"], d["cB"]), loss=L)
    return L


# ─── nested optimisation ─────────────────────────────────────────────────────────────────────────
def step_to_profile(p):
    """A step-coupling optimum as a profile start: K_ttx = f_ttx K_pre, no dip (f_min → 1), no rebound (f_max → 0)."""
    q = dict(p)
    q["f_ttx"] = float(np.clip(p["K_ttx"] / max(p["K_pre"], 1e-12), 1e-6, 1 - 1e-6))
    q["f_min"], q["f_max"], q["t_rec"], q["w_rec"], q["tau_wash"] = 1 - 1e-6, 1e-6, 72.0, 12.0, 24.0
    return q


def warm(p, add=(), drop=()):
    q = dict(p)
    for k in add:
        q[k] = SMALL.get(k, 1e-3)
    for k in drop:
        q[k] = DEFAULTS[k]
    return q


# ─── staged, start-parallel nested optimisation ──────────────────────────────────────────────────
_CTX = {}                                                  # per-process cache: (slice, kind, dt) -> context


def context(i, kind, dt):
    key = (i, kind, dt)
    if key not in _CTX:
        d = TG.load(i); model = mixture(i, kind, d)
        _CTX[key] = (d, model, prepare(d), dt, GQ_FRAC * float(model["De"].min()))
    return _CTX[key]


def generic_starts(keys, n, seed):
    """The hand-picked STARTS, then uniform samples in the transformed space (x ∈ (−X_SPREAD, X_SPREAD))."""
    X = [to_x(start_params(s), keys) for s in STARTS][:n]
    rng = np.random.default_rng(seed)
    while len(X) < n:
        X.append(rng.uniform(-X_SPREAD, X_SPREAD, len(free(keys))))
    return X


def _task(args):
    """One local search: (tag, x, loss). mode 'nm' (Nelder–Mead), 'bh' (basinhopping) or 'polish'."""
    tag, i, kind, name, x0, mode, mpd, dt, opt = args
    d, model, prep, dt_, gq = context(i, kind, dt)
    _GQ_MAX[0] = gq; OPT.update(opt)
    keys = MODELS[name]; ctx = (d, model, prep, dt)
    if mode == "bh":
        L0 = loss(x0, keys, *ctx)
        r = basinhopping(loss, x0, niter=OPT["bh_niter"], T=OPT["bh_T"] * L0, stepsize=OPT["bh_step"],
                         minimizer_kwargs=dict(args=(keys,) + ctx, method="Nelder-Mead",
                                               options=dict(maxiter=OPT["bh_mpd"] * len(free(keys)), xatol=1e-3,
                                                            fatol=1e-7)),
                         seed=OPT["seed"] + 7919 * tag[2] + 31 * i)
        return tag, r.x, r.fun
    tol = (1e-4, 1e-8) if mode == "polish" else (1e-3, 1e-7)
    r = minimize(loss, x0, args=(keys,) + ctx, method="Nelder-Mead",
                 options=dict(maxiter=mpd * len(free(keys)), xatol=tol[0], fatol=tol[1]))
    return tag, r.x, r.fun


class Stage:
    """Collects the starts of several (unit, model) searches and runs them in one pool map; the best run of each
    search is then polished once (a second map)."""
    def __init__(self, pool, mpd, dt):
        self.pool, self.mpd, self.dt, self.jobs = pool, mpd, dt, []

    def add(self, unit, name, X):
        i, kind = unit[:2]
        mode = "bh" if OPT["optimizer"] == "basinhopping" else "nm"
        if mode == "bh":                                    # basinhopping from the most promising starts only
            X = X[:OPT["bh_starts"]]
        for k, x in enumerate(X):
            self.jobs.append(((unit, name, k), i, kind, name, np.asarray(x, float), mode, self.mpd, self.dt,
                              dict(OPT)))

    def run(self, label):
        t0 = time.time(); best = {}
        for (unit, name, _), x, f in self.pool.imap_unordered(_task, self.jobs, chunksize=1):
            if (unit, name) not in best or f < best[unit, name][1]:
                best[unit, name] = (x, f)
        pol = [((unit, name, -1), unit[0], unit[1], name, x, "polish", self.mpd, self.dt, dict(OPT))
               for (unit, name), (x, f) in best.items()]
        for (unit, name, _), x, f in self.pool.imap_unordered(_task, pol, chunksize=1):
            if f < best[unit, name][1]:
                best[unit, name] = (x, f)
        print(f"[stage] {label}: {len(self.jobs)} starts + {len(pol)} polishes, {(time.time() - t0) / 60:.1f} min",
              flush=True)
        return best


def run(slices, kinds, models, procs, dt, mpd, n_base, n_ext):
    """Nested fits of all requested models; stages = base | single mechanisms | reset+noise | base re-polish |
    refits of extended models above a nested one. Every start of a stage is a separate pool task."""
    gq = {}
    for i in slices:                                       # targets and mixtures, serially (cached)
        d = TG.load(i)
        for k in kinds:
            gq[i, k] = GQ_FRAC * float(mixture(i, k, d)["De"].min())
    fams = {}
    for m in models:
        fams.setdefault(family_of(m), []).append(m)
    units = []                                             # (slice, kind, prefix, suffix, names)
    for i in slices:
        for k in kinds:
            for f, ms in fams.items():
                pre = prefix_of(f); suf = "+Ki" if f.endswith("+Ki") else ""
                need = list(ms)                            # add the warm-start ancestors of every model
                for n in ms:
                    for q in ancestors(mech_of(n)):
                        if q != "base" and pre + q + suf not in need:
                            need.append(pre + q + suf)
                units.append((i, k, pre, suf, tuple(need)))
    name = lambda u, m: u[2] + m + u[3]
    seed = lambda u, m: (1000 * u[0] + 10 * KINDS.index(u[1]) + sum(map(ord, name(u, m)))) % (2 ** 31)

    def conv(u, p, n):                                     # raw parameter dict -> x for model n of unit u
        _GQ_MAX[0] = gq[u[0], u[1]]
        return to_x(p, MODELS[n])

    def P(u, n):
        _GQ_MAX[0] = gq[u[0], u[1]]
        return from_x(res[u, n][0], MODELS[n])

    extra = lambda n, m: [k for k in MODELS[n] if k not in MODELS[m]]
    res = {}
    t_all = time.time()
    with Pool(procs) as pool:
        st = Stage(pool, mpd, dt)                          # 1: base
        for u in units:
            b = name(u, "base")
            _GQ_MAX[0] = gq[u[0], u[1]]
            st.add(u, b, generic_starts(MODELS[b], n_base, seed(u, "base")))
        res.update(st.run("base"))
        ext = [(u, n) for u in units for n in u[4] if mech_of(n) != "base"]
        for lev in range(1, 1 + max([depth(mech_of(n)) for _, n in ext], default=0)):   # 2…: by nesting depth
            st = Stage(pool, mpd, dt); todo = 0
            for u, n in ext:
                m = mech_of(n)
                if depth(m) != lev:
                    continue
                X = [conv(u, warm(P(u, name(u, q)), add=extra(n, name(u, q))), n)
                     for q in WARM.get(m, ["base"]) if (u, name(u, q)) in res]
                _GQ_MAX[0] = gq[u[0], u[1]]
                st.add(u, n, X + generic_starts(MODELS[n], n_ext, seed(u, m))); todo += 1
            if todo:
                res.update(st.run(f"nesting level {lev} ({todo})"))
        step_unit = {(u[0], u[1], u[3]): u for u in units if not u[2]}   # the step family of each profile unit
        is_prof = lambda u: u[2] == "prof-"                 # step ⊂ sigmoid profile; homeo is not nested
        st = Stage(pool, mpd, dt)                          # 4: base re-polished from the extended optima
        for u in units:                                    #    (profile base also from the step base optimum)
            b = name(u, "base")
            X = [res[u, b][0]] + [conv(u, warm(P(u, n), drop=extra(n, b)), b)
                                  for (uu, n) in res if uu == u and n != b and b in [name(u, q) for q in
                                                                                     NESTED.get(mech_of(n), ["base"])]]
            us = step_unit.get((u[0], u[1], u[3]))
            if is_prof(u) and us is not None and (us, name(us, "base")) in res:
                X.append(conv(u, step_to_profile(P(us, name(us, "base"))), b))
            st.add(u, b, X)
        for k_, v in st.run("base re-polish").items():
            if v[1] < res[k_][1]:
                res[k_] = v
        for rnd in range(REFIT_ROUNDS):                    # 5: repeated until no model is above a nested one
            st = Stage(pool, mpd, dt)                          # 5: refit models above a nested model; nested in a
            todo = 0                                           #    model = its base / sub-mechanisms (same family)
            for (u, n) in list(res):                           #    and, for profile models, the same step model
                if mech_of(n) == "base" and not is_prof(u):
                    continue
                nested = [name(u, q) for q in NESTED.get(mech_of(n), ["base"]) if (u, name(u, q)) in res] \
                    if mech_of(n) != "base" else []
                cand = [(res[u, q][1], conv(u, warm(P(u, q), add=extra(n, q)), n)) for q in nested]
                us = step_unit.get((u[0], u[1], u[3]))
                ns = n.replace("prof-", "")
                if is_prof(u) and us is not None and (us, ns) in res:
                    cand.append((res[us, ns][1], conv(u, step_to_profile(P(us, ns)), n)))
                if cand and res[u, n][1] > min(c[0] for c in cand) * (1 + 1e-6):
                    st.add(u, n, [c[1] for c in sorted(cand, key=lambda c: c[0])] + [res[u, n][0]]); todo += 1
            if not todo:
                break
            for k_, v in st.run(f"refits, round {rnd + 1} ({todo})").items():
                if v[1] < res[k_][1]:
                    res[k_] = v
    print(f"[fit] all stages: {(time.time() - t_all) / 60:.1f} min", flush=True)
    for (u, n), (x, f) in sorted(res.items(), key=lambda kv: (kv[0][0][0], kv[0][0][1], kv[0][1])):
        i, k = u[0], u[1]
        d, model, prep, _, g = context(i, k, dt); _GQ_MAX[0] = g
        L, r = loss(x, MODELS[n], d, model, prep, dt, detail=True)
        r.update(slice=i, kind=k, model_name=n, keys=MODELS[n], x=x, n_free=len(free(MODELS[n])), model=model, dt=dt,
                 gq_max=g, optimizer=dict(OPT), n_base=n_base, n_ext=n_ext)
        np.save(os.path.join(OUT, f"scn{i}_{k}_{n}.npy"), r, allow_pickle=True)
        p = r["params"]
        shown = free(r["keys"]) + (["K_ttx", "K_min", "K_max"] if "f_ttx" in r["keys"] else []) + \
            (["lam"] if "d_theta" in r["keys"] else [])
        print(f"[fit] SCN{i} {k:7s} {n:16s} RMSE {r['rmse']:.3f} (A {r['rmse_A']:.3f}, B {r['rmse_B']:.3f}) "
              f"loss {r['loss']:.6f} | " + ", ".join(f"{q} {p[q]:.4g}" for q in shown), flush=True)


def timing(i, dt, kind="floor"):
    d = TG.load(i); model = mixture(i, kind, d); prep = prepare(d); _GQ_MAX[0] = GQ_FRAC * float(model["De"].min())
    keys = MODELS["homeo-noise"]; x = to_x(start_params(STARTS[0]), keys)
    t0 = time.time()
    for _ in range(5):
        loss(x, keys, d, model, prep, dt)
    el = (time.time() - t0) / 5
    ref = loss(x, keys, d, model, prep, 0.25, detail=True)[1]
    cur = loss(x, keys, d, model, prep, dt, detail=True)[1]
    print(f"SCN{i} {kind}: M = {model['Om'].size}, one loss evaluation at dt = {dt}: {el:.3f} s; "
          f"max |ΔR| vs dt = 0.25: {np.max(np.abs(np.abs(cur['Zg']) - np.abs(ref['Zg']))):.2e}")


# ─── output ──────────────────────────────────────────────────────────────────────────────────────
def collect(slices, kinds, models):
    R = {}
    for i in slices:
        for k in kinds:
            for m in models:
                f = os.path.join(OUT, f"scn{i}_{k}_{m}.npy")
                if os.path.exists(f):
                    R[i, k, m] = np.load(f, allow_pickle=True).item()
    return R


def summary(R, slices, kinds, models):
    print("\nRMSE (slice, masked, ceiling-corrected) per slice  [loss]")
    for k in kinds:
        for pre in SHAPES:
            ms = [m for m in models if prefix_of(m) == pre]
            if not ms:
                continue
            print(f"  ── {k}, {SHAPE_NAME[pre]}:  " + "  ".join(f"{mech_of(m):>18s}" for m in ms))
            for i in slices:
                print(f"  SCN{i}  " + "  ".join(f"{R[i, k, m]['rmse']:7.3f} [{R[i, k, m]['loss']:.5f}]"
                                              if (i, k, m) in R else f"{'—':>18s}" for m in ms))
    rows = []
    for (i, k, m), r in R.items():
        rows.append(dict(slice=i, kind=k, model=m, n_free=r["n_free"], loss=r["loss"], rmse=r["rmse"],
                         rmse_A=r["rmse_A"], rmse_B=r["rmse_B"],
                         **{p: r["params"].get(p, np.nan) for p in list(DEFAULTS) + ["K_min", "K_max", "lam"]}))
    if rows:
        with open(os.path.join(OUT, "scn_model_fit_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def figure(R, slices, kinds, models):
    """Rows: slice R for each coupling shape | K_s(t) of all fits | lobe A, lobe B of the best fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kind = "floor" if "floor" in kinds else kinds[0]
    shapes = [pre for pre in SHAPES if any(prefix_of(m) == pre for m in models)]
    nr = len(shapes) + 3
    fig, axs = plt.subplots(nr, len(slices), figsize=(3.2 * len(slices), 2.2 * nr), layout="constrained",
                            squeeze=False)
    CL = ["#1b7837", "#762a83"]; LS = {"": ":", "prof-": "--", "homeo-": "-", "h0-": "-.", "hf-": "-"}
    for j, i in enumerate(slices):
        d = TG.load(i); T = int(d["T"]); td = np.arange(T) / 24; v0, v1 = S.valid_window(T)
        vw = lambda x: np.where((np.arange(T) >= v0) & (np.arange(T) < v1), x, np.nan)
        fits = {m: R[i, kind, m] for m in models if (i, kind, m) in R}
        best = min(fits, key=lambda m: fits[m]["loss"]) if fits else None
        kr = len(shapes)                                    # K_s row
        for row in range(nr):
            a = axs[row, j]
            a.axvspan(d["ttx"] / 24, d["wash"] / 24, color="0.92", zorder=0)
            if row != kr:
                for t_c in (d["ttx"], d["wash"]):
                    a.axvspan((t_c - d["mask_h"]) / 24, (t_c + d["mask_h"]) / 24, color="none", hatch="////",
                              ec="0.75", lw=0, zorder=0)
                a.set_ylim(0, 1.02)
        for row, pre in enumerate(shapes):
            a = axs[row, j]
            a.plot(td, vw(d["R"]), color="k", lw=1.3, label="data")
            for m, r in fits.items():
                if prefix_of(m) != pre:
                    continue
                ti = r["tm"].astype(int)
                a.plot(r["tm"] / 24, d["c"][ti] * np.abs(r["Zg"]), color=C_MODEL[mech_of(m)], lw=0.9,
                       label=f"{mech_of(m)} ({r['rmse']:.3f})")
            a.set_title(f"SCN{i}: slice R, {SHAPE_NAME[pre]} K_s ({kind})", fontsize=8)
            a.legend(fontsize=5, frameon=False, loc="lower left")
            if j == 0:
                a.set_ylabel(r"$R$ vs $c(t)|Z|$")
        a = axs[kr, j]
        tt = np.arange(T, dtype=float)
        for m, r in fits.items():
            K = Kfun(r["params"], d)
            a.plot(tt / 24, [float(K(x)) for x in tt], color=C_MODEL[mech_of(m)], lw=0.9, ls=LS[prefix_of(m)],
                   label=f"{SHAPE_NAME[prefix_of(m)]}, {mech_of(m)}")
        a.set_yscale("log"); a.set_ylim(1e-4, 1.2)
        a.set_ylabel(r"$K_s(t)$ (rad/h)" if j == 0 else ""); a.set_title("TTX-sensitive coupling", fontsize=8)
        a.legend(fontsize=4.5, frameon=False, loc="lower right")
        for row, (lab, key, ck, zk) in ((kr + 1, ("lobe A", "RA", "cA", "ZA")), (kr + 2, ("lobe B", "RB", "cB", "ZB"))):
            a = axs[row, j]
            a.plot(td, vw(d[key]), color=CL[row - kr - 1], lw=1.2, label=f"data, {lab}")
            if best:
                r = fits[best]; ti = r["tm"].astype(int)
                a.plot(r["tm"] / 24, d[ck][ti] * np.abs(r[zk]), color=C_MODEL[mech_of(best)], lw=0.9, ls="--",
                       label=f"best: {best}")
            a.set_title(f"SCN{i}: {lab}", fontsize=8); a.legend(fontsize=5.5, frameon=False, loc="lower left")
            if j == 0:
                a.set_ylabel(lab)
        axs[-1, j].set_xlabel("time (days)")
    out = os.path.join(OUT, f"scn_model_fit_{kind}")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
    lst = lambda s_: [v for v in s_.split(",")]
    slices = arg("--slices", list(DEFAULT_SLICES), lambda s_: [int(v) for v in s_.split(",")])
    models = [ALIASES.get(m, m) for m in arg("--models", list(DEFAULT_MODELS), lst)]
    kinds = arg("--kinds", ["floor"], lst)
    dt = arg("--dt", DT_FIT, float); mpd = arg("--maxiter-per-dim", MAXITER_PER_DIM, int)
    OPT.update(optimizer=arg("--optimizer", "nm"), bh_niter=arg("--bh-niter", OPT["bh_niter"], int),
               bh_T=arg("--bh-T", OPT["bh_T"], float), bh_step=arg("--bh-step", OPT["bh_step"], float),
               bh_mpd=arg("--bh-maxiter-per-dim", OPT["bh_mpd"], int), bh_starts=arg("--bh-starts", 8, int),
               seed=arg("--seed", OPT["seed"], int))
    if OPT["optimizer"] == "basinhopping":                # separate results, for the comparison with 'nm'
        OUT = dp.ensure(os.path.join(MIX, "nested_bh"))
    if "--out-tag" in sys.argv:                           # e.g. separate floor / no-floor runs on the server
        OUT = dp.ensure(os.path.join(OUT, arg("--out-tag", "")))
    if "time" in sys.argv:
        for i in slices:
            for k in kinds:
                timing(i, dt, k)
    if "fit" in sys.argv:
        run(slices, kinds, models, arg("--procs", os.cpu_count(), int), dt, mpd, arg("--n-base", N_BASE, int),
            arg("--n-ext", N_EXT, int))
    if "figure" in sys.argv or "summary" in sys.argv:
        R = collect(slices, kinds, models)
        summary(R, slices, kinds, models)
        if "figure" in sys.argv:
            figure(R, slices, kinds, models)
