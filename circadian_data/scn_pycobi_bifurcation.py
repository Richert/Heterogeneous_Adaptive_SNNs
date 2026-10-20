r"""
PyCoBi bifurcation analysis of the SCN frequency mixtures (minimal model: one global coupling)
=============================================================================================

Slices SCN1 (bimodal), SCN3 (broad, unimodal + side peak), SCN4 (narrow, unimodal); mixtures "nofloor" (detailed),
"floor" (noise-floor Lorentzian mixture; for SCN4 identical to "single") and "single" (one Lorentzian per lobe) from
scn_model_fit.mixture (λ = 1e-5), all fitted per LOBE; "pfloor" / "psingle" = noise-floor mixture / ONE Lorentzian
fitted to the slice's POOLED frequencies (the minimal model has no lobe structure). Mean field
(components m, weights w_m, centres Ω_m relative to their weighted mean, widths Δ_m):

    ż_m = (iΩ_m − h Δ_m) z_m + ½ (H − H̄ z_m²),   H = K e^{−iα} Σ_n w_n z_n,   R = |Σ_n w_n z_n|

  K  global coupling (main continuation parameter); α  coupling phase lag (0 in 1-D);
  h  WIDTH scale (global heterogeneity of the components; 1 = fitted). Scaling the centre deviations AND widths
     together is a pure time rescaling (bifurcation curves would be K/h = const), so h scales the widths only.

REDUCED COORDINATES (rotation symmetry removed): the component with the largest weight is pinned real (y_0 ≡ 0,
x_0 = r_0) and the frame co-rotates with it (Ω_rot = Im F_0 / x_0):
    ẋ_0 = Re F_0,   ẋ_i = Re F_i + Ω_rot y_i,   ẏ_i = Im F_i − Ω_rot x_i   (i ≥ 1),   dimension 2M − 1.
(Partially / fully synchronised states are fixed points; beating of clusters is a limit cycle.) The incoherent
state (singular in this frame) is treated analytically: K_c = smallest K at which it is linearly unstable.

Per (slice, mixture), ONE Auto session per process (run sequentially, never in parallel):
  1. IVP settle on the synchronised state at K = K_START (T_SETTLE h)
  2. equilibrium continuation in K, bidirectional, K ∈ [K_MIN, K_MAX] (folds LP, Hopf HB; stability from the
     analytical Jacobian, dense NPR)
  3. limit cycles from every HB (IPS = 2, ICP = [K, 11]); R min/max along the stored orbits
  4. 2-D loci of every LP and HB in (K, α) and (K, h) (ISW = 2, bidirectional, positive DS)
  5. analytic K_c(α) and K_c(h) of the incoherent state
Results: data_paths.MPMF/circadian_data/pycobi_bif/scn<i>_<kind>.npy; figure with --figure.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_pycobi_bifurcation.py run [--cases 4:single,4:floor] [--no-loci]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python scn_pycobi_bifurcation.py figure
    (internal: ... one <slice> <kind>  — a single case in this process)
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import subprocess
import time
import numpy as np
import data_paths as dp

OUT = dp.ensure(dp.mpmf("circadian_data", "pycobi_bif"))
AUTO_DIR = "~/PycharmProjects/auto-07p"
CASES = [f"{i}:{k}" for i in (1, 3, 4) for k in ("single", "floor", "nofloor")]
K_MIN, K_MAX = 0.003, 0.5
K_START = K_MAX                              # settle at the top, continue downward
CASE_TIMEOUT = 1800                          # s per (slice, mixture)
FORT9_MAX = 2e9                              # bytes; Auto diagnostics runaway guard
LC_DSMAX, LC_PERIOD_MAX = 0.5, 2e4           # limit-cycle continuation: max step, stop period (h; homoclinic)
T_SETTLE = 4000.0
A_RANGE = (-1.2, 1.2)
H_RANGE = (0.05, 4.0)


# ─── model ───────────────────────────────────────────────────────────────────────────────────────
def _num(v):
    return f"{float(v):.12f}"                # fixed point: avoids Fortran line-wrap of exponents


def pooled_mixture(i, kind):
    """ONE mixture fitted to the slice's pooled frequencies (both lobes; the minimal model has no lobe structure):
    "pfloor" = noise-floor Lorentzian mixture (free M), "psingle" = one Lorentzian (M = 1); λ, width floor as
    scn_model_fit. Cached in OUT."""
    import scn_model_fit as F, scn_targets as TG, scn_lmmf_sim as SIM
    f = os.path.join(OUT, f"scn{i}_{kind}_mixture.npy")
    if os.path.exists(f):
        return np.load(f, allow_pickle=True).item()
    mod = SIM.fit_mixture(TG.load(i)["om"], F.LAM, True, F.WIDTH_K, M_max=1 if kind == "psingle" else 64)
    np.save(f, mod, allow_pickle=True)
    return mod


def mixture(i, kind):
    """(w, Ω − Ω̄, Δ) of the slice's mixture, pinned component (largest weight) first."""
    import scn_model_fit as F, scn_targets as TG
    mod = pooled_mixture(i, kind) if kind in ("pfloor", "psingle") else F.mixture(i, kind, TG.load(i))
    w = np.asarray(mod["w"], float); w = w / w.sum()
    Om = np.asarray(mod["Om"], float); Om = Om - np.sum(w * Om); De = np.asarray(mod["De"], float)
    order = np.argsort(-w)
    return w[order], Om[order], De[order]


def equations(w, Om, De):
    M = w.size; W = [_num(v) for v in w]; O = [_num(v) for v in Om]; D = [_num(v) for v in De]
    xs = [f"x{j}" for j in range(M)]; ys = ["0"] + [f"y{j}" for j in range(1, M)]
    X = "(" + " + ".join(f"{W[j]}*{xs[j]}" for j in range(M)) + ")"
    Y = "(" + " + ".join(f"{W[j]}*{ys[j]}" for j in range(1, M)) + ")" if M > 1 else "0"
    Hr = f"(K*(cos(alpha)*{X} + sin(alpha)*{Y}))"
    Hi = f"(K*(cos(alpha)*{Y} - sin(alpha)*{X}))"

    def F(m):
        x, y = xs[m], ys[m]
        if m == 0:      # y_0 ≡ 0
            re = f"(-h*{D[0]}*x0 + 0.5*({Hr} - x0^2*{Hr}))"
            im = f"({O[0]}*x0 + 0.5*({Hi} + x0^2*{Hi}))"
        else:
            re = (f"(-h*{D[m]}*{x} - ({O[m]})*{y} + 0.5*({Hr} - (({x}^2 - {y}^2)*{Hr} + 2*{x}*{y}*{Hi})))")
            im = (f"(-h*{D[m]}*{y} + ({O[m]})*{x} + 0.5*({Hi} - (2*{x}*{y}*{Hr} - ({x}^2 - {y}^2)*{Hi})))")
        return re, im

    re0, im0 = F(0)
    Orot = f"(({im0})/x0)"
    eqs = [f"d/dt * x0 = {re0}"]
    for m in range(1, M):
        re, im = F(m)
        eqs.append(f"d/dt * x{m} = {re} + {Orot}*y{m}")
        eqs.append(f"d/dt * y{m} = {im} - {Orot}*x{m}")
    return eqs


def state_names(M):
    return ["x0"] + [v for m in range(1, M) for v in (f"x{m}", f"y{m}")]


def circuit(w, Om, De, tag, r0=0.95):
    from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate
    M = w.size
    var = {"x0": f"output({r0})"}
    for m in range(1, M):
        var[f"x{m}"] = f"variable({r0})"; var[f"y{m}"] = "variable(0.0)"
    var.update(K=float(K_START), alpha=0.0, h=1.0)
    op = OperatorTemplate(name=f"scnb_op_{tag}", equations=equations(w, Om, De), variables=var)
    return CircuitTemplate(name=f"scnb_{tag}", nodes={"p": NodeTemplate(name=f"scnb_node_{tag}", operators=[op])})


def K_crit(w, Om, De, alpha=0.0, h=1.0, K_hi=20.0):
    """Smallest K at which the incoherent state z = 0 is linearly unstable."""
    D0 = np.diag(1j * Om - h * De); G = 0.5 * np.exp(-1j * alpha) * np.outer(np.ones(w.size), w)
    growth = lambda K: np.max(np.linalg.eigvals(D0 + K * G).real)
    if growth(0.0) >= 0:
        return 0.0
    if growth(K_hi) < 0:
        return np.nan
    lo, hi = 0.0, K_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if growth(mid) < 0 else (lo, mid)
    return hi


# ─── Auto helpers ────────────────────────────────────────────────────────────────────────────────
def _col(s, name):
    for c in s.columns:
        head = c[0] if isinstance(c, tuple) else c
        if head == name or (isinstance(head, str) and head.endswith("/" + name)):
            return c
    raise KeyError(name)


def branch(ode, cont, w, M):
    """K, R, stability, bifurcation labels of an equilibrium branch (R from the stored states)."""
    s = ode.get_summary(cont)
    K = np.asarray(s[_col(s, "K")], float)
    x = {n: np.asarray(s[_col(s, n)], float) for n in state_names(M)}
    X = sum(w[m] * x[f"x{m}"] for m in range(M)); Y = sum(w[m] * x[f"y{m}"] for m in range(1, M)) if M > 1 else 0
    R = np.sqrt(X ** 2 + Y ** 2)
    stab = np.asarray(s[("stability", "")]).astype(bool) if ("stability", "") in s.columns else np.ones(K.size, bool)
    return dict(K=K, R=R, stable=stab, bif=np.asarray(s[("bifurcation", "")]).astype(str))


def lc_branch(ode, cont, w, M):
    """K, period, stability and R min/max along the stored orbits of a limit-cycle branch."""
    s = ode.get_summary(cont)
    K = np.asarray(s[_col(s, "K")], float)
    period = np.asarray(s[_col(s, "PAR(11)")], float) if any("11" in str(c) for c in s.columns) else np.full(K.size, np.nan)
    stab = np.asarray(s[("stability", "")]).astype(bool) if ("stability", "") in s.columns else np.ones(K.size, bool)
    Rmin, Rmax = np.full(K.size, np.nan), np.full(K.size, np.nan)
    labels = np.asarray(s.index)
    for r, lab in enumerate(labels):
        try:
            sol, *_ = ode.get_solution(point=int(lab), cont=cont)
            if hasattr(sol, "b") and isinstance(getattr(sol, "b", None), dict):
                sol = sol.b["solution"]
            arr = {c.split("/")[-1]: np.asarray(sol[c], float).ravel() for c in sol.coordnames}
            X = sum(w[m] * arr[f"x{m}"] for m in range(M))
            Y = sum(w[m] * arr[f"y{m}"] for m in range(1, M)) if M > 1 else 0
            R = np.sqrt(X ** 2 + Y ** 2); Rmin[r], Rmax[r] = R.min(), R.max()
        except Exception:
            pass
    return dict(K=K, period=period, stable=stab, Rmin=Rmin, Rmax=Rmax,
                bif=np.asarray(s[("bifurcation", "")]).astype(str))


def trim_lc(lc, amp_on=5e-3, amp_off=1e-3):
    """Cut a limit-cycle branch where its R amplitude collapses again (it reached another Hopf point; Auto then walks
    the zero-amplitude period multiples). None if the amplitude never exceeds amp_on (degenerate branch)."""
    amp = lc["Rmax"] - lc["Rmin"]
    on = np.flatnonzero(amp > amp_on)
    if on.size == 0:
        return None
    off = np.flatnonzero((np.arange(amp.size) > on[0]) & (amp < amp_off))
    end = off[0] + 1 if off.size else amp.size
    return {k: (v[:end] if isinstance(v, np.ndarray) and v.shape[:1] == amp.shape else v) for k, v in lc.items()}


def locus(ode, origin, point, name, p2, rng):
    ode.run(origin=origin, starting_point=point, name=name, c="eq", ICP=["K", p2], bidirectional=True,
            RL0=K_MIN, RL1=K_MAX, UZSTOP={p2: list(rng)}, IPS=1, ISW=2, ILP=0, ISP=0, NMX=8000, NPR=5,
            DS=1e-3, DSMIN=1e-10, DSMAX=5e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, get_stability=False)
    s = ode.get_summary(name)
    return np.column_stack([np.asarray(s[_col(s, "K")], float), np.asarray(s[_col(s, p2)], float)])


# ─── one case ────────────────────────────────────────────────────────────────────────────────────
def run_case(i, kind, loci_on=True):
    from pycobi import ODESystem
    from pyrates import clear
    os.chdir(dp.local_work("scn_pycobi_bif"))               # Auto executables must not live on the share
    t0 = time.time()
    w, Om, De = mixture(i, kind); M = w.size
    tag = f"s{i}{kind}"
    circ = circuit(w, Om, De, tag)
    ode = ODESystem.from_template(circ, auto_dir=AUTO_DIR, init_cont=False, analytical_jacobian=True,
                                  auto_constants=("ivp", "eq"))
    log = []
    # 1. settle on the synchronised state
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000, EPSL=1e-8, EPSU=1e-8, EPSS=1e-6,
            UZR={14: T_SETTLE}, STOP={"UZ1"})
    # 2. equilibrium branch in K: downward from K_MAX only. The synchronised branch ends on the incoherent state
    #    z = 0 (BP at ~K_c), where the pinned frame is singular (Ω_rot = 0/0) -> stop there, no branch switching
    ode.run(origin="time", starting_point="UZ1", name="eq", c="eq", ICP="K", RL0=K_MIN, RL1=K_MAX * 1.01,
            IPS=1, ILP=1, ISP=2, ISW=1, MXBF=0, STOP=["BP1"], NMX=8000, NPR=5, DS=-1e-3, DSMIN=1e-10,
            DSMAX=2e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, get_stability=True)
    eq = branch(ode, "eq", w, M)
    n_lp, n_hb = int(np.sum(eq["bif"] == "LP")), int(np.sum(eq["bif"] == "HB"))
    log.append(f"eq: {eq['K'].size} pts, LP at K = {np.round(eq['K'][eq['bif'] == 'LP'], 4)}, "
               f"HB at K = {np.round(eq['K'][eq['bif'] == 'HB'], 4)}")
    # 3. limit cycles from the Hopf points. Periods are O(10²–10³ h), so the arclength is dominated by the period:
    #    a large DSMAX is needed to get anywhere in K (5e-3 stalled at NMX); stop at very long periods (homoclinic)
    lcs = []
    for k in range(1, n_hb + 1):
        nm = f"lc{k}"
        try:
            ode.run(origin="eq", starting_point=f"HB{k}", name=nm, c="eq", IPS=2, ISW=-1, ICP=["K", 11],
                    NTST=400, NCOL=4, ILP=1, ISP=2, NMX=8000, NPR=10, DS=1e-3, DSMIN=1e-12, DSMAX=LC_DSMAX,
                    EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, RL0=K_MIN, RL1=K_MAX, UZSTOP={11: LC_PERIOD_MAX},
                    get_stability=True)
            lc = lc_branch(ode, nm, w, M); lcs.append(lc)
            log.append(f"{nm}: {lc['K'].size} pts, K ∈ [{np.nanmin(lc['K']):.4f}, {np.nanmax(lc['K']):.4f}], "
                       f"period ∈ [{np.nanmin(lc['period']):.0f}, {np.nanmax(lc['period']):.0f}] h, "
                       f"bifs {sorted(set(lc['bif']) - {'', 'nan'})}")
        except Exception as e:
            log.append(f"{nm}: FAILED ({type(e).__name__}: {e})")
    # 4. 2-D loci of folds and Hopf points
    loci = {}
    for lab, n in ((("LP", n_lp), ("HB", n_hb)) if loci_on else ()):
        for k in range(1, n + 1):
            for p2, rng in (("alpha", A_RANGE), ("h", H_RANGE)):
                nm = f"{lab}{k}_{p2}"
                try:
                    loci[nm] = locus(ode, "eq", f"{lab}{k}", nm, p2, rng)
                    log.append(f"{nm}: {loci[nm].shape[0]} pts")
                except Exception as e:
                    log.append(f"{nm}: FAILED ({type(e).__name__}: {e})")
    ode.close_session(clear_files=True); clear(circ)
    # 5. analytic incoherence thresholds
    A = np.linspace(*A_RANGE, 121); Hh = np.geomspace(*H_RANGE, 121)
    kc = dict(K_c=K_crit(w, Om, De), alpha=A, Kc_alpha=np.array([K_crit(w, Om, De, alpha=a) for a in A]),
              h=Hh, Kc_h=np.array([K_crit(w, Om, De, h=x) for x in Hh]))
    np.save(os.path.join(OUT, f"scn{i}_{kind}.npy"),
            dict(slice=i, kind=kind, M=M, w=w, Om=Om, De=De, eq=eq, lc=lcs, loci=loci, kc=kc, log=log),
            allow_pickle=True)
    print(f"[auto] SCN{i} {kind} (M = {M}, dim {2 * M - 1}, K_c = {kc['K_c']:.4f}; {time.time() - t0:.0f} s)\n   "
          + "\n   ".join(log), flush=True)


def run_all(cases, loci_on=True):
    for c in cases:
        i, kind = c.split(":")
        wd = dp.local_work("scn_pycobi_bif"); logf = os.path.join(wd, f"case_s{i}{kind}.log")
        fh = open(logf, "w")
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "one", i, kind]
                             + ([] if loci_on else ["--no-loci"]),
                             stdout=fh, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        f9, t0, why = os.path.join(wd, "fort.9"), time.time(), ""
        while p.poll() is None:                      # guard: wall time and Auto's diagnostic file
            time.sleep(2)
            if time.time() - t0 > CASE_TIMEOUT:
                why = "timeout"
            elif os.path.exists(f9) and os.path.getsize(f9) > FORT9_MAX:
                why = "fort.9 runaway"
            if why:
                os.killpg(p.pid, 9); p.wait()
                if os.path.exists(f9):
                    os.remove(f9)
                break
        fh.close(); out = open(logf).read()
        lines = [l for l in out.splitlines() if l.startswith("[auto]") or l.startswith("   ")]
        print("\n".join(lines) if lines else f"[auto] SCN{i} {kind}: FAILED {why}\n{out[-1500:]}", flush=True)


# ─── figure ──────────────────────────────────────────────────────────────────────────────────────
def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prl_style import set_prl_style, panel_label, COL_DOUBLE
    set_prl_style("prl", **{"axes.titleweight": "bold", "axes.titlelocation": "center"})
    sl = (1, 3, 4)
    kinds = (("nofloor", "#2166ac", "no floor"), ("floor", "#c0392b", "noise floor"), ("single", "0.35", "single"))
    nk = len(kinds)
    fig, axs = plt.subplots(len(sl), nk + 2, figsize=(COL_DOUBLE, 1.6 * len(sl) + 0.3), squeeze=False,
                            gridspec_kw=dict(hspace=0.6, wspace=0.4, left=0.06, right=0.99, top=0.95, bottom=0.08))
    from matplotlib.lines import Line2D
    C_LC = "#1a9850"

    def draw_1d(a, g, col, marker_size=3.5):
        eq = g["eq"]
        for st, ls in ((True, "-"), (False, "--")):
            a.plot(eq["K"], np.where(eq["stable"] == st, eq["R"], np.nan), color=col, ls=ls, lw=1.0)
        for b_, mk in (("LP", "o"), ("HB", "s")):
            s_ = eq["bif"] == b_
            a.plot(eq["K"][s_], eq["R"][s_], mk, ms=marker_size, mfc="none", mec="k", mew=0.7)
        for lc in g["lc"]:
            for st, ls in ((True, "-"), (False, ":")):
                for key in ("Rmax", "Rmin"):
                    a.plot(lc["K"], np.where(lc["stable"] == st, lc[key], np.nan), color=C_LC, ls=ls, lw=0.8)
        a.plot([K_MIN, g["kc"]["K_c"]], [0, 0], color=col, lw=1.0)

    def unique_loci(g, p2):
        """LP1/LP2 (HB1/HB2) usually lie on the same 2-par curve: draw each curve once."""
        seen = []
        for nm, cur in g["loci"].items():
            if nm.endswith("_" + p2) and not any(cur.shape == s.shape and np.allclose(cur, s) for _, s in seen):
                seen.append((nm, cur))
        return seen

    for r, i in enumerate(sl):
        for c, (kind, col, lab) in enumerate(kinds):
            a = axs[r, c]; f = os.path.join(OUT, f"scn{i}_{kind}.npy")
            if not os.path.exists(f):
                a.text(0.5, 0.5, "not run", transform=a.transAxes, ha="center"); continue
            g = np.load(f, allow_pickle=True).item()
            draw_1d(a, g, col)
            a.axvline(g["kc"]["K_c"], color="0.5", lw=0.6, ls=":")
            a.set_xscale("log"); a.set_xlim(K_MIN, K_MAX); a.set_ylim(-0.02, 1.02)
            a.set_title(f"SCN{i}, {lab} (M = {g['M']})", fontsize=6.5); a.set_xlabel(r"$K$ (rad/h)")
            if c == 0:
                a.set_ylabel(r"$R$")
            if np.any(np.isin(g["eq"]["bif"], ("LP", "HB"))):         # zoom on the fold / Hopf window
                ins = a.inset_axes([0.07, 0.47, 0.42, 0.48])
                draw_1d(ins, g, col, marker_size=2.5)
                ins.set_xlim(0.026, 0.038); ins.set_ylim(0.22, 0.7)
                ins.tick_params(labelsize=4.5, length=1.5, pad=1)
                ins.set_xticks([0.03, 0.035]); ins.set_yticks([0.3, 0.5])
        for c, (p2, key, rng) in enumerate((("alpha", "Kc_alpha", A_RANGE), ("h", "Kc_h", H_RANGE))):
            a = axs[r, nk + c]
            for kind, col, lab in kinds:
                f = os.path.join(OUT, f"scn{i}_{kind}.npy")
                if not os.path.exists(f):
                    continue
                g = np.load(f, allow_pickle=True).item()
                a.plot(g["kc"][key], g["kc"][p2], color=col, lw=0.8, ls=":")
                for nm, cur in unique_loci(g, p2):
                    a.plot(cur[:, 0], cur[:, 1], color=col, lw=1.0 if nm.startswith("HB") else 0.8,
                           ls="-" if nm.startswith("HB") else "--")
            a.axhline(0 if p2 == "alpha" else 1, color="0.7", lw=0.5, zorder=0)
            a.set_xscale("log"); a.set_xlim(K_MIN, K_MAX); a.set_ylim(*rng)
            if p2 == "h":
                a.set_yscale("log")
            a.set_xlabel(r"$K$ (rad/h)"); a.set_ylabel(r"$\alpha$ (rad)" if p2 == "alpha" else r"width scale $h$")
            a.set_title(f"SCN{i}: ({'K, α' if p2 == 'alpha' else 'K, h'})", fontsize=6.5)
    for r in range(len(sl)):
        panel_label(axs[r, 0], "abc"[r], dx=-22, dy=3)
    hd = ([Line2D([], [], color=col, lw=1.0, label=lab) for _, col, lab in kinds] +
          [Line2D([], [], color="0.3", lw=1.0, label="stable eq."),
           Line2D([], [], color="0.3", lw=1.0, ls="--", label="unstable eq."),
           Line2D([], [], color=C_LC, lw=0.8, label="limit cycle (R min/max)"),
           Line2D([], [], ls="none", marker="o", mfc="none", mec="k", label="fold (LP)"),
           Line2D([], [], ls="none", marker="s", mfc="none", mec="k", label="Hopf (HB)"),
           Line2D([], [], color="0.3", lw=0.8, ls=":", label=r"incoherence threshold $K_c$"),
           Line2D([], [], color="0.3", lw=0.8, ls="--", label="fold locus"),
           Line2D([], [], color="0.3", lw=1.0, label="Hopf locus")])
    fig.legend(handles=hd, loc="lower center", ncol=6, frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, -0.04))
    out = os.path.join(OUT, "scn_pycobi_bifurcation")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.pdf/.png")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "one":
        run_case(int(sys.argv[2]), sys.argv[3], loci_on="--no-loci" not in sys.argv)
    elif mode == "run":
        arg = lambda k, d_, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d_
        run_all(arg("--cases", CASES, lambda s_: s_.split(",")), loci_on="--no-loci" not in sys.argv)
    elif mode == "figure":
        figure()
