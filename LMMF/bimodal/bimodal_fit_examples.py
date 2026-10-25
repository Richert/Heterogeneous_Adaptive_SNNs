r"""
PyCoBi bifurcation diagrams (α = 0, K-continuation) of the example LMMF fits of bimodal_fit_ensemble.py
======================================================================================================

Per sample size N: sample SEED_EX, two no-floor fits from its trace —
    low   the selection at (M_max, λ) = LOW  (few components)
    high  the selection at (M_max, λ) = HIGH (many components)
Diagram (pinned co-rotating Cartesian frame of circadian_data/scn_pycobi_bifurcation: the component with the largest
weight is real, the frame rotates with it; dimension 2M − 1): single-cluster / partially synchronised states are
EQUILIBRIA, the two-cluster standing wave (beating, R oscillates) is a LIMIT CYCLE. The incoherent state z = 0 is singular
in this frame and drawn analytically (stable below the onset K of bimodal_fit_ensemble.onset).
  1. IVP settle at K = K_START (fully synchronised)
  2. equilibrium continuation in K downward to K_MIN (stop at the branch point on z = 0; no branch switching), LP / HB
  3. limit cycles from every HB (IPS = 2, ICP = [K, 11], large DSMAX; R min / max along the orbits; trimmed where the
     amplitude collapses, scn_pycobi_bifurcation.trim_lc)
One Auto session per process; cases sequential in subprocesses (wall-time / fort.9 guards).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python bimodal_fit_examples.py run
Output: data_paths.MPMF/bimodal_fit_ensemble/example_N<N>_<low|high>.npy
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared"),
                os.path.join(_HERE, "..", "..", "circadian_data")]
import subprocess
import time
import numpy as np
import data_paths as dp
import bimodal_fit_ensemble as BE

SEED_EX = BE.SEED0
LOW, HIGH = (2, 1e-3), (16, 1e-6)
K_START, K_MIN = 8.0, 0.01
CASE_TIMEOUT, FORT9_MAX = 1800, 2e9
K_LC_MAX = 3.3                                       # limit cycles are continued within the ramp range
LC_NTST, LC_NMX, LC_DSMAX = 150, 4000, 0.05
OSC_MIN = 0.05                                       # SD of R in a ramp window above which it oscillates


def lc_seed(N, m, w, Om, De, T_settle=2000.0, dt=0.01):
    """Simulation seed of the standing-wave limit cycle in the pinned co-rotating frame (component 0 = largest weight,
    real), at the ramp K (≤ 3) where this model's R oscillates most. Returns (K, period, orbit (n, 2M−1), names) or
    None if the model never oscillates along the ramp."""
    d, _ = BE.load_run(N, SEED_EX)
    Rw = d["R_lmmf"][m["k"]]                                   # (nK, ns)
    sd = Rw[:, Rw.shape[1] // 2:].std(1)
    if sd.max() < OSC_MIN:
        return None
    Ks = float(d["K"][int(np.argmax(sd))])
    a = 1j * Om - De; z = np.full(w.size, 0.9 + 0j) * np.exp(1j * np.linspace(0, 1, w.size))
    F = lambda z_: a * z_ + 0.5 * (Ks * (z_ @ w) - np.conj(Ks * (z_ @ w)) * z_ * z_)
    def step(z_):
        k1 = F(z_); k2 = F(z_ + 0.5 * dt * k1); k3 = F(z_ + 0.5 * dt * k2); k4 = F(z_ + dt * k3)
        return z_ + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    for _ in range(int(T_settle / dt)):
        z = step(z)
    pin = lambda z_: z_ * np.exp(-1j * np.angle(z_[0]))       # co-rotating, component 0 real
    traj = [pin(z)]
    for _ in range(int(200.0 / dt)):
        z = step(z); traj.append(pin(z))
    traj = np.array(traj); R = np.abs(traj @ w)
    up = np.flatnonzero((R[:-1] < R.mean()) & (R[1:] >= R.mean()))
    if up.size < 3:
        return None
    # fundamental period: smallest multiple of the R-crossing interval that closes the orbit in the pinned frame
    base = int(np.median(np.diff(up))); i0 = up[1]
    best = min(((np.abs(traj[i0 + q * base] - traj[i0]).max(), q) for q in (1, 2, 3)
                if i0 + q * base < traj.shape[0]))
    n_per = best[1] * base
    orbit = traj[i0:i0 + n_per + 1]
    cols = {"x0": orbit[:, 0].real}
    for j in range(1, w.size):
        cols[f"x{j}"] = orbit[:, j].real; cols[f"y{j}"] = orbit[:, j].imag
    return Ks, n_per * dt, cols, None


def par_index(name):
    import re, glob
    for f in glob.glob("c.*"):
        mt = re.search(r"parnames\s*=\s*(\{[^}]*\})", open(f).read())
        if mt:
            inv = {v: k for k, v in eval(mt.group(1)).items()}
            if name in inv:
                return inv[name]
    return 1


def unames():
    import re, glob
    for f in glob.glob("c.*"):
        mt = re.search(r"unames\s*=\s*(\{[^}]*\})", open(f).read())
        if mt:
            d_ = eval(mt.group(1))
            return [d_[k] for k in sorted(d_)]
    raise RuntimeError("no unames in the Auto constants files")


def write_lc_seed(cols, T, _names, path):
    """.dat seed in Auto's state order (read from the constants file PyCoBi wrote)."""
    import pandas as pd
    from pycobi.utility import write_auto_dat
    order = unames()
    n = len(next(iter(cols.values())))
    write_auto_dat(pd.DataFrame({k: cols[k] for k in order}, index=np.linspace(0, T, n)), path, normalize_time=False)
WD = "bimodal_fit_examples"


def example_models(N, seed=SEED_EX):
    d, trace = BE.load_run(N, seed)
    out = {}
    for tag, (Mm, lam) in (("low", LOW), ("high", HIGH)):
        k = BE.select_nofloor(trace, Mm, lam)
        out[tag] = dict(BE.model(d, k), k=k, M_max=Mm, lam=lam, D=trace[k]["D"], rmse=float(d["rmse"][k]),
                        K_onset=float(d["K_onset"][k]), Om_onset=float(d["Om_onset"][k]))
    return out


def out_file(N, tag):
    return os.path.join(BE.OUT, f"example_N{N}_{tag}.npy")


def run_case(N, tag):
    import scn_pycobi_bifurcation as SB
    from pycobi import ODESystem
    from pyrates import clear
    os.chdir(dp.local_work(WD)); t0 = time.time(); log = []
    m = example_models(N)[tag]
    w, Om, De = m["w"], m["Om"], m["De"]
    o = np.argsort(-w); w, Om, De = w[o], Om[o] - np.sum(w * Om), De[o]
    M = w.size
    eq, lcs = None, []
    circ = SB.circuit(w, Om, De, f"bfe{N}{tag}", K0=K_START)
    ode = ODESystem.from_template(circ, auto_dir=SB.AUTO_DIR, init_cont=False, analytical_jacobian=True,
                                  auto_constants=("ivp", "eq", "lc"))
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000, EPSL=1e-8, EPSU=1e-8, EPSS=1e-6,
            UZR={14: 3000.0}, STOP={"UZ1"})
    ode.run(origin="time", starting_point="UZ1", name="eq", c="eq", ICP="K", RL0=K_MIN, RL1=K_START * 1.01,
            IPS=1, ILP=1, ISP=2, ISW=1, MXBF=0, STOP=["BP1"], NMX=8000, NPR=5, DS=-1e-3, DSMIN=1e-10,
            DSMAX=5e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, get_stability=True)
    eq = SB.branch(ode, "eq", w, M)
    n_hb = int(np.sum(eq["bif"] == "HB"))
    log.append(f"eq: {eq['K'].size} pts, LP {np.round(eq['K'][eq['bif'] == 'LP'], 3)}, "
               f"HB {np.round(eq['K'][eq['bif'] == 'HB'], 3)}")
    # limit cycle (standing wave / cluster beating): seeded from a simulation at the K of strongest oscillation
    seed = lc_seed(N, m, w, Om, De)
    if seed is None:
        log.append("lc: no oscillation along the ramp (K ≤ 3) -> no limit cycle")
    else:
        Ks, T, orbit, names = seed
        write_lc_seed(orbit, T, names, "lcseed.dat")
        par = par_index("K")
        parts = []
        for d_, ds in (("a", 1e-3), ("b", -1e-3)):
            nm = "lc" + d_
            try:
                ode.run(name=nm, dat="lcseed", c="lc", NDIM=2 * M - 1, NPAR=36, PAR={par: Ks}, IPS=2, ISW=1,
                        ICP=["K", 11], NTST=LC_NTST, NCOL=4, ILP=1, ISP=2, NMX=LC_NMX, NPR=10, DS=ds, DSMIN=1e-10,
                        DSMAX=LC_DSMAX, EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, RL0=K_MIN, RL1=K_LC_MAX,
                        THL={11: 0.0}, UZSTOP={11: SB.LC_PERIOD_MAX}, get_stability=True)
                lc = SB.lc_branch(ode, nm, w, M)
                parts.append(lc)
            except Exception as e:
                log.append(f"{nm}: FAILED ({type(e).__name__}: {str(e)[:80]})")
        if parts:
            lc = {k: (np.concatenate([parts[0][k][::-1]] + [p_[k] for p_ in parts[1:]])
                      if isinstance(parts[0][k], np.ndarray) else parts[0][k]) for k in parts[0]}
            lcs.append(lc)
            log.append(f"lc (seed K = {Ks:.2f}, T = {T:.2f}): K ∈ [{np.nanmin(lc['K']):.3f}, {np.nanmax(lc['K']):.3f}],"
                       f" R ∈ [{np.nanmin(lc['Rmin']):.2f}, {np.nanmax(lc['Rmax']):.2f}], stable "
                       f"{np.mean(lc['stable']):.0%}")
    ode.close_session(clear_files=True); clear(circ)
    np.save(out_file(N, tag), dict(N=N, tag=tag, model=m, eq=eq, lc=lcs, log=log), allow_pickle=True)
    print(f"[auto] N = {N} {tag} (M = {M}; {time.time() - t0:.0f} s)\n   " + "\n   ".join(log), flush=True)


def run_all():
    wd = dp.local_work(WD)
    for N in BE.N_LIST:
        for tag in ("low", "high"):
            logf = os.path.join(wd, f"case_{N}_{tag}.log"); fh = open(logf, "w")
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "one", str(N), tag], stdout=fh,
                                 stderr=subprocess.STDOUT, text=True, start_new_session=True)
            f9, t0, why = os.path.join(wd, "fort.9"), time.time(), ""
            while p.poll() is None:
                time.sleep(2)
                why = ("timeout" if time.time() - t0 > CASE_TIMEOUT else
                       "fort.9 runaway" if os.path.exists(f9) and os.path.getsize(f9) > FORT9_MAX else "")
                if why:
                    os.killpg(p.pid, 9); p.wait()
                    if os.path.exists(f9):
                        os.remove(f9)
                    break
            fh.close(); out = open(logf).read()
            lines = [l for l in out.splitlines() if l.startswith("[auto]") or l.startswith("   ")]
            print("\n".join(lines) if lines else f"[auto] N = {N} {tag}: FAILED {why}\n{out[-1500:]}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "one":
        run_case(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "run":
        run_all()
