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
CLOSE_TOL = 0.1                                      # rough closure of the simulated seed orbit (Auto then converges it;
                                                     # windows without a matching STABLE continued cycle = quasi-periodic)
MAX_LC = 4                                           # limit-cycle seeds per example
LC_NPR = 3


def ramp_windows(N, m):
    """The example's own LMMF backward ramp (bimodal_fit_ensemble): K grid and R min / max / SD per window (2nd half)."""
    d, _ = BE.load_run(N, SEED_EX)
    Rw = d["R_lmmf"][m["k"]][:, d["R_lmmf"].shape[-1] // 2:]
    return d["K"], Rw.min(1), Rw.max(1), Rw.std(1)


def lc_seed(m, w, Om, De, K_target, dt=0.02, hold=BE.HOLD, close_tol=CLOSE_TOL):
    """Simulation seed of a limit cycle in the pinned co-rotating frame (component 0 = largest weight, real) at
    K_target, reached like the ramp: from a synchronised start at the top of the K grid, stepping DOWN the grid
    (hold time units per step) to K_target (same attractor as the ramp). The orbit is accepted if it closes after 1-3
    R-oscillation periods (max |Δz| < close_tol); otherwise the motion is quasi-periodic -> returns (None, err).
    Returns ((period, columns), err)."""
    a = 1j * Om - De; z = np.full(w.size, 0.9 + 0j) * np.exp(1j * np.linspace(0, 1, w.size))
    def step(z_, K):
        F = lambda u: a * u + 0.5 * (K * (u @ w) - np.conj(K * (u @ w)) * u * u)
        k1 = F(z_); k2 = F(z_ + 0.5 * dt * k1); k3 = F(z_ + 0.5 * dt * k2); k4 = F(z_ + dt * k3)
        return z_ + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    for K in BE.K_GRID[::-1]:
        if K < K_target - 1e-9:
            break
        for _ in range(int(hold / dt)):
            z = step(z, K)
    for _ in range(int(1000.0 / dt)):                         # extra settling at K_target
        z = step(z, K_target)
    pin = lambda z_: z_ * np.exp(-1j * np.angle(z_[0]))
    traj = [pin(z)]
    for _ in range(int(200.0 / dt)):
        z = step(z, K_target); traj.append(pin(z))
    traj = np.array(traj); R = np.abs(traj @ w)
    up = np.flatnonzero((R[:-1] < R.mean()) & (R[1:] >= R.mean()))
    if up.size < 4:
        return None, np.inf
    base = int(np.median(np.diff(up))); i0 = up[1]
    err, q = min(((np.abs(traj[i0 + q * base] - traj[i0]).max(), q) for q in (1, 2, 3)
                  if i0 + q * base < traj.shape[0]))
    if err > close_tol:
        return None, err
    orbit = traj[i0:i0 + q * base + 1]
    cols = {"x0": orbit[:, 0].real}
    for j in range(1, w.size):
        cols[f"x{j}"] = orbit[:, j].real; cols[f"y{j}"] = orbit[:, j].imag
    return (q * base * dt, cols), err


def covered(lcs, K, Rmax, tol=0.05):
    """True if a STABLE continued limit cycle passes through K with R max within tol of Rmax (linear interpolation
    between consecutive stored points that are both stable; the stored points are sparse in K)."""
    for lc in lcs:
        Kb, st, Rb = lc["K"], lc["stable"].astype(bool), lc["Rmax"]
        for j in range(Kb.size - 1):
            if st[j] and st[j + 1] and Kb[j] != Kb[j + 1] and min(Kb[j], Kb[j + 1]) <= K <= max(Kb[j], Kb[j + 1]):
                if abs(Rb[j] + (Rb[j + 1] - Rb[j]) * (K - Kb[j]) / (Kb[j + 1] - Kb[j]) - Rmax) < tol:
                    return True
    return False


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
    # limit cycles (standing waves / cluster beating), seeded from simulations of the ramp: first at the window of
    # strongest oscillation, then in every oscillating window not yet covered by a stable continued cycle; windows whose
    # orbit does not close are quasi-periodic (drawn from the simulation in the figure)
    Kg, Rlo, Rhi, Rsd = ramp_windows(N, m)
    osc = np.flatnonzero(Rsd > OSC_MIN)
    par = par_index("K"); tried, qp = [], []
    order = list(osc[np.argsort(-Rsd[osc])])
    while order and len(tried) < MAX_LC:
        i = order.pop(0)
        if covered(lcs, Kg[i], Rhi[i]) or any(abs(Kg[i] - t_) < 0.15 for t_ in tried):
            continue
        tried.append(Kg[i])
        seed, err = lc_seed(m, w, Om, De, float(Kg[i]))
        if seed is None:
            qp.append(float(Kg[i])); log.append(f"seed K = {Kg[i]:.2f}: orbit does not close (err {err:.1e}) -> quasi-periodic")
            continue
        T, orbit = seed
        q = len(lcs)
        write_lc_seed(orbit, T, None, f"lcseed{q}.dat")
        parts = []
        for d_, ds in (("a", 1e-3), ("b", -1e-3)):
            nm = f"lc{q}{d_}"
            try:
                ode.run(name=nm, dat=f"lcseed{q}", c="lc", NDIM=2 * M - 1, NPAR=36, PAR={par: float(Kg[i])}, IPS=2,
                        ISW=1, ICP=["K", 11], NTST=LC_NTST, NCOL=4, ILP=1, ISP=2, NMX=LC_NMX, NPR=LC_NPR, DS=ds,
                        DSMIN=1e-10, DSMAX=LC_DSMAX, EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, RL0=K_MIN, RL1=K_LC_MAX,
                        THL={11: 0.0}, UZSTOP={11: SB.LC_PERIOD_MAX}, get_stability=True)
                parts.append(SB.lc_branch(ode, nm, w, M))
            except Exception as e:
                log.append(f"{nm}: FAILED ({type(e).__name__}: {str(e)[:80]})")
        if parts:
            lc = {k: (np.concatenate([parts[0][k][::-1]] + [p_[k] for p_ in parts[1:]])
                      if isinstance(parts[0][k], np.ndarray) else parts[0][k]) for k in parts[0]}
            lcs.append(lc)
            log.append(f"lc{q} (seed K = {Kg[i]:.2f}, T = {T:.2f}): K ∈ [{np.nanmin(lc['K']):.3f}, "
                       f"{np.nanmax(lc['K']):.3f}], R ∈ [{np.nanmin(lc['Rmin']):.2f}, {np.nanmax(lc['Rmax']):.2f}], "
                       f"stable {np.mean(lc['stable']):.0%}")
    uncov = [float(Kg[i]) for i in osc if not covered(lcs, Kg[i], Rhi[i])]
    log.append(f"oscillating windows without a stable continued cycle: {np.round(uncov, 2).tolist()}")
    ode.close_session(clear_files=True); clear(circ)
    np.save(out_file(N, tag), dict(N=N, tag=tag, model=m, eq=eq, lc=lcs, log=log, quasi_periodic=qp,
                                   uncovered=uncov, ramp=dict(K=Kg, Rmin=Rlo, Rmax=Rhi, Rsd=Rsd)), allow_pickle=True)
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
