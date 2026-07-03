r"""
Kuramoto heterogeneity bifurcation — coupling K and two heterogeneity knobs (h_Delta, h_ombar)
==============================================================================================

PyCoBi/Auto-07p bifurcation analysis of the Lorentzian-mixture (LMMF) Ott-Antonsen mean field of a
globally coupled Kuramoto network with an arbitrary (Gaussian-mixture) frequency distribution.
Two heterogeneity knobs rescale the frequency distribution about its weighted mean Ombar = sum_m w_m Omega_m:

    width spread :  Delta_m(hD) = hD * Delta_m^0
    centre spread:  Omega_m(hC) = Ombar + hC * (Omega_m^0 - Ombar)            (hC == h_ombar)

`combined=True` ties them into a single lumped knob hD=hC=h (h=1 data, h->0 homogeneous).

Reduced OA in co-rotating Cartesian order parameters z_m=x_m+i y_m (pin y_0:=0, x_0=r_0; carry the
global rotation Omega=Im F_0/x_0), so the partially-synchronized state is a genuine fixed point.  No
plasticity (A_ml=1): field = K sum_j w_j z_j.  K, hD, hC are Auto-continuable; w_m, Omega_m^0, Delta_m^0
inlined.  Order parameter R = |sum_m w_m z_m|.

Modes (CLI, one ODESystem per process — two in one process collide):
    lumped   : single knob h. 1-D bif R(h) at fixed K + 2-D fold/Hopf loci in the (h, K) plane.
    twoknob  : 1-D bif R(hD) at fixed (hC, K) + 2-D fold/Hopf loci in the (hD, hC) plane
               (expected multistability at small hD / large hC).
    both     : run lumped then twoknob is NOT allowed in one process; call twice.

Run in the ``pycobi`` conda env (reads kmo_het_fit.npz from kmo_heterogeneity_sim.py fit):
    python kmo_heterogeneity_bifurcation.py lumped
    python kmo_heterogeneity_bifurcation.py twoknob
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
from pycobi import ODESystem
from pycobi.utility import write_auto_dat

AUTO_DIR = "~/PycharmProjects/auto-07p"
FIT_NPZ = "/home/rgast/data/mpmf_simulations/kmo_het_fit.npz"
_HERE = os.path.dirname(os.path.abspath(__file__))

# PyRates emits PAR slots for the constant params in declaration order: K=PAR(1), h=PAR(2)
# (lumped) ; PAR(11) is Auto's reserved period slot.
PAR_H = 2
PAR_PERIOD = 11

CONFIG = dict(
    K_fix=1.0,                 # coupling for the 1-D cuts / the (hD,hC) plane
    h_settle=0.30,             # lumped: settle h ;  twoknob: settle hD (with hC=hC_settle)
    hC_settle=0.30,            # twoknob: settle hC small (collapsed centres -> coherent), then the
                               # fold is continued out into the large-hC / small-hD multistable region
    r0_start=0.8, T_settle=2000.0,
    H_MIN=0.02, H_MAX=1.3,     # heterogeneity-knob continuation range (hD, hC, or lumped h)
    K_MIN=0.2, K_MAX=4.0,      # K range for the (h, K) loci (twoknob plane only)
    lc_seed_offset=0.02, per_max=4000.0, T_settle_lc=2500.0,   # limit-cycle seeding past the Hopf
)


def _xj(j):
    return f"x_{j}"


def _yj(j):
    return "0" if j == 0 else f"y_{j}"


def state_var_order(M):
    """State-variable / .dat column order emitted by PyRates: x_0, (x_1,y_1), ..., (x_{M-1},y_{M-1})."""
    names = ["x_0"]
    for i in range(1, M):
        names += [f"x_{i}", f"y_{i}"]
    return names


def build_equations(M, omega, delta, weights, combined=False):
    """Reduced OA (co-rotating Cartesian, y_0:=0). Width knob hD scales Delta_m, centre knob hC
    scales the centre spread about Ombar; combined ties them to a single h. dim = 2M-1."""
    w = [float(x) for x in weights]; om = [float(x) for x in omega]; dl = [float(x) for x in delta]
    Ombar = repr(float(np.asarray(weights, float) @ np.asarray(omega, float)))
    hw, hc = ("h", "h") if combined else ("hD", "hC")

    def De(i):
        return f"({hw}*{dl[i]})"

    def Om(i):
        return f"({Ombar} + {hc}*({om[i]} - {Ombar}))"

    rH = "K*(" + " + ".join(f"{w[j]}*{_xj(j)}" for j in range(M)) + ")"
    _imt = [f"{w[j]}*{_yj(j)}" for j in range(M) if j != 0]
    iH = "K*(" + (" + ".join(_imt) if _imt else "0") + ")"

    def ReF(i):
        xi, yi = _xj(i), _yj(i)
        rezz = f"(x_0^2)*({rH})" if i == 0 else f"(({xi})^2-({yi})^2)*({rH}) + 2*{xi}*{yi}*({iH})"
        bias = f"-{De(i)}*{xi}" if i == 0 else f"-{De(i)}*{xi} - {Om(i)}*{yi}"
        return f"({bias} + 0.5*(({rH}) - ({rezz})))"

    def ImF(i):
        xi, yi = _xj(i), _yj(i)
        if i == 0:
            imzz, bias = f"-(x_0^2)*({iH})", f"{Om(0)}*x_0"
        else:
            imzz = f"2*{xi}*{yi}*({rH}) - (({xi})^2-({yi})^2)*({iH})"
            bias = f"-{De(i)}*{yi} + {Om(i)}*{xi}"
        return f"({bias} + 0.5*(({iH}) - ({imzz})))"

    Omega = f"(({ImF(0)})/x_0)"
    eqs = [f"d/dt * x_0 = {ReF(0)}"]
    for i in range(1, M):
        eqs.append(f"d/dt * x_{i} = {ReF(i)} + {Omega}*y_{i}")
        eqs.append(f"d/dt * y_{i} = {ImF(i)} - {Omega}*x_{i}")
    return eqs


# ── numpy mirror of the reduced field (lumped h), only to synthesise an LC seed orbit ────────
def cart_rhs_h(t, s, M, K, h, omega0, delta0, weights, Ombar):
    x = np.empty(M); yv = np.empty(M)
    x[0] = s[0]; yv[0] = 0.0
    x[1:] = s[1:1 + 2 * (M - 1):2]; yv[1:] = s[2:1 + 2 * (M - 1):2]
    z = x + 1j * yv
    De = h * delta0; Om = Ombar + h * (omega0 - Ombar)
    H = K * (weights @ z)
    F = (-De + 1j * Om) * z + 0.5 * (H - z ** 2 * np.conj(H))
    Omg = F.imag[0] / x[0]
    dx = F.real + Omg * yv; dy = F.imag - Omg * x
    out = np.empty_like(s)
    out[0] = dx[0]
    out[1:1 + 2 * (M - 1):2] = dx[1:]; out[2:1 + 2 * (M - 1):2] = dy[1:]
    return out


def settle_and_classify_h(M, K, h, omega0, delta0, weights, Ombar, r0val,
                          T_settle=4000.0, n_samples=201, amp_tol=2e-3, closure_tol=3e-2):
    """Integrate from a synchronized IC at this h and classify the attractor:
    'fixedpoint' (x_0 stops oscillating), 'cycle' (closes at its fundamental period -> one-period
    DataFrame for write_auto_dat), or 'torus' (does not close)."""
    s0 = np.zeros(2 * M - 1); s0[0] = r0val
    s0[1:1 + 2 * (M - 1):2] = r0val
    sol = solve_ivp(cart_rhs_h, (0.0, T_settle), s0, args=(M, K, h, omega0, delta0, weights, Ombar),
                    method="LSODA", rtol=1e-9, atol=1e-11, dense_output=True, max_step=2.0)
    tt = np.linspace(0.7 * T_settle, T_settle, 40000)
    x0 = sol.sol(tt)[0]
    amp = float(x0.max() - x0.min())
    if amp < amp_tol:
        return "fixedpoint", None, None, amp, None
    xc = x0 - x0.mean()
    cr = np.where((xc[:-1] < 0) & (xc[1:] >= 0))[0]
    if cr.size < 2:
        return "torus", None, None, amp, None
    base = float(np.diff(tt[cr]).mean())
    t_anchor = 0.7 * T_settle; s_anchor = sol.sol(t_anchor)
    best = (1e9, base)
    for k in range(1, 9):
        Tc = base * k
        err = float(np.max(np.abs(sol.sol(t_anchor + Tc) - s_anchor)))
        if err < best[0]:
            best = (err, Tc)
    err, Tfund = best
    if err > closure_tol:
        return "torus", None, None, amp, err
    tau = np.linspace(0.0, 1.0, n_samples)
    states = np.array([sol.sol(t_anchor + s * Tfund) for s in tau])
    df = pd.DataFrame(states, columns=state_var_order(M), index=tau * Tfund)
    return "cycle", Tfund, df, amp, err


def build_circuit(M, K, omega, delta, weights, r0, combined=False,
                  h0=1.0, hD0=1.0, hC0=1.0, name="kmo_het"):
    eqs = build_equations(M, omega, delta, weights, combined=combined)
    variables = {"x_0": f"output({float(r0[0])})"}
    for i in range(1, M):
        variables[f"x_{i}"] = f"variable({float(r0[i])})"
        variables[f"y_{i}"] = "variable(0.0)"
    variables["K"] = float(K)
    if combined:
        variables["h"] = float(h0)
    else:
        variables["hD"] = float(hD0)
        variables["hC"] = float(hC0)
    op = OperatorTemplate(name=f"{name}_op", equations=eqs, variables=variables)
    node = NodeTemplate(name=f"{name}_node", operators=[op])
    return CircuitTemplate(name=name, nodes={"p": node})


def load_fit():
    d = np.load(FIT_NPZ, allow_pickle=False)
    return (np.asarray(d["weights"], float), np.asarray(d["omega"], float),
            np.asarray(d["delta"], float), int(d["M"]))


# ── summary helpers ───────────────────────────────────────────────────────────
def _pcol(df, name):
    for c in df.columns:
        head = c[0] if isinstance(c, tuple) else c
        sub = c[1] if isinstance(c, tuple) and len(c) > 1 else ""
        if (head == name or (isinstance(head, str) and head.endswith("/" + name))) and sub in ("", 0):
            return c
    raise KeyError(name)


def _bif_vals(df, label, pname):
    pc = _pcol(df, pname)
    rows = df[df[("bifurcation", "")] == label]
    return sorted(float(v) for v in rows[pc]) if len(rows) else []


def _branch(ode, cont, M, weights, xname, yname, with_stab=True):
    """(x, y, R=|sum w_m z_m| min/max envelope, stability, bifurcation) from a summary."""
    summ = ode.get_summary(cont)
    head = lambda n: [c for c in summ.columns if (c[0] if isinstance(c, tuple) else c) == n]
    col = lambda n: np.column_stack([np.asarray(summ[c], float) for c in head(n)]) if head(n) else None
    x = col(xname)[:, 0]; y = col(yname)[:, 0]
    bif = np.asarray(summ[head("bifurcation")[0]]).astype(str)
    sc = head("stability")
    stab = np.asarray(summ[sc[0]], bool) if (with_stab and sc) else np.ones(x.size, bool)
    ncol = col("x_0").shape[1]
    R = np.zeros((x.size, ncol))
    for c in range(ncol):
        z = np.zeros(x.size, complex)
        for m in range(M):
            xm = col(f"x_{m}")[:, c]
            ym = col(f"y_{m}")[:, c] if m != 0 else np.zeros(x.size)
            z = z + weights[m] * (xm + 1j * ym)
        R[:, c] = np.abs(z)
    return dict(x=x, y=y, Rmin=R.min(axis=1), Rmax=R.max(axis=1), stab=stab, bif=bif)


def _settle(ode, cfg):
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=1.0, NMX=500000,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, UZR={14: cfg["T_settle"]}, STOP={"UZ1"})


def _eq1d(ode, knob, cfg, stop=None):
    kw = dict(STOP=stop) if stop else {}
    eq, _ = ode.run(origin="time", starting_point="UZ1", name="eq1", c="eq",
                    ICP=knob, bidirectional=True, RL0=cfg["H_MIN"], RL1=cfg["H_MAX"],
                    IPS=1, ILP=1, ISP=2, ISW=1, NMX=8000, NPR=20,
                    DSMIN=1e-9, DSMAX=5e-3, EPSL=1e-7, EPSU=1e-7, EPSS=1e-5, get_stability=True, **kw)
    return eq


def _loci_2d(ode, knob, second, cfg, bounds2):
    """Trace fold (LP) and Hopf (HB) points of eq1 in the (knob, second) plane."""
    loci = {}
    common = dict(ICP=[knob, second], bidirectional=True, RL0=cfg["H_MIN"], RL1=cfg["H_MAX"],
                  UZSTOP={second: list(bounds2)}, NMX=8000, NPR=20, DS=1e-3, DSMIN=1e-10,
                  DSMAX=2e-2, EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, get_stability=False)
    eq = ode.get_summary("eq1")
    for label, tag in (("LP", "fold"), ("HB", "hopf")):
        n = len(eq[eq[("bifurcation", "")] == label])
        for k in range(1, n + 1):
            nm = f"{tag}2d_{k}"
            try:
                ode.run(origin="eq1", starting_point=f"{label}{k}", name=nm, c="eq",
                        IPS=1, ISW=2, ILP=0, ISP=0, **common)
                a = _branch(ode, nm, cfg["M"], cfg["w"], knob, second, with_stab=False)
                loci[f"{tag}_{k}"] = dict(x=a["x"], y=a["y"])
                print(f"   {nm}: {a['x'].size} pts, {knob}∈[{a['x'].min():.3f},{a['x'].max():.3f}] "
                      f"{second}∈[{a['y'].min():.3f},{a['y'].max():.3f}]")
            except Exception as e:
                print(f"   {nm}: FAILED ({type(e).__name__}: {e})")
    return loci


def run_lumped(cfg):
    w, Om, De, M = load_fit(); cfg = dict(cfg, M=M, w=w)
    Ombar = float(w @ Om); DIM = 2 * M - 1
    print(f"== lumped-h:  M={M} (dim {DIM})  Ombar={Ombar:.4f}  K={cfg['K_fix']} ==")
    circ = build_circuit(M, cfg["K_fix"], Om, De, w, np.full(M, cfg["r0_start"]),
                         combined=True, h0=cfg["h_settle"])
    ode = ODESystem.from_template(circ, auto_dir=AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq", "lc"))
    _settle(ode, cfg)
    eq = _eq1d(ode, "h", cfg, stop={"BP1"})                   # stop the coherent branch at the BP (R->0)
    hbs = sorted(_bif_vals(eq, "HB", "h"), reverse=True)      # highest-h Hopf encountered first
    print(f"   LP at h={[round(x,4) for x in _bif_vals(eq,'LP','h')]};  HB at h={[round(x,4) for x in sorted(hbs)]}")

    # limit cycle: settle just past each Hopf (highest h first), find a clean cycle, save a
    # one-period .dat, and continue it.  (Some Hopfs give a quasiperiodic torus, not continuable.)
    lc_data = []
    for hb in hbs:
        h_seed = hb - cfg["lc_seed_offset"]
        try:
            status, T, orbit, amp, err = settle_and_classify_h(
                M, cfg["K_fix"], h_seed, Om, De, w, Ombar, cfg["r0_start"],
                T_settle=cfg["T_settle_lc"], closure_tol=6e-2)
        except Exception as e:
            print(f"   LC probe at h={h_seed:.3f} (past HB {hb:.3f}): settle FAILED ({type(e).__name__}: {e})")
            continue
        print(f"   LC probe at h={h_seed:.3f} (past HB {hb:.3f}): {status}  amp={amp:.3g}  "
              f"T={None if T is None else round(T,2)}  closure={None if err is None else f'{err:.1e}'}")
        if status == "cycle":
            datstem = os.path.join(_HERE, "kmo_lc_seed")
            write_auto_dat(orbit, datstem + ".dat", normalize_time=False)
            try:
                lc, _ = ode.run(name="lc1", dat=datstem, c="lc", NDIM=DIM, NPAR=36,
                                PAR={PAR_H: h_seed}, IPS=2, ISP=2, ILP=1, ISW=1,
                                ICP=[PAR_H, PAR_PERIOD], NTST=200, NCOL=4,
                                RL0=cfg["H_MIN"], RL1=cfg["H_MAX"], NMX=8000, NPR=100,
                                DS=1e-3, DSMIN=1e-11, DSMAX=1e-2, EPSL=1e-8, EPSU=1e-8, EPSS=1e-7,
                                THL={PAR_PERIOD: 0.0}, UZSTOP={PAR_PERIOD: cfg["per_max"]},
                                bidirectional=True, get_period=True, get_stability=True)
                a = _branch(ode, "lc1", M, w, "h", "h")
                lc_data = [a]
                print(f"   lc1: {a['x'].size} pts, h∈[{a['x'].min():.3f},{a['x'].max():.3f}], "
                      f"R∈[{a['Rmin'].min():.3f},{a['Rmax'].max():.3f}], "
                      f"stable frac={a['stab'].mean():.2f}")
                break
            except Exception as e:
                print(f"   lc1: FAILED ({type(e).__name__}: {e})")
                break

    # 2-D fold/Hopf loci in the (h, K) plane (coupling vs lumped heterogeneity)
    loci = _loci_2d(ode, "h", "K", cfg, (cfg["K_MIN"], cfg["K_MAX"]))
    data = dict(mode="lumped", M=int(M), Ombar=Ombar, K_fix=float(cfg["K_fix"]),
                eq=_branch(ode, "eq1", M, w, "h", "K"), lc=lc_data, loci=loci)
    np.save(os.path.join(_HERE, "kmo_het_bif_lumped.npy"), data, allow_pickle=True)
    print(f"   [saved] kmo_het_bif_lumped.npy")
    ode.close_session(clear_files=True); clear(circ)


def run_twoknob(cfg):
    w, Om, De, M = load_fit(); cfg = dict(cfg, M=M, w=w)
    Ombar = float(w @ Om)
    print(f"== two-knob:  M={M}  Ombar={Ombar:.4f}  K={cfg['K_fix']}  (1-D in hD at hC={cfg['hC_settle']}) ==")
    circ = build_circuit(M, cfg["K_fix"], Om, De, w, np.full(M, cfg["r0_start"]),
                         combined=False, hD0=cfg["h_settle"], hC0=cfg["hC_settle"])
    ode = ODESystem.from_template(circ, auto_dir=AUTO_DIR, init_cont=False,
                                  analytical_jacobian=True, auto_constants=("ivp", "eq"))
    _settle(ode, cfg)
    eq = _eq1d(ode, "hC", cfg)        # continue the centre spread (the synchronization control here)
    print(f"   LP at hC={[round(x,4) for x in _bif_vals(eq,'LP','hC')]}; "
          f"HB at hC={[round(x,4) for x in _bif_vals(eq,'HB','hC')]}")
    loci = _loci_2d(ode, "hD", "hC", cfg, (cfg["H_MIN"], cfg["H_MAX"]))
    data = dict(mode="twoknob", M=int(M), Ombar=Ombar, K_fix=float(cfg["K_fix"]),
                hC_settle=float(cfg["hC_settle"]),
                eq=_branch(ode, "eq1", M, w, "hD", "hC"), loci=loci)
    np.save(os.path.join(_HERE, "kmo_het_bif_twoknob.npy"), data, allow_pickle=True)
    print(f"   [saved] kmo_het_bif_twoknob.npy")
    ode.close_session(clear_files=True); clear(circ)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "lumped"
    {"lumped": run_lumped, "twoknob": run_twoknob}[mode](CONFIG)
