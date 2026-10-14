r"""
Check: LMMF incoherence boundaries from the eigenvalue maps vs PyCoBi Hopf continuations
=========================================================================================

The figures draw the LMMF incoherence boundaries as zero contours of the largest (G1) and
second-largest (G2) real part of the eigenvalues of the linearisation at z = 0. Here the same
boundaries are computed independently with Auto-07p (PyCoBi): the trivial state z = 0 of the lab-frame
Cartesian LMMF (2M dims, parameters K and α) is continued in K at fixed α0, every Hopf point on it is
continued in (K, α) (ISW = 2), and the curves are overlaid on the eigenvalue contours.

  dx_m = −Δ_m x_m − Ω_m y_m + ½ (H_r − [H_r (x_m² − y_m²) + 2 x_m y_m H_i])
  dy_m = −Δ_m y_m + Ω_m x_m + ½ (H_i − [2 x_m y_m H_r − H_i (x_m² − y_m²)])
  H_r = K (cos α X + sin α Y),  H_i = K (cos α Y − sin α X),  X = Σ w x,  Y = Σ w y.

Fits: bimodal floor / nofloor (bimodal_kalpha), unimodal gauss, g8, dgauss × pop / floor / nofloor
(fit_scenarios, M_max = 64, width floor 10). One Auto session per process, run sequentially.

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python lmmf_hopf_check.py run [--fits bimodal:floor,gauss:nofloor]
    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python lmmf_hopf_check.py figure
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "unimodal"),
                os.path.join(_HERE, "..", "..", "shared"), os.path.join(_HERE, "..", "continuity")]
import subprocess
import time
import numpy as np
import data_paths as dp

OUT = dp.ensure(os.path.join(dp.mpmf("bimodal_kalpha"), "hopf_check"))
AUTO_DIR = "~/PycharmProjects/auto-07p"
SEEDS_A = (0.7, 0.2, 1.2)                # α0 of the K-continuations (all used; curves de-duplicated)
A_MAX = 1.45
FITS = ["bimodal:floor", "bimodal:nofloor"] + [f"{d}:{s}" for d in ("gauss", "g8", "dgauss")
                                               for s in ("pop", "floor", "nofloor")]


def load(key):
    """-> (fit dict, K_max of the figure window)."""
    dist, v = key.split(":")
    if dist == "bimodal":
        import bimodal_kalpha as BK
        return BK.lmmf_fit(v), 6.0
    import fit_scenarios as FS
    f = np.load(FS.fpath(dist, v, 0, 1000, 64, 10), allow_pickle=True).item()
    return dict(w=f["w"], Om=f["Om"], De=f["De"], M=int(f["M"])), 4.0


def eig_maps(f, K, A):
    """G1, G2 = largest and second-largest Re eigenvalue at z = 0 on the (A × K) grid."""
    A0 = np.diag(1j * f["Om"] - f["De"]); P = 0.5 * np.outer(np.ones(f["M"]), f["w"])
    G1 = np.empty((A.size, K.size)); G2 = np.empty_like(G1)
    for i, a in enumerate(A):
        for j, k in enumerate(K):
            ev = np.sort(np.linalg.eigvals(A0 + k * np.exp(-1j * a) * P).real)[::-1]
            G1[i, j], G2[i, j] = ev[0], ev[1]
    return G1, G2


def _num(v):
    return f"{float(v):.10f}"                # fixed point: avoids the PyRates Fortran line-wrap of exponents


def _circuit(f, K0, a0, tag):
    from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate
    M = f["M"]; w = [_num(v) for v in f["w"]]; Om = [_num(v) for v in f["Om"]]; De = [_num(v) for v in f["De"]]
    X = "(" + " + ".join(f"{w[j]}*x_{j}" for j in range(M)) + ")"
    Y = "(" + " + ".join(f"{w[j]}*y_{j}" for j in range(M)) + ")"
    Hr = f"K*(cos(alpha)*{X} + sin(alpha)*{Y})"
    Hi = f"K*(cos(alpha)*{Y} - sin(alpha)*{X})"
    eqs, var = [], {}
    for m in range(M):
        x, y = f"x_{m}", f"y_{m}"
        eqs.append(f"d/dt * {x} = -{De[m]}*{x} - ({Om[m]})*{y} + 0.5*(({Hr}) - "
                   f"((({x})^2 - ({y})^2)*({Hr}) + 2*{x}*{y}*({Hi})))")
        eqs.append(f"d/dt * {y} = -{De[m]}*{y} + ({Om[m]})*{x} + 0.5*(({Hi}) - "
                   f"(2*{x}*{y}*({Hr}) - (({x})^2 - ({y})^2)*({Hi})))")
        var[x] = "output(0.0)" if m == 0 else "variable(0.0)"
        var[y] = "variable(0.0)"
    var["K"] = float(K0); var["alpha"] = float(a0)
    op = OperatorTemplate(name="hc_op", equations=eqs, variables=var)
    return CircuitTemplate(name=f"hopf_{tag}", nodes={"p": NodeTemplate(name="hc_node", operators=[op])})


def _cols(ode, name):
    s = ode.get_summary(name)
    head = lambda n: [c for c in s.columns if (c[0] if isinstance(c, tuple) else c) == n]
    col = lambda n: np.asarray(s[head(n)[0]], float)
    return col, np.asarray(s[head("bifurcation")[0]]).astype(str)


def hopf_curves(key, a0):
    """One Auto session: z = 0 continued in K at α0 -> Hopf points -> (K, α) continuation of each."""
    from pycobi import ODESystem
    from pyrates import clear
    f, kmax = load(key)
    circ = _circuit(f, 0.05, a0, f"{key.replace(':', '_')}_{int(round(100 * a0))}")
    ode = ODESystem.from_template(circ, auto_dir=AUTO_DIR, init_cont=False, analytical_jacobian=True,
                                  auto_constants=("ivp", "eq"))
    ode.run(c="ivp", name="time", DS=1e-3, DSMIN=1e-9, DSMAX=0.5, NMX=20000, UZR={14: 5.0}, STOP={"UZ1"})
    ode.run(origin="time", starting_point="UZ1", name="incoh", c="eq", ICP="K", RL0=0.01, RL1=kmax + 0.5,
            IPS=1, ILP=0, ISP=2, ISW=1, NMX=8000, NPR=50, DS=1e-3, DSMIN=1e-9, DSMAX=5e-3,
            EPSL=1e-8, EPSU=1e-8, EPSS=1e-6, get_stability=False)
    col, bif = _cols(ode, "incoh")
    K_hb = col("K")[bif == "HB"]
    curves = []
    for k in range(1, K_hb.size + 1):
        nm = f"hb2d_{k}"
        try:
            ode.run(origin="incoh", starting_point=f"HB{k}", name=nm, c="eq", ICP=["K", "alpha"],
                    bidirectional=True, RL0=0.01, RL1=kmax + 0.5, UZSTOP={"alpha": [-0.05, A_MAX + 0.05]},
                    IPS=1, ISW=2, ILP=0, ISP=0, NMX=8000, NPR=20, DS=1e-3, DSMIN=1e-10, DSMAX=1e-2,
                    EPSL=1e-7, EPSU=1e-7, EPSS=1e-6, get_stability=False)
            c2, _ = _cols(ode, nm)
            curves.append(np.column_stack([c2("K"), c2("alpha")]))
        except Exception as e:
            print(f"   {key} α0={a0:g} HB{k}: 2-D continuation failed ({type(e).__name__}: {e})", flush=True)
    ode.close_session(clear_files=True); clear(circ)
    return K_hb, curves


def run_one(key, a0):
    os.chdir(dp.local_work("lmmf_hopf_check_auto"))   # executables must not live on the share
    t0 = time.time()
    K_hb, curves = hopf_curves(key, a0)
    np.save(os.path.join(OUT, f"{key.replace(':', '_')}_a{a0:g}.npy"),
            dict(K_hb=K_hb, curves=np.array(curves, dtype=object), a0=a0), allow_pickle=True)
    print(f"[auto] {key} α0={a0:g}: HB at K = {np.round(K_hb, 4)}; {len(curves)} curve(s) ({time.time() - t0:.0f} s)",
          flush=True)


def run_all(keys):
    for key in keys:
        for a0 in SEEDS_A:
            if os.path.exists(os.path.join(OUT, f"{key.replace(':', '_')}_a{a0:g}.npy")):
                continue
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "one", key, str(a0)],
                               capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("[auto]")]
            print(line[-1] if line else f"[auto] {key} α0={a0:g}: FAILED\n{r.stderr[-800:]}", flush=True)


def check_points(f, curves, K, A, G1, G2):
    """At every Auto Hopf point inside the window: smallest |Re λ| over all eigenvalues at z = 0 (should
    be ~0) and whether the point lies on the G1 / G2 zero contour (|G| small) or on a deeper mode."""
    A0 = np.diag(1j * f["Om"] - f["De"]); P = 0.5 * np.outer(np.ones(f["M"]), f["w"])
    res, on12 = [], 0
    for c in curves:
        for k, a in c:
            if not (K[0] <= k <= K[-1] and 0 <= a <= A_MAX):
                continue
            ev = np.sort(np.linalg.eigvals(A0 + k * np.exp(-1j * a) * P).real)[::-1]
            res.append(np.min(np.abs(ev)))
            on12 += int(min(abs(ev[0]), abs(ev[1])) < 1e-4)
    return (np.max(res) if res else np.nan), on12, len(res)


def figure(keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prl_style import set_prl_style
    set_prl_style("prl")
    n = len(keys); nc = 4; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.0, 1.75 * nr + 0.3), layout="constrained")
    axs = np.atleast_1d(axs).ravel()
    for ax, key in zip(axs, keys):
        f, kmax = load(key)
        K = np.linspace(0.02, kmax, 400); A = np.linspace(0.0, A_MAX, 291)
        G1, G2 = eig_maps(f, K, A)
        ax.contour(K, A, G1, levels=[0.0], colors=["#eb6834"], linewidths=2.2, alpha=0.55)
        ax.contour(K, A, G2, levels=[0.0], colors=["#2a9d6f"], linewidths=2.2, alpha=0.55)
        curves = []
        for a0 in SEEDS_A:
            p = os.path.join(OUT, f"{key.replace(':', '_')}_a{a0:g}.npy")
            if os.path.exists(p):
                curves += list(np.load(p, allow_pickle=True).item()["curves"])
        for c in curves:
            ax.plot(c[:, 0], c[:, 1], "-", color="k", lw=0.6)
        worst, on12, npts = check_points(f, curves, K, A, G1, G2)
        print(f"{key:16s} M={f['M']:2d}: {len(curves)} Auto Hopf curves; max |Re λ|min at Auto points = {worst:.1e} "
              f"({npts} points in window, {on12} on the G1/G2 contours)")
        ax.set_xlim(0, kmax); ax.set_ylim(0, A_MAX)
        ax.set_title(f"{key} (M = {f['M']})", fontsize=6, pad=2)
        ax.set_xlabel(r"$K$", labelpad=1); ax.set_ylabel(r"$\alpha$", labelpad=1)
    for ax in axs[n:]:
        ax.axis("off")
    h = [plt.Line2D([], [], color="#eb6834", lw=2.2, alpha=0.55, label="eigenvalue map: largest Re λ = 0"),
         plt.Line2D([], [], color="#2a9d6f", lw=2.2, alpha=0.55, label="second-largest Re λ = 0"),
         plt.Line2D([], [], color="k", lw=0.6, label="PyCoBi Hopf continuations")]
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False)
    out = os.path.join(OUT, "hopf_check")
    fig.savefig(out + ".png", dpi=250); fig.savefig(out + ".pdf")
    print(f"[saved] {out}.png/.pdf")


if __name__ == "__main__":
    mode = sys.argv[1]
    arg = lambda k, d, t=str: t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    keys = arg("--fits", FITS, lambda s_: s_.split(","))
    if mode == "run":
        run_all(keys)
    elif mode == "one":
        run_one(sys.argv[2], float(sys.argv[3]))
    elif mode == "figure":
        figure(keys)
    else:
        raise SystemExit(__doc__)
