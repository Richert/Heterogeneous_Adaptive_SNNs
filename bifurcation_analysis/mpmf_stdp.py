"""Bifurcation analysis of a QIF mean-field model with synaptic plasticity.

The multi-population mean-field model is now defined in YAML
(``config/qif_stdp_eq.yaml``) and instantiated through PyRates, so PyCoBi can
generate the Fortran source + analytical Jacobian on the fly.  The hand-written
``config/qif_stdp.f90`` is no longer needed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from numpy.exceptions import AxisError

from pyrates import CircuitTemplate, NodeTemplate
from pyrates.frontend.template.population import PopulationTemplate
from pycobi import ODESystem


# preparations
##############

# plotting params
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16.0
plt.rcParams['axes.titlesize'] = 16.0
plt.rcParams['axes.labelsize'] = 16.0
plt.rcParams['lines.linewidth'] = 2.0
markersize = 8

# directories
auto_dir = os.path.expanduser("~/PycharmProjects/auto-07p")
config_dir = os.path.join(os.path.dirname(__file__), os.pardir, "config")
yaml_path = os.path.abspath(os.path.join(config_dir, "qif_stdp_eq.yaml"))

# config
continue_lcs = True
continue_2d = True
N = 10  # number of QIF sub-populations (heterogeneity bins)


# model assembly
################

def build_circuit(n: int) -> CircuitTemplate:
    """Two-population circuit: N qif/sd/trace units plus N*N plastic weights, wired
    via per-element index edges with unit scalar weight (the matrix-valued
    ``Connectivity`` API would expose those weight matrices as PARs, which
    auto-07p rejects).  Matches the qif_stdp.f90 ODE."""

    qif_node = NodeTemplate.from_yaml(f"{yaml_path}/qif_stdp_node")
    w_node = NodeTemplate.from_yaml(f"{yaml_path}/weight_node")

    # Lorentzian-bin offsets: D*((k-1)/(M-1) - 0.5) in the Fortran original.
    offsets = (np.arange(n) / (n - 1) - 0.5) if n > 1 else np.zeros(1)

    qif_pop = PopulationTemplate(
        name="p", node=qif_node, n=n,
        params={"qif_het_op/eta_offset": offsets,
                "qif_het_op/Mpop": float(n)},
    )
    w_pop = PopulationTemplate(name="w", node=w_node, n=n * n)

    # Weight index k = j*n + i  →  i = source, j = target.
    edges = []
    for k in range(n * n):
        j, i = divmod(k, n)
        # gather: w[k] inputs come from p[i] (pre side) and p[j] (post side)
        edges.append(("p/sd_op/s",      "w/weight_op/pre_s",   None,
                      {"weight": 1.0, "source_idx": i, "target_idx": k}))
        edges.append(("p/sd_op/s",      "w/weight_op/post_s",  None,
                      {"weight": 1.0, "source_idx": j, "target_idx": k}))
        edges.append(("p/trace_op/u_p", "w/weight_op/pre_up",  None,
                      {"weight": 1.0, "source_idx": i, "target_idx": k}))
        edges.append(("p/trace_op/u_d", "w/weight_op/post_ud", None,
                      {"weight": 1.0, "source_idx": j, "target_idx": k}))
        # scatter+sum: s_in[j] = sum_i s_out[j*n+i]
        edges.append(("w/weight_op/s_out", "p/qif_het_op/s_in", None,
                      {"weight": 1.0, "source_idx": k, "target_idx": j}))

    return CircuitTemplate(
        name="qif_stdp_mp",
        populations={"p": qif_pop, "w": w_pop},
        edges=edges,
    )


circuit = build_circuit(N)
ODESystem.reset_auto_state()
ode = ODESystem.from_template(
    template=circuit,
    working_dir=config_dir,
    auto_dir=auto_dir,
    analytical_jacobian=True,
    auto_constants=('ivp', 'eq', 'lc'),
    init_cont=False,
)


# initial continuation in time to converge to fixed point
##########################################################

t_sols, t_cont = ode.run(
    c='ivp', name='t',
    DS=1e-4, DSMIN=1e-10, DSMAX=0.1, EPSL=1e-6, EPSU=1e-6, EPSS=1e-5,
    NPR=1000, NMX=90000,
    UZR={14: 1000.0}, STOP={'UZ1'},
)


########################
# bifurcation analysis #
########################

# pick a representative mid-population firing rate as the y-axis variable
mid_idx = N // 2
y_var = ("p/qif_het_op/r", mid_idx)


# 1. ramp J up to 15.0
c0_sols, c0_cont = ode.run(
    starting_point='UZ1', c='eq', ICP='J', name='J:0', origin='t',
    UZR={'J': 15.0}, STOP=['UZ1'],
    RL0=-30.0, RL1=30.0, NPR=20, NMX=8000, DSMAX=0.2,
    EPSL=1e-7, EPSU=1e-7, EPSS=1e-5,
    bidirectional=True,
)

# 2. set the STDP base amplitude a0 = 0.1
ode.run(starting_point='UZ1', ICP='a0', name='a0:0', origin=c0_cont,
        UZR={'a0': [0.1]}, STOP=['UZ1'],
        RL0=0.0, RL1=1.0, DSMAX=0.02, DS=1e-4)

# 3. set the trace-timescale ratio tau_r = 2.0
ode.run(starting_point='UZ1', ICP='tau_r', name='tau_r:0', origin='a0:0',
        UZR={'tau_r': [2.0]}, STOP=['UZ1'],
        RL0=0.0, RL1=3.0, DSMAX=0.02, DS=1e-4)

# 4. continuation in the critical STDP parameter a_r
a_r_vals = [1.2]
c1_sols, c1_cont = ode.run(
    starting_point='UZ1', ICP='a_r', name='a_r:0', origin='tau_r:0',
    UZR={'a_r': a_r_vals}, STOP=[],
    RL0=0.0, RL1=5.0, DSMAX=0.02, DS=1e-4,
)

# 5. eta-continuations + limit cycles at each captured a_r value
for i, a_r in enumerate(a_r_vals):
    eta_cont = f'eta:{i+1}'
    c2_sols, c2_cont = ode.run(
        starting_point=f'UZ{i+1}', ICP='eta', name=eta_cont,
        origin=c1_cont, STOP=[], NPR=5,
        RL0=-10.0, RL1=5.0, DSMAX=0.05, DS=1e-4,
        variables=[y_var], bidirectional=False,
    )

    lc_names = []
    if continue_lcs:
        for hb_label, lc_suffix in (("HB1", "lc:1"), ("HB6", "lc:2")):
            lc_name = f"{eta_cont}:{lc_suffix}"
            try:
                ode.run(
                    starting_point=hb_label, ICP=['eta', 11], name=lc_name,
                    origin=c2_cont, ISW=-1, IPS=2, NCOL=4, NTST=100, NMX=2000,
                    DSMAX=0.1, NPR=10, STOP=["LP6", "BP2"],
                    EPSL=1e-6, EPSU=1e-6, EPSS=1e-4,
                    variables=[y_var],
                )
                lc_names.append(lc_name)
            except KeyError:
                pass

    # 1D bifurcation diagram for this a_r value
    fig, ax = plt.subplots(figsize=(12, 4))
    ode.plot_continuation('PAR(eta)', y_var, cont=eta_cont, ax=ax,
                          bifurcation_legend=True, default_size=markersize)
    for lc_name in lc_names:
        try:
            ode.plot_continuation('PAR(eta)', y_var, cont=lc_name, ax=ax,
                                  bifurcation_legend=True, ignore=["BP"],
                                  default_size=markersize)
        except (KeyError, AxisError):
            pass
    ax.set_title(fr"1D bifurcation diagram for $a_p/a_d$ = {a_r}")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$r_{i}$")
    plt.tight_layout()


# 2D continuations
##################

if continue_2d:
    a_r_focus = 0
    eta_origin = f'eta:{a_r_focus+1}'
    lc_origin = f'{eta_origin}:lc:2'
    NMX = 2000
    dsmax = 0.1

    def codim2(starting_point, p1, p1_min, p1_max, *, ips, origin):
        """Wrap the boilerplate of a single 2D continuation; returns True if it ran."""
        try:
            ode.run(
                starting_point=starting_point, ICP=[p1, 'eta'],
                name=f"{p1}/eta:{starting_point.lower()}", origin=origin,
                NMX=NMX, DSMAX=dsmax, NPR=10,
                RL0=p1_min, RL1=p1_max, bidirectional=True,
                ILP=0, IPS=ips, ISW=2, NTST=400,
                STOP=["BP2"], variables=[y_var],
                get_stability=False,
            )
            return True
        except KeyError:
            return False

    flags = {
        "lp1": False, "lp2": False, "hb1": False, "hb2": False,
        "pd1": False, "tr1": False,
    }

    # 2D continuation I: (a_r, eta)
    flags["lp1"] = codim2("LP1", "a_r", 0.5, 2.0, ips=1, origin=eta_origin)
    flags["lp2"] = codim2("LP2", "a_r", 0.5, 2.0, ips=1, origin=eta_origin)
    flags["hb1"] = codim2("HB1", "a_r", 0.5, 2.0, ips=1, origin=eta_origin)
    flags["hb2"] = codim2("HB6", "a_r", 0.5, 2.0, ips=1, origin=eta_origin)

    pd_bifurcation = False
    tr_bifurcation = False
    if pd_bifurcation:
        flags["pd1"] = codim2("PD1", "a_r", 0.5, 2.0, ips=2, origin=lc_origin)
    if tr_bifurcation:
        flags["tr1"] = codim2("TR1", "a_r", 0.5, 2.0, ips=2, origin=lc_origin)

    # 2D continuation II: (Delta, eta) and (tau, eta)
    extra_params = [("D", 1e-4, 5.0), ("tau", 0.5, 20.0)]
    for p2, p2_min, p2_max in extra_params:
        if flags["lp1"]:
            codim2("LP1", p2, p2_min, p2_max, ips=1, origin=eta_origin)
        if flags["lp2"]:
            codim2("LP2", p2, p2_min, p2_max, ips=1, origin=eta_origin)
        if flags["hb1"]:
            codim2("HB1", p2, p2_min, p2_max, ips=1, origin=eta_origin)
        if flags["hb2"]:
            codim2("HB6", p2, p2_min, p2_max, ips=1, origin=eta_origin)
        if pd_bifurcation and flags["pd1"]:
            codim2("PD1", p2, p2_min, p2_max, ips=2, origin=lc_origin)
        if tr_bifurcation and flags["tr1"]:
            codim2("TR1", p2, p2_min, p2_max, ips=2, origin=lc_origin)

    for label, present in flags.items():
        print(f"{label.upper():>4}: {'yes' if present else 'no'}")

    # 2D plots
    for p2_name in ["a_r", "D", "tau"]:
        fig, ax = plt.subplots(figsize=(12, 4))
        for lp_label in ("lp1", "lp2"):
            try:
                ode.plot_continuation('PAR(eta)', f'PAR({p2_name})',
                                      cont=f'{p2_name}/eta:{lp_label}',
                                      ax=ax, bifurcation_legend=True,
                                      get_stability=False,
                                      default_size=markersize)
            except KeyError:
                pass
        for hb_label in ("hb1", "hb2"):
            try:
                ode.plot_continuation('PAR(eta)', f'PAR({p2_name})',
                                      cont=f'{p2_name}/eta:{hb_label}',
                                      ax=ax, bifurcation_legend=True,
                                      line_color_stable="green",
                                      get_stability=False,
                                      default_size=markersize)
            except KeyError:
                pass
        for special_label, color in (("pd1", "blue"), ("tr1", "red")):
            try:
                ode.plot_continuation(
                    'PAR(eta)', f'PAR({p2_name})',
                    cont=f'{p2_name}/eta:{special_label}',
                    ax=ax, bifurcation_legend=True,
                    line_color_stable=color, line_style_stable="dashed",
                    ignore=["BP"], get_stability=False,
                    default_size=markersize,
                    custom_bf_styles={
                        'R1': {'marker': 's', 'color': 'k'},
                        'R2': {'marker': 'o', 'color': 'k'},
                        'R3': {'marker': 'v', 'color': 'k'},
                        'R4': {'marker': 'd', 'color': 'k'},
                    },
                )
            except KeyError:
                pass
        fig.suptitle(f"2d bifurcations for eta/{p2_name}")
        ax.set_xlabel("eta")
        ax.set_ylabel(p2_name)
        plt.tight_layout()

plt.show()
