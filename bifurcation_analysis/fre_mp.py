from pycobi import ODESystem
from pyrates import CircuitTemplate
import sys
import matplotlib.pyplot as plt
from numpy.exceptions import AxisError

"""
Bifurcation analysis of a QIF mean-field model with multiple populations.
"""

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
auto_dir = "~/PycharmProjects/auto-07p"
config_dir = "../config"

# config
continue_lcs = False
N = 20
n_dim = int(3*N)
n_params = 4
ode = ODESystem(eq_file="qif_mp", working_dir=config_dir, auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = ode.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                         EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 1000.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# continuation in independent parameter
p1 = "J"
p1_idx = 2
p1_vals = [5.0, 10.0, 15.0]
p1_min, p1_max = 0.0, 20.0
ncol = 6
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='1d', ICP=p1_idx, NPAR=n_params, NDIM=n_dim, name=f'{p1}:0',
                           origin="t", NMX=8000, DSMAX=0.05, UZR={p1_idx: p1_vals}, STOP=[],
                           NPR=20, RL1=p1_max, RL0=p1_min, EPSL=1e-7, EPSU=1e-7, EPSS=1e-4, bidirectional=True)

# continuations in eta
eta_idx = 1
eta_min, eta_max = -10.0, 10.0
for i, p1_val in enumerate(p1_vals):

    c2_sols, c2_cont = ode.run(starting_point=f'UZ{i+1}', ICP=eta_idx, name=f'eta:{i+1}', DSMAX=0.01,
                               origin=c1_cont, UZR={}, STOP=[], NPR=5, RL1=eta_max, RL0=eta_min, bidirectional=False,
                               variables=["U(1)"], DS=1e-4)

    # plot 1D bifurcation diagram
    fig, ax = plt.subplots(figsize=(12, 4))
    ode.plot_continuation(f"PAR({eta_idx})", "U(1)", cont=f"eta:{i+1}", ax=ax, bifurcation_legend=True,
                          default_size=markersize)
    ax.set_title(f"1D bifurcation diagram for {p1} = {p1_val}")
    plt.tight_layout()

# 2D continuation I
p1_val_idx = 0
dsmax = 0.1
NMX = 2000
NTST = 200
fold_bifurcations = True
try:
    ode.run(starting_point='LP1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:lp1', origin=f"eta:{p1_val_idx+1}", NMX=NMX,
            DSMAX=dsmax, NPR=10, RL1=p1_max, RL0=p1_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=NTST,
            variables=["U(1)"], get_stability=False)
    ode.run(starting_point='LP2', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:lp2', origin=f"eta:{p1_val_idx+1}",
            NMX=NMX, DSMAX=dsmax, NPR=10, RL1=p1_max, RL0=p1_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=NTST,
            variables=["U(1)"], get_stability=False)
except KeyError:
    fold_bifurcations = False

# 2D continuation II
params_2d = ["Delta"]
params_idx = [3]
params_min = [0.01]
params_max = [10.0]
for p2, p2_idx, p2_min, p2_max in zip(params_2d, params_idx, params_min, params_max):
    if fold_bifurcations:
        ode.run(starting_point='LP1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:lp1', origin=f"eta:{p1_val_idx+1}",
                NMX=NMX, DSMAX=dsmax, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=NTST,
                get_stability=False, STOP=["BP2"], variables=["U(1)"])
        ode.run(starting_point='LP2', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:lp2', origin=f"eta:{p1_val_idx+1}", NMX=NMX,
                DSMAX=dsmax, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=NTST, STOP=["BP2"],
                variables=["U(1)"], get_stability=False)
print(f"Fold bifurcations: {'yes' if fold_bifurcations else 'no'}")

# plot 2D bifurcation diagrams
for idx1, key1 in zip([p1_idx,] + params_idx, [p1,] + params_2d):
    fig, ax = plt.subplots(figsize=(12, 4))
    if fold_bifurcations:
        for lp in [1, 2]:
            try:
                ode.plot_continuation(f"PAR({eta_idx})", f"PAR({idx1})", cont=f"{key1}/eta:lp{lp}",
                                      ax=ax, bifurcation_legend=True, get_stability=False, default_size=markersize)
            except KeyError:
                pass
    fig.suptitle(f"2d bifurcations: {key1}/eta for {p1} = {p1_vals[p1_val_idx]}")
    plt.tight_layout()

plt.show()
