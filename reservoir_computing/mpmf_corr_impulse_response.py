import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from scipy.signal import welch
from config.utility_functions import *
from scipy.ndimage import gaussian_filter1d

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
trial = 0
syn = "exc"
stp = "sd"

# set model parameters
M = 50
Delta = 3.0
p = 1.0
eta = -1.5
b = 0.5
tau_s = 1.0
tau_a = 40.0
kappa = 0.2
etas = uniform(M, eta, Delta)
Delta2 = Delta/(2*M)
c0 = 30.0
c1 = 20.0
c2 = -1.0
c3 = -1.0
node_vars = {"eta": etas, "Delta": Delta2}
syn_vars = {"tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}

# simulation parameters
T = 50.0
dt = 2e-5
dts = 1e-2
sr = int(dts/dt)
stim_amp = 3.0
stim_dur = int(1.0/dt)
noise = 1e-4

# stimulation parameters
p_in = 0.2
sigma = 10
cycle_steps = int(T/dt)
n_inputs = int(p_in*M)
center = int(M*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_stdp_{stp}", "qif_op", f"syn_{stp}_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")

# create network
fr1 = 1.0 / (1.0 + np.exp(-c2*etas))
fr2 = 1.0 / (1.0 + np.exp(-c3*etas))
edges = []
W = (c0 + c1 * np.outer(fr1, fr2)) / M
for i in range(M):
    for j in range(M):
        edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", None,
                      {"weight": W[i, j]}))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})

# show connectivity
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(W, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
ax.set_title("Connectivity")

# generate run function
inp = np.zeros((int(T/dt), M), dtype=np.float32)
func, args, arg_keys, _ = net.get_run_func(f"{syn}_{stp}_vectorfield", file_name=f"{syn}_{stp}_run",
                                           step_size=dt, backend="numpy", solver="heun", float_precision="float32",
                                           vectorize=True, inputs={f"all/{node_op}/I_ext": inp}, clear=False)
func_njit = njit(func)
func_njit(*args)
rhs = func_njit

# find argument positions of free parameters
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")
eta_idx = arg_keys.index(f"p0/{node_op}/eta")
args = list(args)

# set initial state
init_hist, y0 = integrate(args[1], rhs, tuple(args[2:]), T, dt, dts)

# collect network responses
inp = np.zeros((cycle_steps, M))
inp[:stim_dur, inp_indices] = stim_amp
args[inp_idx] = np.asarray(inp, dtype=np.float32)
y_init = np.asarray(y0 + noise*np.random.randn(*y0.shape), dtype=np.float32)
y_init[:M] = np.abs(y_init[:M])
y, _ = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
signal = gaussian_filter1d(y[:, :M], sigma=sigma, axis=0)

# plot results
_, axes = plt.subplots(nrows=2, figsize=(12, 5))
ax = axes[0]
ax.plot(np.mean(signal, axis=1))
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.set_title(f"Mean signal")
ax = axes[1]
im = ax.imshow(signal.T, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.4)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title(f"Reservoir signal")
plt.tight_layout()

# clear files up
plt.show()
clear(net)
