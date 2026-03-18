import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from scipy.signal import welch
from config.utility_functions import *

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
trial = 0
syn = "exc"
stp = "sd"

# set model parameters
M = 10
J = 15.0 / (0.5*M)
Delta = 2.0
p = 1.0
eta = -0.85
b = 0.5
tau_s = 0.5
tau_a = 20.0
kappa = 0.1
etas = uniform(M, eta, Delta)
c1 = -1.0
c2 = 5.0
c3 = 1.0
c4 = 5.0
node_vars = {"eta": etas, "Delta": Delta/(2*M)}
syn_vars = {"tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}

# simulation parameters
cutoff = 0.0
T = 2000.0 + cutoff
dt = 1e-3
dts = 1.0
noise_tau = 200.0
noise_scale = 0.02

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_stdp_{stp}", "qif_op", f"syn_{stp}_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")

# create network
edges = []
for i in range(M):
    for j in range(M):
        edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", None,
                      {"weight": (c1*(etas[j] - eta) + c2)*(c3*(etas[i] - eta))}))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})

# generate run function
inp = np.zeros((int(T/dt), 1), dtype=np.float32)
func, args, arg_keys, _ = net.get_run_func(f"{syn}_{stp}_vectorfield", file_name=f"{syn}_{stp}_run",
                                           step_size=dt, backend="numpy", solver="heun", float_precision="float32",
                                           vectorize=True, inputs={f"all/{node_op}/I_ext": inp}, clear=False)
func_njit = njit(func)
func_njit(*args)
rhs = func_njit

# find argument positions of free parameters
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")
eta_idx = arg_keys.index(f"p0/{node_op}/eta")

# define extrinsic input
noise = np.asarray(generate_colored_noise(int(T/dt), noise_tau, noise_scale), dtype=np.float32)
args = list(args)
args[inp_idx] = noise

# set initial state
init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)

# run simulation
args[inp_idx] = noise
y0_hist, y0 = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)

# calculate network covariance eigenvalues
rs = y0_hist[:, :M] * 100.0
eigvals, eigvecs, C = get_eigs(rs)

# transform etas into covariance eigenvector space
etas = args[eta_idx]
etas_proj = np.dot(eigvecs.T, etas)

# get PSD of first PC
pc1 = np.dot(rs, eigvecs[:, 0])
fs, ps = welch(pc1, fs=100.0/dts, nperseg=512)
f_max = fs[np.argmax(ps)]
p_sum = (fs[1] - fs[0]) * np.sum(ps)

# calculate fano factors
ff = get_ff(rs)

# report some basic stats
#########################

print(f"Neuron type: {syn}")
print(f"STP type: {stp}")

# plotting
##########

fig = plt.figure(figsize=(16, 5), layout="constrained")
grid = fig.add_gridspec(ncols=4, nrows=2)

# plotting dynamics
ax = fig.add_subplot(grid[0, :])
time = np.linspace(0.0, T, int(T/dts)) / 100.0
ax.plot(time, np.mean(rs, axis=1), label="average")
ax.plot(time, rs[:, 0], label="lowest FR neuron")
ax.plot(time, rs[:, -1], label="highest FR neuron")
ax.legend()
ax.set_ylabel(r"$r$ (Hz)")
ax.set_title("network dynamics")
ax.set_xlabel(r"$t$ (s)")

# plotting covariances
ax = fig.add_subplot(grid[1, 0])
im = ax.imshow(C, interpolation="none", aspect="auto", vmin=-1.0, vmax=1.0, cmap="berlin")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Network Covariance")
ax.set_xlabel("n")
ax.set_ylabel("n")

# plotting statistics
ax = fig.add_subplot(grid[1, 1])
ax.plot(eigvals)
ax.legend()
ax.set_xlabel(r"index")
ax.set_ylabel(r"$\lambda$")
ax.set_title("eigenvalues")
ax = fig.add_subplot(grid[1, 2])
ax.plot(ff)
ax.legend()
ax.set_xlabel(r"index")
ax.set_ylabel(r"$ff$")
ax.set_title("fano factors")

# plotting connectivity
ax = fig.add_subplot(grid[1, 3])
ax.plot(etas, c1*(etas-eta) + c2, label="in")
ax.plot(etas, c3*(etas-eta) + c4, label="out")
ax.legend()
ax.set_xlabel(r"$\eta$")
ax.set_ylabel(r"$W_{in/out}$")
ax.set_title("connectivity")

fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
fig.canvas.draw()
plt.show()

# clear files up
clear(net)
