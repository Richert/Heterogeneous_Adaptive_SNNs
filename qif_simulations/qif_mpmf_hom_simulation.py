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

# set model parameters
M = 10
Delta = 2.0
p = 1.0
eta = -0.85
tau_s = 0.5
tau_a = 100.0
J = 40.0 / (0.5*M)
alpha = 0.1
node_vars = {"eta": uniform(M, eta, Delta), "Delta": Delta/(2*M), "tau_a": tau_a, "J": J, "alpha": alpha}
syn_vars = {"tau_s": tau_s}
node_vars["r0"] = np.sqrt(np.maximum(0.0, node_vars["eta"]))/np.pi

# simulation parameters
cutoff = 100.0
T = 4000.0 + cutoff
dt = 1e-3
dts = 1.0
noise_tau = 200.0
noise_scale = 0.02

# node and edge template initiation
node, node_op, syn_op = "qif_hom", "qif_hom_op", "syn_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")

# create network
edges = []
W0 = np.random.uniform(low=0.0, high=1.0, size=(M, M))
for i in range(M):
    for j in range(M):
        edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", None, {"weight": W0[i, j]}))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})

# generate run function
inp = np.zeros((int(T/dt), 1), dtype=np.float32)
func, args, arg_keys, _ = net.get_run_func(f"{syn}_hom_vectorfield", file_name=f"{syn}_hom_run",
                                           step_size=dt, backend="numpy", solver="heun", float_precision="float32",
                                           vectorize=True, inputs={f"all/{node_op}/I_ext": inp}, clear=False)
func_njit = njit(func)
func_njit(*args)
rhs = func_njit

# find argument positions of free parameters
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")
eta_idx = arg_keys.index(f"p0/{node_op}/eta")
args = list(args)

# define extrinsic input
noise = np.asarray(generate_colored_noise(int(T/dt), noise_tau, noise_scale), dtype=np.float32)
args[inp_idx] = noise

# set initial state
init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)

# define extrinsic input
noise = np.asarray(generate_colored_noise(int(T/dt), noise_tau, noise_scale), dtype=np.float32)

# run simulation
args[inp_idx] = noise
y0_hist, y0 = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)

# calculate network covariance eigenvalues
rs = y0_hist[:, :M] * 100.0
a = 1.0/(1.0 + np.exp(-y0_hist[:, 2*M:3*M]))
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

# plotting
##########

fig = plt.figure(figsize=(16, 5), layout="constrained")
grid = fig.add_gridspec(ncols=3, nrows=3)

# plotting dynamics
ax = fig.add_subplot(grid[0, :])
time = np.linspace(0.0, T, int(T/dts)) / 100.0
ax.plot(time, np.mean(rs, axis=1), label="average")
ax.plot(time, rs[:, 0], label="lowest FR neuron")
ax.plot(time, rs[:, -1], label="highest FR neuron")
ax.legend()
ax.set_ylabel(r"$r$ (Hz)")
ax.set_title("network dynamics")
ax = fig.add_subplot(grid[1, :])
ax.plot(time, np.mean(a, axis=1), label="average")
ax.plot(time, a[:, 0], label="lowest FR neuron")
ax.plot(time, a[:, -1], label="highest FR neuron")
ax.legend()
ax.set_ylabel(r"$a$ (Hz)")
ax.set_xlabel(r"$t$ (s)")

# plotting covariances
ax = fig.add_subplot(grid[2, 0])
im = ax.imshow(C, interpolation="none", aspect="auto", vmin=-1.0, vmax=1.0, cmap="berlin")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Network Covariance")
ax.set_xlabel("n")
ax.set_ylabel("n")

# plotting statistics
ax = fig.add_subplot(grid[2, 1])
ax.plot(eigvals)
ax.legend()
ax.set_xlabel(r"index")
ax.set_ylabel(r"$\lambda$")
ax.set_title("eigenvalues")
ax = fig.add_subplot(grid[2, 2])
ax.plot(ff)
ax.legend()
ax.set_xlabel(r"index")
ax.set_ylabel(r"$ff$")
ax.set_title("fano factors")

fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
fig.canvas.draw()
plt.show()

# clear files up
clear(net)
