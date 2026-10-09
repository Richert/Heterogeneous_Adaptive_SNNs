import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from scipy.signal import welch
from config.utility_functions import *

def ssr(eta: np.ndarray, Delta: float):
    return np.sqrt(eta + np.sqrt(eta**2 + Delta**2))/(2*np.pi)

# define data directory
path = "/mnt/kennedy_labdata/richard_turbulence/data/mpmf_simulations"

# read condition
trial = 0
syn = "exc"
stp = "sd"

# define stdp parameters
c1 = -1.0
c2 = 0.01
c3 = 0.005

# set model parameters
M = 10
J = 30.0 / (0.5*M)
Delta = 2.0
p = 1.0
eta = -10.0
tau_s = 0.5
tau_a = 20.0
tau_u = 2.0
tau_z = 200.0
kappa = 0.1
alpha = 0.5
etas = uniform(M, eta, Delta)
r0 = np.linspace(0.05, 0.2, M)[::-1]
node_vars = {"eta": etas, "Delta": Delta/(2*M), "alpha": alpha, "tau_z": tau_z, "r0": r0}
edge_vars = {"c1": c1, "c2": c2, "c3": c3}
syn_vars = {"tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}

# simulation parameters
cutoff = 0.0
T = 2000.0 + cutoff
dt = 5e-4
dts = 1.0
noise_tau = 200.0
noise_scale = 0.02

# node and edge template initiation
edge, edge_op = "pitchfork_hom_edge", "pitchfork_hom_op"
node, node_op, syn_op = f"qif_hom", "qif_hom_op", f"syn_{stp}_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                      {"weight": J,
                       f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                       f"{edge}/{edge_op}/u_post": f"p{i}/ltp_op/u_p",
                       f"{edge}/{edge_op}/u_pre": f"p{j}/ltp_op/u_p",
                       f"{edge}/{edge_op}/z_post": f"p{i}/{node_op}/z",
                       }))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})
net.update_var(node_vars={f"all/ltp_op/tau_p": tau_u})

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
c_idx = arg_keys.index(f"{edge}/{edge_op}/c3")
eta_idx = arg_keys.index(f"p0/{node_op}/eta")
args = list(args)

# set random initial connectivity
W0 = np.random.uniform(low=0.0, high=1.0, size=(M, M))
args[1][-int(M*M):] = W0.reshape((int(M*M),))

# define extrinsic input
noise = np.asarray(generate_colored_noise(int(T/dt), noise_tau, noise_scale), dtype=np.float32)
args[inp_idx] = noise

# set initial state
init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)

# run initial simulation
args[inp_idx] = noise
y0_hist, y0 = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)

# turn on synaptric plasticity and run simulation again
args[c_idx] = c3
y1_hist, y1 = integrate(y0, rhs, tuple(args[2:]), T, dt, dts)
W1 = y1[-int(M*M):].reshape(M, M)

# turn off synaptic plasticity and run simulation a final time
args[c_idx] = 0.0
y2_hist, y2 = integrate(y1, rhs, tuple(args[2:]), T, dt, dts)

# calculate in- and out-degrees
in_degree_pre = np.sum(W0, axis=1)
out_degree_pre = np.sum(W0, axis=0)
in_degree_post = np.sum(W1, axis=1)
out_degree_post = np.sum(W1, axis=0)

# calculate network covariance eigenvalues
r0, r1, r2 = y0_hist[:, :M], y1_hist[:, :M], y2_hist[:, :M]
eigvals_pre, eigvecs_pre, C_pre = get_eigs(r0)
eigvals_post, eigvecs_post, C_post = get_eigs(r2)
C_pre[np.eye(M) > 0.0] = 0.0
C_post[np.eye(M) > 0.0] = 0.0

# transform etas into covariance eigenvector space
etas = args[eta_idx]
etas_pre = np.dot(eigvecs_pre.T, etas)
etas_post = np.dot(eigvecs_post.T, etas)

# get PSD of first PC
pc1_pre, pc1_post = np.dot(r0*100.0, eigvecs_pre[:, 0]), np.dot(r2, eigvecs_post[:, 0])
fs_pre, ps_pre = welch(pc1_pre, fs=100.0/dts, nperseg=512)
fs_post, ps_post = welch(pc1_post, fs=100.0/dts, nperseg=512)
f_max_pre, f_max_post = fs_pre[np.argmax(ps_pre)], fs_post[np.argmax(ps_post)]
pow_pre = (fs_pre[1] - fs_pre[0]) * np.sum(ps_pre)
pow_post = (fs_post[1] - fs_post[0]) * np.sum(ps_post)

# calculate fano factors
ff_pre = get_ff(r0)
ff_post = get_ff(r2)

# report some basic stats
#########################

print(f"Neuron type: {syn}")
print(f"STP type: {stp}")

# plotting
##########

fig = plt.figure(figsize=(16, 5), layout="constrained")
grid = fig.add_gridspec(ncols=5, nrows=6)

# plotting dynamics
ax = fig.add_subplot(grid[:2, :2])
time = np.linspace(0.0, T, int(T/dts)) / 100.0
ax.plot(time, np.mean(r0, axis=1)*100.0, label="T0: no plasticity")
ax.plot(time, np.mean(r1, axis=1)*100.0, label="T1: plasticity")
ax.plot(time, np.mean(r2, axis=1)*100.0, label="T2: no plasticity")
ax.legend()
ax.set_ylabel(r"$r$ (Hz)")
ax.set_title("network dynamics")
ax = fig.add_subplot(grid[2:4, :2])
z = y1_hist[:, 2*M:3*M]
ax.plot(time, z)
ax.set_ylabel(r"$z$")
ax = fig.add_subplot(grid[4:6, :2])
x = y1_hist[:, -2*int(M*M):-int(M*M)]
ax.plot(time, x)
ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"$x$")

# plotting weights
ax = fig.add_subplot(grid[:3, 2])
im = ax.imshow(W0, interpolation="none", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Connectivity Weights")
ax.set_xlabel("n")
ax.set_ylabel("n")
ax = fig.add_subplot(grid[3:, 2])
im = ax.imshow(W1, interpolation="none", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
ax.set_xlabel("n")
ax.set_ylabel("n")
plt.colorbar(im, ax=ax, shrink=0.8)

# plotting covariances
c_max = np.max(C_post)
ax = fig.add_subplot(grid[:3, 3])
im = ax.imshow(C_pre, interpolation="none", aspect="auto", vmin=-c_max, vmax=c_max, cmap="berlin")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Network Covariance")
ax.set_xlabel("n")
ax.set_ylabel("n")
ax = fig.add_subplot(grid[3:, 3])
im = ax.imshow(C_post, interpolation="none", aspect="auto", vmin=-c_max, vmax=c_max, cmap="berlin")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel("n")
ax.set_ylabel("n")

# plotting DV relationships
ax = fig.add_subplot(grid[:3, 4])
ax.plot(etas, in_degree_pre, color="royalblue", linestyle="dashed", label=r"$w_{in}$ (T0)")
ax.plot(etas, out_degree_pre, color="darkorange", linestyle="dashed", label=r"$w_{out}$ (T0)")
ax.plot(etas, in_degree_post, color="royalblue", linestyle="solid", label=r"$w_{in}$ (T2)")
ax.plot(etas, out_degree_post, color="darkorange", linestyle="solid", label=r"$w_{out}$ (T2)")
ax.legend()
ax.set_xlabel(r"$\eta$")
ax.set_ylabel(r"$w_{in/out}$")
ax.set_title("Network Statistics")
ax = fig.add_subplot(grid[3:, 4])
ax.bar(np.arange(0, M), eigvals_pre, label="T0", alpha=0.5)
ax.bar(np.arange(0, M), eigvals_post, label="T2", alpha=0.5)
ax.legend()
ax.set_xlabel(r"eigenvalue index")
ax.set_ylabel(r"$\lambda$")

fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
fig.canvas.draw()
plt.show()

# clear files up
clear(net)
