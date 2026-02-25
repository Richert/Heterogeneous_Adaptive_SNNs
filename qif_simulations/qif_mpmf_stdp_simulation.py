import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from scipy.signal import welch

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

def uniform(N: int, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.linspace(-0.5, 0.5, N)

@njit
def integrate_noise(x, inp, scale, tau):
    return x + scale * inp - x / tau

def generate_colored_noise(num_samples, tau, scale=1.0):
    """
    Generates Brownian noise by integrating white noise.

    Args:
        num_samples (int): The number of samples in the output Brownian noise.
        scale (float): A scaling factor for the noise amplitude.

    Returns:
        numpy.ndarray: An array containing the generated Brownian noise.
    """
    white_noise = np.random.randn(num_samples)
    x = 0.0
    colored_noise = np.zeros_like(white_noise)
    for sample in range(num_samples):
        x = integrate_noise(x, white_noise[sample], scale, tau)
        colored_noise[sample] = x
    return colored_noise

def integrate(y: np.ndarray, func, args, T, dt, dts):

    steps = int(T / dt)
    store_step = int(dts / dt)
    state_rec = []

    # solve ivp with Heun's method
    for step in range(steps):
        if step % store_step == 0:
            state_rec.append(y[:])
        rhs = func(step, y, *args)
        y_0 = y + dt * rhs
        y = y + (rhs + func(step, y_0, *args)) * dt/2

    return np.asarray(state_rec), y

def get_eigs(rates: np.ndarray, epsilon: float = 1e-12) -> tuple:

    rates_centered = np.zeros_like(rates)
    for i in range(rates.shape[1]):
        rates_centered[:, i] = rates[:, i] - np.mean(rates[:, i])
        rates_centered[:, i] /= (np.std(rates[:, i]) + epsilon)
    C = np.cov(rates_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx], C

def get_ff(rates: np.ndarray) -> np.ndarray:
    n = rates.shape[1]
    ff = np.zeros((n,))
    for i in range(n):
        ff[i] = np.var(rates[:, i]) / np.mean(rates[:, i])
    return ff

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
trial = 0
syn = "exc"
stp = "sf"
group = "antihebbian"

# define stdp parameters
a = 0.01
a_r = 2.0
tau = 30.0
tau_r = 2.0
a_p = a*a_r
a_d = a/a_r
tau_p = tau
tau_d = tau*tau_r
tau_ratio = tau_p / tau_d
a_ratio = a_p / a_d
stdp_ratio = tau_ratio*a_ratio

# set model parameters
M = 10
J = 20.0
Delta = 2.0
p = 1.0
eta = -0.85
b = 0.5
tau_s = 0.5
tau_a = 20.0
kappa = 0.1
node_vars = {"eta": uniform(M, eta, Delta), "Delta": Delta/(2*M), "J": J/(0.5*M)}
edge_vars = {"a_p": 0.0, "a_d": 0.0, "b": b}
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
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        if group == "stdp_asym":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/p2": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{i}/ltd_op/u_d",
                           }))
        elif group == "stdp_sym":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{i}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{i}/ltd_op/u_d",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        elif group == "antihebbian":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p2": f"p{i}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        elif group == "oja":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{i}/ltd_op/u_d",
                           }))
        elif group == "antioja":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        else:
            raise ValueError(f"Unknown group {group}")
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
a_p_idx = arg_keys.index(f"{edge}/{edge_op}/a_p")
a_d_idx = arg_keys.index(f"{edge}/{edge_op}/a_d")
tau_p_idx = arg_keys.index(f"p0/ltp_op/tau_p")
tau_d_idx = arg_keys.index(f"p0/ltd_op/tau_d")
eta_idx = arg_keys.index(f"p0/{node_op}/eta")

# set LTP/LTD time constants
args = list(args)
args[tau_p_idx] = tau_p
args[tau_d_idx] = tau_d

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
args[a_p_idx] = a_p
args[a_d_idx] = a_d
y1_hist, y1 = integrate(y0, rhs, tuple(args[2:]), T, dt, dts)
W1 = y1[-int(M * M):].reshape(M, M)

# turn off synaptic plasticity and run simulation a final time
args[a_p_idx] = 0.0
args[a_d_idx] = 0.0
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
print(f"Plasticity rule: {group}")
print(f"Log LTP/LTD ratio: {np.log(stdp_ratio)}")

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
w = y1_hist[:, 6*M:]
w = w.reshape(w.shape[0], M, M)
ax.plot(time, np.sum(w, axis=2))
ax.set_ylabel(r"$w_{in}$")
ax = fig.add_subplot(grid[4:6, :2])
ax.plot(time, np.sum(w, axis=1))
ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"$w_{out}$")

# plotting weights
ax = fig.add_subplot(grid[:3, 2])
ax.imshow(W0, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
ax.set_title("Connectivity Weights")
ax.set_xlabel("n")
ax.set_ylabel("n")
ax = fig.add_subplot(grid[3:, 2])
ax.imshow(W1, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
ax.set_xlabel("n")
ax.set_ylabel("n")

# plotting covariances
ax = fig.add_subplot(grid[:3, 3])
ax.imshow(C_pre, interpolation="none", aspect="auto", vmin=-1.0, vmax=1.0, cmap="managua")
ax.set_title("Network Covariance")
ax.set_xlabel("n")
ax.set_ylabel("n")
ax = fig.add_subplot(grid[3:, 3])
ax.imshow(C_post, interpolation="none", aspect="auto", vmin=-1.0, vmax=1.0, cmap="managua")
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
fig.savefig("/home/rgast/data/qif_plasticity/exc_stdp_dynamics.png", dpi=300)
plt.show()

# clear files up
clear(net)
