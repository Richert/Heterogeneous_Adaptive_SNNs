import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

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

def integrate(y: np.ndarray, func, args, T, dt, dts, cutoff, N):

    steps = int(T / dt)
    cutoff_steps = int(cutoff / dt)
    store_step = int(dts / dt)
    store_steps = int((T - cutoff) / dts)
    state_rec = []
    N2 = int(N*N)

    # solve ivp with Heun's method
    for step in range(steps):
        if step > cutoff_steps and step % store_step == 0:
            state_rec.append(y[:-N2])
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
    idx = np.argsort(eigvals)
    return eigvals[idx], eigvecs[:, idx]

def get_ff(rates: np.ndarray) -> np.ndarray:
    n = rates.shape[1]
    ff = np.zeros((n,))
    for i in range(n):
        ff[i] = np.var(rates[:, i]) / np.mean(rates[:, i])
    return ff

# define data directory
path = "/home/richard/data/mpmf_simulations"

# read condition
trial = 0
syn = "exc"
stp = "sd"
group = "antioja"

# define stdp parameters
a_p = 0.005
a_d = 0.005
tau_p = 10.0
tau_d = 80.0

# set model parameters
M = 10
J = 20.0
Delta = 2.0
p = 1.0
eta = -0.9
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

# run initial simulation
y0_hist, y0 = integrate(args[1], rhs, tuple(args[2:]), T, dt, dts, cutoff, M)

# turn on synaptric plasticity and run simulation again
args[a_p_idx] = a_p
args[a_d_idx] = a_d
y1_hist, y1 = integrate(y0, rhs, tuple(args[2:]), T, dt, dts, cutoff, M)
W1 = y1[-int(M * M):].reshape(M, M)

# turn off synaptic plasticity and run simulation a final time
args[a_p_idx] = 0.0
args[a_d_idx] = 0.0
y2_hist, y2 = integrate(y1, rhs, tuple(args[2:]), T, dt, dts, cutoff, M)

# calculate in- and out-degrees
in_degree_pre = np.sum(W0, axis=1)
out_degree_pre = np.sum(W0, axis=0)
in_degree_post = np.sum(W1, axis=1)
out_degree_post = np.sum(W1, axis=0)

# calculate network covariance eigenvalues
r0, r1, r2 = y0_hist[:, :M], y1_hist[:, :M], y2_hist[:, :M]
eigvals_pre, eigvecs_pre = get_eigs(r0)
eigvals_post, eigvecs_post = get_eigs(r2)

# transform etas into covariance eigenvector space
etas = args[eta_idx]
etas_pre = np.dot(eigvecs_pre.T, etas)
etas_post = np.dot(eigvecs_post.T, etas)

# calculate fano factors
ff_pre = get_ff(r0)
ff_post = get_ff(r2)

# plotting weights
fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
ax = axes[0]
im = ax.imshow(W0, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0)
ax.set_title("Initial Weights")
ax = axes[1]
ax.imshow(W1, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0)
ax.set_title("Final Weights")

# plotting dynamics
fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
ax = axes[0]
ax.plot(np.mean(r0, axis=1), label="r0")
ax.plot(np.mean(r1, axis=1), label="r1")
ax.plot(np.mean(r2, axis=1), label="r2")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("r")
ax.set_title("firing rate")
ax = axes[1]
u = y1_hist[:, 4*M:5*M]
ax.plot(u)
ax.set_xlabel("time")
ax.set_ylabel("u")
ax.set_title("LTP trace variables")
ax = axes[2]
u = y1_hist[:, 5 * M:6 * M]
ax.plot(u)
ax.set_xlabel("time")
ax.set_ylabel("u")
ax.set_title("LTD trace variables")
plt.tight_layout()

# plotting DV relationships
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
ax = axes[0]
ax.plot(etas, in_degree_pre, color="royalblue", linestyle="dashed", label="in-degree (pre)")
ax.plot(etas, out_degree_pre, color="darkorange", linestyle="dashed", label="out-degree (pre)")
ax.plot(etas, in_degree_post, color="royalblue", linestyle="solid", label="in-degree (post)")
ax.plot(etas, out_degree_post, color="darkorange", linestyle="solid", label="out-degree (post)")
ax.legend()
ax.set_xlabel("eta")
ax.set_ylabel("degree")
ax.set_title("Nodal Connectivity")
ax = axes[1]
ax.plot(etas, ff_pre, label="pre")
ax.plot(etas, ff_post, label="post")
ax.legend()
ax.set_xlabel("eta")
ax.set_ylabel("fano factor")
ax.set_title("Nodal Dynamics")
ax = axes[2]
ax.scatter(etas_pre, np.log(eigvals_pre + 1e-12), label="pre")
ax.scatter(etas_post, np.log(eigvals_post + 1e-12), label="post")
ax.legend()
ax.set_xlabel("sum(eta*v)")
ax.set_ylabel("log(lambda)")
ax.set_title("Nodal Covariance")
plt.tight_layout()

plt.show()

# clear files up
clear(net)
