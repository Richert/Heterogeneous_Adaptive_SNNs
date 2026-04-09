from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from scipy.signal import welch
from config.utility_functions import *
np.random.seed(42)

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
trial = 0
syn = "inh"
group = "oja"

# define stdp parameters
a = 0.01
a_r = 1.5
tau = 2.0
tau_r = 2.0
a_p = a*a_r
a_d = a/a_r
tau_p = tau
tau_d = tau*tau_r
tau_ratio = tau_p / tau_d
a_ratio = a_p / a_d
stdp_ratio = tau_ratio*a_ratio

# set model parameters
M = 50
J = -10.0 / (0.5*M)
Delta = 1.0
p = 1.0
eta = 0.0
b = 0.5
tau_s = 1.0
node_vars = {"eta": uniform(M, eta, Delta), "Delta": Delta/(2*M)}
edge_vars = {"a_p": 0.0, "a_d": 0.0, "b": b}
syn_vars = {"tau_s": tau_s}

# simulation parameters
cutoff = 100.0
T = 2000.0
dt = 1e-3
dts = 1.0
inp_amp = 3.0
inp_freq = 0.005
inp_dur = 5.0

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_stdp", "qif_op", f"syn_op"
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
                          {"weight": J,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{i}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{i}/ltd_op/u_d",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        elif group == "antihebbian":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": J,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p2": f"p{i}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        elif group == "oja":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": J,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{i}/ltd_op/u_d",
                           }))
        elif group == "antioja":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": J,
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
func, args, arg_keys, _ = net.get_run_func(f"{syn}_vectorfield", file_name=f"{syn}_run",
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

# set initial state
init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)

# generate intrinsic input
steps = int(T/dt)
inp = np.zeros((steps,))
period = int(1.0/(inp_freq*dt))
dur = int(inp_dur/dt)
step = 0
while step < steps:
    inp[step:step+dur] += inp_amp
    step += period
# plt.plot(inp)
# plt.show()
args[inp_idx] = inp

# run initial simulation
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
w = y1_hist[:, -int(M*M):]
w = w.reshape(w.shape[0], M, M)
ax.plot(time, np.sum(w, axis=2))
ax.set_ylabel(r"$w_{in}$")
ax = fig.add_subplot(grid[4:6, :2])
ax.plot(time, np.sum(w, axis=1))
ax.set_xlabel(r"$t$ (s)")
ax.set_ylabel(r"$w_{out}$")

# plotting weights
ax = fig.add_subplot(grid[:3, 2])
im = ax.imshow(W0, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Connectivity Weights")
ax.set_xlabel("n")
ax.set_ylabel("n")
ax = fig.add_subplot(grid[3:, 2])
im = ax.imshow(W1, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
ax.set_xlabel("n")
ax.set_ylabel("n")
plt.colorbar(im, ax=ax, shrink=0.8)

# plotting covariances
ax = fig.add_subplot(grid[:3, 3])
im = ax.imshow(C_pre, interpolation="none", aspect="auto", vmin=-1.0, vmax=1.0, cmap="berlin")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Network Covariance")
ax.set_xlabel("n")
ax.set_ylabel("n")
ax = fig.add_subplot(grid[3:, 3])
im = ax.imshow(C_post, interpolation="none", aspect="auto", vmin=-1.0, vmax=1.0, cmap="berlin")
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
