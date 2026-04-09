import matplotlib.pyplot as plt
from pyrates import NodeTemplate, CircuitTemplate, EdgeTemplate, OperatorTemplate, clear
from copy import deepcopy
import matplotlib
matplotlib.use('tkagg')
from config.utility_functions import *

# preparations
##############

# read condition
syn = "exc"
stp = "sd"
group = "stdp_asym"

# define stdp parameters
a = 0.01
a_r = 1.2
tau = 1.0
tau_r = 2.0
a_p = a*a_r
a_d = a/a_r
tau_p = tau
tau_d = tau*tau_r
# tau_ratio = tau_p / tau_d
# a_ratio = a_p / a_d
# stdp_ratio = tau_ratio*a_ratio

# set model parameters
M = 10
J = 15.0
Delta = 1.5
eta = -1.5
eta2 = 0.1
Delta2 = 0.05
b = 0.5
tau_s = 1.0
tau_a = 10.0
kappa = 2.0
indices = np.arange(1, M+1)
node_vars = {"eta": uniform(M, eta, Delta), #uniform(M, eta, Delta)
             "Delta": uniform(M, eta2, Delta2), #Delta/(2*M)
             }
edge_vars = {"a_p": 0.0, "a_d": 0.0, "b": b}
syn_vars = {"tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}

# simulation parameters
cutoff = 100.0
T = 2000.0
dt = 1e-3
dts = 1.0
freq = 0.002
amp = 2.0
pow = 20.0
noise_lvl = 3.0
noise_tau = 10.0

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_stdp_{stp}", "qif_op", f"alpha_{stp}_op"
node_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/{node_op}")
syn_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/{syn_op}")
ltp_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/ltp_op")
ltd_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/ltd_op")
node_temp = NodeTemplate(name=node_op, operators=[node_op_temp, syn_op_temp, ltp_op_temp, ltd_op_temp])
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create coupling matrix
W0 = np.zeros((M, M))
for i in range(M):
    start = i-1 if i > 0 else 0
    stop = i+2 if i < M-1 else M
    W0[i, start:stop] = np.random.uniform(0.1, 0.9, size=(stop-start,))

# create network
edges = []
for i in range(M):
    for j in range(M):
        if W0[i, j] > 0.0:
            if group == "stdp_asym":
                edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                              {"weight": J,
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

# define input
steps = int(T/dt)
time = np.arange(0, steps) * dt
inp = np.zeros((steps, M))
inp_indices = [0]
for i in range(M):
    noise = generate_colored_noise(steps, noise_tau, noise_lvl)
    inp[:, i] = noise
    if i in inp_indices:
        inp[:, i] += amp * np.sin(2.0*np.pi*freq*time)**pow

# fig, ax = plt.subplots(figsize=(12, 3))
# ax.plot(inp[:, inp_indices[0]])
# plt.show()

# generate run function
func, args, arg_keys, _ = net.get_run_func(f"{syn}_{stp}_vectorfield", file_name=f"{syn}_{stp}_run",
                                           step_size=dt, backend="numpy", solver="heun", float_precision="float32",
                                           vectorize=True, inputs={f"all/{node_op}/I_ext": np.zeros_like(inp)},
                                           clear=False)
func_njit = njit(func)
func_njit(*args)
rhs = func_njit

# find argument positions of free parameters
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")
a_p_idx = arg_keys.index(f"{edge}/{edge_op}/a_p")
a_d_idx = arg_keys.index(f"{edge}/{edge_op}/a_d")

# set connectivity
args = list(args)
conn_indices = np.argwhere(W0 > 0.0).squeeze()
args[1][-conn_indices.shape[0]:] = W0[conn_indices[:, 0], conn_indices[:, 1]]

# simulation
##############

# set initial state
init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)

# run initial simulation
args[inp_idx] = inp
y0_hist, y0 = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)

# turn on synaptric plasticity and run simulation again
args[a_p_idx] = a_p
args[a_d_idx] = a_d
y1_hist, y1 = integrate(y0, rhs, tuple(args[2:]), T, dt, dts)
W1 = np.zeros_like(W0)
W1[conn_indices[:, 0], conn_indices[:, 1]] = y1[-conn_indices.shape[0]:]

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

# plotting
##########

fig = plt.figure(figsize=(16, 5), layout="constrained")
grid = fig.add_gridspec(ncols=5, nrows=6)

# plotting dynamics
s, a, w_tmp = y1_hist[:, 2*M:3*M], y1_hist[:, 4*M:5*M], y1_hist[:, -len(conn_indices):]
w_step = 5
ax = fig.add_subplot(grid[:2, :2])
time = np.linspace(0.0, T, int(T/dts)) / 100.0
# ax.plot(time, np.mean(r0, axis=1)*100.0, label="T0: no plasticity")
# ax.plot(time, np.mean(r1, axis=1)*100.0, label="T1: plasticity")
ax.plot(time, s) #label="T2: no plasticity"
# ax.legend()
ax.set_ylabel(r"$s$")
ax.set_title("network dynamics")
ax = fig.add_subplot(grid[2:4, :2])
ax.plot(time, a)
ax.set_ylabel(r"$a$")
ax = fig.add_subplot(grid[4:6, :2])
w = np.zeros((w_tmp.shape[0], M, M))
w[:, conn_indices[:, 0], conn_indices[:, 1]] = w_tmp
ax.plot(time, np.sum(w, axis=2)[:, ::w_step])
ax.set_ylabel(r"$w_{in}$")
ax.set_xlabel("time (s)")

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
ax.imshow(C_pre, interpolation="none", aspect="auto", vmin=-1.0, vmax=1.0, cmap="managua_r")
ax.set_title("Network Covariance")
ax.set_xlabel("n")
ax.set_ylabel("n")
ax = fig.add_subplot(grid[3:, 3])
ax.imshow(C_post, interpolation="none", aspect="auto", vmin=-1.0, vmax=1.0, cmap="managua_r")
ax.set_xlabel("n")
ax.set_ylabel("n")

# plotting DV relationships
etas = node_vars["eta"]
ax = fig.add_subplot(grid[:3, 4])
idx = np.argsort(etas)
ax.plot(etas[idx], in_degree_pre[idx], color="royalblue", linestyle="dashed", label=r"$w_{in}$ (T0)")
ax.plot(etas[idx], out_degree_pre[idx], color="darkorange", linestyle="dashed", label=r"$w_{out}$ (T0)")
ax.plot(etas[idx], in_degree_post[idx], color="royalblue", linestyle="solid", label=r"$w_{in}$ (T2)")
ax.plot(etas[idx], out_degree_post[idx], color="darkorange", linestyle="solid", label=r"$w_{out}$ (T2)")
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