from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, OperatorTemplate, clear
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from config.utility_functions import *

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
trials = 10
extinction_trials = 3
syn = "exc"
stp = "sd"
group = "stdp_asym"

# define stdp parameters
a = 1.0
a_r = 1.5
tau = 3.0
tau_r = 2.0
a_p = a*a_r
a_d = a/a_r
tau_p = tau
tau_d = tau*tau_r
tau_ratio = tau_p / tau_d
a_ratio = a_p / a_d if a_d > 0.0 else 1.0
stdp_ratio = tau_ratio*a_ratio

# set model parameters
M = 10
J = 20.0 / (0.5*M)
Delta = 2.0
eta = -0.8
b = 0.5
tau_s = 0.5
tau_a = 20.0
kappa = 0.1
node_vars = {"eta": uniform(M, eta, Delta), "Delta": Delta/(2*M)}
edge_vars = {"a_p": 0.0, "a_d": 0.0, "b": b}
syn_vars = {"tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}

# simulation parameters
cutoff = 50.0
T = 100.0
dt = 5e-4
dts = 0.1
dur = int(10.0/dt)
amp = 2.0
offset = -1.0
input_sequence = np.random.permutation(np.arange(M))

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_stdp_{stp}", "qif_op", f"syn_{stp}_op"
node_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/{node_op}")
syn_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/{syn_op}")
ltp_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/ltp_op")
ltd_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/ltd_op")
node_temp = NodeTemplate(name=node_op, operators=[node_op_temp, syn_op_temp, ltp_op_temp, ltd_op_temp])
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)
nodes = {f"p{i}": node_temp for i in range(M)}

# create network
edges = []
for i in range(M):
    for j in range(M):
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

# finalize network
net = CircuitTemplate(name=node, nodes=nodes, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})

# generate run function
inp = np.zeros((int(T/dt), M), dtype=np.float32) + offset
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
inp = np.zeros((int(T/dt), M)) + offset
for i, s in enumerate(input_sequence):
    inp[i*dur:(i+1)*dur, s] += amp
# fig, ax = plt.subplots(figsize=(16, 4))
# ax.imshow(inp.T, aspect='auto', interpolation='none')
# plt.show()

# set initial state
init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)

# run simulations
args[inp_idx] = inp
args[a_p_idx] = a_p
args[a_d_idx] = a_d
Ws, Rs, Us = [], [], []
for trial in range(trials):
    y_hist, y = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
    y_init = y_hist[-1]
    Ws.append(y[-int(M * M):].reshape(M, M))
    Rs.append(y_hist[:, 2*M:3*M])
    Us.append(y_hist[:, 4*M:5*M])
args[inp_idx] = inp - offset
for trial in range(extinction_trials):
    y_hist, y = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
    y_init = y_hist[-1]
    Ws.append(y[-int(M * M):].reshape(M, M))
    Rs.append(y_hist[:, 2*M:3*M])
    Us.append(y_hist[:, 4*M:5*M])

# plotting
##########

for trial in np.arange(0, trials + extinction_trials, step=2, dtype=np.int32):

    fig = plt.figure(figsize=(16, 4))
    grid = fig.add_gridspec(2, 4)
    fig.suptitle(f"Memory formation trial {trial}" if trial < trials else f"Memory extinction trial {trial - trials}")

    # plotting dynamics
    ax = fig.add_subplot(grid[0, :3])
    time = np.linspace(0.0, T, int(T/dts)) / 100.0
    ax.plot(time, Rs[trial] * 100.0)
    ax.set_ylabel(r"$r$")
    ax.set_title("network dynamics")
    ax = fig.add_subplot(grid[1, :3])
    ax.plot(time, Us[trial])
    ax.set_ylabel(r"$u$")
    ax.set_xlabel("time (s)")
    ax.set_title("network dynamics")

    # plotting weights
    ax = fig.add_subplot(grid[:, 3])
    ax.imshow(Ws[trial], interpolation="none", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Connectivity Weights")
    ax.set_xlabel("n")
    ax.set_ylabel("n")

    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
    fig.canvas.draw()

plt.show()

# clear files up
clear(net)

