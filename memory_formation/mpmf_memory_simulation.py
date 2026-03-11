from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, OperatorTemplate, clear
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from config.utility_functions import *

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
learn_trials = 10
extinction_trials = 2
syn = "exc"
stp = "sd"
group = "stdp_asym"

# define stdp parameters
a = 0.1
a_r = 1.5
tau = 3.0
tau_r = 2.0
a_p = a*a_r
a_d = a/a_r
tau_p = tau
tau_d = tau*tau_r
tau_ratio = tau_p / tau_d
a_ratio = a_p / a_d
stdp_ratio = tau_ratio*a_ratio

# set within-cluster model parameters
clusters = 4
N = 10
M = clusters * N
Delta_exc = 2.0
eta_exc = -0.85
tau_exc = 1.0
tau_s_exc = 0.5
eta_inh = -1.0
Delta_inh = 0.05
tau_inh = 0.3
J_ee_in = 20.0 / (0.5*N)
J_ie_in = 20.0 / (0.5*N)
J_ei_in = -20.0
node_vars = {"eta": np.asarray(uniform(N, eta=eta_exc, Delta=Delta_exc).tolist() + [eta_inh]),
             "Delta": np.asarray([Delta_exc/(2*N)]*N + [Delta_inh]),
             "tau": np.asarray([tau_exc]*N + [tau_inh])}
edge_vars = {"a_p": 0.0, "a_d": 0.0, "b": 0.5}

# set between-cluster model parameters
J_ee_out = 120.0 / (0.5*M)
J_ie_out = 40.0 / (0.5*M)

# simulation parameters
cutoff = 50.0
T = 50.0
dt = 5e-4
dts = 0.1
dur = int(5.0/dt)
amp = 2.0
offset = -1.0
input_sequence = np.random.permutation(np.arange(clusters))

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_exc_op, syn_inh_op = f"qif_stdp_{stp}", "qif_op", f"syn_{stp}_op", f"syn_op"
node_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/{node_op}")
syn_exc_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/{syn_exc_op}")
syn_inh_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/{syn_inh_op}")
ltp_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/ltp_op")
ltd_op_temp = OperatorTemplate.from_yaml(f"../config/fre_equations/ltd_op")
exc_temp = NodeTemplate(name=node_op, operators=[node_op_temp, syn_exc_temp, ltp_op_temp, ltd_op_temp])
inh_temp = NodeTemplate(name=node_op, operators=[node_op_temp, syn_inh_temp])
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create node list for local cluster
cluster_nodes = {f"exc_{i}": exc_temp for i in range(N)}
cluster_nodes["inh"] = inh_temp

# create list of cluster edges
cluster_edges = []
for i in range(N):
    for j in range(N):
        cluster_edges.append((f"exc_{j}/{syn_exc_op}/s", f"exc_{i}/{node_op}/s_in", None, {"weight": J_ee_in}))
    cluster_edges.append((f"exc_{i}/{syn_exc_op}/s", f"inh/{node_op}/s_in", None, {"weight": J_ie_in}))
    cluster_edges.append((f"inh/{syn_inh_op}/s", f"exc_{i}/{node_op}/s_in", None, {"weight": J_ei_in}))

# create cluster network
net_c = CircuitTemplate(name="cluster", nodes=cluster_nodes, edges=cluster_edges)
net_c.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# create full network
edges = []
for i in range(clusters):
    for j in range(clusters):
        if i != j:
            for n in range(N):
                for m in range(N):
                    if group == "stdp_asym":
                        edges.append((f"p{j}/exc_{m}/{syn_exc_op}/s", f"p{i}/exc_{n}/{node_op}/s_in", deepcopy(edge_temp),
                                      {"weight": J_ee_out,
                                       f"{edge}/{edge_op}/s_in": f"p{j}/exc_{m}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/p1": f"p{i}/exc_{n}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/p2": f"p{j}/exc_{m}/ltp_op/u_p",
                                       f"{edge}/{edge_op}/d1": f"p{j}/exc_{m}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/d2": f"p{i}/exc_{n}/ltd_op/u_d",
                                       }))
                    elif group == "stdp_sym":
                        edges.append((f"p{j}/exc_{m}/{syn_exc_op}/s", f"p{i}/exc_{n}/{node_op}/s_in", deepcopy(edge_temp),
                                      {"weight": J_ee_out,
                                       f"{edge}/{edge_op}/s_in": f"p{j}/exc_{m}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/p1": f"p{j}/exc_{m}/ltp_op/u_p",
                                       f"{edge}/{edge_op}/p2": f"p{i}/exc_{n}/ltp_op/u_p",
                                       f"{edge}/{edge_op}/d1": f"p{i}/exc_{n}/ltd_op/u_d",
                                       f"{edge}/{edge_op}/d2": f"p{j}/exc_{m}/ltd_op/u_d",
                                       }))
                    elif group == "antihebbian":
                        edges.append((f"p{j}/exc_{m}/{syn_exc_op}/s", f"p{i}/exc_{n}/{node_op}/s_in", deepcopy(edge_temp),
                                      {"weight": J_ee_out,
                                       f"{edge}/{edge_op}/s_in": f"p{j}/exc_{m}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/p1": f"p{j}/exc_{m}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/p2": f"p{i}/exc_{n}/ltp_op/u_p",
                                       f"{edge}/{edge_op}/d1": f"p{i}/exc_{n}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/d2": f"p{j}/exc_{m}/ltd_op/u_d",
                                       }))
                    elif group == "oja":
                        edges.append((f"p{j}/exc_{m}/{syn_exc_op}/s", f"p{i}/exc_{n}/{node_op}/s_in", deepcopy(edge_temp),
                                      {"weight": J_ee_out,
                                       f"{edge}/{edge_op}/s_in": f"p{j}/exc_{m}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/p1": f"p{j}/exc_{m}/ltp_op/u_p",
                                       f"{edge}/{edge_op}/p2": f"p{i}/exc_{n}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/d1": f"p{i}/exc_{n}/{syn_exc_op}/s",
                                       f"{edge}/{edge_op}/d2": f"p{i}/exc_{n}/ltd_op/u_d",
                                       }))
                    # elif group == "antioja":
                    #     edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                    #                   {"weight": J,
                    #                    f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                    #                    f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                    #                    f"{edge}/{edge_op}/p2": f"p{j}/{syn_op}/s",
                    #                    f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                    #                    f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                    #                    }))
                    else:
                        raise ValueError(f"Unknown group {group}")

                edges.append((f"p{j}/exc_{n}/{syn_exc_op}/s", f"p{i}/inh/{node_op}/s_in", None,
                              {"weight": J_ie_out}))

# finalize network
net = CircuitTemplate(name=node, circuits={f"p{i}": net_c for i in range(clusters)}, edges=edges)
net.update_var(node_vars={})

# generate run function
inp = np.zeros((int(T/dt), clusters*(N+1)), dtype=np.float32) + offset
func, args, arg_keys, _ = net.get_run_func(f"{syn}_{stp}_vectorfield", file_name=f"{syn}_{stp}_run",
                                           step_size=dt, backend="numpy", solver="heun", float_precision="float32",
                                           vectorize=True, inputs={f"all/all/{node_op}/I_ext": inp}, clear=False)
func_njit = njit(func)
func_njit(*args)
rhs = func_njit

# find argument positions of free parameters
inp_idx = arg_keys.index(f"input_lvl_0/I_ext_input_node/I_ext_input_op/I_ext_input")
a_p_idx = arg_keys.index(f"{edge}/{edge_op}/a_p")
a_d_idx = arg_keys.index(f"{edge}/{edge_op}/a_d")
tau_p_idx = arg_keys.index(f"p0/exc_0/ltp_op/tau_p")
tau_d_idx = arg_keys.index(f"p0/exc_0/ltd_op/tau_d")
eta_idx = arg_keys.index(f"p0/exc_0/{node_op}/eta")

# set LTP/LTD time constants
args = list(args)
args[tau_p_idx] = tau_p
args[tau_d_idx] = tau_d

# set random initial connectivity
W0 = np.random.uniform(low=0.01, high=0.1, size=(M, M))
for i in range(clusters):
    W0[i*N:(i+1)*N, i*N:(i+1)*N] = 0.0
conn_idx = np.argwhere(W0 > 0).squeeze()
args[1][-len(conn_idx):] = W0[conn_idx[:, 0], conn_idx[:, 1]]

# define extrinsic input
inp_t = np.zeros_like(inp) + offset
for i, s in enumerate(input_sequence):
    inp_t[i*dur:(i+1)*dur, s*(N+1):(s+1)*(N+1)-1] += amp
    # inp_t[:, (s+1)*(N+1)-1] += 0.5
fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(inp_t[::int(dts/dt)].T, aspect='auto', interpolation='none')
ax.set_title("teacher input pattern")
# plt.show()

# set initial state
init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)

# learning phase
args[inp_idx] = inp_t
args[a_p_idx] = a_p
args[a_d_idx] = a_d
Ws, REs, RIs, Is = [], [], [], []
for trial in range(learn_trials):
    y_hist, y = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
    y_init = y_hist[-1]
    if trial == 0 or trial == learn_trials - 1:
        W = np.zeros_like(W0)
        W[conn_idx[:, 0], conn_idx[:, 1]] = y[-len(conn_idx):]
        Ws.append(W)
        REs.append(y_hist[:, 2*M:3*M])
        RIs.append(y_hist[:, 6*M+2*clusters:6*M+3*clusters])
        Is.append(inp_t[::int(dts/dt)])

# recall phase
inp_r = np.zeros_like(inp_t) + offset
inp_r[:dur, input_sequence[0]*(N+1):(input_sequence[0]+1)*(N+1)-1] += amp
args[inp_idx] = inp_r
args[a_p_idx] = 0.0
args[a_d_idx] = 0.0
y_hist, y = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
y_init = y_hist[-1]
W = np.zeros_like(W0)
W[conn_idx[:, 0], conn_idx[:, 1]] = y[-len(conn_idx):]
Ws.append(W)
REs.append(y_hist[:, 2*M:3*M])
RIs.append(y_hist[:, 6*M+2*clusters:6*M+3*clusters])
Is.append(inp_r[::int(dts/dt)])

# extinction phase
args[inp_idx] = inp - offset
args[a_p_idx] = a_p
args[a_d_idx] = a_d
for trial in range(extinction_trials):
    y_hist, y = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
    y_init = y_hist[-1]
    if trial == 0 or trial == extinction_trials - 1:
        W = np.zeros_like(W0)
        W[conn_idx[:, 0], conn_idx[:, 1]] = y[-len(conn_idx):]
        Ws.append(W)
        REs.append(y_hist[:, 2*M:3*M])
        RIs.append(y_hist[:, 6 * M + 2 * clusters:6 * M + 3 * clusters])
        Is.append(inp[::int(dts/dt)] - offset)

# plotting
##########

for trial in np.arange(len(Ws)):

    fig = plt.figure(figsize=(16, 4))
    grid = fig.add_gridspec(2, 4)
    if trial == 0:
        fig.suptitle(f"Initial memory formation trial")
    elif trial == 1:
        fig.suptitle(f"Final memory formation trial")
    elif trial == 2:
        fig.suptitle(f"Memory recall trial")
    elif trial == 3:
        fig.suptitle(f"Initial memory extinction trial")
    else:
        fig.suptitle(f"Final memory extinction trial")

    # plotting dynamics
    ax = fig.add_subplot(grid[0, :3])
    time = np.linspace(0.0, T, int(T/dts)) / 100.0
    for c in range(clusters):
        ax.plot(time, np.mean(REs[trial][:, c*N:(c+1)*N], axis=1) * 100.0, label=f"p{c}")
    ax.legend()
    ax.set_ylabel(r"$r_e$")
    ax.set_title("network dynamics")
    ax = fig.add_subplot(grid[1, :3])
    for c in range(clusters):
        ax.plot(time, RIs[trial][:, c] * 100.0, label=f"p{c}")
    ax.legend()
    ax.set_ylabel(r"$r_i$")
    ax.set_xlabel("time (s)")

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
