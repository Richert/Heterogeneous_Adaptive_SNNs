from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = "sans"
from config.utility_functions import *

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
trial = 0
syn = "exc"
stp = "sd"

# set model parameters
M = 50
Delta = 2.0
p = 1.0
eta = -1.0
b = 0.5
tau_s = 1.0
tau_a = 40.0
kappa = 0.1
etas = uniform(M, eta, Delta)
Delta2 = Delta/(2*M)
c0 = 5.0
c1 = 20.0
c2 = 1.0
c3 = 1.0
node_vars = {"eta": etas, "Delta": Delta2}
syn_vars = {"tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}

# simulation parameters
T = 50.0
dt = 2e-5
dts = 1e-2
sr = int(dts/dt)
stim_amp = 3.0
stim_dur = int(1.0/dt)

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
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(W, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
ax.set_title("Connectivity")
plt.tight_layout()

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
inp = np.zeros((int(T/dt), 1))
inp[:stim_dur, :] = stim_amp
args[inp_idx] = np.asarray(inp, dtype=np.float32)
y_init = np.asarray(y0, dtype=np.float32)
y_init[:M] = np.abs(y_init[:M])
y, _ = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
signal = y[:, :M]

# plot results
_, axes = plt.subplots(nrows=2, figsize=(12, 5))
ax = axes[0]
time = np.linspace(0, T, signal.shape[0]) * 10.0
ax.plot(time, np.mean(signal, axis=1)*100.0/tau_s)
ax.set_ylabel("mean(r)")
ax.set_title(f"Network dynamics")
ax.set_xlim(time[0], time[-1])
ticks = list(ax.get_xticks())
labels = list(ax.get_xticklabels())
ax = axes[1]
im = ax.imshow(signal.T, aspect="auto", interpolation="none")
# plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel('time (s)')
ax.set_ylabel('neurons')
ax.set_xticks([np.argmin(np.abs(time-t)) for t in ticks], labels=labels)
plt.tight_layout()

# clear files up
plt.show()
clear(net)
