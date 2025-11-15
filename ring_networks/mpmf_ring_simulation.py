import numpy as np
import matplotlib.pyplot as plt
from rectipy import circular_connectivity
from pyrates import NodeTemplate, CircuitTemplate, EdgeTemplate, clear
import torch
from time import perf_counter
from scipy.stats import rv_discrete
from copy import deepcopy

def dist(x: int, method: str = "inverse", zero_val: float = 1.0, inverse_pow: float = 1.0) -> float:
    if method == "inverse":
        return 1/x**inverse_pow if x > 0 else zero_val
    if method == "exp":
        return np.exp(-x) if x > 0 else zero_val
    else:
        raise ValueError("Invalid method.")

# preparations
##############

# general parameters
numpy_precision = np.float32
torch_precision = torch.float32
device = "cpu"

# model parameters
node = "qif_stdp"
edge = "stdp"
delay_coupling = False
M = 50
p = 0.2
p_in = 0.2
conn_pow = 1.5
indices = np.arange(1, M+1)
node_vars = {"tau": 1.0, "J": 3.0, "eta": 0.0, "Delta": 1.0, "tau_s": 1.0, "tau_u": 50.0}
edge_vars = {"a": 0.005, "b": 1.0}

# input parameters
freq = 0.16
amp = 1.0
pow = 1.0
noise = 1.0

# simulation parameters
dt = 1e-3
dts = 1e-1
cutoff = 200.0
T = 1000.0 + cutoff
steps = int(T/dt)

# node and edge template initiation
edge_op = f"{edge}_op"
node_op = f"{node}_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}_edge")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(M, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=True)
# W[np.eye(M) > 0.1] = 1.0
edges = []
for i in range(M):
    for j in range(M):
        if W[i, j] > 1e-4:
            edge_tmp = deepcopy(edge_temp)
            edge_tmp.update_var(edge_op, "w", W[i, j])
            edges.append((f"p{j}/{node_op}/s", f"p{i}/{node_op}/s_in", edge_tmp,
                          {"weight": 1.0,
                           f"{edge}_edge/{edge_op}/r_s": f"p{j}/{node_op}/s",
                           f"{edge}_edge/{edge_op}/r_t": f"p{i}/{node_op}/s",
                           f"{edge}_edge/{edge_op}/x_s": f"p{j}/{node_op}/u",
                           f"{edge}_edge/{edge_op}/x_t": f"p{i}/{node_op}/u",
                           }))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(W, aspect="auto", interpolation="none")
# plt.show()

# simulation
##############

# define input
n_inputs = int(p_in*M)
center = int(M*0.5)
time = np.arange(0, steps) * dt
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
inp = noise * np.sqrt(dt) * np.random.randn(steps, M)
for idx in inp_indices:
    inp[:, idx] += amp * np.sin(2.0*np.pi*freq*time)**pow

print("Starting simulation...")
t0 = perf_counter()
obs = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, inputs={f"all/{node_op}/I_ext": inp},
              outputs={"s": f"all/{node_op}/s"}, solver="heun", clear=False, cutoff=cutoff)
t1 = perf_counter()
print(f'Finished network state collection after {t1 - t0} s.')

# extract synaptic weights
mapping, weights, deltas = net._ir["weight"].value, net.state["w"], net._ir["Delta"].value
idx = np.arange(M) #np.argsort(deltas)
W1 = np.zeros_like(W)
for i in idx:
    W1[i, W[i, :] > 1e-4] = weights[mapping[i, :] > 0.0]
clear(net)
# np.fill_diagonal(W, 0)


# plotting
##########

fig = plt.figure(figsize=(12, 8))
grid = fig.add_gridspec(nrows=3, ncols=2)
ax0 = fig.add_subplot(grid[0, :])
ax0.plot(inp[:, inp_indices[0]])
ax0.set_title('input')
ax0.set_xlabel('steps')
ax0.set_ylabel('amplitude')
ax1 = fig.add_subplot(grid[1, :])
ax1.imshow(obs["s"].T, aspect="auto", interpolation="none", cmap="Greys")
ax1.set_title('model dynamics')
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('s (Hz)')
ax2 = fig.add_subplot(grid[2, 0])
ax2.imshow(W, vmin=0.0, vmax=1.0, aspect="auto", interpolation="none")
ax2.set_title('initial weights')
ax2.set_xlabel('n')
ax2.set_ylabel('n')
ax3 = fig.add_subplot(grid[2, 1])
ax3.imshow(W1, vmin=0.0, vmax=1.0, aspect="auto", interpolation="none")
ax3.set_title('final weights')
ax3.set_xlabel('n')
ax3.set_ylabel('n')
plt.tight_layout()
plt.show()
