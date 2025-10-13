import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
from numba import njit

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

# parameters
M = 50
p = 1.0
edge_vars = {
    "a": 0.1, "b": 0.1
}
Delta = 0.5
eta = -1.05
# etas = eta + Delta * np.linspace(-0.5, 0.5, num=M)
indices = np.arange(1, M+1)
etas = eta + Delta*np.tan(0.5*np.pi*(2*indices-M-1)/(M+1))
deltas = Delta*(np.tan(0.5*np.pi*(2*indices-M-0.5)/(M+1))-np.tan(0.5*np.pi*(2*indices-M-1.5)/(M+1)))
node_vars = {"tau": 1.0, "J": 30.0 / (p*M), "eta": etas, "tau_u": 30.0, "tau_s": 1.0, "Delta": deltas,
             "tau_a": 20.0, "kappa": 0.2, "A0": 0.0}
T = 1000.0
dt = 1e-3
dts = 1.0
noise_lvl = 0.0
noise_sigma = 1000.0

# node and edge template initiation
edge, edge_op = "oja_edge", "oja_op"
node, node_op = "qif_sp", "qif_sp_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        edge_tmp = deepcopy(edge_temp)
        if np.random.uniform() <= p:
            w = float(np.random.uniform(0.0, 1.0))
        else:
            w = 0.0
        edge_tmp.update_var(edge_op, "w", w)
        edges.append((f"p{j}/{node_op}/s", f"p{i}/{node_op}/s_in", edge_tmp,
                      {"weight": 1.0,
                       f"{edge}/{edge_op}/r_s": f"p{j}/{node_op}/s",
                       f"{edge}/{edge_op}/r_t": f"p{i}/{node_op}/s",
                       f"{edge}/{edge_op}/x_s": f"p{j}/{node_op}/u",
                       f"{edge}/{edge_op}/x_t": f"p{i}/{node_op}/u",
                       }))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# define extrinsic input
inp = np.zeros((int(T/dt),))
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# run simulation
res = net.run(simulation_time=T, step_size=dt, inputs={f"all/{node_op}/I_ext": inp},
              outputs={"s": f"all/{node_op}/s", "u": f"all/{node_op}/u", "a": f"all/{node_op}/a"},
              solver="scipy", clear=False, sampling_step_size=dts, decorator=njit, float_precision="float64",
              method="RK23", atol=1e-5, min_step=dt)

# extract synaptic weights
mapping, weights, etas = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
idx = np.argsort(etas)
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]
clear(net)
# np.fill_diagonal(W, 0)

# calculate matrix quantities
in_edges = np.sum(W, axis=1)
corr = correlate(in_edges, etas[idx])

# plotting
time = res.index * 10.0
fig = plt.figure(figsize=(16, 6))
grid = fig.add_gridspec(nrows=3, ncols=3)
ax1 = fig.add_subplot(grid[0, :2])
ax1.plot(time, res["s"])
ax1.plot(time, np.mean(res["s"].values, axis=1), color="black")
ax1.set_ylabel("r (Hz)")
ax = fig.add_subplot(grid[1, :2])
ax.sharex(ax1)
ax.plot(time, res["u"])
ax.set_ylabel("u")
ax = fig.add_subplot(grid[2, :2])
ax.sharex(ax1)
ax.plot(time, res["a"])
ax.set_ylabel("a")
ax.set_xlabel("time")
ax = fig.add_subplot(grid[:, 2])
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="cividis")
plt.colorbar(im, ax=ax)
step = 4
labels = np.round(etas[idx][::step], decimals=2)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title(f"C(etas, sum(W, 1)) = {corr}")
plt.tight_layout()
plt.show()
