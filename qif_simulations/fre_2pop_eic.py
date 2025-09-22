import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.random.randn(N)

# parameters
M = 50
p = 0.5
exc_edge_vars = {"a": 10.0, "b": 0.01}
inh_edge_vars = {"a": 10.0, "b": 0.0}
Delta_e, Delta_i = 0.5, 0.8
eta_e, eta_i = 1.0, 0.0
etas_e = gaussian(M, eta_e, Delta_e)
etas_i = gaussian(M, eta_i, Delta_i)
exc_vars = {"tau": 1.2, "J": -8.0 / M, "eta": etas_e, "tau_u": 10.0, "tau_s": 0.6, "Delta": 0.05}
inh_vars = {"tau": 0.8, "J": 5.0 / M, "eta": etas_i, "tau_u": 10.0, "tau_s": 0.2, "Delta": 0.05}
T = 1000.0
dt = 5e-4
dts = 1e-1

# node and edge template initiation
exc_edge, exc_edge_op = "stdp_edge", "stdp_op"
inh_edge, inh_edge_op = "stdp_ah_edge", "stdp_ah_op"
node, node_op = "qif_stdp", "qif_stdp_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
exc_edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{exc_edge}")
inh_edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{inh_edge}")
for key, val in exc_edge_vars.items():
    exc_edge_temp.update_var(exc_edge_op, key, val)
for key, val in inh_edge_vars.items():
    inh_edge_temp.update_var(inh_edge_op, key, val)

# initialize network connections
edges_1, edges_2 = [], []
w1 = np.zeros((M, M))
w2 = np.zeros((M, M))
for i in range(M):
    for j in range(M):

        # forward edge
        edge_tmp = deepcopy(exc_edge_temp)
        if np.random.uniform() <= p:
            w = float(np.random.uniform(0.0, 1.0))
        else:
            w = 0.0
        edge_tmp.update_var(exc_edge_op, "w", w)
        w1[i, j] = w
        edges_1.append((f"exc_{j}/{node_op}/s", f"inh_{i}/{node_op}/s_in", edge_tmp,
                        {"weight": 1.0,
                         f"{exc_edge}/{exc_edge_op}/r_s": f"exc_{j}/{node_op}/s",
                         f"{exc_edge}/{exc_edge_op}/r_t": f"inh_{i}/{node_op}/s",
                         f"{exc_edge}/{exc_edge_op}/x_s": f"exc_{j}/{node_op}/u",
                         f"{exc_edge}/{exc_edge_op}/x_t": f"inh_{i}/{node_op}/u",
                         }))

        # backward edge
        edge_tmp = deepcopy(inh_edge_temp)
        if np.random.uniform() <= p:
            w = float(np.random.uniform(0.0, 1.0))
        else:
            w = 0.0
        edge_tmp.update_var(inh_edge_op, "w", w)
        w2[i, j] = w
        edges_2.append((f"inh_{j}/{node_op}/s", f"exc_{i}/{node_op}/s_in", edge_tmp,
                        {"weight": 1.0,
                         f"{inh_edge}/{inh_edge_op}/r_s": f"inh_{j}/{node_op}/s",
                         f"{inh_edge}/{inh_edge_op}/r_t": f"exc_{i}/{node_op}/s",
                         f"{inh_edge}/{inh_edge_op}/x_s": f"inh_{j}/{node_op}/u",
                         f"{inh_edge}/{inh_edge_op}/x_t": f"exc_{i}/{node_op}/u",
                         }))

# initialize network
nodes = {}
for i in range(M):
    exc_temp = deepcopy(node_temp)
    for key, val in exc_vars.items():
        exc_temp.update_var(node_op, key, val[i] if type(val) is np.ndarray else val)
    nodes[f"exc_{i}"] = exc_temp
    inh_temp = deepcopy(node_temp)
    for key, val in inh_vars.items():
        inh_temp.update_var(node_op, key, val[i] if type(val) is np.ndarray else val)
    nodes[f"inh_{i}"] = inh_temp
net = CircuitTemplate(name=node, nodes=nodes, edges=edges_1 + edges_2)

# run simulation
res = net.run(simulation_time=T, step_size=dt,
              outputs={"r": f"all/{node_op}/r", "u": f"all/{node_op}/u"},
              solver="scipy", clear=False, sampling_step_size=dts)

# extract synaptic weights
exc_map, exc_weights = net._ir["weight_in0"].value, net.state["w"]
inh_map, inh_weights = net._ir["weight_in1"].value, net.state["w_v1"]
etas = net._ir["eta"].value
idx_exc = np.argsort(etas[::2])[::-1]
idx_inh = np.argsort(etas[1::2])[::-1]
w1_final = np.asarray([exc_weights[exc_map[i, :] > 0.0] for i in idx_exc])
w1_final = w1_final[:, idx_inh]
w2_final = np.asarray([inh_weights[inh_map[i, :] > 0.0] for i in idx_inh])
w2_final = w2_final[:, idx_exc]
clear(net)
w1 = w1[idx_exc, :]
w1 = w1[:, idx_inh]
w2 = w2[idx_exc, :]
w2 = w2[:, idx_inh]
# np.fill_diagonal(W, 0)

# plotting
time = res.index * 10.0
r, u = res["r"].values, res["u"].values
r_e, u_e = r[:, ::2], u[:, ::2]
r_i, u_i = r[:, 1::2], u[:, 1::2]
fig = plt.figure(figsize=(15, 8), layout="constrained")
grid = fig.add_gridspec(nrows=2, ncols=4)
ax1 = fig.add_subplot(grid[0, :2])
ax1.plot(time, r_e * 100.0)
ax1.plot(time, np.mean(r_e, axis=1) * 100.0, color="black")
ax1.set_ylabel("r_s (Hz)")
ax1.set_ylim([0.0, 100.0])
ax1.set_title("Excitatory rate dynamics")
ax = fig.add_subplot(grid[1, :2])
ax.sharex(ax1)
ax.plot(time, r_i * 100.0)
ax.plot(time, np.mean(r_i, axis=1) * 100.0, color="black")
ax.set_ylabel("r_t (Hz)")
ax.set_ylim([0.0, 100.0])
ax.set_xlabel("time")
ax.set_title("Inhibitory rate dynamics")
ax = fig.add_subplot(grid[0, 2])
im = ax.imshow(w1, aspect="auto", interpolation="none", cmap="viridis", vmin=0.0, vmax=1.0)
step = int(M/5)
row_labels = np.round(etas[::2][idx_exc][::step], decimals=1)
col_labels = np.round(etas[1::2][idx_inh][::step], decimals=1)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("I -> E: Initial Weights")
ax = fig.add_subplot(grid[0, 3])
im = ax.imshow(w1_final, aspect="auto", interpolation="none", cmap="viridis", vmin=0.0, vmax=1.0)
plt.colorbar(im, ax=ax)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("I -> E: Final Weights")
ax = fig.add_subplot(grid[1, 2])
im = ax.imshow(w2, aspect="auto", interpolation="none", cmap="viridis", vmin=0.0, vmax=1.0)
step = int(M/5)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("E -> I: Initial Weights")
ax = fig.add_subplot(grid[1, 3])
im = ax.imshow(w2_final, aspect="auto", interpolation="none", cmap="viridis", vmin=0.0, vmax=1.0)
plt.colorbar(im, ax=ax)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("E -> I: Final Weights")
plt.show()

# perform tda
# W[W > 0.3] = 1.0
# W[W < 0.3] = 0.0
# X_ggd = GraphGeodesicDistance(directed=True, unweighted=False).fit_transform([W])
# X_fp = FlagserPersistence(directed=True).fit_transform(X_ggd)
# fig1 = plot_diagram(X_fp[1])
# write_images(fig1, file="/home/richard-gast/Documents/test.pdf")
