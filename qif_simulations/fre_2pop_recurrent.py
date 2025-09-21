import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.random.randn(N)

# parameters
M = 50
p = 0.5
edge_vars = {
    "a": 10.0, "b": 0.01
}
Delta_source, Delta_target = 1.0, 1.0
eta_source, eta_target = 0.5, 0.0
etas_source = gaussian(M, eta_source, Delta_source)
etas_target = gaussian(M, eta_target, Delta_target)
etas = np.asarray(etas_source.tolist() + etas_target.tolist())
node_vars = {"tau": 1.0, "J": 4.0 / M, "eta": etas, "tau_u": 10.0, "tau_s": 0.5, "Delta": 0.05}
T = 500.0
dt = 5e-4
dts = 1e-1

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op = "qif_stdp", "qif_stdp_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges_1, edges_2 = [], []
w1 = np.zeros((M, M))
w2 = np.zeros((M, M))
for i in range(M):
    for j in range(M):

        # forward edge
        edge_tmp = deepcopy(edge_temp)
        if np.random.uniform() <= p:
            w = float(np.random.uniform(0.0, 1.0))
        else:
            w = 0.0
        edge_tmp.update_var(edge_op, "w", w)
        w1[i, j] = w
        edges_1.append((f"s{j}/{node_op}/s", f"t{i}/{node_op}/s_in", edge_tmp,
                        {"weight": 1.0,
                         f"{edge}/{edge_op}/r_s": f"s{j}/{node_op}/s",
                         f"{edge}/{edge_op}/r_t": f"t{i}/{node_op}/s",
                         f"{edge}/{edge_op}/x_s": f"s{j}/{node_op}/u",
                         f"{edge}/{edge_op}/x_t": f"t{i}/{node_op}/u",
                         }))

        # backward edge
        edge_tmp = deepcopy(edge_temp)
        if np.random.uniform() <= p:
            w = float(np.random.uniform(0.0, 1.0))
        else:
            w = 0.0
        edge_tmp.update_var(edge_op, "w", w)
        w2[i, j] = w
        edges_2.append((f"t{j}/{node_op}/s", f"s{i}/{node_op}/s_in", edge_tmp,
                        {"weight": 1.0,
                         f"{edge}/{edge_op}/r_s": f"t{j}/{node_op}/s",
                         f"{edge}/{edge_op}/r_t": f"s{i}/{node_op}/s",
                         f"{edge}/{edge_op}/x_s": f"t{j}/{node_op}/u",
                         f"{edge}/{edge_op}/x_t": f"s{i}/{node_op}/u",
                         }))

nodes = {f"s{i}": node_temp for i in range(M)}
nodes.update({f"t{i}": node_temp for i in range(M)})
net = CircuitTemplate(name=node, nodes=nodes, edges=edges_1 + edges_2)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# run simulation
res = net.run(simulation_time=T, step_size=dt,
              outputs={"r": f"all/{node_op}/r", "u": f"all/{node_op}/u"},
              solver="scipy", clear=False, sampling_step_size=dts)

# extract synaptic weights
mapping, weights, etas = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
idx_1 = np.argsort(etas[:M])[::-1]
idx_2 = np.argsort(etas[M:])[::-1]
w1_final = np.asarray([weights[mapping[i, :] > 0.0] for i in idx_1])
w1_final = w1_final[:, idx_2]
w2_final = np.asarray([weights[mapping[i, :] > 0.0] for i in idx_2])
w2_final = w2_final[:, idx_1]
clear(net)
w1 = w1[idx_1, :]
w1 = w1[:, idx_2]
w2 = w2[idx_2, :]
w2 = w2[:, idx_1]
# np.fill_diagonal(W, 0)

# plotting
time = res.index * 10.0
r, u = res["r"].values, res["u"].values
r_source, u_source = r[:, :M], u[:, :M]
r_target, u_target = r[:, M:], u[:, M:]
fig = plt.figure(figsize=(15, 8), layout="constrained")
grid = fig.add_gridspec(nrows=2, ncols=4)
ax1 = fig.add_subplot(grid[0, :2])
ax1.plot(time, r_source*100.0)
ax1.plot(time, np.mean(r_source, axis=1)*100.0, color="black")
ax1.set_ylabel("r_s (Hz)")
ax1.set_ylim([0.0, 100.0])
ax1.set_title("Firing rate dynamics")
ax = fig.add_subplot(grid[1, :2])
ax.sharex(ax1)
ax.plot(time, r_target*100.0)
ax.plot(time, np.mean(r_target, axis=1)*100.0, color="black")
ax.set_ylabel("r_t (Hz)")
ax.set_ylim([0.0, 100.0])
ax.set_xlabel("time")
ax = fig.add_subplot(grid[0, 2])
im = ax.imshow(w1, aspect="auto", interpolation="none", cmap="viridis", vmin=0.0, vmax=1.0)
step = int(M/5)
row_labels = np.round(etas[M+idx_1][::step], decimals=1)
col_labels = np.round(etas[idx_2][::step], decimals=1)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("S -> T: Initial Weights")
ax = fig.add_subplot(grid[0, 3])
im = ax.imshow(w1_final, aspect="auto", interpolation="none", cmap="viridis", vmin=0.0, vmax=1.0)
plt.colorbar(im, ax=ax)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("S -> T: Final Weights")
ax = fig.add_subplot(grid[1, 2])
im = ax.imshow(w2, aspect="auto", interpolation="none", cmap="viridis", vmin=0.0, vmax=1.0)
step = int(M/5)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("T -> S: Initial Weights")
ax = fig.add_subplot(grid[1, 3])
im = ax.imshow(w2_final, aspect="auto", interpolation="none", cmap="viridis", vmin=0.0, vmax=1.0)
plt.colorbar(im, ax=ax)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("T -> S: Final Weights")
plt.show()

# perform tda
# W[W > 0.3] = 1.0
# W[W < 0.3] = 0.0
# X_ggd = GraphGeodesicDistance(directed=True, unweighted=False).fit_transform([W])
# X_fp = FlagserPersistence(directed=True).fit_transform(X_ggd)
# fig1 = plot_diagram(X_fp[1])
# write_images(fig1, file="/home/richard-gast/Documents/test.pdf")
