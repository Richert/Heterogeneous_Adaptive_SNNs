import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy

# parameters
M = 100
edge_vars = {
    "a": 1.0, "b": 1.0
}
Delta = 1.0
eta = 1.0
indices = np.arange(1, M+1)
etas = eta + Delta*np.tan(0.5*np.pi*(2*indices-M-1)/(M+1))
deltas = Delta*(np.tan(0.5*np.pi*(2*indices-M-0.5)/(M+1))-np.tan(0.5*np.pi*(2*indices-M-1.5)/(M+1)))
deltas = np.asarray(deltas.tolist() + deltas.tolist())
etas = np.asarray(etas.tolist() + etas.tolist())
node_vars = {"tau": 1.0, "J": 10.0 / M, "eta": etas, "tau_u": 10.0, "tau_s": 1.0, "Delta": deltas}
T = 200.0
dt = 1e-4
dts = 1e-1

# node and edge template initiation
edge, edge_op = "oja_edge", "oja_op"
node, node_op = "qif_stdp", "qif_stdp_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        edge_tmp = deepcopy(edge_temp)
        edge_tmp.update_var(edge_op, "w", float(np.random.choice([0.0, 0.33, 0.66, 1.0])))
        edges.append((f"s{j}/{node_op}/s", f"t{i}/{node_op}/s_in", edge_tmp,
                      {"weight": 1.0,
                       f"{edge}/{edge_op}/r_s": f"s{j}/{node_op}/s",
                       f"{edge}/{edge_op}/r_t": f"t{i}/{node_op}/s",
                       f"{edge}/{edge_op}/x_s": f"s{j}/{node_op}/u",
                       f"{edge}/{edge_op}/x_t": f"t{i}/{node_op}/u",
                       }))
nodes = {f"s{i}": node_temp for i in range(M)}
nodes.update({f"t{i}": node_temp for i in range(M)})
net = CircuitTemplate(name=node, nodes=nodes, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# run simulation
res = net.run(simulation_time=T, step_size=dt,
              outputs={"r": f"all/{node_op}/r"},
              solver="scipy", clear=False, sampling_step_size=dts, max_step=1e-2)

# extract synaptic weights
mapping, weights, etas = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
idx_col = np.argsort(etas[:M])[::-1]
idx_row = np.argsort(etas[M:])[::-1]
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx_row])
W = W[:, idx_col]
clear(net)
# np.fill_diagonal(W, 0)

# plotting
time = res.index * 10.0
r = res["r"].values
r_source = r[:, :M]
r_target = r[:, M:]
fig = plt.figure(figsize=(13, 4))
grid = fig.add_gridspec(nrows=2, ncols=3)
ax1 = fig.add_subplot(grid[0, :2])
ax1.plot(time, r_source*100.0)
ax1.plot(time, np.mean(r_source, axis=1)*100.0, color="black")
ax1.set_ylabel("r (Hz)")
ax1.set_title("Source population dynamics")
ax = fig.add_subplot(grid[1, :2])
ax.sharex(ax1)
ax.plot(time, r_target*100.0)
ax.plot(time, np.mean(r_target, axis=1)*100.0, color="black")
ax.set_ylabel("r (Hz)")
ax.set_xlabel("time")
ax.set_title("Target population dynamics")
ax = fig.add_subplot(grid[:, 2])
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar(im, ax=ax)
step = 4
row_labels = np.round(etas[M+idx_row][::step], decimals=2)
col_labels = np.round(etas[idx_col][::step], decimals=2)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=col_labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=row_labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("Synaptic Weights (FFWD)")
plt.tight_layout()
plt.show()

# perform tda
# W[W > 0.3] = 1.0
# W[W < 0.3] = 0.0
# X_ggd = GraphGeodesicDistance(directed=True, unweighted=False).fit_transform([W])
# X_fp = FlagserPersistence(directed=True).fit_transform(X_ggd)
# fig1 = plot_diagram(X_fp[1])
# write_images(fig1, file="/home/richard-gast/Documents/test.pdf")
