import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
from numba import njit
import sys
import pickle
import matplotlib.pyplot as plt

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

def uniform(N: int, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.linspace(-0.5, 0.5, N)

# parameters
path = "/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs"
Delta = 1.0 #float(sys.argv[-3])
noise_lvl = 0.01 #float(sys.argv[-4])
M = 10
p = 1.0
edge_vars = {
    "a_ltp": 50.0, "a_ltd": 50.0, "gamma_ltp": 50.0, "gamma_ltd": 20.0, "theta_ltp": 0.8, "theta_ltd": 1.0,
    "tau_w": 100.0, "W0": 0.5
}
eta = 0.5
etas = uniform(M, eta, Delta)
node_vars = {"tau": 1.0, "J": -2.0 / (0.5*p*M), "eta": etas, "tau_u": 30.0, "tau_s": 1.0, "Delta": Delta/(2*M)}
T = 1000.0
dt = 1e-3
dts = 1.0
global_noise = 10.0
noise_sigma = 1.0/dt

# node and edge template initiation
edge, edge_op = "brunel_edge", "brunel_op"
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
        if np.random.uniform() <= p:
            w = float(np.random.uniform(0.0, 1.0))
        else:
            w = 0.0
        edge_tmp.update_var(edge_op, "w", w)
        edges.append((f"p{j}/{node_op}/s", f"p{i}/{node_op}/s_in", edge_tmp,
                      {"weight": 1.0,
                       f"{edge}/{edge_op}/s_pre": f"p{j}/{node_op}/s",
                       f"{edge}/{edge_op}/u_pre": f"p{j}/{node_op}/u",
                       f"{edge}/{edge_op}/u_post": f"p{i}/{node_op}/u",
                       }))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# define extrinsic input
inp = np.zeros((int(T/dt), M))
noise = noise_lvl*np.random.randn(*inp.shape) + global_noise * np.random.randn(inp.shape[0], 1)
noise = gaussian_filter1d(noise, sigma=noise_sigma, axis=0)
inp += noise

# run simulation
res = net.run(simulation_time=T, step_size=dt, inputs={f"all/{node_op}/I_ext": inp},
              outputs={"s": f"all/{node_op}/s"}, solver="heun", clear=False, sampling_step_size=dts,
              float_precision="float32", vectorize=True)

# extract synaptic weights
mapping, weights, etas_tmp = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
if len(etas) == M:
    etas = etas_tmp
idx = np.argsort(etas)
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]
clear(net)
# np.fill_diagonal(W, 0)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(W, vmin=0.0, vmax=1.0, aspect="auto", interpolation="none")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(res["s"])
ax.plot(np.mean(res["s"], axis=1), color="black")
plt.show()
# pickle.dump(
#     {"W": W, "eta": etas[idx], "b": b, "Delta": Delta, "noise": noise_lvl, "s": res["s"]},
#     open(f"{path}/results/rnn_results/fre_mp_{int(b*10)}_{int(noise_lvl)}_{int(Delta*10.0)}_{rep}.pkl", "wb")
# )
