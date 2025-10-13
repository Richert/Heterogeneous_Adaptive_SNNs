import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
from numba import njit
import sys
import pickle

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

# parameters
path = "/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs"
rep = int(sys.argv[-1])
b = float(sys.argv[-2])
Delta = float(sys.argv[-3])
noise_lvl = float(sys.argv[-4])
M = 50
p = 1.0
edge_vars = {
    "a": 0.1, "b": b
}
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
inp = np.zeros((int(T/dt), M))
noise = noise_lvl*np.random.randn(*inp.shape)
noise = gaussian_filter1d(noise, sigma=noise_sigma, axis=0)
inp += noise

# run simulation
res = net.run(simulation_time=T, step_size=dt, inputs={f"p{i}/{node_op}/I_ext": inp[:, i] for i in range(M)},
              outputs={"s": f"all/{node_op}/s"},
              solver="heun", clear=False, sampling_step_size=dts, float_precision="float64"
              )

# extract synaptic weights
mapping, weights, etas = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
idx = np.argsort(etas)
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]
clear(net)
# np.fill_diagonal(W, 0)

pickle.dump(
    {"W": W, "eta": etas, "b": b, "Delta": Delta, "noise": noise_lvl},
    open(f"{path}/results/rnn_results/fre_simulation_{int(b*10)}_{int(noise_lvl)}_{int(Delta*10.0)}_{rep}.pkl", "wb")
)
