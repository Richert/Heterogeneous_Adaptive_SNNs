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

# model parameters
C = 50.0
k = 1.0
v_r = -80.0
v_t = -30.0
Delta = 0.8
eta = 0.0
kappa = 250.0
a = 0.01
b = -20.0
tau_s = 8.0
tau_u = 50.0
tau_ca = 150.0
g_i = 4.0
E_i = -60.0
g_e = 10.0
E_e = 0.0
tau_ltd = 20.0
tau_ltp = 15.0
a_ltd = 6e-3
a_ltp = 8e-3
gamma_ltp = 50.0
gamma_ltd = 50.0
noise_lvl = 0.1
N, M = 1, 10
p = 1.0
etas = uniform(M, eta, Delta)
edge_vars = {
    "a_ltp": a_ltp, "a_ltd": a_ltd, "gamma_ltp": gamma_ltp, "gamma_ltd": gamma_ltd, "theta_ltp": v_t, "theta_ltd": v_r
}
node_vars = {"C": C, "k": k, "eta": etas, "Delta": Delta/(2*M), "v_r": v_r, "v_t": v_t, "tau_s": tau_s, "tau_u": tau_u,
             "tau_ca": tau_ca,  "tau_ltp": tau_ltp, "tau_ltd": tau_ltd, "g_e": g_e, "g_i": g_i, "E_e": E_e, "E_i": E_i,
             "kappa": kappa}
T = 3000.0
dt = 1e-3
dts = 1.0
global_noise = 100.0
noise_sigma = 1.0/dt

# node and edge template initiation
edge, edge_op = "clopath_edge", "clopath_op"
node, node_op = "ik_clopath", "ik_clopath_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
w0 = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        edge_tmp = deepcopy(edge_temp)
        if np.random.uniform() <= p:
            w = float(np.random.uniform(0.0, 1.0))
        else:
            w = 0.0
        w0[i, j] = w
        edge_tmp.update_var(edge_op, "w", w)
        edges.append((f"p{j}/{node_op}/s", f"p{i}/{node_op}/s_in", edge_tmp,
                      {"weight": 1.0,
                       f"{edge}/{edge_op}/s_pre": f"p{j}/{node_op}/s",
                       f"{edge}/{edge_op}/u_pre": f"p{j}/{node_op}/u",
                       f"{edge}/{edge_op}/v_post": f"p{i}/{node_op}/v",
                       f"{edge}/{edge_op}/v_post_ltd": f"p{i}/{node_op}/v_ltd",
                       f"{edge}/{edge_op}/v_post_ltp": f"p{i}/{node_op}/v_ltp",
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
w0 = w0[idx, :]
w0 = w0[:, idx]
# clear(net)
# np.fill_diagonal(W, 0)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(w0, vmin=0.0, vmax=1.0, aspect="auto", interpolation="none")
ax.set_title("initial weights")
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(W, vmin=0.0, vmax=1.0, aspect="auto", interpolation="none")
ax.set_title("final weights")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(res["s"])
ax.plot(np.mean(res["s"], axis=1), color="black")
ax.set_title("network dynamics")
plt.show()
# pickle.dump(
#     {"W": W, "eta": etas[idx], "b": b, "Delta": Delta, "noise": noise_lvl, "s": res["s"]},
#     open(f"{path}/results/rnn_results/fre_mp_{int(b*10)}_{int(noise_lvl)}_{int(Delta*10.0)}_{rep}.pkl", "wb")
# )
