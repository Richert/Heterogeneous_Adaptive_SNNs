import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

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

@njit
def integrate_noise(x, inp, scale, tau):
    return x + scale * inp - x / tau

def generate_colored_noise(num_samples, tau, scale=1.0):
    """
    Generates Brownian noise by integrating white noise.

    Args:
        num_samples (int): The number of samples in the output Brownian noise.
        scale (float): A scaling factor for the noise amplitude.

    Returns:
        numpy.ndarray: An array containing the generated Brownian noise.
    """
    white_noise = np.random.randn(num_samples)
    x = 0.0
    colored_noise = np.zeros_like(white_noise)
    for sample in range(num_samples):
        x = integrate_noise(x, white_noise[sample], scale, tau)
        colored_noise[sample] = x
    return colored_noise

# parameters
path = "/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs"
rep = 0 #int(sys.argv[-1])
b = 0.2 #float(sys.argv[-2])
Delta = 2.0 #float(sys.argv[-3])
M = 10
p = 1.0
eta = -1.1
etas = uniform(M, eta, Delta)
node_vars = {"tau": 1.0, "J": 15.0 / (0.5*p*M), "eta": etas, "Delta": Delta/(2*M)}
syn_vars = {"tau_s": 0.5, "tau_a": 20.0, "A0": 0.2}
ca_vars = {"tau_u": 100.0}
edge_vars = {"a": 0.0, "b": b}
T = 2000.0
dt = 1e-3
dts = 1.0
noise_tau = 100.0
noise_scale = 0.2

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = "qif_sf", "qif_op", "syn_sf_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
W0 = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        edge_tmp = deepcopy(edge_temp)
        if np.random.uniform() <= p:
            w = 1.0 #float(np.random.uniform(0.0, 1.0))
        else:
            w = 0.0
        W0[i, j] = w
        edge_tmp.update_var(edge_op, "w", w)
        edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", edge_tmp,
                      {"weight": 1.0,
                       f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                       f"{edge}/{edge_op}/p1": f"p{j}/{syn_op}/s",
                       f"{edge}/{edge_op}/p2": f"p{i}/ca_op/u",
                       f"{edge}/{edge_op}/d1": f"p{j}/ca_op/u",
                       f"{edge}/{edge_op}/d2": f"p{i}/{node_op}/r",
                       }))
        # edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", None, {"weight": 1.0}))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})
net.update_var(node_vars={f"all/ca_op/{key}": val for key, val in ca_vars.items()})

# define extrinsic input
inp = np.zeros((int(T/dt), 1))
# inp[int(300/dt):int(600/dt), 0] = 0.1
inp[:, 0] += generate_colored_noise(int(T/dt), noise_tau, noise_scale)

# run simulation
res = net.run(simulation_time=T, step_size=dt, inputs={f"all/{node_op}/I_ext": inp},
              outputs={"r": f"all/{node_op}/r", "a": f"all/{syn_op}/a"}, solver="heun", clear=False,
              sampling_step_size=dts, float_precision="float64")

# extract synaptic weights
mapping, weights, etas_tmp = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
if len(etas) == M:
    etas = etas_tmp
idx = np.argsort(etas)
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]
clear(net)
np.fill_diagonal(W, 0)

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
ax = axes[1]
ax.imshow(W, interpolation="none", aspect="auto")
ax.set_title("Final Weights")
ax = axes[0]
ax.imshow(W0, interpolation="none", aspect="auto")
ax.set_title("Initial Weights")
fig, axes = plt.subplots(nrows=2, figsize=(10, 6))
ax = axes[0]
ax.plot(res["r"])
ax.plot(res["r"].iloc[:, int(M/2)-1], color="black")
ax.set_title("firing rate")
ax = axes[1]
ax.plot(res["a"])
ax.plot(res["a"].iloc[:, int(M/2)-1], color="black")
ax.set_title("synaptic adaptation")
plt.show()
# pickle.dump(
#     {"W": W, "eta": etas[idx], "b": b, "Delta": Delta, "noise": noise_lvl, "s": np.mean(res["s"].values, axis=1)},
#     open(f"{path}/results/rnn_results/fre_inh_{int(b*10)}_{int(noise_lvl)}_{int(Delta*10.0)}_{rep}.pkl", "wb")
# )
