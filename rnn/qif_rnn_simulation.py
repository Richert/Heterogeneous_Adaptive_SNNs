import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate
from rectipy import Network
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
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

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

# parameters
path = "/home/richard-gast/PycharmProjects/Heterogeneous_Adaptive_SNNs"
rep = 0 #int(sys.argv[-1])
b = 0.1 #float(sys.argv[-2])
Delta = 0.5 #float(sys.argv[-3])
noise_lvl = 10.0 #float(sys.argv[-4])
N = 100
p = 1.0
edge_vars = {
    "a": 0.005, "b": b
}
eta = -1.25
etas = lorentzian(N, eta, Delta)
node_vars = {"tau": 1.0, "J": 30.0 / (p*N), "eta": etas, "tau_u": 30.0, "tau_s": 1.0,
             "tau_a": 20.0, "kappa": 0.2, "A0": 0.0}
T = 1000.0
dt = 1e-2
dts = 1.0
noise_sigma = 1000.0

# node and edge template initiation
edge, edge_op = "oja_edge", "oja_op"
node, node_op = "qif_sd", "qif_sd_op"
node_temp = NodeTemplate.from_yaml(f"../config/snn_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
for i in range(N):
    for j in range(N):
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
template = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(N)}, edges=edges)
template.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# generate rectipy network
net = Network(dt=dt, device="cpu")
net.add_diffeq_node(label="qif", node=template, input_var="I_ext", output_var="s",
                    source_var="s", target_var="s_in", node_vars={key: val for key, val in node_vars.items()},
                    op=node_op, clear=False, spike_var="spike", reset_var="v")

# define extrinsic input
inp = np.zeros((int(T/dt), N))
noise = noise_lvl*np.random.randn(*inp.shape)
noise = gaussian_filter1d(noise, sigma=noise_sigma, axis=0)
inp += noise

# run simulation
res = net.run(inputs=inp, sampling_steps=int(dts/dt), enable_grad=False)
s = res.to_numpy("out")

# extract synaptic weights
mapping = net["qif"]["node"]["weights"].cpu().numpy()
weights = net["qif"]["node"].y[int(4*N):]
etas = net["qif"]["node"][f"{node_op}/eta"].cpu().numpy()
idx = np.argsort(etas)
etas = etas[idx]
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]
net.clear()

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(W)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(s)
ax.plot(np.mean(s, axis=1), color="black")
plt.show()

pickle.dump(
    {"W": W, "eta": etas, "b": b, "Delta": Delta, "noise": noise_lvl},
    open(f"{path}/results/rnn_results/qif_simulation_{int(b*10)}_{int(noise_lvl)}_{int(Delta*10.0)}_{rep}.pkl", "wb")
)
