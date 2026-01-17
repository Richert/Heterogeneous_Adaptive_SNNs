import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import sys
import h5py

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

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read sweep parameters
rep = int(sys.argv[-1])
syn = str(sys.argv[-2])
stp = str(sys.argv[-3])
ltp = str(sys.argv[-4])
tau_p = float(sys.argv[-5])
tau_d = float(sys.argv[-6])
a_p = float(sys.argv[-7])
a_d = float(sys.argv[-8])

# load condition parameters
f = h5py.File(f"{path}/1pop_data.hdf5", "a")
ds = f[f"{syn}_{stp}_{ltp}"]
M = ds.attrs["M"]
node_params = ["eta", "Delta", "J"]
syn_params = ["tau_s", "tau_a", "kappa"]
plasticity_params = ["b"]
node_vars, syn_vars, edge_vars = {}, {}, {}
for d, keys in zip([node_vars, syn_vars, edge_vars], [node_params, syn_params, plasticity_params]):
    for key in keys:
        d[key] = ds.attrs[key]
node_vars["eta"] = uniform(M, node_vars["eta"], node_vars["Delta"])
node_vars["Delta"] = node_vars["Delta"]/(2*M)
node_vars["J"] = node_vars["J"]/(0.5*M)

# simulation parameters
cutoff = 200.0
T = 6000.0 + cutoff
dt = 1e-3
dts = 1.0
noise_tau = 200.0
noise_scale = 0.05

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_{stp}", "qif_op", f"syn_{stp}_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# set synaptic plasticity variables
ltp_source = f"{syn_op}/s"
ltp_target = {"oja": f"{syn_op}/s", "stdp": "ltp_op/u_p"}
ltd_source = {"oja": f"{syn_op}/s", "stdp": "ltd_op/u_d"}
ltd_target = f"{syn_op}/s"

# create network
edges = []
W0 = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        edge_tmp = deepcopy(edge_temp)
        w = float(np.random.uniform(0.0, 1.0))
        W0[i, j] = w
        edge_tmp.update_var(edge_op, "w", w)
        edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", edge_tmp,
                      {"weight": 1.0,
                       f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                       f"{edge}/{edge_op}/p1": f"p{j}/{ltp_source}",
                       f"{edge}/{edge_op}/p2": f"p{i}/{ltp_target[ltp]}",
                       f"{edge}/{edge_op}/d1": f"p{i if ltp == 'oja' else j}/{ltd_source[ltp]}",
                       f"{edge}/{edge_op}/d2": f"p{i}/{ltd_target}",
                       }))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})
net.update_var(node_vars={f"all/ltp_op/tau_p": tau_p, f"all/ltd_op/tau_d": tau_d})

# define extrinsic input
inp = np.zeros((int(T/dt), 1))
inp[:, 0] += generate_colored_noise(int(T/dt), noise_tau, noise_scale)

# run simulation
res = net.run(simulation_time=T, step_size=dt, inputs={f"all/{node_op}/I_ext": inp},
              outputs={"r": f"all/{node_op}/r", "a": f"all/{syn_op}/a"}, solver="heun", clear=False,
              sampling_step_size=dts, float_precision="float64", decorator=njit, cutoff=cutoff)

# extract synaptic weights
mapping, weights, etas = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
idx = np.argsort(etas)
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]
clear(net)

# calculate in- and out-degrees
in_degree = np.sum(W, axis=1)
out_degree = np.sum(W, axis=0)

# calculate network covariance eigenvalues
rates = res.loc[:, "r"].values
rates_centered = np.zeros_like(rates)
for i in range(rates.shape[1]):
    rates_centered[:, i] = rates[:, i] - np.mean(rates[:, i])
    rates_centered[:, i] /= np.std(rates[:, i])
C = np.cov(rates_centered, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(C)
# lambdas = np.abs(eigvals)
idx = np.argsort(eigvals)
lambdas = eigvals[idx]

# transform etas into covariance eigenvector space
etas_transformed = np.dot(eigvecs.T, etas)
etas_transformed = etas_transformed[idx]

# calculate fano factors
ff = np.zeros_like(lambdas)
for i in range(M):
    ff[i] = np.var(rates[:, i]) / np.mean(rates[:, i])

# save results
results = {"etas": etas, "in-degrees": in_degree, "out-degrees": out_degree, "lambdas": lambdas, "fano-factors": ff,
           "etas_transformed": etas_transformed}
for i, key in enumerate(ds.attrs["column_vars"]):
    ds[rep, i] = results[key]
# ds = f["results"]
# for key, var in results.items():
#     ds.fields[key][rep, :] = var

# plotting weights
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
ax = axes[2]
im = ax.imshow(C, interpolation="none", aspect="auto", cmap="magma")
ax.set_title("Covariance Matrix")
# plt.colorbar(im, ax=ax)
ax = axes[1]
ax.imshow(W, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0)
ax.set_title("Final Weights")
ax = axes[0]
ax.imshow(W0, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0)
ax.set_title("Initial Weights")

# plotting dynamics
fig, axes = plt.subplots(nrows=2, figsize=(10, 6))
ax = axes[0]
ax.plot(res["r"])
ax.plot(np.mean(res["r"], axis=1), color="black")
ax.set_title("firing rate")
ax = axes[1]
ax.plot(res["a"])
ax.plot(np.mean(res["a"], axis=1), color="black")
ax.set_title("synaptic adaptation")

# plotting DV relationships
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
ax = axes[0]
ax.plot(etas, in_degree, label="in-degree")
ax.plot(etas, out_degree, label="out-degree")
ax.legend()
ax.set_xlabel("eta")
ax.set_ylabel("degree")
ax = axes[1]
ax.plot(etas, ff)
ax.set_xlabel("eta")
ax.set_ylabel("fano factor")
ax = axes[2]
ax.scatter(etas_transformed, np.log(lambdas + 1e-12), color="blue")
ax.set_xlabel("sum(eta*v)")
ax.set_ylabel("log(lambda)")
plt.tight_layout()

plt.show()
