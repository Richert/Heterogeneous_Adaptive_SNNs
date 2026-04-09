import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from scipy.signal import welch
from config.utility_functions import *
from scipy.ndimage import gaussian_filter1d
from typing import Callable


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


def get_c(X: np.ndarray, alpha: float = 1e-4):
    """
    """
    return X @ X.T + alpha*np.eye(X.shape[0])

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
trial = 0
syn = "exc"
stp = "sd"

# set model parameters
M = 50
Delta = 2.0
p = 1.0
eta = -1.0
b = 0.5
tau_s = 1.0
tau_a = 20.0
kappa = 0.1
etas = uniform(M, eta, Delta)
Delta2 = Delta/(2*M)
fr_scale = 1.0
c0 = 30.0
c1 = 10.0
c2 = -10.0
node_vars = {"eta": etas, "Delta": Delta2}
syn_vars = {"tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}

# training parameters
T = 50.0
dt = 1e-4
dts = 1e-2
sr = int(dts/dt)
n_stims = 10
n_tests = 2
stim_amp = 2.0
stim_dur = int(1.0/dt)
noise = 1e-2

# other analysis parameters
K_width = 100
sigma = 10
margin = 100
seq_range = 50
indices = np.arange(0, M, dtype=np.int32)
gamma = 1e-3

# stimulation parameters
p_in = 0.2
cycle_steps = int(T/dt)
stim_onsets = np.linspace(0.0, 50.0, num=n_stims+1)[:-1]
stim_phases = 2.0*np.pi*stim_onsets/T
stim_onsets = [int(onset/dt) for onset in stim_onsets]
stim_width = int(20.0/dt)
n_inputs = int(p_in*M)
center = int(M*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
test_trials = list(np.arange(0, n_stims, n_tests))
train_trials = list(np.arange(0, n_stims))
for t in test_trials:
    train_trials.pop(train_trials.index(t))

# create two target signals to fit
delay = 1500
steps = int(np.round(cycle_steps / sr))
target_1 = np.zeros((steps,))
target_1[delay] = 1.0
target_1 = gaussian_filter1d(target_1, sigma=int(delay*0.1))
t = np.linspace(0, T*1e-2, steps)
f1 = 2.0
f2 = 5.0
target_2 = np.sin(2.0*np.pi*f1*t) * np.sin(2.0*np.pi*f2*t)
targets = [target_1, target_2]

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_stdp_{stp}", "qif_op", f"syn_{stp}_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")

# create network
frs = 1.0 / (1.0 + np.exp(-fr_scale*etas))
edges = []
W = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        W[i, j] = (c0 + c1*frs[i]*c2*frs[j]) / M
        edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", None,
                      {"weight": W[i, j]}))
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})

# show connectivity
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(W, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax)
plt.show()

# generate run function
inp = np.zeros((int(T/dt), M), dtype=np.float32)
func, args, arg_keys, _ = net.get_run_func(f"{syn}_{stp}_vectorfield", file_name=f"{syn}_{stp}_run",
                                           step_size=dt, backend="numpy", solver="heun", float_precision="float32",
                                           vectorize=True, inputs={f"all/{node_op}/I_ext": inp}, clear=False)
func_njit = njit(func)
func_njit(*args)
rhs = func_njit

# find argument positions of free parameters
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")
eta_idx = arg_keys.index(f"p0/{node_op}/eta")
args = list(args)

# set initial state
init_hist, y0 = integrate(args[1], rhs, tuple(args[2:]), T, dt, dts)

# collect network responses
signals = []
inputs = []
for i, stim in enumerate(stim_onsets):
    inp = np.zeros((cycle_steps, M))
    inp[:stim_dur, inp_indices] = stim_amp
    inp = gaussian_filter1d(inp, sigma=sigma, axis=0)
    args[inp_idx] = np.asarray(inp, dtype=np.float32)
    y_init = np.asarray(y0 + noise*np.random.randn(*y0.shape), dtype=np.float32)
    y_init[:M] = np.abs(y_init[:M])
    y, _ = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
    if np.isfinite(y[-1, 0]):
        signals.append(gaussian_filter1d(y[:, :M], sigma=sigma, axis=0).T)
        inputs.append(inp[::sr, inp_indices[0]])
        print(f"Finished {i+1} / {n_stims} training trials.")
    else:
        print(f"Trial {i+1} / {n_stims} failed (contains NaNs).")

# extract results for training and testing
train_signals = [signals[idx] for idx in train_trials]
test_signals = [signals[idx] for idx in test_trials]
train_phases = [stim_phases[idx] for idx in train_trials]
test_phases = [stim_phases[idx] for idx in test_trials]

# calculate network dimensionality and covariance
dims = []
cs = []
for s in train_signals:

    # calculate the network dimensionality
    corr_net = np.cov(s)
    eigs = np.abs(np.linalg.eigvals(corr_net))
    dims.append(np.sum(eigs) ** 2 / np.sum(eigs ** 2))

    # calculate the network covariance matrices
    cs.append(get_c(s, alpha=gamma))

# calculate the network kernel
s_mean = np.mean(train_signals, axis=0)
s_var = np.mean([s_i - s_mean for s_i in train_signals], axis=0)
C_inv = np.linalg.inv(np.mean(cs, axis=0))
w = C_inv @ s_mean
K = s_mean.T @ w
G = s_var.T @ w

# calculate the prediction performance for concrete targets
train_predictions = []
train_distortions = []
test_predictions = []
mses = []
readouts = []
for target in targets:
    train_predictions.append(K @ target)
    train_distortions.append(G @ target)
    w_readout = w @ target
    readouts.append(w_readout)
    test_predictions.append([w_readout @ test_sig for test_sig in test_signals])
    mses.append([mse(target, test_sig) for test_sig in test_signals])

# calculate the network kernel basis functions
K_shifted = np.zeros_like(K)
for j in range(K.shape[0]):
    K_shifted[j, :] = np.roll(K[j, :], shift=int(K.shape[1]/2)-j)
K_mean = np.mean(K_shifted, axis=0)
K_var = np.var(K_shifted, axis=0)
K_diag = np.diag(K)

# plot results
_, axes = plt.subplots(nrows=2, figsize=(12, 5))
s_all = np.concatenate(signals, axis=1)
inp_all = np.concatenate(inputs, axis=0)
s_all /= np.max(s_all)
inp_all /= np.max(inp_all)
ax = axes[0]
ax.plot(np.mean(s_all, axis=0), label="s")
ax.plot(inp_all, label="I_ext")
ax.legend()
ax.set_xlabel("time")
ax.set_title(f"Mean signal")
ax = axes[1]
im = ax.imshow(s_all, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.4)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title(f"Dim = {np.mean(dims)}")
plt.tight_layout()

_, axes = plt.subplots(ncols=2, figsize=(12, 6))
ax = axes[0]
ax.plot(target_1, label="target")
ax.plot(train_predictions[0], label="prediction")
ax.set_xlabel("time")
ax.set_title(f"T1")
ax.legend()
ax = axes[1]
ax.plot(target_2, label="target")
ax.plot(train_predictions[1], label="prediction")
ax.set_xlabel("time")
ax.set_title(f"T2")
ax.legend()
plt.tight_layout()

examples = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
fig, axes = plt.subplots(nrows=len(examples), figsize=(12, 9))
for i, ex in enumerate(examples):
    ax = axes[i]
    ax.plot(test_predictions[ex[0]][ex[1]], label="prediction")
    ax.plot(targets[ex[0]], label="target")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("s")
    ax.set_title(f"Stimulation phase: {test_phases[ex[1]]}, MSE: {mses[ex[0]][ex[1]]}")
plt.tight_layout()

_, axes = plt.subplots(ncols=2, figsize=(12, 6))
ax = axes[0]
im = ax.imshow(K, interpolation="none", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("K")
ax = axes[1]
im = ax.imshow(K_shifted, interpolation="none", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("K_shifted")
plt.tight_layout()

_, axes = plt.subplots(nrows=3, figsize=(12, 9))
ax = axes[0]
ax.plot(K_mean)
ax.set_title("K_mean")
ax = axes[1]
ax.plot(K_var)
ax.set_title("K_var")
ax = axes[2]
ax.plot(K_diag)
ax.set_title("K_diag")
plt.tight_layout()

# clear files up
plt.show()
clear(net)
