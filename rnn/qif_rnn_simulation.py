import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
import sys
import pickle

@njit
def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, trace_source: np.ndarray, trace_target: np.ndarray) -> tuple:
    if condition == "hebbian":
        x = np.outer(fr_target, trace_source)
        y = np.repeat(fr_target*trace_target, N).reshape(N, N)
    elif condition == "antihebbian":
        x = np.repeat(fr_source*trace_source, N).reshape(N, N).T
        y = np.outer(fr_target, trace_source)
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

@njit
def qif_rhs(y: np.ndarray, w: np.ndarray, spikes: np.ndarray, eta: np.ndarray, inp: np.ndarray, tau_s: float,
            tau_u: float, J: float, a: float, b: float) -> tuple:
    v, s, u = y[:N], y[N:2*N], y[2*N:]
    dy = np.zeros_like(y)
    x, y = get_xy(s, s, u, u)
    dy[:N] = v**2 + eta + J*np.dot(w, s) + inp
    dy[N:2*N] = (spikes-s) / tau_s
    dy[2*N:] = (spikes-u) / tau_u
    dw = a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))
    return dy, dw

@njit
def spiking(y: np.ndarray, spikes: np.ndarray):
    idx = np.argwhere((v_cutoff - y[:N]) < 0.0).flatten()
    spikes[:] = 0.0
    y[idx] = -y[idx]
    spikes[idx] = 1.0/dt

def solve_ivp(dt: float, w: np.ndarray, eta: np.ndarray, inp: np.ndarray, tau_s: float, tau_u: float, J: float,
              a: float, b: float, sr: int = 100) -> tuple:

    y = np.zeros((3*N,))
    spikes = np.zeros((N,))

    s_col = np.zeros((int(inp.shape[0]/sr), N))
    ss = 0
    for step in range(inp.shape[0]):
        spiking(y, spikes)
        dy, dw = qif_rhs(y, w, spikes, eta, inp[step], tau_s, tau_u, J, a, b)
        y = y + dt * dy
        w = w + dt * dw
        if step % sr == 0:
            s_col[ss, :] = y[N:2*N]
            ss += 1

    return s_col, w

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

def uniform(N: int, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.linspace(-0.5, 0.5, N)

# parameters
path = "/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs"
rep = int(sys.argv[-1])
b = float(sys.argv[-2])
Delta = float(sys.argv[-3])
noise_lvl = float(sys.argv[-4])
condition = "hebbian"
N = 500
M = 20
p = 1.0
edge_vars = {
    "a": 0.1, "b": b
}
eta = -0.5
etas_mp = uniform(M, eta, Delta)
etas = []
for eta_mp in etas_mp:
    etas.extend(lorentzian(int(N/M), eta_mp, Delta/(2*M)).tolist())
etas = np.asarray(etas)
node_vars = {"J": 5.0 / (0.5*p*N), "eta": etas, "tau_u": 30.0, "tau_s": 1.0}
T = 1000.0
dt = 1e-3
global_noise = 10.0
noise_sigma = 1.0/dt
v_cutoff = 100.0

# define extrinsic input
# inp = np.zeros((int(T/dt), N))
# noise = noise_lvl*np.random.randn(*inp.shape) + global_noise*np.random.randn(inp.shape[0], 1)
inp = np.zeros((int(T/dt), 1))
noise = global_noise*np.random.randn(inp.shape[0], 1)
noise = gaussian_filter1d(noise, sigma=noise_sigma, axis=0)
inp += noise

# run simulation
w0 = np.random.uniform(0.0, 1.0, size=(N, N,))
s, W = solve_ivp(dt, w0, node_vars["eta"], inp, node_vars["tau_s"], node_vars["tau_u"], node_vars["J"],
                 edge_vars["a"], edge_vars["b"])
W[W < 0.0] = 0.0
W[W > 1.0] = 1.0

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(W)
# fig, ax = plt.subplots(figsize=(10, 4))
# ax.plot(s)
# ax.plot(np.mean(s, axis=1), color="black")
# plt.show()

pickle.dump(
    {"W": W, "eta": etas, "b": b, "Delta": Delta, "noise": noise_lvl},
    open(f"{path}/results/rnn_results/qif_{int(b*10)}_{int(noise_lvl)}_{int(Delta*10.0)}_{rep}.pkl", "wb")
)
