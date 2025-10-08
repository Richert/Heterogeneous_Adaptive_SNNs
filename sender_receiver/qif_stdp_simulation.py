import numpy as np
from scipy.stats import entropy
import sys
import pickle
from numba import njit

def get_prob(x, bins: int = 100):
    counts, _ = np.histogram(x, bins=bins)
    return counts / np.sum(counts)

@njit
def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, trace_source: np.ndarray, trace_target: np.ndarray) -> tuple:
    if plasticity == "oja_rate":
        if condition == "hebbian":
            x = trace_source*fr_target
            y = trace_target*fr_target
        elif condition == "antihebbian":
            x = trace_source*fr_source
            y = trace_source*fr_target
        else:
            raise ValueError(f"Invalid condition: {condition}.")
    elif plasticity == "oja_trace":
        if condition == "hebbian":
            x = trace_source*trace_target
            y = trace_target*trace_target
        elif condition == "antihebbian":
            x = trace_source*trace_source
            y = trace_source*trace_target
        else:
            raise ValueError(f"Invalid condition: {condition}.")
    elif plasticity == "stdp":
        if condition == "hebbian":
            x = trace_source*fr_target
            y = trace_target*fr_source
        elif condition == "antihebbian":
            x = trace_target*fr_source
            y = trace_source*fr_target
        else:
            raise ValueError(f"Invalid condition: {condition}.")
    else:
        raise ValueError(f"Invalid plasticity rule: {condition}.")
    return x, y

@njit
def qif_rhs(y: np.ndarray, spikes: np.ndarray, eta: np.ndarray, tau_s: float, tau_u: float, J: float, a: float,
            b: float, N: int):
    v, s, u, w = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:]
    dy = np.zeros_like(y)
    x, y = get_xy(s[:], np.zeros_like(s) + s[-1], u[:], np.zeros_like(u) + u[-1])
    dy[:N] = v**2 + eta + noise * np.random.randn(N) * np.sqrt(dt)
    dy[N-1] += J*np.dot(w[:-1], s[:-1]) / (N-1)
    dy[N:2*N] = (spikes-s) / tau_s
    dy[2 * N:3 * N] = (spikes - u) / tau_u
    dy[3*N:] = a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))
    return dy

@njit
def spiking(y: np.ndarray, spikes: np.ndarray, N: int):
    idx = np.argwhere((v_cutoff - y[:N]) < 0.0).flatten()
    spikes[:] = 0.0
    y[idx] = -y[idx]
    spikes[idx] = 1.0/dt

def solve_ivp(T: float, dt: float, eta: np.ndarray, tau_s: float, tau_u: float, J: float, a: float, b: float, N: int):

    y = np.zeros((4*N,))
    y[3*N:] = np.random.uniform(0.01, 0.99, size=(N,))
    spikes = np.zeros((N,))
    t = 0.0

    while t < T:
        spiking(y, spikes, N)
        dy = qif_rhs(y, spikes, eta, tau_s, tau_u, J, a, b, N)
        y = y + dt * dy
        t += dt

    return y[3*N:-1]

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

# parameter definition
path = "/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/results"
rep = int(sys.argv[-1])
J = float(sys.argv[-2])
plasticity = str(sys.argv[-3])
condition = str(sys.argv[-4])
noise_lvls = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]) * 1e3
distribution = "gaussian"
N = 1000
m = 10
eta = 1.0
deltas = np.linspace(0.1, 1.5, num=m)
target_eta = 0.0 if J > 0 else 2.0
a = 0.01
tau_s = 1.0
tau_u = 30.0
bs = [0.0, 0.05, 0.2]
v_cutoff = 100.0
res = {"b": [], "w": [], "C": [], "H":[], "V": [], "delta": [], "noise": []}

# simulation parameters
T = 10000.0
dt = 1e-3
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    for noise in noise_lvls:
        for Delta in deltas:

            # define source firing rate distribution
            etas = np.asarray(f(N, eta, Delta).tolist() + [target_eta])

            # solve equations
            w = solve_ivp(T, dt, etas, tau_s, tau_u, J, a, b, len(etas))

            # calculate entropy of weight distribution
            h_w = entropy(get_prob(w))

            # calculate variance of weight distribution
            v = np.var(w)

            # calculate correlation between source etas and weights
            c = np.corrcoef(etas[:-1], w)[0, 1]

            # save results
            print(f"Finished simulations for b = {b}, noise = {int(noise)}, and Delta = {np.round(Delta, decimals=1)}")
            res["b"].append(b)
            res["delta"].append(Delta)
            res["noise"].append(noise)
            res["w"].append(w)
            res["C"].append(c)
            res["H"].append(h_w)
            res["V"].append(v)

# save results
conn = int(J)
f = open(f"{path}/qif_{plasticity}_{condition}_{int(tau_u)}_{conn}_{rep}.pkl", "wb")
pickle.dump({"trial": rep, "J": J, "tau_u": tau_u, "condition": condition, "results": res}, f)
