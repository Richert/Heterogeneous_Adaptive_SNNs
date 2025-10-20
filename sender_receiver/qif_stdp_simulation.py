import numpy as np
import sys
import pickle
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

@njit
def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, trace_source: np.ndarray, trace_target: np.ndarray) -> tuple:
    if condition == "hebbian":
        x = trace_source*fr_target
        y = trace_target*fr_source
    elif condition == "antihebbian":
        x = trace_target*fr_source
        y = trace_source*fr_target
    else:
        raise ValueError(f"Invalid condition: {condition}.")
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

def solve_ivp(T: float, dt: float, w0: np.ndarray, eta: np.ndarray, tau_s: float, tau_u: float, J: float, a: float,
              b: float, N: int):

    y = np.zeros((4*N,))
    y[3*N:] = w0
    spikes = np.zeros((N,))
    t = 0.0

    while t < T:
        spiking(y, spikes, N)
        dy = qif_rhs(y, spikes, eta, tau_s, tau_u, J, a, b, N)
        y = y + dt * dy
        t += dt

    return y[3 * N:-1]

# parameter definition
path = "/home/richard-gast/PycharmProjects/Heterogeneous_Adaptive_SNNs"
save_results = False
condition = "hebbian" #str(sys.argv[-3])
noise = 0.0 #float(sys.argv[-2]) * 1e3
b = 0.1 #float(sys.argv[-1])
J = 5.0
N = 200
m = 10
eta_t = 0.0 if J > 0 else 2.0
eta_min, eta_max = -2.0, 3.0
eta_s = np.linspace(eta_min, eta_max, N)
w0s = np.linspace(start=0.0, stop=1.0, num=m)
a = 0.1
tau_s = 1.0
tau_u = 30.0
v_cutoff = 100.0
res = {"eta": [], "w": [], "w0": []}

# simulation
T = 1000.0
dt = 1e-3
for w0 in w0s:

    # get weight solutions
    etas = np.asarray(eta_s.tolist() + [eta_t])
    ws = solve_ivp(T, dt, np.zeros_like(etas) + w0, etas, tau_s, tau_u, J, a, b, N+1)
    ws[ws < 0.0] = 0.0
    ws[ws > 1.0] = 1.0

    # save results
    for eta, w in zip(eta_s, ws):
        res["w0"].append(w0)
        res["w"].append(w)
        res["eta"].append(eta)

# save results
conn = int(J)
if save_results:
    pickle.dump(
        res,
        open(f"{path}/results/qif_stdp_J{conn}_{plasticity}_{condition}_{int(noise)}_{int(b*100)}.pkl", "wb")
    )

# plotting
res = pd.DataFrame.from_dict(res)
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 2

fig, ax = plt.subplots(figsize=(4, 2))
res = pd.DataFrame.from_dict(res)
sb.lineplot(res, x="eta", y="w", palette="Dark2", ax=ax, errorbar=("pi", 90), legend=False)
ax.set_xlabel(r"$\eta$")
ax.set_ylabel(r"$w$")
ax.set_title(r"Solutions for $w$")
fig.canvas.draw()
# plt.savefig(f"../results/figures/weight_update_rule.svg")
plt.show()