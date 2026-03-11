import numpy as np
import sys
import pickle
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

@njit
def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, up_source: np.ndarray, up_target: np.ndarray,
           ud_source: np.ndarray, ud_target: np.ndarray) -> tuple:
    if condition == "oja":
        ltp = fr_source*fr_target
        ltd = fr_target*fr_target
    elif condition == "stdp_sym":
        ltp = up_target*up_source
        ltd = ud_target*ud_source
    elif condition == "stdp_asym":
        ltp = fr_target*up_source
        ltd = fr_source*ud_target
    elif condition == "antihebbian":
        ltp = fr_source*fr_source
        ltd = fr_target*ud_source
    else:
        raise ValueError(f"Invalid plasticity condition: {condition}.")
    return ltp, ltd

@njit
def qif_rhs(y: np.ndarray, spikes: np.ndarray, eta: np.ndarray, tau_s: float, tau_up: float, tau_ud: float, J: float,
            ap: float, ad: float, r: float, N: int):
    v, s, up, ud, w, x = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:4*N], y[4*N:5*N], y[5*N:6*N]
    dy = np.zeros_like(y)
    ltp, ltd = get_xy(s[:], np.zeros_like(s) + s[-1], up[:], np.zeros_like(up) + up[-1], ud[:], np.zeros_like(ud) + ud[-1])
    dy[:N] = v**2 + eta
    dy[N-1] += J*np.dot(w[:-1], s[:-1]) / (N-1)
    dy[N:2*N] = (spikes-s) / tau_s
    dy[2*N:3*N] = (spikes-up) / tau_up
    dy[3*N:4*N] = (spikes-ud) / tau_ud
    dy[4*N:5*N] = x*w*(0.5-w)*(1-w) + noise * np.random.randn(N) * np.sqrt(dt)
    dy[5*N:6*N] = ap*ltp - ad*ltd + r*x - x**3
    return dy

@njit
def spiking(y: np.ndarray, spikes: np.ndarray, N: int):
    idx = np.argwhere((v_cutoff - y[:N]) < 0.0).flatten()
    spikes[:] = 0.0
    y[idx] = -y[idx]
    spikes[idx] = 1.0/dt

def solve_ivp(T: float, dt: float, w0: np.ndarray, eta: np.ndarray, tau_s: float, tau_up: float, tau_ud: float, J: float,
            ap: float, ad: float, r: float, N: int):

    y = np.zeros((6*N,))
    y[4*N:5*N] = w0
    spikes = np.zeros((N,))
    t = 0.0

    while t < T:
        spiking(y, spikes, N)
        dy = qif_rhs(y, spikes, eta, tau_s, tau_up, tau_ud, J, ap, ad, r, N)
        y = y + dt * dy
        t += dt

    return y[4*N:5*N-1]

# parameter definition
condition = "stdp_asym"
noise = 0.01
J = 5.0
N = 100
m = 5
eta_t = 0.0 if J > 0 else 2.0
eta_min, eta_max = -2.0, 3.0
eta_s = np.linspace(eta_min, eta_max, N)
w0s = np.linspace(start=0.01, stop=0.99, num=m)
tau_s = 1.0
tau_up = 10.0
tau_ud = 20.0
a = 0.01
ar = 1.5
ap = a * ar
ad = a / ar
r = 1.0
v_cutoff = 100.0
res = {"eta": [], "w": [], "w0": []}

# simulation
T = 1000.0
dt = 1e-3
for w0 in w0s:

    # get weight solutions
    etas = np.asarray(eta_s.tolist() + [eta_t])
    ws = solve_ivp(T, dt, np.zeros_like(etas) + w0, etas, tau_s, tau_up, tau_ud, J, ap, ad, r, N+1)
    ws[ws < 0.0] = 0.0
    ws[ws > 1.0] = 1.0

    # save results
    for eta, w in zip(eta_s, ws):
        res["w0"].append(w0)
        res["w"].append(w)
        res["eta"].append(eta)

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
sb.lineplot(res, x="eta", y="w", hue="w0", palette="Dark2", ax=ax, legend=True)
ax.set_xlabel(r"$\eta$")
ax.set_ylabel(r"$w$")
ax.set_title(r"Solutions for $w$")
fig.canvas.draw()
# plt.savefig(f"../results/figures/weight_update_rule.svg")
plt.show()