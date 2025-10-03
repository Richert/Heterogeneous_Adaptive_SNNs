import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import sys
import pickle
from numba import njit

def get_prob(x, bins: int = 100):
    counts, _ = np.histogram(x, bins=bins)
    return counts / np.sum(counts)

@njit
def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = fr_source*fr_target
        y = fr_target*fr_target
    elif condition == "antihebbian":
        x = fr_source*fr_source
        y = fr_source*fr_target
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

@njit
def qif_rhs(y: np.ndarray, spikes: np.ndarray, eta: np.ndarray, tau_s: float, tau_u: float, J: float, a: float,
            b: float, N: int, condition: str):
    v, s, w = y[:N], y[N:2*N], y[2*N:]
    dy = np.zeros_like(y)
    x, y = get_xy(s[:], np.zeros_like(s) + s[-1], condition=condition)
    dy[:N] = v**2 + eta
    dy[N-1] += J*np.dot(w[:-1], s[:-1]) / (N-1)
    dy[N:2*N] = (spikes-s) / tau_s
    dy[2*N:] = a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))
    return dy

@njit
def spiking(y: np.ndarray, spikes: np.ndarray, dt: float, v_cutoff: float, N: int):
    idx = np.argwhere((v_cutoff - y[:N]) < 0.0).flatten()
    spikes[:] = 0.0
    y[idx] = -y[idx]
    spikes[idx] = 1.0/dt

def solve_ivp(T: float, dt: float, eta: np.ndarray, tau_s: float, tau_u: float, J: float,
              a: float, b: float, v_cutoff: float, N: int, condition: str):

    y = np.zeros((3*N,))
    y[2*N:] = np.random.uniform(0.01, 0.99, size=(N,))
    spikes = np.zeros((N,))
    t = 0.0

    while t < T:
        spiking(y, spikes, dt, v_cutoff, N)
        dy = qif_rhs(y, spikes, eta, tau_s, tau_u, J, a, b, N, condition)
        y = y + dt * dy
        t += dt

    return y[2*N:-1]

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

# parameter definition
rep = int(sys.argv[-1])
J = float(sys.argv[-2])
tau = float(sys.argv[-3])
condition = str(sys.argv[-4])
distribution = "gaussian"
N = 1000
m = 10
eta = 1.0
deltas = np.linspace(0.1, 1.5, num=m)
target_eta = 0.0 if J > 0 else 2.0
a = 0.1
bs = [0.0, 0.05, 0.2]
tau_s = 10.0
tau_u = 10.0
v_cutoff = 100.0
res = {"b": bs, "w": {}, "C": {}, "H": {}, "V": {}, "deltas": deltas}

# simulation parameters
T = 2000.0
dt = 1e-3
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    ws, cs, hs, vs = [], [], [], []
    for Delta in deltas:

        # define source firing rate distribution
        etas = np.asarray(f(N, eta, Delta).tolist() + [target_eta])

        # solve equations
        w = solve_ivp(T, dt, etas, tau_s, tau_u, J, a, b, v_cutoff, N+1, condition)
        print(f"Finished simulations for b = {b} and Delta = {np.round(Delta, decimals=1)}")

        # calculate entropy of weight distribution
        h_w = entropy(get_prob(w))

        # calculate variance of weight distribution
        v = np.var(w)

        # calculate correlation between source etas and weights
        c = np.corrcoef(etas[:-1], w)[0, 1]

        # save results
        ws.append(w)
        cs.append(c)
        hs.append(h_w)
        vs.append(v)

    # save results
    res["w"][b] = np.asarray(ws)
    res["H"][b] = np.asarray(hs)
    res["C"][b] = np.asarray(cs)
    res["V"][b] = np.asarray(vs)

# save results
conn = int(J)
f = open(f"/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/results/qif_oja_simulation_{condition}_{conn}_{rep}.pkl", "wb")
pickle.dump(res, f)

# # plotting
# fig, axes = plt.subplots(nrows=3, ncols=len(bs), figsize=(3*len(bs), 5), layout="constrained")
# ticks = np.arange(0, m, int(m/5))
# for i, b in enumerate(bs):
#
#     # weight distribution
#     ax = axes[0, i]
#     im = ax.imshow(np.asarray(res["w"][b]).T, aspect="auto", interpolation="none", cmap="viridis", vmax=1.0, vmin=0.0)
#     ax.set_ylabel("neuron")
#     ax.set_xlabel("Delta")
#     ax.set_xticks(ticks, labels=np.round(deltas[ticks], decimals=1))
#     if i == len(bs) - 1:
#         plt.colorbar(im, ax=ax, shrink=0.8)
#     ax.set_title(f"w (b = {b})")
#
#     # correlation
#     ax = axes[1, i]
#     ax.plot(deltas, res["C"][b])
#     ax.set_xlabel("Delta")
#     ax.set_ylabel("C")
#     ax.set_title("correlation(w, eta)")
#
#     # entropy
#     ax = axes[2, i]
#     ax.plot(deltas, res["H"][b])
#     ax.set_xlabel("Delta")
#     ax.set_ylabel("H")
#     ax.set_title("entropy(w)")
#
#     # # variance
#     # ax = axes[3, i]
#     # ax.plot(deltas, res["V"][b])
#     # ax.set_xlabel("Delta")
#     # ax.set_ylabel("var")
#     # ax.set_title("variance(w)")
#
# fig.suptitle(f"{'Anti-Hebbian' if 'antihebbian' in condition else 'Hebbian'} Learning (J = {conn}, QIF Simulation)")
# fig.canvas.draw()
# plt.savefig(f"../results/qif_oja_simulation_{condition}_{int(J)}.svg")
# plt.show()
