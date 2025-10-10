import numpy as np
import sys
import pickle
from numba import njit

@njit
def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, trace_source: np.ndarray, trace_target: np.ndarray) -> tuple:
    if plasticity == "oja_rate":
        if condition == "hebbian":
            x = fr_source*fr_target
            y = fr_target*fr_target
        elif condition == "antihebbian":
            x = fr_source*fr_source
            y = fr_source*fr_target
        else:
            raise ValueError(f"Invalid condition: {condition}.")
    elif plasticity == "oja_trace":
        if condition == "hebbian":
            x = trace_source*trace_target
            y = trace_target*fr_target
        elif condition == "antihebbian":
            x = trace_source*fr_source
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
path = "/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs"
save_results = True
plasticity = str(sys.argv[-4])
condition = str(sys.argv[-3])
noise = float(sys.argv[-2]) * 1e3
b = float(sys.argv[-1])
J = 5.0
N = 1000
m = 10
eta_t = 0.0 if J > 0 else 2.0
eta_min, eta_max = -2.0, 3.0
eta_s = np.linspace(eta_min, eta_max, N)
w0s = np.linspace(start=0.0, stop=1.0, num=m)
a = 0.01
tau_s = 1.0
tau_u = 30.0
v_cutoff = 100.0
res = {"eta": [], "w": [], "w0": []}

# simulation
T = 10000.0
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
        open(f"{path}/results/qif_simulation_J{conn}_{plasticity}_{condition}_{int(noise)}_{int(b*100)}.pkl", "wb")
    )

# # plotting
# res = pd.DataFrame.from_dict(res)
# print(f"Plotting backend: {plt.rcParams['backend']}")
# plt.rcParams["font.family"] = "sans"
# plt.rc('text', usetex=True)
# plt.rcParams['figure.constrained_layout.use'] = True
# plt.rcParams['figure.dpi'] = 200
# plt.rcParams['font.size'] = 12.0
# plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['axes.labelsize'] = 12
# plt.rcParams['lines.linewidth'] = 1.0
# markersize = 2
#
# fig, axes = plt.subplots(ncols=2, nrows=len(noise_lvls), figsize=(6, 1.5*len(noise_lvls)))
# for j, c in enumerate(conditions):
#     res_tmp = res.loc[res.loc[:, "condition"] == c, :]
#     for i, noise in enumerate(noise_lvls):
#         res_tmp2 = res_tmp.loc[res_tmp.loc[:, "noise"] == noise, :]
#         ax = axes[i, j]
#         sb.lineplot(res_tmp2, x="eta", y="w", hue="b", palette="Dark2", ax=ax, errorbar=("pi", 100), legend=False)
#         ax.set_xlabel(r"$\eta$")
#         ax.set_ylabel(r"$w$")
#         # ax.get_legend().set_title(r"$b$")
#         c_title = "Hebbian Learning" if c == "hebbian" else "Anti-Hebbian Learning"
#         ax.set_title(f"{c}, noise lvl = {noise}")
#
# fig.canvas.draw()
# # plt.savefig(f"../results/figures/weight_update_rule.svg")
# plt.show()