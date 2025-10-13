import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import entropy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from typing import Callable
import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

def get_prob(x, bins: int = 100):
    bins = np.linspace(0.0, 1.0, num=bins)
    counts, _ = np.histogram(x, bins=bins)
    return counts / np.sum(counts)

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = np.outer(fr_target, fr_source)
        y = np.repeat((fr_target**2).reshape(len(fr_target), 1), len(fr_source), axis=1)
    elif condition == "antihebbian":
        x = np.repeat((fr_source**2).reshape(1, len(fr_source)), len(fr_target), axis=0)
        y = np.outer(fr_target, fr_source)
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def delta_w(t: float, w: np.ndarray, r: np.ndarray, eta: np.ndarray, J: float, b: float, a: float, condition: str,
            t_old: np.ndarray, inp_f: Callable) -> np.ndarray:
    dt = t - t_old[0]
    t_old[0] = t
    w = w.reshape(N, N)
    r[:] = get_qif_fr(inp_f(t) + eta + noise * np.random.randn() * np.sqrt(dt) + J*np.dot(w, r) / N)
    x, y = get_xy(r, r, condition=condition)
    return (a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))).flatten()

def get_w_solution(inp: Callable, w0: np.ndarray, fr: np.ndarray, eta: np.ndarray, J: float, b: float, a: float,
                   T: float, **kwargs) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(t, w, fr, eta, J, b, a, condition, np.zeros(1,), inp), t_span=(0.0, T),
                     y0=np.asarray(w0), **kwargs)
    return sols.y[:, -1].reshape(N, N)

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# parameter definition
save_results = False
condition = "hebbian"
distribution = "lorentzian"
Deltas = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
J = 30.0
N = 50
m = 5
eta = -5.0 if J > 0 else 2.0
a = 0.1
bs = [0.0, 0.05, 0.2]
weights = {"b": [], "Delta": [], "source": [], "target": [], "w": [], "w0": []}
results = {"b": [], "Delta": [], "eta": [], "c_s": [], "c_t": [], "v_s": [], "v_t": [], "h_s": [], "h_t": [],
           "c2_s": [], "c2_t": []}

# simulation parameters
T = 1000.0
dt = 1e-3
noise = 0.005
inp_noise = 80.0
inp_sigma = 1000.0
inp = np.zeros((int(T/dt)+1,))
inp += inp_noise * np.random.randn(*inp.shape)
inp = gaussian_filter1d(inp, sigma=inp_sigma)
time = np.arange(0.0, T+dt, dt)
inp_f = interp1d(time, inp)
solver_kwargs = {"t_eval": [0.0, T], "method": "RK23", "atol": 1e-5}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    for Delta in Deltas:

        # define initial condition
        etas = f(N, eta, Delta)
        fr = get_qif_fr(etas)
        w0 = np.random.uniform(0.01, 0.99, size=(int(N*N),))

        # get weight solutions
        w = get_w_solution(inp_f, w0, fr, etas, J, b, a, T, **solver_kwargs)
        w0 = w0.reshape(N, N)

        for i in range(N):

            # calculate weight statistics
            results["c_s"].append(np.corrcoef(etas, w[i, :])[0, 1])
            results["c_t"].append(np.corrcoef(etas, w[:, i])[0, 1])
            results["c2_s"].append(np.corrcoef(etas, np.mean(w, axis=0))[0, 1])
            results["c2_t"].append(np.corrcoef(etas, np.mean(w, axis=1))[0, 1])
            results["v_s"].append(np.var(w[i, :]))
            results["v_t"].append(np.var(w[:, i]))
            results["h_s"].append(entropy(get_prob(w[i, :])))
            results["h_t"].append(entropy(get_prob(w[:, i])))
            results["eta"].append(etas[i])
            results["Delta"].append(Delta)
            results["b"].append(b)

            # save weights
            for j in range(N):
                weights["b"].append(b)
                weights["Delta"].append(Delta)
                weights["source"].append(j)
                weights["target"].append(i)
                weights["w"].append(w[i, j])
                weights["w0"].append(w0[i, j])
        print(f"Finished simulations for b = {b} and Delta = {Delta}.")

# save results
conn = int(J)
if save_results:
    pickle.dump({"condition": condition, "J": J, "noise": noise, "results": results, "weights": weights},
                open(f"../results/rate_rnn_simulation_{condition}_{conn}.pkl", "wb"))

# create dataframes
results = pd.DataFrame.from_dict(results)
weights = pd.DataFrame.from_dict(weights)

# plotting
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

# # initial weights
# fig, axes = plt.subplots(ncols=len(bs), nrows=len(Deltas), figsize=(3*len(bs), 2*len(Deltas)))
# for j, b in enumerate(bs):
#     res_tmp = res.loc[res.loc[:, "b"] == b, :]
#     for i, Delta in enumerate(Deltas):
#         res_tmp2 = res_tmp.loc[res_tmp.loc[:, "Delta"] == Delta, :]
#         w = res_tmp2.pivot(index="target", columns="source", values="w0")
#         ax = axes[i, j]
#         sb.heatmap(w, vmin=0.0, vmax=1.0, ax=ax)
#         ax.set_title(fr"$b = {b}$, $\Delta = {Delta}$")
# fig.suptitle("Initial Weights")
# fig.canvas.draw()

# final weights
fig, axes = plt.subplots(ncols=len(bs), nrows=len(Deltas), figsize=(3*len(bs), 2*len(Deltas)))
for j, b in enumerate(bs):
    res_tmp = weights.loc[weights.loc[:, "b"] == b, :]
    for i, Delta in enumerate(Deltas):
        res_tmp2 = res_tmp.loc[res_tmp.loc[:, "Delta"] == Delta, :]
        w = res_tmp2.pivot(index="target", columns="source", values="w")
        ax = axes[i, j]
        sb.heatmap(w, vmin=0.0, vmax=1.0, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(fr"$b = {b}$, $\Delta = {Delta}$")
fig.suptitle("Final Weights")
fig.canvas.draw()

# weight statistics
fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(12, 5))
ax = axes[0, 0]
sb.lineplot(results, x="Delta", y="c_s", hue="b", ax=ax, palette="Dark2")
ax = axes[1, 0]
sb.lineplot(results, x="Delta", y="c_t", hue="b", legend=False, ax=ax, palette="Dark2")
ax = axes[0, 1]
sb.lineplot(results, x="Delta", y="v_s", hue="b", legend=False, ax=ax, palette="Dark2")
ax = axes[1, 1]
sb.lineplot(results, x="Delta", y="v_t", hue="b", legend=False, ax=ax, palette="Dark2")
ax = axes[0, 2]
sb.lineplot(results, x="Delta", y="h_s", hue="b", legend=False, ax=ax, palette="Dark2")
ax = axes[1, 2]
sb.lineplot(results, x="Delta", y="h_t", hue="b", legend=False, ax=ax, palette="Dark2")
ax = axes[0, 3]
sb.lineplot(results, x="Delta", y="c2_s", hue="b", legend=False, ax=ax, palette="Dark2")
ax = axes[1, 3]
sb.lineplot(results, x="Delta", y="c2_t", hue="b", legend=False, ax=ax, palette="Dark2")
fig.suptitle("Weight Statistics")
fig.canvas.draw()

plt.show()
