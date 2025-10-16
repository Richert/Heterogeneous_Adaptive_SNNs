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

def normalize(x):
    x = x - np.mean(x)
    x_std = np.std(x)
    return x / x_std if x_std > 0 else x

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

def get_prob(x, bins: int = 100):
    bins = np.linspace(0.0, 1.0, num=bins)
    counts, _ = np.histogram(x, bins=bins)
    count_sum = np.sum(counts)
    if count_sum > 0:
        return counts / count_sum
    return counts

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray) -> tuple:
    x = np.outer(fr_target, fr_source)
    y = np.repeat((fr_target**2).reshape(len(fr_target), 1), len(fr_source), axis=1)
    return x, y

def delta_w(t: float, w: np.ndarray, r: np.ndarray, eta: np.ndarray, J: float, b: float, a: float,
            t_old: np.ndarray, inp_f: Callable) -> np.ndarray:
    dt = t - t_old[0]
    t_old[0] = t
    w = w.reshape(N, N)
    r[:] = get_qif_fr(inp_f(t) + eta + noise * np.random.randn() * np.sqrt(dt) + J*np.dot(w, r))
    x, y = get_xy(r, r)
    return (a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))).flatten()

def get_w_solution(inp: Callable, w0: np.ndarray, fr: np.ndarray, eta: np.ndarray, J: float, b: float, a: float,
                   T: float, **kwargs) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(t, w, fr, eta, J, b, a, np.zeros(1,), inp), t_span=(0.0, T),
                     y0=np.asarray(w0), **kwargs)
    return sols.y[:, -1].reshape(N, N)

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

def uniform(N: int, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.linspace(-0.5, 0.5, N)

# meta parameters
path = "../results/rnn_results"
models = ["rate", "fre"]
fre_res = {"b": [], "Delta": [], "noise": [], "c_s": [], "c_t": [], "v": [], "h": []}
bs = [0.1]
noises = [0.0]
deltas = np.arange(0.2, 2.1, step=0.2)
n_reps = 10
N = 500

# rate model parameters
J = 5.0 / (0.5 * N)
eta = -0.5
a = 0.1
weights = {"Delta": [], "w_qif": [], "w_fre": [], "w_rate": []}
results = {"b": [], "Delta": [], "noise": [], "c_s": [], "c_t": [], "v": [], "h": [], "model": []}

# simulation parameters
T = 2000.0
dt = 1e-3
noise = 0.0
inp_noise = 10.0
inp_sigma = 1.0/dt
time = np.arange(0.0, T+dt, dt)
inp = np.zeros((int(T/dt)+1,))
solver_kwargs = {"t_eval": [0.0, T], "method": "RK23", "atol": 1e-5}

# calculate weight statistics
for b in bs:
    for noise in noises:
        for delta in deltas:
            for rep in range(n_reps):

                try:

                    # fre model
                    if "fre" in models:
                        data = pickle.load(open(f"{path}/fre_mp_{int(b*10)}_{int(noise)}_{int(delta*10.0)}_{rep}.pkl",
                                                "rb"))
                        w_fre = data["W"]
                        etas = data["eta"]
                        w_s = np.mean(w_fre, axis=1)
                        w_t = np.mean(w_fre, axis=0)

                        results["b"].append(b)
                        results["Delta"].append(delta)
                        results["noise"].append(noise)
                        results["c_s"].append(correlate(etas, w_s))
                        results["c_t"].append(correlate(etas, w_t))
                        results["v"].append(np.var(w_fre.flatten()))
                        results["h"].append(entropy(get_prob(w_fre.flatten())))
                        results["model"].append("MFE")
                        if rep == 0:
                            weights["Delta"].append(delta)
                            weights["w_fre"].append(w_fre)

                    # qif model
                    if "qif" in models:
                        data = pickle.load(open(f"{path}/qif_{int(b * 10)}_{int(noise)}_{int(delta * 10.0)}_{rep}.pkl",
                                                "rb"))
                        w_qif = data["W"]
                        etas = data["eta"]
                        w_s = np.mean(w_qif, axis=1)
                        w_t = np.mean(w_qif, axis=0)

                        results["b"].append(b)
                        results["Delta"].append(delta)
                        results["noise"].append(noise)
                        results["c_s"].append(correlate(etas, w_s))
                        results["c_t"].append(correlate(etas, w_t))
                        results["v"].append(np.var(w_qif.flatten()))
                        results["h"].append(entropy(get_prob(w_qif.flatten())))
                        results["model"].append("QIF")
                        if rep == 0:
                            weights["w_qif"].append(w_qif)

                    # rate model simulation
                    if "rate" in models:
                        etas = uniform(N, eta, delta)
                        fr = get_qif_fr(etas)
                        w0 = np.random.uniform(0.0, 1.0, size=(int(N * N),))
                        inp = inp_noise * np.random.randn(*inp.shape)
                        inp = gaussian_filter1d(inp, sigma=inp_sigma)
                        w_rate = get_w_solution(interp1d(time, inp), w0, fr, etas, J, b, a, T, **solver_kwargs)
                        w0 = w0.reshape(N, N)
                        w_s = np.mean(w_rate, axis=1)
                        w_t = np.mean(w_rate, axis=0)

                        results["b"].append(b)
                        results["Delta"].append(delta)
                        results["noise"].append(noise)
                        results["c_s"].append(correlate(etas, w_s))
                        results["c_t"].append(correlate(etas, w_t))
                        results["v"].append(np.var(w_rate.flatten()))
                        results["h"].append(entropy(get_prob(w_rate.flatten())))
                        results["model"].append("SSR")
                        if rep == 0:
                            weights["w_rate"].append(w_rate)

                    print(f"Finished {rep+1} out of {n_reps} simulations for b = {b}, noise = {noise} and Delta = {delta}.")

                except FileNotFoundError:
                    print(f"FAILED {rep+1} out of {n_reps} simulations for b = {b}, noise = {noise} and Delta = {delta}.")

results = pd.DataFrame.from_dict(results)

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

fig = plt.figure(figsize=(6, 3.5))
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0., wspace=0.)
grid = fig.add_gridspec(nrows=2, ncols=3)

# source correlation
ax = fig.add_subplot(grid[0, 0])
sb.lineplot(results, x="Delta", y="c_s", hue="model", ax=ax, palette="Dark2", legend="auto")
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$R(\eta_j, w_j)$")

# target correlation
ax = fig.add_subplot(grid[0, 1])
sb.lineplot(results, x="Delta", y="c_t", hue="model", legend=False, ax=ax, palette="Dark2")
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$R(\eta_i, w_i)$")

# weight matrices
delta = 0.9
idx = np.argmin(np.abs(np.asarray(weights["Delta"]) - delta)).squeeze()
ws = [weights[f"w_{m}"][idx] for m in models]
for i, (w, title) in enumerate(zip(ws, models)):
    ax = fig.add_subplot(grid[1, i])
    im = ax.imshow(w, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    if i == 2:
        plt.colorbar(im, ax=ax, shrink=0.8)
    if i == 0:
        ax.set_ylabel("neuron")
    ax.set_title(title)
fig.canvas.draw()
plt.savefig(f"../results/figures/rnn_results.svg")
plt.show()
