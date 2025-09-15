import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = np.outer(fr_target, fr_source)
        y = np.repeat((fr_target**2).reshape(len(fr_target), 1), len(fr_source), axis=1)
    elif condition == "antihebbian":
        x = np.repeat((fr_source**2).reshape(len(fr_source), 1), len(fr_target), axis=1)
        y = np.outer(fr_target, fr_source)
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def delta_w(w: np.ndarray, etas: np.ndarray, r: np.ndarray, J: float, b: float, a: float, condition: str) -> np.ndarray:
    w = np.reshape(w, (N, N))
    r[:] = get_qif_fr(etas + J* (w @ r))
    x, y = get_xy(r, r, condition=condition)
    return (a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))).flatten()

def get_w_solution(w0: np.ndarray, etas: np.ndarray, J: float, b: float, a: float, T: float, condition: str,
                   **kwargs) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(w, etas, get_qif_fr(etas), J, b, a, condition), t_span=(0.0, T), y0=w0,
                     **kwargs)
    ws = sols.y[:, -1]
    return np.reshape(ws, (N, N))

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta*np.random.randn(N)
    return np.sort(etas)

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# parameter definition
condition = "hebbian"
distribution = "gaussian"
N = 100
m = 4
eta = 0.0
deltas = np.linspace(0.1, 3.0, num=m)
a = 0.1
J = 0.1
bs = [0.0, 0.25]
res = {"b": [], "delta": [], "w": [], "eta": []}

# simulation parameters
T = 400.0
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    for Delta in deltas:

        # define initial condition
        inp = f(N, eta, Delta)
        w0 = np.random.uniform(0.01, 0.99, size=(int(N*N),))

        # get weight solutions
        w = get_w_solution(w0, inp, J, b, a, T, condition, **solver_kwargs)

        # save results
        res["w"].append(w)
        res["b"].append(b)
        res["delta"].append(Delta)
        res["eta"].append(inp)
        print(f"Finished simulations for b = {b} and Delta = {Delta}")

# plotting
fig, axes = plt.subplots(ncols=len(deltas), nrows=len(bs), figsize=(12, 6))
ticks = np.arange(0, N, int(N/5))
for i, b in enumerate(bs):
    for j, Delta in enumerate(deltas):

        # weight distribution
        ax = axes[i, j]
        idx1 = np.asarray(res["b"]) == b
        idx2 = np.asarray(res["delta"]) == Delta
        idx = np.argwhere((idx1 * idx2) > 0.5).squeeze()
        im = ax.imshow(np.asarray(res["w"][idx]), aspect="auto", interpolation="none", cmap="viridis",
                       vmax=1.0, vmin=0.0)
        ax.set_xlabel("source neuron eta")
        ax.set_ylabel("target neuron eta")
        ax.set_yticks(ticks, labels=np.round(res["eta"][idx][ticks], decimals=1))
        ax.set_xticks(ticks, labels=np.round(res["eta"][idx][ticks], decimals=1))
        ax.set_title(f"W (b = {b}, Delta = {np.round(Delta, decimals=1)})")

        # # firing rate distribution
        # ax = axes[1, i]
        # im = ax.imshow(np.asarray(res["data"][b]["fr"]), aspect="auto", interpolation="none", cmap="cividis", vmax=fr_max)
        # plt.colorbar(im, ax=ax)
        # ax.set_xlabel("eta")
        # ax.set_ylabel("Delta")
        # ax.set_title(f"Firing Rates (b = {b})")

fig.suptitle(f"Weight Distribution for {'Hebbian' if condition == 'hebbian' else 'Anti-Hebbian'} Learning (Simulation)")
plt.tight_layout()
fig.canvas.draw()
plt.savefig(f"../results/rnn_weights_2d_{condition}.svg")
plt.show()
