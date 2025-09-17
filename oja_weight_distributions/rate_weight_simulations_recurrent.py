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

def delta_w(w: np.ndarray, r0: np.ndarray, eta: np.ndarray, J: float, b: float, a: float, condition: str) -> np.ndarray:
    w = w.reshape(N, N)
    r1 = get_qif_fr(eta + J*np.dot(w, r0) / N)
    x, y = get_xy(r1, r1, condition=condition)
    return (a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))).flatten()

def get_w_solution(w0: np.ndarray, fr: np.ndarray, eta: np.ndarray, J: float, b: float, a: float, T: float, **kwargs
                   ) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(w, fr, eta, J, b, a, condition), t_span=(0.0, T), y0=np.asarray(w0),
                     **kwargs)
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
condition = "antihebbian"
distribution = "gaussian"
N = 200
m = 100
Deltas = [0.25, 0.5, 1.0]
eta = 0.0
a = 0.2
J = 10.0
bs = [0.0, 0.125]
res = {"b": [], "w": [], "delta": [], "eta": []}

# simulation parameters
T = 1000.0
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    for Delta in Deltas:

        # define initial condition
        etas = f(N, eta, Delta)
        fr = get_qif_fr(etas)
        w0 = np.random.uniform(0.01, 0.99, size=(int(N*N),))

        # get weight solutions
        w = get_w_solution(w0, fr, etas, J, b, a, T, **solver_kwargs)

        # save results
        res["b"].append(b)
        res["delta"].append(Delta)
        res["w"].append(w)
        res["eta"].append(etas)
        print(f"Finished simulations for b = {b}, Delta = {Delta}")

# plotting
fig, axes = plt.subplots(ncols=len(Deltas), nrows=len(bs), figsize=(12, 6), layout="constrained")
ticks = np.arange(0, N, int(N/5))
for i, b in enumerate(bs):
    for j, Delta in enumerate(Deltas):

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
        ax.set_title(f"W (b = {b}, Delta = {Delta})")
        if j == len(Deltas) - 1:
            plt.colorbar(im, ax=ax, shrink=0.8)

        # # firing rate distribution
        # ax = axes[1, i]
        # im = ax.imshow(np.asarray(res["data"][b]["fr"]), aspect="auto", interpolation="none", cmap="cividis", vmax=fr_max)
        # plt.colorbar(im, ax=ax)
        # ax.set_xlabel("eta")
        # ax.set_ylabel("Delta")
        # ax.set_title(f"Firing Rates (b = {b})")

fig.suptitle(f"{'Hebbian' if condition == 'hebbian' else 'Anti-Hebbian'} Learning (J = {int(J)}, Theory)")
fig.canvas.draw()
plt.savefig(f"../results/rate_weight_simulations_recurrent_{condition}_{int(J)}.svg")
plt.show()
