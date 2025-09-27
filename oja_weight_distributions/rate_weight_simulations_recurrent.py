import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import entropy

def get_prob(x):
    unique, count = np.unique(x, return_counts=True, axis=0)
    return count / len(x)

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

def delta_w(w: np.ndarray, r: np.ndarray, eta: np.ndarray, J: float, b: float, a: float, condition: str) -> np.ndarray:
    w = w.reshape(N, N)
    r[:] = get_qif_fr(eta + J*np.dot(w, r) / N)
    x, y = get_xy(r, r, condition=condition)
    return (a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))).flatten()

def get_w_solution(w0: np.ndarray, fr: np.ndarray, eta: np.ndarray, J: float, b: float, a: float, T: float, **kwargs
                   ) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(w, fr, eta, J, b, a, condition), t_span=(0.0, T), y0=np.asarray(w0), **kwargs)
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
N = 500
m = 10
Deltas = np.linspace(0.1, 1.0, num=m)
eta = 0.5
a = 0.1
J = 5.0
bs = [0.0, 0.01, 0.1]
res = {"b": [], "w": [], "delta": [], "eta": [], "H": {}, "C": {}, "V": {}}

# simulation parameters
T = 1000.0
solver_kwargs = {"t_eval": [0.0, T]}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    h_col, c_col, v_col = [], [], []
    for Delta in Deltas:

        # define initial condition
        etas = f(N, eta, Delta)
        fr = get_qif_fr(etas)
        w0 = np.random.uniform(0.01, 0.99, size=(int(N*N),))

        # get weight solutions
        w = get_w_solution(w0, fr, etas, J, b, a, T, **solver_kwargs)

        hs, cs, vs = [], [], []
        for i in range(N):
            hs.append(entropy(get_prob(w[i, :])))
            vs.append(np.var(w[i, :]))
            cs.append(np.corrcoef(etas, w[i, :])[0, 1])
        h_col.append(hs)
        c_col.append(cs)
        v_col.append(vs)

        # save results
        res["b"].append(b)
        res["delta"].append(Delta)
        res["w"].append(w)
        res["eta"].append(etas)
        print(f"Finished simulations for b = {b}, Delta = {Delta}")

    res["H"][b] = np.asarray(h_col)
    res["V"][b] = np.asarray(v_col)
    res["C"][b] = np.asarray(c_col)

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
        ax.set_title(f"W (b = {b}, Delta = {np.round(Delta, decimals=1)})")
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

# plotting
fig, axes = plt.subplots(nrows=3, ncols=len(bs), figsize=(3*len(bs), 6), layout="constrained")
ticks = np.arange(0, m, int(m/5))
for i, b in enumerate(bs):

    # weight entropy distribution
    ax = axes[0, i]
    im = ax.imshow(np.asarray(res["H"][b]).T, aspect="auto", interpolation="none", cmap="viridis")
    ax.set_ylabel("neuron")
    ax.set_xlabel("Delta")
    ax.set_xticks(ticks, labels=np.round(Deltas[ticks], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"H(w) for b = {b}")

    # weight variance distribution
    ax = axes[1, i]
    im = ax.imshow(np.asarray(res["V"][b]).T, aspect="auto", interpolation="none", cmap="viridis")
    ax.set_ylabel("neuron")
    ax.set_xlabel("Delta")
    ax.set_xticks(ticks, labels=np.round(Deltas[ticks], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"var(w) for b = {b}")

    # weight-eta correlation distribution
    ax = axes[2, i]
    im = ax.imshow(np.asarray(res["C"][b]).T, aspect="auto", interpolation="none", cmap="viridis")
    ax.set_ylabel("neuron")
    ax.set_xlabel("Delta")
    ax.set_xticks(ticks, labels=np.round(Deltas[ticks], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"corr(w, eta) for b = {b}")

conn = int(J)
conn = f"{conn}_inh" if conn < 0 else f"{conn}"
plt.savefig(f"../results/rate_weight_simulations_recurrent_{condition}_{conn}.svg")
plt.show()
