import numpy as np
import matplotlib.pyplot as plt

def second_derivative(w, x, y, b):
    return -b*(x+y)  + (1-b)*(x-y) - 2*w*(1-b)*(x-y)

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

def get_w_solution(w0: np.ndarray, x: np.ndarray, y: np.ndarray, b: float) -> np.ndarray:
    w = np.zeros_like(w0) + w0
    if b < 1.0:
        mask = x != y
        a_term = 2*(b-1) * (x-y)
        b_term = x*(2*b-1) + y
        sqrt_term = np.sqrt((x-y)**2 + 4*x*y*b**2)
        w1 = (b_term + sqrt_term) / a_term
        w2 = (b_term - sqrt_term) / a_term
        for w_tmp in (w1, w2):
            sd = second_derivative(w_tmp, x, y, b)
            mask2 = (0 <= w_tmp) * (w_tmp <= 1) * (sd <= 0.0)
            idx = (mask * mask2) > 0.5
            w[idx] = w_tmp[idx]
    else:
        idx2 = np.argwhere(x + y > 0.0).squeeze()
        w[idx2] = x[idx2] / (x[idx2] + y[idx2])
    return w

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# parameter definition
condition = "hebbian"
distribution = "gaussian"
N = 200
Deltas = [0.25, 0.5, 1.0]
eta = 0.0
bs = [0.0, 0.001]
J = 4.0
n_reps = 10
res = {"b": [], "delta": [], "w": [], "eta": []}

for b in bs:
    for Delta in Deltas:

        # define source firing rate distribution
        etas = gaussian(N, eta, Delta)
        fr = get_qif_fr(etas)

        # get weight solutions
        w = np.random.uniform(0.01, 0.99, size=(N, N))
        for _ in range(n_reps):
            fr = get_qif_fr(etas + J * np.dot(w, fr) / N)
            x, y = get_xy(fr, fr, condition=condition)
            w[:, :] = get_w_solution(w, x, y, b)

        # save results
        res["w"].append(np.asarray(w))
        res["b"].append(b)
        res["delta"].append(Delta)
        res["eta"].append(etas)

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
plt.savefig(f"../results/ss_weight_solutions_recurrent_{condition}_{int(J)}.svg")
plt.show()
