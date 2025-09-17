import numpy as np
import matplotlib.pyplot as plt

def second_derivative(w, x, y, b):
    return -b*(x+y)  + (1-b)*(x-y) - 2*w*(1-b)*(x-y)

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = fr_target * fr_source
        y = fr_target**2
    elif condition == "antihebbian":
        x = fr_source**2
        y = fr_target*fr_source
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def get_w_solution(w0: np.ndarray, x: np.ndarray, y: np.ndarray, b: float) -> np.ndarray:
    w = np.zeros_like(w0) + w0
    idx = []
    if b < 1.0:
        idx = np.argwhere(x != y).squeeze()
        a_term = 2*(b-1) * (x-y)
        b_term = x*(2*b-1) + y
        sqrt_term = np.sqrt((x-y)**2 + 4*x*y*b**2)
        for i in idx:
            w1 = (b_term[i] + sqrt_term[i]) / a_term[i]
            w2 = (b_term[i] - sqrt_term[i]) / a_term[i]
            ws = []
            for w_tmp in (w1, w2):
                sd = second_derivative(w_tmp, x[i], y, b)
                if 0 <= w_tmp <= 1 and (sd <= 0.0 or np.abs(w0[i] - w_tmp) < 1e-6):
                    ws.append(w_tmp)
            w[i] = np.random.choice(ws)
    idx2 = np.argwhere(x + y > 0.0).squeeze()
    for i in idx2:
        if i not in idx:
            w[i] = x / (x + y)
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
Deltas_source = [0.25, 0.5, 1.0]
Delta_target = 1.0
eta_source, eta_target = 1.0, 0.0
bs = [0.0, 0.125]
J = 8.0
n_reps = 10
res = {"b": [], "delta": [], "w": [], "eta_source": [], "eta_target": []}

for b in bs:
    for Delta_source in Deltas_source:

        # define source firing rate distribution
        etas_source = gaussian(N, eta_source, Delta_source)
        etas_target = gaussian(N, eta_target, Delta_target)
        fr_source = get_qif_fr(etas_source)

        # get weight solutions
        w = np.random.uniform(0.01, 0.99, size=(N, N))
        for _ in range(n_reps):
            for i, eta in enumerate(etas_target):
                fr_target = get_qif_fr(eta + J * np.dot(w[i, :], fr_source) / N)
                x, y = get_xy(fr_source, fr_target, condition=condition)
                w[i, :] = get_w_solution(w[i, :], x, y, b)

        # save results
        res["w"].append(np.asarray(w))
        res["b"].append(b)
        res["delta"].append(Delta_source)
        res["eta_source"].append(etas_source)
        res["eta_target"].append(etas_target)

# plotting
fig, axes = plt.subplots(ncols=len(Deltas_source), nrows=len(bs), figsize=(12, 6), layout="constrained")
ticks = np.arange(0, N, int(N/5))
for i, b in enumerate(bs):
    for j, Delta in enumerate(Deltas_source):

        # weight distribution
        ax = axes[i, j]
        idx1 = np.asarray(res["b"]) == b
        idx2 = np.asarray(res["delta"]) == Delta
        idx = np.argwhere((idx1 * idx2) > 0.5).squeeze()
        im = ax.imshow(np.asarray(res["w"][idx]), aspect="auto", interpolation="none", cmap="viridis",
                       vmax=1.0, vmin=0.0)
        ax.set_xlabel("source neuron eta")
        ax.set_ylabel("target neuron eta")
        ax.set_yticks(ticks, labels=np.round(res["eta_target"][idx][ticks], decimals=1))
        ax.set_xticks(ticks, labels=np.round(res["eta_source"][idx][ticks], decimals=1))
        ax.set_title(f"W (b = {b}, Delta = {Delta})")
        if j == len(Deltas_source) - 1:
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
plt.savefig(f"../results/ss_weight_solutions_2d_{condition}_{int(J)}.svg")
plt.show()
