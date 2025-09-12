import numpy as np
import matplotlib.pyplot as plt

def second_derivative(w, x, y, b):
    return -b*(x+y)  + (1-b)*(x-y) - 2*w*(1-b)*(x-y)

def get_xy(fr_source: float, fr_target: float, condition: str) -> tuple:
    if condition == "hebbian":
        x = fr_target * fr_source
        y = fr_target**2
    elif condition == "antihebbian":
        x = fr_source**2
        y = fr_target*fr_source
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def get_w_solution(w0: float, x: float, y: float, b: float) -> float:
    if b < 1.0 and x != y:
        a_term = 2*(b-1) * (x-y)
        b_term = x*(2*b-1) + y
        sqrt_term = np.sqrt((x-y)**2 + 4*x*y*b**2)
        w1 = (b_term + sqrt_term) / a_term
        w2 = (b_term - sqrt_term) / a_term
        ws = []
        for w in (w1, w2):
            sd = second_derivative(w, x, y, b)
            if 0 <= w <= 1 and (sd <= 0.0 or np.abs(w0 - w) < 1e-6):
                ws.append(w)
        return np.random.choice(ws)
    elif x + y > 0.0:
        return x / (x + y)
    else:
        return w0

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
Deltas_source = [0.5, 1.0, 2.0]
Delta_target = 1.0
eta_source, eta_target = 1.0, 1.0
bs = [0.0, 0.5]
res = {"b": [], "delta": [], "w": [], "eta_source": [], "eta_target": []}

for b in bs:
    for Delta_source in Deltas_source:

        # define source firing rate distribution
        etas_source = gaussian(N, eta_source, Delta_source)
        etas_target = gaussian(N, eta_target, Delta_target)
        fr_source = get_qif_fr(etas_source)
        fr_target = get_qif_fr(etas_target)

        # get weight solutions
        ws = []
        for target_fr in fr_target:
            row = []
            for source_fr in fr_source:
                x, y = get_xy(source_fr, target_fr, condition=condition)
                w0 = np.random.choice([0.0, 0.33, 0.66, 1.0])
                w = get_w_solution(w0, x, y, b)
                row.append(w)
            ws.append(row)

        # save results
        res["w"].append(np.asarray(ws))
        res["b"].append(b)
        res["delta"].append(Delta_source)
        res["eta_source"].append(etas_source)
        res["eta_target"].append(etas_target)

# plotting
fig, axes = plt.subplots(ncols=len(Deltas_source), nrows=len(bs), figsize=(12, 6))
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

        # # firing rate distribution
        # ax = axes[1, i]
        # im = ax.imshow(np.asarray(res["data"][b]["fr"]), aspect="auto", interpolation="none", cmap="cividis", vmax=fr_max)
        # plt.colorbar(im, ax=ax)
        # ax.set_xlabel("eta")
        # ax.set_ylabel("Delta")
        # ax.set_title(f"Firing Rates (b = {b})")

fig.suptitle(f"Weight Distribution for {'Hebbian' if condition == 'hebbian' else 'Anti-Hebbian'} Learning (Theory)")
plt.tight_layout()
fig.canvas.draw()
plt.savefig(f"../results/ss_weight_distribution_2d_{condition}.svg")
plt.show()
