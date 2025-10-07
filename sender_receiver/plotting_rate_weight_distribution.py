import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

# load data
condition = "hebbian"
J = 5
data = pickle.load(open(f"../results/rate_weight_simulations_{condition}_{int(J)}.pkl"))

# create data frames
weights = {"b": [], "neuron": [], "delta": [], "noise": [], "w": []}
for b, delta, noise, ws in zip(data["b"], data["delta"], data["noise"], data["w"]):
    ...

# plotting
fig, axes = plt.subplots(nrows=3, ncols=len(bs), figsize=(3 * len(bs), 5), layout="constrained")
ticks = np.arange(0, m, int(m / 5))
for i, b in enumerate(bs):

    # weight distribution
    ax = axes[0, i]
    sb.heatmap()
    # im = ax.imshow(np.asarray(res["w"][b]).T, aspect="auto", interpolation="none", cmap="viridis", vmax=1.0, vmin=0.0)
    ax.set_ylabel("neuron")
    ax.set_xlabel("Delta")
    ax.set_xticks(ticks, labels=np.round(deltas[ticks], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"w (b = {b})")

    # correlation
    ax = axes[1, i]
    ax.plot(deltas, res["C"][b])
    ax.set_xlabel("Delta")
    ax.set_ylabel("C")
    ax.set_title("correlation(w, eta)")

    # entropy
    ax = axes[2, i]
    ax.plot(deltas, res["H"][b])
    ax.set_xlabel("Delta")
    ax.set_ylabel("H")
    ax.set_title("entropy(w)")

    # # variance
    # ax = axes[3, i]
    # ax.plot(deltas, res["V"][b])
    # ax.set_xlabel("Delta")
    # ax.set_ylabel("var")
    # ax.set_title("variance(w)")

fig.suptitle(f"{'Anti-Hebbian' if 'antihebbian' in condition else 'Hebbian'} Learning (J = {int(J)}, Rate Simulation)")
fig.canvas.draw()
plt.savefig(f"../results/ss_weight_simulation_{condition}_{conn}.svg")
plt.show()
