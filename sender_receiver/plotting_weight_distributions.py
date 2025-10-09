import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

# load data
neuron_type = "rate"
condition = "hebbian"
J = -5
rep = 0
tau = 4.0
if "rate" in neuron_type:
    data = pickle.load(open(f"../results/{neuron_type}_simulation_{condition}_{int(J)}.pkl", "rb"))["results"]
else:
    data = pickle.load(open(f"../results/{neuron_type}_simulation_{condition}_{int(tau)}_{int(J)}_{rep}.pkl", "rb"))["results"]

# create data frames
weights = {"b": [], "neuron": [], "delta": [], "noise": [], "w": []}
for b, delta, noise, ws in zip(data["b"], data["delta"], data["noise"], data["w"]):
    for neuron, w in enumerate(ws):
        weights["b"].append(b)
        weights["neuron"].append(neuron)
        weights["delta"].append(np.round(delta, decimals=2))
        weights["noise"].append(noise)
        weights["w"].append(w)
weights = pd.DataFrame.from_dict(weights)

# plotting weight distributions
bs, noise_lvls = np.unique(data["b"]), np.unique(data["noise"])
fig, axes = plt.subplots(ncols=len(bs), nrows=len(noise_lvls), figsize=(3*len(bs), 2*len(noise_lvls)), layout="constrained")
for j, b in enumerate(bs):
    for i, noise in enumerate(noise_lvls):

        w = weights.loc[(weights.loc[:, "b"] == b) & (weights.loc[:, "noise"] == noise)]
        w = w.pivot(index="neuron", columns="delta", values="w")

        # weight distribution
        ax = axes[i, j]
        sb.heatmap(w, vmin=0.0, vmax=1.0, ax=ax)
        ax.set_title(f"w (b = {b})")

# plotting statistics
data.pop("w")
data = pd.DataFrame.from_dict(data)
fig, axes = plt.subplots(nrows=3, ncols=len(bs), figsize=(3 * len(bs), 6), layout="constrained")
for j, b in enumerate(bs):

    df = data.loc[data.loc[:, "b"] == b]

    # correlation
    ax = axes[0, j]
    sb.lineplot(df, x="delta", y="C", hue="noise", ax=ax, palette="dark")
    ax.set_title("correlation(w, eta)")

    # entropy
    ax = axes[1, j]
    sb.lineplot(df, x="delta", y="H", hue="noise", ax=ax, palette="dark")
    ax.set_title("entropy(w)")

    # variance
    ax = axes[2, j]
    sb.lineplot(df, x="delta", y="V", hue="noise", ax=ax, palette="dark")
    ax.set_title("variance(w)")

fig.suptitle(f"{'Anti-Hebbian' if 'antihebbian' in condition else 'Hebbian'} Learning (J = {int(J)}, Rate Simulation)")
fig.canvas.draw()
plt.show()
