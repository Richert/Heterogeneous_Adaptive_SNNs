import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from scipy.stats import entropy

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

def get_prob(x, bins: int = 100):
    bins = np.linspace(0.0, 1.0, num=bins)
    counts, _ = np.histogram(x, bins=bins)
    count_sum = np.sum(counts)
    return counts / count_sum if count_sum > 0 else counts

# parameters
path = "../results/rnn_results"
results = {"b": [], "Delta": [], "noise": [], "c_s": [], "c_t": [], "v_s": [], "v_t": [], "h_s": [], "h_t": []}
bs = [0.0, 0.1, 0.2]
noises = [0.0, 1.0, 10.0]
deltas = np.arange(0.0, 2.1, step=0.2)
n_reps = 10

# calculate weight statistics
for b in bs:
    for noise in noises:
        for delta in deltas:
            for rep in range(n_reps):

                try:

                    data = pickle.load(open(f"{path}/fre_mp_{int(b*10)}_{int(noise)}_{int(delta*10.0)}_{rep}.pkl",
                                            "rb"))
                    w = data["W"]
                    etas = data["eta"]
                    w_s = np.mean(w, axis=1)
                    w_t = np.mean(w, axis=0)

                    results["b"].append(b)
                    results["Delta"].append(delta)
                    results["noise"].append(noise)
                    results["c_s"].append(correlate(etas, w_s))
                    results["c_t"].append(correlate(etas, w_t))
                    results["v_s"].append(np.var(w_s))
                    results["v_t"].append(np.var(w_t))
                    results["h_s"].append(entropy(get_prob(w_s)))
                    results["h_t"].append(entropy(get_prob(w_t)))

                except FileNotFoundError:
                    pass

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

fig = plt.figure(figsize=(12, 9))
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0., wspace=0.)
grid = fig.add_gridspec(nrows=4, ncols=3)
for i, b in enumerate(bs):

    # source correlation
    ax = fig.add_subplot(grid[0, i])
    sb.lineplot(results, x="Delta", y="c_s", hue="noise", ax=ax, palette="Dark2")
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel(r"$R(\eta_j, w_j)$")
    else:
        ax.set_ylabel("")
    ax.set_title(rf"$b = {b}$")

    # target correlation
    ax = fig.add_subplot(grid[1, i])
    sb.lineplot(results, x="Delta", y="c_t", hue="noise", legend=False, ax=ax, palette="Dark2")
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel(r"$R(\eta_i, w_i)$")
    else:
        ax.set_ylabel("")

    # source weight variance
    ax = fig.add_subplot(grid[2, i])
    sb.lineplot(results, x="Delta", y="v_s", hue="noise", legend=False, ax=ax, palette="Dark2")
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel(r"$var(w_j)$")
    else:
        ax.set_ylabel("")

    # target weight variance
    ax = fig.add_subplot(grid[3, i])
    sb.lineplot(results, x="Delta", y="v_t", hue="noise", legend=False, ax=ax, palette="Dark2")
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel(r"$var(w_i)$")
    else:
        ax.set_ylabel("")

    # # source weight entropy
    # ax = fig.add_subplot(grid[2, i])
    # sb.lineplot(results, x="Delta", y="h_s", hue="noise", legend=False, ax=ax, palette="Dark2")
    # ax.set_xlabel("")
    # if i == 0:
    #     ax.set_ylabel(r"$H(w_j)$")
    # else:
    #     ax.set_ylabel("")
    #
    # # target weight variance
    # ax = fig.add_subplot(grid[3, i])
    # sb.lineplot(results, x="Delta", y="h_t", hue="noise", legend=False, ax=ax, palette="Dark2")
    # ax.set_xlabel(r"$\Delta$")
    # if i == 0:
    #     ax.set_ylabel(r"$H(w_i)$")
    # else:
    #     ax.set_ylabel("")

fig.suptitle("Weight Statistics")
fig.canvas.draw()
plt.show()
