import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn as sb

# conditions
plasticity = ["hebbian", "antihebbian"]
projection = [5, -5]
reps = 10
rate_deltas = np.linspace(0.1, 1.5, num=100)
spike_deltas = np.linspace(0.1, 1.5, num=15)

# load data
path = "/home/richard-gast/PycharmProjects/Heterogeneous_Adaptive_SNNs/results"
rate_data = {"delta": [], "b": [], "plasticity": [], "projection": [], "correlation": [], "entropy": []}
spike_data = {"trial": [], "delta": [], "b": [], "plasticity": [], "projection": [], "correlation": [], "entropy": []}
rate_conn, spike_conn = {}, {}
for p in plasticity:
    for J in projection:

        conn = int(J)
        conn = f"{conn}_inh" if conn < 0 else f"{conn}"

        # load rate data
        rates = pickle.load(open(f"{path}/rate_weight_simulations_{p}_{conn}.pkl", "rb"))
        for b in rates["b"]:
            for i, delta in enumerate(rate_deltas):
                rate_data["delta"].append(delta)
                rate_data["b"].append(b)
                rate_data["plasticity"].append(p)
                rate_data["projection"].append(J)
                rate_data["correlation"].append(rates["C"][b][i])
                rate_data["entropy"].append(rates["H"][b][i])
            rate_conn[(b, p, J)] = rates["w"][b]

        # load SNN data
        for rep in range(reps):
            spikes = pickle.load(open(f"{path}/qif_weight_simulation_oja_{p}_{conn}_{rep}.pkl", "rb"))
            for b in rates["b"]:
                for i, delta in enumerate(spike_deltas):
                    spike_data["trial"].append(rep)
                    spike_data["delta"].append(delta)
                    spike_data["b"].append(b)
                    spike_data["plasticity"].append(p)
                    spike_data["projection"].append(J)
                    spike_data["correlation"].append(spikes["C"][b][i])
                    spike_data["entropy"].append(spikes["H"][b][i])
                spike_conn[(b, p, J, rep)] = spikes["w"][b]

rate_data = pd.DataFrame.from_dict(rate_data)
spike_data = pd.DataFrame.from_dict(spike_data)

# plotting
##########

# select data to plot
bs = [0.0, 0.05, 0.2]
p = "hebbian"
J = -5
rep = 2

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 2

# plotting
fig, axes = plt.subplots(nrows=4, ncols=len(bs), layout="constrained")
ticks = np.arange(0, len(rate_deltas), int(len(rate_deltas) / 4))
ticks2 = np.arange(0, len(spike_deltas), int(len(spike_deltas) / 4))
for i, b in enumerate(bs):

    # rate weight distribution
    ax = axes[0, i]
    im = ax.imshow(np.asarray(rate_conn[(b, p, J)]).T, aspect="auto", interpolation="none", cmap="viridis",
                   vmax=1.0, vmin=0.0)
    ax.set_ylabel("neuron")
    ax.set_xticks(ticks, labels=np.round(rate_deltas[ticks], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(fr"$w$ ($b = {b}$)")

    # spike weight distribution
    ax = axes[1, i]
    im = ax.imshow(np.asarray(spike_conn[(b, p, J, rep)]).T, aspect="auto", interpolation="none", cmap="viridis",
                   vmax=1.0, vmin=0.0)
    ax.set_ylabel("neuron")
    ax.set_xlabel(r"$\Delta$")
    ax.set_xticks(ticks2, labels=np.round(spike_deltas[ticks2], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)

    # correlation
    ax = axes[2, i]
    sb.pointplot(spike_data.loc[(spike_data.loc[:, "b"] == b) & (spike_data.loc[:, "projection"] == 5), :],
                 x="delta", y="correlation", hue="plasticity", capsize=0.3, native_scale=True, linestyle="none",
                 errorbar=None, marker="*", markersize=markersize, markeredgewidth=3, ax=ax, legend=False)
    sb.pointplot(spike_data.loc[(spike_data.loc[:, "b"] == b) & (spike_data.loc[:, "projection"] == -5), :],
                 x="delta", y="correlation", hue="plasticity", capsize=0.3, native_scale=True, linestyle="none",
                 errorbar=None, marker="*", markersize=markersize, markeredgewidth=3, ax=ax, legend=False)
    sb.lineplot(rate_data.loc[rate_data.loc[:, "b"] == b, :],
                x="delta", y="correlation", hue="plasticity", style="projection",
                ax=ax, legend="auto" if i == len(bs) else False)
    ax.set_xticks(rate_deltas[ticks], labels=np.round(rate_deltas[ticks], decimals=1))
    ax.set_ylabel(r"$R$")
    ax.set_xlabel("")
    ax.set_title(r"$R = corr(w_i, \eta_i)$")

    # entropy
    ax = axes[3, i]
    sb.pointplot(spike_data.loc[(spike_data.loc[:, "b"] == b) & (spike_data.loc[:, "projection"] == 5), :],
                 x="delta", y="entropy", hue="plasticity", capsize=0.3, native_scale=True, linestyle="none",
                 errorbar=None, marker="*", markersize=markersize, markeredgewidth=3, ax=ax, legend=False)
    sb.pointplot(spike_data.loc[(spike_data.loc[:, "b"] == b) & (spike_data.loc[:, "projection"] == -5), :],
                 x="delta", y="entropy", hue="plasticity", capsize=0.3, native_scale=True, linestyle="none",
                 errorbar=None, marker="*", markersize=markersize, markeredgewidth=3, ax=ax, legend=False)
    sb.lineplot(rate_data.loc[rate_data.loc[:, "b"] == b, :],
                x="delta", y="entropy", hue="plasticity", style="projection",
                ax=ax, legend="auto" if i == len(bs) else False)
    ax.set_xticks(rate_deltas[ticks], labels=np.round(rate_deltas[ticks], decimals=1))
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$H$")
    ax.set_title(r"$H = entropy(w_i)$")

fig.canvas.draw()
plt.savefig(f"../results/figures/weight_distribution_sender_receiver.svg")
plt.show()