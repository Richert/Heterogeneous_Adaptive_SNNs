import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn as sb

# conditions
plasticity = ["hebbian", "antihebbian"]
projection = [5, -5]
taus = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
target_tau = 8.0
reps = 10
rate_deltas = np.linspace(0.1, 1.5, num=100)
spike_deltas = np.linspace(0.1, 1.5, num=10)

# load data
path = "/results"
rate_data = {"delta": [], "b": [], "plasticity": [], "projection": [], "correlation": [], "entropy": []}
spike_data = {"trial": [], "delta": [], "b": [], "tau": [], "plasticity": [], "projection": [], "correlation": [], "entropy": []}
rate_conn, spike_conn = {}, {}
for p in plasticity:
    for J in projection:

        conn = int(J)

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
        for tau in taus:
            for rep in range(reps):
                spikes = pickle.load(open(f"{path}/qif_oja_simulation_{p}_{int(tau)}_{conn}_{rep}.pkl", "rb"))
                for b in rates["b"]:
                    for i, delta in enumerate(spike_deltas):
                        spike_data["trial"].append(rep)
                        spike_data["delta"].append(delta)
                        spike_data["b"].append(b)
                        spike_data["plasticity"].append(p)
                        spike_data["projection"].append(J)
                        spike_data["tau"].append(spikes["tau"])
                        spike_data["correlation"].append(spikes["C"][b][i])
                        spike_data["entropy"].append(spikes["H"][b][i])
                    spike_conn[(b, p, J, rep)] = spikes["w"][b]

rate_data = pd.DataFrame.from_dict(rate_data)
spike_data = pd.DataFrame.from_dict(spike_data)
spike_data_r = spike_data.loc[spike_data.loc[:, "tau"] == target_tau, :]

# calculate errors between rate and QIF models
errors = {"tau": [], "error": [], "type": []}
for tau in np.unique(spike_data.loc[:, "tau"]):
    spike_data_tmp = spike_data.loc[spike_data.loc[:, "tau"] == tau, :]
    h_error, c_error = [], []
    for delta in spike_deltas:
        rate_delta = rate_deltas[np.argmin(np.abs(rate_deltas - delta))]
        rate_data_tmp = rate_data.loc[rate_data.loc[:, "delta"] == rate_delta]
        spike_data_tmp2 = spike_data_tmp.loc[spike_data_tmp.loc[:, "delta"] == delta, :]
        for trial in np.unique(spike_data_tmp2.loc[:, "trial"]):
            spike_data_tmp3 = spike_data_tmp2.loc[spike_data_tmp2.loc[:, "trial"] == trial, :]
            hdiff = np.mean(((rate_data_tmp.loc[:, "entropy"].values - spike_data_tmp3.loc[:, "entropy"].values) / np.max(rate_data.loc[:, "entropy"]))**2)
            cdiff = np.mean((rate_data_tmp.loc[:, "correlation"].values - spike_data_tmp3.loc[:, "correlation"].values)**2)
            h_error.append(hdiff)
            c_error.append(cdiff)
    for e, type in zip([h_error, c_error], ["entropy", "correlation"]):
        errors["tau"].append(int(tau))
        errors["error"].append(np.mean(e))
        errors["type"].append(type)
errors = pd.DataFrame.from_dict(errors)

# plotting
##########

# select data to plot
bs = [0.0, 0.05, 0.2]
p = "antihebbian"
J = -5
rep = 2

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 7)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 2

# plotting
fig, axes = plt.subplots(nrows=5, ncols=len(bs), layout="constrained")
ticks = np.arange(0, len(rate_deltas), int(len(rate_deltas) / 4))
ticks2 = np.arange(0, len(spike_deltas), int(len(spike_deltas) / 4))

# error between rate and QIF model
ax = axes[0, 2]
sb.barplot(errors, x="tau", y="error", hue="type", ax=ax)
ax.set_xlabel(r"$\tau_s$")
ax.set_ylabel("MSE")
ax.set_title("Rate-QIF difference")

# other plots as a function of b
for i, b in enumerate(bs):

    # rate weight distribution
    ax = axes[1, i]
    im = ax.imshow(np.asarray(rate_conn[(b, p, J)]).T, aspect="auto", interpolation="none", cmap="viridis",
                   vmax=1.0, vmin=0.0)
    ax.set_xticks(ticks, labels=np.round(rate_deltas[ticks], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    elif i == 0:
        ax.set_ylabel("neuron")
    ax.set_title(fr"$w$ ($b = {b}$)")

    # spike weight distribution
    ax = axes[2, i]
    im = ax.imshow(np.asarray(spike_conn[(b, p, J, rep)]).T, aspect="auto", interpolation="none", cmap="viridis",
                   vmax=1.0, vmin=0.0)
    ax.set_xlabel(r"$\Delta$")
    ax.set_xticks(ticks2, labels=np.round(spike_deltas[ticks2], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    elif i == 0:
        ax.set_ylabel("neuron")

    # correlation
    ax = axes[3, i]
    sb.pointplot(spike_data_r.loc[(spike_data_r.loc[:, "b"] == b) & (spike_data_r.loc[:, "projection"] == 5), :],
                 x="delta", y="correlation", hue="plasticity", capsize=0.3, native_scale=True, linestyle="none",
                 errorbar=None, marker="*", markersize=markersize, markeredgewidth=3, ax=ax, legend=False)
    sb.pointplot(spike_data_r.loc[(spike_data_r.loc[:, "b"] == b) & (spike_data_r.loc[:, "projection"] == -5), :],
                 x="delta", y="correlation", hue="plasticity", capsize=0.3, native_scale=True, linestyle="none",
                 errorbar=None, marker="*", markersize=markersize, markeredgewidth=3, ax=ax, legend=False)
    sb.lineplot(rate_data.loc[rate_data.loc[:, "b"] == b, :],
                x="delta", y="correlation", hue="plasticity", style="projection",
                ax=ax, legend="auto" if i == len(bs) else False)
    ax.set_xticks(rate_deltas[ticks], labels=np.round(rate_deltas[ticks], decimals=1))
    if i == 0:
        ax.set_ylabel(r"$R$")
    else:
        ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(r"$R = corr(w_i, \eta_i)$")

    # entropy
    ax = axes[4, i]
    sb.pointplot(spike_data_r.loc[(spike_data_r.loc[:, "b"] == b) & (spike_data_r.loc[:, "projection"] == 5), :],
                 x="delta", y="entropy", hue="plasticity", capsize=0.3, native_scale=True, linestyle="none",
                 errorbar=None, marker="*", markersize=markersize, markeredgewidth=3, ax=ax, legend=False)
    sb.pointplot(spike_data_r.loc[(spike_data_r.loc[:, "b"] == b) & (spike_data_r.loc[:, "projection"] == -5), :],
                 x="delta", y="entropy", hue="plasticity", capsize=0.3, native_scale=True, linestyle="none",
                 errorbar=None, marker="*", markersize=markersize, markeredgewidth=3, ax=ax, legend=False)
    sb.lineplot(rate_data.loc[rate_data.loc[:, "b"] == b, :],
                x="delta", y="entropy", hue="plasticity", style="projection",
                ax=ax, legend="auto" if i == len(bs) else False)
    ax.set_xticks(rate_deltas[ticks], labels=np.round(rate_deltas[ticks], decimals=1))
    ax.set_xlabel(r"$\Delta$")
    if i == 0:
        ax.set_ylabel(r"$H$")
    else:
        ax.set_ylabel("")
    ax.set_title(r"$H = entropy(w_i)$")

fig.canvas.draw()
plt.savefig(f"../results/figures/weight_distribution_sender_receiver.svg")
plt.show()