import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sb
import numpy as np

# set parameters
path = "/home/richard-gast/PycharmProjects/Heterogeneous_Adaptive_SNNs/results/qif_simulations_J5"
plasticity = "oja_trace"
conditions = ["hebbian"]
noise_lvls = [0.0, 1.0]
rate_noises = [0.0, 0.006]
bs = [0.0, 0.01, 0.1, 1.0]
res = {"condition": [], "b": [], "noise": [], "eta": [], "w": [], "type": [], "w0": []}
J = 5.0

# load data
theory_data = pickle.load(open(f"{path}/../weight_solutions.pkl", "rb"))
theory_res = pd.DataFrame.from_dict(theory_data)
rate_data = pickle.load(open(f"{path}/../rate_simulation_J{int(J)}.pkl", "rb"))
rate_w0s = np.asarray(rate_data["w0"])
rate_ws = np.asarray(rate_data["w"])
rate_etas = np.asarray(rate_data["eta"])
rate_noise_lvls = np.asarray(rate_data["noise"])
rate_bs = np.asarray(rate_data["b"])
rate_conditions = np.asarray(rate_data["condition"])
for condition in conditions:
    idx1 = rate_conditions == condition
    for b in bs:
        idx2 = rate_bs == b
        for noise, rate_noise in zip(noise_lvls, rate_noises):
            idx3 = rate_noise_lvls == rate_noise
            data = pickle.load(open(f"{path}/qif_simulation_J{int(J)}_{plasticity}_{condition}_{int(noise*1e3)}_{int(b*100)}.pkl", "rb"))
            for w, w0, eta in zip(data["w"], data["w0"], data["eta"]):
                idx4 = rate_etas == eta
                idx5 = rate_w0s == w0
                w_rate = rate_ws[(idx1*idx2*idx3*idx4*idx5) > 0.0][0]
                for model_type, w_tmp in zip(["qif", "rate"], [w, w_rate]):
                    res["condition"].append(condition)
                    res["b"].append(b)
                    res["noise"].append(noise)
                    res["eta"].append(eta)
                    res["w"].append(w_tmp)
                    res["type"].append(model_type)
                    res["w0"].append(w0)
res = pd.DataFrame.from_dict(res)

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

fig = plt.figure(figsize=(6, 1.6*len(conditions)))
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0., wspace=0.)
grid = fig.add_gridspec(ncols=3, nrows=len(conditions))
for i, c in enumerate(conditions):

    res_tmp = res.loc[res.loc[:, "condition"] == c, :]
    tres = theory_res.loc[theory_res.loc[:, "condition"] == c, :]

    # solutions
    ax = fig.add_subplot(grid[i, 0])
    sb.lineplot(tres, x="eta", y="w", hue="b", palette="Dark2", ax=ax, errorbar=("pi", 90),
                legend=False if i > 0 else "auto")
    ax.set_ylabel(r"$w$")
    if i == 0:
        ax.get_legend().set_title(r"$b$")
        ax.set_title(r"Theory")
        ax.set_xlabel("")
    else:
        ax.set_xlabel(r"$\eta$")

    for j, noise in enumerate(noise_lvls):
        res_tmp2 = res_tmp.loc[res_tmp.loc[:, "noise"] == noise, :]

        ax = fig.add_subplot(grid[i, j+1])
        sb.lineplot(res_tmp2, x="eta", y="w", hue="b", style="type", palette="Dark2", ax=ax, errorbar=("pi", 90),
                    legend=False, err_kws={"alpha": 0.0})
        if i == 1:
            ax.set_xlabel(r"$\eta$")
        else:
            ax.set_xlabel("")
        ax.set_ylabel("")
        n_title = "no noise" if noise == 0.0 else "with noise"
        ax.set_title(f"Simulations, {n_title}")

fig.canvas.draw()
plt.savefig(f"../results/figures/sender_receiver_weight_solutions.svg")
plt.show()