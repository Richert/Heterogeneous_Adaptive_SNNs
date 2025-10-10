import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sb

# set parameters
path = "/home/richard-gast/PycharmProjects/Heterogeneous_Adaptive_SNNs/results"
plasticity = ["oja_rate", "oja_trace"]
conditions = ["hebbian", "antihebbian"]
noise_lvls = [0.0, 1.0, 4.0, 16.0]
bs = [0.0, 0.01, 0.1, 1.0]
res = {"plasticity": [], "condition": [], "b": [], "noise": [], "eta": [], "w": [], "w0": []}
J = 5.0

# load data
for p in plasticity:
    for condition in conditions:
        for b in bs:
            for noise in noise_lvls:
                data = pickle.load(open(f"{path}/qif_simulation_J{int(J)}_{p}_{condition}_{int(noise*1e3)}_{int(b*100)}.pkl", "rb"))
                for w, w0, eta in zip(data["w"], data["w0"], data["eta"]):
                    res["plasticity"].append(p)
                    res["condition"].append(condition)
                    res["b"].append(b)
                    res["noise"].append(noise)
                    res["eta"].append(eta)
                    res["w0"].append(w0)
                    res["w"].append(w)
res = pd.DataFrame.from_dict(res)

# plotting
res = pd.DataFrame.from_dict(res)
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

for p in plasticity:
    fig, axes = plt.subplots(ncols=2, nrows=len(noise_lvls), figsize=(6, 1.5*len(noise_lvls)))
    fig.suptitle(
        "Oja's rule with synaptic + trace variables" if p == "oja_rate" else "Oja's rule with trace variables only"
    )
    res_tmp = res.loc[res.loc[:, "plasticity"] == p, :]
    for j, c in enumerate(conditions):
        res_tmp2 = res_tmp.loc[res_tmp.loc[:, "condition"] == c, :]
        for i, noise in enumerate(noise_lvls):
            res_tmp3 = res_tmp2.loc[res_tmp2.loc[:, "noise"] == noise, :]
            ax = axes[i, j]
            sb.lineplot(res_tmp3, x="eta", y="w", hue="b", palette="Dark2", ax=ax, errorbar=("pi", 90), legend=False)
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$w$")
            # ax.get_legend().set_title(r"$b$")
            c_title = "Hebbian Learning" if c == "hebbian" else "Anti-Hebbian Learning"
            ax.set_title(f"{c}, noise lvl = {noise}")

    fig.canvas.draw()
    # plt.savefig(f"../results/figures/weight_update_rule.svg")
plt.show()