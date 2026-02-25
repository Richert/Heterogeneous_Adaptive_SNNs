import h5py
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["axes.labelsize"] = 12
import seaborn as sb
from sklearn.manifold import TSNE, Isomap

# load data file
path = "/home/rgast/data/mpmf_simulations"
f = h5py.File(f"{path}/qif_1pop_data.hdf5", "r")
stdp_conditions = ["stdp_asym", "stdp_sym", "antihebbian"]

# get results
results = {"plasticity rule": [], "synapse type": [], "stp": [], "trial": [], "tau": [], "a": [], "LTP/LTD": [],
           "w_in": [], "w_out": [], "FF": [], "dimensionality": [], "corr(fr,w_in)": [], "corr(fr,w_out)": [], "corr(w_in,w_out)": [],
           "mean(w)": [], "var(w)": [], "max. frequency": []}
categorical_result_vars = ["plasticity rule", "synapse type", "stp", "tau", "a"]
bidirectional_result_vars = ["corr(fr,w_in)", "corr(fr,w_out)", "corr(w_in,w_out)", "LTP/LTD", "input correlation", "FF"]
for stdp_condition in stdp_conditions:

    # get plasticity rule specific data
    gr = f[stdp_condition]
    result_vars = gr.attrs["result_vars"].tolist()
    param_sweep = gr["param_sweep"]
    params = param_sweep.attrs["parameters"]
    conditions = list(gr.keys())
    conditions.pop(conditions.index("param_sweep"))

    for c in conditions:

        # get condition-specific variables
        syn, stp = c.split("_")
        ds = gr[c]
        etas = ds[0, 0, result_vars.index("etas"), :]

        for i, (tau_p, tau_d, a_p, a_d) in enumerate(param_sweep):

            # calculate stdp parameters
            tau = tau_p
            tau_ratio = tau_p / tau_d
            a = np.sqrt(a_p * a_d)
            a_ratio = a_p / a_d

            for trial in range(ds.shape[0]):

                # calculate weight statistics
                in_degrees = ds[trial, i, result_vars.index("in-degrees_post"), :]
                out_degrees = ds[trial, i, result_vars.index("out-degrees_post"), :]
                weight_mean = (np.mean(in_degrees) + np.mean(out_degrees)) / 2
                weight_var = (np.var(in_degrees) + np.var(out_degrees)) / 2

                # get fano factors
                ff1 = ds[trial, i, result_vars.index("fano-factors_pre"), :]
                ff2 = ds[trial, i, result_vars.index("fano-factors_post"), :]
                ff_corr = np.corrcoef(ff2-ff1, etas)[0][1]

                # calculate participation ratio
                evs_1 = ds[trial, i, result_vars.index("eigvals_pre"), :]
                evs_2 = ds[trial, i, result_vars.index("eigvals_post"), :]
                pr1 = np.sum(evs_1)**2/(np.sum(evs_1**2)*len(evs_1))
                pr2 = np.sum(evs_2) ** 2 / (np.sum(evs_2 ** 2) * len(evs_2))

                # calculate correlation between degrees and etas
                c_eta_in = np.corrcoef(in_degrees, etas)[0][1]
                c_eta_out = np.corrcoef(out_degrees, etas)[0][1]
                c_in_out = np.corrcoef(in_degrees, out_degrees)[0][1]

                # extract remaining variables
                max_freq_1 = ds[trial, i, result_vars.index("max-freq-pre"), 0]
                max_freq_2 = ds[trial, i, result_vars.index("max-freq-post"), 0]

                # save results to dict
                results["plasticity rule"].append(stdp_condition)
                results["synapse type"].append(syn)
                results["stp"].append(stp)
                results["trial"].append(trial)
                results["tau"].append(np.round(tau, decimals=0))
                results["a"].append(np.round(a, decimals=3))
                results["LTP/LTD"].append(np.log(a_ratio*tau_ratio))
                results["w_in"].append(in_degrees)
                results["w_out"].append(out_degrees)
                results["FF"].append(ff_corr)
                results["dimensionality"].append(pr2)
                results["corr(fr,w_in)"].append(c_eta_in)
                results["corr(fr,w_out)"].append(c_eta_out)
                results["corr(w_in,w_out)"].append(c_in_out)
                results["mean(w)"].append(weight_mean)
                results["var(w)"].append(weight_var)
                results["max. frequency"].append(max_freq_2)

# low-dim embedding analysis
embedding = "tsne"
X = np.concatenate([results["w_in"], results["w_out"]], axis=1)
if embedding == "isomap":
    model = Isomap(n_components=2, n_neighbors=10)
else:
    model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=15.0)
X_t = model.fit_transform(X)

# plotting
genetic_vars = ["plasticity rule", "LTP/LTD", "tau"]
connectivity_vars = ["corr(fr,w_in)", "corr(fr,w_out)", "mean(w)"]
function_vars = ["FF", "dimensionality", "max. frequency"]
levels = ["plasticity parameters", "synaptic connectivity", "network dynamics"]
level_colors = [(210/255, 210/255, 210/255), (240/255, 240/255, 240/255), (210/255, 210/255, 210/255)]
for syn in ["exc"]:
    idx1 = np.argwhere(np.asarray(results["synapse type"])== syn).squeeze()
    idx2 = np.argwhere(np.asarray(results["synapse type"]) != syn).squeeze()
    fig = plt.figure(figsize=(16, 15), layout="constrained")
    subfigs = fig.subfigures(len(levels), 1, wspace=0.05, hspace=0.05)
    for row, (level, vars) in enumerate(zip(levels, [genetic_vars, connectivity_vars, function_vars])):
        axes = subfigs[row].subplots(ncols=len(vars))
        for col, color_var in enumerate(vars):
            res = {color_var: np.asarray(results[color_var]), "t-SNE dim 1": X_t[:, 0], "t-SNE dim 2": X_t[:, 1]}
            df = DataFrame.from_dict(res)
            ax = axes[col]
            sb.scatterplot(df.iloc[idx2, :], x="t-SNE dim 1", y="t-SNE dim 2", color="black", ax=ax, alpha=0.05, s=5,
                           legend=False)
            if color_var in categorical_result_vars:
                sb.scatterplot(df.iloc[idx1, :], x="t-SNE dim 1", y="t-SNE dim 2", hue=color_var, ax=ax, s=12, legend=True,
                               palette="tab10")
            else:
                norm = plt.Normalize(df.iloc[idx2, :].loc[:, color_var].min(), df.iloc[idx2, :].loc[:, color_var].max())
                cmap = "Spectral_r" if color_var in bidirectional_result_vars else "viridis"
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sb.scatterplot(df.iloc[idx1, :], x="t-SNE dim 1", y="t-SNE dim 2", hue=color_var, ax=ax, s=12, legend=False,
                               palette=cmap)
                ax.figure.colorbar(sm, ax=ax)
            if row < len(levels) - 1:
                ax.set_xlabel("")
            if col > 0:
                ax.set_ylabel("")
            ax.set_title(f"{color_var}")
        subfigs[row].suptitle(level)
        subfigs[row].set_facecolor(level_colors[row])
    fig.canvas.draw()
    fig.savefig(f"/home/rgast/data/qif_plasticity/{embedding}_{syn}_embeddings.png", dpi=300)
plt.show()
