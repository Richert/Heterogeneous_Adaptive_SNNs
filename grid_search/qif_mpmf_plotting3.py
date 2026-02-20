import h5py
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import seaborn as sb
from sklearn.manifold import TSNE, Isomap

# load data file
path = "/home/rgast/data/mpmf_simulations"
f = h5py.File(f"{path}/qif_1pop_data.hdf5", "r")
gr = f["stdp_asym"]

# load parameter sweep
param_sweep = gr["param_sweep"]
params = param_sweep.attrs["parameters"]

# define different sweep conditions
conditions = list(gr.keys())
conditions.pop(conditions.index("param_sweep"))

# get results
results = {"synapse": [], "stp": [], "trial": [], "tau": [], "a": [], "stdp_ratio": [],
           "in_degree": [], "out_degree": [], "ff": [], "pr": [], "eta_in": [], "eta_out": []}
result_vars = gr.attrs["result_vars"].tolist()
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

            # get in-degree
            in_degrees = ds[trial, i, result_vars.index("in-degrees_post"), :]

            # get out-degree
            out_degrees = ds[trial, i, result_vars.index("out-degrees_post"), :]

            # get fano factors
            ff = ds[trial, i, result_vars.index("fano-factors_post"), :]

            # calculate participation ratio
            evs = ds[trial, i, result_vars.index("eigvals_post"), :]
            pr = np.sum(evs)**2/(np.sum(evs**2)*len(evs))

            # calculate correlation between degrees and etas
            c_eta_in = np.corrcoef(in_degrees, etas)[0][1]
            c_eta_out = np.corrcoef(out_degrees, etas)[0][1]

            # save results to dict
            results["synapse"].append(syn)
            results["stp"].append(stp)
            results["trial"].append(trial)
            results["tau"].append(tau)
            results["a"].append(a)
            results["stdp_ratio"].append(a_ratio*tau_ratio)
            results["in_degree"].append(in_degrees)
            results["out_degree"].append(out_degrees)
            results["ff"].append(np.mean(ff))
            results["pr"].append(pr)
            results["eta_in"].append(c_eta_in)
            results["eta_out"].append(c_eta_out)

# low-dim embedding analysis
embedding = "isomap"
for tau in np.unique(results["tau"]):
    for a in np.unique(results["a"]):
        idx1 = np.argwhere(results["tau"] == tau).squeeze()
        idx2 = np.argwhere(np.asarray(results["a"])[idx1] == a).squeeze()
        X = np.concatenate([results["in_degree"], results["out_degree"]], axis=1)
        if embedding == "isomap":
            model = Isomap(n_components=2, n_neighbors=100)
        else:
            model = TSNE(n_components=2, perplexity=50.0, early_exaggeration=15.0)
        X_t = model.fit_transform(X[idx1, :][idx2, :])
        for color_var in ["synapse", "stp", "stdp_ratio", "ff", "pr", "eta_in", "eta_out"]:
            res = {color_var: np.asarray(results[color_var])[idx1][idx2], "dim1": X_t[:, 0], "dim2": X_t[:, 1]}
            df = DataFrame.from_dict(res)
            fig, ax = plt.subplots(figsize=(10, 10))
            sb.scatterplot(df, x="dim1", y="dim2", hue=color_var, palette="vlag", ax=ax)
            ax.set_title(f"tau = {np.round(tau, decimals=0)}, a = {np.round(a, decimals=3)} (color-coding: {color_var})")
            plt.tight_layout()
        plt.show()
