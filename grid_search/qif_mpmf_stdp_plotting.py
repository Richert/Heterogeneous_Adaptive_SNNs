import h5py
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import seaborn as sb

# load data file
path = "/home/rgast/data/mpmf_simulations"
f = h5py.File(f"{path}/mpmf_1pop_data.hdf5", "r")
gr = f["stdp"]

# load parameter sweep
param_sweep = gr["param_sweep"]
params = param_sweep.attrs["parameters"]

# define different sweep conditions
conditions = list(gr.keys())
conditions.pop(conditions.index("param_sweep"))

# get results
results = {key: {"tau_ratio": [], "a_ratio": [], "in_degree": [], "in_degree_diff": [], "out_degree": [], "out_degree_diff": [],
                 "ff": [], "ff_diff": [], "pr": [], "pr_diff": []} for key in conditions}
result_vars = gr.attrs["result_vars"].tolist()
for c in conditions:

    ds = gr[c]

    for i, (tau_p, tau_d, a_p, a_d) in enumerate(param_sweep):

        tau_ratio = tau_p / tau_d
        a_ratio = a_p / a_d
        etas = ds[0, i, result_vars.index("etas"), :]

        in_degrees_0 = ds[:, i, result_vars.index("in-degrees_pre"), :]
        in_degrees_1 = ds[:, i, result_vars.index("in-degrees_post"), :]
        in_degree_diff = np.mean(in_degrees_1 - in_degrees_0, axis=0)
        in_degrees = np.mean(in_degrees_1, axis=0)

        out_degrees_0 = ds[:, i, result_vars.index("out-degrees_pre"), :]
        out_degrees_1 = ds[:, i, result_vars.index("out-degrees_post"), :]
        out_degree_diff = np.mean(out_degrees_1 - out_degrees_0, axis=0)
        out_degrees = np.mean(out_degrees_1, axis=0)

        ff_0 = ds[:, i, result_vars.index("fano-factors_pre"), :]
        ff_1 = ds[:, i, result_vars.index("fano-factors_post"), :]
        ff_diff = np.mean(ff_1 - ff_0, axis=0)

        evs_0 = ds[:, i, result_vars.index("eigvals_pre"), :]
        evs_1 = ds[:, i, result_vars.index("eigvals_post"), :]
        pr0 = np.sum(evs_0)**2/(np.sum(evs_0**2)*len(evs_0))
        pr1 = np.sum(evs_1)**2/(np.sum(evs_1**2)*len(evs_1))
        pr_diff = pr1 - pr0

        for j in range(len(etas)):
            results[c]["tau_ratio"].append(np.round(tau_ratio, decimals=2))
            results[c]["a_ratio"].append(np.round(a_ratio, decimals=2))
            results[c]["in_degree"].append(np.corrcoef(in_degrees, etas)[0][1])
            results[c]["in_degree_diff"].append(np.corrcoef(in_degree_diff, etas)[0][1])
            results[c]["out_degree"].append(np.corrcoef(out_degrees, etas)[0][1])
            results[c]["out_degree_diff"].append(np.corrcoef(out_degree_diff, etas)[0][1])
            results[c]["ff"].append(np.corrcoef(ff_1, etas)[0][1])
            results[c]["ff_diff"].append(np.corrcoef(ff_diff, etas)[0][1])
            results[c]["pr"].append(pr1)
            results[c]["pr_diff"].append(pr_diff)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 9))
    df = DataFrame.from_dict(results[c])
    ax = axes[0, 0]
    sb.heatmap(df.pivot_table(index="tau_ratio", columns="a_ratio", values="in_degree_diff"), vmin=-1.0, vmax=1.0, ax=ax, cmap="icefire")
    ax.set_title("corr(eta, delta(d_in))")
    ax = axes[0, 1]
    sb.heatmap(df.pivot_table(index="tau_ratio", columns="a_ratio", values="out_degree_diff"), vmin=-1.0, vmax=1.0, ax=ax, cmap="icefire")
    ax.set_title("corr(eta, delta(d_out))")
    ax = axes[1, 0]
    sb.heatmap(df.pivot_table(index="tau_ratio", columns="a_ratio", values="ff_diff"), ax=ax, cmap="icefire")
    ax.set_title("corr(eta, delta(ff))")
    ax = axes[1, 1]
    sb.heatmap(df.pivot_table(index="tau_ratio", columns="a_ratio", values="pr_diff"), ax=ax, cmap="icefire")
    ax.set_title("delta(pr)")
    plt.tight_layout()
    fig.suptitle(c)

plt.show()