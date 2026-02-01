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
gr = f["stdp_sym"]

# load parameter sweep
param_sweep = gr["param_sweep"]
params = param_sweep.attrs["parameters"]

# define different sweep conditions
conditions = list(gr.keys())
conditions.pop(conditions.index("param_sweep"))

# get results
results = {key: {"eta": [], "stdp_ratio": [], "in_degree": [], "in_degree_diff": [], "out_degree": [],
                 "out_degree_diff": [], "ff": [], "ff_diff": [], "pr": [], "pr_diff": []} for key in conditions}
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
        ff = np.mean(ff_1, axis=0)

        evs_0 = ds[:, i, result_vars.index("eigvals_pre"), :]
        evs_1 = ds[:, i, result_vars.index("eigvals_post"), :]
        pr0 = np.sum(evs_0)**2/(np.sum(evs_0**2)*len(evs_0))
        pr1 = np.sum(evs_1)**2/(np.sum(evs_1**2)*len(evs_1))
        pr_diff = pr1 - pr0

        for j in range(len(etas)):
            results[c]["eta"].append(etas[j])
            results[c]["stdp_ratio"].append(np.round(tau_ratio*a_ratio, decimals=2))
            results[c]["in_degree"].append(in_degrees[j])
            results[c]["in_degree_diff"].append(in_degree_diff[j])
            results[c]["out_degree"].append(out_degrees[j])
            results[c]["out_degree_diff"].append(out_degree_diff[j])
            results[c]["ff"].append(ff[j])
            results[c]["ff_diff"].append(ff_diff[j])
            results[c]["pr"].append(pr1)
            results[c]["pr_diff"].append(pr_diff)

    # visualization of changes
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 9))
    df = DataFrame.from_dict(results[c])
    ax = axes[0, 0]
    sb.heatmap(df.pivot_table(index="stdp_ratio", columns="eta", values="in_degree_diff"), ax=ax, cmap="icefire")
    ax.set_title("delta(d_in)")
    ax = axes[0, 1]
    sb.heatmap(df.pivot_table(index="stdp_ratio", columns="eta", values="out_degree_diff"), ax=ax, cmap="icefire")
    ax.set_title("delta(d_out)")
    ax = axes[1, 0]
    sb.heatmap(df.pivot_table(index="stdp_ratio", columns="eta", values="ff_diff"), ax=ax, cmap="icefire")
    ax.set_title("delta(ff)")
    ax = axes[1, 1]
    sb.lineplot(df, x="stdp_ratio", y="pr_diff", ax=ax)
    ax.set_xscale("log")
    ax.set_title("delta(pr)")
    fig.suptitle(f"{c}, t1-t0")
    plt.tight_layout()

    # visualization of final network state
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 9))
    df = DataFrame.from_dict(results[c])
    ax = axes[0, 0]
    sb.heatmap(df.pivot_table(index="stdp_ratio", columns="eta", values="in_degree"), ax=ax, cmap="icefire")
    ax.set_title("d_in")
    ax = axes[0, 1]
    sb.heatmap(df.pivot_table(index="stdp_ratio", columns="eta", values="out_degree"), ax=ax, cmap="icefire")
    ax.set_title("d_out")
    ax = axes[1, 0]
    sb.heatmap(df.pivot_table(index="stdp_ratio", columns="eta", values="ff"), ax=ax, cmap="icefire")
    ax.set_title("ff")
    ax = axes[1, 1]
    sb.lineplot(df, x="stdp_ratio", y="pr", ax=ax)
    ax.set_xscale("log")
    ax.set_title("pr")
    fig.suptitle(f"{c}, t1")
    plt.tight_layout()

plt.show()