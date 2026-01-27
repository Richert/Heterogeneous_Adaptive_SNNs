import h5py
import numpy as np
from mpi4py import MPI

# directory
path = "/home/richard/data/mpmf_simulations"

# sweep conditions
group = "antihebbian"
syn_types = ["exc", "inh"]
stp_types = ["sd", "sf"]

# sweep parameters
tau_p = np.arange(20.0, 110.0, 20.0)
tau_d = np.arange(20.0, 110.0, 20.0)
a_p = np.arange(0.001, 0.01, 0.002)
a_d = np.arange(0.001, 0.01, 0.002)
sweep_params = []
for tp in tau_p:
    for td in tau_d:
        for ap in a_p:
            for ad in a_d:
                sweep_params.append((tp, td, ap, ad))
n_params = len(sweep_params)
sweep_keys = ["tau_p", "tau_d", "a_p", "a_d"]

# default model parameters
n_reps = 10
M = 10
J = 0.0
Delta = 2.0
p = 1.0
eta = 0.0
b = 0.5
tau_s = 0.5
tau_a = 20.0
kappa = 0.0
default_params = {"J": J, "Delta": Delta, "M": M, "p": p, "eta": eta, "b": b, "tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}
result_vars = ["etas", "etas_pre", "etas_post",
               "in-degrees_pre", "out-degrees_pre", "in-degrees_post", "out-degrees_post",
               "fano-factors_pre", "eigvals_pre",  "fano-factors_post", "eigvals_post"
               ]

# condition-specific parameters
conditions = {"exc_sd": {"eta": -0.9, "J": 20.0, "kappa": 0.1},
              "exc_sf": {"eta": -0.5, "J": 10.0, "kappa": 0.6},
              "inh_sd": {"eta": 2.5, "J": -20.0, "kappa": 0.1},
              "inh_sf": {"eta": 2.5, "J": -15.0, "kappa": 0.6}}

# create storage file
f = h5py.File(f"{path}/mpmf_1pop_data.hdf5", "a", driver='mpio', comm=MPI.COMM_WORLD)
try:
    gr = f[group]
except KeyError:
    gr = f.create_group(group)

# store parameter sweep
if "param_sweep" in gr.keys():
    del gr["param_sweep"]
sw = gr.create_dataset("param_sweep", shape=(n_params, len(sweep_keys)))
sw[:] = np.asarray(sweep_params)
sw.attrs["parameters"] = sweep_keys

# create results datasets
gr.attrs["result_dims"] = ["trial_idx", "sweep_idx", "result_vars", "population_idx"]
gr.attrs["result_vars"] = result_vars
for syn in syn_types:
    for stp in stp_types:

        # create dataset
        ds_key = f"{syn}_{stp}"
        if ds_key in gr.keys():
            del gr[ds_key]
        ds = gr.create_dataset(ds_key, shape=(n_reps, n_params, len(result_vars), M))

        # store parameter values
        cond_params = default_params.copy()
        cond_params.update(conditions[f"{syn}_{stp}"])
        for key, val in cond_params.items():
            ds.attrs[key] = val

f.close()
