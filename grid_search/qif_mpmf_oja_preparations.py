import h5py
import numpy as np

# directory
path = "/home/rgast/data/mpmf_simulations"

# sweep conditions
syn_types = ["exc", "inh"]
stp_types = ["sd", "sf"]

# sweep parameters
a_p = np.linspace(start=0.01, stop=0.1, num=10)
a_d = np.linspace(start=0.01, stop=0.1, num=10)
n_sweep = len(a_p)*len(a_d)

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
conditions = {"exc_sd": {"eta": -0.9, "J": 20.0, "kappa": 0.1}, "exc_sf": {"eta": -0.35, "J": 10.0, "kappa": 0.4},
              "inh_sd": {"eta": 3.0, "J": -10.0, "kappa": 0.2}, "inh_sf": {"eta": 1.0, "J": -10.0, "kappa": 0.2}}

# prepare storage file
f = h5py.File(f"{path}/1pop_data.hdf5", "a")
try:
    gr = f["oja"]
except KeyError:
    gr = f.create_group("oja")
for syn in syn_types:
    for stp in stp_types:

        # create dataset
        ds_key = f"{syn}_{stp}"
        if ds_key in gr.keys():
            del gr[ds_key]
        ds = gr.create_dataset(ds_key, shape=(n_reps, n_sweep, len(result_vars), M))

        # store parameter values
        cond_params = default_params.copy()
        cond_params.update(conditions[f"{syn}_{stp}"])
        for key, val in cond_params.items():
            ds.attrs[key] = val
        ds.attrs["column_vars"] = result_vars

