import h5py

# directory
path = "/home/rgast/data/mpmf_simulations"

# sweep conditions
syn_types = ["exc", "inh"]
stp_types = ["sd", "sf"]
ltp_types = ["oja", "stdp"]

# default model parameters
n_reps = 10
M = 10
J = 0.0
Delta = 2.0
p = 1.0
eta = 0.0
b = 0.2
tau_s = 0.5
tau_a = 20.0
kappa = 0.0
default_params = {"J": J, "Delta": Delta, "M": M, "p": p, "eta": eta, "b": b, "tau_s": tau_s, "tau_a": tau_a, "kappa": kappa}
result_vars = ["etas", "in-degrees", "out-degrees", "fano-factors", "lambdas", "etas_transformed"]

# condition-specific parameters
conditions = {"exc_sd": {"eta": -1.1, "J": 15.0, "kappa": 0.2}, "exc_sf": {"eta": -1.1, "J": 15.0, "kappa": 0.2},
              "inh_sd": {"eta": 1.0, "J": -15.0, "kappa": 0.2}, "inh_sf": {"eta": 1.0, "J": -15.0, "kappa": 0.2}}

# prepare storage file
f = h5py.File(f"{path}/1pop_data.hdf5", "a")
for syn in syn_types:
    for stp in stp_types:
        for ltp in ltp_types:

            # create dataset
            ds_key = f"{syn}_{stp}_{ltp}"
            if ds_key in f.keys():
                del f[ds_key]
            ds = f.create_dataset(ds_key, shape=(n_reps, len(result_vars), M))

            # store parameter values
            cond_params = default_params.copy()
            cond_params.update(conditions[f"{syn}_{stp}"])
            for key, val in cond_params.items():
                ds.attrs[key] = val
            ds.attrs["column_vars"] = result_vars
