import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
from numba import njit
from scipy.signal import welch
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('tkagg')
import sys
from mpi4py import MPI
import h5py
from config.utility_functions import *

# define data directory
path = "/home/richard/data/mpmf_simulations"

# read sweep condition
trial = int(sys.argv[-1])
syn = str(sys.argv[-2])
stp = str(sys.argv[-3])
group = str(sys.argv[-4])

# load data file
f = h5py.File(f"{path}/qif_1pop_data.hdf5", "r+", driver='mpio', comm=MPI.COMM_WORLD)
gr = f[group]

# load sweep parameters
sweep_params = gr["param_sweep"]
sweep_keys = sweep_params.attrs["parameters"]

# load condition-specific model parameters
ds = gr[f"{syn}_{stp}"]
M = ds.attrs["M"]

# set model parameters
node_params = ["eta", "Delta", "J"]
syn_params = ["tau_s", "tau_a", "kappa"]
plasticity_params = ["b"]
node_vars, syn_vars, edge_vars = {}, {}, {}
for d, keys in zip([node_vars, syn_vars, edge_vars], [node_params, syn_params, plasticity_params]):
    for key in keys:
        d[key] = ds.attrs[key]
node_vars["eta"] = uniform(M, node_vars["eta"], node_vars["Delta"])
node_vars["Delta"] = node_vars["Delta"]/(2*M)
node_vars["J"] = node_vars["J"]/(0.5*M)
edge_vars["a_p"] = 0.0
edge_vars["a_d"] = 0.0

# simulation parameters
cutoff = 100.0
T = 2000.0
dt = 1e-3
dts = 1.0
noise_tau = 200.0
noise_scale = 0.02

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_stdp_{stp}", "qif_op", f"syn_{stp}_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        if group == "stdp_asym":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/p2": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{i}/ltd_op/u_d",
                           }))
        elif group == "stdp_sym":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{i}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{i}/ltd_op/u_d",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        elif group == "antihebbian":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p2": f"p{i}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        elif group == "oja":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{i}/ltd_op/u_d",
                           }))
        elif group == "antioja":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": 1.0,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        else:
            raise ValueError(f"Unknown group {group}")
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})

# generate run function
inp = np.zeros((int(T/dt), 1), dtype=np.float32)
func, args, arg_keys, _ = net.get_run_func(f"{syn}_{stp}_vectorfield", file_name=f"{syn}_{stp}_run",
                                           step_size=dt, backend="numpy", solver="heun", float_precision="float32",
                                           vectorize=True, inputs={f"all/{node_op}/I_ext": inp}, clear=False)
func_njit = njit(func)
func_njit(*args)
rhs = func_njit

# find argument positions of free parameters
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")
a_p_idx = arg_keys.index(f"{edge}/{edge_op}/a_p")
a_d_idx = arg_keys.index(f"{edge}/{edge_op}/a_d")
tau_p_idx = arg_keys.index(f"p0/ltp_op/tau_p")
tau_d_idx = arg_keys.index(f"p0/ltd_op/tau_d")
eta_idx = arg_keys.index(f"p0/{node_op}/eta")

args = list(args)
for i, (tau_p, tau_d, a_p, a_d) in enumerate(sweep_params):

    # set LTP/LTD time constants
    args[tau_p_idx] = tau_p
    args[tau_d_idx] = tau_d

    # set random initial connectivity
    W0 = np.random.uniform(low=0.0, high=1.0, size=(M, M))
    args[1][-int(M*M):] = W0.reshape((int(M*M),))

    # define extrinsic input
    noise = np.asarray(generate_colored_noise(int(T/dt), noise_tau, noise_scale), dtype=np.float32)

    # set initial state
    init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)

    # run initial simulation
    args[inp_idx] = noise
    y0_hist, y0 = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)

    # turn on synaptric plasticity and run simulation again
    args[a_p_idx] = a_p
    args[a_d_idx] = a_d
    y1_hist, y1 = integrate(y0, rhs, tuple(args[2:]), T, dt, dts)
    W1 = y1[-int(M * M):].reshape(M, M)

    # turn off synaptic plasticity and run simulation a final time
    args[a_p_idx] = 0.0
    args[a_d_idx] = 0.0
    y2_hist, y2 = integrate(y1, rhs, tuple(args[2:]), T, dt, dts)

    # calculate in- and out-degrees
    in_degree_pre = np.sum(W0, axis=1)
    out_degree_pre = np.sum(W0, axis=0)
    in_degree_post = np.sum(W1, axis=1)
    out_degree_post = np.sum(W1, axis=0)

    # calculate network covariance eigenvalues
    r0, r1, r2 = y0_hist[:, :M], y1_hist[:, :M], y2_hist[:, :M]
    eigvals_pre, eigvecs_pre = get_eigs(r0)
    eigvals_post, eigvecs_post = get_eigs(r2)

    # transform etas into covariance eigenvector space
    etas = args[eta_idx]
    etas_pre = np.dot(eigvecs_pre.T, etas)
    etas_post = np.dot(eigvecs_post.T, etas)

    # get PSD of first PC
    pc1_pre, pc1_post = np.dot(r0*100.0, eigvecs_pre[:, 0]), np.dot(r2, eigvecs_post[:, 0])
    fs_pre, ps_pre = welch(pc1_pre, fs=100.0/dts, nperseg=512)
    fs_post, ps_post = welch(pc1_post, fs=100.0/dts, nperseg=512)
    f_max_pre, f_max_post = fs_pre[np.argmax(ps_pre)], fs_post[np.argmax(ps_post)]
    pow_pre = (fs_pre[1] - fs_pre[0]) * np.sum(ps_pre)
    pow_post = (fs_post[1] - fs_post[0]) * np.sum(ps_post)

    # calculate correlation between input and first PC
    in_corr_pre = np.corrcoef(pc1_pre, noise[::int(dts/dt)])[0, 1]
    in_corr_post = np.corrcoef(pc1_post, noise[::int(dts/dt)])[0, 1]

    # calculate fano factors
    ff_pre = get_ff(r0)
    ff_post = get_ff(r2)

    # save results
    results = {"etas": etas, "etas-pre": etas_pre, "etas-post": etas_post,
               "in-degrees-pre": in_degree_pre, "out-degrees-pre": out_degree_pre,
               "in-degrees-post": in_degree_post, "out-degrees-post": out_degree_post,
               "eigvals-pre": eigvals_pre, "eigvals-post": eigvals_post,
               "fano-factors-pre": ff_pre, "fano-factors-post": ff_post,
               "sig-pow-pre": pow_pre, "sig-pow-post": pow_post,
               "max-freq-pre": f_max_pre, "max-freq-post": f_max_post,
               "in-corr-pre": in_corr_pre, "in-corr-post": in_corr_post,
               }
    for j, key in enumerate(gr.attrs["result_vars"]):
        ds[trial, i, j, :] = results[key]

    # plotting weights
    # fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    # ax = axes[0]
    # im = ax.imshow(W0, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0)
    # ax.set_title("Initial Weights")
    # ax = axes[1]
    # ax.imshow(W1, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0)
    # ax.set_title("Final Weights")
    #
    # # plotting dynamics
    # fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
    # ax = axes[0]
    # ax.plot(np.mean(r0, axis=1), label="r0")
    # ax.plot(np.mean(r1, axis=1), label="r1")
    # ax.plot(np.mean(r2, axis=1), label="r2")
    # ax.legend()
    # ax.set_xlabel("time")
    # ax.set_ylabel("r")
    # ax.set_title("firing rate")
    # ax = axes[1]
    # u = y1_hist[:, 4*M:5*M]
    # ax.plot(u)
    # ax.set_xlabel("time")
    # ax.set_ylabel("u")
    # ax.set_title("LTP trace variables")
    # ax = axes[2]
    # u = y1_hist[:, 5 * M:6 * M]
    # ax.plot(u)
    # ax.set_xlabel("time")
    # ax.set_ylabel("u")
    # ax.set_title("LTD trace variables")
    # plt.tight_layout()
    #
    # # plotting DV relationships
    # fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    # ax = axes[0]
    # ax.plot(etas, in_degree_pre, color="royalblue", linestyle="dashed", label="in-degree (pre)")
    # ax.plot(etas, out_degree_pre, color="darkorange", linestyle="dashed", label="out-degree (pre)")
    # ax.plot(etas, in_degree_post, color="royalblue", linestyle="solid", label="in-degree (post)")
    # ax.plot(etas, out_degree_post, color="darkorange", linestyle="solid", label="out-degree (post)")
    # ax.legend()
    # ax.set_xlabel("eta")
    # ax.set_ylabel("degree")
    # ax.set_title("Nodal Connectivity")
    # ax = axes[1]
    # ax.plot(etas, ff_pre, label="pre")
    # ax.plot(etas, ff_post, label="post")
    # ax.legend()
    # ax.set_xlabel("eta")
    # ax.set_ylabel("fano factor")
    # ax.set_title("Nodal Dynamics")
    # ax = axes[2]
    # ax.scatter(etas_pre, np.log(eigvals_pre + 1e-12), label="pre")
    # ax.scatter(etas_post, np.log(eigvals_post + 1e-12), label="post")
    # ax.legend()
    # ax.set_xlabel("sum(eta*v)")
    # ax.set_ylabel("log(lambda)")
    # ax.set_title("Nodal Covariance")
    # plt.tight_layout()
    #
    # plt.show()

# clear files up
f.close()
clear(net)
