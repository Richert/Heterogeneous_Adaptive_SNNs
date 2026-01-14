import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

@njit
def integrate_noise(x, inp, scale, tau):
    return x + scale * inp - x / tau

def generate_colored_noise(num_samples, tau, scale=1.0):
    """
    Generates Brownian noise by integrating white noise.

    Args:
        num_samples (int): The number of samples in the output Brownian noise.
        scale (float): A scaling factor for the noise amplitude.

    Returns:
        numpy.ndarray: An array containing the generated Brownian noise.
    """
    white_noise = np.random.randn(num_samples)
    x = 0.0
    colored_noise = np.zeros_like(white_noise)
    for sample in range(num_samples):
        x = integrate_noise(x, white_noise[sample], scale, tau)
        colored_noise[sample] = x
    return colored_noise

# parameters
path = "/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs"
rep = 0 #int(sys.argv[-1])
b = 0.1 #float(sys.argv[-2])
Delta = 0.5 #float(sys.argv[-3])
eta = -1.9
node_vars = {"tau": 1.0, "J": 15.0, "eta": eta, "Delta": Delta}
syn_vars = {"tau_s": 1.0, "tau_a": 20.0, "kappa": 0.1}
ca_vars = {"tau_u": 100.0}
T = 1000.0
dt = 5e-4
dts = 1.0
noise_tau = 100.0
noise_scale = 0.1

# create network
model = "qif_sd"
node_op, syn_op = "qif_op", "syn_sd_op"
net = CircuitTemplate.from_yaml(f"../config/fre_equations/{model}")
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})
net.update_var(node_vars={f"all/ca_op/{key}": val for key, val in ca_vars.items()})

# define extrinsic input
inp = generate_colored_noise(int(T/dt), noise_tau, noise_scale)

# run simulation
res = net.run(simulation_time=T, step_size=dt, inputs={f"p/{node_op}/I_ext": inp},
              outputs={"r": f"p/{node_op}/r"}, solver="heun", clear=False, sampling_step_size=dts,
              float_precision="float64")

# plotting
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(res["r"])
plt.show()
# pickle.dump(
#     {"W": W, "eta": etas[idx], "b": b, "Delta": Delta, "noise": noise_lvl, "s": np.mean(res["s"].values, axis=1)},
#     open(f"{path}/results/rnn_results/fre_inh_{int(b*10)}_{int(noise_lvl)}_{int(Delta*10.0)}_{rep}.pkl", "wb")
# )
