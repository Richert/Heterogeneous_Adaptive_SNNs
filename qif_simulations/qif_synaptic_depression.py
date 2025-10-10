from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
tau = 1.0
Delta = 1.0
eta = -4.5
tau_a = 20.0
A0 = 0.0
J = 15.0
kappa = 0.0
tau_s = 0.5
I_ext = 0.0
noise_lvl = 80.0
noise_sigma = 1000.0

params = {
    'tau': tau, 'Delta': Delta, 'eta': eta, 'tau_a': tau_a, 'J': J, 'tau_s': tau_s, 'kappa': kappa, 'A0': A0
}

# define inputs
cutoff = 0.0
T = 1000.0 + cutoff
dt = 1e-3
dts = 1e-1
start = 500.0 + cutoff
stop = 1500.0 + cutoff
inp = np.zeros((int(T/dt),))
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise
inp[int(start/dt):int(stop/dt)] += I_ext

# run the mean-field model
##########################

# initialize model
op = "qif_sd_op"
ik = CircuitTemplate.from_yaml("../config/fre_equations/qif_sd")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                outputs={'r': f'p/{op}/r', 'a': f'p/{op}/a'},
                inputs={f'p/{op}/I_ext': inp}, clear=False)

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 5), sharex=True)
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r"])
ax.set_ylabel(r'$r(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("average firing rate")
ax = axes[1]
ax.plot(res_mf.index, res_mf["a"])
ax.set_ylabel(r'$a(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("Synaptic efficacy")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
