import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def exp_conv(time, x, tau):
    y = 0.0
    dt = time[1] - time[0]
    y_col = np.zeros_like(x)
    for n in range(len(time)):
        y = y + dt*(x[n] - y/tau)
        y_col[n] = y
    return y_col

# define time parameters
n_time = 5000
time = np.linspace(0, 600.0, n_time)
spike_times = np.arange(0, n_time, step=50)
spike_pre = np.zeros((n_time,))
t_pre = int(0.5*n_time)
dt = time[1]-time[0]
spike_pre[t_pre] = 1.0
spike_post = np.zeros((n_time,))

# define stdp parameters
tau_s = 5.0
a = 1e-2
a_r = 2.0
tau_p = 10.0
tau_d = 20.0
a_p = a*a_r
a_d = a/a_r

# calculate pre-synaptic trace variables
s_pre = exp_conv(time, spike_pre, tau_s)
u_pre_ltp = exp_conv(time, spike_pre, tau_p)
u_pre_ltd = exp_conv(time, spike_pre, tau_d)

results = { "spike_time_diff": [], "stdp_sym": [], "stdp_asym": [], "oja": [], "anti": [], "anti_oja": []}
for t_post in spike_times:

    results["spike_time_diff"].append(time[t_post] - time[t_pre])

    # set spike time
    spike_post[:] = 0.0
    spike_post[t_post] = 1.0

    # calculate post-synaptic trace variables
    s_post = exp_conv(time, spike_post, tau_s)
    u_post_ltp = exp_conv(time, spike_post, tau_p)
    u_post_ltd = exp_conv(time, spike_post, tau_d)

    # combine synaptic trace variables to asymmetric stdp update
    ltp = a_p*u_pre_ltp*s_post
    ltd = a_d*s_pre*u_post_ltd
    results["stdp_asym"].append(np.sum(ltp - ltd))

    # combine synaptic trace variables to symmetric stdp update
    ltp = a_p*u_pre_ltp*u_post_ltp
    ltd = a_d*u_pre_ltd*u_post_ltd
    results["stdp_sym"].append(np.sum(ltp - ltd))

    # combine synaptic trace variables to antihebbian stdp update
    ltp = a_p*s_pre*u_post_ltp
    ltd = a_d*u_pre_ltd*s_post
    results["anti"].append(np.sum(ltp - ltd))

    # combine synaptic trace variables to oja update
    ltp = a_p*u_pre_ltp*s_post
    ltd = a_d*s_post*u_post_ltd
    results["oja"].append(np.sum(ltp - ltd))

    # combine synaptic trace variables to anti-oja update
    ltp = a_p*u_pre_ltp*s_pre
    ltd = a_d*u_pre_ltd*s_post
    results["anti_oja"].append(np.sum(ltp - ltd))

    # test plotting
    # fig, ax = plt.subplots(figsize=(10,3))
    # ax.plot(time, ltp, label="LTP")
    # ax.plot(time, ltd, label="LTD")
    # ax.plot(time, ltp-ltd, label="STDP")
    # ax.legend()
    # fig, ax = plt.subplots(figsize=(10, 3))
    # ax.plot(time, s_pre, label="s_pre")
    # ax.plot(time, s_post, label="s_post")
    # ax.legend()
    # fig, ax = plt.subplots(figsize=(10, 3))
    # ax.plot(time, u_pre_ltd, label="u_pre_ltd")
    # ax.plot(time, u_pre_ltp, label="u_pre_ltp")
    # ax.plot(time, u_post_ltd, label="u_post_ltd")
    # ax.plot(time, u_post_ltp, label="u_post_ltp")
    # ax.legend()
    # plt.show()

# plotting
keys = ["stdp_sym", "stdp_asym", "anti", "oja", "anti_oja"]
fig, axes = plt.subplots(ncols=len(keys), sharex=True, sharey=True, figsize=(16,4))
for i, key in enumerate(keys):
    ax = axes[i]
    ax.plot(results["spike_time_diff"], results[key])
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xlabel("t_post-t_pre (ms)")
    ax.set_ylabel("dW")
    ax.set_title(key)
fig.tight_layout()
plt.show()