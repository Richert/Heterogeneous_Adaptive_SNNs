import numpy as np
import matplotlib.pyplot as plt

def get_xy(fr_source: float, fr_target: float, trace_source: float, trace_target: float, condition: str) -> tuple:
    if condition == "oja_hebbian":
        x = fr_target*trace_source
        y = fr_target*trace_target
    elif condition == "oja_antihebbian":
        x = fr_source*trace_source
        y = fr_source*trace_target
    elif condition == "stdp_hebbian":
        x = fr_target * trace_source
        y = fr_source * trace_target
    elif condition == "stdp_antihebbian":
        x = fr_source * trace_target
        y = fr_target * trace_source
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def qif_rhs(y: np.ndarray, eta: np.ndarray, tau_u: np.ndarray, tau_s: np.ndarray, J: np.ndarray,
            spikes: np.ndarray, N: int):
    v, s, u = y[:N], y[N:2*N], y[2*N:]
    dv = v**2 + eta + J @ s + noise*np.random.randn(2)*np.sqrt(dt)
    ds = (spikes-s) / tau_s
    du = (spikes-u) / tau_u
    return np.concatenate([dv, ds, du], axis=0)

def spiking(y: np.ndarray, spikes: np.ndarray, dt: float, v_cutoff: float):
    idx = np.argwhere((v_cutoff - y[:N]) < 0.0).squeeze()
    spikes[:] = 0.0
    y[idx] = v_reset
    spikes[idx] = 1.0/dt

def solve_ivp(T: float, dt: float, eta: np.ndarray, tau_u: np.ndarray, tau_s: np.ndarray, J: np.ndarray,
              v_cutoff: float, N: int):

    y = np.zeros((3*N,))
    spikes = np.zeros((N,))
    t = 0.0
    time, y_col = [], []
    while t < T:
        y_col.append(y[:])
        time.append(t)
        spiking(y, spikes, dt, v_cutoff)
        dy = qif_rhs(y, eta, tau_u, tau_s, J, spikes, N)
        y = y + dt * dy
        t += dt

    return np.asarray(y_col), np.asarray(time)

# parameter definition
N = 2
T = 20.0
dt = 1e-3
noise = 1e3
etas = np.asarray([0.8, 1.0])
tau_u = np.asarray([20.0, 20.0])
tau_s = np.asarray([5.0, 5.0])
J = np.zeros((N, N))
J[1, 0] = 5.0
v_cutoff = 60.0
v_reset = -20.0

# solve equations
ys, time = solve_ivp(T, dt, etas, tau_u, tau_s, J, v_cutoff, N)

# calculate quantities of interest
s_source, s_target, trace_source, trace_target = ys[:, N], ys[:, N+1], ys[:, 2*N], ys[:, 2*N+1]
x_h_oja, y_h_oja = get_xy(s_source, s_target, trace_source, trace_target, condition="oja_hebbian")
x_ah_oja, y_ah_oja = get_xy(s_source, s_target, trace_source, trace_target, condition="oja_antihebbian")
# x_h_stdp, y_h_stdp = get_xy(s_source, s_target, trace_source, trace_target, condition="stdp_hebbian")
# x_ah_stdp, y_ah_stdp = get_xy(s_source, s_target, trace_source, trace_target, condition="stdp_antihebbian")

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (3, 1.6)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 2

# plotting
fig, axes = plt.subplots(nrows=2, layout="constrained")
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0., wspace=0.)
ax = axes[0]
ax.plot(time, ys[:, 0], label="sender")
ax.plot(time, ys[:, 1], label="receiver")
ax.set_xticklabels(["" for tick in ax.get_xticks()])
# ax.legend()
ax.set_ylabel(r"$v$")
ax.set_title("Plasticity rule")
ax = axes[1]
ax.plot(time, x_ah_oja, label="LTP", color="black")
ax.plot(time, y_ah_oja, label="LTD", color="darkorange")
# ax.plot(time, x_ah_oja, label="LTP, anti-Hebbian", color="black", linestyle="dashed")
# ax.plot(time, y_ah_oja, label="LTD, anti-Hebbian", color="darkorange", linestyle="dashed")
# ax.legend()
# ax1.legend()
ax.set_ylabel(r"model")
ax.set_xlabel("rate")
# ax.set_title(r"$w$ dynamics")
# ax = axes[2]
# ax.plot(time, x_h_stdp, label="x (hebbian)")
# ax.plot(time, y_h_stdp, label="y (hebbian)")
# ax.plot(time, x_ah_stdp, label="x (anti-hebbian)")
# ax.plot(time, y_ah_stdp, label="y (anti-hebbian)")
# ax.legend()
# ax.set_ylabel("LTP/LTD")
# ax.set_title("Plasticity drivers for STDP")

fig.canvas.draw()
plt.savefig(f"../results/figures/qif_sender_receiver_dynamics.svg")
plt.show()