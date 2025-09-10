import numpy as np
import matplotlib.pyplot as plt

def get_xy(fr_source: float, fr_target: float, trace_source: float, trace_target: float, condition: str) -> tuple:
    if condition == "oja_hebbian":
        x = fr_source*trace_target
        y = trace_target*fr_target
    elif condition == "oja_antihebbian":
        x = trace_source*fr_source
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
    dv = v**2 + eta + J @ s
    ds = (spikes-s) / tau_s
    du = (spikes-u) / tau_u
    return np.concatenate([dv, ds, du], axis=0)

def spiking(y: np.ndarray, spikes: np.ndarray, dt: float, v_cutoff: float):
    idx = np.argwhere((v_cutoff - y[:N]) < 0.0).squeeze()
    spikes[:] = 0.0
    y[idx] = -v_cutoff
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
T = 100.0
dt = 1e-3
etas = np.asarray([0.5, 0.8])
tau_u = np.asarray([20.0, 20.0])
tau_s = np.asarray([5.0, 5.0])
J = np.zeros((N, N))
J[1, 0] = 5.0
v_cutoff = 100.0

# solve equations
ys, time = solve_ivp(T, dt, etas, tau_u, tau_s, J, v_cutoff, N)

# calculate quantities of interest
condition = "stdp_antihebbian"
s_source, s_target, trace_source, trace_target = ys[:, N], ys[:, N+1], ys[:, 2*N], ys[:, 2*N+1]
x, y = get_xy(s_source, s_target, trace_source, trace_target, condition=condition)

# plotting
fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
ax = axes[0]
ax.plot(time, ys[:, 0], label="sender")
ax.plot(time, ys[:, 1], label="receiver")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("v")
ax.set_title("QIF dynamics")
ax = axes[1]
ax.plot(time, ys[:, N], label="sender")
ax.plot(time, ys[:, N+1], label="receiver")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("s")
ax = axes[2]
ax.plot(time, x, label="LTP")
ax.plot(time, y, label="LTD")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("trace")
ax.set_title("plasticity drivers")
plt.tight_layout()
plt.show()
