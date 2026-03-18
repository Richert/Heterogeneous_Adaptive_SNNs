import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def rhs(y, eta, tau_u, r0, alpha):
    v, u, x = y[:]
    dy = np.zeros_like(y)
    dy[0] = v**2 + eta + alpha*u
    dy[1] = x
    dy[2] = r0 - (2*x + u/tau_u)/tau_u
    return dy

def integrate(T, dt, dts, *args):
    steps = int(T/dt)
    sampling_steps = int(dts/dt)
    y_col = np.zeros((int(T/dts), 3))
    y = y_col[0, :]
    idx = 0
    for step in range(steps):
        y = y + dt*rhs(y, *args)
        if y[0] > v_p:
            y[0] = v_r
            y[2] -= 1.0
        if step % sampling_steps == 0:
            y_col[idx, :] = y[:]
            idx += 1
    return y_col

# define parameters
T = 500.0
dt = 1e-4
dts = 1e-2
eta = -10.0
tau_u = 50.0
r0 = 0.1 #np.sqrt(eta)/np.pi + 0.01
alpha = 0.5
v_p = 100.0
v_r = -10.0
args = (eta, tau_u, r0, alpha)

# simulation
y = integrate(T, dt, dts, *args)

# plotting
fig, axes = plt.subplots(nrows=3, figsize=(16, 6))
for i, v in enumerate(["v", "u", "x"]):
    ax = axes[i]
    ax.plot(y[:, i])
    ax.set_ylabel(v)
ax.set_xlabel("time")
plt.tight_layout()
plt.show()
