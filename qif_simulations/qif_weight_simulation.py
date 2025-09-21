import numpy as np
import matplotlib.pyplot as plt

def get_xy(fr_source: float, fr_target: float, trace_source: float, trace_target: float, condition: str) -> tuple:
    if condition == "oja_hebbian":
        x = trace_source*fr_target
        y = trace_target*fr_target
    elif condition == "oja_antihebbian":
        x = trace_source*fr_source
        y = trace_source*fr_target
    elif condition == "stdp_hebbian":
        x = trace_source*fr_target
        y = trace_target*fr_source
    elif condition == "stdp_antihebbian":
        x = trace_target*fr_source
        y = trace_source*fr_target
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def qif_rhs(y: np.ndarray, spikes: np.ndarray, eta: np.ndarray, tau_s: float, tau_u: float, J: float, a: float,
            b: float, N: int, condition: str):
    v, s, u, w = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:]
    x, y = get_xy(s[:], s[-1], u[:], u[-1], condition=condition)
    dv = v**2 + eta
    dv[-1] += J*np.dot(w[:-1], s[:-1]) / (N-1)
    ds = (spikes-s) / tau_s
    du = (spikes-u) / tau_u
    dw = a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))
    return np.concatenate([dv, ds, du, dw], axis=0)

def spiking(y: np.ndarray, spikes: np.ndarray, dt: float, v_cutoff: float, N: int):
    idx = np.argwhere((v_cutoff - y[:N]) < 0.0).squeeze()
    spikes[:] = 0.0
    y[idx] = -y[idx]
    spikes[idx] = 1.0/dt

def solve_ivp(T: float, dt: float, eta: np.ndarray, tau_s: float, tau_u: float, J: float,
              a: float, b: float, v_cutoff: float, N: int, condition: str):

    y = np.zeros((4*N,))
    y[3*N:] = np.random.uniform(0.01, 0.99, size=(N,))
    spikes = np.zeros((N,))
    t = 0.0

    while t < T:
        spiking(y, spikes, dt, v_cutoff, N)
        dy = qif_rhs(y, spikes, eta, tau_s, tau_u, J, a, b, N, condition)
        y = y + dt * dy
        t += dt

    return y[3*N:-1]

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

# parameter definition
condition = "stdp_antihebbian"
distribution = "gaussian"
N = 10
m = 10
eta = 1.0
deltas = np.linspace(0.1, 3.0, num=m)
target_eta = 0.5
a = 0.2
bs = [0.0, 0.125]
tau_s = 1.0
tau_u = 10.0
J = 10.0
v_cutoff = 100.0
res = {"b": bs, "w": {}}

# simulation parameters
T = 2000.0
dt = 1e-3
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    ws = []
    for Delta in deltas:

        # define source firing rate distribution
        etas = np.asarray(f(N, eta, Delta).tolist() + [target_eta])

        # solve equations
        ws.append(solve_ivp(T, dt, etas, tau_s, tau_u, J, a, b, v_cutoff, N+1, condition))
        print(f"Finished simulations for b = {b} and Delta = {np.round(Delta, decimals=1)}")

    # save results
    res["w"][b] = np.asarray(ws)

# plotting
fig, axes = plt.subplots(ncols=len(bs), figsize=(3*len(bs), 3), layout="constrained")
ticks = np.arange(0, m, int(m/5))
for i, b in enumerate(bs):

    # weight distribution
    ax = axes[i]
    im = ax.imshow(np.asarray(res["w"][b]), aspect="auto", interpolation="none", cmap="viridis", vmax=1.0, vmin=0.0)
    ax.set_xlabel("neuron")
    ax.set_ylabel("Delta")
    ax.set_yticks(ticks, labels=np.round(deltas[ticks], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"W (b = {b})")

    # # firing rate distribution
    # ax = axes[1, i]
    # im = ax.imshow(np.asarray(res["data"][b]["fr"]), aspect="auto", interpolation="none", cmap="cividis", vmax=fr_max)
    # plt.colorbar(im, ax=ax)
    # ax.set_xlabel("eta")
    # ax.set_ylabel("Delta")
    # ax.set_title(f"Firing Rates (b = {b})")

fig.suptitle(f"{'Anti-Hebbian' if 'antihebbian' in condition else 'Hebbian'} Learning (J = {int(J)}, QIF Simulation)")
fig.canvas.draw()
plt.savefig(f"../results/qif_weight_simulation_{condition}_{int(J)}.svg")
plt.show()
