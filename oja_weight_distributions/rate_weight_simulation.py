import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = fr_target * fr_source
        y = fr_target**2
    elif condition == "antihebbian":
        x = fr_source**2
        y = fr_target*fr_source
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def delta_w(w: np.ndarray, r_source: np.ndarray, eta: float, J: float, b: float, a: float, condition: str) -> np.ndarray:
    r_target = get_qif_fr(eta + J*np.dot(w, r_source) / N)
    x, y = get_xy(r_source, r_target, condition=condition)
    return a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))

def get_w_solution(w0: np.ndarray, r_source: np.ndarray, eta: float, J: float, b: float, a: float, T: float, **kwargs
                   ) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(w, r_source, eta, J, b, a, condition), t_span=(0.0, T), y0=np.asarray(w0),
                     **kwargs)
    return sols.y[:, -1]

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# parameter definition
condition = "antihebbian"
distribution = "gaussian"
N = 200
m = 100
eta = 1.0
deltas = np.linspace(0.1, 3.0, num=m)
target_eta = 0.2
a = 0.2
J = 8.0
bs = [0.0, 0.125]
res = {"b": bs, "w": {}}

# simulation parameters
T = 1000.0
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    ws = []
    for Delta in deltas:

        # define initial condition
        inp = f(N, eta, Delta)
        fr_source = get_qif_fr(inp)
        w0 = np.random.uniform(0.01, 0.99, size=(N,))

        # get weight solutions
        ws.append(get_w_solution(w0, fr_source, target_eta, J, b, a, T, **solver_kwargs))

    res["w"][b] = np.asarray(ws)
    print(f"Finished simulations for b = {b}")

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

fig.suptitle(f"{'Hebbian' if condition == 'hebbian' else 'Anti-Hebbian'} Learning (J = {int(J)}, Rate Simulation)")
fig.canvas.draw()
plt.savefig(f"../results/ss_weight_simulation_{condition}_{int(J)}.svg")
plt.show()
