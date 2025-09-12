import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def get_xy(fr_source: float, fr_target: float, condition: str) -> tuple:
    if condition == "hebbian":
        x = fr_target * fr_source
        y = fr_target**2
    elif condition == "antihebbian":
        x = fr_source**2
        y = fr_target*fr_source
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def delta_w(w: float, x: float, y: float, b: float, a: float) -> float:
    return a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))

def get_w_solution(x: float, y: float, b: float, a: float, T: float, **kwargs) -> float:
    w0 = [0.0, 0.33, 0.66, 1.0]
    sols = solve_ivp(lambda t, w: delta_w(w, x, y, b, a), t_span=(0.0, T), y0=np.asarray(w0), **kwargs)
    ws = []
    for w in sols.y[:, -1]:
        if 0 <= w <= 1:
            ws.append(w)
    return np.random.choice(ws)

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.random.randn(N)

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# parameter definition
condition = "hebbian"
distribution = "lorentzian"
N = 1000
m = 100
eta = 1.0
deltas = np.linspace(0.1, 2.0, num=m)
target_fr = 0.2
a = 1.0
bs = [0.0, 0.125, 0.25, 0.5, 1.0]
res = {"b": bs, "delta": deltas, "data": {}}

# simulation parameters
T = 200.0
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    data = {"w": [], "fr": []}
    for Delta in deltas:

        # define source firing rate distribution
        inp = f(N, eta, Delta)
        fr_source = get_qif_fr(inp)

        # get weight solutions
        ws = []
        for source_fr in fr_source:
            x, y = get_xy(source_fr, target_fr, condition=condition)
            w = get_w_solution(x, y, b, a, T, **solver_kwargs)
            ws.append(w)

        # save results
        data["w"].append(ws)
        data["fr"].append(fr_source)

    print(f"Finished simulations for b = {b}")
    res["data"][b] = data

# plotting
fig, axes = plt.subplots(ncols=len(bs), figsize=(12, 3))
ticks = np.arange(0, m, int(m/5))
for i, b in enumerate(bs):

    # weight distribution
    ax = axes[i]
    im = ax.imshow(np.asarray(res["data"][b]["w"]), aspect="auto", interpolation="none", cmap="viridis",
                   vmax=1.0, vmin=0.0)
    ax.set_xlabel("neuron")
    ax.set_ylabel("Delta")
    ax.set_yticks(ticks, labels=np.round(deltas[ticks], decimals=1))
    # ax.set_title(f"W (b = {b})")

    # # firing rate distribution
    # ax = axes[1, i]
    # im = ax.imshow(np.asarray(res["data"][b]["fr"]), aspect="auto", interpolation="none", cmap="cividis", vmax=fr_max)
    # plt.colorbar(im, ax=ax)
    # ax.set_xlabel("eta")
    # ax.set_ylabel("Delta")
    # ax.set_title(f"Firing Rates (b = {b})")

fig.suptitle("Weight Distribution for Hebbian Learning (Rate Simulation)")
plt.tight_layout()
fig.canvas.draw()
plt.savefig(f"../results/ss_weight_distribution_{condition}_simulation.svg")
plt.show()
