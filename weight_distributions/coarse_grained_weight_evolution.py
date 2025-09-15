import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def second_derivative(w, x, y, b):
    return -b*(x+y)  + (1-b)*(x-y) - 2*w*(1-b)*(x-y)

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = np.outer(fr_target, fr_source)
        y = np.repeat((fr_target**2).reshape(len(fr_target), 1), len(fr_source), axis=1)
    elif condition == "antihebbian":
        x = np.repeat((fr_source**2).reshape(len(fr_source), 1), len(fr_target), axis=1)
        y = np.outer(fr_target, fr_source)
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def get_w_solution(w0: float, x: float, y: float, b: float) -> float:
    if b < 1.0 and x != y:
        a_term = 2*(b-1) * (x-y)
        b_term = x*(2*b-1) + y
        sqrt_term = np.sqrt((x-y)**2 + 4*x*y*b**2)
        w1 = (b_term + sqrt_term) / a_term
        w2 = (b_term - sqrt_term) / a_term
        ws = []
        for w in (w1, w2):
            sd = second_derivative(w, x, y, b)
            if 0 <= w <= 1 and (sd <= 0.0 or np.abs(w0 - w) < 1e-6):
                ws.append(w)
        return np.random.choice(ws)
    elif x + y > 0.0:
        return x / (x + y)
    else:
        return w0

def evolve_w(w: np.ndarray, etas: np.ndarray, J: float, b: float, steps: int, condition: str) -> np.ndarray:
    r = get_qif_fr(etas)
    for step in range(steps):
        r = get_qif_fr(etas + J * (w @ r))
        x, y = get_xy(r, r, condition)
        for i in range(w0.shape[0]):
            for j in range(w0.shape[1]):
                w[i, j] = get_w_solution(w[i, j], x[i, j], y[i, j], b)
    return w

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta*np.random.randn(N)
    return np.sort(etas)

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# parameter definition
condition = "hebbian"
distribution = "gaussian"
N = 100
m = 4
eta = 0.0
deltas = np.linspace(0.1, 3.0, num=m)
J = 0.1
bs = [0.0, 0.25]
res = {"b": [], "delta": [], "w": [], "eta": []}

# simulation parameters
steps = 10
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    for Delta in deltas:

        # define initial condition
        inp = f(N, eta, Delta)
        w0 = np.random.uniform(0.01, 0.99, size=(N, N))

        # get weight solutions
        w = evolve_w(w0, inp, J, b, steps, condition)

        # save results
        res["w"].append(w)
        res["b"].append(b)
        res["delta"].append(Delta)
        res["eta"].append(inp)
        print(f"Finished simulations for b = {b} and Delta = {Delta}")

# plotting
fig, axes = plt.subplots(ncols=len(deltas), nrows=len(bs), figsize=(12, 6))
ticks = np.arange(0, N, int(N/5))
for i, b in enumerate(bs):
    for j, Delta in enumerate(deltas):

        # weight distribution
        ax = axes[i, j]
        idx1 = np.asarray(res["b"]) == b
        idx2 = np.asarray(res["delta"]) == Delta
        idx = np.argwhere((idx1 * idx2) > 0.5).squeeze()
        im = ax.imshow(np.asarray(res["w"][idx]), aspect="auto", interpolation="none", cmap="viridis",
                       vmax=1.0, vmin=0.0)
        ax.set_xlabel("source neuron eta")
        ax.set_ylabel("target neuron eta")
        ax.set_yticks(ticks, labels=np.round(res["eta"][idx][ticks], decimals=1))
        ax.set_xticks(ticks, labels=np.round(res["eta"][idx][ticks], decimals=1))
        ax.set_title(f"W (b = {b}, Delta = {np.round(Delta, decimals=1)}")

        # # firing rate distribution
        # ax = axes[1, i]
        # im = ax.imshow(np.asarray(res["data"][b]["fr"]), aspect="auto", interpolation="none", cmap="cividis", vmax=fr_max)
        # plt.colorbar(im, ax=ax)
        # ax.set_xlabel("eta")
        # ax.set_ylabel("Delta")
        # ax.set_title(f"Firing Rates (b = {b})")

fig.suptitle(f"Weight Distribution for {'Hebbian' if condition == 'hebbian' else 'Anti-Hebbian'} Learning (Simulation)")
plt.tight_layout()
fig.canvas.draw()
plt.savefig(f"../results/rnn_weights_2d_{condition}.svg")
plt.show()
