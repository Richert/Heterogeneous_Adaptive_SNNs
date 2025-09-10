import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def delta_w(w: float, x: np.ndarray, y: np.ndarray, b: float) -> np.ndarray:
    return b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2)

def get_w_vectorfield(x: np.ndarray, y:np.ndarray, w: float, b: float) -> np.ndarray:
    deltas = delta_w(w, x, y, b)
    return deltas

# parameter definition
condition = "hebbian"
N = 1000
m = 1000
bs = np.asarray([0.0, 0.125, 0.25, 0.5, 1.0])
ws = np.linspace(0.0, 1.0, num=N)
xs = np.linspace(0.0, 1.0, num=m)
res = {"b": bs, "delta_w": {}, "w": ws, "x": xs}

# simulation parameters
x0 = np.zeros((m,))
for b in bs:
    data = {"ltp": [], "ltd": []}
    for w in ws:

        # get vectorfield values
        data["ltp"].append(get_w_vectorfield(xs, x0, w, b))
        data["ltd"].append(get_w_vectorfield(x0, xs, w, b))

    res["delta_w"][b] = data

# plotting
fig, axes = plt.subplots(ncols=len(bs), figsize=(12, 3))
ticks = np.arange(0, 2*N+1, step=int(N/2), dtype=np.int16)
ticks_half = ticks[:2]
xlabels = np.round(xs[ticks_half], decimals=1)
xlabels = xlabels.tolist() + [1.0] + (-xlabels[::-1]).tolist()
for i, b in enumerate(bs):

    # concatenate matrices
    VF = np.concatenate([np.asarray(res["delta_w"][b]["ltp"]), np.asarray(res["delta_w"][b]["ltd"])[:, ::-1]], axis=1)

    # LTP/LTD distribution
    ax = axes[i]
    im = ax.imshow(VF, aspect="auto", interpolation="none", cmap="berlin")
    ax.set_xlabel("x")
    ax.set_ylabel("w")
    ax.set_title(f"b = {b}")
    ax.set_xticks(ticks, labels=xlabels)
    ax.set_yticks(ticks_half, labels=np.round(ws[ticks_half], decimals=1))

fig.suptitle("LTP / LTD profiles")
plt.tight_layout()
fig.canvas.draw()
plt.savefig("../results/synaptic_weight_vf.svg")
plt.show()
