import numpy as np
import matplotlib.pyplot as plt

def second_derivative(w, x, y, b):
    return -b*(x+y)  + (1-b)*(x-y) - 2*w*(1-b)*(x-y)

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

def get_w_solution(x: float, y: float, b: float) -> float:
    if b < 1.0 and x != y:
        a_term = 2*(b-1) * (x-y)
        b_term = x*(2*b-1) + y
        sqrt_term = np.sqrt((x-y)**2 + 4*x*y*b**2)
        w1 = (b_term + sqrt_term) / a_term
        w2 = (b_term - sqrt_term) / a_term
        ws = []
        for w in (w1, w2):
            sd = second_derivative(w, x, y, b)
            if 0 <= w <= 1 and sd <= 0.0:
                ws.append(w)
        return np.random.choice(ws)
    elif x + y > 0.0:
        return x / (x + y)
    else:
        return np.random.uniform(low=0.0, high=1.0)

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / (2*np.pi)

# parameter definition
condition = "hebbian"
distribution = "lorentzian"
N = 10000
m = 100
eta = 1.0
deltas = np.linspace(0.1, 2.0, num=m)
target_fr = 0.2
bs = [0.0, 0.125, 0.25, 0.5, 1.0]
res = {"b": bs, "delta": deltas, "data": {}}

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
            w = get_w_solution(x, y, b)
            ws.append(w)

        # save results
        data["w"].append(ws)
        data["fr"].append(fr_source)

    res["data"][b] = data

# plotting
fig, axes = plt.subplots(ncols=len(bs), figsize=(12, 4))
for i, b in enumerate(bs):

    # weight distribution
    ax = axes[i]
    im = ax.imshow(np.asarray(res["data"][b]["w"]), aspect="auto", interpolation="none", cmap="cividis",
                   vmax=1.0, vmin=0.0)
    ax.set_xlabel("eta")
    ax.set_ylabel("Delta")
    ax.set_title(f"Weights (b = {b})")

    # # firing rate distribution
    # ax = axes[1, i]
    # im = ax.imshow(np.asarray(res["data"][b]["fr"]), aspect="auto", interpolation="none", cmap="cividis", vmax=fr_max)
    # plt.colorbar(im, ax=ax)
    # ax.set_xlabel("eta")
    # ax.set_ylabel("Delta")
    # ax.set_title(f"Firing Rates (b = {b})")

fig.suptitle("Theory")
plt.tight_layout()
plt.show()
