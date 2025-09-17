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
J = 8.0
deltas = np.linspace(0.1, 3.0, num=m)
target_eta = 0.2
bs = [0.0, 0.125]
res = {"b": bs, "w": {}}
n_reps = 10

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    ws = []
    for Delta in deltas:

        # define source firing rate distribution
        inp = f(N, eta, Delta)
        fr_source = get_qif_fr(inp)

        # get weight solutions
        w = np.random.uniform(0.01, 0.99, size=(N,))
        for _ in range(n_reps):
            target_fr = get_qif_fr(target_eta + J * np.dot(w, fr_source) / N)
            for i, source_fr in enumerate(fr_source):
                x, y = get_xy(source_fr, target_fr, condition=condition)
                w[i] = get_w_solution(w[i], x, y, b)

        ws.append(w)
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

fig.suptitle(f"{'Hebbian' if condition == 'hebbian' else 'Anti-Hebbian'} Learning (J = {int(J)}, Theory)")
fig.canvas.draw()
plt.savefig(f"../results/ss_weight_solutions_{condition}_{int(J)}.svg")
plt.show()
