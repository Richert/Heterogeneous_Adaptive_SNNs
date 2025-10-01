import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import entropy

def get_prob(x, bins: int = 100):
    counts, _ = np.histogram(x, bins=bins)
    return counts / np.sum(counts)

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = fr_target*fr_source
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
condition = "hebbian"
distribution = "gaussian"
N = 10000
m = 100
eta = 1.0
deltas = np.linspace(0.1, 1.5, num=m)
target_eta = 2.0
a = 0.1
J = -5.0
bs = [0.0, 0.05, 0.2]
res = {"b": bs, "w": {}, "C": {}, "H": {}, "V": {}}

# simulation parameters
T = 2000.0
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    ws, cs, hs, vs = [], [], [], []
    for Delta in deltas:

        c = 0.0
        attempt = 0
        while c < 0.01 and attempt < 10:

            # define initial condition
            inp = f(N, eta, Delta)
            fr_source = get_qif_fr(inp)
            w0 = np.random.uniform(0.01, 0.99, size=(N,))

            # get weight solutions
            w = get_w_solution(w0, fr_source, target_eta, J, b, a, T, **solver_kwargs)
            w[w < 0.0] = 0.0
            w[w > 1.0] = 1.0

            # calculate entropy of weight distribution
            h_w = entropy(get_prob(w))

            # calculate variance of weight distribution
            v = np.var(w)

            # calculate correlation between source etas and weights
            c = np.corrcoef(inp, w)[1, 0]

        # save results
        print(f"Finished simulations for b = {b} and Delta = {np.round(Delta, decimals=1)}")
        ws.append(w)
        cs.append(c)
        hs.append(h_w)
        vs.append(v)

        # save results
    res["w"][b] = np.asarray(ws)
    res["H"][b] = np.asarray(hs)
    res["C"][b] = np.asarray(cs)
    res["V"][b] = np.asarray(vs)

# plotting
fig, axes = plt.subplots(nrows=3, ncols=len(bs), figsize=(3 * len(bs), 5), layout="constrained")
ticks = np.arange(0, m, int(m / 5))
for i, b in enumerate(bs):

    # weight distribution
    ax = axes[0, i]
    im = ax.imshow(np.asarray(res["w"][b]).T, aspect="auto", interpolation="none", cmap="viridis", vmax=1.0, vmin=0.0)
    ax.set_ylabel("neuron")
    ax.set_xlabel("Delta")
    ax.set_xticks(ticks, labels=np.round(deltas[ticks], decimals=1))
    if i == len(bs) - 1:
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"w (b = {b})")

    # mutual information
    ax = axes[1, i]
    ax.plot(deltas, res["C"][b])
    ax.set_xlabel("Delta")
    ax.set_ylabel("C")
    ax.set_title("correlation(w, eta)")

    # entropy
    ax = axes[2, i]
    ax.plot(deltas, res["H"][b])
    ax.set_xlabel("Delta")
    ax.set_ylabel("H")
    ax.set_title("entropy(w)")

    # # variance
    # ax = axes[3, i]
    # ax.plot(deltas, res["V"][b])
    # ax.set_xlabel("Delta")
    # ax.set_ylabel("var")
    # ax.set_title("variance(w)")

fig.suptitle(f"{'Anti-Hebbian' if 'antihebbian' in condition else 'Hebbian'} Learning (J = {int(J)}, Rate Simulation)")
fig.canvas.draw()
plt.savefig(f"../results/ss_weight_simulation_{condition}_{int(J)}.svg")
plt.show()
