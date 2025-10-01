import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def get_prob(x, bins: int = 100):
    counts, _ = np.histogram(x, bins=bins)
    return counts / np.sum(counts)

def second_derivative(w, x, y, b):
    return -b*(x+y)  + (1-b)*(x-y) - 2*w*(1-b)*(x-y)

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

def get_w_solution(w0: np.ndarray, x: np.ndarray, y: np.ndarray, b: float) -> np.ndarray:
    w = np.zeros_like(w0) + w0
    idx = []
    if b < 1.0:
        idx = np.argwhere(x != y).squeeze()
        a_term = 2*(b-1) * (x-y)
        b_term = x*(2*b-1) + y
        sqrt_term = np.sqrt((x-y)**2 + 4*x*y*b**2)
        for i in idx:
            w1 = (b_term[i] + sqrt_term[i]) / a_term[i]
            w2 = (b_term[i] - sqrt_term[i]) / a_term[i]
            ws = []
            for w_tmp in (w1, w2):
                sd = second_derivative(w_tmp, x[i], y[i], b)
                if 0 <= w_tmp <= 1 and (sd <= 0.0 or np.abs(w0[i] - w_tmp) < 1e-6):
                    ws.append(w_tmp)
            w[i] = np.random.choice(ws)
    idx2 = np.argwhere(x + y > 0.0).squeeze()
    for i in idx2:
        if i not in idx:
            w[i] = x[i] / (x[i] + y[i])
    return w

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
N = 1000
m = 100
eta = 1.0
J = 5.0
deltas = np.linspace(0.1, 1.5, num=m)
target_eta = 0.0
bs = [0.0, 0.05, 0.2]
res = {"b": bs, "w": {}, "C": {}, "H": {}, "V": {}}
n_reps = 4

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    ws, cs, hs, vs = [], [], [], []
    for Delta in deltas:

        # define source firing rate distribution
        inp = f(N, eta, Delta)
        fr_source = get_qif_fr(inp)

        # get weight solutions
        w = np.random.uniform(0.01, 0.99, size=(N,))
        for _ in range(n_reps):
            target_fr = get_qif_fr(target_eta + J * np.dot(w, fr_source) / N)
            x, y = get_xy(fr_source, np.zeros_like(fr_source) + target_fr, condition=condition)
            w = get_w_solution(w, x, y, b)
        print(f"Finished simulations for b = {b}, Delta = {Delta}")

        # calculate entropy of weight distribution
        h_w = entropy(get_prob(w))

        # calculate variance of weight distribution
        v = np.var(w)

        # calculate correlation between source etas and weights
        c = np.corrcoef(inp, w)[0, 1]

        # save results
        ws.append(w)
        cs.append(c)
        hs.append(h_w)
        vs.append(v)

    res["w"][b] = np.asarray(ws)
    res["H"][b] = np.asarray(hs)
    res["C"][b] = np.asarray(cs)
    res["V"][b] = np.asarray(vs)

# plotting
fig, axes = plt.subplots(nrows=3, ncols=len(bs), figsize=(3*len(bs), 5), layout="constrained")
ticks = np.arange(0, m, int(m/5))
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

fig.suptitle(f"{'Hebbian' if condition == 'hebbian' else 'Anti-Hebbian'} Learning (J = {int(J)}, Theory)")
fig.canvas.draw()
plt.savefig(f"../results/ss_weight_solutions_{condition}_{int(J)}.svg")
plt.show()
