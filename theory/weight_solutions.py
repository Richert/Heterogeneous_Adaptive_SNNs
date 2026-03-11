import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import pickle

def first_derivative(w: float, x: np.ndarray, y: np.ndarray, b: float) -> np.ndarray:
    return b*(x*(1-w) - y*w) + (1-b)*(x-y)*(1-w)*w

def second_derivative(w: float, x: float, y: float, b: float) -> float:
    return -b*(x+y) + (1-b)*(x-y)*(1-2*w)

def get_xy(fr_source: float, fr_target: float, condition: str) -> tuple:
    if condition == "hebbian":
        x = fr_target*fr_source
        y = fr_target**2
    elif condition == "antihebbian":
        x = fr_source**2
        y = fr_target*fr_source
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def symmetric_solution(w0: float, x: float, y: float, eps: float = 1e-10) -> float:
    diff = y-x
    if np.abs(diff) < eps:
        return w0
    elif diff > 0.0:
        return 0.0
    else:
        return 1.0

def asymmetric_solution(w0: float, x: float, y: float, eps: float = 1e-10) -> float:
    if x < eps and y < eps:
        return w0
    else:
        return x / (x + y)

def full_solution(w0: float, x: float, y: float, b: float) -> float:
    if b == 0:
        return symmetric_solution(w0, x, y)
    elif b < 1.0 and x != y:
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
    else:
        return asymmetric_solution(w0, x, y)

def get_qif_fr(x: float) -> float:
    return np.sqrt(x) / np.pi if x > 0 else 0.0

# parameter definition
save_results = True
conditions = ["hebbian", "antihebbian"]
N = 100
eta_min, eta_max = -2.0, 3.0
eta_target = 0.5
etas_source = np.linspace(eta_min, eta_max, N)
bs = np.asarray([0.0, 0.01, 0.1, 1.0])
w0 = np.linspace(start=0.0, stop=1.0, num=N)

# get weight solutions
res = {"condition": [], "b": [], "eta": [], "w": [], "w0": []}
fr_t = get_qif_fr(eta_target)
for condition in conditions:
    for b in bs:
        for eta in etas_source:
            fr_s = get_qif_fr(eta)
            x, y = get_xy(fr_s, fr_t, condition)
            for w in w0:
                res["condition"].append(condition)
                res["b"].append(b)
                res["eta"].append(eta)
                res["w"].append(full_solution(w, x, y, b))
                res["w0"].append(w)
res = pd.DataFrame.from_dict(res)

if save_results:
    pickle.dump(res, open(f"../results/weight_solutions.pkl", "wb"))

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 2

# plotting
fig = plt.figure(constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
grid = fig.add_gridspec(nrows=2, ncols=3)
for i, condition in enumerate(conditions):

    # reduce data to condition
    res_tmp = res.loc[res.loc[:, "condition"] == condition, :]
    # subfig.suptitle("Hebbian Learning" if condition == "hebbian" else "Anti-Hebbian Learning")

    # solutions
    ax = fig.add_subplot(grid[i, 0])
    sb.lineplot(res_tmp, x="eta", y="w", hue="b", palette="Dark2", ax=ax, errorbar=("pi", 90),
                legend=False if i > 0 else "auto")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$w$")
    if i == 0:
        ax.get_legend().set_title(r"$b$")
        ax.set_title(r"Solutions for $w$")

fig.canvas.draw()
plt.savefig(f"../results/figures/weight_update_rule.svg")
plt.show()
