import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

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

def delta_w(t: float, w: np.ndarray, eta_s: np.ndarray, eta_t: float, J: float, b: float, a: float, condition: str,
            t_old: np.ndarray) -> np.ndarray:
    dt = t - t_old[0]
    t_old[0] = t
    r_s = get_qif_fr(eta_s + noise * np.random.randn(len(eta_s)) * np.sqrt(dt))
    r_t = get_qif_fr(eta_t + noise * np.random.randn() * np.sqrt(dt) + J*np.dot(w, r_s) / N)
    x, y = get_xy(r_s, r_t, condition=condition)
    return a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))

def get_w_solution(w0: np.ndarray, eta_source: np.ndarray, eta: float, J: float, b: float, a: float, T: float, **kwargs
                   ) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(t, w, eta_source, eta, J, b, a, condition, np.zeros(1,)), t_span=(0.0, T), y0=np.asarray(w0),
                     **kwargs)
    return sols.y[:, -1]

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# parameter definition
save_results = True
conditions = ["hebbian", "antihebbian"]
noise_lvls = [0.0, 0.003, 0.006]
bs = [0.0, 0.01, 0.1, 1.0]
J = 5.0
N = 1000
m = 10
eta_t = 0.0 if J > 0 else 2.0
eta_min, eta_max = -2.0, 3.0
eta_s = np.linspace(eta_min, eta_max, N)
w0s = np.linspace(start=0.0, stop=1.0, num=m)
a = 0.01
res = {"condition": [], "b": [], "noise": [], "eta": [], "w": [], "w0": []}

# simulation
T = 18000.0
solver_kwargs = {"method": "RK23", "t_eval": [0.0, T], "atol": 1e-5}

for condition in conditions:
    for b in bs:
        for noise in noise_lvls:
            for w0 in w0s:

                # get weight solutions
                ws = get_w_solution(np.zeros_like(eta_s) + w0, eta_s, eta_t, J, b, a, T, **solver_kwargs)
                ws[ws < 0.0] = 0.0
                ws[ws > 1.0] = 1.0

                # save results
                for eta, w in zip(eta_s, ws):
                    res["condition"].append(condition)
                    res["b"].append(b)
                    res["noise"].append(noise)
                    res["w0"].append(w0)
                    res["w"].append(w)
                    res["eta"].append(eta)

            # save results
            print(f"Finished simulations for c = {condition}, b = {b}, and noise = {noise}.")

# save results
conn = int(J)
if save_results:
    pickle.dump(res, open(f"../results/rate_simulation_J{conn}.pkl", "wb"))

# plotting
res = pd.DataFrame.from_dict(res)
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 2

fig, axes = plt.subplots(ncols=2, nrows=len(noise_lvls), figsize=(6, 1.5*len(noise_lvls)))
for j, c in enumerate(conditions):
    res_tmp = res.loc[res.loc[:, "condition"] == c, :]
    for i, noise in enumerate(noise_lvls):
        res_tmp2 = res_tmp.loc[res_tmp.loc[:, "noise"] == noise, :]
        ax = axes[i, j]
        sb.lineplot(res_tmp2, x="eta", y="w", hue="b", palette="Dark2", ax=ax, errorbar=("pi", 90), legend=False)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$w$")
        # ax.get_legend().set_title(r"$b$")
        c_title = "Hebbian Learning" if c == "hebbian" else "Anti-Hebbian Learning"
        ax.set_title(f"{c}, noise lvl = {noise}")

fig.suptitle("Oja's rule, rate prediction")
fig.canvas.draw()
# plt.savefig(f"../results/figures/weight_update_rule.svg")
plt.show()