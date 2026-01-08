import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_prob(x, bins: int = 100):
    counts, _ = np.histogram(x, bins=bins)
    return counts / np.sum(counts)

def delta_w(t: float, w: np.ndarray, eta_s: np.ndarray, eta_t: float, J: float, a_ltp: float, a_ltd: float,
            gamma_ltp: float, gamma_ltd: float, theta_ltp: float, theta_ltd: float, tau_w: float, t_old: np.ndarray
            ) -> np.ndarray:
    dt = t - t_old[0]
    t_old[0] = t
    r_s = get_qif_fr(eta_s + noise * np.random.randn(len(eta_s)) * np.sqrt(dt))
    r_t = get_qif_fr(eta_t + noise * np.random.randn() * np.sqrt(dt) + J*np.dot(w, r_s) / N)
    c = r_s + r_t
    return (a_ltp*(1-w)*sigmoid(gamma_ltp*(c-theta_ltp)) - a_ltd*w*sigmoid(gamma_ltd*(c-theta_ltd)) - w*(1-w)*(0.5-w)) / tau_w

def get_w_solution(w0: np.ndarray, eta_source: np.ndarray, eta: float, J: float, a_ltp: float, a_ltd: float,
                   gamma_ltp: float, gamma_ltd: float, theta_ltp: float, theta_ltd: float, tau_w: float, T: float,
                   **kwargs) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(t, w, eta_source, eta, J, a_ltp, a_ltd, gamma_ltp, gamma_ltd, theta_ltp,
                                          theta_ltd, tau_w, np.zeros(1,)), t_span=(0.0, T), y0=np.asarray(w0), **kwargs)
    return sols.y[:, -1]

def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# parameter definition
save_results = True
noise_lvls = [0.0, 0.003, 0.006]
J = 5.0
N = 1000
m = 10
eta_t = 0.0 if J > 0 else 2.0
eta_min, eta_max = -2.0, 3.0
eta_s = np.linspace(eta_min, eta_max, N)
w0s = np.linspace(start=0.0, stop=1.0, num=m)
a_ltp = 50.0
a_ltd = 50.0
gamma_ltp = 50.0
gamma_ltd = 20.0
theta_ltp = 0.8
theta_ltd = 1.0
tau_w = 100.0
res = {"noise": [], "eta": [], "w": [], "w0": []}

# simulation
T = 2000.0
solver_kwargs = {"method": "RK23", "t_eval": [0.0, T], "atol": 1e-5}
for noise in noise_lvls:
    for w0 in w0s:

        # get weight solutions
        ws = get_w_solution(np.zeros_like(eta_s) + w0, eta_s, eta_t, J, a_ltp, a_ltd, gamma_ltp, gamma_ltd,
                            theta_ltp, theta_ltd, tau_w, T, **solver_kwargs)
        ws[ws < 0.0] = 0.0
        ws[ws > 1.0] = 1.0

        # save results
        for eta, w in zip(eta_s, ws):
            res["noise"].append(noise)
            res["w0"].append(w0)
            res["w"].append(w)
            res["eta"].append(eta)

    # save results
    print(f"Finished simulations for noise = {noise}.")

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

fig, ax = plt.subplots(figsize=(6, 3))
sb.lineplot(res, x="eta", y="w", hue="noise", palette="Dark2", ax=ax, errorbar=("pi", 90), legend=False)
ax.set_xlabel(r"$\eta$")
ax.set_ylabel(r"$w$")

fig.suptitle("Brunel's rule, rate prediction")
fig.canvas.draw()
# plt.savefig(f"../results/figures/weight_update_rule.svg")
plt.show()