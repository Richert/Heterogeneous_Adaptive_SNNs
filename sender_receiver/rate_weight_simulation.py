import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import entropy
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

def delta_w(w: np.ndarray, eta_s: np.ndarray, eta_t: float, J: float, b: float, a: float, condition: str) -> np.ndarray:
    r_s = get_qif_fr(eta_s + noise * np.random.randn(len(eta_s)))
    r_t = get_qif_fr(eta_t + noise * np.random.randn() + J*np.dot(w, r_s) / N)
    x, y = get_xy(r_s, r_t, condition=condition)
    return a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))

def get_w_solution(w0: np.ndarray, eta_source: np.ndarray, eta: float, J: float, b: float, a: float, T: float, **kwargs
                   ) -> np.ndarray:
    sols = solve_ivp(lambda t, w: delta_w(w, eta_source, eta, J, b, a, condition), t_span=(0.0, T), y0=np.asarray(w0),
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
save_results = False
condition = "hebbian"
distribution = "gaussian"
noise_lvls = [0.0, 0.5, 1.0]
J = 5.0
N = 10000
m = 100
eta = 1.0
deltas = np.linspace(0.1, 1.5, num=m)
target_eta = 0.0 if J > 0 else 2.0
a = 0.1
bs = [0.0, 0.05, 0.2]
res = {"b": [], "w": [], "C": [], "H":[], "V": [], "delta": [], "noise": []}

# simulation parameters
T = 2000.0
solver_kwargs = {}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    for noise in noise_lvls:
        for Delta in deltas:

            diff = 1.0
            attempt = 0
            while diff > 0.2 and attempt < 20:

                # define initial condition
                eta_source = f(N, eta, Delta)
                w0 = np.random.uniform(0.01, 0.99, size=(N,))

                # get weight solutions
                w = get_w_solution(w0, eta_source, target_eta, J, b, a, T, **solver_kwargs)
                w[w < 0.0] = 0.0
                w[w > 1.0] = 1.0

                # calculate entropy of weight distribution
                h_w = entropy(get_prob(w))

                # calculate variance of weight distribution
                v = np.var(w)

                # calculate correlation between source etas and weights
                c = np.corrcoef(eta_source, w)[1, 0]

                try:
                    diff = np.abs(c - res["C"][-1]) + np.abs(h_w - res["H"][-1])
                except IndexError:
                    diff = 0.0
                attempt += 1

            # save results
            res["b"].append(b)
            res["delta"].append(Delta)
            res["noise"].append(noise)
            res["w"].append(w)
            res["C"].append(c)
            res["H"].append(h_w)
            res["V"].append(v)

            # save results
            print(f"Finished simulations for b = {b}, noise = {noise} and Delta = {np.round(Delta, decimals=2)} after {attempt} attempts")

# save results
conn = int(J)
if save_results:
    pickle.dump({"condition": condition, "J": J, "results": res},
                open(f"../results/rate_weight_simulations_{condition}_{conn}.pkl", "wb"))
