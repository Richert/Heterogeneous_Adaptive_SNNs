import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = np.outer(fr_target, fr_source)
        y = np.repeat((fr_target**2).reshape(len(fr_target), 1), len(fr_source), axis=1)
    elif condition == "antihebbian":
        x = np.repeat((fr_source**2).reshape(len(fr_source), 1), len(fr_target), axis=1).T
        y = np.outer(fr_target, fr_source)
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def extract_weights(w: np.ndarray) -> tuple:

    n_ee = int(N_e * N_e)
    n_ei = int(N_e * N_i)

    w_ee = np.reshape(w[:n_ee], (N_e, N_e))
    w_ei = np.reshape(w[n_ee:n_ee + n_ei], (N_e, N_i))
    w_ie = np.reshape(w[n_ee + n_ei:n_ee + 2 * n_ei], (N_i, N_e))
    w_ii = np.reshape(w[n_ee + 2 * n_ei:], (N_i, N_i))

    return w_ee, w_ei, w_ie, w_ii

def delta_w(w: np.ndarray, r_source: np.ndarray, r_target: np.ndarray, b: float, a: float) -> np.ndarray:
    x, y = get_xy(r_source, r_target, condition=condition)
    return (a*(b*((1-w)*x - w*y) + (1-b)*(x-y)*(w-w**2))).flatten()

def eic_update(w: np.ndarray, etas_e: np.ndarray, etas_i: np.ndarray, r_e: np.ndarray, r_i: np.ndarray,
               J_ee: float, J_ei: float, J_ie: float, J_ii: float, b: float, a: float
               ) -> np.ndarray:

    w_ee, w_ei, w_ie, w_ii = extract_weights(w)

    r_e[:] = get_qif_fr(etas_e + J_ee*np.dot(w_ee, r_e) - J_ei*np.dot(w_ei, r_i))
    r_i[:] = get_qif_fr(etas_i + J_ie*np.dot(w_ie, r_e) - J_ii*np.dot(w_ii, r_i))

    dw_ee = delta_w(w_ee, r_e, r_e, b, a)
    dw_ei = delta_w(w_ei, r_i, r_e, b, a)
    dw_ie = delta_w(w_ie, r_e, r_i, b, a)
    dw_ii = delta_w(w_ii, r_i, r_i, b, a)

    return np.concatenate([dw_ee, dw_ei, dw_ie, dw_ii], axis=0)

def get_w_solution(w0: np.ndarray, etas_e: np.ndarray, etas_i: np.ndarray,
                   J_ee: float, J_ei: float, J_ie: float, J_ii: float,
                   b: float, a: float, T: float, **kwargs) -> tuple:

    r_e = get_qif_fr(etas_e)
    r_i = get_qif_fr(etas_i)

    sols = solve_ivp(lambda t, w: eic_update(w, etas_e, etas_i, r_e, r_i, J_ee, J_ei, J_ie, J_ii, b, a),
                     t_span=(0.0, T), y0=np.asarray(w0), **kwargs)

    return extract_weights(sols.y[:, -1])

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
N_e = 200
N_i = 50
Deltas_e = [0.1, 1.0]
Delta_i = 1.0
eta_e, eta_i = 1.0, 0.0
a = 0.1
J_ee = 5.0
J_ei = 5.0
J_ie = 5.0
J_ii = 5.0
bs = [0.01]
res = {"b": [], "w_ee": [], "w_ei": [], "w_ie": [], "w_ii": [], "delta": [], "eta_e": [], "eta_i": []}

# simulation parameters
T = 200.0
solver_kwargs = {"t_eval": [T]}

f = lorentzian if distribution == "lorentzian" else gaussian
for b in bs:
    for Delta in Deltas_e:

        # define initial condition
        etas_e = f(N_e, eta_e, Delta)
        etas_i = f(N_i, eta_i, Delta_i)
        fr_e = get_qif_fr(etas_e)
        fr_i = get_qif_fr(etas_i)
        w0 = np.random.uniform(0.01, 0.99, size=(int((N_e + N_i)**2),))

        # get weight solutions
        w_ee, w_ei, w_ie, w_ii = get_w_solution(w0, etas_e, etas_i, J_ee, J_ei, J_ie, J_ii, b, a, T, **solver_kwargs)

        # save results
        res["b"].append(b)
        res["delta"].append(Delta)
        res["w_ee"].append(w_ee)
        res["w_ei"].append(w_ei)
        res["w_ie"].append(w_ie)
        res["w_ii"].append(w_ii)
        res["eta_e"].append(etas_e)
        res["eta_i"].append(etas_i)
        print(f"Finished simulations for b = {b}, Delta = {Delta}")

# plotting
fig, axes = plt.subplots(nrows=len(Deltas_e), ncols=len(bs), figsize=(3 * len(bs), 3 * len(Deltas_e)),
                         layout="constrained")
ticks = np.arange(0, N_e, int(N_e/5))
for j, b in enumerate(bs):
    for i, Delta in enumerate(Deltas_e):

        # weight distribution
        ax = axes[i, j]
        idx1 = np.asarray(res["b"]) == b
        idx2 = np.asarray(res["delta"]) == Delta
        idx = np.argwhere((idx1 * idx2) > 0.5).squeeze()
        im = ax.imshow(np.asarray(res["w_ee"][idx]), aspect="auto", interpolation="none", cmap="viridis",
                       vmax=1.0, vmin=0.0)
        ax.set_xlabel("source neuron eta")
        ax.set_ylabel("target neuron eta")
        ax.set_yticks(ticks, labels=np.round(res["eta_e"][idx][ticks], decimals=1))
        ax.set_xticks(ticks, labels=np.round(res["eta_e"][idx][ticks], decimals=1))
        ax.set_title(f"W_ee (b = {b}, Delta = {Delta})")
        if j == len(bs) - 1:
            plt.colorbar(im, ax=ax, shrink=0.8)

        # # firing rate distribution
        # ax = axes[1, i]
        # im = ax.imshow(np.asarray(res["data"][b]["fr"]), aspect="auto", interpolation="none", cmap="cividis", vmax=fr_max)
        # plt.colorbar(im, ax=ax)
        # ax.set_xlabel("eta")
        # ax.set_ylabel("Delta")
        # ax.set_title(f"Firing Rates (b = {b})")

fig.suptitle(f"{'Hebbian' if condition == 'hebbian' else 'Anti-Hebbian'} Learning (J = {int(J)}, Rate Simulation)")
fig.canvas.draw()
conn = int(J_ee)
conn = f"{conn}_inh" if conn < 0 else f"{conn}"
plt.savefig(f"../results/rate_weight_simulation_eic_{condition}_{conn}.svg")
plt.show()
