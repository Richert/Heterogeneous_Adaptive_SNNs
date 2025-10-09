import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

def second_derivative(w: float, x: float, y: float, b: float) -> float:
    return -b*(x+y)  + (1-b)*(x-y) - 2*w*(1-b)*(x-y)

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
conditions = ["hebbian", "antihebbian"]
N = 1000
eta_min, eta_max = -0.5, 3.0
eta_target = 0.5
etas_source = np.linspace(eta_min, eta_max, N)
bs = np.asarray([0.0, 0.01, 0.1, 1.0])
w0 = np.linspace(start=0.0, stop=1.0, num=10)

# get weight solutions
weights = {"condition": [], "b": [], "w": [], "eta": []}
fr_t = get_qif_fr(eta_target)
for condition in conditions:
    for b in bs:
        for eta in etas_source:
            fr_s = get_qif_fr(eta)
            x, y = get_xy(fr_s, fr_t, condition)
            weights["condition"].append(condition)
            weights["b"].append(b)
            weights["eta"].append(eta)
            weights["w"].append(full_solution(np.random.choice(w0), x, y, b))
weights = pd.DataFrame.from_dict(weights)

# plotting
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")
for i, condition in enumerate(conditions):
    ax = axes[i]
    weights_tmp = weights.loc[weights.loc[:, "condition"] == condition, :]
    sb.lineplot(weights_tmp, x="eta", y="w", hue="b", palette="Dark2", ax=ax)
    ax.set_title("Hebbian Learning" if condition == "hebbian" else "Anti-Hebbian Learning")
fig.canvas.draw()
plt.show()
