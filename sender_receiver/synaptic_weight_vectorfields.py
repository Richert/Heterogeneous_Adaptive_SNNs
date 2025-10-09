import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

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

def delta_w(w: float, x: np.ndarray, y: np.ndarray, b: float, alpha: float, beta: float) -> np.ndarray:
    return b*(x*(1-w)**alpha - y*w**beta) + (1-b)*(x-y)*(1-w)*w

def get_qif_fr(x: float) -> float:
    return np.sqrt(x) / np.pi if x > 0 else 0.0

# parameter definition
condition = "antihebbian"
N = 1000
m = 10
eta_min, eta_max = -0.5, 1.5
fr_target = get_qif_fr(0.5)
fr_source = np.asarray([get_qif_fr(eta) for eta in np.linspace(eta_min, eta_max, m)])
bs = np.asarray([0.1])
alpha = 1.0
beta = 1.0

# simulation parameters
w0 = np.linspace(0.0, 1.0, num=N)
res = {"b": [], "x-y": [], "dw/dt": [], "w": []}
for b in bs:
    for fr_s in fr_source:
        for w in w0:
            x, y = get_xy(fr_s, fr_target, condition)
            res["b"].append(b)
            res["w"].append(w)
            res["x-y"].append(x-y)
            res["dw/dt"].append(delta_w(w, x, y, b, alpha, beta))
res = pd.DataFrame.from_dict(res)

# plotting
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
sb.lineplot(res, x="w", y="dw/dt", hue="x-y", palette="Dark2")
fig.canvas.draw()
plt.show()
