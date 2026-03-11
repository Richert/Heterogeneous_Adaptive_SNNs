import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

def delta_w(w: float, x: float, y: float, b: float, alpha: float, beta: float) -> np.ndarray:
    return b*(x*(1-w)**alpha - y*w**beta) + (1-b)*(x-y)*(1-w)*w

def get_qif_fr(x: float) -> float:
    return np.sqrt(x) / np.pi if x > 0 else 0.0

# parameter definition
condition = "antihebbian"
N = 1000
m = 4
bs = np.asarray([0.0, 0.01, 0.1, 1.0])
xys = [(0.0, 0.1), (0.1, 0.0)]
alpha = 1.0
beta = 1.0

# simulation parameters
w0 = np.linspace(0.0, 1.0, num=N)
res = {"b": [], "ltp/ltd": [], "dw/dt": [], "w": []}
for b in bs:
    for x, y in xys:
        for w in w0:
            res["b"].append(b)
            res["w"].append(w)
            res["ltp/ltd"].append("LTP" if x > y else "LTD")
            res["dw/dt"].append(delta_w(w, x, y, b, alpha, beta))
res = pd.DataFrame.from_dict(res)

# figure settings
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

# plotting
fig, ax = plt.subplots(figsize=(2, 1.5), layout="constrained")
for i, ltp in enumerate(["LTP", "LTD"]):
    sb.lineplot(res.loc[res.loc[:, "ltp/ltd"] == ltp], x="w", y="dw/dt", hue="b", palette="Dark2", ax=ax, legend=False)
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$dw/dt$")
ax.set_title("")
fig.canvas.draw()
plt.savefig(f"../results/figures/oja_weight_rule.svg")
plt.show()
