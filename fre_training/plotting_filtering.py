import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from pandas import DataFrame
import os

# preparations
##############

# load data
path = "/home/richard-gast/Documents/results/filter_training"
results = {"model": [], "noise": [], "frequency": [], "trial": [], "loss": [],
           "Delta": [], "J": [], "eta": []}
for file in os.listdir(path):
    if "filtering_" in file:
        data = pickle.load(open(f"{path}/{file}", "rb"))
        if np.isfinite(data["test_loss"]):
            results["model"].append(file.split("_")[1])
            results["noise"].append(data["noise"])
            results["frequency"].append(data["frequency"])
            results["trial"].append(data["trial"])
            results["loss"].append(data["test_loss"])
            results["Delta"].append(np.abs(data["Delta"][0]))
            results["eta"].append(data["eta"][0])
            results["J"].append(data["J"][0])
df = DataFrame.from_dict(results)

# plot settings
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
##########

# figure layout
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=4, nrows=2)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0., wspace=0.)

# plot heatmaps for fitted parameters (and loss)
for i, dv in enumerate(["loss", "eta", "Delta", "J"]):
    ax = fig.add_subplot(grid[0, i])
    pv = df.pivot_table(index="noise", columns="frequency", values=dv)
    sb.heatmap(pv, ax=ax)
    ax.set_title(dv)

# plot scatter plots for fitted parameters
for i, (x, y) in enumerate([("eta", "Delta"), ("eta", "J"), ("Delta", "J")]):
    ax = fig.add_subplot(grid[1, i])
    sb.scatterplot(df, x=x, y=y, ax=ax)
ax = fig.add_subplot(grid[1, 3])

# finish plot
fig.suptitle("Filtering Results")
fig.canvas.draw()
plt.show()
