import numpy as np
import matplotlib.pyplot as plt

def g(x, a1, a2, a3):
    return a1*np.cos(x)**1 + a2*np.sin(x)**1 + a3*np.abs(np.cos(x))

N = 1000
a1_vals = [-0.5, 0.0, 0.5]
a2_vals = [-0.5, 0.0, 0.5]
phases = np.linspace(-np.pi, np.pi, N)
phase_differences = []
for a1 in a1_vals:
    for a2 in a2_vals:
        phase_differences.append(g(phases, a1, a2, a3=0.5))

fig, ax = plt.subplots(figsize=(16,6 ))
for pd in phase_differences:
    ax.plot(phases, pd, "o-")
ax.set_xlabel("phase diff")
ax.set_ylabel("G(phase diff)")
plt.tight_layout()
plt.show()