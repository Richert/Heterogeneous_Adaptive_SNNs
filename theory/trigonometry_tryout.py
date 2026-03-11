import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2*np.pi, 1000)
f1 = 2.0
a = np.sin(f1*t)**4
b = 3/8 - 1/2 * np.cos(2*f1*t) + 1/8 * np.cos(4*f1*t)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, a, label='sin^2')
ax.plot(t, b, label='cos')
ax.legend()
plt.show()