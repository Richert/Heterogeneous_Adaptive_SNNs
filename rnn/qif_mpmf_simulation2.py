import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

@njit
def vector_field(t, y, dy, Delta, J, eta, tau_s, tau_a, kappa):

    r = y[0:M]
    v = y[M:2*M]
    s = y[2*M:3*M]
    a = y[3*M:4*M]
    s_in = np.sum(s) * J / M
    PI = 4*np.atan(1.0)

    dy[0:M] = Delta / PI + 2.0*r*v
    dy[M:2*M] = v**2 + s_in*tau + eta - (PI*r)**2
    dy[2*M:3*M] = (a*r - s) / tau_s
    dy[3*M:4*M] = -kappa*a*r + (1 - a) / tau_a

    return dy

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

def uniform(N: int, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.asarray([-0.5 + (n-1)/(N-1) for n in range(1, N+1)])

@njit
def integrate_noise(x, inp, scale, tau):
    return x + scale * inp - x / tau

def generate_colored_noise(num_samples, tau, scale=1.0):
    """
    Generates Brownian noise by integrating white noise.

    Args:
        num_samples (int): The number of samples in the output Brownian noise.
        scale (float): A scaling factor for the noise amplitude.

    Returns:
        numpy.ndarray: An array containing the generated Brownian noise.
    """
    white_noise = np.random.randn(num_samples)
    x = 0.0
    colored_noise = np.zeros_like(white_noise)
    for sample in range(num_samples):
        x = integrate_noise(x, white_noise[sample], scale, tau)
        colored_noise[sample] = x
    return colored_noise

# parameters
path = "/home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs"
M = 10
p = 1.0
eta = 2.5
Delta = 2.0
etas = uniform(M, eta, Delta)
tau = 1.0
J = -15.0
Delta2 = Delta / (2*M)
tau_s = 0.5
tau_a = 20.0
kappa = 0.2
T = 1000.0
dt = 1e-4
dts = 1.0

# run simulation
y0 = np.zeros((4*M,))
y0[3*M:] = 1.0
dy = np.zeros_like(y0)
args = (dy, Delta2, J, etas, tau_s, tau_a, kappa)
f = lambda t, y: vector_field(t, y, *args)
res = solve_ivp(f, t_span=(0.0, T), y0=y0, t_eval=np.linspace(0.0, T, int(T/dts)), atol=1e-7, rtol=1e-4)
r = res.y[:M, :]
a = res.y[3*M:4*M, :]
time = res.t

# extract synaptic weights
# mapping, weights, etas_tmp = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
# if len(etas) == M:
#     etas = etas_tmp
# idx = np.argsort(etas)
# W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
# W = W[:, idx]
# clear(net)
# np.fill_diagonal(W, 0)

# fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
# ax = axes[1]
# ax.imshow(W, interpolation="none", aspect="auto")
# ax.set_title("Final Weights")
# ax = axes[0]
# ax.imshow(W0, interpolation="none", aspect="auto")
# ax.set_title("Initial Weights")
fig, axes = plt.subplots(nrows=2, figsize=(10, 6))
ax = axes[0]
ax.plot(time, r.T)
ax.set_title("firing rates")
ax = axes[1]
ax.plot(time, a.T)
ax.set_title("synaptic adaptation")
plt.show()
