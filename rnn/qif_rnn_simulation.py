import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('tkagg')
np.random.seed(42)
from numba import njit, config
config.THREADING_LAYER = "tbb"

@njit(parallel=True)
def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, p_source: np.ndarray, p_target: np.ndarray,
           d_source: np.ndarray, d_target: np.ndarray) -> tuple:
    if condition == "hebbian":
        x = np.outer(fr_target, p_source)
        y = np.repeat(fr_target*d_target, N).reshape(N, N)
    elif condition == "antihebbian":
        x = np.repeat(fr_source*p_source, N).reshape(N, N).T
        y = np.outer(fr_target, d_source)
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

@njit
def qif_rhs(y: np.ndarray, w: np.ndarray, spikes: np.ndarray, eta: np.ndarray, inp: float, J: float, tau_s: float,
            tau_p: float, tau_d: float, a_p: float, a_d: float, b: float) -> tuple:
    v, s, u_p, u_d = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:4*N]
    dy = np.zeros_like(y)
    x, y = get_xy(s, s, u_p, u_p, u_d, u_d)
    dy[:N] = v**2 + eta + J*np.dot(w, s) + inp
    dy[N:2*N] = spikes - s/tau_s
    dy[2*N:3*N] = spikes - u_p/tau_p
    dy[3*N:4*N] = spikes - u_d/tau_d
    dw = b*(a_p*(1-w)*x - a_d*w*y) + (1-b)*(a_p*x-a_d*y)*(w-w**2)
    return dy, dw

@njit
def spiking(y: np.ndarray, spikes: np.ndarray):
    idx = np.argwhere((v_cutoff - y[:N]) < 0.0).flatten()
    spikes[:] = 0.0
    y[idx] = -y[idx]
    spikes[idx] = 1.0/dt

def solve_ivp(dt: float, w: np.ndarray, eta: np.ndarray, inp: np.ndarray, J: float, tau_s: float, tau_p: float, tau_d: float,
              a_p: float, a_d: float, b: float, sr: int = 100) -> tuple:

    y = np.zeros((4*N,))
    spikes = np.zeros((N,))

    s_col = np.zeros((int(inp.shape[0]/sr), N))
    ss = 0
    for step in range(inp.shape[0]):
        spiking(y, spikes)
        dy, dw = qif_rhs(y, w, spikes, eta, inp[step], J, tau_s, tau_p, tau_d, a_p, a_d, b)
        y = y + dt * dy
        w = w + dt * dw
        if step % sr == 0:
            s_col[ss, :] = y[N:2*N]
            ss += 1

    return s_col, w

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def uniform(N: int, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.linspace(-0.5, 0.5, N)

# parameters
path = "/home/richard-gast/PycharmProjects/Heterogeneous_Adaptive_SNNs"
condition = "hebbian"

# define stdp parameters
a = 0.01
a_r = 1.5
tau = 2.0
tau_r = 2.0
a_p = a*a_r
a_d = a/a_r
tau_p = tau
tau_d = tau*tau_r
tau_ratio = tau_p / tau_d
a_ratio = a_p / a_d
stdp_ratio = tau_ratio*a_ratio

# set model parameters
N = 200
M = 20
J = -10.0 / (0.5*N)
Delta = 1.0
eta = 0.0
b = 0.5
tau_s = 1.0
v_cutoff = 100.0
etas_mp = uniform(M, eta, Delta)
etas = []
for eta_mp in etas_mp:
    etas.extend(lorentzian(int(N/M), eta_mp, Delta/(2*M)).tolist())
etas = np.asarray(etas)
edge_vars = {"a_p": a_p, "a_d": a_d, "b": b}
node_vars = {"J": J, "eta": etas, "tau_p": tau_p, "tau_d": tau_d, "tau_s": tau_s}

# simulation parameters
cutoff = 100.0
T = 2000.0
dt = 5e-3
dts = 1.0
inp_amp = 3.0
inp_freq = 0.005
inp_dur = 5.0

# generate intrinsic input
steps = int(T/dt)
inp = np.zeros((steps,))
period = int(1.0/(inp_freq*dt))
dur = int(inp_dur/dt)
step = 0
while step < steps:
    inp[step:step+dur] += inp_amp
    step += period
# plt.plot(inp)
# plt.show()

# run simulation
w0 = np.random.uniform(0.0, 1.0, size=(N, N,))
s, w = solve_ivp(dt, w0, node_vars["eta"], inp, node_vars["J"], node_vars["tau_s"], node_vars["tau_p"], node_vars["tau_d"],
                 edge_vars["a_p"], edge_vars["a_d"], edge_vars["b"])

# plotting
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 2

fig, axes = plt.subplots(figsize=(8, 3), ncols=2)
for i, (W, title) in enumerate(zip([w0, w], ["initial weights", "final weights"])):
    ax = axes[i]
    im = ax.imshow(W, aspect="auto", interpolation="none", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
fig.suptitle("connectivity")
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.mean(s, axis=1))
ax.set_title("dynamics")
ax.set_ylabel("s")
ax.set_xlabel("time")
plt.tight_layout()
plt.show()

