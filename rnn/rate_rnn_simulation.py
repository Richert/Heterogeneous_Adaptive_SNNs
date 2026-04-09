from config.utility_functions import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
np.random.seed(42)

def get_prob(x, bins: int = 100):
    bins = np.linspace(0.0, 1.0, num=bins)
    counts, _ = np.histogram(x, bins=bins)
    return counts / np.sum(counts)

def get_xy(fr_source: np.ndarray, fr_target: np.ndarray, condition: str) -> tuple:
    if condition == "hebbian":
        x = np.outer(fr_target*tau_s, fr_source*tau_p)
        y = np.repeat((tau_s*tau_d*fr_target**2).reshape(len(fr_target), 1), len(fr_source), axis=1)
    elif condition == "antihebbian":
        x = np.repeat((fr_source**2).reshape(1, len(fr_source)), len(fr_target), axis=0)
        y = np.outer(fr_target, fr_source)
    else:
        raise ValueError(f"Invalid condition: {condition}.")
    return x, y

def delta_w(t: int, w: np.ndarray, r: np.ndarray, inp: np.ndarray, eta: np.ndarray, J: float, b: float, a: float,
            condition: str) -> np.ndarray:
    w = w.reshape(N, N)
    r[:] = get_qif_fr(inp[t] + eta + J*np.dot(w, r))
    x, y = get_xy(r, r, condition=condition)
    return (b*(a_p*(1-w)*x - a_d*w*y) + (1-b)*(a_p*x-a_d*y)*(w-w**2)).flatten()

def lorentzian(N: int, eta: float, Delta: float) -> np.ndarray:
    x = np.arange(1, N+1)
    return eta + Delta*np.tan(0.5*np.pi*(2*x-N-1)/(N+1))

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

def uniform(N: int, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.linspace(-0.5, 0.5, N)

@njit
def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

# parameter definition
save_results = False
condition = "hebbian"
distribution = "uniform"

# model parameters
Delta = 1.0
N = 200
J = -10.0 / (0.5*N)
eta = 0.0
b = 0.5
tau_s = 1.0

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

# simulation parameters
T = 2000.0
dt = 1e-2
dts = 1.0
inp_amp = 3.0
inp_freq = 0.005
inp_dur = 5.0

# choose sampling distribution
if distribution == "lorentzian":
    f = lorentzian
elif distribution == "gaussian":
    f = gaussian
else:
    f = uniform

# define initial condition
etas = f(N, eta, Delta)
fr = get_qif_fr(etas)
w0 = np.random.uniform(0.0, 1.0, size=(int(N*N),))

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

# get weight solutions
args = (fr, inp, etas, J, b, a, condition)
w_hist, w = integrate(w0, delta_w, args, T, dt, dts)
w0 = w0.reshape(N, N)
w = w.reshape(N, N)

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
plt.tight_layout()
plt.show()
