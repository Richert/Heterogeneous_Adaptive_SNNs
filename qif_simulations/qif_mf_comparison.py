from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from numba import config
config.THREADING_LAYER = "tbb"
from config.utility_functions import *
from time import perf_counter
# np.random.seed(42)

@njit
def get_xy(s: np.ndarray, u_p: np.ndarray, u_d: np.ndarray) -> tuple:
    if plasticity == "oja":
        x = np.outer(s, u_p)
        y = np.repeat(s*u_d, N).reshape(N, N)
    elif plasticity == "antioja":
        x = np.repeat(s*u_p, N).reshape(N, N).T
        y = np.outer(s, u_d)
    elif plasticity == "stdp_asym":
        x = np.outer(s, u_p)
        y = np.outer(u_d, s)
    elif plasticity == "stdp_sym":
        x = np.outer(u_p, u_p)
        y = np.outer(u_d, u_d)
    else:
        raise ValueError(f"Invalid condition: {plasticity}.")
    return x, y

@njit
def qif_rhs(y: np.ndarray, w: np.ndarray, spikes: np.ndarray, eta: np.ndarray, inp: float, J: float, tau_s: float,
            tau_p: float, tau_d: float, a_p: float, a_d: float, b: float) -> tuple:
    v, s, u_p, u_d = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:4*N]
    dy = np.zeros_like(y)
    x, y = get_xy(s, u_p, u_d)
    dy[:N] = v**2 + eta + J*np.dot(w, s) + inp
    dy[N:2*N] = spikes - s/tau_s
    dy[2*N:3*N] = s - u_p/tau_p
    dy[3*N:4*N] = s - u_d/tau_d
    dw = b*(a_p*(1-w)*x - a_d*w*y) + (1-b)*(a_p*x-a_d*y)*(w-w**2)
    return dy, dw

@njit
def spiking(y: np.ndarray, spikes: np.ndarray, dt: float):
    idx = np.argwhere((v_p - y[:N]) < 0.0).flatten()
    spikes[:] = 0.0
    y[idx] = -y[idx]
    spikes[idx] = 1.0/dt

def solve_ivp(dt: float, y: np.ndarray, w: np.ndarray, eta: np.ndarray, inp: np.ndarray, J: float, tau_s: float,
              tau_p: float, tau_d: float, a_p: float, a_d: float, b: float, sr: int = 100) -> tuple:

    spikes = np.zeros((N,))

    y_col = np.zeros((int(inp.shape[0]/sr), 4*N))
    ss = 0
    for step in range(inp.shape[0]):
        spiking(y, spikes, dt)
        dy, dw = qif_rhs(y, w, spikes, eta, inp[step], J, tau_s, tau_p, tau_d, a_p, a_d, b)
        y = y + dt * dy
        w = w + dt * dw
        if step % sr == 0:
            y_col[ss, :] = y[:]
            ss += 1

    return y_col, w

def solve_fr(dt: float, w: np.ndarray, r: np.ndarray, eta: np.ndarray, inp: np.ndarray, J: float,
             a_p: float, a_d: float, b: float, sr: int = 100) -> tuple:

    s_col = np.zeros((int(inp.shape[0]/sr), N))
    ss = 0
    for step in range(inp.shape[0]):
        r, dw = delta_w(w, r, inp[step], eta, J, b, a_p, a_d)
        w = w + dt * dw
        if step % sr == 0:
            s_col[ss, :] = r[:]
            ss += 1

    return s_col, w

@njit
def delta_w(w: np.ndarray, r: np.ndarray, inp: np.ndarray, eta: np.ndarray, J: float, b: float, a_p: float,
            a_d: float) -> tuple:
    r = get_qif_fr(inp + eta + J*np.dot(w, r))
    x, y = get_xy(r*tau_s, r*tau_p, r*tau_d)
    return r, b*(a_p*(1-w)*x - a_d*w*y) + (1-b)*(a_p*x-a_d*y)*(w-w**2)

@njit
def get_qif_fr(x: np.ndarray) -> np.ndarray:
    fr = np.zeros_like(x)
    fr[x > 0] = np.sqrt(x[x > 0])
    return fr / np.pi

# define data directory
path = "/home/rgast/data/mpmf_simulations"

# read condition
trial = 0
syn = "exc"
plasticity = "stdp_asym"

# define stdp parameters
a = 0.1
a_r = 1.2
tau = 10.0
tau_r = 2.0
a_p = a*a_r
a_d = a/a_r
tau_p = tau
tau_d = tau*tau_r
tau_ratio = tau_p / tau_d
a_ratio = a_p / a_d
stdp_ratio = tau_ratio*a_ratio

# set model parameters
M = 10
N = 200
J = 15.0
J_mp = J / (0.5*M)
J_qif = J / (0.5*N)
Delta = 2.0
eta = -0.65
b = 0.5
tau_s = 0.5
v_p = 100.0
etas_mp = uniform(M, eta, Delta)
etas = uniform(N, eta, Delta)
node_vars = {"eta": etas_mp, "Delta": Delta/(2*M)}
edge_vars = {"a_p": 0.0, "a_d": 0.0, "b": b}
syn_vars = {"tau_s": tau_s}

# simulation parameters
cutoff = 100.0
T = 2000.0
dt = 1e-3
dt2 = 5e-3
dts = 1e-1
inp_amp = 0.0
inp_freq = 0.005
inp_dur = 5.0
sr = int(dts/dt)
sr2 = int(dts/dt2)
sr3 = int(dt2/dt)

# node and edge template initiation
edge, edge_op = "stdp_edge", "stdp_op"
node, node_op, syn_op = f"qif_stdp", "qif_op", f"syn_op"
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        if plasticity == "stdp_asym":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": J_mp,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/p2": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{i}/ltd_op/u_d",
                           }))
        elif plasticity == "stdp_sym":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": J_mp,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{i}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{i}/ltd_op/u_d",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        elif plasticity == "antihebbian":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": J_mp,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p2": f"p{i}/ltp_op/u_p",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        elif plasticity == "oja":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": J_mp,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{i}/ltd_op/u_d",
                           }))
        elif plasticity == "antioja":
            edges.append((f"p{j}/{syn_op}/s", f"p{i}/{node_op}/s_in", deepcopy(edge_temp),
                          {"weight": J_mp,
                           f"{edge}/{edge_op}/s_in": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/p1": f"p{j}/ltp_op/u_p",
                           f"{edge}/{edge_op}/p2": f"p{j}/{syn_op}/s",
                           f"{edge}/{edge_op}/d1": f"p{i}/{syn_op}/s",
                           f"{edge}/{edge_op}/d2": f"p{j}/ltd_op/u_d",
                           }))
        else:
            raise ValueError(f"Unknown plasticity {plasticity}")
net = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})
net.update_var(node_vars={f"all/{syn_op}/{key}": val for key, val in syn_vars.items()})

# generate run function
inp = np.zeros((int(T/dt), 1), dtype=np.float32)
func, args, arg_keys, _ = net.get_run_func(f"{syn}_vectorfield", file_name=f"{syn}_run",
                                           step_size=dt, backend="numpy", solver="heun", float_precision="float32",
                                           vectorize=True, inputs={f"all/{node_op}/I_ext": inp}, clear=False)
func_njit = njit(func)
func_njit(*args)
rhs = func_njit

# find argument positions of free parameters
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")
a_p_idx = arg_keys.index(f"{edge}/{edge_op}/a_p")
a_d_idx = arg_keys.index(f"{edge}/{edge_op}/a_d")
tau_p_idx = arg_keys.index(f"p0/ltp_op/tau_p")
tau_d_idx = arg_keys.index(f"p0/ltd_op/tau_d")
eta_idx = arg_keys.index(f"p0/{node_op}/eta")

# set LTP/LTD time constants
args = list(args)
args[tau_p_idx] = tau_p
args[tau_d_idx] = tau_d

# set random initial connectivity
W0 = np.random.uniform(low=0.49, high=0.51, size=(M, M))
w0 = np.random.uniform(low=0.49, high=0.51, size=(N, N))
args[1][-int(M*M):] = W0.reshape((int(M*M),))

# get initial rate model rate
fr = get_qif_fr(etas)

# set initial state
t0 = perf_counter()
print("Starting initial washout simulations ...")
init_hist, y_init = integrate(args[1], rhs, tuple(args[2:]), cutoff, dt, dts)
s_init, _ = solve_ivp(dt2, np.zeros((4*N,)), w0, etas, inp[::sr3, 0], J_qif, tau_s, tau_p, tau_d, 0.0, 0.0, b, sr=sr2)
fr_init, _ = solve_fr(dt2, w0, fr, etas, inp[::sr3, 0], J_qif, 0.0, 0.0, b, sr=sr2)
t1 = perf_counter()
print(f"Finished initial washout simulations after {t1-t0:.2f} seconds.")

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
args[inp_idx] = inp

# run initial simulations
print("Starting T0 simulations ...")
y0_hist, y0 = integrate(y_init, rhs, tuple(args[2:]), T, dt, dts)
w0 = np.random.uniform(0.0, 1.0, size=(N, N,))
s0_hist, _ = solve_ivp(dt2, s_init[-1], w0, etas, inp[::sr3], J_qif, tau_s, tau_p, tau_d, 0.0, 0.0, b, sr=sr2)
fr0, _ = solve_fr(dt2, w0, fr_init[-1], etas, inp[::sr3], J_qif, 0.0, 0.0, b, sr=sr2)
print("finished T0 simulations.")

# turn on synaptic plasticity and run simulations again
print("Starting T1 simulations ...")
args[a_p_idx] = a_p
args[a_d_idx] = a_d
y1_hist, y1 = integrate(y0, rhs, tuple(args[2:]), T, dt, dts)
W1 = y1[-int(M * M):].reshape(M, M)
s1_hist, w1 = solve_ivp(dt2, s_init[-1], w0, etas, inp[::sr3], J_qif, tau_s, tau_p, tau_d, a_p, a_d, b, sr=sr2)
fr1, wr1 = solve_fr(dt2, w0, fr_init[-1], etas, inp[::sr3], J_qif, a_p, a_d, b, sr=sr2)
print("Finished T1 simulations.")

# turn off synaptic plasticity and run simulation a final time
print("Starting T2 simulations ...")
args[a_p_idx] = 0.0
args[a_d_idx] = 0.0
y2_hist, y2 = integrate(y1, rhs, tuple(args[2:]), T, dt, dts)
s2_hist, _ = solve_ivp(dt2, s_init[-1], w1, etas, inp[::sr3], J_qif, tau_s, tau_p, tau_d, 0.0, 0.0, b, sr=sr2)
fr2, _ = solve_fr(dt2, wr1, fr_init[-1], etas, inp[::sr3], J_qif, 0.0, 0.0, b, sr=sr2)
print("Finished T2 simulations.")

# collect state variables from MF model
r0, r1, r2 = y0_hist[:, 2*M:3*M], y1_hist[:, 2*M:3*M], y2_hist[:, 2*M:3*M]
s0, s1, s2 = s0_hist[:, N:2*N], s1_hist[:, N:2*N], s2_hist[:, N:2*N]

# report some basic stats
#########################

print(f"Neuron type: {syn}")
print(f"Plasticity rule: {plasticity}")
print(f"Log LTP/LTD ratio: {np.log(stdp_ratio)}")

# plotting
##########

fig = plt.figure(figsize=(16, 8), layout="constrained")
grid = fig.add_gridspec(ncols=3, nrows=4)

# plotting dynamics
time = np.linspace(0.0, T, int(T/dts)) / 100.0
for i, (s, r, fr) in enumerate(zip([s0, s1, s2], [r0, r1, r2], [fr0, fr1, fr2])):
    ax = fig.add_subplot(grid[i, :])
    ax.plot(time, np.mean(s, axis=1)*tau_s*100.0, label="QIF")
    ax.plot(time, np.mean(r, axis=1)*tau_s*100.0, label="MFE")
    ax.plot(time, np.mean(fr, axis=1)*tau_s*100.0, label="SSR")
    ax.legend()
    ax.set_ylabel("r (Hz)")
ax.set_xlabel("time (s)")

# plotting weights
for j, (w, title) in enumerate(zip([w1, W1, wr1], ["QIF", "MFE", "SSR"])):
    ax = fig.add_subplot(grid[3, j])
    im = ax.imshow(w, interpolation="none", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("n")
plt.colorbar(im, ax=ax, shrink=0.8)

fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
fig.canvas.draw()
plt.show()

# clear files up
clear(net)
