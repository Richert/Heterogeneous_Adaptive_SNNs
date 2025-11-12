import numpy as np
import matplotlib.pyplot as plt
from rectipy import Network
from pyrates import EdgeTemplate, NodeTemplate, CircuitTemplate
import torch
from copy import deepcopy
from time import perf_counter

def vanderpol(y: np.ndarray, x: float = 1.0, tau: float = 1.0) -> np.ndarray:
    y1, y2 = y[0], y[1]
    y1_dot = y2 / tau
    y2_dot = (y2*x*(1 - y1**2) - y1) / tau
    return np.asarray([y1_dot, y2_dot])

# preparations
##############

# general parameters
float_precision = "float64"
device = "cpu"

# Van der Pol parameters
vdp_vars = {"x": 1.0, "tau": 5.0}

# model parameters
node, node_op = "fre", "fre_op"
edge, edge_op = "oja_simple_edge", "oja_simple_op"
M = 100
delta_min, delta_max = 0.0, 1.0
eta_min, eta_max = -1.0, 1.0
tau_min, tau_max = 0.99, 1.01
indices = np.arange(1, M+1)
etas = np.random.uniform(eta_min, eta_max, M)
deltas = np.random.uniform(delta_min, delta_max, M)
taus = np.random.uniform(tau_min, tau_max, M)
node_vars = {"tau": taus, "J": 0.5 / np.sqrt(M), "eta": etas, "Delta": deltas}
edge_vars = {"a": 0.01}

# training parameters
alpha = 1e-3
dt = 1e-3
sampling_steps = 1
init_noise = vdp_vars["tau"] / 100.0
inp_noise = 1.0
init_steps = int(500.0/dt)
init_steps2 = int(30.0/dt)
epoch_steps = int(100.0/dt)
train_epochs = 20
test_epochs = 2

# initialize node template and weights
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")
edge_temp = EdgeTemplate.from_yaml(f"../config/fre_equations/{edge}")
for key, val in edge_vars.items():
    edge_temp.update_var(edge_op, key, val)

# create network
edges = []
w0 = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        edge_tmp = deepcopy(edge_temp)
        w = float(np.random.randn())
        w0[i, j] = w
        edge_tmp.update_var(edge_op, "w", w)
        edges.append((f"p{j}/{node_op}/r", f"p{i}/{node_op}/r_syn", edge_tmp,
                      {"weight": 1.0,
                       f"{edge}/{edge_op}/r_s": f"p{j}/{node_op}/r",
                       f"{edge}/{edge_op}/r_t": f"p{i}/{node_op}/r",
                       }))
template = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)}, edges=edges)
template.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# generate rectipy network
net = Network(dt=dt, device=device)
net.add_diffeq_node(label="qif", node=template, input_var="I_ext", output_var="r",
                    node_vars={key: val for key, val in node_vars.items()},
                    op=node_op, clear=False, float_precision=float_precision, N=M)

# add input layer
m = 2
W_in = torch.as_tensor(np.random.randn(M, m) / m, dtype=torch.float64, device=device)
net.add_func_node("inp", m, activation_function="identity")
net.add_edge("inp", "qif", weights=W_in, train=None)

# optimization
##############

# get random initial condition
obs = net.run(inp_noise*np.random.randn(init_steps, m), verbose=False, sampling_steps=10, enable_grad=False)

# get weight matrix
mapping = net["qif"]["node"]["weights"].cpu().numpy()
weights = net["qif"]["node"].y[M:]
etas = net["qif"]["node"][f"{node_op}/eta"].cpu().numpy()
idx = np.argsort(etas)
w1 = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
w1 = w1[:, idx]
w0 = w0[idx, :]
w0 = w0[:, idx]

plt.plot(obs.to_dataframe("out"))
fig, axes = plt.subplots(ncols=3, figsize=(12, 8))
for i, w in enumerate([w0, w1]):

    ax = axes[i]
    im = ax.imshow(w, aspect="auto", interpolation="none")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"W{i+1}")

plt.tight_layout()
plt.show()

print("Starting optimization...")
network_states, targets = [], []
for epoch in range(train_epochs):

    # generate lorenz input for random initial condition
    x = np.asarray([-1.0, 1.0]) + init_noise*np.random.randn(2)
    lorenz_states = []
    for step in range(epoch_steps):
        x = x + dt * vanderpol(x, **vdp_vars)
        lorenz_states.append(x)
    lorenz_states = np.asarray(lorenz_states)
    inputs = torch.tensor(lorenz_states[:-1], device=device, dtype=torch.float64)
    targets.append(lorenz_states[1::sampling_steps])
    # plt.plot(inputs)
    # plt.show()

    # get random initial condition
    obs = net.run(inp_noise*np.random.randn(init_steps2, m), verbose=False, sampling_steps=init_steps2, enable_grad=False)
    # plt.plot(obs.to_dataframe("out"))
    # plt.show()

    # collect network states
    t0 = perf_counter()
    obs = net.run(inputs=inputs, sampling_steps=sampling_steps, verbose=False, enable_grad=False)
    t1 = perf_counter()
    print(f'Finished network state collection of train epoch {epoch+1} after {t1 - t0} s.')
    network_states.append(torch.stack(obs["out"], dim=0))
    # plt.plot(obs.to_dataframe("out"))
    # plt.show()

# train read-out classifier
###########################

# ridge regression formula
X = torch.concatenate(network_states)
targets = torch.tensor(targets, device=device, dtype=torch.float64).reshape(X.shape[0], m)
X_t = X.T
w_out = torch.inverse(X_t @ X + alpha * torch.eye(X.shape[1])) @ (X_t @ targets)
y_train = X @ w_out

# progress report
t1 = perf_counter()
print(f'Finished fitting of read-out weights after {t1 - t0} s.')

# model testing
###############

# get weight matrix
mapping = net["qif"]["node"]["weights"].cpu().numpy()
weights = net["qif"]["node"].y[M:]
etas = net["qif"]["node"][f"{node_op}/eta"].cpu().numpy()
idx = np.argsort(etas)
w2 = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
w2 = w2[:, idx]

print("Starting testing...")
y_test, targets_test = [], []
for epoch in range(test_epochs):

    # generate lorenz input for random initial condition
    x = np.asarray([-1.0, 1.0]) + init_noise*np.random.randn(2)
    lorenz_states = []
    for step in range(epoch_steps):
        x = x + dt * vanderpol(x, **vdp_vars)
        lorenz_states.append(x)
    lorenz_states = np.asarray(lorenz_states)
    inputs = torch.tensor(lorenz_states[:-1], device=device, dtype=torch.float64)
    targets_test.append(lorenz_states[1::sampling_steps])

    # get random initial condition
    net.run(inp_noise*np.random.randn(init_steps2, m), verbose=False, sampling_steps=init_steps2, enable_grad=False)

    # collect network states
    t0 = perf_counter()
    with torch.no_grad():
        for step in range(epoch_steps):
            inp = inputs[step] if step < 100 else y @ w_out
            y = net.forward(inp)
            if step % sampling_steps == 0:
                y_test.append(y.detach().cpu().numpy())
    t1 = perf_counter()
    print(f'Finished network state collection of test epoch {epoch + 1} after {t1 - t0} s.')

# make prediction
y_test = np.asarray(y_test)
targets_test = np.concatenate(targets_test, axis=0)

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax1 = axes[0]
ax1.plot(y_test @ w_out.detach().cpu().numpy())
ax1.set_title('predictions (testing)')
ax1.set_xlabel('steps')
ax1.set_ylabel('y')
ax2 = axes[1]
ax2.plot(targets_test)
ax2.set_title('targets (testing)')
ax2.set_xlabel('steps')
ax2.set_ylabel('y')
ax3 = axes[2]
ax3.plot(y_test)
ax3.set_title('network dynamics')
ax3.set_xlabel('epochs')
ax3.set_ylabel('r')
plt.tight_layout()

fig, axes = plt.subplots(ncols=3, figsize=(12, 8))
for i, w in enumerate([w0, w1, w2]):

    ax = axes[i]
    im = ax.imshow(w, aspect="auto", interpolation="none")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"W{i+1}")

plt.tight_layout()
plt.show()