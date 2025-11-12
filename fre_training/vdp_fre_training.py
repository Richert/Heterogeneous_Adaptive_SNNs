import numpy as np
import matplotlib.pyplot as plt
from rectipy import Network
from pyrates import NodeTemplate, CircuitTemplate
import torch
from time import perf_counter

def vanderpol(y: np.ndarray, x: float = 1.0, tau: float = 5.0) -> np.ndarray:
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
vdp_vars = {"x": 1.0, "tau": 10.0}

# model parameters
node, node_op = "fre", "fre_op"
M = 100
p = 0.5
delta_min, delta_max = 0.0, 0.0
eta_min, eta_max = -0.1, 0.1
tau_min, tau_max = 0.99, 1.01
indices = np.arange(1, M+1)
etas = np.random.uniform(eta_min, eta_max, M)
deltas = np.random.uniform(delta_min, delta_max, M)
taus = np.random.uniform(tau_min, tau_max, M)
node_vars = {"tau": taus, "J": 0.5 / np.sqrt(M*p), "eta": etas, "Delta": deltas}

# training parameters
alpha = 1e-3
dt = 1e-3
sampling_steps = 1
init_noise = vdp_vars["tau"] / 100.0
inp_noise = 1.0
init_steps = int(20.0/dt)
epoch_steps = int(100.0/dt)
train_epochs = 20
test_epochs = 4

# initialize node template and weights
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")

# create network
w0 = np.random.randn(M, M)
template = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)})
template.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# generate rectipy network
net = Network(dt=dt, device=device)
net.add_diffeq_node(label="qif", node=template, input_var="I_ext", output_var="r",
                    node_vars={key: val for key, val in node_vars.items()},
                    op=node_op, clear=False, float_precision=float_precision,
                    weights=w0, source_var="r", target_var="r_syn")

# add input layer
m = 2
W_in = torch.as_tensor(np.random.randn(M, m) / m, dtype=torch.float64, device=device)
net.add_func_node("inp", m, activation_function="identity")
net.add_edge("inp", "qif", weights=W_in, train=None)

# optimization
##############

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
    net.run(inp_noise*np.random.randn(init_steps, m), verbose=False, sampling_steps=init_steps, enable_grad=False)
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
    net.run(inp_noise*np.random.randn(init_steps, m), verbose=False, sampling_steps=init_steps, enable_grad=False)

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
plt.show()
