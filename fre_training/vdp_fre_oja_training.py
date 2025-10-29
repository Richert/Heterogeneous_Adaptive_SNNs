import numpy as np
import matplotlib.pyplot as plt
from rectipy import Network
from pyrates import EdgeTemplate, NodeTemplate, CircuitTemplate
import torch
from copy import deepcopy

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
lorenz_vars = {"x": 1.0, "tau": 1.0}

# model parameters
node, node_op = "fre", "fre_op"
edge, edge_op = "oja_simple_edge", "oja_simple_op"
M = 100
p = 0.5
delta_min, delta_max = 0.01, 1.0
eta_min, eta_max = -2.0, 0.0
tau_min, tau_max = 0.2, 1.0
indices = np.arange(1, M+1)
etas = np.random.uniform(eta_min, eta_max, M)
deltas = np.random.uniform(delta_min, delta_max, M)
taus = np.random.uniform(tau_min, tau_max, M)
node_vars = {"tau": taus, "J": 10.0 / M, "eta": etas, "Delta": deltas}
edge_vars = {"a": 0.1}

# training parameters
alpha = 1e-5
dt = 1e-3
sampling_steps = 1
init_steps = int(200.0/dt)
train_steps = int(400.0/dt)
test_steps = int(200.0/dt)

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
        if np.random.uniform() <= p:
            w = float(np.random.uniform(-1.0, 1.0))
        else:
            w = 0.0
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

# wash out initial condition
obs = net.run(np.zeros((init_steps, 1)), verbose=False, sampling_steps=sampling_steps, enable_grad=False)
plt.plot(obs.to_dataframe("out"))
plt.show()
y0 = {key: val.clone() for key, val in net.state.items()}

# generate lorenz input
x = np.random.rand(2)
lorenz_states = []
for step in range(train_steps + test_steps):
    x = x + 0.1* dt * vanderpol(x, **lorenz_vars)
    lorenz_states.append(x)
lorenz_states = np.asarray(lorenz_states)
inputs = lorenz_states[:train_steps-1]
targets = lorenz_states[1:train_steps:sampling_steps]
# plt.plot(inp)
# plt.show()

# add input layer
m = lorenz_states.shape[-1]
W_in = torch.as_tensor(np.random.randn(M, m) / m, dtype=torch.float64, device=device)
net.add_func_node("inp", m, activation_function="identity")
net.add_edge("inp", "qif", weights=W_in, train=None)

# normalize input
for i in range(m):
    inputs[:, i] /= np.max(np.abs(inputs[:, i]))

# optimization
##############

print("Starting optimization...")
obs = net.fit_ridge(inputs=inputs, targets=targets, sampling_steps=sampling_steps, alpha=alpha, add_readout_node=False,
                    enable_grad=False)
w_out = obs["w_out"]

# model testing
###############

print("Starting testing...")
inputs = lorenz_states[train_steps:train_steps+test_steps-1]
targets = lorenz_states[train_steps+1::sampling_steps]
predictions, rates = [], []
for step in range(test_steps-1):
    r = net.forward(inputs[step])
    prediction = r @ w_out
    rates.append(r.detach().cpu().numpy())
    predictions.append(prediction.detach().cpu().numpy())
print("Finished.")

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax1 = axes[0]
ax1.plot(predictions)
ax1.set_title('predictions (testing)')
ax1.set_xlabel('steps')
ax1.set_ylabel('y')
ax2 = axes[1]
ax2.plot(inputs)
ax2.set_title('targets (testing)')
ax2.set_xlabel('steps')
ax2.set_ylabel('y')
ax3 = axes[2]
ax3.plot(rates)
ax3.set_title('network dynamics')
ax3.set_xlabel('epochs')
ax3.set_ylabel('r')
plt.tight_layout()
plt.show()
