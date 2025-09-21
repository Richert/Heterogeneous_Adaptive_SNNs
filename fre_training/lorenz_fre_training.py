import numpy as np
import matplotlib.pyplot as plt
from rectipy import Network
import torch

def lorenz(x: np.ndarray, s: float = 10.0, r: float = 28.0, b: float = 2.667) -> np.ndarray:
    """
    Parameters
    ----------
    x: np.ndarray
        State variables.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    np.ndarray
       Vectorfield of the Lorenz equations.
    """
    x1, x2, x3 = x[:]
    x1_dot = s*(x2 - x1)
    x2_dot = r*x1 - x2 - x1*x3
    x3_dot = x1*x2 - b*x3
    return np.asarray([x1_dot, x2_dot, x3_dot])

# preparations
##############

# general parameters
float_precision = "float64"
device = "cpu"

# lorenz parameters
lorenz_vars = {"s": 10.0, "r": 28.0, "b": 2.667}

# model parameters
node, node_op = "qif_sd", "qif_sd_op"
M = 50
Delta = 0.5
eta = -0.5
indices = np.arange(1, M+1)
etas = eta + Delta*np.tan(0.5*np.pi*(2*indices-M-1)/(M+1))
deltas = Delta*(np.tan(0.5*np.pi*(2*indices-M-0.5)/(M+1))-np.tan(0.5*np.pi*(2*indices-M-1.5)/(M+1)))
node_vars = {"tau": 1.0, "J": 10.0 / M, "eta": etas, "tau_s": 0.5, "Delta": deltas, "tau_a": 20.0,
             "kappa": 0.1, "A0": 0.5}

# training parameters
dt = 1e-3
init_steps = int(100.0/dt)
epoch_steps = int(50.0/dt)
n_epochs = 300
tol = 1e-3
n_cutoff = 1000

# initialize node template and weights
w0 = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        w = float(np.random.uniform(0.0, 1.0))
        w0[i, j] = w

# generate rectipy network
net = Network(dt=dt, device=device)
net.add_diffeq_node(label="qif", node=f"../config/fre_equations/{node}_pop", input_var="I_ext", output_var="r",
                    source_var="s", target_var="s_in", weights=w0,
                    node_vars={key: val for key, val in node_vars.items()},
                    op=node_op, clear=True, float_precision=float_precision, train_params=["tau", "eta", "J"])

# wash out initial condition
net.run(np.zeros((init_steps, 1)), verbose=False, sampling_steps=init_steps+1)
y0 = {key: val.clone() for key, val in net.state.items()}

# generate lorenz input
x = np.random.randn(3)
lorenz_states = []
for step in range(epoch_steps):
    x = x + dt * lorenz(x, **lorenz_vars)
    lorenz_states.append(x)
inp = np.asarray(lorenz_states)
# plt.plot(inp)
# plt.show()

# add input layer
m = inp.shape[-1]
W_in = torch.as_tensor(np.random.randn(M, m), dtype=torch.float64, device=device)
net.add_func_node("inp", m, activation_function="identity")
net.add_edge("inp", "qif", weights=W_in, train="gd")

# add output layer
W_out = torch.as_tensor(np.random.randn(m, M), dtype=torch.float64, device=device)
net.add_func_node("out", m, activation_function="tanh")
net.add_edge("qif", "out", weights=W_out, train="gd")
net.compile()

# normalize input
for i in range(m):
    inp[:, i] /= np.max(np.abs(inp[:, i]))
inp = torch.as_tensor(inp, dtype=torch.float64, device=device)

# optimization
##############

# loss function
loss = torch.nn.MSELoss()

# optimizer definition
opt = torch.optim.Rprop(net.parameters(), lr=0.01, etas=(0.5, 1.1), step_sizes=(1e-5, 0.5))

# optimization loop
print("Starting optimization...")
losses = []
error_tmp = torch.zeros(1)
error = 1.0
epoch = 0
while error > tol and epoch < n_epochs:

    # error calculation epoch
    losses_tmp = []
    net.reset(y0)
    for step in range(epoch_steps-1):
        target = inp[step+1]
        prediction = net.forward(inp[step])
        error_tmp += loss(prediction, target)
        losses_tmp.append(error_tmp.item())
        if step+1 % n_cutoff == 0:
            net.detach()

    # optimization step
    opt.zero_grad()
    error_tmp.backward()
    opt.step()
    error_tmp = torch.zeros(1)

    # save results and display progress
    error = np.mean(losses_tmp)
    losses.append(error)
    epoch += 1
    print(f"Training epoch #{epoch} finished. Mean epoch loss: {error}.")

# model testing
###############

print("Starting testing...")
net.reset(y0)
predictions = []
for step in range(epoch_steps):
    prediction = net.forward(inp[step])
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
ax2.plot(inp)
ax2.set_title('targets (testing)')
ax2.set_xlabel('steps')
ax2.set_ylabel('y')
ax3 = axes[2]
ax3.plot(losses)
ax3.set_title('loss (training)')
ax3.set_xlabel('epochs')
ax3.set_ylabel('MSE')
plt.tight_layout()
plt.show()
