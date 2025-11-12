import numpy as np
import matplotlib.pyplot as plt
from rectipy import Network
from pyrates import NodeTemplate, CircuitTemplate
import torch
from time import perf_counter

# preparations
##############

# general parameters
numpy_precision = np.float32
torch_precision = torch.float32
device = "cpu"

# model parameters
node, node_op = "fre", "fre_op"
M = 100
p = 0.2
delta_min, delta_max = 0.0, 1.0
eta_min, eta_max = -0.5, 0.5
tau_min, tau_max = 0.9, 1.1
indices = np.arange(1, M+1)
etas = np.random.uniform(eta_min, eta_max, M)
deltas = np.random.uniform(delta_min, delta_max, M)
node_vars = {"tau": 1.0, "J": 1.0 / np.sqrt(M*p), "eta": etas, "Delta": deltas}

# training parameters
dt = 2e-3
lr = 1e-4
betas = (0.9, 0.999)
sampling_steps = 1
inp_noise = 0.1 * np.sqrt(dt)
init_steps = int(5.0/dt)
trial_steps = int(20.0/dt)
batch_size = 10
train_epochs = 50
test_trials = 10

# initialize node template and weights
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")

# create network
w0 = np.asarray(np.random.randn(M, M), dtype=numpy_precision)
template = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)})
template.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# generate rectipy network
net = Network(dt=dt, device=device, dtype=torch_precision)
net.add_diffeq_node(label="qif", node=template, input_var="I_ext", output_var="r",
                    node_vars={key: val for key, val in node_vars.items()},
                    op=node_op, clear=False, dtype=torch_precision,
                    weights=w0, source_var="r", target_var="r_syn", train_params=["eta", "J"])

# add input layer
m_in = 2
W_in = torch.as_tensor(np.random.randn(M, m_in) / m_in, dtype=torch_precision, device=device)
net.add_func_node("inp", m_in, activation_function="identity")
net.add_edge("inp", "qif", weights=W_in, train=None)

# add output layer
m_out = 2
W_out = torch.as_tensor(np.random.randn(m_out, M), dtype=torch_precision, device=device)
net.add_func_node("out", m_out, activation_function="softmax")
net.add_edge("qif", "out", weights=W_out, train="gd")
net.compile()

# optimization
##############

# loss function
loss = torch.nn.CrossEntropyLoss()

# optimizer definition
opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)

print("Starting optimization...")
losses = []
for epoch in range(train_epochs):

    error = torch.zeros(1, device=device)
    for trial in range(batch_size):

        # get random initial condition
        net.run(inp_noise*np.random.randn(init_steps, m_in), verbose=False, sampling_steps=init_steps, enable_grad=False)
        net.detach()

        # create input and target
        channel = np.random.choice(np.arange(m_in))
        inp = inp_noise * np.random.randn(trial_steps, m_in)
        inp[:, channel] += 1.0
        targets = torch.zeros((trial_steps, m_out), device=device, dtype=torch_precision)
        targets[:, channel] = 1.0

        # collect network states
        t0 = perf_counter()
        obs = net.run(inputs=torch.tensor(inp, device=device, dtype=torch_precision),
                      sampling_steps=sampling_steps, verbose=False, enable_grad=True)
        t1 = perf_counter()
        # print(f'Finished network state collection of training trial {trial+1} after {t1 - t0} s.')

        # calculate loss
        predictions = torch.stack(obs["out"], dim=0)
        error += loss(predictions, targets)

    # optimization step
    opt.zero_grad()
    error.backward()
    opt.step()
    losses.append(error.item())
    print(f"Training batch #{epoch+1} of {train_epochs} finished. Total epoch loss: {losses[-1]}.")


# model testing
###############

print("Starting testing...")
targets, predictions = [], []
test_error = 0
for trial in range(test_trials):

    # get random initial condition
    net.run(inp_noise * np.random.randn(init_steps, m_in), verbose=False, sampling_steps=init_steps, enable_grad=False)
    net.detach()

    # create input and target
    channel = np.random.choice(np.arange(m_in))
    inp = inp_noise * np.random.randn(trial_steps, m_in)
    inp[:, channel] += 1.0
    target = torch.zeros((trial_steps, m_out), device=device, dtype=torch_precision)
    target[:, channel] = 1.0

    # collect network states
    t0 = perf_counter()
    obs = net.run(inputs=torch.tensor(inp, device=device, dtype=torch_precision),
                  sampling_steps=sampling_steps, verbose=False, enable_grad=False)
    t1 = perf_counter()
    print(f'Finished network state collection of test trial {trial + 1} after {t1 - t0} s.')

    # calculate loss
    prediction = torch.stack(obs["out"], dim=0)
    test_error += loss(prediction, target).item()
    targets.append(target.detach().cpu().numpy())
    predictions.append(prediction.detach().cpu().numpy())

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax1 = axes[0]
ax1.plot(np.concatenate(predictions, axis=0))
ax1.set_title(f'predictions (test MSE = {np.round(test_error, decimals=3)})')
ax1.set_xlabel('steps')
ax1.set_ylabel('y')
ax2 = axes[1]
ax2.plot(np.concatenate(targets, axis=0))
ax2.set_title('targets')
ax2.set_xlabel('steps')
ax2.set_ylabel('y')
ax3 = axes[2]
ax3.plot(losses)
ax3.set_title('loss (training)')
ax3.set_xlabel('epochs')
ax3.set_ylabel('MSE')
plt.tight_layout()
plt.show()
