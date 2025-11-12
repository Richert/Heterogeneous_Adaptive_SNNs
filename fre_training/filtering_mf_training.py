import numpy as np
from rectipy import Network, circular_connectivity
from pyrates import NodeTemplate, CircuitTemplate
import torch
from time import perf_counter
from scipy.stats import rv_discrete
import sys
import pickle

def dist(x: int, method: str = "inverse", zero_val: float = 1.0, inverse_pow: float = 1.0) -> float:
    if method == "inverse":
        return 1/x**inverse_pow if x > 0 else zero_val
    if method == "exp":
        return np.exp(-x) if x > 0 else zero_val
    else:
        raise ValueError("Invalid method.")

# preparations
##############

# general parameters
numpy_precision = np.float32
torch_precision = torch.float32
device = "cpu"
path = "/home/richard/results/filter_training"

# task parameters
f = float(sys.argv[-1])
init_noise = 0.0001
inp_noise = float(sys.argv[-2])
rep = float(sys.argv[-3])

# model parameters
node = str(sys.argv[-4])
node_op = f"{node}_op"
M = 50
p = 0.2
p_in = 0.2
conn_pow = 1.5
indices = np.arange(1, M+1)
node_vars = {"tau": 1.0, "J": 10.0, "eta": 0.0, "Delta": 0.1}
train_params = ["J", "eta", "Delta"]

# training parameters
dt = 1e-2
lr = 1e-2
betas = (0.9, 0.99)
sampling_steps = 1
init_steps = int(10.0/dt)
trial_steps = int(50.0/dt)
batch_size = 10
train_epochs = 100
test_trials = 10
gradient_cutoff = 1e4

# create targets
time = np.arange(0, trial_steps) * dt
target_signal = np.sin(2.0*np.pi*f*time)

# initialize node template and weights
node_temp = NodeTemplate.from_yaml(f"../config/fre_equations/{node}_pop")

# create network
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(M, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=True)
W[np.eye(M) > 0.1] = 1.0
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(W, aspect="auto", interpolation="none")
# plt.show()

template = CircuitTemplate(name=node, nodes={f"p{i}": node_temp for i in range(M)})
template.update_var(node_vars={f"all/{node_op}/{key}": val for key, val in node_vars.items()})

# generate rectipy network
net = Network(dt=dt, device=device, dtype=torch_precision)
net.add_diffeq_node(label="qif", node=template, input_var="I_ext", output_var="r",
                    node_vars={key: val for key, val in node_vars.items()},
                    op=node_op, clear=False, dtype=torch_precision, in_place=False, to_file=False,
                    weights=W, source_var="r", target_var="r_in", train_params=train_params)

# add input layer
W_in = np.zeros((M, 1))
n_inputs = int(p_in*M)
center = int(M*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
W_in[inp_indices, 0] = np.random.rand(len(inp_indices))
W_in = torch.as_tensor(W_in, dtype=torch_precision, device=device)
net.add_func_node("inp", 1, activation_function="identity")
net.add_edge("inp", "qif", weights=W_in, train=None)

# add output layer
W_out = torch.as_tensor(np.random.randn(1, M), dtype=torch_precision, device=device)
net.add_func_node("out", 1, activation_function="identity")
net.add_edge("qif", "out", weights=W_out, train="gd")
net.compile()

# optimization
##############

# loss function
loss = torch.nn.MSELoss()

# optimizer definition
opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
for p in net.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -gradient_cutoff, gradient_cutoff))

print("Starting optimization...")
losses = []
for epoch in range(train_epochs):

    error = torch.zeros(1, device=device)
    for trial in range(batch_size):

        # get random initial condition
        net.run(init_noise*np.sqrt(dt)*np.random.randn(init_steps, 1), verbose=False, sampling_steps=init_steps, enable_grad=False)
        net.detach()

        # create input and target
        inp = inp_noise * np.sqrt(dt) * np.random.randn(trial_steps, 1)
        inp[:, 0] += target_signal
        target = np.zeros((trial_steps, 1))
        target[:, 0] += target_signal
        target = torch.tensor(target, device=device, dtype=torch_precision)
        # fig, ax = plt.subplots(figsize=(12, 4))
        # ax.plot(inp, label="inputs")
        # ax.plot(target[:, 0], label="target")
        # ax.legend()
        # plt.show()

        # collect network states
        t0 = perf_counter()
        obs = net.run(inputs=torch.tensor(inp, device=device, dtype=torch_precision),
                      sampling_steps=sampling_steps, verbose=False, enable_grad=True)
        t1 = perf_counter()
        # print(f'Finished network state collection of training trial {trial+1} after {t1 - t0} s.')

        # calculate loss
        prediction = torch.stack(obs["out"], dim=0)
        error += loss(prediction, target)

    # optimization step
    opt.zero_grad()
    error.backward()
    opt.step()
    losses.append(error.item())
    print(f"Training batch #{epoch+1} of {train_epochs} finished. Total epoch loss: {losses[-1]}.")


# model testing
###############

print("Starting testing...")
inputs, target_signals, predictions = [], [], []
test_error = 0
for trial in range(test_trials):

    # get random initial condition
    net.run(init_noise*np.sqrt(dt)*np.random.randn(init_steps, 1), verbose=False, sampling_steps=init_steps, enable_grad=False)
    net.detach()

    # create input and target
    inp = inp_noise * np.sqrt(dt) * np.random.randn(trial_steps, 1)
    inp[:, 0] += target_signal
    target = np.zeros((trial_steps, 1))
    target[:, 0] += target_signal
    target = torch.tensor(target, device=device, dtype=torch_precision)

    # collect network states
    t0 = perf_counter()
    obs = net.run(inputs=torch.tensor(inp, device=device, dtype=torch_precision),
                  sampling_steps=sampling_steps, verbose=False, enable_grad=False)
    t1 = perf_counter()
    print(f'Finished network state collection of test trial {trial + 1} after {t1 - t0} s.')

    # calculate loss
    prediction = torch.stack(obs["out"], dim=0)
    test_error += loss(prediction, target).item()
    inputs.append(inp[::sampling_steps])
    target_signals.append(target.detach().cpu().numpy())
    predictions.append(prediction.detach().cpu().numpy())

# save results
##############

results = {"test_loss": test_error, "trial": rep, "noise": inp_noise, "frequency": f}
for param_key, param_tensor in zip(train_params, net["qif"]["node"].train_params):
    results[param_key] = param_tensor.detach().cpu().numpy()
pickle.dump(results,
            open(f"{path}/filtering_{node}_n{int(inp_noise)}_f{int(f*100.0)}_r{int(rep)}.pkl", "wb"))
