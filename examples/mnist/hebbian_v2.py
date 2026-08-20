import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.learning import Hebbian

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=500)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
examples = args.examples
n_workers = args.n_workers
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)


# Create simple Torch NN
network = Network(dt=dt)

# Add traces=True here
inpt = Input(784, shape=(1, 28, 28), traces=True)
network.add_layer(inpt, name="I")


from bindsnet.network.nodes import AdaptiveLIFNodes

output = AdaptiveLIFNodes(
    n_neurons, 
    thresh=-52.0,       # Base threshold
    theta_plus=0.01,    # How much the threshold increases per spike
    theta_decay=1e-7,   # How fast the threshold decays back to base
    traces=True
)

# Add traces=True here
# output = LIFNodes(
#     n_neurons, 
#     thresh=-52 + np.random.randn(n_neurons).astype(float), 
#     traces=True
# )
network.add_layer(output, name="O")




# 1st ver ~84.43% accuracy
C1 = Connection(
    source=inpt, 
    target=output, 
    w=0.5 * torch.randn(inpt.n, output.n),
    update_rule=Hebbian, 
    nu=(1e-6, 1e-4)      
)

# 1. Define sparsity level (e.g., 80% sparse = only 20% connections active)
sparsity = 0.8 

# 2. Create a random mask: 1s where connection exists, 0s where it is blocked
mask = (torch.rand(n_neurons, n_neurons) > sparsity).float()

# 3. Create the base inhibitory matrix (all negative)
base_weights = -0.5 * (torch.ones(n_neurons, n_neurons) - torch.eye(n_neurons))

# 4. Apply the mask to the weights
sparse_weights = base_weights * mask

# 5. Initialize C2 with the sparse matrix
C2 = Connection(
    source=output, 
    target=output, 
    w=sparse_weights
)

# C2 = Connection(
#     source=output, 
#     target=output, 
#     w= -0.5 * (torch.ones(output.n, output.n) - torch.eye(output.n)) 
# )


#orig
# C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n))
# C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n))


#change 2 56%
# C1 is now a STATIC, fixed random projection (Classic Reservoir)
# C1 = Connection(
#     source=inpt, 
#     target=output, 
#     w=0.5 * torch.randn(inpt.n, output.n)
#     # No update_rule here!
# )

# # C2 is PLASTIC. The reservoir learns its own internal dynamics.
# C2 = Connection(
#     source=output, 
#     target=output, 
#     w=0.5 * torch.randn(output.n, output.n),
#     update_rule=Hebbian, 
#     nu=(1e-5, 1e-4), # (decay rate, strengthen rate)
#     wmin=-2.0,       # Crucial for recurrent connections
#     wmax=2.0
# )

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

# Monitors for visualizing activity
spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time, device=device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {"O": Monitor(network.layers["O"], ["v"], time=time, device=device)}
network.add_monitor(voltages["O"], name="O_voltages")

# Directs network to GPU
if gpu:
    network.to("cuda")

# Get MNIST training images and labels.
# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

def extract_features(s_tensor, v_tensor):
    """
    Extracts multi-dimensional features from the reservoir's activity.
    s_tensor shape: (time, n_neurons)
    v_tensor shape: (time, n_neurons)
    """
    # 1. Spike Frequency
    freq = s_tensor.sum(dim=0).float()
    
    # 2. Time-to-First-Spike (TFS)
    # If a neuron fires, get the normalized time (0 to 1). 
    # If it never fires, default to 1.0 (maximum time).
    tfs = torch.where(
        freq > 0, 
        s_tensor.float().argmax(dim=0).float() / s_tensor.shape[0], 
        torch.ones_like(freq)
    )
    
    # 3. Final Membrane Voltage
    final_v = v_tensor[-1, :].float()
    
    # Concatenate into a single 1D tensor of size (3 * n_neurons)
    return torch.cat([freq, tfs, final_v])

def adaptive_synaptic_scaling(connection, spikes_tensor, target_rate=0.05, base_decay=1e-4):
    """
    Dynamically decays weights based on the neuron's firing rate.
    Hyperactive neurons receive stronger weight decay.
    """
    # Calculate actual firing rate of each output neuron
    firing_rate = spikes_tensor.float().mean(dim=0) 
    
    # Calculate how far above the target rate the neuron is
    rate_error = firing_rate - target_rate
    
    # Dynamic decay: base_decay + penalty for hyperactivity
    # torch.relu ensures we only penalize neurons that fire too much
    dynamic_decay = base_decay * (1.0 + torch.relu(rate_error * 10))
    
    # Apply the adaptive decay to the target connection weights
    connection.w.data -= connection.w.data * dynamic_decay
    
    # Keep weights bounded
    connection.w.data = torch.clamp(connection.w.data, -2.0, 2.0)

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
# Note: Because this is a reservoir network, no adjustments of neuron parameters occurs in this phase.
n_iters = examples
training_pairs = []
network.train(mode=True)
pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break

    # Extract & resize the MNIST samples image data for training
    #       int(time / dt)  -> length of spike train
    #       28 x 28         -> size of sample
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

    # Run network on sample image
    # network.run(inputs={"I": datum}, time=time)
    
    # # Normalize feedforward weights so their sum equals a constant (e.g., 78.4)
    # C1.w.data = C1.w.data / C1.w.data.sum(dim=0) * 78.4
    # training_pairs.append([spikes["O"].get("s"), label])

    # Run network on sample image
    network.run(inputs={"I": datum}, time=time)
    
    # 1. Grab the raw data tensors
    s_data = spikes["O"].get("s")
    v_data = voltages["O"].get("v")
    
    # 2. Extract augmented features (Freq + TFS + Voltage)
    features = extract_features(s_data, v_data)
    training_pairs.append([features, label])
    
    # 3. Apply dynamic learning rate / adaptive decay to C1
    adaptive_synaptic_scaling(C1, s_data)
    
    # Plot spiking activity using monitors
    if plot:
        # Plot the current image and reconstructed/encoded image
        inpt_axes, inpt_ims = plot_input(
            dataPoint["image"].view(28, 28),
            datum.view(int(time / dt), 784).sum(0).view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        # Plot spikes
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        # Plot voltages
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(time, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
        )
        # Plot weights between input and output
        weights_im = plot_weights(
            get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
        )
        # Plot weights between output and output
        weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

        plt.pause(1e-8)
    network.reset_state_variables()


# Define logistic regression model using PyTorch.
# These neurons will take the reservoirs output as its input, and be trained to classify the images.
# class NN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NN, self).__init__()
#         # h = int(input_size/2)
#         self.linear_1 = nn.Linear(input_size, num_classes)
#         # self.linear_1 = nn.Linear(input_size, h)
#         # self.linear_2 = nn.Linear(h, num_classes)

#     def forward(self, x):
#         out = torch.sigmoid(self.linear_1(x.float().view(-1)))
#         # out = torch.sigmoid(self.linear_2(out))
#         return out

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # LayerNorm automatically scales the crazy inputs (Freq, TFS, Voltages)
        # to have a mean of 0 and std of 1.
        self.norm = nn.LayerNorm(input_size)
        self.linear_1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.float().view(1, -1)
        x = self.norm(x) # Normalize before the linear layer!
        
        # Remove the sigmoid. CrossEntropyLoss expects raw "logits" (un-squashed values)
        out = self.linear_1(x) 
        return out

# Initialize with 3 * n_neurons (because of Freq + TFS + Voltage)
# Create model
model = NN(3 * n_neurons, 10).to(device)

# CrossEntropy is FAR better for 10-class classification than MSE
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam handles messy data better than SGD

# Training the Model
print("\n Training the read out")
pbar = tqdm(enumerate(range(n_epochs)))
for epoch, _ in pbar:
    avg_loss = 0

    for i, (s, l) in enumerate(training_pairs):
        optimizer.zero_grad()

        outputs = model(s) # outputs is shape (1, 10)

        # CrossEntropyLoss just needs the raw integer label, no one-hot vectors required!
        label_tensor = torch.tensor([l], dtype=torch.long, device=device)
        
        loss = criterion(outputs, label_tensor)
        avg_loss += loss.item() # Use .item() to safely get the float value

        loss.backward()
        optimizer.step()

    pbar.set_description_str(
        "Epoch: %d/%d, Loss: %.4f"
        % (epoch + 1, n_epochs, avg_loss / len(training_pairs))
    )

# Run same simulation on reservoir with testing data instead of training data
# (see training section for intuition)
n_iters = examples
test_pairs = []
network.train(mode=False)

pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))

    # network.run(inputs={"I": datum}, time=time)
    # test_pairs.append([spikes["O"].get("s"), label])

    # During Testing (No adaptive scaling here, just feature extraction)
    network.run(inputs={"I": datum}, time=time)
    
    s_data = spikes["O"].get("s")
    v_data = voltages["O"].get("v")
    
    features = extract_features(s_data, v_data)
    test_pairs.append([features, label])
    if plot:
        inpt_axes, inpt_ims = plot_input(
            dataPoint["image"].view(28, 28),
            datum.view(time, 784).sum(0).view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(time, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
        )
        weights_im = plot_weights(
            get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
        )
        weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

        plt.pause(1e-8)
    network.reset_state_variables()

# Test model with previously trained logistic regression classifier
# Test model with previously trained logistic regression classifier
correct, total = 0, 0

for s, label in test_pairs:
    outputs = model(s) # outputs is shape (1, 10)
    
    # Get the index of the highest probability (the predicted digit)
    predicted = torch.argmax(outputs, dim=1)
    
    total += 1
    
    # .item() safely pulls the single value out of the tensor
    # We cast both to int to ensure a safe comparison
    if int(predicted.item()) == int(label):
        correct += 1

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (n_iters, 100 * correct / total)
)
