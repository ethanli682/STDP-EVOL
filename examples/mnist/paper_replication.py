import argparse
import os
import time

import matplotlib.pyplot as plt
from sympy import true
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
from bindsnet.learning import Hebbian, PostPre

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights
from bindsnet.network.nodes import AdaptiveLIFNodes


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=300)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=80)
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
# --- 1. DEFINE THE THREE LAYERS ---
inpt = Input(784, shape=(1, 28, 28), traces=True)
network.add_layer(inpt, name="I")

exc = AdaptiveLIFNodes(
    n_neurons, 
    thresh=-40.0,       
    theta_plus=0.1,    
    tc_theta_decay=1e7, 
    traces=True
)
network.add_layer(exc, name="O")

# Inhibitory Layer (Standard LIF)
inh = LIFNodes(n_neurons, thresh=-40.0, traces=True)
network.add_layer(inh, name="Inh")

# --- 2. DEFINE THE CONNECTIONS ---

# C1: Input -> Excitatory (Visual Cortex, Plastic)
C1 = Connection(
    source=inpt, 
    target=exc, 
    w=0.3 * torch.rand(inpt.n, exc.n), 
    update_rule=PostPre,                 
    nu=(1e-4, 1e-2),                       
    wmin=0.0,                              
    wmax=1.0,
    norm=78.4  # Forces strict energy budget per neuron                            
)

# C2: Excitatory -> Inhibitory (The Twin Trigger, Static)
# An identity matrix mapping each excitatory neuron strictly to its inhibitory twin
w_exc_inh = torch.eye(n_neurons) * 22.5 
C2 = Connection(
    source=exc, 
    target=inh, 
    w=w_exc_inh,
    wmin=0.0,
    wmax=22.5
)

# C3: Inhibitory -> Excitatory (Flat Lateral Inhibition, Static)
# A flat matrix of -120.0 penalty to everyone EXCEPT the diagonal (which is 0.0)
w_inh_exc = -120.0 * (torch.ones(n_neurons, n_neurons) - torch.eye(n_neurons))
C3 = Connection(
    source=inh, 
    target=exc, 
    w=w_inh_exc, 
    wmin=-120.0, 
    wmax=0.0
)

# Add all connections to the network
network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="Inh")
network.add_connection(C3, source="Inh", target="O")
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
delta_weights_im = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)
with torch.no_grad():
    C1.w.data = C1.w.data * (78.4 / C1.w.data.sum(dim=0, keepdim=True))
initial_c2_w = C2.w.clone().detach()

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
# Note: Because this is a reservoir network, no adjustments of neuron parameters occurs in this phase.
n_iters = examples
training_pairs = []
network.train(mode=True)
pbar = tqdm(enumerate(dataloader))

neuron_class_spikes = torch.zeros(n_neurons, 10, device=device)

for i, dataPoint in pbar:
    if i > n_iters:
        break

    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"].item() # Get the integer label
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

    network.run(inputs={"I": datum}, time=time)
    
    # Get total spike counts for each neuron in this sample
    s_counts = spikes["O"].get("s").sum(dim=0).squeeze().float() 
    active_neurons = (s_counts > 0).sum().item()
    max_spikes = s_counts.max().item()
    neuron_class_spikes[:, label] += s_counts

    print(f"Active Neurons: {active_neurons}/500 | Max Spikes by one neuron: {max_spikes}")

    # --- NEW: Extract and print the IDs of every neuron that fired ---
    spiking_neuron_ids = torch.nonzero(s_counts > 0).squeeze(-1).tolist()
    print(f"All spiking neurons for this image: {spiking_neuron_ids}")

    # --- FIRST TO FIRE (LATENCY) MONITOR ---
    # Get the raw spike matrix [time, n_neurons]
    spike_matrix = spikes["O"].get("s").squeeze()
    
    # torch.nonzero returns all coordinates [time, neuron_id] where a spike occurred
    all_spikes = torch.nonzero(spike_matrix)

    # See how high the thresholds have climbed
    current_thetas = network.layers["O"].theta
    print(f"Max Threshold: {current_thetas.max().item():.2f} mV | Min Threshold: {current_thetas.min().item():.2f} mV")
    
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
        
        # Plot weights between input and output
        weights_im = plot_weights(
            get_square_weights(C1.w, 23, 28), 
            im=weights_im, 
            wmin=0.0,  # Floor of C1
            wmax=1.0   # Ceiling of C1
        )
        
        # DELTA GRAPH

        plt.pause(1e-8)
    network.reset_state_variables()
    # input("Press Enter to continue to the next training image...")
    


neuron_assignments = torch.argmax(neuron_class_spikes, dim=1)



n_iters = examples
test_pairs = []

test_spike_tally = torch.zeros(10, n_neurons, device=device)

network.train(mode=False)
pbar = tqdm(enumerate(dataloader))

for i, dataPoint in pbar:
    
    if i > n_iters:
        break
        
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"].item()
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))

    network.run(inputs={"I": datum}, time=time)
    
    # Extract the flat spike counts for this test image
    s_counts = spikes["O"].get("s").sum(dim=0).squeeze().float()
    test_pairs.append([s_counts, label])

    # --- STEP 2: UPDATE THE SCOREBOARD ---
    # Add this image's spikes to the row corresponding to the actual label
    test_spike_tally[label, :] += s_counts

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



# --- EVALUATE PATTERN MATCHING & CONFUSION MATRIX ---
correct, total = 0, 0

# Create a 10x10 matrix to track Actual vs. Predicted
# Rows = Actual Label, Columns = Predicted Label
confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)

for s_counts, label in test_pairs:
    class_votes = torch.zeros(10, device=device)
    
    for c in range(10):
        assigned_mask = (neuron_assignments == c)
        class_votes[c] = s_counts[assigned_mask].sum()
    
    predicted = torch.argmax(class_votes)
    
    # Safely extract integers for the matrix
    actual_label = int(label)
    predicted_label = int(predicted.item())
    
    # Log the prediction in the confusion matrix
    confusion_matrix[actual_label][predicted_label] += 1
    
    total += 1
    if predicted_label == actual_label:
        correct += 1

print(f"\nOverall Accuracy on {total} test images: {100 * correct / total:.2f} %")

# --- DEEP DIVE ANALYSIS PRINTOUT ---
print("\n--- Accuracy by Digit ---")
for i in range(10):
    # Total times this digit actually appeared
    total_actual = confusion_matrix[i].sum().item() 
    # Times the network correctly guessed this digit
    correct_predictions = confusion_matrix[i][i].item() 
    
    if total_actual > 0:
        acc = 100 * correct_predictions / total_actual
        print(f"Digit {i}: {acc:>5.1f}%  ({correct_predictions}/{total_actual})")
    else:
        print(f"Digit {i}: No samples in test set.")

# print("\n--- Top Confusions (Where the network gets tricked) ---")
for actual in range(10):
    for predicted in range(10):
        # Skip correct guesses
        if actual == predicted:
            continue
            
        mistakes = confusion_matrix[actual][predicted].item()
        # If it made this specific mistake more than 5 times, flag it
        if mistakes > 5: 
            print(f"Actual '{actual}' was mistaken for '{predicted}' -> {mistakes} times")
