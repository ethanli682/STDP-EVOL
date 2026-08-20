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
parser.set_defaults(plot=False, gpu=False, train=True)


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
inpt = Input(784, shape=(1, 28, 28), traces = True)
network.add_layer(inpt, name="I")
base_theta_plus = torch.full((n_neurons,), 0.15, device=device)

output = AdaptiveLIFNodes(
    n_neurons, 
    thresh=-40.0,       # Base threshold
    theta_plus=base_theta_plus,    # How much the threshold increases per spike
    theta_decay=1e-7,   # How fast the threshold decays back to base
    traces=True
)

network.add_layer(output, name="O")

C1 = Connection(
    source=inpt, 
    target=output, 
    w=0.25 * torch.rand(inpt.n, output.n),   
)




C2 = Connection(
    source=output, 
    target=output, 
    w=-15.0 * torch.rand(output.n, output.n),
    update_rule=Hebbian,
    nu=[-1e-4, -1e-2],
    wmin=-20.0,       # Floor
    wmax=0.0
)

# with torch.no_grad():
#     diag_mask = torch.eye(C2.w.size(0), device=C2.w.device).bool()
#     C2.w.masked_fill_(diag_mask, 0.0)

# C1 = Connection(
#     source=inpt, 
#     target=output, 
#     w=0.15 * torch.rand(inpt.n, output.n), 
#     update_rule=PostPre,                 
#     nu=(1e-4, 1e-4),                       
#     wmin=0.0,                              
#     wmax=1.0                               
# )

# # --- C2: STATIC LATERAL INHIBITION ---
# C2 = Connection(
#     source=output, 
#     target=output, 
#     w=-10.0 * torch.rand(output.n, output.n), # FIXED: Used rand() so it is strictly negative
#     wmin=-10.0,       
#     wmax=0.0
#     # FIXED: Removed Hebbian update_rule so brakes are permanent
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
delta_weights_im = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

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

    print(f"Active Neurons: {active_neurons}/500 | Max Spikes by one neuron: {max_spikes}")

    # --- NEW: Extract and print the IDs of every neuron that fired ---
    spiking_neuron_ids = torch.nonzero(s_counts > 0).squeeze(-1).tolist()
    print(f"All spiking neurons for this image: {spiking_neuron_ids}")

    # --- FIRST TO FIRE (LATENCY) MONITOR ---
    # Get the raw spike matrix [time, n_neurons]
    spike_matrix = spikes["O"].get("s").squeeze()
    
    # torch.nonzero returns all coordinates [time, neuron_id] where a spike occurred
    all_spikes = torch.nonzero(spike_matrix)
    
    # if len(all_spikes) > 0:
    #     # The first row contains the earliest timestep a spike was recorded
    #     first_timestep = all_spikes[0, 0].item()
        
    #     # Find all neurons that fired at this exact starting millisecond
    #     first_neurons = torch.where(spike_matrix[first_timestep] > 0)[0].tolist()
        
    #     print(f"First Spike at t={first_timestep:03d} ms | Neurons involved: {first_neurons}")
    # else:
    #     print("First Spike: NONE (Coma)")
    
    # # Add these spikes to the running tally for the current digit class
    neuron_class_spikes[:, label] += s_counts
    
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
        # weights_im = plot_weights(
        #     get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
        # )
        # Plot weights between output and output
        weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

        # DELTA GRAPH
        delta_w = C2.w - initial_c2_w
        delta_weights_im = plot_weights(
            delta_w, 
            im=delta_weights_im, 
            wmin=-2.0,  # Negative values mean the inhibitory connection got stronger (more negative)
            wmax=2.0    # Positive values mean the connection weakened (pushed toward 0.0)
        )

        plt.pause(1e-8)
    network.reset_state_variables()
    if i == 100:
        print("\n" + "="*40)
        print("INTERVENTION AT ITERATION 100")
        
        # Calculate current assignments based on the tally so far
        current_assignments = torch.argmax(neuron_class_spikes, dim=1)
        
        # 1. Create a boolean mask of the neurons currently voting for '0'
        zero_mask = (current_assignments == 0)
        
        # 2. Reach directly into the output layer and overwrite the physics for ONLY those neurons
        # We increase their fatigue penalty by 5x (from 0.5 to 2.5)
        network.layers["O"].theta_plus[zero_mask] = 2.5
        
        zero_neurons = torch.nonzero(zero_mask).squeeze(-1).tolist()
        print(f"Identified {len(zero_neurons)} neurons assigned to '0'.")
        print("Hyper-fatigue (theta_plus=2.5) successfully applied to Team 0.")
        print("="*40 + "\n")


neuron_assignments = torch.argmax(neuron_class_spikes, dim=1)


# # --- DOMINANT EXCLUSIVITY FILTER (Soft Margin) ---
# PURITY_THRESHOLD = 0.85  # A neuron must dedicate 85% of its spikes to ONE digit

# # 1. Find the highest spike count for a single digit, and what digit that was
# max_spikes, best_classes = torch.max(neuron_class_spikes, dim=1)

# # 2. Find total spikes across all 10 digits for each neuron
# total_spikes = neuron_class_spikes.sum(dim=1)

# # 3. Prevent division by zero for dead neurons
# active_mask = total_spikes > 0

# # 4. Calculate the purity ratio (e.g., 850 spikes for '0' / 1000 total spikes = 0.85)
# purity_ratio = torch.zeros(n_neurons, device=device)
# purity_ratio[active_mask] = max_spikes[active_mask] / total_spikes[active_mask]

# # 5. Create a mask of neurons that meet or exceed the threshold
# pure_mask = (purity_ratio >= PURITY_THRESHOLD) & active_mask

# # 6. Initialize all assignments to -1, then assign the ones that passed the test
# neuron_assignments = torch.full((n_neurons,), -1, dtype=torch.long, device=device)
# neuron_assignments[pure_mask] = best_classes[pure_mask]

# print("\n" + "="*45)
# print(f"--- Dominant Exclusivity Filter ({PURITY_THRESHOLD*100}%) ---")
# print(f"Total 'Specialist' Neurons: {pure_mask.sum().item()} / {n_neurons}")
# print("="*45 + "\n")


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

# # --- SPIKE-TRIGGERED AVERAGE (FEATURE VISUALIZATION) ---
# print("\n--- Running Spike-Triggered Average (STA) Analysis ---")

# # Let's analyze the features for Digit "0" (you can change this loop to check other digits)
# target_digit = 0

# # 1. Find the "Captain" Neuron for this digit
# # (The neuron assigned to '0' that had the highest total spikes across the test set)
# assigned_to_target = (neuron_assignments == target_digit)
# # We need to find which of these assigned neurons fired the absolute most.
# # For simplicity, we'll just pick the first one in the list for now, 
# # or you can track the max spiking neuron during the test loop.
# target_neurons = torch.nonzero(assigned_to_target).squeeze()

# if len(target_neurons.shape) == 0 or len(target_neurons) == 0:
#     print(f"No neurons assigned to digit {target_digit}.")
# else:
#     captain_neuron = target_neurons[0].item() # Pick the first neuron assigned to '0'
#     print(f"Analyzing Receptive Field for Neuron #{captain_neuron} (Assigned to '{target_digit}')")

#     # 2. Compute the STA
#     sta_image = torch.zeros(28, 28, device=device)
#     total_neuron_spikes = 0.0

#     # We need to do one more quick pass over the dataloader to get the raw images
#     network.train(mode=False)
#     pbar = tqdm(enumerate(dataloader), total=examples)
    
#     for i, dataPoint in pbar:
#         if i >= examples: break
        
#         raw_image = dataPoint["image"].view(28, 28).to(device)
#         datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
        
#         network.run(inputs={"I": datum}, time=time)
        
#         # Get how many times our captain neuron spiked for this specific image
#         spike_count = spikes["O"].get("s")[:, 0, captain_neuron].sum().item()
        
#         if spike_count > 0:
#             # Multiply the raw image by the spike count and add it to our average
#             sta_image += (raw_image * spike_count)
#             total_neuron_spikes += spike_count
            
#         network.reset_state_variables()

#     # 3. Visualize the resulting feature
#     if total_neuron_spikes > 0:
#         sta_image = sta_image / total_neuron_spikes # Normalize
        
#         plt.figure(figsize=(5, 5))
#         plt.title(f"Preferred Feature for Neuron #{captain_neuron} (Digit {target_digit})")
#         plt.imshow(sta_image.cpu().numpy(), cmap='hot', interpolation='nearest')
#         plt.colorbar(label='Feature Importance')
#         plt.show()
#     else:
#         print("This neuron never spiked during the analysis pass.")

# print("\n" + "="*60)
# print("🏆 NEURON PERFORMANCE LEADERBOARD (TEST SET) 🏆")
# print("="*60)

# # How many top neurons you want to see per digit
# top_n = 5 

# for digit in range(10):
#     # 1. Get the tally row for this specific digit
#     digit_spikes = test_spike_tally[digit, :]
    
#     # 2. Sort the neurons by how much they fired for this digit (descending)
#     sorted_spikes, sorted_indices = torch.sort(digit_spikes, descending=True)
    
#     print(f"\nDigit '{digit}' - Top {top_n} Star Neurons:")
#     print("-" * 55)
#     print(f"{'Neuron ID':<15} | {'Total Spikes':<15} | {'Assigned Team':<15}")
#     print("-" * 55)
    
#     for i in range(top_n):
#         n_id = sorted_indices[i].item()
#         total_fired = int(sorted_spikes[i].item())
#         team = neuron_assignments[n_id].item()
        
#         # Highlight if a neuron is doing heavy lifting for a team it doesn't belong to!
#         if team != digit:
#             team_str = f"Team {team} (ROGUE!)"
#         else:
#             team_str = f"Team {team}"
            
#         print(f"Neuron {n_id:<12} | {total_fired:<15} | {team_str}")