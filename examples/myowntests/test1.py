import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.connections import Connection
from bindsnet.monitor import Monitor

# 1. Initialize Network
network = Network()

# 2. Create Layers
# Input layer (e.g., 784 neurons for MNIST)
input_layer = Input(n=784)
# Hidden layer with Leaky Integrate-and-Fire neurons
hidden_layer = LIFNodes(n=100)

# 3. Create Connections (Synapses)
# connects input_layer to hidden_layer with random weights
conn = Connection(source=input_layer, target=hidden_layer)

# 4. Add components to the network
network.add_layer(input_layer, name='A')
network.add_layer(hidden_layer, name='B')
network.add_edge(conn, source='A', target='B')

# 5. Set up a Monitor to watch spikes in the hidden layer
monitor = Monitor(hidden_layer, state_vars=['spikes'])
network.add_monitor(monitor, name='B_spikes')

# 6. Run the simulation
# We provide input data (spikes) and run for 100 milliseconds (time steps)
inputs = {'A': torch.bernoulli(0.1 * torch.ones(100, 784))} # Random spike train
network.run(inputs=inputs, time=100)

# 7. Check the results
spikes = monitor.get('spikes')
print(f"Total spikes in hidden layer: {spikes.sum()}")