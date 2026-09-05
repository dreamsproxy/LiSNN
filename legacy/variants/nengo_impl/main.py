import pandas as pd
import numpy as np
import nengo

# Define some parameters
n_neurons = 100  # Number of neurons in each ensemble
sim_time = 1.0  # Simulation time (in seconds)
dt = 0.001  # Simulation time step
repeat_factor = 10  # Number of times to repeat each row

# Load your pandas dataframe
# Replace this with your actual data loading code
data = pd.read_csv('./dataset/train.csv').drop(columns=['date'])

# Normalize the data between -1 and 1
data_normalized = (2 * (data - data.min()) / (data.max() - data.min())) - 1

# Repeat each row n times
data_repeated = np.repeat(data_normalized.values, repeat_factor, axis=0)

# Define the model
with nengo.Network() as model:
    # Define input node
    input_node = nengo.Node(output=data_repeated[0])

    # Create ensembles for each column in the dataframe
    ensembles = []
    for i in range(data_normalized.shape[1]):
        ensembles.append(nengo.Ensemble(n_neurons, dimensions=1))

    # Connect input node to each ensemble
    for i, ensemble in enumerate(ensembles):
        nengo.Connection(input_node[i], ensemble)

    # Define connections between ensembles with recurrent connection (to implement time constant)
    for i in range(len(ensembles)):
        for j in range(len(ensembles)):
            if i != j:
                nengo.Connection(ensembles[i], ensembles[j], transform=(1.0), synapse=0.1)

    # Define the BCM learning rule
    bcm = nengo.BCM(learning_rate=1e-8)

    # Connect the BCM learning rule to all ensembles
    for ensemble in ensembles:
        conn = nengo.Connection(ensemble.neurons, ensemble.neurons, transform = np.random.randn(n_neurons, n_neurons), learning_rule_type=bcm)

    # Probe the output of each ensemble
    probes = [nengo.Probe(ensemble, synapse=0.01) for ensemble in ensembles]

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(sim_time)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i, probe in enumerate(probes):
    plt.subplot(len(probes), 1, i + 1)
    plt.plot(sim.trange(), sim.data[probe])
    plt.title(f'Column {i+1}')
plt.tight_layout()
plt.show()
