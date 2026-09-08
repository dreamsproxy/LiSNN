import numpy as np
import NeuronModels as nm
import debug
import Network

model = Network.create_nn(
    population=8,
    neuron_type="GLIF5",
    fill=np.float32(0.0),
    randomize_params=False,
)
print(model.pool.shape)
print(model.type_counts)
print(model.type_slices)
print(model.homogeneous)

model = Network.create_nn(
    population=8,
    neuron_type={
        "default": "LIF",
        "GLIF5": 2,
        "AdEx": 3,
    }, # type: ignore

    randomize_params=True,
    seed=1,
)

print(model.pool.shape)
print(model.type_counts)
print(model.type_slices)
print(model.homogeneous)

for neuron_type, s in self.type_slices.items():
    spike_fn(
        self.pool[s],
        input_current[s],
        dt,
    )