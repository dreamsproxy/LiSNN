# LiSNN

LiSNN is an experimental liquid spiking neural-network substrate focused on
closed-loop learning, heterogeneous neuron populations, and biologically
inspired local dynamics.

The active architecture was reset in September 2026 to support experiments
inspired by closed-loop biological-computing work such as DishBrain. Previous
LiSNN/LiNN/BioEmulator experiments are intentionally frozen under `legacy/`.

## Current development boundary

The active implementation currently provides:

- a shared `float32` neuron population matrix with shape `(N, 43)`;
- vectorized NumPy neuron kernels;
- deterministic or physiologically randomized parameter initialization;
- homogeneous and heterogeneous neuron-population construction;
- contiguous neuron-type slices for future vectorized network stepping;
- population smoke testing and forced post-spike state verification.

Synaptic weights, connectivity, propagation, plasticity, and closed-loop
feedback learning are deliberately not part of the network constructor yet.

## Supported neuron models

- LIF
- Adaptive LIF
- Izhikevich
- AdEx
- GLIF3
- GLIF4
- GLIF5
- CAdEx
- experimental CAdEx-GLIF hybrid

## Package layout

```text
lisnn/
  neurons/
    layout.py       # shared neuron-matrix ABI
    models.py       # vectorized neuron-step import surface
    population.py   # typed population construction
    registry.py     # NeuronType + step registry
  network/
    spec.py         # homogeneous/mixed population specification
    core.py         # SNN constructor/container
  debugging/
    __init__.py     # debugging API
  types.py          # shared NumPy/type aliases
```

The original root modules remain compatibility facades or implementation
surfaces so existing experiments do not need to change immediately.

## Network construction

```python
import Network

model = Network.create_nn(
    population=8,
    neuron_type="GLIF5",
    randomize_params=True,
    seed=1,
)
```

Mixed populations use explicit counts and a required default type. Unassigned
neurons are assigned to the default group:

```python
model = Network.create_nn(
    population=8,
    neuron_type={
        "default": "LIF",
        "GLIF5": 2,
        "AdEx": 3,
    },
    randomize_params=True,
    seed=1,
)
```

The same API can be used through the package directly:

```python
from lisnn import NeuronType, create_nn

model = create_nn(
    population=8,
    neuron_type=NeuronType.GLIF5,
)
```

## Population debugging

```python
import debugging as debug

debug.population_smoke_test(
    neuron_count=8,
    n_steps=8,
    watch_spikes=True,
    verbose=True,
)
```

The smoke test checks alternating-input integration and explicitly exercises
threshold crossing, spike/reset behavior, refractory state, and model-specific
post-spike state transitions.

## Design invariant

Neuron objects are not instantiated individually. The neuronal substrate stays
vectorized:

```text
population -> float32 ndarray[N, 43]
neuron type -> contiguous population slice
step kernel -> vectorized operation over that slice
```

This invariant should be preserved as connectivity, synapses, plasticity, and
closed-loop feedback are added.
