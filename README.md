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

Sparse synapses and fixed integrated-current propagation are available as
independent modules. `FixedWeightRuntime` composes heterogeneous kernels with
causal spike buffers and separate pA current channels. Plasticity and task-driven
closed-loop learning are not implemented yet. See the
[propagation contract and Izhikevich adapters](lisnn/synapses/PROPAGATION.md).

## Supported neuron models

See [neuron units and timestep contract](lisnn/neurons/UNITS.md) for electrical
units, the resistance conversion correction, Izhikevich's native input scale,
and a hand-checkable example (`python -m examples.foundation_contracts`).

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
    runtime.py      # causal fixed-weight stepping
  synapses/
    core.py         # sparse topology and unsigned efficacy
    propagation.py  # independent integrated-current propagation
  debugging/
    runner.py       # full smoke-suite CLI
    *_smoke.py      # independently callable diagnostics
  types.py          # shared NumPy/type aliases
```

Root `main.py` demonstrates construction; root `debug.py` runs every active
smoke test. Implementation and import APIs live under `lisnn/`.

An initial synthetic text medium lives in `lisnn/media/`. It maps 7-bit ASCII
character IDs through fixed seeded embeddings into bounded pA currents on an
explicit input port. Presentation timing remains an experiment decision; see
`python -m examples.character_input` and `lisnn/media/README.md`.

## Network construction

```python
from lisnn import network

model = network.create_nn(
    population=8,
    neuron_type="GLIF5",
    randomize_params=True,
    seed=1,
)
```

Mixed populations use explicit counts and a required default type. Unassigned
neurons are assigned to the default group:

```python
model = network.create_nn(
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
from lisnn import debugging as debug

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

## Full smoke suite

```bash
python debug.py
python debug.py --verbose
python debug.py --only propagation
python -m lisnn.debugging --list
```

The default command runs all eight active suites and shows progress and PASS/FAIL
for each. It writes suite diagnostics and an aggregate JSON report into a
timestamped `logs/` directory; use `--log-dir PATH` to choose one. The process
returns a failing exit code when any suite fails. See the
[debugging guide](lisnn/debugging/README.md) for callable functions and reports.
The full regression suite remains `python -m pytest -q`.

## Import migration

| Old root import | Canonical replacement |
| --- | --- |
| `import Network` | `from lisnn import network` |
| `import NeuronModels as nm` | `from lisnn.neurons import kernels as nm` |
| `import debugging as debug` or `import debug` | `from lisnn import debugging as debug` |

`debug.py` is now the executable smoke-test entrypoint. Root `Network.py`,
`NeuronModels.py` and `debugging.py` are removed. External callers using those
names must update their imports. The frozen `legacy/` tree remains untouched.

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


## Fixed-weight propagation

```bash
python -m examples.fixed_weight_propagation
python -m lisnn.debugging.propagation_smoke
```

The example prints a hand-checkable three-tick pathway. The independently
callable smoke test covers causal timing, integrated impulses and shared-unit
Izhikevich observations. Propagation uses binary events and unsigned efficacy;
all runtime current channels are pA and membrane voltages are mV.
