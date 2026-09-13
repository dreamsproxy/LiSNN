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

Sparse synaptic topology and unsigned efficacy are available separately through
`lisnn.synapses`. Propagation, plasticity, and closed-loop feedback learning
are not implemented yet.

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
  debugging/
    runner.py       # all-suite runner and CLI
    *_smoke.py      # independently callable diagnostics
  synapses/
    core.py         # sparse topology and unsigned efficacy
  spatial/          # optional placement and morphology shell
  types.py          # shared NumPy/type aliases
```

Root `main.py` demonstrates construction; root `debug.py` runs the smoke suite.
Implementations and import APIs live exclusively under `lisnn/`.

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

## Run all smoke tests

```bash
python debug.py
python debug.py --verbose
python debug.py --only units indices
python -m lisnn.debugging --list
```

The default command runs seven active suites with a progress bar, reports PASS
or FAIL for each, and writes detailed JSON reports plus the neuron trace log
under a timestamped `logs/` directory. Use `--log-dir PATH` for a fixed report
location (existing files for those suites are replaced). Exit status is 0 when
all selected suites pass and 1 on failure. An exception in one suite is recorded
with its traceback and the remaining suites still run.

See [debugging documentation](lisnn/debugging/README.md) for callable APIs and
how to register more smoke tests. These checks cover the active implementation;
they do not establish correctness of future propagation/plasticity or validate
biological learning. The full regression suite remains `python -m pytest -q`.

## Import migration

The redundant root wrappers have been removed. Update active external callers:

| Previous import | Canonical replacement |
| --- | --- |
| `import Network` | `from lisnn import network` |
| `import NeuronModels as nm` | `from lisnn.neurons import kernels as nm` |
| `import debugging as debug` or `import debug` | `from lisnn import debugging as debug` |

Use `network.create_nn(...)` in place of `Network.create_nn(...)`.
`debug.py` is now an executable entrypoint, not an API re-export. The frozen
`legacy/` tree retains its historical imports and is outside this migration.

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
