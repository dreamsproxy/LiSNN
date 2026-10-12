# LiSNN

LiSNN is an experimental toolkit for constructing, stimulating, observing and
adapting heterogeneous spiking neural substrates. Its Phase 1 goal is a usable
playground for DishBrain-inspired controlled adaptation experiments. Task
semantics and feedback decisions belong to external experiments.

## Start here

Use Python 3.12 from the repository root. The active framework needs NumPy.

```bash
python -m pip install -r requirements/core.txt
python -m examples.playground --plasticity pair --steps 200
python debug.py
```

For regression tests and development:

```bash
python -m pip install -r requirements/test.txt
python -m pytest -q
python debug.py --verbose
python -m lisnn.debugging --list
```

`debug.py` runs every registered smoke suite (currently 19), prints progress,
and returns a nonzero exit status on failure. Diagnostics and aggregate JSON
are saved under `logs/`, or a chosen `--log-dir PATH`. To isolate a component,
use `python debug.py --only propagation`. See the [debugging guide](lisnn/debugging/README.md).

`requirements/legacy.txt` preserves the old dependency inventory for archived
experiments. `requirements/media.txt` and `requirements/visualization.txt` are
optional future/example dependencies. `requirements/performance.txt` lists
optional performance packages; none is required by the current NumPy runtime.
Live microphone/camera adapters are future work. ASCII input needs no media library.

## Available now

| Module | Current capability |
| --- | --- |
| `lisnn.neurons` | Nine vectorized neuron families; shared float32 `(N, 43)` pool; pre-reset observations |
| `lisnn.network` | Homogeneous/mixed construction; explicit causal stepping; learner selection; snapshot/reset |
| `lisnn.spatial` | Optional geometry shell, micrometre coordinates and morphology interface reservation |
| `lisnn.synapses` | Sparse directed edges, unsigned efficacy and integrated-current propagation |
| `lisnn.plasticity` | Independent Pair, Triplet and voltage-gated STDP alternatives; bounded optional trace history |
| `lisnn.io` | Declared ports, current transduction, probes, causal feedback, ACTIVE/SLEEP policies and bounded literal replay |
| `lisnn.media` | Fixed seeded ASCII character embeddings into bounded pA current patterns |
| `lisnn.experiments` | Small controlled ACTIVE/SLEEP/ACTIVE engineering example |
| `lisnn.debugging` | Independently callable smoke suites and aggregate runner |
| `benchmarks` | Component/scaling profiles, reference comparisons and numerical/memory regression gates |

Supported kernels: LIF, Adaptive LIF, Izhikevich, AdEx, GLIF3, GLIF4, GLIF5,
CAdEx, and the experimental CAdEx-GLIF hybrid. Numerical model identity is
separate from biological cell taxonomy, port membership and spatial geometry.

## Construct, stimulate, step and observe

```python
import numpy as np
from lisnn import create_nn
from lisnn.synapses import create_synapses

model = create_nn(4, neuron_type={"default": "LIF", "Izhikevich": 1}, seed=7)
edges = create_synapses(4, [0, 1, 2], [1, 2, 3], weights=0.1)
model.configure_runtime(
    edges, dt=0.1, impulse_scale=100,
    plasticity="pair",
    plasticity_params=dict(tau_plus_ms=20, tau_minus_ms=20,
                           a_plus=0.001, a_minus=0.001, w_max=1),
)
current_pA = np.array([600, 0, 0, 0], dtype=np.float32)
observation = model.step(current_pA)
print(observation.current_spikes)
print(observation.plasticity_voltage_mV)  # Integrated voltage before reset.
print(observation.voltage_mV)             # Applied post-step voltage.
print(observation.weights_after)
```

`configure_runtime` copies the construction pool and edges. Inspect the evolving
state through `model.runtime.pool` and `model.runtime.weights`; `model.pool`
remains the construction state. Choose exactly one of `off`, `pair`, `triplet`
or `voltage`, with that learner's explicit parameters. The
[editable playground](examples/playground.py) shows all four choices.

A tick propagates the previous spikes with its starting weights, sums declared
pA currents, integrates each contiguous model slice, captures spikes/pre-reset
mV, and applies the selected local learner. New weights affect later ticks.
Task answers and outcome errors never become supervised weight deltas.
See the [complete tick and ownership contract](docs/TIMESTEP.md),
[neuron units](lisnn/neurons/UNITS.md), [propagation contract](lisnn/synapses/PROPAGATION.md),
and [plasticity rules](lisnn/plasticity/README.md).

Izhikevich has an explicit input/output adapter: global pA input maps to its
native drive through `C_M`; membrane observations return mV and equivalent
intrinsic/recovery contributions return pA. Native recovery state and binary
spikes keep their meanings. The calibration is not a measured biophysical model.

## Small editable experiments

Run these from the repository root:

| Command | Inspect or edit |
| --- | --- |
| `python -m examples.foundation_contracts` | Hand-checkable electrical units and neuron integration |
| `python -m examples.fixed_weight_propagation` | Three-tick causal pathway and integrated impulse |
| `python -m examples.playground --plasticity triplet --steps 200` | Mixed neurons, sparse connectivity, currents, learner selection and progress |
| `python -m examples.recurrent_learning` | Two-cell learning-on/off comparison |
| `python -m examples.stream_routing` | Explicit sensory channels and subsequent feedback |
| `python -m examples.active_sleep` | Declared regime transitions and internal perturbations |
| `python -m examples.internal_replay` | Bounded recorded samples through internal ports |
| `python -m examples.controlled_cycle --json logs/cycle.json` | Same-snapshot quiet/ordered/shuffled/learning-off arms with raw observations |
| `python -m examples.character_input` | ASCII synthetic medium and an explicit experiment-side presentation duration |

The character medium accepts ASCII IDs 0..127. It supplies spatial current
patterns, not an intrinsic clock or token-per-tick policy. How long a pattern
is presented remains an experiment choice. It has no trained linguistic
semantics, decoder or language-training objective. See [media contracts](lisnn/media/README.md).

The small state-cycle example verifies engineering controls. Broader adaptation,
recall, transfer and biological claims need the selected experiments and evidence
in [#66](https://github.com/dreamsproxy/LiSNN/issues/66). Literal replay records
stimulation; it does not implement endogenous reconstruction or imagination.

## Numerical and performance checks

```bash
python -m benchmarks.regression_gate --output logs/regression-gate.json
python -m benchmarks.profile_playground --sizes small --output logs/profile.json
```

The gate compares reference and optimized NumPy topology validation under the
same seeds, sizes, edge orders and integration steps. CI enforces correctness
and practical memory checks. Timing is recorded; controlled-host timing
thresholds are opt-in. See [benchmark policy and results](benchmarks/README.md).

For an exploratory session using the reference path:

```python
from lisnn.synapses import topology_validation

with topology_validation("reference"):
    observation = model.step(current_pA)
```

Set `LISNN_TOPOLOGY_VALIDATION=reference` before starting Python to select it
for a whole process. Both paths validate inputs and preserve edge accumulation
order. This selector changes topology checking, not the neuron or learning equations.

## Ownership, provenance and roadmap

Active implementation and import APIs live under `lisnn/`. Root `main.py` is a
construction example; `debug.py` is the smoke CLI. The old root `Network.py`,
`NeuronModels.py` and `debugging.py` were removed in #81.

| Previous import | Current import |
| --- | --- |
| `import Network` | `from lisnn import network` |
| `import NeuronModels as nm` | `from lisnn.neurons import kernels as nm` |
| `import debugging as debug` or `import debug` | `from lisnn import debugging as debug` |

Historical LiSNN/LiNN/BioEmulator experiments remain frozen under `legacy/`.
Their dependency inventory and assumptions do not define the active runtime.

- [Program #62](https://github.com/dreamsproxy/LiSNN/issues/62)
- [Phase 1 #63](https://github.com/dreamsproxy/LiSNN/issues/63): playground, conditional performance work, controlled DishBrain-inspired experiments
- [Phase 2 #64](https://github.com/dreamsproxy/LiSNN/issues/64): future cellular, synaptic and extracellular realism
- [Phase 3 #65](https://github.com/dreamsproxy/LiSNN/issues/65): future biological taxonomy, network import and connectome compatibility

All original M1-M7 work belonged to Phase 1. The [spatial TODO](lisnn/spatial/TODO.md)
uses separate local development stages; these are not additional project phases.
Ionic fields, chemical transport, detailed morphology and biological import
remain deferred. Milestone organization remains a separate repository task.
