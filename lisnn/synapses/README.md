# LiSNN synapse substrate

This package owns synaptic topology and edge-local state. It is intentionally
separate from neuron physiology (`pool[N, 43]`), environment I/O, spatial
geometry, propagation, and plasticity.

## M2.1 canonical representation

Each directed synaptic contact is one entry in parallel arrays:

```text
pre_idx[E]   int32
post_idx[E]  int32
weight[E]    float32
```

`weight` is an unsigned efficacy magnitude. Excitatory/inhibitory polarity is
not encoded by a negative weight; later Dale/receptor handling owns that
semantic.

The canonical path does not require a dense `N x N` matrix.

## Invariants

- Edges are directed: `i -> j` and `j -> i` are independent.
- Autapses are rejected by default and can be enabled explicitly.
- Duplicate ordered edges are rejected by default. Multiple biological contacts
  between one ordered neuron pair can be represented later by explicit contact
  state or enabled for targeted sub-experiments.
- Explicit constructors preserve caller edge order.
- Fixed-fan-out construction selects unique targets for every presynaptic neuron.
- Random topology/weight initialization is deterministic for a fixed seed.
- Synaptic storage is independent of neuron model type, so homogeneous and mixed
  populations use exactly the same edge representation.

## Public construction API

```python
from lisnn.synapses import create_synapses, create_fixed_out_degree

edges = create_synapses(
    population=4,
    pre_idx=[0, 0, 2],
    post_idx=[1, 3, 1],
    weights=[0.3, 0.5, 0.8],
)

edges = create_fixed_out_degree(
    population=1024,
    synapses_per_neuron=64,
    seed=1,
)
```

## Deliberately deferred beyond M2.1

The following are not implemented in this package yet:

- spike propagation and postsynaptic accumulation
- axonal delays
- conductance/receptor dynamics
- Dale identity and receptor classes
- short-term facilitation/depression
- Pair, Triplet, or Voltage-Based STDP
- per-edge calcium/eligibility traces
- structural plasticity
- spatial axon/dendrite paths

These features should extend the edge substrate rather than change the neuron
matrix ABI.
