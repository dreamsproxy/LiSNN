# LiSNN Spatial Substrate

`lisnn.spatial` is the common physical coordinate frame for the long-horizon
biophysical branch of LiSNN.

## Milestone 1 scope - 2026-09-08

The current implementation is deliberately geometry-only.

It provides:

- an optional axis-aligned `SpatialVolume`;
- physical coordinates in micrometres (`um`);
- one `(x, y, z)` soma-center coordinate per neuron;
- deterministic uniform placement from the network seed;
- explicit user-supplied soma placement;
- closed-volume boundary metadata;
- an inactive vectorized morphology table interface for future compartments.

It does **not** currently change neuron dynamics, synaptic propagation,
plasticity, or connectivity. Voltage-based STDP remains independent of space.

## Coordinate contract

```text
space.positions.shape == (N, 3)
space.positions.dtype == float32
space.positions[n]     == soma center of neuron n
units                  == micrometres
```

Soma coordinates must never be reinterpreted as dendritic or axonal paths.
Future morphology is represented separately as compartment/path geometry.

## Why space is first-class

The eventual simulator needs one shared frame in which multiple physical
systems can coexist:

```text
SpatialVolume
|-- soma coordinates
|-- dendrite geometry
|-- axon geometry
|-- nodes of Ranvier / myelination geometry
|-- grey- and white-matter regions
|-- extracellular-space geometry
|-- extracellular ion concentration fields
|-- free neurotransmitter / neuromodulator fields
|-- other extracellular solute fields
|-- probes / electrode contacts
|-- vasculature / CSF / clearance geometry
`-- future tissue boundaries and interfaces
```

Keeping this separate from the neuron-state matrix preserves the existing
`(N, NEURON_WIDTH)` electrophysiology ABI while allowing spatial resolution to
increase independently.

## Future field model

Extracellular chemistry should eventually be represented as fields over space,
not as extra columns attached to individual neurons. Conceptually:

```text
C_species(x, y, z, t)
```

Initial ionic species of interest include:

- Ca2+
- Na+
- K+

Later phases may add free extracellular neurotransmitters, neuromodulators,
and other substances as independent species with their own source, sink,
diffusion, uptake, binding, and clearance behavior.

No such field solver exists yet.

## Future morphology model

The reserved `MorphologyTable` uses vectorized arrays for future compartments:

```text
owner_neuron
parent
position[x, y, z]
radius
compartment_type
```

Reserved compartment labels currently include soma, dendrite, axon, axon
terminal, and node of Ranvier. Their existence is an interface commitment, not
a statement that those structures are currently simulated.

## Design rule

**Do not add approximate biophysics merely to make the spatial layer appear
more complete.** Add geometry first, then introduce each physical mechanism as
an independently testable subsystem with explicit units and conservation
assumptions.
