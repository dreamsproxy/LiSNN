# Spatial TODO

This file preserves the long-range spatial/biophysical development horizon so
work can resume without reconstructing the architecture from chat history.

## Phase 1 - Spatial shell

- [x] First-class `SpatialVolume`.
- [x] Micrometre coordinate convention.
- [x] Soma-center `positions[N, 3]` array.
- [x] Deterministic uniform placement.
- [x] Explicit placement validation.
- [x] Closed-boundary metadata.
- [x] Optional integration with `SNN` / `create_nn()`.
- [x] Reserve morphology array interface.
- [ ] Add spatial debug/visualization helper when needed.

## Phase 2 - Morphology and physical wiring

- [ ] Define soma radius/volume representation.
- [ ] Generate/import dendritic trees.
- [ ] Generate/import axonal paths.
- [ ] Represent axon terminals and synaptic contact coordinates.
- [ ] Represent myelinated segments and nodes of Ranvier.
- [ ] Measure path length rather than Euclidean soma distance for conduction.
- [ ] Add physical conduction delay derived from path/axon properties.
- [ ] Add collision/packing constraints.
- [ ] Quantify maximum feasible wiring density for a specified tissue volume.
- [ ] Separate grey-matter and white-matter spatial regions.
- [ ] Preserve support for imported biological morphologies.

## Phase 3 - Tissue and probe geometry

- [ ] Define tissue-region masks/material properties.
- [ ] Add extracellular-space volume fraction and tortuosity fields.
- [ ] Add probe/electrode contact geometry: position, orientation, radius,
      length/contact surface.
- [ ] Compute distance/orientation relationships between compartments and probes.
- [ ] Introduce an extracellular-potential forward model only after current
      source assumptions are explicit.
- [ ] Support virtual probe placement at arbitrary coordinates.

## Phase 4 - Extracellular ionic fields

- [ ] Define voxel/adaptive-grid field API with explicit physical units.
- [ ] Add extracellular Ca2+ concentration field.
- [ ] Add extracellular Na+ concentration field.
- [ ] Add extracellular K+ concentration field.
- [ ] Add membrane source/sink interfaces without coupling them to a specific
      ion-channel model prematurely.
- [ ] Add diffusion with explicit boundary conditions.
- [ ] Add uptake/pump/buffer interfaces.
- [ ] Add glial buffering/transport only after its model assumptions are chosen.
- [ ] Validate conservation and numerical stability independently from SNN
      learning experiments.

## Phase 4/5 - Free neurotransmitters and other extracellular substances

- [ ] Generalize extracellular fields to arbitrary chemical species.
- [ ] Add free neurotransmitter species only after release, uptake, degradation,
      and receptor-coupling assumptions are specified.
- [ ] Add neuromodulator species as independently configurable fields.
- [ ] Add other extracellular substances as separate species rather than a
      generic scalar "chemical" field.
- [ ] Allow different diffusion coefficients, binding, uptake, and decay per
      species.
- [ ] Couple synaptic terminal geometry to local release sites.

## Phase 5+ - Clearance, metabolism, and state cycles

- [ ] Add interstitial/CSF transport geometry.
- [ ] Add macromolecular/interstitial solute fields such as tau only as separate
      species with explicit transport/clearance assumptions.
- [ ] Investigate state-dependent clearance/sleep-cycle abstractions.
- [ ] Add vascular/metabolic support if required by the selected resolution.
- [ ] Keep ionic homeostasis distinct from macromolecular waste clearance.

## Long-horizon constraints

- Spatial physics must remain optional so non-spatial subexperiments stay cheap.
- New physical mechanisms require explicit units.
- Do not place morphology or extracellular state into the neuron `(N, P)` ABI.
- Prefer contiguous NumPy arrays and sparse/indexed geometry over Python object
  graphs in hot loops.
- Profile before introducing Numba or other optimization layers.
- Every new physical subsystem should have an independent validation test before
  it is coupled to learning/plasticity.
