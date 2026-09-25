# Generic stream and interface contract

`StreamFrame` declares `source` (`EXTERNAL` or `INTERNAL`), `kind`
(`CONTINUOUS`, `EVENT`, or `MODULATORY`), free-form explicit `modality`,
channel/time axes, timestamp in ms, optional sample period, and mapping
provenance. Its float32 values have a declared channel axis. A two-dimensional
frame must declare `("time", "channel")` or `("channel", "time")` and a sample
period; `.sample(index)` selects one sample and preserves source, modality,
provenance, and its computed timestamp. It rejects negative/out-of-range
indices. `ContinuousStream`, `EventStream`, and `ModulatoryStream` hold ordered,
nonoverlapping frames from one source. Modulatory frames require explicit
destination and effect; neither is inferred from value shape.

`InterfacePort` references ordinary neuron indices with an explicit source,
input/readout role, optional biological identity, and currently one-to-one
channel mapping. Roles can overlap in membership. Its capacity check reports
required/available channels, input structure, mapping policy and an explicit
remedy; no channels are truncated, wrapped, resized, or compressed.
`Transducer.to_current(frame)` maps a single sample into a float32 population
current vector. Continuous values are already pA; binary events need an
explicit `event_amplitude_pA`. Modulatory values can become currents only
when they declare destination `external_current` or `feedback_current` and
effect `add_pA`. Other effects remain metadata for an experiment-specific
consumer, never disguised as current injection.

`StreamScheduler(max_frames)` holds a bounded FIFO of frames keyed by explicit
delivery ticks. An internal frame must give the tick it was observed and may
only be delivered on a later tick. The producer must declare any timestamp to
tick/rate mapping before scheduling; the scheduler never silently rounds
physical time. `Probe(port).capture(tick_result)` copies raw spikes, voltage,
pre-reset voltage, per-channel current, and optional learning diagnostics;
it cannot change neuron, synapse, or trace state by modifying the result.

These independent components establish the transport boundary for #71
external routing, #75 regimes and #76 internal replay. They do not infer a
semantic modality from shape, decode tasks, stimulate receptors implicitly,
or assign InputNeuron/OutputNeuron biological classes. The runtime continues
to accept direct per-neuron pA vectors without using this package. Run
`python debug.py --only io_contracts` for a small deterministic check.
