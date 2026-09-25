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

## External routing and outcome feedback (#71)

`StreamRuntime(configured_snn, transducers={port_name: Transducer(port), ...})`
maps declared sample timestamps to the configured `dt` on an exact-tick grid
(within float32 representation tolerance). Off-grid timestamps require
experiment-side resampling; no implicit floor, nearest-sample choice, or
interpolation happens. `schedule_frame` maps a one-sample frame or validates
and schedules each sample of a time-series frame atomically, with declared
sample-period and channel-axis semantics. Ports remain source-specific.

At each `StreamRuntime.step(direct_current)` the ordered due frames become
per-neuron pA currents, summed with the direct current and passed to the
ordinary `SNN.step`. Modulatory frames may enter the feedback-current channel
only when destination/effect allow it. `RoutedTick` includes the neural result
and per-frame `Delivery` logs: source, port, kind, modality, delivery tick,
original timestamp, mapping/outcome mode, count of stimulated neurons and
sum of absolute pA. Failed stepping restores due frames to the queue.

`FeedbackPolicy` lives at this experiment-facing boundary. It supports
contingent success stimulation, seeded unpredictable error/no-response
perturbation, no-feedback control, random independent success selection,
predeclared yoked currents and inverted contingency. Its `schedule` method
uses `StreamRuntime.schedule_feedback` and requires a delay of at least one
tick from the observed outcome. An outcome is an experiment-defined boolean;
there is no task target/error in the neuron, synapse, or learner. Pass a
separate internal port even when its neuron membership overlaps an external
sensory port. A snapshot copies network state, queue and delivery logs;
for controlled experiments also snapshot the separate feedback policy's RNG
state (e.g. with `copy.deepcopy`).

`python -m examples.stream_routing` demonstrates two-channel microphone
samples, two-site visual current, binary sparse events, and a subsequent
closed-loop perturbation without a hardware dependency. A fixed frame
schedule replays deterministically from the same initial network snapshot.
