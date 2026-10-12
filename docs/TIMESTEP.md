# Tick, units and ownership contract

This document closes the common contract in issue #68. Detailed model equations
remain in [neuron units](../lisnn/neurons/UNITS.md),
[propagation](../lisnn/synapses/PROPAGATION.md) and
[local learning](../lisnn/plasticity/README.md).

## Units and one tick

Neural time is ms, voltage mV, current pA, capacitance pF, conductance nS and
resistance MOhm. Neuron state and current vectors are float32. Resistance is
converted to GOhm where mV / resistance must yield pA. Stream timestamp and
sample-period metadata retain float64 precision. Timestep must remain finite,
positive and representable as a positive float32 value at the kernel boundary.

For a directed edge i -> j, a binary spike emitted on the previous tick
contributes `spike[i] * weight[i,j] * J0 / dt` pA now. `J0` is pA*ms; efficacy is
nonnegative and dimensionless. Fan-in adds without normalization. Holding J0
fixed preserves the rectangular impulse integral when dt changes; it does not
guarantee identical Euler trajectories, event times or learning trajectories.

At tick t:

1. Collect previously scheduled external/internal/background and feedback input.
2. Propagate spikes from t-1 using weights available at the start of t.
3. Sum the separately observable external, synaptic and feedback current channels.
   Internal/background/replay deliveries retain source and mode provenance in
   routing logs, and currently enter the feedback-current aggregate.
4. Integrate each contiguous neuron-model slice with the shared dt.
5. Capture current spikes and integrated pre-reset `plasticity_voltage_mV`;
   separately expose pre-step and applied post-step/reset voltage.
6. Evaluate the one selected learner using the conventions below and commit its
   bounded efficacy update. Those new weights first affect later propagation.
7. Return owned observations. Experiment-side probes/outcomes may schedule
   consequences only after an executed observation and for a later tick.

`TickResult.tick` and `time_ms` identify the beginning of the integration
interval. Emitted spikes share its ending time for local event matching.
No same-tick output recursively stimulates the tick that produced it.
A refractory cell's plasticity voltage is its actually applied held/reset
voltage, not an integration candidate discarded by the kernel.

## Learner ordering

| Learner | Update uses | State advanced afterward |
| --- | --- | --- |
| Off | Fixed efficacy | Neuron/clock state only |
| Pair | Decayed old pre/post traces and current events; current simultaneous events do not pair together | Add current spikes to the two traces |
| Triplet | Four decayed old traces and current events; pair and triplet terms are independently observable | Add current spikes to fast/slow pre/post traces |
| Voltage | Decayed old presynaptic trace, previous voltage filters and current pre-reset voltage; LTD is event-based, LTP includes dt | Add current presynaptic events and update voltage filters |

Pair, Triplet and Voltage are alternative learners, not additive defaults.
`plasticity_enabled=False` freezes efficacy while the selected learner's
history continues evolving; `plasticity="off"` creates the fixed runtime.
Trace helper voltage filtering uses cancellation-safe exponential coefficients
with a final float32 state cast. Very small changes can still round at the
float32 storage boundary; no hidden clipping or normalization is introduced.

## Izhikevich boundary

The native kernel remains explicitly phenomenological. Runtime input converts
`I_native = I_pA / C_M_pF`. Observations expose global mV and pA, including signed
equivalent intrinsic/recovery currents scaled by the same C_M. Native recovery
variables remain native internal state; spike events remain binary. This is a
calibrated common interface, not evidence of biological electrical equivalence.

## Ownership, clocks and schedules

- `configure_runtime` copies construction arrays and edges. Runtime state is
  independent of `SNN.pool`; returned observations/weights are owned copies.
- Snapshots include neuron state, topology, weights, traces, clock and queued
  feedback. Restore accepts the same configuration lineage, including a deep
  copy for controlled experimental arms. An unrelated network or a new
  `configure_runtime` call cannot silently import another configuration.
- Stream routing snapshots additionally retain ports, queues and delivery logs;
  regime snapshots include transition policy and RNG state. Separately snapshot
  experiment-owned feedback policies and replay controllers when needed.
- Scheduling validates the whole frame before enqueueing. Same-tick scheduled
  entries preserve FIFO submission order. Internal frames must name an executed
  observation tick. Timestamp mapping tolerates floating representation error,
  capped at 0.001 tick; materially off-grid inputs need explicit resampling.
  Use the runtime's effective dt when constructing long schedules.
- Input/readout roles are enforced: transducers accept input/overlap ports;
  probes accept readout/overlap. Shared neuron membership remains allowed.
- Literal replay is checked against SLEEP both at scheduling and delivery. A
  conflicting later ACTIVE transition raises before advancing or consuming the
  due replay. Shuffled replay draws a non-identity permutation and requires at
  least two entries; identical payloads may still make an order contrast inert.
- Failed neural/adaptive steps preserve clock and working state. Failed routed
  ticks restore due inputs; regime failures restore policy/RNG state as well.

Trace histories and replay buffers have explicit capacity/time bounds. Frame
queues have explicit capacity. Delivery audit logs and caller-held snapshots
can grow with run length; these bounds do not promise constant total memory for
an indefinitely recorded experiment. Long runs must choose recording budgets.

## Hand checks and regression evidence

```bash
python -m examples.foundation_contracts
python -m examples.fixed_weight_propagation
python -m examples.recurrent_learning
python debug.py
python -m pytest -q
```

Tests cover all nine neuron families, adapters, current composition, prior-spike
causality, pre-reset observations, rule ordering, invalid inputs, ownership,
snapshot lineage, fractional timestamps, future feedback and replay boundaries.
The controlled-cycle example records every SLEEP tick, including its initial
quiet transition tick, with frozen evaluation probes and identical branch starts.
