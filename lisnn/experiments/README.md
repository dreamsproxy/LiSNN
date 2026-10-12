# Controlled P1.2 state-cycle example

`python -m examples.controlled_cycle` runs a small, deterministic experiment.
Use `python -m examples.controlled_cycle --json cycle.json` to export complete
per-tick observations and configuration to a chosen local path.
It has one directed LIF connection, initial unsigned efficacy `0.5`, a
selected Pair STDP learner, `dt=1 ms`, and two explicitly mapped EXTERNAL
training frames: auditory input to the source cell at tick 0, visual input to
the target cell at tick 1. A bounded recorder retains the two delivered pA
samples with their timestamps, ports and mapping provenance. Pair/Triplet/
Voltage learners are independently selectable in `SNN.configure_runtime`;
the controlled example picks Pair so its two-event expectation can be checked
by hand. This experiment is an engineering probe, not DishBrain emulation or
a biological recall claim.

The experiment waits 100 quiet ticks after experience. It snapshots neuron
state, synapses, learner traces, clock, queues, regime/RNG state and replay
buffer. The baseline recall probe sends a direct current to the source cell
and reads target spikes, voltage and next-tick synaptic pA. It freezes
plasticity and restores the *same* post-experience snapshot afterward, so
the first probe cannot change any later starting state. Every branch begins
from that saved snapshot:

| Branch | SLEEP stimulation | Plasticity during interval |
| --- | --- | --- |
| quiet | none | configured learner on |
| ordered | literal auditory then visual | on |
| shuffled | same two samples, same tick slots and gain, reversed by seeded shuffle | on |
| ordered_learning_off | same ordered samples | weights frozen; traces still evolve |

The initial SLEEP tick is quiet. Ordered/shuffled replay enters only later
through separate INTERNAL ports. After six SLEEP ticks the controller returns
to ACTIVE; every arm then waits another 100 quiet ticks with weight updates
frozen before its second probe. The declared response metric is target
synaptic pA one tick after the direct source probe; target spike/no-spike is
also returned. Each arm reports both final response and signed difference
from baseline, weights after SLEEP, delivery source/port/tick/provenance, and
all raw SLEEP spikes, currents, voltages, trace vectors and per-edge update
components. The example prints branch progress for longer runs. Pass a
`progress` callback to `run_controlled_cycle` to route progress elsewhere.

In the fixed default configuration, ordered replay raises the later probe's
synaptic current, shuffled replay lowers it, and quiet/learning-off controls
produce no change. These are properties of this chosen tiny pattern and
parameters, not a required scientific result. The experiment retains null
and negative outcomes; change the inputs, model composition or parameters
before drawing conclusions about persistence or generalization. A 100 ms
washout reduces transient activity but does not prove long-term memory.
Run `python debug.py --only controlled_cycle` for the small engineering gate.
