# Physically interpretable fixed integrated-current propagation

Issue #18 implements the supplied P1.2 contract, with an explicit bidirectional
Izhikevich adapter approved during the subsequent discussion. This is a reduced
current-based point-synapse abstraction, not a complete biological synapse or
an extracellular electrode/MEA recording model.

## Independent propagator

```python
from lisnn.synapses import create_synapses, propagate

edges = create_synapses(3, [0, 1], [2, 2], [0.5, 0.25])
current_pA = propagate([1, 1, 0], edges, dt=0.1, impulse_scale=100)
# [0, 0, 750] pA; integrated current at N2 is 75 pA*ms.
```

For edge e, `J_e = previous_spikes[pre_idx[e]] * weight[e] * impulse_scale`
in pA*ms, and `I_e = J_e / dt` in pA. `impulse_scale` is the explicit J0
calibration, finite and nonnegative; no hidden default is chosen. One pA*ms
is one fC. Presynaptic events are strictly binary, not analog activations.
Weights stay unsigned efficacy, independent of current units or future polarity.

The implementation gathers events and uses `np.add.at` to sum edge currents
into a new float32 population vector. Fan-in adds, every outgoing edge acts
independently, and explicitly allowed duplicate contacts each contribute.
There is no normalization, fan-in/degree division, clipping, or saturation.
Self-contacts, if explicitly enabled in SynapseEdges, use the same tick delay.

`return_details=True` returns `PropagationResult` with active-edge flags,
per-edge impulse/current, and the population current. Arrays preserve caller
edge order and do not alias the input spike buffer. The propagator knows
nothing about neuron models, SNN, tasks, plasticity, or simulation clocks.

`dt` is a finite positive float32-representable value in ms. Spike values are
checked before narrowing; edge bounds/weights are revalidated because the
existing edge container is mutable. Malformed/non-finite inputs and numerical
overflow are errors, never silently repaired. Float32 precision applies;
reordering large sums may change rounding, but cannot alter causal tick order.

## Runtime and ownership

```python
from lisnn.network import create_nn, FixedWeightRuntime
from lisnn.synapses import create_synapses

model = create_nn(3, neuron_type={'default': 'LIF', 'Izhikevich': 1})
edges = create_synapses(3, [0, 1], [1, 2], [0.5, 0.25])
runtime = FixedWeightRuntime(model, edges, dt=0.1, impulse_scale=100)
result = runtime.step(external_current=[100, 0, 0])
runtime.schedule_feedback(runtime.tick, [0, 25, 0])
next_result = runtime.step()
```

The runtime copies the initial SNN population and connectivity. Its internal
edge arrays are fixed snapshots; modifying the original edges or SNN does not
modify the running simulation. The public `pool` is native model state for
explicit initialization/debugging, not a shared-unit export of every column.
Parameters and model-specific states retain their kernel meanings.

Each tick performs:

1. Validate state and arguments; read copied previous-tick spikes.
2. Propagate them into a separate synaptic-current vector.
3. Copy supplied external and feedback currents; add previously queued feedback.
4. Compose `total = external + synaptic + feedback`, all in pA.
5. Adapt input where needed and dispatch independent contiguous neuron slices.
6. Collect current binary spikes and shared-unit observations.
7. Commit state and advance spike buffers, logical tick and elapsed ms.

Initial previous spikes default to zero; an explicit initial vector is available
for controlled experiments. Spikes emitted on tick t influence targets only
on tick t+1. This is a causal scheduling approximation, not a physical axonal
or synaptic delay model. Reordering neuron-family dispatch does not introduce
within-tick propagation.

Queued feedback targets an unexecuted logical tick. Multiple contributions to
the same tick add. After observing a completed tick t, the earliest queue target
is t+1. Direct feedback passed to `step` must already be available at call start;
there are no within-tick callbacks. Queued values are consumed only after a
successful tick. No runtime state is advanced after invalid inputs or numerical
failure, including failures in a later model slice.

`TickResult` contains independent snapshots of previous/current spikes,
propagation details, all current channels, voltage before integration and after
reset, integrated `plasticity_voltage_mV` before spike reset, plus Izhikevich
observations indexed by global neuron indices. Its
`time_ms` is the interval start and `dt_ms` its width. Returned-array edits do not
modify the simulator. Persistent pre-reset traces are maintained separately
by `PlasticityTraces` when requested by a caller.

## Izhikevich input and output adapter

All runtime current channels use pA. Existing Izhikevich kernel inputs remain
native additive voltage-rate terms. At its dispatch boundary:

```text
I_native = I_total_pA / C_M_pF
I_equivalent_pA = native_rate * C_M_pF
```

For this model, `C_M` is explicitly a reference calibration capacitance, default
200 pF. It must be finite and positive. It is not an inferred membrane
capacitance in the original phenomenological equations. Record its configured
values when comparing model compositions or loading a designed connectome.
Other electrical models already consume pA and receive current unchanged.

The outgoing shared-unit observation interface is explicit:

| Observation | Meaning |
| --- | --- |
| `voltage_mV` | Membrane voltage, already in mV; copied without rescaling |
| binary `current_spikes` | AP events; never converted into current amplitude |
| `input_current_pA` (before-step adapter diagnostics) | Adapted input converted back to pA |
| `recovery_current_pA` | Signed equivalent contribution `-C_M * u` |
| `intrinsic_current_pA` | Equivalent polynomial contribution `C_M * (0.04*V^2 + 5*V + 140)` |
| `reference_capacitance_pF` | Per-neuron calibration used for both directions |

`observe_izhikevich` reports quantities at the supplied instant. The runtime
provides separate before/after observations; recovery may jump after a spike.
The two equivalent intrinsic contributions are diagnostic quantities, not
measured ionic currents, and are never fed back as additional runtime inputs.
Voltage jumps from reset must not be interpreted as current by taking a finite
difference. Native `u` stays in the kernel's state; converting it in place would
break the equations.

Outgoing propagation remains binary event -> outgoing efficacy -> J0 -> pA,
regardless of the source model. A native input or recovery value never becomes
an outgoing spike amplitude. If SI amperes are needed externally, convert the
labelled pA observations explicitly by multiplying by 1e-12.

For each event, the input-induced Euler increment is `dt * (J/dt)/C = J/C`.
Full nonlinear trajectories and spike timing need not match at different dt,
nor do matched injected charges guarantee matched firing rates across models.

## Demonstration and checks

```bash
python -m examples.fixed_weight_propagation
python -m lisnn.debugging.propagation_smoke
python -m pytest -q
```

The demonstration prints every requested current channel, prior/current spikes,
active edges, edge impulses/currents and membrane voltages. It uses dt=0.1 ms,
J0=100 pA*ms and the supplied four-node graph. N2 is explicitly initialized to
-50.1 mV, making the second hop visible without changing model parameters or
using hidden stabilization. A 32000 pA external pulse makes N0 fire on tick 0;
N2/N3 receive 500/800 pA on tick 1; N2 fires and N3 receives 400 pA on tick 2.

Tests cover empty/disconnected paths, no spikes, fan-out, fan-in, duplicates,
impulse invariance, malformed inputs, finite state, all nine model families,
Izhikevich round-trip observations, deterministic causal timing, buffer isolation,
feedback scheduling and failed-tick rollback.

## Recorded future work, not implemented

After the P1 playground matures, consider a causal temporal PSC with normalized
integral, such as `I(t) = (J/tau_syn) * exp(-t/tau_syn)`. A discrete implementation
must preserve the intended integral, rather than assuming point samples do so.
Difference-of-exponentials/alpha kernels and later transmitter/receptor
conductance models are separate refinements requiring explicit choices and
validation. No physical time constant or biological substrate is selected here.

Edge-specific physical delays can later replace the one-tick arrival rule.
Plasticity, homeostasis, inhibition/receptor identity, STP, morphology and
extracellular diffusion are not introduced by this step.

Electrode/MEA voltage and SNR require a separate extracellular forward model,
geometry/conductivity, electrode transfer function, filtering and noise. Internal
pA currents or equivalent Izhikevich contributions are not electrode voltages.
