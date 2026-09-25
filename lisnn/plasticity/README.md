# P1.2 observation boundary

Every neuron kernel still returns a float32 spike vector by default. Pass
`return_observation=True` to receive `NeuronObservation(spikes,
plasticity_voltage_mV)`. The latter records the integrated voltage before a
spike-triggered reset, even for GLIF kernels whose reset equation uses the
beginning-of-tick voltage. A cell refractory at tick start reports its actual
post-step held/reset voltage, rather than a discarded integration candidate.
Izhikevich observations are in mV; `FixedWeightRuntime` adapts pA input before
the native kernel call and exposes both global pA current and mV voltage.

`FixedWeightRuntime.step()` adds `TickResult.plasticity_voltage_mV` alongside
existing pre-step `voltage_before_mV` and post-reset `voltage_mV`. All three
arrays and the spike vector are separate owned snapshots. Tick `t` consumes
the previous spike vector to propagate, integrates every slice, then reports
current events and observations. No weight changes occur in this runtime.

The independently usable `PlasticityTraces` keeps float32 state with
`x_new = exp(-dt/tau_spike) * x_old + spike_current` and
`u_new = exp(-dt/tau_voltage) * u_old +
(1 - exp(-dt/tau_voltage)) * plasticity_voltage_current`.
All times and time constants are in ms. `advance` reports the state after both
decay and same-tick events; read `spike_trace` and `voltage_trace_mV` before
calling it if a rule needs the old state. Co-occurring pre/post spikes have
the same time; no order or weight-update convention is implied here. Histories
are off by default. Specify both `history_ms` and `history_max_samples` to
retain snapshots within an elapsed-time horizon and a hard sample count. The
returned snapshots, readback history, and inputs cannot mutate stored state.

Run `python debug.py --only plasticity_observation` for the independent smoke
check. The rule-specific old/new ordering and weight-update schedule belong
to subsequent learner implementations.
