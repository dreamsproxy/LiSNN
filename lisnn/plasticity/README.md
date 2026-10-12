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

## Pair STDP reference rule

`PairSTDP` implements the additive, all-to-all pair rule in Gerstner et al.,
[Neuronal Dynamics, section 19.2.2, equations 19.10–19.14](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html).
It is a control learner, independent of the future voltage primary rule. For
edge `i -> j`, a pre event leaves trace `x_i`, a post event leaves trace `y_j`:

```text
x_old' = exp(-dt/tau_plus_ms) * x_old
y_old' = exp(-dt/tau_minus_ms) * y_old
LTP_ij = a_plus * x_old'[i] * post[j]
LTD_ij = -a_minus * y_old'[j] * pre[i]
w_ij' = clip(w_ij + LTP_ij + LTD_ij, 0, w_max)
x_new = x_old' + pre; y_new = y_old' + post
```

All spikes observed within one tick share its ending time. Traces decay over
the interval before that tick's events are matched; simultaneous events do
not pair with each other. Each pre/post event can pair with all earlier
opposite-cell events. `a_plus` and `a_minus` are positive efficacy increments
per isolated pair, `tau_*` and `dt` are in ms, weights remain unsigned and
dimensionless. There is no factor of `dt` multiplying discrete spike updates;
the elapsed time appears in the exponential. Finite exponential tails are
retained: an isolated pair at a 200 ms separation with 10 ms tau changes
weight by less than `1e-7` for amplitude `0.1` (float32 may round that change
to zero). Weights saturate at hard bounds after the combined LTP/LTD event.

Call `PairSTDP.step(spikes, dt_ms)` independently to inspect per-edge LTP,
LTD, applied bounded change and before/after weights. `current_edges()` gives
an owned edge snapshot for a later propagation tick; `FixedWeightRuntime`
still takes a fixed snapshot and does not integrate this learner until #23.
Run `python debug.py --only pair_stdp` for the hand-checkable three-cell
trajectory.

## Triplet STDP reference rule

`TripletSTDP` follows Pfister and Gerstner,
[Triplets of Spikes in a Model of Spike Timing-Dependent Plasticity (2006)](https://www.jneurosci.org/content/26/38/9673),
full four-trace, all-to-all formulation. At each interval, decay the existing
fast pre trace `r1` by `tau_plus_ms`, fast post trace `o1` by `tau_minus_ms`,
slow pre trace `r2` by `tau_x_ms`, and slow post trace `o2` by `tau_y_ms`.
For each edge `i -> j`, after decay and before adding the current events:

```text
pair LTP     =  a2_plus  * r1[i] * post[j]
pair LTD     = -a2_minus * o1[j] * pre[i]
triplet LTP  =  a3_plus  * r1[i] * o2[j] * post[j]
triplet LTD  = -a3_minus * o1[j] * r2[i] * pre[i]
w' = clip(w + all four components, 0, w_max)
```

Then add current pre spikes to `r1,r2` and current post spikes to `o1,o2`.
An initial simultaneous pre/post pair contributes zero; earlier events can
still affect both updates on a later simultaneous tick. All four amplitudes
are nonnegative efficacy increments, and all time constants are positive ms.
The rule keeps the four components inspectable and is not combined with
`PairSTDP` or a voltage rule. `a3_plus=a3_minus=0` reduces it to the Pair rule
when pair amplitudes and fast time constants match. A pre-post-post sequence
adds triplet LTP on the second post; post-pre-pre adds triplet LTD on the second
pre. A pre-pre-post sequence adds two pair LTP terms, but no false triplet
component without earlier post history. `current_edges()` returns an owned
snapshot to feed into later propagation.
