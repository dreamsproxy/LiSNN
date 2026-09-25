# Neuron units and timestep contract

This specifies the independently callable neuron kernels for Phase 1 (#68).
Propagation, feedback queues, efficacy-to-current conversion, and plasticity
event ordering remain a separate discussion. This document does not complete
those parts of #68.

## Electrical units

For LIF, Adaptive LIF, GLIF3/4/5, AdEx, CAdEx, and CAdEx-GLIF:

| Quantity / columns | Unit |
| --- | --- |
| Voltage, thresholds, voltage increments and exponential widths | mV |
| `dt`, `REFRACTORY`, `T_REF`, `TAU_ADAPT`, `TAU_W`, `TAU_G_A` | ms |
| `input_current`, `ASC_1/2`, `DELTA_I_1/2`, `B_W` | pA |
| `C_M` | pF |
| `R_M` | MOhm |
| `G_L`, `A_W`, `G_A_BAR`, `DELTA_G_A` | nS |
| `B_S`, `K_1/2`, `A_V`, `B_V` | 1/ms |
| `F_V`, `F_1/2` | dimensionless |

`ADAPT` is model-specific: mV for Adaptive LIF, pA for AdEx, and nS
for CAdEx and its hybrid. `DELTA_ADAPT` is mV. Unused columns do not acquire
physical meaning merely because they are allocated in the shared matrix.

The useful numerical identities are:

```text
1 nS * 1 mV = 1 pA
1 pA / 1 pF = 1 mV/ms
1 MOhm = 0.001 GOhm
I_leak[pA] = (V - E_L)[mV] / (0.001 * R_M[MOhm])
tau_LIF[ms] = 0.001 * R_M[MOhm] * C_M[pF]
tau_AdEx_leak[ms] = C_M[pF] / G_L[nS]
```

Defaults `R_M=100`, `C_M=200`, and `G_L=10` therefore give a 20 ms
passive time constant in both formulations. AdEx also includes exponential
initiation and adaptation, so this does not imply equal full trajectories.
Randomized `R_M` and `G_L` remain independently sampled; reciprocal matching
is not imposed on heterogeneous populations.

### Numerical correction and compatibility

At main commit `a29aa0169d503db406a9e51fe8cb41e7160676a5`, resistance-based
kernels divided mV directly by the stored `R_M=100`, then divided by `C_M=200`.
The resulting literal time constant was 20,000 ms. Previous documentation said
only "model-compatible" units, so it did not establish the author's intended
physical scale. This change explicitly adopts the units above and applies the
missing MOhm conversion in LIF, Adaptive LIF, and the shared GLIF derivative.

Default values, randomized parameter ranges, matrix layout, and public function
signatures are preserved. Resistance-based voltage trajectories do change.
Old active experiments that compensated for the slower leak need recalibration.
An old saved resistance value can reproduce its former numerical leak by
multiplying that value by 1000 under this contract; this is an explicit legacy
conversion, not an automatic migration or a physiological recommendation.
The frozen `legacy/` implementation is unaffected.

### Izhikevich exception

The implemented equation is `dV/dt = 0.04*V**2 + 5*V + 140 - u + I`.
It does not divide `I` by capacitance. Time remains ms and voltage mV, but
input `I`, recovery `u` (`ADAPT`), and `IZH_D` use the equation's native scale
(additive voltage-rate terms), not the pA convention of the electrical models.
`IZH_A` and `IZH_B` carry the scaling implicit in those native equations.
The fixed-weight runtime now explicitly converts total pA input through
`I_native = I_pA / C_M_pF`. For Izhikevich, `C_M` is a reference input/output
calibration capacitance, not a claim of a physical capacitance in the original
phenomenological model. Equivalent current observations multiply native rates
by `C_M`; voltage remains mV and spikes remain binary events. See the
[propagation/runtime contract](../synapses/PROPAGATION.md). Independently called
Izhikevich kernels still receive native inputs.

## A kernel call advances one interval

`step(pool, input_current, dt)` advances continuous state with Forward Euler.
`dt` must be a real scalar, finite and strictly positive after float32
conversion. Zero, negative values, NaN, infinity, float32 overflow/underflow
to zero, and arrays are rejected. Boolean, string, and complex inputs are
rejected rather than coerced into timesteps.

`input_current` is a real scalar or a vector of shape `(N,)`, interpreted as
current amplitude held over this interval. It must remain finite in float32.
Invalid timestep/current arguments raise `TypeError` or `ValueError` before
any neuron state, including refractory timers, is mutated. Scalar current is
broadcast. Signed currents are allowed; no clipping or normalization is added.

The caller supplies one total input current. Built-in model currents such as
adaptation and after-spike currents are accounted for by the model itself.
Input stimulation enters the derivative; it does not assign membrane voltage.
Initialization or explicitly setting state for a controlled test is separate.

The existing refractory convention is retained: a neuron refractory at interval
start stays refractory throughout that call, even if its timer reaches zero.
Its next call may integrate voltage. Auxiliary-state evolution and reset rules
remain model-specific. Kernels mutate the supplied population in place and
return a new float32 `(N,)` vector of 0/1 spike events. They do not own a global
clock or expose a new pre-reset observation API.

Finite positive `dt` validates the argument, not numerical stability. Model
parameters must be suitable for the chosen experiment; no new parameter
clamping or guarantee against trajectory overflow is introduced. Use timestep
refinement to check integration error and spike-time discretization.

## Current duration versus timestep

For an electrical-model input, rectangular current exposure has integral
`I[pA] * duration[ms]` in fC. Holding 100 pA for 1 ms injects 100 fC, whether
represented by one 1 ms interval or ten 0.1 ms intervals. Euler trajectories
can differ because the state is reevaluated at each interval.

A current supplied for only one call has duration `dt`; changing `dt` therefore
changes that call's current integral. Preserving a physical pulse requires
preserving its duration and amplitude (and hence an appropriate step count).
This observation does not select a synaptic pulse shape, width, or gain.

## Hand calculation

With `E_L=-65 mV`, `V=-55 mV`, `R_M=100 MOhm`, `C_M=200 pF`, and no input:

```text
I_leak = 10 / 0.1 = 100 pA
dV/dt = -100 / 200 = -0.5 mV/ms
V_after_1_ms = -55.5 mV
```

At rest, 100 pA instead gives `V_after_1_ms=-64.5 mV`, below threshold.
Run the independent example from the repository root:

```bash
python -m examples.foundation_contracts
python -m pytest -q
```

## Reserved for the propagation discussion

Issue #18 now defines fixed integrated impulses, positive synaptic effects,
one-tick causal scheduling, channel ownership and feedback queues in the
[propagation/runtime contract](../synapses/PROPAGATION.md). Weights remain
unsigned efficacy, not pA. Receptor polarity, learning-rule ordering and
pre-reset plasticity observations remain future work.
