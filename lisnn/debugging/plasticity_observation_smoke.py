"""Callable #19 pre-reset voltage and trace smoke diagnostics."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.network import create_nn
from lisnn.network.runtime import FixedWeightRuntime
from lisnn.neurons import kernels as k
from lisnn.neurons.registry import NeuronType, get_step_function
from lisnn.plasticity import PlasticityTraces
from lisnn.synapses import create_synapses


def plasticity_observation_smoke_test(verbose=True):
    checks = {}
    for model in NeuronType:
        initial = create_nn(1, neuron_type=model).pool
        input_current = 0.5 if model == NeuronType.IZHIKEVICH else 100
        step = get_step_function(model)
        baseline = step(initial.copy(), input_current, 0.1, return_observation=True)
        initial[:, k.THETA_INF] = baseline.plasticity_voltage_mV - 1
        initial[:, k.V_DETECT] = baseline.plasticity_voltage_mV - 1
        event = step(initial, input_current, 0.1, return_observation=True)
        checks[f"{model.value}_pre_reset_voltage"] = (
            event.spikes[0] == 1
            and np.allclose(event.plasticity_voltage_mV, baseline.plasticity_voltage_mV)
            and not np.array_equal(event.plasticity_voltage_mV, initial[:, k.V])
        )
    network = create_nn(3, neuron_type={"default": "lif", "izhikevich": 1, "glif5": 1})
    runtime = FixedWeightRuntime(network, create_synapses(3, [], []), dt=0.1, impulse_scale=100)
    tick = runtime.step(100)
    checks["mixed_global_voltage_mV"] = (tick.plasticity_voltage_mV.shape == (3,)
                                         and np.all(np.isfinite(tick.plasticity_voltage_mV)))
    traces = PlasticityTraces(2, tau_spike_ms=1 / np.log(2),
                              tau_voltage_ms=1 / np.log(2),
                              history_ms=1, history_max_samples=2)
    first = traces.advance([1, 1], [-40, -60], 1)
    second = traces.advance([0, 1], [-60, -40], 1)
    checks["simultaneous_and_decay"] = (np.allclose(first.spike_trace, [1, 1])
                                         and np.allclose(second.spike_trace, [0.5, 1.5]))
    checks["bounded_history"] = len(traces.history) == 2
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if plasticity_observation_smoke_test()["passed"] else 1)
