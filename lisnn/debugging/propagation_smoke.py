"""Independently callable propagation/adapter diagnostics for issue #18."""

import numpy as np

from lisnn.network import create_nn
from lisnn.network.runtime import FixedWeightRuntime
from lisnn.neurons import kernels as k
from lisnn.synapses import create_synapses
from lisnn.synapses.propagation import propagate


def demonstration():
    model = create_nn(4, neuron_type='LIF')
    # Put N2 0.1 mV below threshold to make the second hop visible with J0=100.
    # This is an explicit initial condition, not input-induced voltage assignment.
    model.pool[2, k.V] = -50.1
    edges = create_synapses(4, [0, 0, 1, 2], [2, 3, 2, 3], [0.5, 0.8, 0.25, 0.4])
    runtime = FixedWeightRuntime(model, edges, dt=0.1, impulse_scale=100)
    # At rest: 32000 pA * 0.1 ms / 200 pF = 16 mV -> N0 spikes.
    results = [runtime.step([32000, 0, 0, 0]), runtime.step(), runtime.step()]
    return results



def propagation_smoke_test(verbose=True):
    results = demonstration()
    checks = {
        'tick0_no_synaptic_effect': np.all(results[0].propagation.synaptic_current_pA == 0),
        'tick0_source_spike': np.array_equal(results[0].current_spikes, [1, 0, 0, 0]),
        'tick1_first_hop': np.allclose(results[1].propagation.synaptic_current_pA, [0, 0, 500, 800]),
        'tick1_next_spike': np.array_equal(results[1].current_spikes, [0, 0, 1, 0]),
        'tick2_second_hop': np.allclose(results[2].propagation.synaptic_current_pA, [0, 0, 0, 400]),
        'finite_voltages': all(np.all(np.isfinite(r.voltage_mV)) for r in results),
    }
    edges = create_synapses(2, [0, 0], [1, 1], [0.4, 0.2], allow_duplicates=True)
    for dt in (1, 0.5, 0.1):
        current = propagate([1, 0], edges, dt, 100)
        checks[f'impulse_and_duplicates_dt_{dt}'] = np.allclose(current * dt, [0, 60])
    mixed = create_nn(3, neuron_type={'default': 'LIF', 'Izhikevich': 1, 'AdEx': 1})
    runtime = FixedWeightRuntime(mixed, create_synapses(3, [0], [2]), dt=0.1, impulse_scale=100)
    output = runtime.step(100)
    checks['izh_input_roundtrip_pA'] = np.allclose(output.izhikevich_before['input_current_pA'], [100])
    checks['izh_recovery_output_pA'] = np.allclose(output.izhikevich_before['recovery_current_pA'], [2600])
    checks['mixed_finite'] = np.all(np.isfinite(output.voltage_mV))
    checks = {name: bool(value) for name, value in checks.items()}
    if verbose:
        for name, value in checks.items():
            print(f"{'PASS' if value else 'FAIL'} {name}")
    return {'passed': all(checks.values()), 'checks': checks}


if __name__ == '__main__':
    raise SystemExit(0 if propagation_smoke_test()['passed'] else 1)
