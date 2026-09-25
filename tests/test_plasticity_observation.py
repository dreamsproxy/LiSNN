"""Regression checks for the independent #19 observation/trace contract."""

import numpy as np
import pytest

from lisnn.network import create_nn
from lisnn.network.runtime import FixedWeightRuntime
from lisnn.neurons import kernels as k
from lisnn.neurons.registry import NeuronType, get_step_function
from lisnn.plasticity import PlasticityTraces
from lisnn.synapses import create_synapses


@pytest.mark.parametrize("model", list(NeuronType))
def test_integrated_spike_voltage_survives_reset_for_every_model(model):
    initial = create_nn(1, neuron_type=model).pool
    current = 0.5 if model == NeuronType.IZHIKEVICH else 100
    step = get_step_function(model)
    unforced = step(initial.copy(), current, 0.1, return_observation=True)
    forced = initial.copy()
    forced[:, k.THETA_INF] = unforced.plasticity_voltage_mV - 1
    forced[:, k.V_DETECT] = unforced.plasticity_voltage_mV - 1
    observed = step(forced, current, 0.1, return_observation=True)
    assert observed.spikes.dtype == np.float32
    np.testing.assert_array_equal(observed.spikes, [1])
    np.testing.assert_allclose(observed.plasticity_voltage_mV, unforced.plasticity_voltage_mV)
    assert float(observed.plasticity_voltage_mV[0]) != float(forced[0, k.V])
    observed.plasticity_voltage_mV[:] = 999
    assert forced[0, k.V] != 999
    assert isinstance(step(initial.copy(), current, 0.1), np.ndarray)


@pytest.mark.parametrize("model", [m for m in NeuronType if m != NeuronType.IZHIKEVICH])
def test_refractory_voltage_is_physically_applied_not_discarded_candidate(model):
    pool = create_nn(1, neuron_type=model).pool
    pool[:, k.V] = -62
    pool[:, k.REFRACTORY] = 0.2
    observed = get_step_function(model)(pool, 100, 0.1, return_observation=True)
    np.testing.assert_array_equal(observed.spikes, [0])
    np.testing.assert_array_equal(observed.plasticity_voltage_mV, pool[:, k.V])


def test_mixed_runtime_voltage_order_and_snapshot_ownership():
    model = create_nn(9, neuron_type={"default": "lif", **{m.value: 1 for m in NeuronType if m != NeuronType.LIF}})
    runtime = FixedWeightRuntime(model, create_synapses(9, [], []), dt=0.1, impulse_scale=100)
    expected = model.pool.copy()
    for kind, section in model.type_slices.items():
        current = np.full(section.stop - section.start, 100, dtype=np.float32)
        if kind == NeuronType.IZHIKEVICH:
            current /= expected[section, k.C_M]
        observation = get_step_function(kind)(expected[section], current, 0.1, return_observation=True)
        if section.start == 0:
            voltage = np.empty(9, dtype=np.float32)
        voltage[section] = observation.plasticity_voltage_mV
    result = runtime.step(100)
    np.testing.assert_array_equal(result.plasticity_voltage_mV, voltage)
    assert result.plasticity_voltage_mV.dtype == np.float32
    result.plasticity_voltage_mV[:] = 123
    assert not np.any(runtime.pool[:, k.V] == 123)


def test_traces_analytic_decay_simultaneous_events_history_and_aliases():
    traces = PlasticityTraces(2, tau_spike_ms=1 / np.log(2),
                             tau_voltage_ms=1 / np.log(2),
                             history_ms=0.5, history_max_samples=2)
    spikes = np.array([1, 1], dtype=np.float32)
    volts = np.array([-40, -60], dtype=np.float32)
    first = traces.advance(spikes, volts, 1)
    np.testing.assert_allclose(first.spike_trace, [1, 1], rtol=1e-6)
    np.testing.assert_allclose(first.voltage_trace_mV, [-20, -30], rtol=1e-6)
    spikes[:] = 0
    volts[:] = 0
    second = traces.advance([0, 1], [-60, -40], 1)
    np.testing.assert_allclose(second.spike_trace, [0.5, 1.5], rtol=1e-6)
    np.testing.assert_allclose(second.voltage_trace_mV, [-40, -35], rtol=1e-6)
    assert len(traces.history) == 1 and traces.history[0].time_ms == second.time_ms
    second.spike_trace[:] = 99
    traces.history[0].voltage_trace_mV[:] = 99
    np.testing.assert_allclose(traces.history[0].spike_trace, [0.5, 1.5], rtol=1e-6)
    np.testing.assert_allclose(traces.spike_trace, [0.5, 1.5], rtol=1e-6)
    np.testing.assert_allclose(traces.voltage_trace_mV, [-40, -35], rtol=1e-6)


def test_trace_validation_and_no_partial_update():
    traces = PlasticityTraces(2, tau_spike_ms=20, tau_voltage_ms=10)
    assert traces.history == ()
    for spike, volt, dt in [([1, 0], [0, np.nan], 1), ([2, 0], [0, 0], 1),
                             ([0, 0], [0, 0], 0)]:
        with pytest.raises(ValueError):
            traces.advance(spike, volt, dt)
        assert traces.time_ms == 0
        np.testing.assert_array_equal(traces.spike_trace, [0, 0])
    with pytest.raises(ValueError):
        PlasticityTraces(2, tau_spike_ms=1, tau_voltage_ms=1, history_ms=5)
