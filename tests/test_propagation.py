"""Contract regressions for sparse propagation and causal heterogeneous runtime."""

from collections import OrderedDict

import numpy as np
import pytest

from lisnn.debugging.propagation_smoke import demonstration, propagation_smoke_test
from lisnn.network import create_nn
from lisnn.network.runtime import FixedWeightRuntime
from lisnn.neurons import kernels as k
from lisnn.neurons.adapters import current_to_izhikevich, izhikevich_to_current, observe_izhikevich
from lisnn.neurons.registry import NeuronType, get_step_function
from lisnn.synapses import create_synapses
from lisnn.synapses.propagation import propagate


@pytest.mark.parametrize('pre,post,weight,spikes,expected,duplicates', [
    ([], [], [], [1, 0, 0, 0], [0, 0, 0, 0], False),
    ([0], [1], [0.5], [0, 0, 0, 0], [0, 0, 0, 0], False),
    ([0], [1], [0.5], [1, 0, 0, 0], [0, 50, 0, 0], False),
    ([0, 0], [1, 2], [0.5, 0.8], [1, 0, 0, 0], [0, 50, 80, 0], False),
    ([0, 2], [1, 1], [0.5, 0.2], [1, 0, 1, 0], [0, 70, 0, 0], False),
    ([0, 0], [1, 1], [0.4, 0.2], [1, 0, 0, 0], [0, 60, 0, 0], True),
])
def test_topology(pre, post, weight, spikes, expected, duplicates):
    edges = create_synapses(4, pre, post, weight, allow_duplicates=duplicates)
    spikes = np.array(spikes, dtype=np.float32)
    saved = spikes.copy()
    result = propagate(spikes, edges, 1, 100)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(spikes, saved)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, spikes)
    np.testing.assert_array_equal(edges.weight, np.array(weight, dtype=np.float32))


@pytest.mark.parametrize('dt', [1, 0.5, 0.1, 0.01])
def test_event_impulse_invariant(dt):
    edges = create_synapses(2, [0], [1], [0.5])
    result = propagate([1, 0], edges, dt, 100, return_details=True)
    np.testing.assert_allclose(result.edge_current_pA * dt, [50])
    np.testing.assert_allclose(result.synaptic_current_pA * dt, [0, 50])


@pytest.mark.parametrize('spikes', [[0.5, 0], [np.nan, 0], [1 + 1e-10, 0], [1], [-1, 0]])
def test_invalid_spikes(spikes):
    with pytest.raises(ValueError):
        propagate(spikes, create_synapses(2, [0], [1]), 1, 100)


@pytest.mark.parametrize('dt', [0, -1, np.nan, np.inf, 1e40, 1e-50])
def test_invalid_dt(dt):
    with pytest.raises(ValueError):
        propagate([1, 0], create_synapses(2, [0], [1]), dt, 100)


@pytest.mark.parametrize('scale', [-1, np.inf, np.nan])
def test_invalid_impulse_scale(scale):
    with pytest.raises(ValueError):
        propagate([1, 0], create_synapses(2, [0], [1]), 1, scale)


def test_mutated_edges_and_overflow_are_rejected():
    edges = create_synapses(2, [0], [1])
    edges.pre_idx = np.array([2**32], dtype=np.int64)
    with pytest.raises(IndexError):
        propagate([1, 0], edges, 1, 100)
    edges = create_synapses(2, [0], [1], [np.finfo(np.float32).max])
    with pytest.raises(FloatingPointError):
        propagate([1, 0], edges, 0.1, 100)


def test_demonstration_causality_and_reproducibility():
    first, second = demonstration(), demonstration()
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.voltage_mV, b.voltage_mV)
        np.testing.assert_array_equal(a.current_spikes, b.current_spikes)
    np.testing.assert_array_equal(first[0].current_spikes, [1, 0, 0, 0])
    np.testing.assert_array_equal(first[1].current_spikes, [0, 0, 1, 0])
    np.testing.assert_allclose(first[1].propagation.synaptic_current_pA, [0, 0, 500, 800])
    np.testing.assert_allclose(first[2].propagation.synaptic_current_pA, [0, 0, 0, 400])
    assert propagation_smoke_test(False)['passed']


@pytest.mark.parametrize('cap', [100, 200, 300])
def test_izhikevich_adapter_and_equivalent_outputs(cap):
    pool = create_nn(1, neuron_type='Izhikevich').pool
    pool[:, k.C_M] = cap
    native = current_to_izhikevich([100], [cap])
    np.testing.assert_allclose(native, [100 / cap])
    np.testing.assert_allclose(izhikevich_to_current(native, [cap]), [100])
    obs = observe_izhikevich(pool)
    np.testing.assert_allclose(obs['voltage_mV'], [-65])
    np.testing.assert_allclose(obs['recovery_current_pA'], [13 * cap])
    np.testing.assert_allclose(obs['intrinsic_current_pA'], [-16 * cap], atol=0.01)
    # No integration reset: signed equivalent currents predict dV numerically.
    expected_dv = (obs['intrinsic_current_pA'] + obs['recovery_current_pA'] + 100) / cap
    k.izhikevich_step(pool, native, 0.1)
    np.testing.assert_allclose(pool[:, k.V], -65 + 0.1 * expected_dv, atol=1e-5)


@pytest.mark.parametrize('cap', [0, -1, np.inf, np.nan])
def test_bad_izh_capacitance(cap):
    with pytest.raises(ValueError):
        current_to_izhikevich([100], [cap])


def test_all_model_slices_match_independent_kernels_and_order():
    specification = {'default': 'LIF', **{model.value: 1 for model in NeuronType if model != NeuronType.LIF}}
    model = create_nn(9, neuron_type=specification)
    empty = create_synapses(9, [], [])
    runtime = FixedWeightRuntime(model, empty, dt=0.1, impulse_scale=100)
    expected = model.pool.copy()
    for kind, section in model.type_slices.items():
        value = np.full(section.stop - section.start, 100, dtype=np.float32)
        if kind == NeuronType.IZHIKEVICH:
            value /= expected[section, k.C_M]
        get_step_function(kind)(expected[section], value, 0.1)
    model.type_slices = OrderedDict(reversed(list(model.type_slices.items())))
    reverse = FixedWeightRuntime(model, empty, dt=0.1, impulse_scale=100)
    result, other = runtime.step(100), reverse.step(100)
    np.testing.assert_array_equal(runtime.pool, expected)
    np.testing.assert_array_equal(result.voltage_mV, other.voltage_mV)
    np.testing.assert_array_equal(result.current_spikes, other.current_spikes)
    np.testing.assert_allclose(result.izhikevich_before['input_current_pA'], [100])
    assert np.all(np.isfinite(runtime.pool))


def test_izh_spike_is_binary_and_drives_next_tick_in_global_current_units():
    model = create_nn(2, neuron_type={'default': 'LIF', 'Izhikevich': 1})
    runtime = FixedWeightRuntime(model, create_synapses(2, [0], [1], [0.5]), dt=0.1, impulse_scale=100)
    first = runtime.step([200000, 0])
    np.testing.assert_array_equal(first.current_spikes, [1, 0])
    np.testing.assert_array_equal(first.propagation.synaptic_current_pA, [0, 0])
    second = runtime.step()
    np.testing.assert_allclose(second.propagation.synaptic_current_pA, [0, 500])
    assert second.current_spikes.dtype == np.float32


def test_feedback_buffers_and_failed_tick_are_isolated():
    model = create_nn(2)
    original = model.pool.copy()
    edges = create_synapses(2, [], [])
    runtime = FixedWeightRuntime(model, edges, dt=1, impulse_scale=100)
    value = np.array([100, 0], dtype=np.float32)
    runtime.schedule_feedback(1, value)
    runtime.schedule_feedback(1, value)
    value[:] = 0
    first = runtime.step()
    np.testing.assert_array_equal(first.feedback_current_pA, [0, 0])
    with pytest.raises(ValueError):
        runtime.step([np.nan, 0])
    assert runtime.tick == 1 and runtime.time_ms == 1
    second = runtime.step()
    np.testing.assert_array_equal(second.feedback_current_pA, [200, 0])
    second.current_spikes[:] = 1
    second.voltage_mV[:] = 999
    np.testing.assert_array_equal(runtime.previous_spikes, [0, 0])
    assert np.all(runtime.pool[:, k.V] < 0)
    np.testing.assert_array_equal(model.pool, original)
    with pytest.raises(ValueError):
        runtime.schedule_feedback(1, [0, 0])
    np.testing.assert_array_equal(runtime.step().feedback_current_pA, [0, 0])


def test_failed_later_slice_does_not_partially_advance(monkeypatch):
    import lisnn.network.runtime as module
    model = create_nn(2, neuron_type={'default': 'LIF', 'AdEx': 1})
    runtime = FixedWeightRuntime(model, create_synapses(2, [], []), dt=0.1, impulse_scale=100)
    before = runtime.pool.copy()
    real = module.get_step_function
    def dispatch(kind):
        if kind == NeuronType.LIF:
            def fail(*args):
                raise FloatingPointError('injected later-slice failure')
            return fail
        return real(kind)
    monkeypatch.setattr(module, 'get_step_function', dispatch)
    with pytest.raises(FloatingPointError):
        runtime.step(100)
    np.testing.assert_array_equal(runtime.pool, before)
    assert runtime.tick == 0 and runtime.time_ms == 0
