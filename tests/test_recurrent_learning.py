"""SNN convenience runtime, causal plasticity and snapshot restoration."""

import numpy as np
import pytest

from lisnn.network import create_nn
from lisnn.network.runtime import FixedWeightRuntime
from lisnn.neurons import kernels as k
from lisnn.synapses import create_synapses


PAIR = dict(tau_plus_ms=10, tau_minus_ms=10, a_plus=0.2, a_minus=0.1, w_max=1)
TRIPLET = dict(tau_plus_ms=10, tau_minus_ms=10, tau_x_ms=20, tau_y_ms=20,
               a2_plus=0.2, a2_minus=0.1, a3_plus=0.1, a3_minus=0.1, w_max=1)
VOLTAGE = dict(tau_x_ms=10, tau_minus_ms=2, tau_plus_ms=2,
               theta_minus_mV=-60, theta_plus_mV=-50,
               a_ltd=0.0001, a_ltp=0.001, w_max=1)


def setup(rule, params=None):
    model = create_nn(2, neuron_type="lif")
    model.pool[:, k.T_REF] = 0
    model.pool[1, k.V] = -50.1
    model.configure_runtime(create_synapses(2, [0], [1], [0.5]), dt=0.1,
                            impulse_scale=100, plasticity=rule, plasticity_params=params)
    return model


@pytest.mark.parametrize("name,params", [("off", None), ("pair", PAIR),
                                         ("triplet", TRIPLET), ("voltage", VOLTAGE)])
def test_each_rule_runs_on_small_recurrent_network(name, params):
    model = setup(name, params)
    baseline = model.snapshot_runtime()
    first = model.step([32000, 0])
    second = model.step()
    np.testing.assert_array_equal(first.current_spikes, [1, 0])
    np.testing.assert_array_equal(second.current_spikes, [0, 1])
    assert first.weights_after is None if name == "off" else first.weights_after is not None
    if name != "off":
        assert second.plasticity_update is not None and second.trace_state is not None
        np.testing.assert_array_equal(second.weights_after, model.runtime.weights)
    model.restore_runtime(baseline)
    again = model.step([32000, 0])
    np.testing.assert_array_equal(again.current_spikes, first.current_spikes)
    np.testing.assert_array_equal(again.voltage_mV, first.voltage_mV)
    model.reset_runtime()
    assert model.runtime.tick == 0
    np.testing.assert_array_equal(model.runtime.weights, [0.5])


def test_off_matches_fixed_weight_propagation():
    model = setup("off")
    independent = FixedWeightRuntime(model, create_synapses(2, [0], [1], [0.5]),
                                     dt=0.1, impulse_scale=100)
    for external in ([32000, 0], [0, 0], [32000, 0], [0, 0]):
        a, b = model.step(external), independent.step(external)
        np.testing.assert_array_equal(a.current_spikes, b.current_spikes)
        np.testing.assert_array_equal(a.voltage_mV, b.voltage_mV)
        np.testing.assert_array_equal(a.propagation.synaptic_current_pA,
                                      b.propagation.synaptic_current_pA)


def test_pair_weight_change_affects_only_subsequent_propagation():
    model = setup("pair", PAIR)
    first = model.step([32000, 0])
    second = model.step()
    assert first.weights_after[0] == 0.5
    assert second.weights_after[0] > 0.5
    np.testing.assert_allclose(second.propagation.synaptic_current_pA, [0, 500])
    model.step([32000, 0])
    later = model.step()
    assert later.propagation.synaptic_current_pA[1] > 500
    later.weights_after[:] = 0
    assert model.runtime.weights[0] > 0


def test_mixed_models_and_failed_learning_leave_everything_intact(monkeypatch):
    mixed = create_nn(2, neuron_type={"default": "izhikevich", "lif": 1})
    mixed.configure_runtime(create_synapses(2, [0], [1]), dt=0.1,
                            impulse_scale=100, plasticity="pair", plasticity_params=PAIR)
    first = mixed.step(100)
    assert first.plasticity_voltage_mV.dtype == np.float32
    assert np.isfinite(first.izhikevich_before["input_current_pA"]).all()
    before = mixed.snapshot_runtime()
    def broken(*args, **kwargs):
        raise FloatingPointError("injected learning failure")
    monkeypatch.setattr(mixed.runtime._learner, "step", broken)
    with pytest.raises(FloatingPointError):
        mixed.step(100)
    assert mixed.runtime.tick == before.tick
    np.testing.assert_array_equal(mixed.runtime.weights, before.weights)
    np.testing.assert_array_equal(mixed.runtime.pool, before.pool)


def test_unconfigured_and_invalid_selector():
    model = create_nn(2)
    with pytest.raises(RuntimeError):
        model.step()
    with pytest.raises(ValueError):
        model.configure_runtime(create_synapses(2, [0], [1]), dt=0.1,
                                impulse_scale=100, plasticity="all")
