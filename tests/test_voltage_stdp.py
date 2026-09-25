"""Independent gates, analytic samples and ownership for voltage STDP."""

import numpy as np
import pytest

from lisnn.plasticity import VoltageSTDP
from lisnn.synapses import create_synapses


def learner(*, weight=0.5, initial=-65, a_ltd=0.01, a_ltp=0.001):
    return VoltageSTDP(create_synapses(2, [0], [1], [weight]),
                       tau_x_ms=10, tau_minus_ms=1, tau_plus_ms=1,
                       theta_minus_mV=-60, theta_plus_mV=-50,
                       a_ltd=a_ltd, a_ltp=a_ltp, w_max=1,
                       initial_voltage_mV=initial)


def test_ltd_needs_pre_event_and_recent_depolarization_not_post_spike():
    rule = learner()
    quiet = rule.step([0, 0], [-65, -40], 1)
    assert quiet.depression[0] == 0
    assert quiet.ltd_gate_mV[1] == 0
    updated = rule.step([1, 0], [-65, -65], 1)
    filtered = -65 * np.exp(-1) + -40 * (1 - np.exp(-1))
    np.testing.assert_allclose(updated.ltd_gate_mV[1], filtered - (-60), rtol=1e-5)
    np.testing.assert_allclose(updated.depression, [-0.01 * (filtered + 60)], rtol=1e-5)
    np.testing.assert_array_equal(updated.potentiation, [0])


def test_ltp_requires_each_of_pre_history_current_and_filtered_voltage():
    no_pre = learner()
    no_pre.step([0, 0], [-65, -40], 1)
    assert no_pre.step([0, 0], [-65, -40], 1).potentiation[0] == 0
    no_history = learner()
    no_history.step([1, 0], [-65, -65], 1)
    assert no_history.step([0, 0], [-65, -40], 1).potentiation[0] == 0
    no_current = learner()
    no_current.step([1, 0], [-65, -40], 1)
    assert no_current.step([0, 0], [-65, -55], 1).potentiation[0] == 0
    full = learner()
    full.step([1, 0], [-65, -40], 1)
    out = full.step([0, 0], [-65, -40], 1)
    expected_gate = -65 * np.exp(-1) + -40 * (1 - np.exp(-1)) + 60
    expected_ltp = 0.001 * np.exp(-0.1) * 10 * expected_gate
    np.testing.assert_allclose(out.potentiation, [expected_ltp], rtol=1e-5)
    np.testing.assert_array_equal(out.depression, [0])
    assert out.weights_after[0] > 0.5


def test_ltp_scales_with_interval_ltd_is_event_increment():
    a = learner(initial=[-65, -40])
    b = learner(initial=[-65, -40])
    a.step([1, 0], [-65, -40], 1)
    b.step([1, 0], [-65, -40], 1)
    ua = a.step([1, 0], [-65, -40], 1)
    ub = b.step([1, 0], [-65, -40], 2)
    np.testing.assert_allclose(ua.depression, ub.depression)
    assert ua.potentiation[0] > 0 and ub.potentiation[0] > ua.potentiation[0]


def test_thresholds_bounds_validation_and_atomicity():
    rule = learner(weight=0.99, initial=-40, a_ltp=1)
    rule.step([1, 0], [-40, -40], 1)
    result = rule.step([0, 0], [-40, -40], 1)
    assert result.weights_after[0] == 1
    result.filtered_plus_mV[:] = 999
    result.weights_after[:] = 999
    assert rule.weights[0] == 1
    assert rule.traces[2][1] != 999
    before = rule.traces
    for events, voltage, dt in (([2, 0], [-40, -40], 1),
                                 ([0, 0], [-40, np.nan], 1),
                                 ([0, 0], [-40, -40], 0)):
        with pytest.raises(ValueError):
            rule.step(events, voltage, dt)
        for prior, current in zip(before, rule.traces):
            np.testing.assert_array_equal(prior, current)
    with pytest.raises(ValueError):
        learner(initial=np.inf)
    with pytest.raises(ValueError):
        learner(weight=1.1)


def test_owned_edge_snapshot_and_unsigned_depression_floor():
    rule = learner(weight=0.01, initial=-40, a_ltd=1)
    out = rule.step([1, 0], [-65, -65], 1)
    np.testing.assert_array_equal(out.weights_after, [0])
    snap = rule.current_edges()
    snap.weight[:] = 0.9
    np.testing.assert_array_equal(rule.weights, [0])
