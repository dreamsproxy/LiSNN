"""Analytic tests for Pfister–Gerstner pair and higher-order interactions."""

import numpy as np
import pytest

from lisnn.plasticity import PairSTDP, TripletSTDP
from lisnn.synapses import create_synapses


def learner(a3_plus=0.2, a3_minus=0.15):
    return TripletSTDP(create_synapses(2, [0], [1], [0.5]),
                       tau_plus_ms=10, tau_minus_ms=10,
                       tau_x_ms=20, tau_y_ms=20,
                       a2_plus=0.1, a2_minus=0.12,
                       a3_plus=a3_plus, a3_minus=a3_minus, w_max=1)


def test_pair_reduction_and_initial_simultaneous_neutrality():
    triplet = learner(a3_plus=0, a3_minus=0)
    pair = PairSTDP(create_synapses(2, [0], [1], [0.5]), tau_plus_ms=10,
                    tau_minus_ms=10, a_plus=0.1, a_minus=0.12, w_max=1)
    for event in ([1, 1], [0, 0], [1, 0], [0, 1], [0, 1], [1, 0]):
        a = triplet.step(event, 1)
        b = pair.step(event, 1)
        np.testing.assert_allclose(a.weights_after, b.weights_after, atol=1e-7)
        np.testing.assert_array_equal(a.triplet_ltp, [0])
        np.testing.assert_array_equal(a.triplet_ltd, [0])


def test_pre_post_post_has_extra_ltp_from_slow_post_trace():
    rule = learner()
    rule.step([1, 0], 1)
    first = rule.step([0, 1], 1)
    second = rule.step([0, 1], 1)
    np.testing.assert_array_equal(first.triplet_ltp, [0])
    expected = 0.2 * np.exp(-2 / 10) * np.exp(-1 / 20)
    np.testing.assert_allclose(second.triplet_ltp, [expected], rtol=2e-6)
    assert second.triplet_ltp[0] > 0 and second.pair_ltp[0] > 0


def test_post_pre_pre_has_extra_ltd_from_slow_pre_trace():
    rule = learner()
    rule.step([0, 1], 1)
    first = rule.step([1, 0], 1)
    second = rule.step([1, 0], 1)
    np.testing.assert_array_equal(first.triplet_ltd, [0])
    expected = -0.15 * np.exp(-2 / 10) * np.exp(-1 / 20)
    np.testing.assert_allclose(second.triplet_ltd, [expected], rtol=2e-6)


def test_pre_pre_post_pairs_both_presynaptic_events_without_false_triplet():
    rule = learner()
    rule.step([1, 0], 1)
    rule.step([1, 0], 1)
    out = rule.step([0, 1], 1)
    np.testing.assert_allclose(out.pair_ltp, [0.1 * (np.exp(-2 / 10) + np.exp(-1 / 10))], rtol=2e-6)
    np.testing.assert_array_equal(out.triplet_ltp, [0])
    np.testing.assert_array_equal(out.triplet_ltd, [0])


def test_context_frequency_and_trace_aliasing():
    fast = learner()
    slow = learner()
    for rule, lag in ((fast, 1), (slow, 20)):
        rule.step([1, 0], 1)
        rule.step([0, 1], lag)
    f = fast.step([0, 1], 1)
    s = slow.step([0, 1], 1)
    assert f.triplet_ltp[0] > s.triplet_ltp[0]
    f.weights_after[:] = 0
    f.triplet_ltp[:] = 0
    assert fast.weights[0] > 0
    copy = fast.traces
    copy[:] = 99
    assert np.all(fast.traces != 99)


def test_validation_and_owned_weights():
    edges = create_synapses(2, [0], [1], [0.5])
    rule = learner()
    for spikes, dt in (([2, 0], 1), ([0, 0], -1)):
        with pytest.raises(ValueError):
            rule.step(spikes, dt)
    np.testing.assert_array_equal(rule.traces, np.zeros((4, 2)))
    with pytest.raises(ValueError):
        TripletSTDP(edges, tau_plus_ms=0, tau_minus_ms=1,
                    tau_x_ms=1, tau_y_ms=1, a2_plus=1,
                    a2_minus=1, a3_plus=1, a3_minus=1, w_max=1)
    rule.step([1, 0], 1)
    snap = rule.current_edges()
    snap.weight[:] = 0
    np.testing.assert_array_equal(rule.weights, [0.5])
