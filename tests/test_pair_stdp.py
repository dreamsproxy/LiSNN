"""Scientific timing and numerical contract of the reference Pair rule."""

import numpy as np
import pytest

from lisnn.plasticity import PairSTDP
from lisnn.synapses import create_synapses


def learner(*, weight=0.5, a_plus=0.1, a_minus=0.12, tau=10):
    return PairSTDP(create_synapses(2, [0], [1], [weight]), tau_plus_ms=tau,
                    tau_minus_ms=tau, a_plus=a_plus, a_minus=a_minus, w_max=1)


@pytest.mark.parametrize("lag", [1, 5, 20])
def test_pre_before_post_ltp_has_analytic_monotone_window(lag):
    rule = learner()
    rule.step([1, 0], 1)
    update = rule.step([0, 1], lag)
    np.testing.assert_allclose(update.potentiation, [0.1 * np.exp(-lag / 10)], rtol=2e-6)
    np.testing.assert_array_equal(update.depression, [0])
    np.testing.assert_allclose(rule.weights, 0.5 + update.potentiation, rtol=2e-6)


@pytest.mark.parametrize("lag", [1, 5, 20])
def test_post_before_pre_ltd_has_analytic_monotone_window(lag):
    rule = learner()
    rule.step([0, 1], 1)
    update = rule.step([1, 0], lag)
    np.testing.assert_allclose(update.depression, [-0.12 * np.exp(-lag / 10)], rtol=2e-6)
    np.testing.assert_array_equal(update.potentiation, [0])


def test_simultaneous_events_do_not_pair_but_old_traces_contribute():
    rule = learner()
    simultaneous = rule.step([1, 1], 1)
    np.testing.assert_array_equal(simultaneous.applied_change, [0])
    later = rule.step([1, 1], 1)
    np.testing.assert_allclose(later.potentiation, [0.1 * np.exp(-0.1)])
    np.testing.assert_allclose(later.depression, [-0.12 * np.exp(-0.1)])
    np.testing.assert_allclose(later.applied_change, [-0.02 * np.exp(-0.1)], rtol=2e-6)


def test_remote_pair_exponential_tail_is_below_declared_tolerance():
    rule = learner()
    rule.step([1, 0], 1)
    update = rule.step([0, 1], 200)
    assert abs(update.applied_change[0]) <= 1e-7


def test_boundaries_and_duplicate_edges_independent():
    edges = create_synapses(2, [0, 0], [1, 1], [0.99, 0.01], allow_duplicates=True)
    rule = PairSTDP(edges, tau_plus_ms=10, tau_minus_ms=10,
                    a_plus=10, a_minus=10, w_max=1)
    rule.step([1, 0], 1)
    upper = rule.step([0, 1], 1)
    np.testing.assert_array_equal(upper.weights_after, [1, 1])
    rule.step([0, 1], 1)
    lower = rule.step([1, 0], 1)
    np.testing.assert_array_equal(lower.weights_after, [0, 0])
    np.testing.assert_allclose(edges.weight, [0.99, 0.01])
    snap = rule.current_edges()
    snap.weight[:] = 0.8
    np.testing.assert_array_equal(rule.weights, [0, 0])


def test_invalid_input_atomic_and_result_owned():
    rule = learner()
    with pytest.raises(ValueError):
        rule.step([0.5, 1], 1)
    with pytest.raises(ValueError):
        rule.step([1, 0], 0)
    np.testing.assert_array_equal(rule.weights, [0.5])
    np.testing.assert_array_equal(rule.pre_trace, [0, 0])
    update = rule.step([1, 0], 1)
    update.pre_trace[:] = 99
    np.testing.assert_array_equal(rule.pre_trace, [1, 0])
    with pytest.raises(ValueError):
        learner(weight=1.1)


def test_two_edge_toy_sequence_is_deterministic_and_independent():
    edges = create_synapses(3, [0, 1], [2, 2], [0.5, 0.5])
    runs = [PairSTDP(edges, tau_plus_ms=10, tau_minus_ms=12,
                     a_plus=0.1, a_minus=0.12, w_max=1) for _ in range(2)]
    sequence = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    for events in sequence:
        out = [rule.step(events, 1) for rule in runs]
        np.testing.assert_array_equal(out[0].weights_after, out[1].weights_after)
    assert runs[0].weights[1] > runs[0].weights[0] > 0.5
