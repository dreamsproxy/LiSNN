"""Verify the fast duplicate check against the original NumPy unique oracle."""

import numpy as np
import pytest

from benchmarks.profile_playground import Config, OPERATIONS, fingerprint, workload
from lisnn.synapses import core, create_synapses
from lisnn.synapses.propagation import propagate
from lisnn.network import create_nn
from lisnn.neurons.registry import NeuronType
from tests.test_recurrent_learning import PAIR, TRIPLET, VOLTAGE


def reference_duplicate_check(pre, post, population):
    keys = pre.astype(np.int64) * np.int64(population) + post.astype(np.int64)
    return np.unique(keys).size != keys.size


@pytest.mark.parametrize("seed", range(5))
def test_duplicate_check_matches_oracle_without_mutating_inputs(seed):
    rng = np.random.default_rng(seed)
    for n in (1, 3, 31, np.iinfo(np.int32).max):
        for count in (0, 1, 2, 64):
            pre = rng.integers(0, n, count, dtype=np.int32)
            post = rng.integers(0, n, count, dtype=np.int32)
            if count > 2:
                pre[-1], post[-1] = pre[0], post[0]
            for order in (np.arange(count), np.lexsort((post, pre)), np.arange(count)[::-1]):
                a, b = pre[order].copy(), post[order].copy()
                original_a, original_b = a.copy(), b.copy()
                assert core._has_duplicate_edges(a, b, n) == reference_duplicate_check(a, b, n)
                np.testing.assert_array_equal(a, original_a)
                np.testing.assert_array_equal(b, original_b)


def test_mutated_public_topology_is_still_revalidated():
    edges = create_synapses(4, [0, 0, 2], [1, 3, 1])
    edges.post_idx[1] = 1
    with pytest.raises(ValueError, match="duplicate"):
        propagate(np.ones(4, dtype=np.float32), edges, 1, 1)
    edges.allow_duplicates = True
    np.testing.assert_array_equal(propagate(np.ones(4), edges, 1, 1), [0, 3, 0, 0])


def test_shuffled_edges_keep_float32_accumulation_order():
    pre, post = [2, 0, 1], [3, 3, 3]
    weights = np.array([1, 1e8, 1], dtype=np.float32)
    edges = create_synapses(4, pre, post, weights)
    actual = propagate(np.ones(4), edges, 1, 1)
    expected = np.zeros(4, dtype=np.float32)
    np.add.at(expected, post, weights)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(edges.pre_idx, pre)
    np.testing.assert_array_equal(edges.weight, weights)


@pytest.mark.parametrize("edge_order", ["ordered", "shuffled"])
def test_all_profiled_workloads_have_identical_trajectories(monkeypatch, edge_order):
    fast = core._has_duplicate_edges
    for operation in OPERATIONS:
        monkeypatch.setattr(core, "_has_duplicate_edges", reference_duplicate_check)
        run, _ = workload(Config(32, 4, edge_order=edge_order), operation)
        reference = [fingerprint(run(tick)) for tick in range(25)]
        monkeypatch.setattr(core, "_has_duplicate_edges", fast)
        run, _ = workload(Config(32, 4, edge_order=edge_order), operation)
        assert [fingerprint(run(tick)) for tick in range(25)] == reference, operation


@pytest.mark.parametrize("rule,params", [("off", None), ("pair", PAIR),
                                        ("triplet", TRIPLET), ("voltage", VOLTAGE)])
def test_mixed_neuron_learning_trajectories_match_reference(monkeypatch, rule, params):
    def run():
        n = 2 * len(NeuronType)
        types = {"default": "LIF", **{kind.value: 2 for kind in NeuronType}}
        model = create_nn(n, neuron_type=types, seed=11)
        # Reverse ring deliberately exercises the unordered fallback.
        pre = np.arange(n, dtype=np.int32)[::-1]
        model.configure_runtime(create_synapses(n, pre, (pre + 1) % n, 0.01),
                                dt=0.1, impulse_scale=5, plasticity=rule,
                                plasticity_params=params)
        result = []
        for tick in range(80):
            value = model.step(600 if tick % 4 else 0, plasticity_enabled=(tick < 60))
            result.append(fingerprint(value))
        return result, model.runtime.pool.copy(), model.runtime.weights.copy()

    fast = core._has_duplicate_edges
    monkeypatch.setattr(core, "_has_duplicate_edges", reference_duplicate_check)
    reference, pool, weights = run()
    monkeypatch.setattr(core, "_has_duplicate_edges", fast)
    actual, actual_pool, actual_weights = run()
    assert actual == reference
    np.testing.assert_array_equal(actual_pool, pool)
    np.testing.assert_array_equal(actual_weights, weights)
