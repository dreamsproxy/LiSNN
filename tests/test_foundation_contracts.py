"""Numerical and invalid-input regressions for Phase 1 foundation contracts."""

import numpy as np
import pytest

from lisnn.neurons import kernels as k
from lisnn.neurons.registry import NEURON_STEP_REGISTRY
from lisnn.synapses import SynapseEdges, create_synapses


def make_edges(direct, pre, post, population=4):
    if direct:
        return SynapseEdges(population, pre, post, np.ones(len(pre)))
    return create_synapses(population, pre, post)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("endpoint", ["pre", "post"])
@pytest.mark.parametrize("bad", [
    np.array([2**32], dtype=np.int64),
    np.array([-2**32], dtype=np.int64),
    np.array([2**32], dtype=np.uint64),
    np.array([2**64 - 1], dtype=np.uint64),
    np.array([-2**63], dtype=np.int64),
    np.array([4], dtype=np.int64),
    np.array([-1], dtype=np.int64),
])
def test_indices_rejected_before_narrowing(direct, endpoint, bad):
    pre, post = (bad, [1]) if endpoint == "pre" else ([1], bad)
    with pytest.raises(IndexError, match="outside the population"):
        make_edges(direct, pre, post)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_valid_edges_preserve_order_and_dtype(direct, dtype):
    pre = np.array([2, 0, 1], dtype=dtype)
    post = np.array([1, 2, 0], dtype=dtype)
    edges = make_edges(direct, pre, post)
    np.testing.assert_array_equal(edges.pre_idx, pre)
    np.testing.assert_array_equal(edges.post_idx, post)
    assert edges.pre_idx.dtype == edges.post_idx.dtype == np.int32
    empty = make_edges(direct, [], [])
    assert empty.edge_count == 0
    assert empty.pre_idx.dtype == empty.post_idx.dtype == np.int32


@pytest.mark.parametrize("direct", [False, True])
def test_max_supported_population_without_allocating_neurons(direct):
    population = int(np.iinfo(np.int32).max)
    edges = make_edges(direct, [population - 1], [0], population)
    assert edges.pre_idx[0] == population - 1
    with pytest.raises(IndexError):
        make_edges(direct, [population], [0], population)


@pytest.mark.parametrize("step", list(NEURON_STEP_REGISTRY.values()))
@pytest.mark.parametrize("dt,error", [
    (0, ValueError), (-0.1, ValueError), (np.nan, ValueError),
    (np.inf, ValueError), (1e40, ValueError), (1e-50, ValueError),
    ([0.1], ValueError), (True, TypeError), ("0.1", TypeError),
    (0.1j, TypeError),
])
def test_invalid_dt_does_not_mutate_state(step, dt, error):
    pool = k.new_population(2)
    pool[:, k.REFRACTORY] = 1
    before = pool.copy()
    with pytest.raises(error, match="dt"):
        step(pool, 0, dt)
    np.testing.assert_array_equal(pool, before)


@pytest.mark.parametrize("step", list(NEURON_STEP_REGISTRY.values()))
@pytest.mark.parametrize("current,error", [
    (np.nan, ValueError), ([0, np.inf], ValueError), (1e40, ValueError),
    ([1], ValueError), ([1j, 0], TypeError), ("1", TypeError),
])
def test_invalid_current_does_not_mutate_state(step, current, error):
    pool = k.new_population(2)
    pool[:, k.REFRACTORY] = 1
    before = pool.copy()
    with pytest.raises(error, match="input_current"):
        step(pool, current, 0.1)
    np.testing.assert_array_equal(pool, before)


@pytest.mark.parametrize("step", [
    k.lif_step, k.adaptive_lif_step, k.glif3_step, k.glif4_step, k.glif5_step,
])
def test_resistance_units_against_hand_calculation(step):
    pool = k.new_population(2)
    pool[:, k.V] = [-55, -65]
    spikes = step(pool, [0, 100], 1.0)
    # 100 MOhm = 0.1 GOhm; leak at -55 mV is 100 pA.
    # C = 200 pF: +/-100 pA gives +/-0.5 mV in 1 ms.
    np.testing.assert_allclose(pool[:, k.V], [-55.5, -64.5], atol=1e-5, rtol=0)
    np.testing.assert_array_equal(spikes, [0, 0])
    assert spikes.dtype == np.float32


def test_lif_and_adex_passive_leak_agree():
    lif = k.new_population(1)
    lif[:, k.V] = -55
    adex = lif.copy()
    # Suppress exponential initiation to isolate the passive leak term.
    adex[:, k.V_T] = 100
    k.lif_step(lif, 0, 1)
    k.adex_step(adex, 0, 1)
    np.testing.assert_allclose(lif[:, k.V], adex[:, k.V], atol=1e-5, rtol=0)


def test_smaller_dt_converges_toward_passive_analytic_solution():
    exact = -65 + 10 * np.exp(-1)  # t = tau = 20 ms
    errors = []
    for dt in [1.0, 0.5, 0.1]:
        pool = k.new_population(1)
        pool[:, k.V] = -55
        for _ in range(round(20 / dt)):
            k.lif_step(pool, 0, dt)
        errors.append(abs(float(pool[0, k.V]) - exact))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_refractory_tick_uses_start_of_tick_state():
    pool = k.new_population(1)
    pool[:, k.REFRACTORY] = 0.5
    assert k.lif_step(pool, 100, 1)[0] == 0
    assert pool[0, k.V] == -65
    assert pool[0, k.REFRACTORY] == 0
    k.lif_step(pool, 100, 1)
    assert pool[0, k.V] == -64.5
