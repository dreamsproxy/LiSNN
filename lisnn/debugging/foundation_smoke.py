"""Executable checks for index safety, electrical units and input boundaries."""

import numpy as np

from lisnn.debugging.checks import rejects, report
from lisnn.neurons import kernels as k
from lisnn.neurons.registry import NEURON_STEP_REGISTRY
from lisnn.synapses import SynapseEdges, create_synapses


def index_smoke_test(verbose=True):
    checks = {}
    constructors = {
        "factory": lambda pre, post: create_synapses(4, pre, post),
        "direct": lambda pre, post: SynapseEdges(4, pre, post, np.ones(len(pre))),
    }
    invalid = [
        np.array([2**32], dtype=np.int64),
        np.array([-2**32], dtype=np.int64),
        np.array([2**64 - 1], dtype=np.uint64),
        np.array([-1], dtype=np.int64),
        np.array([4], dtype=np.int64),
    ]
    for name, construct in constructors.items():
        for number, bad in enumerate(invalid):
            checks[f"{name}_pre_reject_{number}"] = rejects(IndexError, construct, bad, [1])
            checks[f"{name}_post_reject_{number}"] = rejects(IndexError, construct, [1], bad)
        edges = construct(np.array([2, 0], dtype=np.int64), [1, 2])
        checks[f"{name}_order"] = edges.pre_idx.tolist() == [2, 0] and edges.post_idx.tolist() == [1, 2]
        checks[f"{name}_dtype"] = edges.pre_idx.dtype == edges.post_idx.dtype == np.int32
        empty = construct([], [])
        checks[f"{name}_empty"] = empty.edge_count == 0 and empty.pre_idx.dtype == np.int32
    return report(checks, verbose)


def units_smoke_test(verbose=True):
    checks = {}
    for step in (k.lif_step, k.adaptive_lif_step, k.glif3_step, k.glif4_step, k.glif5_step):
        pool = k.new_population(2)
        pool[:, k.V] = [-55, -65]
        spikes = step(pool, [0, 100], 1)
        # Independent hand calculation: 100 pA / 200 pF = 0.5 mV/ms.
        checks[f"{step.__name__}_voltage"] = np.allclose(pool[:, k.V], [-55.5, -64.5], atol=1e-5, rtol=0)
        checks[f"{step.__name__}_subthreshold"] = np.array_equal(spikes, [0, 0])
    lif = k.new_population(1)
    lif[:, k.V] = -55
    adex = lif.copy()
    adex[:, k.V_T] = 100  # isolate passive leak from exponential initiation
    k.lif_step(lif, 0, 1)
    k.adex_step(adex, 0, 1)
    checks["passive_lif_adex_agreement"] = np.allclose(lif[:, k.V], adex[:, k.V], atol=1e-5, rtol=0)
    exact = -65 + 10 * np.exp(-1)
    errors = []
    for dt in (1, 0.5, 0.1):
        pool = k.new_population(1)
        pool[:, k.V] = -55
        for _ in range(round(20 / dt)):
            k.lif_step(pool, 0, dt)
        errors.append(abs(float(pool[0, k.V]) - exact))
    checks["timestep_convergence"] = errors[0] > errors[1] > errors[2] and errors[2] < 0.01
    return report(checks, verbose)


def input_smoke_test(verbose=True):
    checks = {}
    invalid_dt = [0, -0.1, np.nan, np.inf, 1e40, 1e-50, [0.1], True, "0.1", 1j]
    invalid_current = [np.nan, [0, np.inf], 1e40, [1], "1", [1j, 0]]
    for name, step in NEURON_STEP_REGISTRY.items():
        for argument, invalid in (("dt", invalid_dt), ("current", invalid_current)):
            for index, value in enumerate(invalid):
                pool = k.new_population(2)
                pool[:, k.REFRACTORY] = 1
                before = pool.copy()
                current, dt = (0, value) if argument == "dt" else (value, 0.1)
                prefix = f"{name.value}_{argument}_{index}"
                checks[f"{prefix}_rejected"] = rejects((TypeError, ValueError), step, pool, current, dt)
                checks[f"{prefix}_state_intact"] = np.array_equal(pool, before)
        # Also exercise a valid scalar and vector, so unconditional rejection fails.
        a, b = k.new_population(2), k.new_population(2)
        sa, sb = step(a, 0, 0.1), step(b, [0, 0], 0.1)
        checks[f"{name.value}_valid_broadcast"] = np.array_equal(a, b) and np.array_equal(sa, sb)
    return report(checks, verbose)
