"""Reproducibility and allocation accounting for the profiling harness."""

import numpy as np

from benchmarks.profile_playground import (Config, OPERATIONS, array_inventory,
                                           fingerprint, measure, workload)


def test_inventory_counts_array_backing_once_through_aliases():
    array = np.zeros((4, 8), dtype=np.float32)
    inventory = array_inventory({"full": array, "view": array[:, 0], "alias": array})
    assert inventory["array_count"] == 1
    assert inventory["unique_backing_bytes"] == array.nbytes


def test_every_workload_restarts_reproducibly():
    config = Config(8, 2)
    for operation in OPERATIONS:
        a, _ = workload(config, operation)
        b, _ = workload(config, operation)
        for tick in range(3):
            assert fingerprint(a(tick)) == fingerprint(b(tick)), operation


def test_measure_separates_initialization_and_diagnostics():
    result = measure(Config(8, 2), "adaptive_tick", steps=2, repeats=2,
                     warmup=1, diagnostics=True)
    assert len(result["seconds_per_operation"]) == 2
    assert len(result["initialization_seconds"]) == 2
    assert result["median_seconds"] > 0
    assert result["allocations"]["peak_bytes"] >= result["allocations"]["retained_bytes"]
    assert result["arrays_after_timing"]["unique_backing_bytes"] > 0
    assert len(result["final_observation_sha256"]) == 64
    assert result["profile_top_own_time"]
