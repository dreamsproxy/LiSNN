"""Semantic, memory and optional timing gates fail for the intended defects."""

from copy import deepcopy

import numpy as np
import pytest

from benchmarks.profile_playground import Config, array_inventory, workload
from benchmarks.regression_gate import assess
from lisnn.synapses import core, topology_validation


def test_reference_selection_is_nested_scoped_and_validated(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "_reference_duplicate_edges", lambda *a: calls.append("reference"))
    monkeypatch.setattr(core, "_optimized_duplicate_edges", lambda *a: calls.append("optimized"))
    with topology_validation("reference"):
        core._has_duplicate_edges(None, None, None)
        with topology_validation("optimized"):
            core._has_duplicate_edges(None, None, None)
        with pytest.raises(RuntimeError):
            with topology_validation("optimized"):
                raise RuntimeError("exercise restoration")
        core._has_duplicate_edges(None, None, None)
    assert calls == ["reference", "optimized", "reference"]
    with pytest.raises(ValueError):
        with topology_validation("skip_validation"):
            pass


@pytest.mark.parametrize("operation,key", [("trace_history", "traces"), ("replay_record", "buffer")])
def test_bounded_recording_retains_constant_array_storage(operation, key):
    run, state = workload(Config(32, 4), operation)
    before = array_inventory(state[key])
    for tick in range(500):
        run(tick)
    after = array_inventory(state[key])
    assert before["array_count"] == after["array_count"]
    assert before["unique_backing_bytes"] == after["unique_backing_bytes"]
    assert after["unique_backing_bytes"] <= 32 * 32 * 4 * 4 + 2 * 32 * 4


def test_gate_detects_semantic_memory_and_opt_in_timing_regressions():
    row = dict(config={"seed": 7}, operation="adaptive_tick",
               final_observation_sha256="same", median_seconds=1,
               arrays_after_timing={"unique_backing_bytes": 100},
               allocations={"peak_bytes": 100000})
    assert assess(row, row)["passed"]
    changed = deepcopy(row)
    changed["final_observation_sha256"] = "different"
    assert assess(row, changed)["failures"] == ["observation mismatch"]
    changed = deepcopy(row)
    changed["arrays_after_timing"]["unique_backing_bytes"] += 4
    assert "persistent array footprint changed" in assess(row, changed)["failures"]
    changed = deepcopy(row)
    changed["allocations"]["peak_bytes"] = 1000000
    assert "temporary allocation peak" in assess(row, changed)["failures"][0]
    changed = deepcopy(row)
    changed["median_seconds"] = 2
    assert assess(row, changed)["passed"]
    assert not assess(row, changed, enforce_timing=True)["passed"]
