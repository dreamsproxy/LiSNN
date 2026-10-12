"""Steady-state history and replay-payload benchmark checks."""
from benchmarks.profile_playground import Config, fingerprint, workload


def test_bounded_workloads_start_full_and_replay_hash_includes_payload():
    config = Config(8, 2)
    run, state = workload(config, "trace_history")
    assert len(state["traces"].history) == 32
    run, state = workload(config, "replay_record")
    assert len(state["buffer"].entries) == 32
    first = fingerprint(run(0))
    state["result"].deliveries[0].channel_current_pA[:] += 1
    assert fingerprint(run(1)) != first
