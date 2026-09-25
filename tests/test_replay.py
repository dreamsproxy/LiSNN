"""Bounded literal auditory/visual replay through distinct internal ports."""

import numpy as np
import pytest

from lisnn.io import (CapacityError, InterfacePort, RegimeController,
                      ReplayBuffer, ReplayController, StreamFrame,
                      StreamRuntime, Transducer)
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def setup(max_queue=20):
    model = create_nn(4)
    model.configure_runtime(create_synapses(4, [], []), dt=1, impulse_scale=100)
    ports = {name: Transducer(InterfacePort(name, 4, indices, source=source))
             for name, source, indices in (
                 ("microphone", "EXTERNAL", [0, 1]),
                 ("receptors", "EXTERNAL", [2, 3]),
                 ("audio_internal", "INTERNAL", [2, 3]),
                 ("visual_internal", "INTERNAL", [0, 1]))}
    runtime = StreamRuntime(model, transducers=ports, max_queued_frames=max_queue)
    return RegimeController(runtime)


def record_known_experience(regime, *, max_frames=3, max_duration_ms=10):
    buffer = ReplayBuffer(max_frames=max_frames, max_duration_ms=max_duration_ms)
    frames = (("microphone", "auditory", [10, 20]),
              ("receptors", "visual", [30, 40]),
              ("microphone", "auditory", [50, 60]))
    for tick, (port, modality, values) in enumerate(frames):
        regime.runtime.schedule_frame(StreamFrame("EXTERNAL", "CONTINUOUS", modality,
                                                  values, ("channel",), tick,
                                                  f"recorded:{port}"), port)
        buffer.record(regime.step().routed)
    regime.transition(regime.runtime.tick, "SLEEP")
    regime.step()
    return buffer


def replay(buffer):
    return ReplayController(buffer, internal_ports={"auditory": "audio_internal",
                                                     "visual": "visual_internal"},
                            max_gain=1, max_repeats=2, max_events=6,
                            max_duration_ticks=6, seed=42)


def test_ordered_replay_known_values_distinct_internal_path_and_no_reset():
    regime = setup()
    buffer = record_known_experience(regime)
    before = regime.runtime.network.runtime.pool.copy()
    recorded = buffer.entries
    assert [(x.modality, x.external_port) for x in recorded] == [
        ("auditory", "microphone"), ("visual", "receptors"),
        ("auditory", "microphone")]
    start = regime.runtime.tick + 1
    plan = replay(buffer).schedule(regime, observed_tick=regime.runtime.tick - 1,
                                   start_tick=start, gain=0.5)
    assert plan == (("audio_internal", start), ("visual_internal", start + 1),
                    ("audio_internal", start + 2))
    assert regime.step().routed.deliveries == ()
    expected = ([0, 0, 5, 10], [15, 20, 0, 0], [0, 0, 25, 30])
    for currents in expected:
        result = regime.step().routed
        np.testing.assert_array_equal(result.neural.feedback_current_pA, currents)
        assert len(result.deliveries) == 1 and result.deliveries[0].source == "INTERNAL"
        assert result.deliveries[0].mode.startswith("literal_replay:")
        np.testing.assert_array_equal(result.neural.external_current_pA, np.zeros(4))
    assert not np.array_equal(regime.runtime.network.runtime.pool, before)


def test_bounded_recording_and_owned_snapshots():
    regime = setup()
    buffer = record_known_experience(regime, max_frames=2, max_duration_ms=1)
    assert len(buffer.entries) == 2
    buffer.entries[0].channel_current_pA[:] = 99
    np.testing.assert_array_equal(buffer.entries[0].channel_current_pA, [30, 40])
    assert len(buffer.snapshot().entries) == 2


def test_budget_failures_and_no_partial_queue():
    regime = setup(max_queue=2)
    buffer = record_known_experience(regime)
    rule = replay(buffer)
    start = regime.runtime.tick + 1
    with pytest.raises(CapacityError):
        rule.schedule(regime, observed_tick=regime.runtime.tick - 1,
                      start_tick=start, repeats=2)
    assert sum(map(len, regime.runtime.scheduler._frames.values())) == 0
    with pytest.raises(ValueError):
        rule.schedule(regime, observed_tick=start, start_tick=start)
    with pytest.raises(ValueError):
        rule.schedule(regime, observed_tick=regime.runtime.tick - 1,
                      start_tick=start, gain=2)
    regime.transition(start + 1, "ACTIVE")
    with pytest.raises(ValueError, match="cross into ACTIVE"):
        rule.schedule(regime, observed_tick=regime.runtime.tick - 1, start_tick=start)
    assert sum(map(len, regime.runtime.scheduler._frames.values())) == 0


def test_seeded_shuffle_budget_and_no_recursive_recording():
    a, b = setup(), setup()
    record_a, record_b = record_known_experience(a), record_known_experience(b)
    ra, rb = replay(record_a), replay(record_b)
    start = a.runtime.tick + 1
    assert ra.schedule(a, observed_tick=start - 2, start_tick=start,
                       order="shuffled", repeats=2) == rb.schedule(
                           b, observed_tick=start - 2, start_tick=start,
                           order="shuffled", repeats=2)
    for _ in range(7):
        out_a, out_b = a.step().routed, b.step().routed
        np.testing.assert_array_equal(out_a.neural.feedback_current_pA,
                                      out_b.neural.feedback_current_pA)
        record_a.record(out_a)
    assert len(record_a.entries) == 3  # INTERNAL replay never re-enters experience buffer.
