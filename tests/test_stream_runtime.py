"""External routes, explicit rate-to-tick timing and feedback controls."""

import numpy as np
import pytest

from lisnn.io import (CapacityError, FeedbackPolicy, InterfacePort,
                      StreamFrame, StreamRuntime, Transducer)
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def setup():
    network = create_nn(4)
    network.configure_runtime(create_synapses(4, [], []), dt=1, impulse_scale=100)
    ports = {
        "auditory": Transducer(InterfacePort("auditory", 4, [0, 1], source="EXTERNAL")),
        "visual": Transducer(InterfacePort("visual", 4, [2, 3], source="EXTERNAL")),
        "events": Transducer(InterfacePort("events", 4, [1, 3], source="EXTERNAL"), event_amplitude_pA=100),
        "feedback": Transducer(InterfacePort("feedback", 4, [0, 1], source="INTERNAL")),
    }
    return StreamRuntime(network, transducers=ports, max_queued_frames=12)


def frame(source, kind, modality, values, time, period=None, provenance="test"):
    axes = ("time", "channel") if period is not None else ("channel",)
    return StreamFrame(source, kind, modality, np.asarray(values), axes, time,
                       provenance, sample_period_ms=period)


def test_waveform_visual_events_and_direct_current_map_without_inference():
    routed = setup()
    routed.schedule_frame(frame("EXTERNAL", "CONTINUOUS", "auditory", [[1, 2], [3, 4]], 0, 1), "auditory")
    routed.schedule_frame(frame("EXTERNAL", "CONTINUOUS", "visual", [5, 6], 1), "visual")
    routed.schedule_frame(frame("EXTERNAL", "EVENT", "generic", [1, 0], 1), "events")
    t0 = routed.step([10, 0, 0, 0])
    np.testing.assert_array_equal(t0.neural.external_current_pA, [11, 2, 0, 0])
    t1 = routed.step()
    np.testing.assert_array_equal(t1.neural.external_current_pA, [3, 104, 5, 6])
    assert [(x.source, x.modality, x.delivery_tick) for x in t1.deliveries] == [
        ("EXTERNAL", "auditory", 1), ("EXTERNAL", "visual", 1),
        ("EXTERNAL", "generic", 1)]
    assert t1.deliveries[2].nonzero_channels == 1


def test_capacity_alignment_source_and_partial_schedule_are_rejected():
    routed = setup()
    with pytest.raises(ValueError, match="does not align"):
        routed.schedule_frame(frame("EXTERNAL", "CONTINUOUS", "auditory", [1, 2], 0.5), "auditory")
    with pytest.raises(CapacityError):
        routed.schedule_frame(frame("EXTERNAL", "CONTINUOUS", "auditory", [1, 2, 3], 0), "auditory")
    with pytest.raises(ValueError, match="sources must match"):
        routed.schedule_frame(frame("INTERNAL", "EVENT", "visual", [1, 0], 1), "visual", observed_tick=0)
    with pytest.raises(ValueError):
        routed.schedule_frame(frame("EXTERNAL", "CONTINUOUS", "auditory", [[1, 2], [3, 4]], 0, 0.5), "auditory")
    assert routed.step().deliveries == ()


def test_feedback_future_tick_log_and_replay_determinism():
    a, b = setup(), setup()
    policy_a = FeedbackPolicy(mode="contingent", port_name="feedback", success_pA=50, error_pA=100, seed=7)
    policy_b = FeedbackPolicy(mode="contingent", port_name="feedback", success_pA=50, error_pA=100, seed=7)
    for runtime, policy in ((a, policy_a), (b, policy_b)):
        runtime.step()
        policy.schedule(runtime, observed_tick=0, success=False)
        policy.schedule(runtime, observed_tick=0, success=True)
    ta, tb = a.step(), b.step()
    np.testing.assert_array_equal(ta.neural.feedback_current_pA, tb.neural.feedback_current_pA)
    assert len(ta.deliveries) == 2
    assert all(x.source == "INTERNAL" and x.delivery_tick == 1 and x.mode == "outcome:contingent"
               for x in ta.deliveries)
    np.testing.assert_array_equal(ta.neural.feedback_current_pA.sum(), 200)
    checkpoint = a.snapshot()
    assert a.step().deliveries == ()
    a.restore(checkpoint)
    assert a.tick == 2 and a.step().deliveries == ()


def test_control_modes_random_yoked_inverted_none():
    a, b = setup(), setup()
    for mode in ("random", "inverted"):
        policy_a = FeedbackPolicy(mode=mode, port_name="feedback", success_pA=50, error_pA=100, seed=13)
        policy_b = FeedbackPolicy(mode=mode, port_name="feedback", success_pA=50, error_pA=100, seed=13)
        for event in (True, False, True):
            np.testing.assert_array_equal(policy_a.schedule(a, observed_tick=0, success=event),
                                          policy_b.schedule(b, observed_tick=0, success=event))
    yoked = FeedbackPolicy(mode="yoked", port_name="feedback", success_pA=50,
                           error_pA=100, yoked_currents=[[12, 34]])
    np.testing.assert_array_equal(yoked.schedule(a, observed_tick=0, success=False), [12, 34])
    none = FeedbackPolicy(mode="none", port_name="feedback", success_pA=50, error_pA=100)
    assert none.schedule(a, observed_tick=0, success=True) is None
    with pytest.raises(ValueError):
        yoked.schedule(a, observed_tick=0, success=True)


def test_failed_neuron_step_preserves_scheduled_input():
    routed = setup()
    routed.schedule_frame(frame("EXTERNAL", "CONTINUOUS", "auditory", [1, 2], 0), "auditory")
    with pytest.raises(ValueError):
        routed.step([np.nan, 0, 0, 0])
    assert len(routed.scheduler._frames[0]) == 1
    np.testing.assert_array_equal(routed.step().neural.external_current_pA, [1, 2, 0, 0])
