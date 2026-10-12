"""ACTIVE/SLEEP transitions, routing, learning freeze and state preservation."""

import numpy as np
import pytest

from lisnn.io import (InterfacePort, RegimeController, RegimePolicy, StreamFrame,
                      StreamRuntime, Transducer)
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def controller(*, sleep=None, background_probability=0, background_max=0):
    model = create_nn(2)
    model.configure_runtime(create_synapses(2, [0], [1], [0.5]), dt=1,
                            impulse_scale=100, plasticity="pair",
                            plasticity_params=dict(tau_plus_ms=10, tau_minus_ms=10,
                                                   a_plus=0.1, a_minus=0.1, w_max=1))
    ports = {
        "world": Transducer(InterfacePort("world", 2, [0], source="EXTERNAL")),
        "internal": Transducer(InterfacePort("internal", 2, [0, 1], source="INTERNAL"))}
    runtime = StreamRuntime(model, transducers=ports, max_queued_frames=10)
    return RegimeController(runtime, sleep=sleep, background_port="internal",
                            background_probability=background_probability,
                            background_amplitude_pA=5,
                            max_background_events=background_max, seed=7)


def test_transitions_at_ticks_and_gain_without_implicit_reset():
    c = controller()
    c.transition(1, "SLEEP")
    c.transition(3, "ACTIVE")
    t0 = c.step([100, 0])
    before = c.runtime.network.runtime.pool.copy()
    t1 = c.step([100, 0])
    np.testing.assert_array_equal(t0.routed.neural.external_current_pA, [100, 0])
    np.testing.assert_array_equal(t1.routed.neural.external_current_pA, [5, 0])
    assert t1.regime.name == "SLEEP" and not t1.regime.actions_enabled
    assert t1.regime.plasticity_enabled
    assert c.runtime.network.runtime.tick == 2
    np.testing.assert_array_equal(c.runtime.network.runtime.pool, before)
    assert c.runtime.network.runtime.pool[0, 0] > -65
    c.step()
    t3 = c.step([100, 0])
    np.testing.assert_array_equal(t3.routed.neural.external_current_pA, [100, 0])
    assert t3.regime.actions_enabled and t3.regime.name == "ACTIVE"


def test_sleep_pulse_and_bounded_seeded_background():
    a, b = controller(background_probability=1, background_max=2), controller(background_probability=1, background_max=2)
    for c in (a, b):
        c.transition(0, "SLEEP")
        c.step()  # Schedules one sparse internal event on tick 1.
        c.pulse(port_name="internal", current_pA=[0, 10])
    for _ in range(3):
        out_a, out_b = a.step(), b.step()
        np.testing.assert_array_equal(out_a.routed.neural.feedback_current_pA,
                                      out_b.routed.neural.feedback_current_pA)
    assert a.background_count == 2
    assert any(d.mode == "outcome:sleep_pulse" for d in a.runtime.deliveries)
    assert any(d.mode == "outcome:sleep_background" for d in a.runtime.deliveries)


def test_learning_policy_freezes_weight_but_advances_traces():
    model = create_nn(2)
    model.configure_runtime(create_synapses(2, [0], [1], [0.5]), dt=1,
                            impulse_scale=100, plasticity="pair",
                            plasticity_params=dict(tau_plus_ms=10, tau_minus_ms=10,
                                                   a_plus=0.1, a_minus=0.1, w_max=1))
    model.runtime._runtime._previous[:] = [1, 0]
    model.runtime._learner._pre_trace[:] = [1, 0]
    model.pool[1, 0] = -50.1
    # Explicit learner freeze does not discard ongoing neural/trace dynamics.
    before = model.runtime.weights.copy()
    out = model.step([0, 32000], plasticity_enabled=False)
    np.testing.assert_array_equal(out.weights_after, before)
    np.testing.assert_array_equal(out.plasticity_update.applied_change, [0])
    assert out.trace_state[1][1] > 0


def test_forbidden_source_and_snapshot_restore_rng_queue_policy():
    sleep = RegimePolicy("SLEEP", 0, 1, False, False, ("INTERNAL",))
    c = controller(sleep=sleep)
    c.transition(0, "SLEEP")
    frame = StreamFrame("EXTERNAL", "CONTINUOUS", "generic", [10],
                        ("channel",), 0, "explicit")
    c.runtime.schedule_frame(frame, "world")
    with pytest.raises(ValueError, match="forbidden"):
        c.step()
    assert c.runtime.tick == 0
    saved = c.snapshot()
    c.runtime.scheduler.consume(0)
    c.step()
    c.restore(saved)
    assert c.runtime.tick == 0 and c.transitions[0] == "SLEEP"
