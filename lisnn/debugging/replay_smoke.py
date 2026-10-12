"""Small known auditory/visual literal replay smoke test."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.io import (InterfacePort, RegimeController, ReplayBuffer,
                      ReplayController, StreamFrame, StreamRuntime, Transducer)
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def replay_smoke_test(verbose=True):
    model = create_nn(2)
    model.configure_runtime(create_synapses(2, [], []), dt=1, impulse_scale=100)
    ports = {"world": Transducer(InterfacePort("world", 2, [0], source="EXTERNAL")),
             "inside": Transducer(InterfacePort("inside", 2, [1], source="INTERNAL"))}
    regime = RegimeController(StreamRuntime(model, transducers=ports))
    buffer = ReplayBuffer(max_frames=2, max_duration_ms=3)
    for tick, modality in enumerate(("auditory", "visual")):
        regime.runtime.schedule_frame(StreamFrame("EXTERNAL", "CONTINUOUS", modality,
                                                  [10 + 10 * tick], ("channel",), tick,
                                                  "known source"), "world")
        buffer.record(regime.step().routed)
    regime.transition(2, "SLEEP")
    regime.step()
    controller = ReplayController(buffer, internal_ports={"auditory": "inside", "visual": "inside"},
                                  max_gain=1, max_repeats=1, max_events=2,
                                  max_duration_ticks=2)
    controller.schedule(regime, observed_tick=2, start_tick=4, gain=0.5)
    regime.step()
    first, second = regime.step().routed, regime.step().routed
    checks = {
        "ordered_values": np.allclose(first.neural.feedback_current_pA, [0, 5]) and
                          np.allclose(second.neural.feedback_current_pA, [0, 10]),
        "internal_source": all(d.source == "INTERNAL" for d in first.deliveries + second.deliveries),
        "no_recursive_record": len(buffer.entries) == 2,
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if replay_smoke_test()["passed"] else 1)
