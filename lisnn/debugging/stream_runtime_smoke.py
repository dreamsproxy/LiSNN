"""External routing and future feedback smoke diagnostics."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.io import FeedbackPolicy, InterfacePort, StreamFrame, StreamRuntime, Transducer
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def stream_runtime_smoke_test(verbose=True):
    model = create_nn(2)
    model.configure_runtime(create_synapses(2, [], []), dt=1, impulse_scale=100)
    runtime = StreamRuntime(model, transducers={
        "sensory": Transducer(InterfacePort("sensory", 2, [0], source="EXTERNAL")),
        "feedback": Transducer(InterfacePort("feedback", 2, [1], source="INTERNAL"))})
    runtime.schedule_frame(StreamFrame("EXTERNAL", "CONTINUOUS", "auditory", [10],
                                       ("channel",), 0, "explicit test route"), "sensory")
    first = runtime.step()
    FeedbackPolicy(mode="contingent", port_name="feedback", success_pA=20,
                   error_pA=50, seed=1).schedule(runtime, observed_tick=0, success=False)
    second = runtime.step()
    checks = {
        "external_current": np.array_equal(first.neural.external_current_pA, [10, 0]),
        "future_feedback": np.array_equal(second.neural.feedback_current_pA, [0, 50]),
        "provenance": second.deliveries[0].source == "INTERNAL" and
                      second.deliveries[0].delivery_tick == 1,
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if stream_runtime_smoke_test()["passed"] else 1)
