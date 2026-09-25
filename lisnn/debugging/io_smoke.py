"""Independent generic stream, mapping and causality smoke suite."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.io import CapacityError, InterfacePort, StreamFrame, StreamScheduler, Transducer


def io_smoke_test(verbose=True):
    frame = StreamFrame("EXTERNAL", "CONTINUOUS", "auditory",
                        np.array([[10, 20], [30, 40]]), ("time", "channel"),
                        0, "two declared samples", sample_period_ms=1)
    port = InterfacePort("microphone", 3, [0, 2], source="EXTERNAL")
    current = Transducer(port).to_current(frame.sample(1))
    scheduler = StreamScheduler(1)
    internal = StreamFrame("INTERNAL", "EVENT", "visual", [1, 0],
                           ("channel",), 1, "explicit internal replay")
    scheduler.schedule(internal, 2, observed_tick=1)
    try:
        Transducer(InterfacePort("small", 3, [0], source="EXTERNAL")).to_current(frame.sample(1))
        rejects_capacity = False
    except CapacityError:
        rejects_capacity = True
    checks = {
        "timed_channels": frame.sample(1).timestamp_ms == 1 and
                          np.array_equal(current, [30, 0, 40]),
        "capacity_failure": rejects_capacity,
        "causal_internal_queue": scheduler.consume(1) == () and scheduler.consume(2) == (internal,),
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if io_smoke_test()["passed"] else 1)
