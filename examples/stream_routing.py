"""Editable, hardware-free auditory, visual, event and feedback routing."""

from lisnn.io import (FeedbackPolicy, InterfacePort, StreamFrame,
                      StreamRuntime, Transducer)
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def main():
    network = create_nn(4)
    network.configure_runtime(create_synapses(4, [], []), dt=1, impulse_scale=100)
    ports = {p.name: Transducer(p, event_amplitude_pA=100) for p in (
        InterfacePort("microphone", 4, [0, 1], source="EXTERNAL"),
        InterfacePort("receptors", 4, [2, 3], source="EXTERNAL"),
        InterfacePort("sparse_events", 4, [1, 3], source="EXTERNAL"),
        InterfacePort("feedback", 4, [0, 1], source="INTERNAL"))}
    runtime = StreamRuntime(network, transducers=ports)
    auditory = StreamFrame("EXTERNAL", "CONTINUOUS", "auditory",
                           [[50, 20], [30, 10]], ("time", "channel"), 0,
                           "microphone sample channels", sample_period_ms=1)
    visual = StreamFrame("EXTERNAL", "CONTINUOUS", "visual", [10, 20],
                         ("channel",), 1, "two explicit receptor sites")
    events = StreamFrame("EXTERNAL", "EVENT", "generic", [1, 0],
                         ("channel",), 1, "named sparse stimulation port")
    for sample, port in ((auditory, "microphone"), (visual, "receptors"),
                         (events, "sparse_events")):
        runtime.schedule_frame(sample, port)
    policy = FeedbackPolicy(mode="contingent", port_name="feedback",
                            success_pA=20, error_pA=50, seed=7)
    for tick in range(3):
        result = runtime.step()
        print(tick, result.neural.external_current_pA.tolist(),
              result.neural.feedback_current_pA.tolist(),
              [(d.source, d.port, d.delivery_tick) for d in result.deliveries])
        if tick == 1:
            # Outcome is defined by the experiment, not inferred in the kernel.
            policy.schedule(runtime, observed_tick=tick, success=False)


if __name__ == "__main__":
    main()
