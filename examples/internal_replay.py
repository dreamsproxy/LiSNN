"""Literal bounded auditory and visual replay through separate internal ports."""

from lisnn.io import (InterfacePort, RegimeController, ReplayBuffer,
                      ReplayController, StreamFrame, StreamRuntime, Transducer)
from lisnn.network import create_nn
from lisnn.synapses import create_synapses


def main():
    model = create_nn(4)
    model.configure_runtime(create_synapses(4, [], []), dt=1, impulse_scale=100)
    ports = {p.name: Transducer(p) for p in (
        InterfacePort("microphone", 4, [0, 1], source="EXTERNAL"),
        InterfacePort("visual", 4, [2, 3], source="EXTERNAL"),
        InterfacePort("auditory_internal", 4, [2, 3], source="INTERNAL"),
        InterfacePort("visual_internal", 4, [0, 1], source="INTERNAL"))}
    regime = RegimeController(StreamRuntime(model, transducers=ports))
    experience = ReplayBuffer(max_frames=4, max_duration_ms=5)
    for tick, (port, modality, values) in enumerate((
        ("microphone", "auditory", [10, 20]),
        ("visual", "visual", [30, 40]))):
        regime.runtime.schedule_frame(StreamFrame("EXTERNAL", "CONTINUOUS",
                                                  modality, values, ("channel",), tick,
                                                  f"experiment:{port}"), port)
        experience.record(regime.step().routed)
    regime.transition(regime.runtime.tick, "SLEEP")
    regime.step()  # First quiet tick.
    replay = ReplayController(experience,
                              internal_ports={"auditory": "auditory_internal",
                                              "visual": "visual_internal"},
                              max_gain=1, max_repeats=1, max_events=2,
                              max_duration_ticks=2, seed=7)
    replay.schedule(regime, observed_tick=regime.runtime.tick - 1,
                    start_tick=regime.runtime.tick + 1, gain=0.5)
    for _ in range(3):
        tick = regime.step()
        print(tick.routed.neural.tick, tick.regime.name,
              tick.routed.neural.feedback_current_pA.tolist(),
              [(d.source, d.modality, d.port) for d in tick.routed.deliveries])


if __name__ == "__main__":
    main()
