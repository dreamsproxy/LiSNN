"""Deterministic stream-to-current routing around the recurrent network."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from lisnn.io.contracts import CapacityError, StreamFrame, StreamScheduler, Transducer
from lisnn.network.core import SNN
from lisnn.validation import scalar32, vector32


@dataclass(frozen=True)
class Delivery:
    source: str
    port: str
    kind: str
    modality: str
    delivery_tick: int
    timestamp_ms: float
    mode: str
    nonzero_channels: int
    total_abs_current_pA: float
    channel_current_pA: np.ndarray


@dataclass(frozen=True)
class RoutedTick:
    neural: object
    deliveries: tuple[Delivery, ...]


class StreamRuntime:
    """Route declared stream samples and feedback to a configured SNN.

    The mapping is exact tick alignment within float32 representation error;
    no interpolation or implicit time rounding. The caller supplies frame
    times/sample periods and separate external/internal ports.
    """

    def __init__(self, network, *, transducers, max_queued_frames=1024):
        if not isinstance(network, SNN):
            raise TypeError("network must be an SNN")
        self.network = network
        self.network.runtime  # Fail early if no recurrent runtime configured.
        self.transducers = dict(transducers)
        if not self.transducers or any(not isinstance(t, Transducer) or key != t.port.name
                                       for key, t in self.transducers.items()):
            raise ValueError("transducers must map port names to transducers")
        if any(t.port.population_size != network.population_size for t in self.transducers.values()):
            raise ValueError("port and network population sizes must match")
        self.scheduler = StreamScheduler(max_queued_frames)
        self.deliveries = []

    @property
    def tick(self):
        return self.network.runtime.tick

    def timestamp_to_tick(self, timestamp_ms):
        dt = float(self.network.runtime.dt if hasattr(self.network.runtime, "dt")
                   else self.network.runtime._runtime.dt)
        relative = float(timestamp_ms) / dt
        tick = round(relative)
        if tick < 0 or not np.isclose(relative, tick, rtol=0, atol=2e-5):
            raise ValueError(f"timestamp {timestamp_ms} ms does not align to dt={dt} ms; "
                             "provide explicit resampling in the experiment")
        return tick

    def schedule_frame(self, frame, port_name, *, observed_tick=None):
        if not isinstance(frame, StreamFrame) or port_name not in self.transducers:
            raise ValueError("known port and StreamFrame required")
        transducer = self.transducers[port_name]
        transducer.port.validate_capacity(frame)
        samples = (frame,) if frame.axes == ("channel",) else tuple(
            frame.sample(i) for i in range(frame.values.shape[frame.axes.index("time")]))
        # Validate every sample, tick and queue capacity before committing any.
        entries = []
        for sample in samples:
            current = transducer.to_current(sample)
            tick = self.timestamp_to_tick(sample.timestamp_ms)
            if tick < self.tick:
                raise ValueError("cannot submit frame for an executed tick")
            if sample.source == "INTERNAL":
                if observed_tick is None or tick <= observed_tick:
                    raise ValueError("internal frame must target a future tick after observation")
            entries.append((sample, tick, current))
        if sum(map(len, self.scheduler._frames.values())) + len(entries) > self.scheduler.max_frames:
            raise CapacityError("routing exceeds declared queued-frame capacity")
        for sample, tick, _ in entries:
            self.scheduler.schedule(sample, tick,
                                    observed_tick=(-1 if observed_tick is None else observed_tick),
                                    metadata=port_name)

    def schedule_feedback(self, observed_tick, current_pA, *, port_name, delivery_tick, mode):
        """Schedule outcome-derived current on an INTERNAL port in the future."""
        if port_name not in self.transducers or self.transducers[port_name].port.source != "INTERNAL":
            raise ValueError("feedback requires a configured INTERNAL port")
        port = self.transducers[port_name].port
        values = vector32("feedback_current_pA", current_pA, len(port.neuron_indices), scalar=False)
        dt = float(self.network.runtime.dt if hasattr(self.network.runtime, "dt")
                   else self.network.runtime._runtime.dt)
        frame = StreamFrame("INTERNAL", "MODULATORY", "generic", values,
                            ("channel",), float(delivery_tick) * dt,
                            f"outcome:{mode}", destination="feedback_current", effect="add_pA")
        self.schedule_frame(frame, port_name, observed_tick=observed_tick)

    def step(self, external_current=0, *, external_gain=1, internal_gain=1,
             plasticity_enabled=True):
        direct = vector32("external_current", external_current, self.network.population_size)
        eg = scalar32("external_gain", external_gain, nonnegative=True)
        ig = scalar32("internal_gain", internal_gain, nonnegative=True)
        scheduled = self.scheduler.consume(self.tick)
        ext = direct * eg
        feedback = np.zeros_like(ext)
        records = []
        try:
            with np.errstate(over="raise", invalid="raise"):
                for frame, name in scheduled:
                    current = self.transducers[name].to_current(frame) * (eg if frame.source == "EXTERNAL" else ig)
                    target = feedback if frame.source == "INTERNAL" or (frame.kind == "MODULATORY" and frame.destination == "feedback_current") else ext
                    target += current
                    records.append(Delivery(frame.source, name, frame.kind, frame.modality,
                                            self.tick, float(frame.timestamp_ms),
                                            frame.mapping_provenance, int(np.count_nonzero(current)),
                                            float(np.sum(np.abs(current), dtype=np.float64)),
                                            current[self.transducers[name].port.neuron_indices].copy()))
            result = self.network.step(ext, feedback, plasticity_enabled=plasticity_enabled)
        except Exception:
            self.scheduler._frames[self.tick] = list(scheduled)
            raise
        self.deliveries.extend(records)
        return RoutedTick(result, tuple(records))

    def snapshot(self):
        return deepcopy(self)

    def restore(self, snapshot):
        if not isinstance(snapshot, StreamRuntime) or snapshot.network.population_size != self.network.population_size:
            raise ValueError("snapshot must match stream runtime population")
        restored = deepcopy(snapshot)
        self.network.restore_runtime(restored.network.snapshot_runtime())
        self.scheduler = restored.scheduler
        self.deliveries = restored.deliveries
        self.transducers = restored.transducers
