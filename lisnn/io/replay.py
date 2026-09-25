"""Bounded literal replay of transduced sensory stimulation."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from lisnn.io.contracts import CapacityError, StreamFrame
from lisnn.io.regimes import RegimeController
from lisnn.io.runtime import RoutedTick
from lisnn.validation import scalar32


@dataclass(frozen=True)
class ReplayEntry:
    modality: str
    original_tick: int
    timestamp_ms: float
    external_port: str
    mapping_provenance: str
    channel_current_pA: np.ndarray


class ReplayBuffer:
    """Record a bounded recent sequence of delivered EXTERNAL sensory pA."""

    def __init__(self, *, max_frames, max_duration_ms):
        if isinstance(max_frames, bool) or not isinstance(max_frames, (int, np.integer)) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self.max_frames = int(max_frames)
        self.max_duration_ms = scalar32("max_duration_ms", max_duration_ms, positive=True)
        self._entries = deque(maxlen=self.max_frames)

    @property
    def entries(self):
        return tuple(ReplayEntry(e.modality, e.original_tick, e.timestamp_ms,
                                 e.external_port, e.mapping_provenance,
                                 e.channel_current_pA.copy()) for e in self._entries)

    def record(self, routed_tick):
        if not isinstance(routed_tick, RoutedTick):
            raise TypeError("record requires a RoutedTick")
        for delivery in routed_tick.deliveries:
            if delivery.source != "EXTERNAL" or delivery.modality not in ("auditory", "visual"):
                continue
            if self._entries and delivery.delivery_tick < self._entries[-1].original_tick:
                raise ValueError("recorded deliveries must be chronological")
            self._entries.append(ReplayEntry(delivery.modality, delivery.delivery_tick,
                                            delivery.timestamp_ms, delivery.port, delivery.mode,
                                            delivery.channel_current_pA.copy()))
            while (self._entries and delivery.timestamp_ms - self._entries[0].timestamp_ms >
                   self.max_duration_ms):
                self._entries.popleft()

    def snapshot(self):
        return deepcopy(self)


class ReplayController:
    """Schedule bounded ordered/shuffled literal samples through INTERNAL ports."""

    def __init__(self, buffer, *, internal_ports, max_gain, max_repeats,
                 max_events, max_duration_ticks, seed=0):
        if not isinstance(buffer, ReplayBuffer):
            raise TypeError("buffer must be ReplayBuffer")
        self.buffer = buffer
        self.internal_ports = dict(internal_ports)
        self.max_gain = scalar32("max_gain", max_gain, positive=True)
        for name, value in (("max_repeats", max_repeats), ("max_events", max_events),
                            ("max_duration_ticks", max_duration_ticks)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_repeats = int(max_repeats)
        self.max_events = int(max_events)
        self.max_duration_ticks = int(max_duration_ticks)
        self._rng = np.random.default_rng(seed)

    def schedule(self, regime, *, observed_tick, start_tick, repeats=1,
                 gain=1, order="ordered"):
        if not isinstance(regime, RegimeController) or regime.current.name != "SLEEP":
            raise ValueError("literal replay requires a SLEEP regime controller")
        if order not in ("ordered", "shuffled"):
            raise ValueError("order must be ordered or shuffled")
        if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or not 1 <= repeats <= self.max_repeats:
            raise ValueError("repeats exceed declared budget")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (observed_tick, start_tick)) or start_tick <= observed_tick or start_tick < regime.runtime.tick:
            raise ValueError("replay must start on a future unexecuted tick")
        g = scalar32("gain", gain, nonnegative=True)
        if g > self.max_gain:
            raise ValueError("replay gain exceeds declared budget")
        entries = self.buffer.entries
        if not entries:
            raise ValueError("cannot replay an empty buffer")
        if len(entries) * repeats > self.max_events:
            raise CapacityError("replay event count exceeds declared budget")
        first, last = entries[0].original_tick, entries[-1].original_tick
        span = last - first + 1
        if span * repeats > self.max_duration_ticks:
            raise CapacityError("replay duration exceeds declared tick budget")
        runtime = regime.runtime
        dt = float(runtime.network.runtime.dt if hasattr(runtime.network.runtime, "dt")
                   else runtime.network.runtime._runtime.dt)
        # Plan/validate every item before enqueuing any of them.
        planned = []
        original_rng = deepcopy(self._rng)
        original_queue = deepcopy(runtime.scheduler)
        try:
            for repeat in range(repeats):
                ordered = list(entries)
                if order == "shuffled":
                    ordered = list(self._rng.permutation(ordered))
                for entry, slot in zip(ordered, entries):
                    tick = start_tick + repeat * span + (slot.original_tick - first)
                    name = self.internal_ports.get(entry.modality)
                    if name not in runtime.transducers or runtime.transducers[name].port.source != "INTERNAL":
                        raise ValueError(f"no declared INTERNAL {entry.modality} port")
                    values = entry.channel_current_pA * g
                    frame = StreamFrame("INTERNAL", "CONTINUOUS", entry.modality, values,
                                        ("channel",), tick * dt,
                                        f"literal_replay:{entry.external_port}:tick={entry.original_tick}:"
                                        f"{entry.mapping_provenance}")
                    runtime.transducers[name].port.validate_capacity(frame)
                    planned.append((frame, name))
            active_transitions = [tick for tick, target in regime.transitions.items()
                                  if target == "ACTIVE"]
            if active_transitions and planned[-1][0].timestamp_ms >= min(active_transitions) * dt:
                raise ValueError("replay schedule would cross into ACTIVE regime")
            queued = sum(map(len, runtime.scheduler._frames.values()))
            if queued + len(planned) > runtime.scheduler.max_frames:
                raise CapacityError("replay exceeds bounded frame queue")
            for frame, name in planned:
                runtime.schedule_frame(frame, name, observed_tick=observed_tick)
        except Exception:
            self._rng = original_rng
            runtime.scheduler = original_queue
            raise
        return tuple((name, runtime.timestamp_to_tick(frame.timestamp_ms)) for frame, name in planned)

    def snapshot(self):
        return deepcopy(self)
