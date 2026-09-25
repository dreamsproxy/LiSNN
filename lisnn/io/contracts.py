"""Generic, experiment-neutral stream interface and causal frame scheduler."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from lisnn.validation import scalar32, vector32


class CapacityError(ValueError):
    """Declared channels cannot be mapped to the available neuron indices."""


@dataclass(frozen=True)
class StreamFrame:
    source: str  # EXTERNAL or INTERNAL
    kind: str  # CONTINUOUS, EVENT or MODULATORY
    modality: str
    values: np.ndarray
    axes: tuple[str, ...]
    timestamp_ms: float
    mapping_provenance: str
    sample_period_ms: float | None = None
    destination: str | None = None
    effect: str | None = None

    def __post_init__(self):
        if self.source not in ("EXTERNAL", "INTERNAL"):
            raise ValueError("frame source must be EXTERNAL or INTERNAL")
        if self.kind not in ("CONTINUOUS", "EVENT", "MODULATORY"):
            raise ValueError("unsupported frame kind")
        if not isinstance(self.modality, str) or not self.modality.strip():
            raise ValueError("modality must be explicit nonempty metadata")
        if not isinstance(self.mapping_provenance, str) or not self.mapping_provenance.strip():
            raise ValueError("mapping provenance must be explicit")
        raw = np.asarray(self.values)
        if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2) or not np.all(np.isfinite(raw)):
            raise ValueError("frame values must be finite real 1D or 2D samples")
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.array(raw, dtype=np.float32, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("frame values exceed float32 range")
        if self.kind == "EVENT" and not np.all((raw == 0) | (raw == 1)):
            raise ValueError("event frame values must be binary")
        if (not isinstance(self.axes, tuple) or len(self.axes) != raw.ndim or
                len(set(self.axes)) != len(self.axes) or self.axes.count("channel") != 1):
            raise ValueError("declare exactly one channel axis and one name per dimension")
        if raw.ndim == 2 and set(self.axes) != {"time", "channel"}:
            raise ValueError("2D frames require explicit time and channel axes")
        if (raw.ndim == 2) != (self.sample_period_ms is not None):
            raise ValueError("sample_period_ms is required exactly for time-series frames")
        timestamp = scalar32("timestamp_ms", self.timestamp_ms, nonnegative=True)
        if self.sample_period_ms is not None:
            object.__setattr__(self, "sample_period_ms", scalar32("sample_period_ms", self.sample_period_ms, positive=True))
        if self.kind == "MODULATORY":
            if not self.destination or not self.effect:
                raise ValueError("modulatory frames require destination and permitted effect")
        elif self.destination is not None or self.effect is not None:
            raise ValueError("only modulatory frames declare destination and effect")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "timestamp_ms", timestamp)

    @property
    def channel_count(self):
        return self.values.shape[self.axes.index("channel")]

    @property
    def duration_ms(self):
        return 0.0 if self.sample_period_ms is None else float(self.sample_period_ms) * self.values.shape[self.axes.index("time")]

    def sample(self, index):
        """Extract one timed sample without changing its modality or provenance."""
        if "time" not in self.axes:
            raise ValueError("frame contains no time axis")
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("sample index must be an integer")
        if index < 0 or index >= self.values.shape[self.axes.index("time")]:
            raise IndexError("sample index outside declared time axis")
        selected = np.take(self.values, index, axis=self.axes.index("time"))
        return StreamFrame(self.source, self.kind, self.modality, selected,
                           ("channel",), float(self.timestamp_ms) + index * float(self.sample_period_ms),
                           self.mapping_provenance, destination=self.destination, effect=self.effect)


class _Stream:
    kind = ""

    def __init__(self, frames):
        self.frames = tuple(frames)
        if not self.frames or any(not isinstance(f, StreamFrame) or f.kind != self.kind for f in self.frames):
            raise ValueError(f"{self.kind} stream requires nonempty matching frames")
        self.source = self.frames[0].source
        if any(f.source != self.source for f in self.frames):
            raise ValueError("a stream has exactly one declared source")
        if any(b.timestamp_ms < a.timestamp_ms + a.duration_ms for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("frames must be ordered and non-overlapping")


class ContinuousStream(_Stream):
    kind = "CONTINUOUS"


class EventStream(_Stream):
    kind = "EVENT"


class ModulatoryStream(_Stream):
    kind = "MODULATORY"


class InterfacePort:
    """Explicit channel-to-neuron membership, independent of neuron class."""

    def __init__(self, name, population_size, neuron_indices, *, source, role="input",
                 mapping_policy="one_to_one", biological_identity=None):
        if not isinstance(name, str) or not name:
            raise ValueError("port requires a name")
        if source not in ("EXTERNAL", "INTERNAL"):
            raise ValueError("port source must be EXTERNAL or INTERNAL")
        if role not in ("input", "readout", "overlap") or mapping_policy != "one_to_one":
            raise ValueError("unsupported port role or mapping policy")
        if isinstance(population_size, bool) or not isinstance(population_size, (int, np.integer)) or population_size <= 0:
            raise ValueError("population_size must be a positive integer")
        raw = np.asarray(neuron_indices)
        if raw.ndim != 1 or raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= population_size) or len(np.unique(raw)) != len(raw):
            raise ValueError("port neuron indices must be unique and in range")
        self.name, self.population_size, self.source, self.role = name, int(population_size), source, role
        self.mapping_policy, self.biological_identity = mapping_policy, biological_identity
        self._indices = raw.astype(np.int32, copy=True)

    @property
    def neuron_indices(self):
        return self._indices.copy()

    def validate_capacity(self, frame):
        if not isinstance(frame, StreamFrame) or frame.source != self.source:
            raise ValueError("frame and port sources must match")
        if frame.channel_count != len(self._indices):
            raise CapacityError(f"{frame.source} {frame.modality} {frame.values.shape} "
                                f"requires {frame.channel_count} channels; available {len(self._indices)} "
                                f"via {self.mapping_policy} on {self.name}; supply an explicit larger "
                                "port or experiment-defined projection")


class Transducer:
    """Project one declared sample to pA without guessing tensor semantics."""

    def __init__(self, port, *, event_amplitude_pA=None):
        if not isinstance(port, InterfacePort):
            raise TypeError("port must be an InterfacePort")
        self.port = port
        self.event_amplitude_pA = None if event_amplitude_pA is None else scalar32("event_amplitude_pA", event_amplitude_pA)

    def to_current(self, frame):
        self.port.validate_capacity(frame)
        if frame.axes != ("channel",):
            raise ValueError("select an explicit time sample before transduction")
        if frame.kind == "MODULATORY":
            if frame.destination not in ("external_current", "feedback_current") or frame.effect != "add_pA":
                raise ValueError("modulatory effect does not permit current injection")
            channels = frame.values
        elif frame.kind == "EVENT":
            if self.event_amplitude_pA is None:
                raise ValueError("event-to-current mapping requires explicit amplitude")
            channels = frame.values * self.event_amplitude_pA
        else:
            channels = frame.values
        output = np.zeros(self.port.population_size, dtype=np.float32)
        output[self.port.neuron_indices] = channels
        return output


class StreamScheduler:
    """Bounded FIFO frame queue; observed activity can only target future ticks."""

    def __init__(self, max_frames):
        if isinstance(max_frames, bool) or not isinstance(max_frames, (int, np.integer)) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self.max_frames = int(max_frames)
        self._frames = defaultdict(list)

    def schedule(self, frame, delivery_tick, *, observed_tick=-1, metadata=None):
        if not isinstance(frame, StreamFrame):
            raise TypeError("scheduled value must be a StreamFrame")
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (delivery_tick, observed_tick)) or delivery_tick <= observed_tick or delivery_tick < 0:
            raise ValueError("delivery tick must be a nonnegative future integer")
        if frame.source == "INTERNAL" and observed_tick < 0:
            raise ValueError("internal frame must declare its observed source tick")
        if sum(map(len, self._frames.values())) >= self.max_frames:
            raise CapacityError(f"frame queue full: required 1 more slot; available 0 of {self.max_frames}")
        self._frames[int(delivery_tick)].append(frame if metadata is None else (frame, metadata))

    def consume(self, tick):
        if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        return tuple(self._frames.pop(int(tick), ()))


class Probe:
    """Read selected neuron observations without mutating runtime or result."""

    def __init__(self, port):
        if not isinstance(port, InterfacePort):
            raise TypeError("port must be an InterfacePort")
        self.port = port

    def capture(self, tick_result):
        ix = self.port.neuron_indices
        if tick_result.current_spikes.shape != (self.port.population_size,):
            raise ValueError("probe population does not match tick")
        result = {name: getattr(tick_result, name)[ix].copy() for name in (
            "current_spikes", "voltage_mV", "plasticity_voltage_mV",
            "total_current_pA", "external_current_pA", "feedback_current_pA")}
        result["synaptic_current_pA"] = tick_result.propagation.synaptic_current_pA[ix].copy()
        result["previous_spikes"] = tick_result.previous_spikes[ix].copy()
        for name in ("trace_state", "plasticity_update", "weights_before", "weights_after"):
            result[name] = deepcopy(getattr(tick_result, name))
        return result
