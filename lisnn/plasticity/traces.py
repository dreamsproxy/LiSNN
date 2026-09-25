"""Persistent float32 traces over observed spikes and pre-reset voltages."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from lisnn.synapses.propagation import binary_spikes
from lisnn.validation import scalar32, vector32


@dataclass(frozen=True)
class TraceSnapshot:
    time_ms: float
    spikes: np.ndarray
    plasticity_voltage_mV: np.ndarray
    spike_trace: np.ndarray
    voltage_trace_mV: np.ndarray


class PlasticityTraces:
    """Update after integration; each snapshot owns independent arrays.

    At each tick, x <- exp(-dt/tau_spike)*x + spike and
    u <- exp(-dt/tau_voltage)*u + (1-exp(-dt/tau_voltage))*voltage.
    Calling rules may use the old state before `advance` or the returned new
    state, but must choose explicitly. Co-occurring events are simultaneous.
    History is opt-in, bounded by both elapsed time and sample count.
    """

    def __init__(self, population_size, *, tau_spike_ms, tau_voltage_ms,
                 history_ms=None, history_max_samples=None):
        if isinstance(population_size, (bool, np.bool_)) or not isinstance(population_size, (int, np.integer)) or population_size <= 0:
            raise ValueError("population_size must be a positive integer")
        self.population_size = int(population_size)
        self.tau_spike_ms = scalar32("tau_spike_ms", tau_spike_ms, positive=True)
        self.tau_voltage_ms = scalar32("tau_voltage_ms", tau_voltage_ms, positive=True)
        if (history_ms is None) != (history_max_samples is None):
            raise ValueError("history_ms and history_max_samples must be specified together")
        self.history_ms = None if history_ms is None else scalar32("history_ms", history_ms, positive=True)
        if history_max_samples is not None and (isinstance(history_max_samples, (bool, np.bool_)) or not isinstance(history_max_samples, (int, np.integer)) or history_max_samples <= 0):
            raise ValueError("history_max_samples must be a positive integer")
        self._history = None if history_ms is None else deque(maxlen=int(history_max_samples))
        self._spike_trace = np.zeros(self.population_size, dtype=np.float32)
        self._voltage_trace = np.zeros(self.population_size, dtype=np.float32)
        self.time_ms = 0.0

    @property
    def spike_trace(self):
        return self._spike_trace.copy()

    @property
    def voltage_trace_mV(self):
        return self._voltage_trace.copy()

    @property
    def history(self):
        return () if self._history is None else tuple(
            TraceSnapshot(s.time_ms, s.spikes.copy(), s.plasticity_voltage_mV.copy(),
                          s.spike_trace.copy(), s.voltage_trace_mV.copy()) for s in self._history
        )

    def advance(self, spikes, plasticity_voltage_mV, dt_ms):
        dt = scalar32("dt_ms", dt_ms, positive=True)
        events = binary_spikes(spikes, self.population_size)
        voltage = vector32("plasticity_voltage_mV", plasticity_voltage_mV, self.population_size)
        spike_decay = np.float32(np.exp(-np.float64(dt) / self.tau_spike_ms))
        voltage_decay = np.float32(np.exp(-np.float64(dt) / self.tau_voltage_ms))
        with np.errstate(over="raise", invalid="raise"):
            new_spike = np.float32(spike_decay * self._spike_trace + events)
            new_voltage = np.float32(voltage_decay * self._voltage_trace +
                                     (np.float32(1) - voltage_decay) * voltage)
        if not np.all(np.isfinite(new_spike)) or not np.all(np.isfinite(new_voltage)):
            raise FloatingPointError("trace update produced non-finite values")
        new_time = self.time_ms + float(dt)
        if not np.isfinite(new_time):
            raise FloatingPointError("trace clock exceeded finite range")
        snapshot = TraceSnapshot(new_time, events.copy(), voltage.copy(), new_spike.copy(), new_voltage.copy())
        self._spike_trace = new_spike
        self._voltage_trace = new_voltage
        self.time_ms = new_time
        if self._history is not None:
            self._history.append(TraceSnapshot(
                new_time, events.copy(), voltage.copy(), new_spike.copy(), new_voltage.copy(),
            ))
            while self._history and self._history[0].time_ms < self.time_ms - float(self.history_ms):
                self._history.popleft()
        return snapshot
