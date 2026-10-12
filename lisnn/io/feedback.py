"""Experiment-side outcome-to-current policies, separate from plasticity."""

import numpy as np

from lisnn.io.runtime import StreamRuntime
from lisnn.validation import scalar32


class FeedbackPolicy:
    """Seeded control modes; schedules only future INTERNAL perturbations."""

    def __init__(self, *, mode, port_name, success_pA, error_pA,
                 seed=0, random_probability=0.5, yoked_currents=()):
        if mode not in ("contingent", "none", "random", "yoked", "inverted"):
            raise ValueError("unsupported feedback mode")
        self.mode, self.port_name = mode, port_name
        self.success_pA = scalar32("success_pA", success_pA, nonnegative=True)
        self.error_pA = scalar32("error_pA", error_pA, nonnegative=True)
        probability = scalar32("random_probability", random_probability, nonnegative=True)
        if probability > 1:
            raise ValueError("random_probability cannot exceed 1")
        self.random_probability = probability
        self._rng = np.random.default_rng(seed)
        self._yoked = tuple(np.array(item, dtype=np.float32, copy=True) for item in yoked_currents)
        self._cursor = 0

    def schedule(self, runtime, *, observed_tick, success, delay_ticks=1):
        if not isinstance(runtime, StreamRuntime):
            raise TypeError("feedback runtime must be StreamRuntime")
        if isinstance(delay_ticks, bool) or not isinstance(delay_ticks, (int, np.integer)) or delay_ticks < 1:
            raise ValueError("feedback delay must be at least one tick")
        if self.mode == "none":
            return None
        port = runtime.transducers[self.port_name].port
        count = len(port.neuron_indices)
        if self.mode == "yoked":
            if self._cursor >= len(self._yoked):
                raise ValueError("yoked schedule exhausted")
            current = self._yoked[self._cursor].copy()
        else:
            actual_success = bool(success)
            if self.mode == "inverted":
                actual_success = not actual_success
            if self.mode == "random":
                actual_success = bool(self._rng.random() < self.random_probability)
            current = np.full(count, self.success_pA, dtype=np.float32) if actual_success else np.zeros(count, dtype=np.float32)
            if not actual_success and count:
                current[self._rng.integers(count)] = self.error_pA
        delivery_tick = int(observed_tick) + int(delay_ticks)
        runtime.schedule_feedback(observed_tick, current, port_name=self.port_name,
                                  delivery_tick=delivery_tick, mode=self.mode)
        if self.mode == "yoked":
            self._cursor += 1
        return current.copy()
