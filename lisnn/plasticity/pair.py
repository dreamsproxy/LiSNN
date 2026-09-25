"""Independent additive pair STDP control rule on unsigned sparse edges."""

from dataclasses import dataclass

import numpy as np

from lisnn.synapses.core import SynapseEdges
from lisnn.synapses.propagation import binary_spikes, validated_edges
from lisnn.validation import scalar32


@dataclass(frozen=True)
class PairUpdate:
    """Owned per-edge components and weights from one completed update."""

    potentiation: np.ndarray
    depression: np.ndarray
    applied_change: np.ndarray
    weights_before: np.ndarray
    weights_after: np.ndarray
    pre_trace: np.ndarray
    post_trace: np.ndarray


class PairSTDP:
    """All-to-all additive spike pair rule; event clocks share a tick.

    Current events are applied against traces of previous events AFTER dt decay.
    Same-tick pre/post events do not pair with each other. Older events may
    still contribute at a tick where both cells fire. The weight is clipped
    once after adding the independent LTP and LTD components per edge.
    """

    def __init__(self, edges, *, tau_plus_ms, tau_minus_ms, a_plus, a_minus, w_max):
        validated_edges(edges)
        self.population_size = edges.population_size
        self._pre_idx = edges.pre_idx.copy()
        self._post_idx = edges.post_idx.copy()
        self._allow_self = edges.allow_self
        self._allow_duplicates = edges.allow_duplicates
        self._weights = edges.weight.copy()
        self.tau_plus_ms = scalar32("tau_plus_ms", tau_plus_ms, positive=True)
        self.tau_minus_ms = scalar32("tau_minus_ms", tau_minus_ms, positive=True)
        self.a_plus = scalar32("a_plus", a_plus, nonnegative=True)
        self.a_minus = scalar32("a_minus", a_minus, nonnegative=True)
        self.w_max = scalar32("w_max", w_max, positive=True)
        if np.any(self._weights > self.w_max):
            raise ValueError("initial edge weights must be within [0, w_max]")
        self._pre_trace = np.zeros(self.population_size, dtype=np.float32)
        self._post_trace = np.zeros(self.population_size, dtype=np.float32)

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def pre_trace(self):
        return self._pre_trace.copy()

    @property
    def post_trace(self):
        return self._post_trace.copy()

    def current_edges(self):
        """Return an independent edge snapshot for the next propagation tick."""
        return SynapseEdges(self.population_size, self._pre_idx.copy(), self._post_idx.copy(),
                            self._weights.copy(), self._allow_self, self._allow_duplicates)

    def step(self, spikes, dt_ms):
        events = binary_spikes(spikes, self.population_size)
        dt = scalar32("dt_ms", dt_ms, positive=True)
        with np.errstate(over="raise", invalid="raise"):
            pre_old = self._pre_trace * np.float32(np.exp(-np.float64(dt) / self.tau_plus_ms))
            post_old = self._post_trace * np.float32(np.exp(-np.float64(dt) / self.tau_minus_ms))
            ltp = self.a_plus * pre_old[self._pre_idx] * events[self._post_idx]
            ltd = -self.a_minus * post_old[self._post_idx] * events[self._pre_idx]
            pre_new = pre_old + events
            post_new = post_old + events
            raw_weights = self._weights + ltp + ltd
        if not all(np.all(np.isfinite(a)) for a in (ltp, ltd, pre_new, post_new, raw_weights)):
            raise FloatingPointError("pair update produced non-finite values")
        before = self._weights.copy()
        after = np.clip(raw_weights, 0, self.w_max).astype(np.float32)
        result = PairUpdate(ltp.copy(), ltd.copy(), after - before, before, after.copy(),
                            pre_new.copy(), post_new.copy())
        self._weights = after
        self._pre_trace = pre_new
        self._post_trace = post_new
        return result
