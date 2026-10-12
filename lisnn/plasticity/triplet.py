"""Pfister–Gerstner all-to-all triplet STDP control learner."""

from dataclasses import dataclass

import numpy as np

from lisnn.synapses.core import SynapseEdges
from lisnn.synapses.propagation import binary_spikes, validated_edges
from lisnn.validation import scalar32


@dataclass(frozen=True)
class TripletUpdate:
    pair_ltp: np.ndarray
    pair_ltd: np.ndarray
    triplet_ltp: np.ndarray
    triplet_ltd: np.ndarray
    applied_change: np.ndarray
    weights_before: np.ndarray
    weights_after: np.ndarray


class TripletSTDP:
    """Four traces and four explicit components, separate from PairSTDP.

    r1 (fast pre), r2 (slow pre), o1 (fast post), o2 (slow post) are
    decayed before the current tick's events are matched. Current spikes
    increment traces after matching, so simultaneous spikes do not pair.
    """

    def __init__(self, edges, *, tau_plus_ms, tau_minus_ms, tau_x_ms, tau_y_ms,
                 a2_plus, a2_minus, a3_plus, a3_minus, w_max):
        edges = validated_edges(edges)
        self.population_size = edges.population_size
        self._pre_idx = edges.pre_idx.copy()
        self._post_idx = edges.post_idx.copy()
        self._allow_self = edges.allow_self
        self._allow_duplicates = edges.allow_duplicates
        self._weights = edges.weight.copy()
        self.taus = tuple(scalar32(name, value, positive=True) for name, value in (
            ("tau_plus_ms", tau_plus_ms), ("tau_minus_ms", tau_minus_ms),
            ("tau_x_ms", tau_x_ms), ("tau_y_ms", tau_y_ms)))
        self.amplitudes = tuple(scalar32(name, value, nonnegative=True) for name, value in (
            ("a2_plus", a2_plus), ("a2_minus", a2_minus),
            ("a3_plus", a3_plus), ("a3_minus", a3_minus)))
        self.w_max = scalar32("w_max", w_max, positive=True)
        if np.any(self._weights > self.w_max):
            raise ValueError("initial weights must be within [0, w_max]")
        self._traces = np.zeros((4, self.population_size), dtype=np.float32)

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def traces(self):
        return self._traces.copy()

    def current_edges(self):
        return SynapseEdges(self.population_size, self._pre_idx.copy(), self._post_idx.copy(),
                            self._weights.copy(), self._allow_self, self._allow_duplicates)

    def step(self, spikes, dt_ms):
        events = binary_spikes(spikes, self.population_size)
        dt = scalar32("dt_ms", dt_ms, positive=True)
        with np.errstate(over="raise", invalid="raise"):
            decays = np.array([np.exp(-np.float64(dt) / tau) for tau in self.taus], dtype=np.float32)
            old = self._traces * decays[:, None]
            r1, o1, r2, o2 = old
            pre, post = events[self._pre_idx], events[self._post_idx]
            a2p, a2m, a3p, a3m = self.amplitudes
            pair_ltp = a2p * r1[self._pre_idx] * post
            pair_ltd = -a2m * o1[self._post_idx] * pre
            triplet_ltp = a3p * r1[self._pre_idx] * o2[self._post_idx] * post
            triplet_ltd = -a3m * o1[self._post_idx] * r2[self._pre_idx] * pre
            new = old + np.stack((events, events, events, events))
            before = self._weights.copy()
            raw = before + pair_ltp + pair_ltd + triplet_ltp + triplet_ltd
        if not np.all(np.isfinite(new)) or not np.all(np.isfinite(raw)):
            raise FloatingPointError("triplet update produced non-finite values")
        after = np.clip(raw, 0, self.w_max).astype(np.float32)
        update = TripletUpdate(pair_ltp.copy(), pair_ltd.copy(), triplet_ltp.copy(),
                               triplet_ltd.copy(), after - before, before, after.copy())
        self._traces = new
        self._weights = after
        return update
