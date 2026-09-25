"""Clopath-style voltage-gated STDP with explicit filter and event timing."""

from dataclasses import dataclass

import numpy as np

from lisnn.synapses.core import SynapseEdges
from lisnn.synapses.propagation import binary_spikes, validated_edges
from lisnn.validation import scalar32, vector32


@dataclass(frozen=True)
class VoltageUpdate:
    ltd_gate_mV: np.ndarray
    ltp_current_gate_mV: np.ndarray
    ltp_history_gate_mV: np.ndarray
    pre_trace: np.ndarray
    filtered_minus_mV: np.ndarray
    filtered_plus_mV: np.ndarray
    depression: np.ndarray
    potentiation: np.ndarray
    applied_change: np.ndarray
    weights_before: np.ndarray
    weights_after: np.ndarray


class VoltageSTDP:
    """Voltage primary learner, without Pair/Triplet or homeostatic stacking.

    Depolarization filters and presynaptic spike history from previous ticks
    are decayed first. Match LTD to current pre events; LTP uses only older pre
    history and current integrated pre-reset voltage. Then incorporate the
    present sample/spike into filters and clip the combined weight update.
    """

    def __init__(self, edges, *, tau_x_ms, tau_minus_ms, tau_plus_ms,
                 theta_minus_mV, theta_plus_mV, a_ltd, a_ltp, w_max,
                 initial_voltage_mV):
        edges = validated_edges(edges)
        self.population_size = edges.population_size
        self._pre_idx = edges.pre_idx.copy()
        self._post_idx = edges.post_idx.copy()
        self._allow_self = edges.allow_self
        self._allow_duplicates = edges.allow_duplicates
        self._weights = edges.weight.copy()
        self.tau_x_ms = scalar32("tau_x_ms", tau_x_ms, positive=True)
        self.tau_minus_ms = scalar32("tau_minus_ms", tau_minus_ms, positive=True)
        self.tau_plus_ms = scalar32("tau_plus_ms", tau_plus_ms, positive=True)
        self.theta_minus_mV = scalar32("theta_minus_mV", theta_minus_mV)
        self.theta_plus_mV = scalar32("theta_plus_mV", theta_plus_mV)
        if self.theta_plus_mV <= self.theta_minus_mV:
            raise ValueError("theta_plus_mV must exceed theta_minus_mV")
        self.a_ltd = scalar32("a_ltd", a_ltd, nonnegative=True)
        self.a_ltp = scalar32("a_ltp", a_ltp, nonnegative=True)
        self.w_max = scalar32("w_max", w_max, positive=True)
        if np.any(self._weights > self.w_max):
            raise ValueError("initial weights must be within [0, w_max]")
        initial = vector32("initial_voltage_mV", initial_voltage_mV, self.population_size)
        self._pre_trace = np.zeros(self.population_size, dtype=np.float32)
        self._u_minus = initial.copy()
        self._u_plus = initial.copy()

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def traces(self):
        return (self._pre_trace.copy(), self._u_minus.copy(), self._u_plus.copy())

    def current_edges(self):
        return SynapseEdges(self.population_size, self._pre_idx.copy(), self._post_idx.copy(),
                            self._weights.copy(), self._allow_self, self._allow_duplicates)

    def step(self, spikes, plasticity_voltage_mV, dt_ms):
        events = binary_spikes(spikes, self.population_size)
        voltage = vector32("plasticity_voltage_mV", plasticity_voltage_mV, self.population_size)
        dt = scalar32("dt_ms", dt_ms, positive=True)
        with np.errstate(over="raise", invalid="raise"):
            dx = np.float32(np.exp(-np.float64(dt) / self.tau_x_ms))
            dm = np.float32(np.exp(-np.float64(dt) / self.tau_minus_ms))
            dp = np.float32(np.exp(-np.float64(dt) / self.tau_plus_ms))
            x_old = self._pre_trace * dx
            # Voltage filters retain earlier samples for both gates. Incorporate
            # the current sample only after this interval's weight update.
            minus_gate = np.maximum(self._u_minus - self.theta_minus_mV, 0)
            current_gate = np.maximum(voltage - self.theta_plus_mV, 0)
            history_gate = np.maximum(self._u_plus - self.theta_minus_mV, 0)
            depression = -self.a_ltd * events[self._pre_idx] * minus_gate[self._post_idx]
            potentiation = self.a_ltp * dt * x_old[self._pre_idx] * (
                current_gate[self._post_idx] * history_gate[self._post_idx])
            before = self._weights.copy()
            raw = before + depression + potentiation
            next_x = x_old + events
            next_minus = dm * self._u_minus + (np.float32(1) - dm) * voltage
            next_plus = dp * self._u_plus + (np.float32(1) - dp) * voltage
        if not all(np.all(np.isfinite(a)) for a in (raw, next_x, next_minus, next_plus)):
            raise FloatingPointError("voltage rule produced non-finite values")
        after = np.clip(raw, 0, self.w_max).astype(np.float32)
        result = VoltageUpdate(minus_gate.copy(), current_gate.copy(), history_gate.copy(),
                               next_x.copy(), next_minus.copy(), next_plus.copy(),
                               depression.copy(), potentiation.copy(), after - before,
                               before, after.copy())
        self._pre_trace = next_x
        self._u_minus = next_minus
        self._u_plus = next_plus
        self._weights = after
        return result
