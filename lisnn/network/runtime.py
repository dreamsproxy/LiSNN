"""Causal fixed-weight composition over contiguous neuron-family slices."""

from dataclasses import dataclass

import numpy as np

from lisnn.network.core import SNN
from lisnn.neurons import kernels as k
from lisnn.neurons.adapters import current_to_izhikevich, izhikevich_to_current, observe_izhikevich
from lisnn.neurons.registry import NeuronType, get_step_function
from lisnn.synapses.core import SynapseEdges
from lisnn.synapses.propagation import PropagationResult, binary_spikes, propagate, validated_edges
from lisnn.validation import scalar32, vector32


@dataclass(frozen=True)
class TickResult:
    tick: int
    time_ms: float
    dt_ms: float
    previous_spikes: np.ndarray
    current_spikes: np.ndarray
    propagation: PropagationResult
    external_current_pA: np.ndarray
    feedback_current_pA: np.ndarray
    total_current_pA: np.ndarray
    voltage_before_mV: np.ndarray
    voltage_mV: np.ndarray
    plasticity_voltage_mV: np.ndarray
    izhikevich_indices: np.ndarray
    izhikevich_before: dict
    izhikevich_after: dict


class FixedWeightRuntime:
    """Own a copied population, fixed edge snapshot, clock and spike buffers.

    The original SNN and edge arrays are not mutated. `pool` is native model
    state, not a shared-unit observation table. TickResult is the observation
    boundary; its arrays are independent snapshots. No weight updates occur.
    """

    def __init__(self, network, edges, *, dt, impulse_scale, previous_spikes=None):
        if not isinstance(network, SNN):
            raise TypeError('network must be an SNN')
        edges = validated_edges(edges)
        if edges.population_size != network.population_size:
            raise ValueError('edge and neuron population sizes must match')
        self.population_size = network.population_size
        self.pool = network.pool.copy()
        self._slices = tuple(network.type_slices.items())
        covered = np.zeros(self.population_size, dtype=np.int32)
        for model, section in self._slices:
            get_step_function(model)
            if not isinstance(section, slice) or section.step not in (None, 1):
                raise ValueError('neuron families require contiguous slices')
            if section.start is None or section.stop is None or not 0 <= section.start <= section.stop <= self.population_size:
                raise ValueError('neuron slice is outside population')
            covered[section] += 1
        if not np.all(covered == 1):
            raise ValueError('neuron slices must cover the population exactly once')
        self._edges = SynapseEdges(edges.population_size, edges.pre_idx.copy(), edges.post_idx.copy(),
                                   edges.weight.copy(), edges.allow_self, edges.allow_duplicates)
        for array in (self._edges.pre_idx, self._edges.post_idx, self._edges.weight):
            array.flags.writeable = False
        self.dt = scalar32('dt', dt, positive=True)
        self.impulse_scale = scalar32('impulse_scale', impulse_scale, nonnegative=True)
        self._previous = np.zeros(self.population_size, dtype=np.float32) if previous_spikes is None else binary_spikes(previous_spikes, self.population_size)
        self.tick = 0
        self.time_ms = 0.0
        self._feedback = {}
        self._validate_pool()

    @property
    def previous_spikes(self):
        return self._previous.copy()

    def _validate_pool(self):
        k._check_population(self.pool)
        if len(self.pool) != self.population_size or not np.all(np.isfinite(self.pool)):
            raise ValueError('population must have the configured size and finite state/parameters')
        for model, section in self._slices:
            if model == NeuronType.IZHIKEVICH:
                observe_izhikevich(self.pool[section])

    def schedule_feedback(self, tick, current_pA):
        """Queue a copied pA vector for an unexecuted tick; collisions add.

        After observing tick t, the earliest schedulable tick is t+1. This
        queue uses logical ticks, not physical axonal arrival times.
        """
        if isinstance(tick, (bool, np.bool_)) or not isinstance(tick, (int, np.integer)):
            raise TypeError('feedback tick must be an integer')
        if tick < self.tick:
            raise ValueError('cannot schedule feedback into an executed tick')
        value = vector32('feedback_current', current_pA, self.population_size)
        with np.errstate(over='raise', invalid='raise'):
            combined = self._feedback.get(int(tick), np.zeros_like(value)) + value
        self._feedback[int(tick)] = combined

    def step(self, external_current=0, feedback_current=0):
        """Advance one tick; invalid inputs/numerical failures leave runtime intact.

        Feedback passed here must already be available before this call. There
        are no within-tick callbacks. New spikes only propagate on the next call.
        """
        self._validate_pool()
        dt = scalar32('dt', self.dt, positive=True)
        external = vector32('external_current', external_current, self.population_size)
        feedback = vector32('feedback_current', feedback_current, self.population_size)
        previous = self._previous.copy()
        propagation = propagate(previous, self._edges, dt, self.impulse_scale, return_details=True)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            feedback = feedback + self._feedback.get(self.tick, np.zeros_like(feedback))
            total = external + propagation.synaptic_current_pA + feedback
            if not np.all(np.isfinite(total)):
                raise FloatingPointError('total current is not finite')
            # Integrate a working copy so later-slice failures cannot half-step state.
            working = self.pool.copy()
            current = np.zeros(self.population_size, dtype=np.float32)
            plasticity_voltage = np.zeros(self.population_size, dtype=np.float32)
            izh_before, izh_after = {}, {}
            izh_indices = np.empty(0, dtype=np.int32)
            for model, section in self._slices:
                kernel_input = total[section].copy()
                if model == NeuronType.IZHIKEVICH:
                    izh_indices = np.arange(section.start, section.stop, dtype=np.int32)
                    izh_before = observe_izhikevich(working[section])
                    kernel_input = current_to_izhikevich(kernel_input, working[section, k.C_M])
                    # Expose the round-trip input in pA, not an unlabelled native value.
                    izh_before['input_current_pA'] = izhikevich_to_current(kernel_input, working[section, k.C_M])
                observation = get_step_function(model)(
                    working[section], kernel_input, dt, return_observation=True,
                )
                current[section] = binary_spikes(observation.spikes, section.stop - section.start)
                plasticity_voltage[section] = observation.plasticity_voltage_mV
                if model == NeuronType.IZHIKEVICH:
                    izh_after = observe_izhikevich(working[section])
            if not np.all(np.isfinite(working)):
                raise FloatingPointError('neuron integration produced non-finite state')
        result = TickResult(
            self.tick, self.time_ms, float(dt), previous, current.copy(), propagation,
            external, feedback, total, self.pool[:, k.V].copy(), working[:, k.V].copy(),
            plasticity_voltage.copy(),
            izh_indices, izh_before, izh_after,
        )
        self.pool[...] = working
        self._previous = current.copy()
        self._feedback.pop(self.tick, None)
        self.tick += 1
        self.time_ms += float(dt)
        return result
