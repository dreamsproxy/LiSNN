"""ACTIVE/SLEEP experimental routing and learning policy."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from lisnn.io.runtime import StreamRuntime
from lisnn.validation import scalar32


@dataclass(frozen=True)
class RegimePolicy:
    name: str
    external_gain: float
    internal_gain: float
    plasticity_enabled: bool
    actions_enabled: bool
    permitted_sources: tuple[str, ...] = ("EXTERNAL", "INTERNAL")

    def __post_init__(self):
        if self.name not in ("ACTIVE", "SLEEP"):
            raise ValueError("regime must be ACTIVE or SLEEP")
        object.__setattr__(self, "external_gain", scalar32("external_gain", self.external_gain, nonnegative=True))
        object.__setattr__(self, "internal_gain", scalar32("internal_gain", self.internal_gain, nonnegative=True))
        if not isinstance(self.plasticity_enabled, bool) or not isinstance(self.actions_enabled, bool):
            raise TypeError("learning/action policies must be boolean")
        if any(source not in ("EXTERNAL", "INTERNAL") for source in self.permitted_sources):
            raise ValueError("invalid permitted source")


@dataclass(frozen=True)
class RegimeTick:
    regime: RegimePolicy
    routed: object


class RegimeController:
    """Apply declared tick transitions, preserving all network state."""

    def __init__(self, runtime, *, active=None, sleep=None, seed=0,
                 background_port=None, background_probability=0,
                 background_amplitude_pA=0, max_background_events=0):
        if not isinstance(runtime, StreamRuntime):
            raise TypeError("regime requires StreamRuntime")
        self.runtime = runtime
        self.active = active or RegimePolicy("ACTIVE", 1, 1, True, True)
        self.sleep = sleep or RegimePolicy("SLEEP", 0.05, 1, True, False)
        if self.active.name != "ACTIVE" or self.sleep.name != "SLEEP":
            raise ValueError("supply matching ACTIVE and SLEEP policies")
        self.current = self.active
        self.transitions = {}
        probability = scalar32("background_probability", background_probability, nonnegative=True)
        if probability > 1:
            raise ValueError("background probability cannot exceed one")
        if isinstance(max_background_events, bool) or not isinstance(max_background_events, (int, np.integer)) or max_background_events < 0:
            raise ValueError("max_background_events must be nonnegative integer")
        self.background_probability = probability
        self.background_amplitude_pA = scalar32("background_amplitude_pA", background_amplitude_pA, nonnegative=True)
        self.max_background_events = int(max_background_events)
        self.background_port = background_port
        if probability > 0 and (background_port not in runtime.transducers or runtime.transducers[background_port].port.source != "INTERNAL"):
            raise ValueError("sparse background needs a declared INTERNAL port")
        self._rng = np.random.default_rng(seed)
        self.background_count = 0

    def transition(self, tick, regime):
        if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick < self.runtime.tick:
            raise ValueError("transition tick must be an unexecuted integer")
        if regime not in ("ACTIVE", "SLEEP") or tick in self.transitions:
            raise ValueError("transition must be unique and name a known regime")
        self.transitions[int(tick)] = regime

    def pulse(self, *, port_name, current_pA, delay_ticks=1):
        """A bounded caller-defined internal perturbation after this tick."""
        if self.current.name != "SLEEP":
            raise ValueError("sleep pulses require SLEEP regime")
        if isinstance(delay_ticks, bool) or not isinstance(delay_ticks, (int, np.integer)) or delay_ticks < 1:
            raise ValueError("pulse delay must be at least one tick")
        self.runtime.schedule_feedback(self.runtime.tick - 1, current_pA,
                                       port_name=port_name,
                                       delivery_tick=self.runtime.tick - 1 + delay_ticks,
                                       mode="sleep_pulse")

    def step(self, external_current=0, *, plasticity_override=None):
        before = self.snapshot()
        tick = self.runtime.tick
        choice = self.transitions.get(tick)
        policy = self.current if choice is None else (self.active if choice == "ACTIVE" else self.sleep)
        if plasticity_override is not None and not isinstance(plasticity_override, bool):
            raise TypeError("plasticity_override must be boolean")
        for item in self.runtime.scheduler._frames.get(tick, ()):
            frame = item[0] if isinstance(item, tuple) else item
            if frame.source not in policy.permitted_sources:
                raise ValueError(f"{frame.source} stream forbidden under {policy.name}")
        try:
            routed = self.runtime.step(external_current, external_gain=policy.external_gain,
                                       internal_gain=policy.internal_gain,
                                       plasticity_enabled=(policy.plasticity_enabled if plasticity_override is None else plasticity_override))
            self.current = policy
            self.transitions.pop(tick, None)
            if (policy.name == "SLEEP" and self.background_count < self.max_background_events and
                    self._rng.random() < self.background_probability):
                port = self.runtime.transducers[self.background_port].port
                values = np.zeros(len(port.neuron_indices), dtype=np.float32)
                if len(values):
                    values[self._rng.integers(len(values))] = self.background_amplitude_pA
                    self.runtime.schedule_feedback(tick, values, port_name=self.background_port,
                                                   delivery_tick=tick + 1, mode="sleep_background")
                    self.background_count += 1
            return RegimeTick(policy, routed)
        except Exception:
            self.restore(before)
            raise

    def snapshot(self):
        return deepcopy(self)

    def restore(self, snapshot):
        if not isinstance(snapshot, RegimeController) or snapshot.runtime.network.population_size != self.runtime.network.population_size:
            raise ValueError("regime snapshot population mismatch")
        copy = deepcopy(snapshot)
        self.runtime.restore(copy.runtime)
        self.active, self.sleep, self.current = copy.active, copy.sleep, copy.current
        self.transitions = copy.transitions
        self.background_probability = copy.background_probability
        self.background_amplitude_pA = copy.background_amplitude_pA
        self.max_background_events = copy.max_background_events
        self.background_port = copy.background_port
        self.background_count = copy.background_count
        self._rng = copy._rng
