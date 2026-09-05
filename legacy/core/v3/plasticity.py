from __future__ import annotations

from dataclasses import dataclass
import argparse

import numpy as np


@dataclass(frozen=True)
class TraceStats:
    mean: float
    std: float
    minimum: float
    maximum: float
    active_fraction: float


def _as_tau_vector(
    tau: float | np.ndarray,
    num_neurons: int,
) -> np.ndarray:
    tau_array = np.asarray(tau, dtype=np.float64)
    if tau_array.ndim == 0:
        tau_array = np.full(
            num_neurons,
            float(tau_array),
            dtype=np.float64,
        )
    elif tau_array.shape != (num_neurons,):
        raise ValueError(
            "tau must be a scalar or shape (num_neurons,)"
        )
    else:
        tau_array = tau_array.copy()

    if not np.all(np.isfinite(tau_array)):
        raise ValueError("tau must contain only finite values")
    if np.any(tau_array <= 0.0):
        raise ValueError("tau must be strictly positive")
    return tau_array


class TemporalTrace:
    """Exponentially decaying trace of recent non-negative spike activity.

    Each update performs:

        trace <- trace * exp(-dt / tau) + spikes

    No normalization, clipping, or weight modification is performed here.
    The trace simply preserves a fading record of recent activity.
    """

    def __init__(
        self,
        num_neurons: int,
        *,
        dt: float = 1.0,
        tau: float | np.ndarray = 20.0,
    ) -> None:
        if num_neurons < 1:
            raise ValueError("num_neurons must be positive")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive value")

        self.num_neurons = int(num_neurons)
        self.dt = np.float64(dt)
        self.tau = _as_tau_vector(tau, self.num_neurons)
        self.decay_factors = np.exp(-self.dt / self.tau)
        self.values = np.zeros(
            self.num_neurons,
            dtype=np.float64,
        )

    @property
    def stats(self) -> TraceStats:
        return TraceStats(
            mean=float(self.values.mean()),
            std=float(self.values.std()),
            minimum=float(self.values.min()),
            maximum=float(self.values.max()),
            active_fraction=float(
                np.count_nonzero(self.values) / self.values.size
            ),
        )

    def reset(self) -> None:
        self.values.fill(0.0)

    def snapshot(self) -> np.ndarray:
        return self.values.copy()

    def decay(self, steps: int = 1) -> np.ndarray:
        """Advance the trace without adding new spikes."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps:
            self.values *= self.decay_factors ** int(steps)
        return self.values

    def update(self, spikes: np.ndarray) -> np.ndarray:
        """Decay one time step, add current spikes, and return the live trace."""
        spikes = np.asarray(spikes, dtype=np.float64)
        if spikes.shape != (self.num_neurons,):
            raise ValueError(
                "spikes must have shape (num_neurons,)"
            )
        if not np.all(np.isfinite(spikes)):
            raise ValueError(
                "spikes must contain only finite values"
            )
        if np.any(spikes < 0.0):
            raise ValueError("spikes must be non-negative")

        self.values *= self.decay_factors
        self.values += spikes
        return self.values


class TracePair:
    """Convenience container for independent pre- and postsynaptic traces."""

    def __init__(
        self,
        num_neurons: int,
        *,
        dt: float = 1.0,
        tau_pre: float | np.ndarray = 20.0,
        tau_post: float | np.ndarray = 20.0,
    ) -> None:
        self.pre = TemporalTrace(
            num_neurons,
            dt=dt,
            tau=tau_pre,
        )
        self.post = TemporalTrace(
            num_neurons,
            dt=dt,
            tau=tau_post,
        )

    @property
    def num_neurons(self) -> int:
        return self.pre.num_neurons

    def reset(self) -> None:
        self.pre.reset()
        self.post.reset()

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pre_trace = self.pre.update(pre_spikes)
        post_trace = self.post.update(post_spikes)
        return pre_trace, post_trace


def _demo() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone v3 temporal-trace diagnostic."
        )
    )
    parser.add_argument("--neurons", type=int, default=4)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--spike-neuron", type=int, default=0)
    parser.add_argument("--spike-magnitude", type=float, default=1.0)
    args = parser.parse_args()

    if args.ticks < 1:
        raise ValueError("--ticks must be positive")
    if args.spike_neuron < 0 or args.spike_neuron >= args.neurons:
        raise ValueError("--spike-neuron is out of range")
    if args.spike_magnitude < 0.0:
        raise ValueError(
            "--spike-magnitude must be non-negative"
        )

    trace = TemporalTrace(
        args.neurons,
        dt=args.dt,
        tau=args.tau,
    )

    spike = np.zeros(args.neurons, dtype=np.float64)
    spike[args.spike_neuron] = args.spike_magnitude

    print(
        "decay factor:",
        float(trace.decay_factors[args.spike_neuron]),
    )
    print("tick  spike  trace")
    for tick in range(args.ticks):
        current = (
            spike
            if tick == 0
            else np.zeros_like(spike)
        )
        trace.update(current)
        print(
            f"{tick:4d}  "
            f"{current[args.spike_neuron]:5.3f}  "
            f"{trace.values[args.spike_neuron]:.9f}"
        )

    expected = args.spike_magnitude * np.exp(
        -args.dt * (args.ticks - 1) / args.tau
    )
    observed = float(
        trace.values[args.spike_neuron]
    )
    print("expected final trace:", expected)
    print("observed final trace:", observed)
    print("absolute error:", abs(expected - observed))


if __name__ == "__main__":
    _demo()
