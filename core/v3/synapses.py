from __future__ import annotations

from dataclasses import dataclass
import argparse

import numpy as np


EXCITATORY = np.int8(1)
INHIBITORY = np.int8(-1)


@dataclass(frozen=True)
class SynapseStats:
    mean_magnitude: float
    std_magnitude: float
    minimum_magnitude: float
    maximum_magnitude: float
    connection_density: float
    excitatory_neurons: int
    inhibitory_neurons: int


def sample_neuron_types(
    num_neurons: int,
    inhibitory_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample immutable Dale-style presynaptic neuron polarities.

    Returns an int8 vector containing +1 for excitatory neurons and -1 for
    inhibitory neurons. The vector can be supplied directly to SynapseMatrix
    or reused by another module so neuron identity remains consistent.
    """
    if num_neurons < 1:
        raise ValueError("num_neurons must be positive")
    if not 0.0 <= inhibitory_fraction <= 1.0:
        raise ValueError("inhibitory_fraction must be in [0, 1]")

    neuron_types = np.full(num_neurons, EXCITATORY, dtype=np.int8)
    inhibitory_count = int(
        np.floor(num_neurons * inhibitory_fraction + 0.5)
    )
    inhibitory_count = min(max(inhibitory_count, 0), num_neurons)

    if inhibitory_count:
        inhibitory_indices = rng.choice(
            num_neurons,
            size=inhibitory_count,
            replace=False,
        )
        neuron_types[inhibitory_indices] = INHIBITORY

    return neuron_types


class SynapseMatrix:
    """Independent dense/sparse recurrent synaptic substrate.

    Rows are postsynaptic neurons and columns are presynaptic neurons.
    Synaptic magnitudes are always non-negative. Dale polarity belongs to the
    presynaptic neuron, so the effective signed weight is:

        W_ij = magnitude_ij * neuron_type_j

    Propagation is a raw weighted sum. There is intentionally no division by
    neuron count, fan-in, row norm, or total connection strength.
    """

    def __init__(
        self,
        num_neurons: int,
        *,
        neuron_types: np.ndarray | None = None,
        inhibitory_fraction: float = 0.2,
        connection_probability: float = 1.0,
        allow_self_connections: bool = False,
        seed: int = 0,
    ) -> None:
        if num_neurons < 1:
            raise ValueError("num_neurons must be positive")
        if not 0.0 <= connection_probability <= 1.0:
            raise ValueError("connection_probability must be in [0, 1]")

        self.num_neurons = int(num_neurons)
        self.connection_probability = float(connection_probability)
        self.allow_self_connections = bool(allow_self_connections)
        self.seed = int(seed)

        type_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 101])
        )
        weight_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 201])
        )
        mask_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 202])
        )

        if neuron_types is None:
            self.neuron_types = sample_neuron_types(
                self.num_neurons,
                inhibitory_fraction,
                type_rng,
            )
        else:
            supplied = np.asarray(neuron_types, dtype=np.int8)
            if supplied.shape != (self.num_neurons,):
                raise ValueError(
                    "neuron_types must have shape (num_neurons,)"
                )
            if not np.all(
                (supplied == EXCITATORY) | (supplied == INHIBITORY)
            ):
                raise ValueError("neuron_types must contain only +1 or -1")
            self.neuron_types = supplied.copy()

        available_fan = (
            self.num_neurons
            if self.allow_self_connections
            else max(self.num_neurons - 1, 0)
        )
        expected_fan = max(
            available_fan * self.connection_probability,
            1.0,
        )
        limit = float(np.sqrt(6.0 / (2.0 * expected_fan)))

        if self.connection_probability < 1.0:
            self.connection_mask = (
                mask_rng.random(
                    (self.num_neurons, self.num_neurons)
                ) < self.connection_probability
            )
        else:
            self.connection_mask = np.ones(
                (self.num_neurons, self.num_neurons),
                dtype=bool,
            )

        if not self.allow_self_connections:
            np.fill_diagonal(self.connection_mask, False)

        self.magnitudes = weight_rng.uniform(
            0.0,
            limit,
            (self.num_neurons, self.num_neurons),
        ).astype(np.float64)
        self.magnitudes *= self.connection_mask

    @property
    def effective_weights(self) -> np.ndarray:
        """Return the signed weight matrix for inspection or diagnostics."""
        return self.magnitudes * self.neuron_types[None, :]

    @property
    def excitatory_count(self) -> int:
        return int(np.sum(self.neuron_types == EXCITATORY))

    @property
    def inhibitory_count(self) -> int:
        return int(np.sum(self.neuron_types == INHIBITORY))

    @property
    def connection_density(self) -> float:
        eligible = self.num_neurons * self.num_neurons
        if not self.allow_self_connections:
            eligible -= self.num_neurons
        if eligible <= 0:
            return 0.0
        return float(np.count_nonzero(self.connection_mask) / eligible)

    @property
    def stats(self) -> SynapseStats:
        nonzero = self.magnitudes[self.magnitudes > 0.0]
        if nonzero.size:
            mean = float(nonzero.mean())
            std = float(nonzero.std())
            minimum = float(nonzero.min())
            maximum = float(nonzero.max())
        else:
            mean = std = minimum = maximum = 0.0

        return SynapseStats(
            mean_magnitude=mean,
            std_magnitude=std,
            minimum_magnitude=minimum,
            maximum_magnitude=maximum,
            connection_density=self.connection_density,
            excitatory_neurons=self.excitatory_count,
            inhibitory_neurons=self.inhibitory_count,
        )

    def propagate(self, spikes: np.ndarray) -> np.ndarray:
        """Propagate spikes as a raw signed weighted sum."""
        spikes = np.asarray(spikes, dtype=np.float64)
        if spikes.shape != (self.num_neurons,):
            raise ValueError("spikes must have shape (num_neurons,)")
        if not np.all(np.isfinite(spikes)):
            raise ValueError("spikes must contain only finite values")
        if np.any(spikes < 0.0):
            raise ValueError("spikes must be non-negative")

        signed_spikes = spikes * self.neuron_types
        return self.magnitudes @ signed_spikes

    def propagate_components(
        self,
        spikes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return excitatory, inhibitory, and net postsynaptic current.

        Excitatory current is non-negative for non-negative spikes.
        Inhibitory current is non-positive for non-negative spikes.
        The returned net current equals excitatory + inhibitory.
        """
        spikes = np.asarray(spikes, dtype=np.float64)
        if spikes.shape != (self.num_neurons,):
            raise ValueError("spikes must have shape (num_neurons,)")
        if not np.all(np.isfinite(spikes)):
            raise ValueError("spikes must contain only finite values")
        if np.any(spikes < 0.0):
            raise ValueError("spikes must be non-negative")

        excitatory = self.neuron_types == EXCITATORY
        inhibitory = ~excitatory

        excitatory_current = np.zeros(self.num_neurons, dtype=np.float64)
        inhibitory_current = np.zeros(self.num_neurons, dtype=np.float64)

        if np.any(excitatory):
            excitatory_current = (
                self.magnitudes[:, excitatory] @ spikes[excitatory]
            )
        if np.any(inhibitory):
            inhibitory_current = -(
                self.magnitudes[:, inhibitory] @ spikes[inhibitory]
            )

        net_current = excitatory_current + inhibitory_current
        return excitatory_current, inhibitory_current, net_current


def _demo() -> None:
    parser = argparse.ArgumentParser(
        description="Run a standalone v3 synaptic propagation diagnostic."
    )
    parser.add_argument("--neurons", type=int, default=16)
    parser.add_argument("--inhibitory-fraction", type=float, default=0.2)
    parser.add_argument("--connection-probability", type=float, default=1.0)
    parser.add_argument("--spike-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-self-connections", action="store_true")
    args = parser.parse_args()

    synapses = SynapseMatrix(
        args.neurons,
        inhibitory_fraction=args.inhibitory_fraction,
        connection_probability=args.connection_probability,
        allow_self_connections=args.allow_self_connections,
        seed=args.seed,
    )

    spike_rng = np.random.default_rng(
        np.random.SeedSequence([args.seed, 301])
    )
    spikes = (
        spike_rng.random(args.neurons) < args.spike_probability
    ).astype(np.float64)

    excitatory, inhibitory, net = synapses.propagate_components(spikes)
    direct = synapses.propagate(spikes)

    print("Synapse stats:", synapses.stats)
    print("Neuron types:", synapses.neuron_types)
    print("Spikes:", spikes)
    print("Excitatory current:", excitatory)
    print("Inhibitory current:", inhibitory)
    print("Net current:", net)
    print(
        "Component/direct max error:",
        float(np.max(np.abs(net - direct))),
    )


if __name__ == "__main__":
    _demo()
