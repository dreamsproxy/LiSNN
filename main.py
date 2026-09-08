"""Minimal LiSNN construction examples.

The executable entrypoint intentionally contains no stepping, weights, or
plasticity until those network layers are implemented.
"""

import numpy as np

import Network


def main() -> None:
    homogeneous = Network.create_nn(
        population=8,
        neuron_type="GLIF5",
        fill=np.float32(0.0),
        randomize_params=False,
    )

    print(homogeneous)
    print(homogeneous.pool.shape)
    print(homogeneous.type_counts)
    print(homogeneous.type_slices)
    print(homogeneous.homogeneous)

    mixed = Network.create_nn(
        population=8,
        neuron_type={
            "default": "LIF",
            "GLIF5": 2,
            "AdEx": 3,
        },
        randomize_params=True,
        seed=1,
    )

    print(mixed)
    print(mixed.pool.shape)
    print(mixed.type_counts)
    print(mixed.type_slices)
    print(mixed.homogeneous)


if __name__ == "__main__":
    main()
