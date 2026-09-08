"""Backward-compatible neuron-model import surface.

Canonical numerical implementation now lives in ``lisnn.neurons.kernels``.
Existing experiments may continue using::

    import NeuronModels as nm

New code should prefer ``lisnn.neurons`` or ``lisnn.neurons.kernels``.
"""

from lisnn.neurons.kernels import *  # noqa: F401,F403
