# LiSNN debugging package

Canonical smoke/debug utilities live here. Root-level `debug.py` and `debugging.py` are compatibility facades only.

## Current smoke tests

- `population_smoke_test(...)`: neuron integration/spike/reset smoke test.
- `synapse_smoke_test(...)`: M2.1 sparse synapse substrate validation.

Future M2 propagation/plasticity debuggers should be added as separate modules under this package rather than expanding one monolithic root debug file.
