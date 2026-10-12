# LiSNN debugging

All active smoke implementations live in this package. Root `debug.py` is the
executable entrypoint; `python -m lisnn.debugging` runs the same CLI.

```bash
python debug.py
python debug.py --list
python debug.py --only population units --verbose
python debug.py --log-dir logs/manual-check
```

Default runs use small deterministic populations and show suite progress.
`--verbose` also prints individual checks and neuron traces. No pytest
dependency is needed: smoke implementations use NumPy and the standard library.

## Suites

| CLI name | Independent callable | Coverage |
| --- | --- | --- |
| `population` | `population_smoke_test` | All nine models: integration, forced spike/reset and auxiliary states |
| `synapses` | `synapse_smoke_test` | Sparse edge layout, degree, weights, seeded topology, autapse/duplicate rejection |
| `indices` | `index_smoke_test` | Signed/unsigned overflow, population bounds, both constructors, order and empty edges |
| `units` | `units_smoke_test` | Hand-calculated voltages, passive LIF/AdEx agreement and timestep convergence |
| `inputs` | `input_smoke_test` | Invalid timestep/current rejection before mutation in all nine kernels; valid broadcasting |
| `network` | `network_smoke_test` | Model construction, initialization, reproducibility and mixed contiguous slices |
| `spatial` | `spatial_smoke_test` | Position layout/bounds, explicit positions, independent RNG and empty morphology shell |
| `propagation` | `propagation_smoke_test` | Causal two-hop delivery, duplicate impulse accumulation, mixed-family pA observations |
| `plasticity_observation` | `plasticity_observation_smoke_test` | Pre-reset voltage across nine families, mixed CVA observation and bounded event traces |
| `pair_stdp` | `pair_smoke_test` | Hand-checkable three-cell sequence, pre-before-post timing and independent fixed source edges |
| `triplet_stdp` | `triplet_smoke_test` | Hand-checkable pre-post-post sequence and isolated higher-order potentiation |

```python
from lisnn.debugging import units_smoke_test, run_all_smoke_tests

result = units_smoke_test(verbose=True)
summary = run_all_smoke_tests(only=["units", "indices"], progress=True)
```

Each independent callable returns a dictionary with boolean `passed`. Most
include named `checks`; population diagnostics retain their existing per-model
structure and `ouput_path` argument spelling. The original detailed diagnostics
are preserved inside the canonical package, without importing root wrappers.

## Reports and failure handling

The runner writes `<suite>.json` for each selected suite, `summary.json` with
all results, and `population.log` when population diagnostics run. Default
paths are timestamped under `logs/`; an explicit existing directory replaces
selected reports, so use a fresh directory to retain previous results.

The CLI returns 0 only when all selected suites pass. A false status, missing
status, or suite exception is a failure. Exceptions include tracebacks in the
report; remaining suites continue. Interrupts are allowed to stop the run.
Report I/O errors also produce a failing CLI status.

GitHub CI runs the regression suite and `debug.py`, then uploads available
smoke reports as `smoke-diagnostics`. Smoke tests are diagnostic samples, not
exhaustive correctness proofs or biological validation.

## Adding a suite

Implement a callable in a focused module here, with its own imports and small,
reproducible inputs. Return `{"passed": bool, ...}` with JSON-compatible
diagnostics (NumPy arrays/scalars are supported by the runner). Export its
public callable in `__init__.py`, then register it in `runner.SMOKE_TESTS`.
Adapters receive `(log_dir, verbose)`; `_suite` adapts a callable needing only
`verbose`. Do not import pytest or a root compatibility wrapper from this
package. Add meaningful regression coverage for new failure modes.

The propagation suite can also run alone with `python -m lisnn.debugging.propagation_smoke`.
`python -m examples.fixed_weight_propagation` prints its full three-tick numerical trace.
