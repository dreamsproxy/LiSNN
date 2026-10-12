# Playground profiling (#25)

From the repository root, using Python with NumPy installed:

```bash
python -m benchmarks.profile_playground --output benchmarks/results/local.json
```

The default run covers 32 neurons / 128 edges, 512 / 8,192, and 4,096 /
131,072. It measures 30 operations after 3 warm-up operations, with 3 fresh,
seeded timing repeats. The console shows progress and each median. For a quick
check or a longer end-to-end sample:

```bash
python -m benchmarks.profile_playground --sizes small --steps 5 --repeats 2 --output /tmp/small.json
python -m benchmarks.profile_playground --sizes stress --operations fixed_tick adaptive_tick regime_tick --steps 100 --repeats 5 --timing-only --output /tmp/runtime.json
```

No optional media dependency, JIT, or GPU is used. Initialization is recorded
separately from steady-state timing. Timing runs do not enable cProfile or
tracemalloc: those measurements run in separate fresh workloads. Each repeat
must produce an identical final observation digest. The JSON records Python,
NumPy, OS, CPU count, thread environment, source digest, seeds, topology size,
all timing samples, allocation peaks, and the largest persistent arrays.
Results depend on the machine and load; no universal timing threshold is
asserted. Exact seeded trajectories are validated separately from timing.

## Workloads and interpretation

| Operation | Included work |
|---|---|
| `neurons` | LIF kernel integration and observation |
| `propagation` | Public topology/input validation, gather, float32 scatter and diagnostics |
| `pair`, `triplet`, `voltage` | One standalone learner update and observations |
| `trace_history` | Trace advance with a 32-sample bounded history |
| `transducer` | Declared 32-channel (or smaller) port to population pA vector |
| `probe` | Copy selected neural observations and complete learning diagnostics |
| `fixed_tick` | Full mixed LIF/AdEx/Izhikevich tick, plasticity off |
| `adaptive_tick` | Full mixed tick with triplet STDP and transactional commit |
| `routed_tick` | Frame construction, scheduling, transduction, adaptive tick and delivery log |
| `regime_tick` | Routed tick plus alternating ACTIVE/SLEEP transitions and rollback snapshot |
| `replay_record` | Record an actual external delivery into a bounded buffer |
| `replay_schedule` | Schedule eight INTERNAL replay samples and drain their queue; excludes neural integration |

Runtime timestep is 0.5 ms: multiply seconds/tick by 2,000 for wall seconds
per simulated second. Component times overlap with full-runtime measurements
and should not be added together. Initialization includes fixture creation;
it is not just the named component's constructor. cProfile's own times identify
cost contributors; cumulative times overlap and must not be summed.

Array inventory deduplicates ndarray backing stores reachable from workload
state. It excludes Python-object overhead; tracemalloc peak measures additional
steady-state allocation, not process RSS or a GPU allocation. Its retained
count includes the final observation. Delivery logs grow during timed routing;
the benchmark reports a finite prefix, not an unlimited-run memory bound.

## Historical NumPy baseline

The four schema-1 reports below preserve the original #25/#26 measurements.
Their trace/replay buffers were not fully populated at measurement start and
`replay_record` fingerprinted only length. Those two workloads cannot support
steady-state allocation or replay-payload equivalence claims. Integration fixes
corrected both; current matched gate reports supersede those claims. The
recorded topology-validation timing comparison remains historical evidence,
not a timing baseline for a changed code revision.

`results/numpy_baseline.json` records the untouched runtime at PR #92 commit
`df5288c36a6fe588ed3381c63d964adbec6653df`, with its `lisnn` source digest in the
report. These are host measurements, not estimates for a user's desktop.

| Operation | 32 neurons | 512 neurons | 4,096 neurons |
|---|---:|---:|---:|
| Propagation | 0.088 ms | 0.845 ms | 19.622 ms |
| Triplet learner | 0.114 ms | 0.192 ms | 2.225 ms |
| Fixed tick | 0.559 ms | 1.273 ms | 20.086 ms |
| Adaptive tick | 0.677 ms | 2.528 ms | 43.100 ms |
| Regime tick | 1.274 ms | 3.179 ms | 44.091 ms |

Measured bottlenecks, ranked on the stress adaptive/regime workloads:

1. Repeated duplicate-edge checks: NumPy unique hashing and sorting take most
   of the profile time. Topology is revalidated on propagation and when learner
   weights become next-tick edges, although the generated topology is ordered.
2. Triplet edge calculations and observation arrays.
3. Transaction/snapshot copies; regime snapshots additionally copy delivery
   history and construction state. Their cost can grow with recording length.
4. Sparse `add.at` accumulation is much smaller than duplicate checking here.

Stress-case traced allocation peaks are about 4.59 MB for a fixed tick,
23.86 MB for adaptive stepping, and 32.81 MB for regime stepping (decimal MB).
This baseline contains no optimization changes. Issue #26 should first reduce
duplicate-check overhead while preserving rejection of malformed mutable
inputs and preserving edge order, then remeasure full trajectories and timing.

## Measured optimization (#26)

`SynapseEdges` now proves strictly ordered `(pre, post)` pairs unique with an
adjacent linear scan. Unordered input uses a single owned int64 key vector,
sorts that vector in place, and checks adjacent equal keys. This removes the
hash-plus-sort work of `np.unique` and avoids int64 key allocations entirely
on the canonical ordered path. It does not sort actual synapses, change float32
scatter order, cache mutable topology, or skip index/weight validation.

The full recorded run is `results/numpy_optimized.json`:

| 4,096 neurons / 131,072 ordered edges | Before | After | Speedup |
|---|---:|---:|---:|
| Public propagation, including validation | 19.622 ms | 0.703 ms | 27.92x |
| Fixed mixed-neuron tick | 20.086 ms | 1.467 ms | 13.69x |
| Adaptive mixed-neuron tick | 43.100 ms | 4.691 ms | 9.19x |
| Routed adaptive tick | 45.374 ms | 4.783 ms | 9.49x |
| ACTIVE/SLEEP regime tick | 44.091 ms | 6.889 ms | 6.40x |

Unordered edges are measured separately in `shuffled_reference.json` and
`shuffled_optimized.json`, using 5 repeats and the same seeded permutation.
Stress adaptive ticks improved from 44.119 to 16.387 ms (2.69x); regime ticks
from 44.657 to 18.141 ms (2.46x). These runs were sequential, not competing
benchmark processes. Ordered results also come from a separate sequential run.
Small-network changes and untouched microcomponents fluctuate: do not infer
a universal speedup or regression from these short host timings.

All 42 ordered and 8 unordered final observation fingerprints match exactly.
The comparison command rejects changed seeds/configuration, workload sets,
step counts, environment, or observation fingerprints before showing speedups:

```bash
python -m benchmarks.compare_profiles benchmarks/results/numpy_baseline.json benchmarks/results/numpy_optimized.json
python -m benchmarks.compare_profiles benchmarks/results/shuffled_reference.json benchmarks/results/shuffled_optimized.json
```

For current comparisons use `regression_gate` below. It switches only the
validation implementation within the same corrected source revision, so later
contract repairs cannot be mistaken for optimization-induced differences.

Historical #26 verification: 346 tests and all 19 smoke suites passed. Dedicated differential
tests compare every tick of all 14 workloads with the original duplicate-check
oracle in ordered and shuffled configurations. An additional 80-tick comparison
covers all nine neuron families together, each of off/Pair/Triplet/Voltage
plasticity, and freezing plasticity partway through the run. Index limits,
empty arrays, duplicate contacts, caller mutations and accumulation order are
also checked. No new JIT backend or learning rule is introduced.

Remaining costs include triplet update arrays and rollback/snapshot copies.
The current regime snapshots and delivery history still need care for long
runs; this change does not claim to bound their lifetime memory. The policy below completes #28. Numba remains conditional under #27.


## Regression policy (#28)

```bash
python -m benchmarks.regression_gate --output logs/regression-gate.json
python -m benchmarks.regression_gate --sizes small medium stress --steps 30 --repeats 5 --output logs/scaling-gate.json
```

Every case runs sequentially with the reference and optimized duplicate check,
using the same current source revision, seeds, configuration, requested warmup,
steps and repeat counts. The first mode alternates between cases. History and
replay-record fixtures are full before warmup. Replay fingerprints include all
retained entry metadata and current arrays, not just length. Schema-2 standalone
profiles also record the backend and these corrected measurement conventions.

Hard gates, including CI:

- Same final observation fingerprint and deterministic repeats in every workload.
- Equal reachable persistent ndarray backing bytes across backends.
- Optimized traced peak <= 1.25 * reference peak + 65,536 bytes, allowing small
  allocator bookkeeping differences while catching material new workspaces.
- Independent tests compare every tick of all workloads and mixed-neuron learner
  runs. Bounded histories/replay buffers keep identical storage after 500 more
  operations. The complete scientific tests and smoke suites run with both
  process defaults in CI.

CI records wall time without enforcing portable milliseconds on shared runners.
For an idle controlled host, opt into an end-to-end slowdown threshold:

```bash
python -m benchmarks.regression_gate --sizes stress --operations fixed_tick adaptive_tick routed_tick regime_tick --steps 100 --repeats 7 --enforce-timing --max-slowdown 1.5 --output logs/performance-gate.json
```

This fails if an optimized median is >1.5 times its paired reference median.
A failure is a reason to inspect load and rerun the same configuration, not to
weaken the threshold automatically. A new optimization must show an end-to-end
benefit on the workload it targets; beating this ceiling alone proves no gain.
Record environment/source/configuration and retain the raw samples with any
performance claim. Compare historical runs only when their measurement protocol
and environment match. Host baselines are evidence, not universal guarantees.

To disable the optimized duplicate-check path in a whole process, set
`LISNN_TOPOLOGY_VALIDATION=reference` before starting Python. Programmatic
`with lisnn.synapses.topology_validation("reference"):` is scoped and restores
the previous mode even on failure. All topology validity checks remain active.
The standalone profiler also accepts `--validation-backend reference`.

The memory gate measures a finite workload, not unlimited recording. Routing
logs and rollback snapshots may grow with experiment length; history/replay
capacity bounds do not bound those logs. No JIT kernel is currently justified
by a measured, validated replacement, so #27 remains deferred.


## Integrated baseline after review repairs

`results/integrated_gate.json` records 84 matched cases: all 14 workloads,
three sizes and both edge orders, with 30 steps, 3 repeats and 3 warmup steps.
All semantic, deterministic-repeat, persistent-storage and allocation gates
passed. Raw timing samples and environment/source hashes are retained.
The stress end-to-end cases also satisfy the optional 1.5x slowdown ceiling:

| Edge order | Operation | Reference ms | Optimized ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| ordered | fixed_tick | 22.427 | 1.467 | 15.29x |
| ordered | adaptive_tick | 43.852 | 4.844 | 9.05x |
| ordered | routed_tick | 45.739 | 5.173 | 8.84x |
| ordered | regime_tick | 48.348 | 6.734 | 7.18x |
| shuffled | fixed_tick | 20.796 | 7.875 | 2.64x |
| shuffled | adaptive_tick | 45.785 | 17.552 | 2.61x |
| shuffled | routed_tick | 44.755 | 17.906 | 2.50x |
| shuffled | regime_tick | 48.021 | 18.246 | 2.63x |

These remain measurements from this host and finite recording duration.
