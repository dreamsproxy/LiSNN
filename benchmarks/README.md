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

## Captured NumPy baseline

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
