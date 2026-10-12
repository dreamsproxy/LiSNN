"""Measure the NumPy playground before optimizing it (#25).

Timing, profiling and allocation tracing run separately. No JIT or optional
media libraries are needed. All stateful trials start from the same seed.
"""

import argparse
from collections import deque
import cProfile
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import time
import tracemalloc

import numpy as np

from lisnn.io import (InterfacePort, Probe, ReplayBuffer, ReplayController,
                      RegimeController, StreamFrame, StreamRuntime, Transducer)
from lisnn.network import create_nn
from lisnn.neurons.registry import get_step_function
from lisnn.plasticity import PairSTDP, TripletSTDP, VoltageSTDP
from lisnn.plasticity.traces import PlasticityTraces
from lisnn.synapses import create_fixed_out_degree
from lisnn.synapses.propagation import propagate


SIZES = {"small": (32, 4), "medium": (512, 16), "stress": (4096, 32)}
OPERATIONS = ("neurons", "propagation", "pair", "triplet", "voltage",
              "trace_history", "transducer", "probe", "fixed_tick",
              "adaptive_tick", "routed_tick", "regime_tick", "replay_record",
              "replay_schedule")
TRIPLET = dict(tau_plus_ms=20, tau_minus_ms=20, tau_x_ms=50, tau_y_ms=50,
               a2_plus=0.0001, a2_minus=0.0001, a3_plus=0.0001,
               a3_minus=0.0001, w_max=1)


@dataclass(frozen=True)
class Config:
    population: int
    degree: int
    seed: int = 7
    dt_ms: float = 0.5


def fingerprint(value):
    """Stable diagnostic of full returned observations, not a timing metric."""
    digest = hashlib.sha256()

    def visit(v):
        if isinstance(v, np.ndarray):
            digest.update(str((v.dtype.str, v.shape)).encode())
            digest.update(v.tobytes())
        elif is_dataclass(v):
            for f in fields(v):
                digest.update(f.name.encode())
                visit(getattr(v, f.name))
        elif isinstance(v, dict):
            for key in sorted(v):
                visit(key)
                visit(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            digest.update(repr(v).encode())

    visit(value)
    return digest.hexdigest()


def array_inventory(root):
    """Count unique ndarray backing allocations reachable from workload state."""
    seen, allocations = set(), {}

    def visit(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            base = value
            while isinstance(base.base, np.ndarray):
                base = base.base
            allocations.setdefault(id(base), dict(path=path, bytes=base.nbytes,
                                                  shape=list(base.shape), dtype=str(base.dtype)))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple, deque)):
            for i, item in enumerate(value):
                visit(item, f"{path}[{i}]")
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
        elif hasattr(value, "__dict__") and not callable(value):
            visit(vars(value), path)

    visit(root, "state")
    entries = sorted(allocations.values(), key=lambda x: x["bytes"], reverse=True)
    return dict(unique_backing_bytes=sum(x["bytes"] for x in entries),
                array_count=len(entries), largest=entries[:8])


def workload(config, operation):
    """Return (one_operation(index), owned_state). Setup is timed separately."""
    n, degree, dt = config.population, config.degree, config.dt_ms
    net = create_nn(n, neuron_type={"default": "LIF", "AdEx": n // 4,
                                    "Izhikevich": n // 4}, seed=config.seed)
    edges = create_fixed_out_degree(n, degree, weight_range=(0.01, 0.02), seed=config.seed)
    indices = np.arange(min(n, 32), dtype=np.int32)
    current = np.full(n, 600, dtype=np.float32)
    spikes = (np.arange(n) % 7 == 0).astype(np.float32)
    voltage = np.full(n, -45, dtype=np.float32)
    state = dict(network=net, edges=edges, current=current, spikes=spikes, voltage=voltage)
    if operation == "neurons":
        # Homogeneous LIF integration isolates the kernel from runtime dispatch.
        net = create_nn(n, seed=config.seed)
        state["network"] = net
        kernel = get_step_function("LIF")
        return lambda _: kernel(net.pool, current, dt, return_observation=True), state
    if operation == "propagation":
        return lambda _: propagate(spikes, edges, dt, 5, return_details=True), state
    if operation in ("pair", "triplet", "voltage"):
        if operation == "pair":
            learner = PairSTDP(edges, tau_plus_ms=20, tau_minus_ms=20,
                               a_plus=0.0001, a_minus=0.0001, w_max=1)
        elif operation == "triplet":
            learner = TripletSTDP(edges, **TRIPLET)
        else:
            learner = VoltageSTDP(edges, tau_x_ms=20, tau_minus_ms=10,
                                  tau_plus_ms=7, theta_minus_mV=-70, theta_plus_mV=-50,
                                  a_ltd=0.000001, a_ltp=0.00000001, w_max=1,
                                  initial_voltage_mV=voltage)
        state["learner"] = learner
        return (lambda _: learner.step(spikes, voltage, dt) if operation == "voltage"
                else learner.step(spikes, dt)), state
    if operation == "trace_history":
        traces = PlasticityTraces(n, tau_spike_ms=20, tau_voltage_ms=10,
                                 history_ms=32 * dt, history_max_samples=32)
        state["traces"] = traces
        return lambda _: traces.advance(spikes, voltage, dt), state
    external = InterfacePort("external", n, indices, source="EXTERNAL")
    internal = InterfacePort("internal", n, indices, source="INTERNAL")
    transducers = {p.name: Transducer(p) for p in (external, internal)}
    frame = StreamFrame("EXTERNAL", "CONTINUOUS", "auditory", current[indices],
                        ("channel",), 0, "benchmark:seeded-port")
    state.update(transducers=transducers, frame=frame)
    if operation == "transducer":
        return lambda _: transducers["external"].to_current(frame), state
    net.configure_runtime(edges, dt=dt, impulse_scale=5,
                          plasticity="off" if operation == "fixed_tick" else "triplet",
                          plasticity_params=None if operation == "fixed_tick" else TRIPLET)
    if operation in ("fixed_tick", "adaptive_tick"):
        return lambda _: net.step(current), state
    if operation == "probe":
        result = net.step(current)
        probe = Probe(external)
        state.update(result=result, probe=probe)
        return lambda _: probe.capture(result), state
    routed = StreamRuntime(net, transducers=transducers, max_queued_frames=256)
    regime = RegimeController(routed)
    state.update(routed=routed, regime=regime)
    buffer = ReplayBuffer(max_frames=32, max_duration_ms=100)
    state["buffer"] = buffer
    if operation in ("routed_tick", "regime_tick"):
        def step(i):
            routed.schedule_frame(StreamFrame("EXTERNAL", "CONTINUOUS", "auditory",
                                  current[indices], ("channel",), routed.tick * dt,
                                  "benchmark:external"), "external")
            if operation == "regime_tick":
                regime.transition(routed.tick, "SLEEP" if i % 2 else "ACTIVE")
                return regime.step(current)
            return routed.step(current)
        return step, state
    # Populate a real delivery buffer once, outside steady-state measurements.
    for i in range(8):
        routed.schedule_frame(StreamFrame("EXTERNAL", "CONTINUOUS", "auditory",
                              current[indices], ("channel",), routed.tick * dt,
                              "benchmark:replay"), "external")
        result = routed.step()
        buffer.record(result)
    if operation == "replay_record":
        state["result"] = result
        def record(_):
            buffer.record(result)
            return len(buffer._entries)
        return record, state
    if operation == "replay_schedule":
        regime.transition(routed.tick, "SLEEP")
        regime.step()
        replay = ReplayController(buffer, internal_ports={"auditory": "internal"},
                                  max_gain=1, max_repeats=1, max_events=16,
                                  max_duration_ticks=16, seed=config.seed)
        state["replay"] = replay
        def schedule(_):
            plan = replay.schedule(regime, observed_tick=routed.tick - 1,
                                   start_tick=routed.tick, order="ordered")
            # Include draining the scheduler, but no neural integration here.
            for _, tick in plan:
                routed.scheduler.consume(tick)
            return plan
        return schedule, state
    raise ValueError(f"unknown operation {operation}")


def profile_rows(profile):
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profile).stats.items():
        path = filename.replace(str(Path(__file__).resolve().parents[1]) + "/", "")
        rows.append(dict(function=f"{path}:{line}:{name}", calls=calls,
                         own_seconds=own, cumulative_seconds=cumulative))
    return sorted(rows, key=lambda row: row["own_seconds"], reverse=True)[:12]


def measure(config, operation, *, steps, repeats, warmup, diagnostics):
    samples, initialization, digests = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        run, state = workload(config, operation)
        initialization.append(time.perf_counter() - start)
        for i in range(warmup):
            run(i)
        start = time.perf_counter()
        for i in range(warmup, warmup + steps):
            result = run(i)
        samples.append((time.perf_counter() - start) / steps)
        digests.append(fingerprint(result))
    if len(set(digests)) != 1:
        raise RuntimeError(f"non-reproducible result: {operation}")
    report = dict(config=asdict(config), operation=operation, edges=config.population * config.degree,
                  seconds_per_operation=samples, median_seconds=statistics.median(samples),
                  initialization_seconds=initialization, final_observation_sha256=digests[0],
                  arrays_after_timing=array_inventory(state))
    if diagnostics:
        run, state = workload(config, operation)
        for i in range(warmup):
            run(i)
        profiler = cProfile.Profile()
        profiler.enable()
        for i in range(warmup, warmup + steps):
            run(i)
        profiler.disable()
        report["profile_top_own_time"] = profile_rows(profiler)
        run, state = workload(config, operation)
        for i in range(warmup):
            run(i)
        # Setup memory is reported via owned arrays. Trace only steady-state allocations.
        tracemalloc.start()
        for i in range(warmup, warmup + steps):
            result = run(i)
        retained, peak = tracemalloc.get_traced_memory()
        allocations = tracemalloc.take_snapshot().statistics("lineno")[:8]
        tracemalloc.stop()
        report["allocations"] = dict(retained_bytes=retained, peak_bytes=peak,
            top_retained=[dict(location=str(s.traceback), bytes=s.size, count=s.count) for s in allocations])
    return report


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "lisnn").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("steps/repeats must be positive and warmup nonnegative")
    report = dict(schema=1, environment=dict(python=platform.python_version(),
        numpy=np.__version__, platform=platform.platform(), processor=platform.processor(),
        cpu_count=os.cpu_count(), thread_environment={k: os.environ.get(k) for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
        source_sha256=source_digest(), steps=args.steps, repeats=args.repeats,
        warmup=args.warmup, jit="none; NumPy reference", results=[])
    jobs = [(size, op) for size in args.sizes for op in args.operations]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, (size, op) in enumerate(jobs, 1):
        n, degree = SIZES[size]
        row = measure(Config(n, degree, args.seed), op, steps=args.steps,
                      repeats=args.repeats, warmup=args.warmup, diagnostics=not args.timing_only)
        row["size"] = size
        report["results"].append(row)
        filled = 20 * index // len(jobs)
        print(f"[{'=' * filled}{'.' * (20 - filled)}] {index}/{len(jobs)} "
              f"{size}/{op}: {row['median_seconds'] * 1000:.3f} ms/op", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
