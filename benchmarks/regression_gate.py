"""Matched reference/optimized checks with optional controlled-host timing gates."""

import argparse
import json
from pathlib import Path

from benchmarks.profile_playground import (Config, OPERATIONS, SIZES, environment,
                                           measure, source_digest)
from lisnn.synapses import topology_validation


END_TO_END = {"fixed_tick", "adaptive_tick", "routed_tick", "regime_tick"}


def assess(reference, optimized, *, enforce_timing=False, max_slowdown=1.5):
    """Hard semantic/storage gates; wall time is opt-in on a controlled host."""
    failures = []
    if reference["config"] != optimized["config"] or reference["operation"] != optimized["operation"]:
        raise ValueError("gate requires the same workload configuration")
    if reference["final_observation_sha256"] != optimized["final_observation_sha256"]:
        failures.append("observation mismatch")
    if reference["arrays_after_timing"]["unique_backing_bytes"] != optimized["arrays_after_timing"]["unique_backing_bytes"]:
        failures.append("persistent array footprint changed")
    # Allow allocator bookkeeping while rejecting material extra workspaces.
    peak_limit = 1.25 * reference["allocations"]["peak_bytes"] + 65536
    if optimized["allocations"]["peak_bytes"] > peak_limit:
        failures.append("temporary allocation peak exceeds reference allowance")
    ratio = optimized["median_seconds"] / reference["median_seconds"]
    if enforce_timing and reference["operation"] in END_TO_END and ratio > max_slowdown:
        failures.append("end-to-end timing regression")
    return dict(passed=not failures, failures=failures, speedup=1 / ratio,
                optimized_peak_limit_bytes=peak_limit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=["small"])
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--edge-orders", nargs="+", choices=("ordered", "shuffled"), default=["ordered", "shuffled"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--enforce-timing", action="store_true")
    parser.add_argument("--max-slowdown", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.steps, args.repeats) <= 0 or args.warmup < 0 or not 1 <= args.max_slowdown < float("inf"):
        parser.error("steps/repeats must be positive, warmup nonnegative, slowdown finite and >= 1")
    jobs = [(size, order, op) for size in args.sizes for order in args.edge_orders for op in args.operations]
    report = dict(schema=1, environment=environment(), source_sha256=source_digest(),
                  steps=args.steps, repeats=args.repeats, warmup=args.warmup,
                  timing_enforced=args.enforce_timing, max_slowdown=args.max_slowdown,
                  bounded_state_prefilled=True, results=[], passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, (size, order, operation) in enumerate(jobs, 1):
        rows = {}
        # Alternate order across cases; never run competing timing processes.
        modes = ("reference", "optimized") if index % 2 else ("optimized", "reference")
        for mode in modes:
            with topology_validation(mode):
                rows[mode] = measure(Config(*SIZES[size], seed=args.seed, edge_order=order),
                                     operation, steps=args.steps, repeats=args.repeats,
                                     warmup=args.warmup, diagnostics=True)
        # Keep raw timing samples and memory totals; detailed profiles belong
        # to profile_playground, avoiding duplicate stacks in gate artifacts.
        for row in rows.values():
            row.pop("profile_top_own_time", None)
            row["allocations"].pop("top_retained", None)
            row["arrays_after_timing"].pop("largest", None)
        verdict = assess(rows["reference"], rows["optimized"],
                         enforce_timing=args.enforce_timing, max_slowdown=args.max_slowdown)
        report["results"].append(dict(size=size, edge_order=order, operation=operation,
                                      **rows, **verdict))
        filled = 20 * index // len(jobs)
        print(f"[{'=' * filled}{'.' * (20 - filled)}] {index}/{len(jobs)} "
              f"{size}/{order}/{operation}: {'PASS' if verdict['passed'] else 'FAIL'} "
              f"({verdict['speedup']:.2f}x)", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["passed"] = all(row["passed"] for row in report["results"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(jobs)} matched cases; {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
