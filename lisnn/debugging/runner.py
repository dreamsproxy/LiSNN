"""Run registered smoke suites with progress, diagnostics and process status."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from lisnn.debugging.foundation_smoke import index_smoke_test, input_smoke_test, units_smoke_test
from lisnn.debugging.network_smoke import network_smoke_test, spatial_smoke_test
from lisnn.debugging.neuron_smoke import population_smoke_test
from lisnn.debugging.synapse_smoke import synapse_smoke_test
from lisnn.debugging.propagation_smoke import propagation_smoke_test
from lisnn.debugging.plasticity_observation_smoke import plasticity_observation_smoke_test
from lisnn.debugging.pair_smoke import pair_smoke_test
from lisnn.debugging.triplet_smoke import triplet_smoke_test
from lisnn.debugging.voltage_smoke import voltage_smoke_test
from lisnn.debugging.recurrent_smoke import recurrent_smoke_test


def _population(log_dir, verbose):
    return population_smoke_test(
        neuron_count=8, n_steps=8, watch_spikes=True,
        ouput_path=str(log_dir / "population.log"), verbose=verbose,
    )


def _suite(function):
    return lambda log_dir, verbose: function(verbose=verbose)


SMOKE_TESTS = {
    "population": _population,
    "synapses": _suite(synapse_smoke_test),
    "indices": _suite(index_smoke_test),
    "units": _suite(units_smoke_test),
    "inputs": _suite(input_smoke_test),
    "network": _suite(network_smoke_test),
    "spatial": _suite(spatial_smoke_test),
    "propagation": _suite(propagation_smoke_test),
    "plasticity_observation": _suite(plasticity_observation_smoke_test),
    "pair_stdp": _suite(pair_smoke_test),
    "triplet_stdp": _suite(triplet_smoke_test),
    "voltage_stdp": _suite(voltage_smoke_test),
    "recurrent_learning": _suite(recurrent_smoke_test),
}


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_all_smoke_tests(log_dir=None, only=None, verbose=False, progress=True):
    """Run all suites (or named selections); return results and write summary.json.

    Suite exceptions become failures with tracebacks; subsequent suites still run.
    KeyboardInterrupt/SystemExit are not swallowed. No pytest dependency is used.
    """
    names = list(SMOKE_TESTS) if only is None else list(dict.fromkeys(only))
    if not names or any(name not in SMOKE_TESTS for name in names):
        raise ValueError(f"Select at least one known smoke suite: {', '.join(SMOKE_TESTS)}")
    if log_dir is None:
        log_dir = Path("logs") / datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    total = len(names)
    for index, name in enumerate(names, 1):
        if progress:
            print(f"[{index}/{total}] Running {name}...", flush=True)
        start = perf_counter()
        try:
            result = SMOKE_TESTS[name](log_dir, verbose)
            # Missing/invalid status must never be reported as a passing suite.
            if not isinstance(result, dict) or not isinstance(result.get("passed"), (bool, np.bool_)):
                raise TypeError("Smoke suite must return a dict with a boolean 'passed'")
            result = dict(result)
            result["passed"] = bool(result["passed"])
        except Exception as exc:
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        result["elapsed_seconds"] = perf_counter() - start
        results[name] = result
        # Keep full details per suite, including arrays from existing diagnostics.
        (log_dir / f"{name}.json").write_text(json.dumps(result, indent=2, default=_json_value) + "\n")
        if progress:
            filled = 20 * index // total
            bar = "=" * filled + "." * (20 - filled)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{bar}] {index}/{total} {status} {name} ({result['elapsed_seconds']:.3f}s)", flush=True)
            if not result["passed"]:
                print(f"  Details: {log_dir / (name + '.json')}", flush=True)
    failed = [name for name, result in results.items() if not result["passed"]]
    summary = {
        "passed": not failed, "total": total, "failed": failed,
        "log_dir": str(log_dir), "results": results,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_value) + "\n")
    if progress:
        print(f"{'PASS' if not failed else 'FAIL'}: {total - len(failed)}/{total} suites passed. Reports: {log_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run all active LiSNN smoke tests.")
    parser.add_argument("--list", action="store_true", help="List available suites without running them")
    parser.add_argument("--only", nargs="+", choices=tuple(SMOKE_TESTS), help="Run selected suites")
    parser.add_argument("--log-dir", type=Path, help="Report directory (default: a new timestamped logs directory)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed checks and neuron traces")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(SMOKE_TESTS))
        return 0
    try:
        result = run_all_smoke_tests(args.log_dir, args.only, args.verbose)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Smoke runner failed: {exc}")
        return 1
    return 0 if result["passed"] else 1
