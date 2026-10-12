"""Deterministic controlled state-cycle engineering exit check."""

import numpy as np

from lisnn.debugging.checks import report
from lisnn.experiments import run_controlled_cycle


def cycle_smoke_test(verbose=True):
    result = run_controlled_cycle(washout_ticks=40)
    arms = result["controls"]
    baseline = result["baseline_weights"][0]
    checks = {
        "same_baseline_negative_controls": np.allclose(arms["quiet"]["sleep_weights"], [baseline])
                                           and np.allclose(arms["ordered_learning_off"]["sleep_weights"], [baseline]),
        "matched_replay_budget": len(arms["ordered"]["deliveries"]) == len(arms["shuffled"]["deliveries"]) == 2,
        "fixed_probe_definition": all(np.isfinite(arm["probe"]["readout_synaptic_current_pA"])
                                      for arm in arms.values()),
        "raw_observations": all(arm["sleep_observations"] for arm in arms.values()),
    }
    return report(checks, verbose)


if __name__ == "__main__":
    raise SystemExit(0 if cycle_smoke_test()["passed"] else 1)
