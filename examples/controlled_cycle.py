"""Run a transparent P1.2 state-cycle comparison with four controls.

Edit `washout_ticks`, `sleep_ticks` or `replay_gain` in the call below to
explore the declared example. No training objective or fit loop is hidden.
"""

import argparse
import json
from pathlib import Path

from lisnn.experiments import run_controlled_cycle


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the controlled P1.2 state-cycle experiment")
    parser.add_argument("--json", type=Path, help="Write complete raw observations to this JSON file")
    args = parser.parse_args(argv)
    result = run_controlled_cycle(
        washout_ticks=100, sleep_ticks=6, replay_gain=1, shuffle_seed=3,
        progress=lambda i, n, label: print(f"[{i}/{n}] {label}", flush=True),
    )
    print("configuration:", json.dumps(result["configuration"], indent=2))
    print("experience:", json.dumps(result["experience"], indent=2))
    print("baseline probe:", json.dumps(result["baseline_probe"], indent=2))
    for name, arm in result["controls"].items():
        print(name, "readout pA:", arm["probe"]["readout_synaptic_current_pA"],
              "change pA:", arm["change_from_baseline_pA"],
              "sleep weights:", arm["sleep_weights"],
              "readout spike:", arm["probe"]["readout_spike"])
        print("  delivery provenance:", arm["deliveries"])
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print("raw observations:", args.json)


if __name__ == "__main__":
    main()
