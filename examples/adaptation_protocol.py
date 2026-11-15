"""Prepare a P1.4 protocol package; never launch adaptation experiments."""

import argparse
import json
from pathlib import Path

from lisnn.experiments.evidence import EvidencePackage
from lisnn.experiments.protocol import engineering_protocol, freeze_protocol
from lisnn.experiments.provenance import capture_provenance


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate protocol field: {key}")
        value[key] = item
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, help="reviewed protocol declaration JSON; default is an engineering-only planning fixture")
    parser.add_argument("--output", type=Path, default=Path("logs/p1.4-plan"),
                        help="new package directory; existing output is never overwritten")
    parser.add_argument("--check", action="store_true", help="validate and enumerate only; write nothing")
    args = parser.parse_args(argv)
    try:
        declaration = engineering_protocol() if args.protocol is None else json.loads(
            args.protocol.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
        frozen = freeze_protocol(declaration)
        print(f"scope: {declaration['scope']}; planned runs: {len(frozen['run_ids'])}; executed runs: 0")
        if args.check:
            return frozen
        provenance = capture_provenance(Path(__file__).resolve().parents[1])
        package = EvidencePackage.create(args.output, frozen, provenance)
        print(f"frozen plan: {package.path / 'protocol.json'}")
        print("No adaptation, recall, retention or transfer evidence has been collected.")
        return package.summary()
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
