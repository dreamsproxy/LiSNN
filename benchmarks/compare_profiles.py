"""Compare matching benchmark workloads and require exact observations."""

import argparse
import json
from pathlib import Path


def compare(before, after):
    for field in ("schema", "steps", "repeats", "warmup", "environment"):
        if before[field] != after[field]:
            raise ValueError(f"benchmark {field} differs; use a matched run")
    def index(report):
        rows = {}
        for row in report["results"]:
            key = (row["size"], row["operation"])
            if key in rows:
                raise ValueError(f"duplicate benchmark case {key}")
            rows[key] = row
        return rows
    a, b = index(before), index(after)
    if a.keys() != b.keys():
        raise ValueError("benchmark case sets differ")
    result = []
    for key in a:
        old, new = a[key], b[key]
        old_config = {"edge_order": "ordered", **old["config"]}
        new_config = {"edge_order": "ordered", **new["config"]}
        if old_config != new_config:
            raise ValueError(f"configuration differs for {key}")
        if old["final_observation_sha256"] != new["final_observation_sha256"]:
            raise ValueError(f"observation differs for {key}")
        result.append(dict(size=key[0], operation=key[1], exact_observation=True,
                           before_ms=1000 * old["median_seconds"],
                           after_ms=1000 * new["median_seconds"],
                           speedup=old["median_seconds"] / new["median_seconds"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    rows = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    print("size / operation                    before ms    after ms    speedup")
    for row in rows:
        print(f"{row['size'] + '/' + row['operation']:34} {row['before_ms']:10.3f} "
              f"{row['after_ms']:11.3f} {row['speedup']:9.2f}x")
    print(f"{len(rows)} cases: exact final observations; timings are host-specific.")


if __name__ == "__main__":
    main()
