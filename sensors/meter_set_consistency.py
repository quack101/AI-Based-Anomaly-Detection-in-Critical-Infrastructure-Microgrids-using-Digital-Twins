"""
Check that a meter JSON, its registry and its corpus describe the SAME
meters. Raises MeterSetMismatch listing every disagreement -- never
returns a partial pass -- so a pipeline step that mixes files from two
meter sets (e.g. set B's registry with set A's corpus) fails loudly
instead of producing numbers.

Checks:
  1. Every meter in the JSON has registry entries with the same
     meter_id prefix (Mxxx_*), and those entries carry the same
     load_name and bus as the JSON.
  2. Every Mxxx_* registry entry belongs to a meter in the JSON.
  3. If a corpus folder is given: every {split}_telemetry.csv header is
     exactly timestamp, real_timestamp, then the registry's enabled ids
     in registry order (the estimator reads columns by these ids).
"""

import csv
import json
from pathlib import Path


class MeterSetMismatch(ValueError):
    pass


def check_meter_set(meters_json, registry_json, corpus_dir=None):
    meters = json.loads(Path(meters_json).read_text())
    registry = json.loads(Path(registry_json).read_text())
    problems = []

    by_id = {m["meter_id"]: m for m in meters}
    meter_entries = [e for e in registry if e["id"][:1] == "M" and "_" in e["id"]]
    seen = set()
    for e in meter_entries:
        mid = e["id"].split("_", 1)[0]
        m = by_id.get(mid)
        if m is None:
            problems.append(f"registry entry {e['id']} has no meter {mid} in {meters_json}")
            continue
        seen.add(mid)
        if str(e.get("node")) != str(m["bus"]):
            problems.append(f"{e['id']}: registry bus {e.get('node')} != meter JSON bus {m['bus']}")
        if "load_name" in e and e["load_name"].lower() != m["load_name"].lower():
            problems.append(f"{e['id']}: registry load {e['load_name']} != meter JSON load {m['load_name']}")
    for mid in sorted(set(by_id) - seen):
        problems.append(f"meter {mid} ({by_id[mid]['load_name']}) has no registry entries in {registry_json}")

    if corpus_dir is not None:
        enabled = [e["id"] for e in registry if e.get("enabled")]
        expected = ["timestamp", "real_timestamp"] + enabled
        for split in ("train", "val", "test"):
            path = Path(corpus_dir) / f"{split}_telemetry.csv"
            if not path.exists():
                problems.append(f"missing corpus file {path}")
                continue
            with open(path, newline="") as f:
                header = next(csv.reader(f))
            if header != expected:
                extra = [c for c in header if c not in expected]
                missing = [c for c in expected if c not in header]
                problems.append(f"{path.name}: header does not match registry enabled ids "
                                f"(missing {missing[:5]}, extra {extra[:5]}, "
                                f"same set but different order: {not extra and not missing})")

    if problems:
        raise MeterSetMismatch(f"{len(problems)} mismatch(es):\n  " + "\n  ".join(problems))
    return {"meters": len(meters), "registry_meter_entries": len(meter_entries),
            "corpus_checked": corpus_dir is not None}
