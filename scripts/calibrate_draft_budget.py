"""Freeze a draft-budget policy from paired engine decode measurements."""
import argparse
import hashlib
import json
from pathlib import Path

from cnu_rag_optimization.speculation_budget import (
    DecodeContext, DecodeMeasurement, DraftBudgetPolicy, calibrate,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    raw = args.measurements.read_bytes()
    for line in raw.splitlines():
        value = json.loads(line)
        value["context"] = DecodeContext(**value["context"])
        rows.append(DecodeMeasurement(**value))
    if not rows:
        raise ValueError("no measurements; do not manufacture a calibration")
    policy = DraftBudgetPolicy(calibrate(rows)).to_dict()
    policy.update(measurements_sha256=hashlib.sha256(raw).hexdigest(),
                  measurement_count=len(rows), live_validation_complete=False,
                  promote_to_full_experiment=False)
    with args.output.open("x") as f:
        json.dump(policy, f, indent=2)
        f.write("\n")
    print(json.dumps({"contexts": len(policy["choices"]), "live_validation_complete": False}))


if __name__ == "__main__":
    main()
