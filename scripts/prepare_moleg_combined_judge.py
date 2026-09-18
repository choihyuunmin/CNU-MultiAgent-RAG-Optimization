"""Build blinded judge input from the combined run's private responses.

One response per (case_id, arm) is taken from the U1, repeat-0 trials (least
contention, first collected). The existing answer judge then scores relevance and
evidence support with the arm name hidden. This is automated silver evaluation,
never expert factual correctness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True, help="experiment root")
    p.add_argument("--judge-subdir", default="judge-u1", help="output subdir under root")
    p.add_argument("--users", type=int, default=1)
    p.add_argument("--repeat", type=int, default=0)
    args = p.parse_args()
    root = args.directory
    cases = json.loads((root / "cases.json").read_text())
    rows = [json.loads(l) for l in (root / "private-responses.jsonl").read_text().splitlines()]
    selected = [r for r in rows if r.get("users") == args.users and r.get("repeat") == args.repeat]
    picked = {}
    for r in selected:
        key = (r["case_id"], r["arm"])
        picked.setdefault(key, r)  # first occurrence
    dest = root / args.judge_subdir
    dest.mkdir(mode=0o700, exist_ok=False)
    (dest / "cases.json").write_text(json.dumps(cases, ensure_ascii=False))
    variants = sorted({r["arm"] for r in selected})
    expected = {(c["case_id"], v) for c in cases if c.get("reference_answer") for v in variants}
    observed = {k for k in picked if any(c["case_id"] == k[0] and c.get("reference_answer") for c in cases)}
    with (dest / "e2e_stream.jsonl").open("x") as sink:
        for (case_id, arm), r in picked.items():
            sink.write(json.dumps({"case_id": case_id, "variant": arm, "repeat": args.repeat,
                                   "result": r.get("result")}, ensure_ascii=False) + "\n")
    for path in dest.iterdir():
        path.chmod(0o600)
    print(json.dumps({"variants": variants, "reference_cases": sum(bool(c.get("reference_answer")) for c in cases),
                      "records_written": len(picked), "reference_records": len(observed),
                      "expected_reference_records": len(expected),
                      "complete_reference_coverage": observed == expected,
                      "scope": "blinded silver judge input; not new inference; not expert gold"}))


if __name__ == "__main__":
    main()
