"""Freeze the two-arm combined-harness experiment (plan, 200 cases, arm configs).

The 200 questions are a stratified selection from the established regression set,
not an independent expert holdout. The search-node source is fingerprinted and
the ProgramHarness transform is compile-checked here, so a mismatched application
revision fails closed before any app is launched.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_moleg_scaling import select_cases, digest  # noqa: E402
from program_harness_attachment import transform  # noqa: E402

USERS = [1, 4, 16]


def fingerprint_and_check(app_src: Path):
    nodes = app_src / "core/query_loop/law_search/nodes.py"
    source = nodes.read_text()
    tree = ast.parse(source)
    func = next(n for n in tree.body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "execute_search")
    transformed = transform(ast.get_source_segment(source, func))
    compile(transformed, "execute_search_overlay", "exec")  # fail closed on bad transform
    return hashlib.sha256(nodes.read_bytes()).hexdigest()


def runtime_digests(repo: Path):
    files = [p for folder in ("src", "scripts", "integrations")
             for p in (repo / folder).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json"} and "__pycache__" not in p.parts]
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def prepare(directory: Path, cases_path: Path, app_src: Path, repo: Path, seed: int, limit: int):
    cases = json.loads(cases_path.read_text())
    if len({c["case_id"] for c in cases}) != len(cases) or len({c["question"] for c in cases}) != len(cases):
        raise ValueError("source case set must have unique ids and questions")
    selected = select_cases(cases, limit, seed)
    remaining = [c for c in cases if c["case_id"] not in {s["case_id"] for s in selected}]
    smoke = select_cases(remaining, 8, seed + 1) if len(remaining) >= 8 else remaining[:8]

    nodes_sha = fingerprint_and_check(app_src)

    baseline_cfg = json.loads((repo / "integrations/2025-moleg-search/workflow-combined-baseline.json").read_text())
    combined_cfg = json.loads((repo / "integrations/2025-moleg-search/workflow-combined.json").read_text())

    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_started",
        "experiment": "combined-harness (baseline vs reasoning-low + program-overlap)",
        "unique_questions": len(selected),
        "smoke_questions": len(smoke),
        "arms": ["baseline", "combined"],
        "users": USERS,
        "repeats": 2,
        "seed": seed,
        "timeout_s": 240,
        "slo_s": 30,
        "planned_primary_requests": len(selected) * len(USERS) * 2 * 2,
        "planned_smoke_requests": len(smoke) * 2,
        "application_admission": False,
        "client_admission": False,
        "emission": "immediate",
        "law_search_nodes_sha256": nodes_sha,
        "source_reference_questions": sum(bool(c.get("reference_answer")) for c in selected),
        "kinds": {k: sum(c["kind"] == k for c in selected) for k in sorted({c["kind"] for c in selected})},
        "scope": "established regression subset, not an independent holdout; latency not pooled with historical runs",
        "rules": [
            "retain every individual failure and incomplete stream",
            "report completion latency separately from time to failure",
            "no claimed acceleration if request success, source recall, or judged support regresses",
            "identical question order per (load, repeat) across arms; arm order rotated",
            "freeze policy before smoke; no silent retuning on smoke results",
            "read-only shared backend; do not restart or reconfigure model servers",
        ],
        "source_cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
    }
    plan["runtime_sha256"] = runtime_digests(repo)

    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    (directory / "apps").mkdir(mode=0o700)
    for name, value in [("cases.json", selected), ("smoke.cases.json", smoke),
                        ("baseline.adapter.json", baseline_cfg),
                        ("combined.adapter.json", combined_cfg), ("plan.json", plan)]:
        fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as sink:
            json.dump(value, sink, ensure_ascii=False, indent=2)
            sink.write("\n")
    return {k: plan[k] for k in ("status", "unique_questions", "smoke_questions",
                                 "planned_primary_requests", "source_reference_questions",
                                 "law_search_nodes_sha256", "kinds")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--app-src", type=Path, required=True, help="frozen application src/ root")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(json.dumps(prepare(args.directory, args.cases, args.app_src, repo, args.seed, args.limit)))


if __name__ == "__main__":
    main()
