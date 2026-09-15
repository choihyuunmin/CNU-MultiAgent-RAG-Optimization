"""Freeze the existing 300-question regression set and two unrestricted arms."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from evaluate_moleg_isolated import trial_schedule
from evaluate_moleg_scaling import digest, select_cases
from prepare_moleg_workflow_campaign import HOOKS


def prepare(directory, cases_path, previous_manifest, repository):
    cases = json.loads(cases_path.read_text())
    previous = json.loads(previous_manifest.read_text())["selected_cases"]
    expected = {c["case_id"]: c["question_sha256"] for c in previous}
    available = {c["case_id"]: c for c in cases}
    if len(previous) != 300 or len(expected) != 300 or len(available) != len(cases):
        raise ValueError("expected the previous 300 unique questions and a unique source dataset")
    if any(i not in available or digest(available[i]["question"]) != h for i, h in expected.items()):
        raise ValueError("previous question identity differs from local source")
    selected = [available[c["case_id"]] for c in previous]
    smoke = select_cases([c for c in cases if c["case_id"] not in expected], 8, 20260911)
    hooks = [dict(module=m, function=f, stage=s) for m, f, s in HOOKS]
    baseline = dict(mode="observe", stage_hooks=hooks)
    candidate = dict(baseline, reasoning_policies={"worker agent": dict(
        effort="low", first_output_timeout_s=60, max_reasoning_characters=32768)})
    users, variants = [1, 2, 4, 8, 16, 32], ["baseline", "reasoning"]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status="prepared_not_started", unique_questions=300, smoke_questions=8,
        planned_smoke_requests=16, planned_primary_requests=7200,
        variants=variants, users=users, repeats=2, seed=20260911, timeout_s=240,
        schedule=[dict(repeat=r, users=u, policy=n) for r, u, n in trial_schedule(variants, users, 2)],
        application_admission=False, client_admission=False, emission="immediate",
        source_reference_questions=sum(bool(c.get("reference_answer")) for c in selected),
        quality_repeat=0, selected_cases=previous,
        smoke_cases=[dict(case_id=c["case_id"], kind=c["kind"], question_sha256=digest(c["question"])) for c in smoke],
        scope="existing regression dataset, not independent holdout; do not pool with historical latency",
        rules=["retain all individual failures and incomplete streams",
               "report successful completion latency separately from time to failure",
               "no claimed acceleration if success or answer support regresses",
               "freeze policy before smoke; no silent holdout retuning",
               "stop escalation on smoke failure, infrastructure fault or undrained app",
               "do not alter production models or deployments"],
        input_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        previous_manifest_sha256=hashlib.sha256(previous_manifest.read_bytes()).hexdigest())
    files = [p for folder in ["src", "scripts", "integrations"]
             for p in (repository / folder).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json"} and "__pycache__" not in p.parts]
    plan["runtime_sha256"] = {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(files)}
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, value in [("cases.json", selected), ("smoke.cases.json", smoke),
                        ("baseline.json", baseline), ("reasoning.json", candidate), ("plan.json", plan)]:
        fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as sink:
            json.dump(value, sink, ensure_ascii=False, indent=2)
            sink.write("\n")
    return {key: plan[key] for key in ["status", "unique_questions", "planned_smoke_requests",
                                      "planned_primary_requests", "source_reference_questions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.cases, args.previous_manifest,
                             Path(__file__).resolve().parents[1])))


if __name__ == "__main__":
    main()
