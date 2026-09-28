"""Extract reproducible, content-free request and trace records from private logs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXPERIMENT = "diagnostic-20260928"
PREFIXES = ("CNU_DIAG_V1 ", "CNU_CAPABILITY_V1 ", "CNU_APP_ADAPTER_V1 ")
OMIT = {"response", "query", "prompt", "content", "message", "headers", "authorization",
        "password", "api_key", "service_key", "error", "user_prompt", "raw_text"}


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items() if k.lower() not in OMIT}
    if isinstance(value, list):
        return [safe(v) for v in value]
    return value


def request(row, arm, round_id):
    response = row.get("response")
    laws = response.get("laws") if isinstance(response, dict) else None
    ids = sorted({str(x.get("law_id") or x.get("item_id") or "").strip()
                  for x in laws or [] if isinstance(x, dict) and (x.get("law_id") or x.get("item_id"))})
    out = {k: row.get(k) for k in ("question_id", "case_id", "attempt_id", "status", "status_code",
                                   "duration_ms", "first_event_ms", "ttft_ms", "started_at", "completed_at",
                                   "law_count", "comment_chars", "response_sha256", "error_type")}
    out.update(experiment_id=EXPERIMENT, arm=arm, round=round_id,
               format_valid=row.get("status") == "ok" and isinstance(laws, list)
               and isinstance(response.get("comment"), str),
               law_ids=ids, law_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
               search_source=response.get("search_source") if isinstance(response, dict) else None,
               hitl=bool(response.get("hitl")) if isinstance(response, dict) else False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out
    (out / "raw" / "metrics").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.results / "protocol.json", out / "protocol.json")
    shutil.copyfile(args.results / "status.json", out / "status.json")
    counts = {"requests": 0, "events": 0, "log_parse_failures": 0}
    with (out / "raw" / "requests.jsonl").open("w") as dest:
        for round_id in (1, 2):
            for arm in ("original", "combined"):
                cell = args.results / "level-100" / f"round-{round_id}"
                path = cell / "responses" / "sweep-trial" / arm / "client_requests.jsonl"
                if not path.exists():
                    continue
                for line in path.open():
                    if line.strip():
                        dest.write(json.dumps(request(json.loads(line), arm, round_id), ensure_ascii=False) + "\n")
                        counts["requests"] += 1
                metrics = cell / f"{arm}-metrics.jsonl"
                if metrics.exists():
                    shutil.copyfile(metrics, out / "raw" / "metrics" / f"r{round_id}-{arm}.jsonl")
    with (out / "raw" / "events.jsonl").open("w") as dest:
        for arm in ("original", "combined"):
            for line in (args.logs / f"{arm}.timed.log").open(errors="replace"):
                prefix = next((prefix for prefix in PREFIXES if prefix in line), None)
                if prefix is None:
                    continue
                try:
                    # The backend's console logger occasionally writes its own
                    # text after one complete JSON event on the same CRI line.
                    # Decode the first object, rather than discarding it.
                    event, _end = json.JSONDecoder().raw_decode(line.split(prefix, 1)[1].lstrip())
                    event = safe(event)
                    event["timestamp"] = line.split(" ", 1)[0]
                    event["arm"] = arm
                    dest.write(json.dumps(event, ensure_ascii=False) + "\n")
                    counts["events"] += 1
                except (json.JSONDecodeError, IndexError):
                    counts["log_parse_failures"] += 1
    (out / "materialization.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(counts)


if __name__ == "__main__":
    main()
