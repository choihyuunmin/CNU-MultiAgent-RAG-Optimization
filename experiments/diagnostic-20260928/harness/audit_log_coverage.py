"""Check whether archived application logs cover the C100 response windows."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--level100", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = {}
    for arm in ("original", "combined"):
        path = args.logs / f"{arm}-application.log"
        digest = hashlib.sha256()
        first = last = None
        diagnostic_lines = 0
        with path.open("rb") as source:
            for line in source:
                digest.update(line)
                if line.startswith(b"2026-"):
                    ts = line.split(b" ", 1)[0].decode("ascii", "ignore")
                    try:
                        t = timestamp(ts)
                    except ValueError:
                        continue
                    first = t if first is None else min(first, t)
                    last = t if last is None else max(last, t)
                diagnostic_lines += b'CNU_APP_ADAPTER_V1 {"event": "request_finished"' in line
        windows = []
        for round_id in (1, 2, 3, 4):
            file = (args.level100 / f"round-{round_id}" / "responses" / "sweep-trial" /
                    arm / "client_requests.jsonl")
            rows = [json.loads(line) for line in file.open() if line.strip()]
            start = min(timestamp(row["started_at"]) for row in rows)
            end = max(timestamp(row["completed_at"]) for row in rows)
            windows.append({"round": round_id, "requests": len(rows),
                            "first_client_start": min(row["started_at"] for row in rows),
                            "last_client_end": max(row["completed_at"] for row in rows),
                            "overlaps_archived_log_time": first <= end and start <= last})
        result[arm] = {"archived_log_sha256": digest.hexdigest(),
                       "archived_log_first_epoch_s": first, "archived_log_last_epoch_s": last,
                       "request_finished_events_total": diagnostic_lines,
                       "c100_windows": windows}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print({arm: [w["overlaps_archived_log_time"] for w in item["c100_windows"]]
           for arm, item in result.items()})


if __name__ == "__main__":
    main()
