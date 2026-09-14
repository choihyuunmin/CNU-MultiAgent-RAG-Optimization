import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_program_harness import harness_detail


def test_harness_events_are_time_scoped_and_aggregated():
    records = [{"started_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:10+00:00"}]
    row = {"event": "workflow_closed", "cancelled": 0, "nodes": [{
        "name": "ontology_search", "status": "ok", "run_ms": 5,
        "join_wait_ms": 0.5, "overlap_before_join_ms": 4.5}]}
    text = "\n".join([
        "2025-01-01T00:00:00+00:00 stdout F CNU_PROGRAM_HARNESS_V1 " + json.dumps(row),
        "2026-01-01T00:00:05+00:00 stdout F CNU_PROGRAM_HARNESS_V1 " + json.dumps(row),
    ])
    result = harness_detail(text, records)
    assert result["workflows"] == 1
    assert result["ontology_success"] == 1
    assert result["ontology_run_mean_ms"] == 5
    assert result["ontology_ready_before_join"] == 1
