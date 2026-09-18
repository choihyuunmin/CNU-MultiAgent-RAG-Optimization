import asyncio
import io
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_moleg_scaling as scaling
from prepare_moleg_reasoning_experiment import prepare
from supervise_moleg_reasoning import campaign_spec
from export_moleg_reasoning import validate_requests
from evaluate_moleg_isolated import trial_schedule
from watch_moleg_reasoning import verify_publication


def test_baseline_trial_has_no_admission_gate(monkeypatch):
    async def run():
        def forbidden(*args, **kwargs):
            raise AssertionError("normal trial must never instantiate a call limiter")
        monkeypatch.setattr(scaling, "PressureGate", forbidden)
        original_client = httpx.AsyncClient
        async def respond(request):
            return httpx.Response(200, text='data: {"stage":"token","delta":"answer"}\n\n'
                'data: {"stage":"done","result":{"comment":"answer","laws":[]}}\n\n')
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
            transport=httpx.MockTransport(respond), **kwargs))
        args = SimpleNamespace(metrics=[], api_key_env=None, timeout=1, base="http://fixture",
                               sample_interval=.1, initial_limit=1, slo=30)
        sink, telemetry = io.StringIO(), io.StringIO()
        cases = [dict(case_id=str(i), kind="fixture", question=f"private-{i}") for i in range(4)]
        result = await scaling.trial(args, cases, 4, None, "baseline", 0, 0, sink, telemetry)
        assert result["success"] == 4 and result["final_gate_limit"] is None
        assert all(json.loads(line)["admission_wait_s"] == 0 for line in sink.getvalue().splitlines())
        assert "private-" not in sink.getvalue()
    asyncio.run(run())


def test_freeze_reuses_question_hashes_and_keeps_smoke_disjoint(tmp_path):
    cases = [dict(case_id=str(i), kind="fixture", question=f"private-{i}") for i in range(308)]
    previous = [dict(case_id=c["case_id"], kind=c["kind"], question_sha256=scaling.digest(c["question"]))
                for c in cases[:300]]
    case_file, manifest = tmp_path / "input.json", tmp_path / "old.json"
    case_file.write_text(json.dumps(cases))
    manifest.write_text(json.dumps(dict(selected_cases=previous)))
    target = tmp_path / "new"
    result = prepare(target, case_file, manifest, tmp_path)
    assert result["planned_primary_requests"] == 7200
    plan = json.loads((target / "plan.json").read_text())
    assert len(plan["schedule"]) == 24
    assert {r["case_id"] for r in plan["smoke_cases"]}.isdisjoint(c["case_id"] for c in previous)
    assert "private-" not in json.dumps(plan)
    assert (target / "cases.json").stat().st_mode & 0o777 == 0o600
    previous[0]["question_sha256"] = "mismatch"
    manifest.write_text(json.dumps(dict(selected_cases=previous)))
    with pytest.raises(ValueError, match="identity"):
        prepare(tmp_path / "bad", case_file, manifest, tmp_path)
    assert not (tmp_path / "bad").exists()


def test_smoke_fallback_blocks_primary_even_when_sse_completed(tmp_path):
    plan = dict(unique_questions=300, users=[1,2,4,8,16,32], repeats=2,
                variants=['baseline','reasoning'], planned_primary_requests=7200)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    path = tmp_path/'smoke-verified'
    path.mkdir()
    trials = [dict(n=8,success=8,pipeline_success=8,traced_internal_error_requests=0) for _ in range(2)]
    (path/'summary.json').write_text(json.dumps(dict(complete=True,trials=trials)))
    assert campaign_spec(tmp_path)['limit']==300
    trials[1]['traced_internal_error_requests']=8
    (path/'summary.json').write_text(json.dumps(dict(complete=True,trials=trials)))
    with pytest.raises(ValueError,match='smoke'):
        campaign_spec(tmp_path)


def test_primary_export_requires_every_question_and_planned_condition():
    schedule=[dict(repeat=r,users=u,policy=p) for r,u,p in
              trial_schedule(['baseline','reasoning'],[1,2,4,8,16,32],2)]
    plan=dict(schedule=schedule,unique_questions=300,planned_primary_requests=7200)
    trials=[dict(trial=i,n=300,success=300,**condition) for i,condition in enumerate(schedule)]
    rows=[dict(trial=i,index=q,case_id=str(q),question_sha256=str(q),ok=True,**condition)
          for i,condition in enumerate(schedule) for q in range(300)]
    summary=dict(complete=True,trials=trials)
    validate_requests(rows,plan,summary)
    rows[-1]['question_sha256']='different-question'
    with pytest.raises(ValueError,match='membership'):
        validate_requests(rows,plan,summary)


def test_collector_verifies_hashes_and_rejects_private_fields(tmp_path):
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(dict(completed_requests=7200)))
    audit=tmp_path/'publication-audit.json'
    def sign():
        audit.write_text(json.dumps(dict(exported_sha256={'summary.json':hashlib.sha256(path.read_bytes()).hexdigest()})))
    sign()
    verify_publication(tmp_path)
    path.write_text(json.dumps(dict(completed_requests=1)))
    with pytest.raises(ValueError,match='checksum'):
        verify_publication(tmp_path)
    path.write_text(json.dumps(dict(api_key='never-export')))
    sign()
    with pytest.raises(ValueError,match='private'):
        verify_publication(tmp_path)
