import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_moleg_isolated import audit_pipeline_trace, trial_schedule
from supervise_moleg_workflow400 import Progress, infrastructure_faults, campaign_spec
from evaluate_moleg_scaling import select_cases
from evaluate_moleg_answer_judge import select_judge_records


def test_full400_schedule_balances_each_arm_position():
    users=[16,32,8,4,2,1]
    schedule=list(trial_schedule(['baseline','fixed16','budget'],users,3))
    assert len(schedule)==54
    assert len(set(schedule))==54
    for u in users:
        orders=[[name for repeat,load,name in schedule if repeat==r and load==u] for r in range(3)]
        for position in range(3):
            assert {o[position] for o in orders}=={'baseline','fixed16','budget'}


def test_recovery300_restarts_whole_block_and_preserves_remaining_order(tmp_path):
    users=[16,32,8,4,2,1]
    variants=['baseline','fixed16','budget']
    full=list(trial_schedule(variants,users,3))
    recovery=list(trial_schedule(variants,users,3,start_repeat=1))
    assert recovery==full[18:]
    assert recovery[:3]==[(1,1,name) for name in variants]
    assert len(recovery)==36 and len(set(recovery))==36
    assert campaign_spec(tmp_path)['limit']==400
    spec=dict(limit=300,start_repeat=1,repeats=3,seed=20260909)
    (tmp_path/'run-spec.json').write_text(json.dumps(spec))
    assert campaign_spec(tmp_path)==spec
    assert len(recovery)*spec['limit']==10800
    spec['start_repeat']=2
    (tmp_path/'run-spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError):campaign_spec(tmp_path)


def test_recovery300_membership_is_fixed_and_not_latency_selected():
    cases=[dict(case_id=f'q{i}',kind='qa' if i<160 else 'source',question=f'question{i}') for i in range(400)]
    selected=select_cases(cases,300,20260909)
    assert len(selected)==len({c['case_id'] for c in selected})==300
    assert selected==select_cases(cases,300,20260909)
    assert sum(c['kind']=='qa' for c in selected)==120


def test_judge_recovery_repeat_is_not_silently_empty():
    cases={'q0':dict(reference_answer='reference'),'q1':dict(reference_answer='')}
    records=[dict(case_id=case_id,variant=arm,repeat=1,result={})
             for case_id in cases for arm in ['baseline','fixed16','budget']]
    selected=select_judge_records(records,cases,repeat=1)
    assert len(selected)==3
    assert all(r['case_id']=='q0' and r['repeat']==1 for r in selected)
    with pytest.raises(ValueError):select_judge_records(records,cases)
    with pytest.raises(ValueError):select_judge_records(records+[records[0]],cases,repeat=1)
    with pytest.raises(ValueError):select_judge_records(records[1:],cases,repeat=1)


def test_judge_preparation_and_selection_preserve_original_repeat(tmp_path):
    import subprocess
    cases=[dict(case_id='q0',reference_answer='reference'),dict(case_id='q1',reference_answer='')]
    (tmp_path/'cases.json').write_text(json.dumps(cases))
    records=[dict(case_id=c['case_id'],policy=arm,repeat=repeat,users=1,result={})
             for repeat in [1,2] for c in cases for arm in ['baseline','fixed16','budget']]
    (tmp_path/'validation.responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    script=Path(__file__).resolve().parents[1]/'scripts/prepare_moleg_workflow_judge.py'
    subprocess.run([sys.executable,str(script),'--directory',str(tmp_path),
                    '--cases',str(tmp_path/'cases.json'),'--repeat','1'],check=True,capture_output=True)
    prepared=[json.loads(line) for line in (tmp_path/'judge-u1/e2e_stream.jsonl').read_text().splitlines()]
    assert len(prepared)==6 and {r['repeat'] for r in prepared}=={1}
    assert len(select_judge_records(prepared,{c['case_id']:c for c in cases},repeat=1))==3


def test_fallback_and_no_return_remain_in_audit(tmp_path):
    path=tmp_path/'trace.jsonl'
    rows=[dict(session_id='scale-test-0',response_chars=10,errors=[dict(error='BadRequestError')]),
          dict(session_id='scale-test-1')]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=audit_pipeline_trace(path,[dict(case_id='q0'),dict(case_id='q1')],0,'baseline',16,0)
    assert result[0]['execute_returned'] and result[0]['internal_error_count']==1
    assert not result[1]['execute_returned']
    assert 'session_id' not in result[0]
    with pytest.raises(ValueError):
        audit_pipeline_trace(path,[dict(case_id='q0')],0,'baseline',16,0)


def test_progress_keeps_partial_line_and_counts_failures(tmp_path):
    path=tmp_path/'requests.jsonl'
    row=dict(ok=False,trial=0,policy='budget',users=16,repeat=0)
    data=json.dumps(row)
    path.write_text(data[:8])
    progress=Progress(path);progress.read()
    assert progress.completed==0
    path.write_text(data+'\n')
    progress.read();progress.read()
    assert progress.completed==1 and progress.sse_ok==0


def test_infrastructure_faults_are_not_request_failure_rates():
    host=dict(vm_counters=dict(oom_kill=4),host_memory_kib=dict(MemAvailable=9*1024**2),
              gpus=[{'temperature.gpu':'83'}])
    assert infrastructure_faults(host,4,20*1024**3)==[]
    assert 'host_oom_kill_increased' in infrastructure_faults(host,3,20*1024**3)
    host['gpus'][0]['temperature.gpu']='90'
    assert 'gpu_temperature_at_least_90c' in infrastructure_faults(host,4,20*1024**3)
