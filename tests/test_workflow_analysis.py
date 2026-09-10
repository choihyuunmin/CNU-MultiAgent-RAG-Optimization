from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_moleg_workflow import paired_comparisons, workflow_summary


def fixtures():
    return [{'users':16, 'policy':policy, 'case_id':f'q{i}', 'repeat':repeat,
             'elapsed_s':seconds, 'question_sha256':f'hash-{i}', 'ok':True,
             'evidence_ids_sha256':['id'], 'source_law_hit':True, 'source_chunk_hit':None}
            for policy, seconds in [('baseline',10),('fixed16',5),('budget',6)]
            for i in range(4) for repeat in range(2)]


def test_additional_adapter_effect_can_be_negative_and_repeats_are_clustered():
    rows = paired_comparisons(fixtures())
    incremental = next(r for r in rows if r['control']=='fixed16')
    assert incremental['paired_requests']==8
    # A slower adapter must not inherit the baseline-to-fixed16 gain.
    assert incremental['latency']['n']==4
    assert incremental['latency']['reduction_pct']==pytest.approx(-20)
    assert incremental['latency']['delta']==1
    assert incremental['source_chunk_hit_change']['n']==0


def test_missing_pairs_and_duplicate_rows_are_rejected():
    rows = fixtures()
    with pytest.raises(ValueError):
        paired_comparisons(rows[:-1])
    with pytest.raises(ValueError):
        paired_comparisons(rows+[rows[0]])


def test_private_join_exports_only_numeric_aggregates(tmp_path):
    data = tmp_path/'validation'
    private = tmp_path/'main-apps'
    data.mkdir(); private.mkdir()
    trials = []
    for i,policy in enumerate(['baseline','fixed16','budget']):
        trials.append(dict(trial=i,policy=policy,users=16,repeat=0,n=1))
        trace = dict(trace_id=policy, diagnosis=dict(dropped_spans=0,
                     before_first_instrumented_stage_s=2), spans=[
            dict(kind='model_rpc_lifetime',stage='preparation',model='orchestrator',start_s=2,end_s=4),
            dict(kind='usage',stage='preparation',model='orchestrator',start_s=2,end_s=4,
                 input_tokens=40,output_tokens=None,input_estimate_error=0)])
        (private/(policy+'.workflow.jsonl')).write_text(json.dumps(trace)+'\n')
        (private/(policy+'.trace.jsonl')).write_text(json.dumps(dict(session_id='scale-secret-0',
            workflow_trace_id=policy,errors=[],private_answer='must not be exported'))+'\n')
    (data/'summary.json').write_text(json.dumps(dict(trials=trials)))
    (data/'app-state.jsonl').write_text('')
    result = workflow_summary(data,private)
    assert len(result['trials'])==3
    assert result['trials'][0]['stage_model_work'][0]['model_work_per_api_s']['mean']==2
    assert result['trials'][0]['model_usage'][0]['reasoning_tokens']['n']==0
    assert 'must not be exported' not in json.dumps(result)
    assert 'scale-secret' not in json.dumps(result)
    path = private/'budget.workflow.jsonl'
    path.write_text(path.read_text()*2)
    with pytest.raises(ValueError, match='duplicate workflow'):
        workflow_summary(data,private)
