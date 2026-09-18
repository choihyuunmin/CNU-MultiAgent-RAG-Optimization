from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from supervise_moleg_sweep import load_level_pools, sweep_schedule
from export_reference_scaling import validate


def test_two_orders_at_each_load_before_escalating():
    assert sweep_schedule([1, 2], 2, repeat_within_level=True) == [(0,0,1),(1,0,1),(0,1,2),(1,1,2)]
    assert sweep_schedule([1, 2], 2) == [(0,0,1),(0,1,2),(1,0,1),(1,1,2)]


def test_load_pools_require_real_unique_requests_for_concurrency():
    cases=[{'case_id':str(i)} for i in range(3)]
    pools={'1':['2','0'],'3':['0','1','2']}
    selected=load_level_pools(cases,pools,[1,3])
    assert [c['case_id'] for c in selected[1]] == ['2','0']
    for invalid in [{'1':['0']}, {'1':['0'],'3':['0','0','1']},
                    {'1':['0'],'3':['0','1','missing']}, {'1':['0'],'3':['0','1']}]:
        with pytest.raises(ValueError):load_level_pools(cases,invalid,[1,3])


def make_audit_fixture(root):
    (root/'main').mkdir()
    (root/'pod_source/src').mkdir(parents=True)
    cases=[{'case_id':str(i),'question_sha256':str(i)} for i in range(2)]
    plan={'planned_requests':4,'arms':[{'name':'baseline'},{'name':'improved'}],
          'levels':[2],'repeats':1,'expected_by_level':{'2':cases},'code_sha256':{}}
    rows=[dict(c,arm=arm,level=2,repeat=0,pipeline_ok=True) for arm in ['baseline','improved'] for c in cases]
    trials=[{'arm':arm,'level':2,'repeat':0,'pipeline_success':2,'max_outstanding':2} for arm in ['baseline','improved']]
    (root/'plan.json').write_text(json.dumps(plan))
    (root/'live-source-manifest.json').write_text('{}')
    (root/'main/campaign-status.json').write_text('{"status":"completed"}')
    (root/'main/summary.json').write_text(json.dumps({'complete':True,'trials':trials}))
    (root/'main/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows,trials


def test_audit_rejects_duplicate_pair_even_when_total_count_matches(tmp_path):
    rows,_=make_audit_fixture(tmp_path)
    assert len(validate(tmp_path)[1])==4
    rows[1]=rows[0]
    (tmp_path/'main/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='membership'):validate(tmp_path)


def test_audit_rejects_unreached_peak_concurrency(tmp_path):
    _,trials=make_audit_fixture(tmp_path)
    trials[0]['max_outstanding']=1
    (tmp_path/'main/summary.json').write_text(json.dumps({'complete':True,'trials':trials}))
    with pytest.raises(ValueError,match='concurrency'):validate(tmp_path)
