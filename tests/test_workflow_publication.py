import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_moleg_workflow300 import public_safe,read_rows,validate_requests
from evaluate_moleg_isolated import trial_schedule


def test_publication_rejects_raw_text_and_credentials():
    public_safe(dict(question_sha256='a'*64,answer_chars=123,ok=False))
    for value in [dict(question='private'),dict(nested=[dict(reason='private')]),
                  dict(value='ghp_'+'a'*36),dict(authorization='redacted')]:
        with pytest.raises(ValueError):public_safe(value)


def test_publication_refuses_partial_tail(tmp_path):
    p=tmp_path/'records.jsonl';p.write_text(json.dumps(dict(ok=True)))
    with pytest.raises(ValueError):read_rows(p)


def test_publication_validates_all_conditions_and_keeps_failures():
    schedule=[dict(repeat=r,users=u,policy=p) for r,u,p in
              trial_schedule(['baseline','fixed16','budget'],[16,32,8,4,2,1],3,1)]
    plan=dict(unique_questions=300,planned_requests=10800,schedule=schedule)
    rows=[dict(trial=t,index=i,case_id=f'q{i}',question_sha256=f'hash{i}',ok=i!=0,**condition)
          for t,condition in enumerate(schedule) for i in range(300)]
    summary=dict(complete=True,trials=[dict(trial=t,n=300,success=299,**c) for t,c in enumerate(schedule)])
    validate_requests(rows,plan,summary)
    with pytest.raises(ValueError):validate_requests(rows[:-1],plan,summary)
    rows[-1]=rows[-2].copy()
    with pytest.raises(ValueError):validate_requests(rows,plan,summary)
