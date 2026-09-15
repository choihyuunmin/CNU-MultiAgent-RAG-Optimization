"""Audit a completed300 campaign and export numeric/hash-only publication files.

No model calls, service changes, raw questions, answers, judge reasons or credentials.
Original evidence is read-only; the destination must not already exist.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

REQUEST_KEYS=set('case_id kind index question_sha256 ok scheduling_lag_s admission_wait_s gate_limit_at_start status elapsed_s wire_s ttft_s first_event_s streamed_chars answer_sha256 evidence_ids_sha256 answer_chars hitl source_law_hit source_chunk_hit pipeline_ok trial policy repeat users arrival_rate error_type'.split())
AUDIT_KEYS=set('trial policy users repeat index case_id execute_returned internal_error_count internal_error_types'.split())
JUDGE_KEYS=set('case_id variant judge_model label_type ok relevance support insufficient_evidence elapsed_s input_tokens evidence_truncated error_type'.split())
PRIVATE_KEYS=set('question prompt messages response streamed_answer reference_answer evidence paragraph_content reason password api_key authorization access_token service_key'.split())
SECRET=re.compile(r'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')


def public_safe(value):
    if isinstance(value,dict):
        if any(k.lower() in PRIVATE_KEYS for k in value):
            raise ValueError('private field in proposed public artifact')
        for item in value.values():public_safe(item)
    elif isinstance(value,list):
        for item in value:public_safe(item)
    elif isinstance(value,str) and SECRET.search(value):
        raise ValueError('credential pattern in proposed public artifact')


def file_hash(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def read_rows(path):
    with path.open() as source:
        rows=[]
        for line in source:
            if not line.endswith('\n'):raise ValueError('incomplete JSONL tail')
            rows.append(json.loads(line))
        return rows


def validate_requests(rows,plan,summary):
    schedule=plan['schedule']
    if plan['unique_questions']!=300 or plan['planned_requests']!=10800 or len(schedule)!=36:
        raise ValueError('not the expected300 protocol')
    if not summary['complete'] or len(summary['trials'])!=36 or len(rows)!=10800:
        raise ValueError('incomplete campaign')
    question_set=None
    for trial_id,condition in enumerate(schedule):
        selected=[r for r in rows if r['trial']==trial_id]
        identities={(r['case_id'],r['question_sha256']) for r in selected}
        if len(selected)!=300 or len(identities)!=300 or {r['index'] for r in selected}!=set(range(300)):
            raise ValueError('missing/duplicate request in trial')
        if question_set is None:question_set=identities
        if identities!=question_set:raise ValueError('different question membership across conditions')
        entry=summary['trials'][trial_id]
        if entry['trial']!=trial_id or entry['n']!=300 or entry['success']!=sum(bool(r['ok']) for r in selected):
            raise ValueError('summary/request mismatch')
        for k,v in condition.items():
            if entry[k]!=v or any(r[k]!=v for r in selected):raise ValueError('condition mismatch')


def export(root,destination):
    status=json.loads((root/'campaign-status.json').read_text())
    plan=json.loads((root/'frozen-plan.json').read_text())
    summary=json.loads((root/'validation/summary.json').read_text())
    if status['status']!='completed_evaluation_not_released' or status['release_allowed']:
        raise ValueError('unexpected terminal or release status')
    if (root/'operator.env').exists():raise ValueError('temporary credential cleanup incomplete')
    for name,digest in plan['sha256'].items():
        if file_hash(root/name)!=digest:raise ValueError('frozen input changed')
    rows=read_rows(root/'validation/requests.jsonl')
    validate_requests(rows,plan,summary)
    if status['completed_requests']!=len(rows) or status['sse_success']!=sum(bool(r['ok']) for r in rows):
        raise ValueError('status/request mismatch')
    audit=read_rows(root/'validation/pipeline-audit.jsonl')
    keyed={(r['trial'],r['index']):r for r in rows}
    if len(audit)!=len(rows) or len({(r['trial'],r['index']) for r in audit})!=len(rows):
        raise ValueError('incomplete or duplicate pipeline health audit')
    for r in audit:
        expected=keyed[(r['trial'],r['index'])]
        if any(r[k]!=expected[k] for k in ['case_id','policy','users','repeat']):
            raise ValueError('pipeline health join mismatch')
    judge_rows=[];judge_summaries={}
    cases=json.loads((root/'cases.json').read_text())
    qa={c['case_id'] for c in cases if c.get('reference_answer')}
    if len(qa)!=plan['reference_answer_questions']:raise ValueError('QA membership count mismatch')
    for users in [1,2,4,8,16,32]:
        data=read_rows(root/f'judge-u{users}/answer_judgments.jsonl')
        expected={(case_id,arm) for case_id in qa for arm in ['baseline','fixed16','budget']}
        if len(data)!=len(expected) or {(r['case_id'],r['variant']) for r in data}!=expected:
            raise ValueError('missing/duplicate judge record; failures must be retained')
        js=json.loads((root/f'judge-u{users}/answer_judge_summary.json').read_text())
        if js['repeat']!=1 or js['completed']!=len(data) or js['expected']!=len(expected):
            raise ValueError('judge summary mismatch')
        if sum(v['valid'] for v in js['variants'].values())!=sum(bool(r['ok']) for r in data):
            raise ValueError('judge validity mismatch')
        judge_summaries[f'judge-u{users}-summary.json']=js
        judge_rows.extend({k:v for k,v in r.items() if k in JUDGE_KEYS}|dict(users=users,repeat=1) for r in data)
    artifacts={
        'campaign-status.json':status,'summary.json':summary,'frozen-plan.json':plan,
        **judge_summaries,
    }
    for name in ['validation/manifest.json','validation/workflow-analysis.json','validation/resources.json',
                 'recovery-lineage.json','data-health-preflight.json','supervised-data-preflight.json',
                 'preflight.json','observe-final.json','budget-final.json','run-spec.json']:
        artifacts[Path(name).name]=json.loads((root/name).read_text())
    row_artifacts={
        'requests.jsonl':[{k:v for k,v in r.items() if k in REQUEST_KEYS} for r in rows],
        'pipeline-audit.jsonl':[{k:v for k,v in r.items() if k in AUDIT_KEYS} for r in audit],
        'judge-scores.jsonl':judge_rows,
    }
    for data in [*artifacts.values(),*row_artifacts.values()]:public_safe(data)
    destination.mkdir(parents=True,exist_ok=False)
    for name,data in artifacts.items():(destination/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for name,data in row_artifacts.items():
        with (destination/name).open('x') as sink:
            for row in data:sink.write(json.dumps(row,ensure_ascii=False)+'\n')
    provenance=dict(exported_utc=datetime.now(timezone.utc).isoformat(),
        scope='numeric/hash-only export; original private evidence unchanged',
        completed_requests=len(rows),sse_success=sum(bool(r['ok']) for r in rows),
        pipeline_ok=sum(bool(r.get('pipeline_ok')) for r in rows),
        traced_internal_error_requests=sum(r['internal_error_count']>0 for r in audit),
        judge_records=len(judge_rows),judge_valid=sum(bool(r['ok']) for r in judge_rows),
        judge_error_types=dict(Counter(r.get('error_type','unknown') for r in judge_rows if not r['ok'])),
        frozen_inputs_verified=True,temporary_environment_removed=True,release_allowed=False,
        omitted=['raw questions','raw answers','private traces','judge reasons','credentials','runtime/source archives'],
        source_sha256={name:file_hash(root/name) for name in ['validation/requests.jsonl','validation/pipeline-audit.jsonl',
             'validation/workflow-analysis.json','validation/resources.json','campaign-status.json']},
        exported_sha256={p.name:file_hash(p) for p in sorted(destination.iterdir())})
    (destination/'publication-audit.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=export(args.directory.resolve(),args.output.resolve())
    print(json.dumps({k:result[k] for k in ['completed_requests','sse_success','pipeline_ok',
        'traced_internal_error_requests','judge_records','judge_valid','judge_error_types']}))


if __name__=='__main__':main()
