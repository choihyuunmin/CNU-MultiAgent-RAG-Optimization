"""Prepare an immutable 300-question recovery segment without editing parent evidence.

Run only on the operator GPU host. Copy credentials only into the new protected
run directory; never print credentials, questions, answers or private traces.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from evaluate_moleg_isolated import trial_schedule
from evaluate_moleg_scaling import select_cases, digest
from supervise_moleg_workflow400 import model_inventory, USERS


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def audit_parent(root):
    cases=json.loads((root/'cases.json').read_text())
    expected={c['case_id']:digest(c['question']) for c in cases}
    if len(cases)!=400 or len(expected)!=400:
        raise ValueError('parent is not the frozen 400-question dataset')
    plan=json.loads((root/'frozen-plan.json').read_text())
    for name,expected_hash in plan['sha256'].items():
        if sha256(root/name)!=expected_hash:
            raise ValueError('parent frozen input changed: '+name)
    summary=json.loads((root/'validation/summary.json').read_text())
    schedule=list(trial_schedule(['baseline','fixed16','budget'],USERS,3))
    # The known reboot interrupted the third arm at U1, repeat index1.
    # Require exactly this checkpoint; never silently skip a different prefix.
    if summary['complete'] or len(summary['trials'])!=20:
        raise ValueError('unexpected recovery checkpoint; requires operator review')
    groups=defaultdict(list)
    partial_tail=False
    with (root/'validation/requests.jsonl').open() as source:
        for line in source:
            if not line.endswith('\n'):
                partial_tail=True
                break
            row=json.loads(line)
            if not 0<=row['trial']<=20:
                raise ValueError('unexpected parent trial')
            if (row['repeat'],row['users'],row['policy'])!=schedule[row['trial']]:
                raise ValueError('parent request schedule differs')
            if expected.get(row['case_id'])!=row['question_sha256']:
                raise ValueError('parent request question identity differs')
            groups[row['trial']].append(row)
    for i,entry in enumerate(summary['trials']):
        rows=groups[i]
        if (entry['repeat'],entry['users'],entry['policy'])!=schedule[i] or entry['n']!=400:
            raise ValueError('parent summary schedule differs')
        if len(rows)!=400 or {r['case_id'] for r in rows}!=set(expected) or {r['index'] for r in rows}!=set(range(400)):
            raise ValueError('parent completed trial is incomplete or duplicated')
    return cases,dict(
        parent_status='interrupted_by_host_reboot_not_completed',
        completed_400_question_trials=20,
        full_first_repeat_trials=18,
        preserved_extra_400_question_trials=[18,19],
        interrupted_trial=20,
        interrupted_trial_recorded_requests=len(groups[20]),
        parent_recorded_requests=sum(map(len,groups.values())),
        parent_sse_success=sum(bool(r['ok']) for rows in groups.values() for r in rows),
        parent_partial_jsonl_tail=partial_tail,
        parent_evidence_sha256={name:sha256(root/name) for name in [
            'frozen-plan.json','campaign-status.json','validation/requests.jsonl','validation/summary.json']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    os.umask(0o077)
    parent,root=args.parent.resolve(),args.directory.resolve()
    if parent==root or root.parent!=parent.parent or not root.name.startswith('workflow300-'):
        parser.error('a new sibling workflow300 directory is required')
    if (root/'recovery-lineage.json').exists() or (root/'operator.env').exists():
        parser.error('do not overwrite a prepared recovery segment')
    cases,audit=audit_parent(parent)
    if sha256(root/'scripts/evaluate_moleg_scaling.py')!=sha256(parent/'scripts/evaluate_moleg_scaling.py'):
        raise ValueError('request primitive must not change across recovery')
    for name in ['moleg_workflow_adapter.py','launch_moleg_scaling_app.py','moleg_adapter_serving.py']:
        if sha256(root/'scripts'/name)!=sha256(parent/'scripts'/name):
            raise ValueError('adapter or app behavior changed during recovery')
    for name in ['source.tar.gz','observe-final.json','budget-final.json','WORKFLOW400_PROTOCOL_20260908.md']:
        shutil.copy2(parent/name,root/name)
    shutil.copytree(parent/'pod_source',root/'pod_source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    expected_source=json.loads((parent/'preflight.json').read_text())['source_files']
    actual_source={str(p.relative_to(root/'pod_source')):sha256(p) for p in (root/'pod_source/src').rglob('*.py')}
    if actual_source!=expected_source:
        raise ValueError('frozen application source changed')
    selected=select_cases(cases,300,20260909)
    if len(selected)!=300 or len({c['case_id'] for c in selected})!=300:
        raise ValueError('300 unique regression questions required')
    (root/'cases.json').write_text(json.dumps(selected,ensure_ascii=False)+'\n')
    (root/'run-spec.json').write_text(json.dumps(dict(limit=300,start_repeat=1,repeats=3,seed=20260909),indent=2)+'\n')
    old={m['port']:m['argv_sha256'] for m in json.loads((parent/'model-inventory-before.json').read_text())}
    models=model_inventory()
    audit.update(utc=datetime.now(timezone.utc).isoformat(),parent_run=parent.name,
        scope='separate post-reboot 300-question segment; never pool with pre-reboot latency',
        selection='proportional by question kind, seed20260909; independent of observed latency and answers',
        selected_case_kinds=dict(Counter(c['kind'] for c in selected)),
        selected_cases=[dict(case_id=c['case_id'],kind=c['kind'],question_sha256=digest(c['question'])) for c in selected],
        reference_answer_questions=sum(bool(c.get('reference_answer')) for c in selected),
        source_label_questions=sum(bool(c.get('source_law_id')) for c in selected),
        remaining_repeats=[1,2],planned_requests=10800,
        serving_command_unchanged={str(m['port']):old.get(m['port'])==m['argv_sha256'] for m in models},
        parent_data_preflight=json.loads((parent/'data-health-preflight.json').read_text()),
        restart_rule='rerun the entire interrupted A/B/C block at300, not only the unfinished arm',
        discarded_failures=0,release_allowed=False)
    (root/'recovery-lineage.json').write_text(json.dumps(audit,indent=2)+'\n')
    shutil.copy2(parent/'operator.env',root/'operator.env')
    (root/'operator.env').chmod(0o600)
    print(json.dumps({k:audit[k] for k in ['parent_recorded_requests','completed_400_question_trials',
        'interrupted_trial_recorded_requests','selected_case_kinds','reference_answer_questions','planned_requests',
        'serving_command_unchanged']}))


if __name__=='__main__':main()
