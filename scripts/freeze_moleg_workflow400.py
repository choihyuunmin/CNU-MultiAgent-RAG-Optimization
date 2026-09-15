"""Freeze this regression campaign's inputs and code before inference starts."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from evaluate_moleg_isolated import trial_schedule
from supervise_moleg_workflow400 import model_inventory,USERS,campaign_spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args();root=args.directory
    spec=campaign_spec(root)
    cases=json.loads((root/'cases.json').read_text())
    if any(n!=spec['limit'] for n in [len(cases),len({c['question'] for c in cases}),len({c['case_id'] for c in cases})]):
        raise ValueError('exact protocol question count required')
    candidate=json.loads((root/'budget-final.json').read_text())
    if candidate['budgets']['orchestrator']!={'max_calls':8,'max_prefill_tokens':46232,'max_kv_tokens':50328}:
        raise ValueError('previously frozen budget changed')
    models=model_inventory()
    files=['runtime.tar.gz','source.tar.gz','cases.json','observe-final.json','budget-final.json',
           'WORKFLOW400_PROTOCOL_20260908.md']
    if (root/'run-spec.json').exists():
        files+=['run-spec.json','WORKFLOW300_RECOVERY_PROTOCOL_20260909.md','recovery-lineage.json']
    data={'utc':datetime.now(timezone.utc).isoformat(),
          'planned_requests':spec['limit']*len(USERS)*3*(spec['repeats']-spec['start_repeat']),
          'unique_questions':spec['limit'],
          'source_label_questions':sum(bool(c.get('source_law_id')) for c in cases),
          'reference_answer_questions':sum(bool(c.get('reference_answer')) for c in cases),
          'scope':'regression dataset, not independent holdout','seed':spec['seed'],
          'users':USERS,'repeats':spec['repeats']-spec['start_repeat'],'start_repeat':spec['start_repeat'],
          'existing_model_processes':len(models),
          'schedule':[dict(repeat=r,users=u,policy=name) for r,u,name in
                      trial_schedule(['baseline','fixed16','budget'],USERS,spec['repeats'],spec['start_repeat'])],
          'sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}
    with (root/'frozen-plan.json').open('x') as sink:
        json.dump(data,sink,indent=2);sink.write('\n')
    print(json.dumps({k:data[k] for k in ['utc','planned_requests','unique_questions','reference_answer_questions','existing_model_processes']}))


if __name__=='__main__':main()
