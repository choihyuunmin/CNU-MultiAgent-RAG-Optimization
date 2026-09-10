"""Export public-safe stopped-campaign evidence and a fail-closed quality decision."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cnu_rag_optimization.quality_gate import QualityEvidence, evaluate_quality_gate


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args = p.parse_args()
    root = args.directory.resolve()
    destination = root/'publication'
    destination.mkdir(exist_ok=False)
    files = ['validation/manifest.json','validation/summary.json','validation/requests.jsonl',
             'validation/workflow-analysis.json','validation/resources.json','validation/app-health.json',
             'validation/failure-audit-v2.json','calibration.json','split-plan.json','preflight.json',
             'observe-final.json','budget-final.json','judge-u16/answer_judge_summary.json']
    for name in files:
        shutil.copyfile(root/name,destination/Path(name).name)
    before = json.loads((root/'environment-before.json').read_text())
    import httpx
    checks = []
    async with httpx.AsyncClient(timeout=10,trust_env=False) as client:
        for server in before['servers']:
            row = {'pid':server['pid'],'gpu':server['gpu'],
                   'model':server['model'] if not server['model'].startswith('/') else 'comparison fine-tuned model'}
            args_before = server['arguments']
            try:
                argv = (Path('/proc')/str(server['pid'])/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                row['registered_pid_alive'] = True
                row['allowlisted_arguments_unchanged'] = all(k in argv and argv[argv.index(k)+1]==v for k,v in args_before.items())
                response = await client.get('http://127.0.0.1:'+args_before['--port']+'/health')
                row['health_status'] = response.status_code
            except Exception as exc:
                row['error_type'] = type(exc).__name__
            row['configuration'] = {k:v for k,v in args_before.items() if k!='--port'}
            checks.append(row)
    analysis = json.loads((root/'validation/workflow-analysis.json').read_text())
    evidence = QualityEvidence(unique_questions=64,paired_complete=analysis['complete'],
        baseline_success_rate=63/64,candidate_success_rate=63/64,
        source_delta_lower=next(r for r in analysis['adapter_comparisons'] if r['control']=='fixed16')['source_law_hit_change']['delta_95ci'][0],
        independent_holdout=True,thresholds_predeclared=False)
    gate = evaluate_quality_gate(evidence)
    gate['scope'] = 'stopped campaign; baseline includes one logged internal fallback error; automated judge is not expert evidence'
    (destination/'quality-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    runtime_paths = [*sorted((root/'src').rglob('*.py')),*sorted((root/'scripts').glob('*.py'))]
    (destination/'provenance.json').write_text(json.dumps({
        'utc':datetime.now(timezone.utc).isoformat(),
        'runtime_archive_sha256':hashlib.sha256((root/'runtime-main.tar.gz').read_bytes()).hexdigest(),
        'script_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_paths},
        'note':'runtime archive froze serving code before main; analysis-only scripts were added after it',
        'models_after':checks,
        'k8s_after_observed':{'generation':36,'ready_replicas':1,'available_replicas':1,'rag_restarts':0},
        'k8s_note':'read-only operator terminal observation; not queried by this GPU-host script',
        'production_mutations':False,
        'campaign_complete':False},indent=2)+'\n')
    print(json.dumps({'exported':len(files),'quality_gate':gate,'models_after':checks}))


if __name__=='__main__':
    asyncio.run(main())
