"""Read-only, content-free progress for the running reference ID load study."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
plan=json.loads((root/'plan.json').read_text())
state=json.loads((root/'main/campaign-status.json').read_text())
rows=[]
for line in (root/'main/requests.jsonl').read_text().splitlines(keepends=True):
    if line.endswith('\n'):
        rows.append(json.loads(line))
current=state.get('current',{})
subset=[r for r in rows if all(r.get(k)==current.get(k) for k in ['level','arm','repeat'])]
result={'status':state['status'],'completed':len(rows),'planned':plan['planned_requests'],
        'pipeline_failures':sum(not r['pipeline_ok'] for r in rows),
        'current':{**current,'completed':len(subset)}}
for level in plan['levels']:
    expected=plan['requests_per_trial'][str(level)]*plan['repeats']
    groups={arm:[r for r in rows if r['level']==level and r['arm']==arm] for arm in ['baseline','improved']}
    if all(len(group)==expected for group in groups.values()):
        means={arm:sum(r['elapsed_s'] for r in group)/len(group) for arm,group in groups.items()}
        result['last_completed_load']={'level':level,'requests':expected*2,
            'baseline_mean_s':round(means['baseline'],3),'improved_mean_s':round(means['improved'],3),
            'reduction_pct':round(100*(1-means['improved']/means['baseline']),2)}
path=root/'main/summary.json'
if path.exists():
    result['last_trials']=[{'level':t['level'],'arm':t['arm'],'repeat':t['repeat'],
        'n':t['n'],'success':t['pipeline_success'],
        'mean_s':round(t['metrics']['elapsed_s']['mean'],3),
        'p95_s':round(t['metrics']['elapsed_s']['p95'],3)}
        for t in json.loads(path.read_text())['trials'][-2:]]
path=root/'finalization-status.json'
if path.exists():result['finalizer']=json.loads(path.read_text())
print(json.dumps(result))
