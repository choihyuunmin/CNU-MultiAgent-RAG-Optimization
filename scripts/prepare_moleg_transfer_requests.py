"""Re-target the frozen orchestrator requests to a different served model
(same messages, same structured response formats, temperature 0) to test
whether the drafting policy transfers across model families."""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--requests', type=Path, required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--reasoning-effort', default='')
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
rows = json.loads(a.requests.read_text())
for r in rows:
    p = r['payload']
    p['model'] = a.model
    p.pop('chat_template_kwargs', None)
    if a.reasoning_effort:
        p['reasoning_effort'] = a.reasoning_effort
a.output.write_text(json.dumps(rows, ensure_ascii=False))
print(len(rows), 'requests ->', a.model)
