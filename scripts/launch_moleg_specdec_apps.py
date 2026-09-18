"""Launch three isolated app processes on the fixed 2026-09-05 source copy:
orig (original proxy path), direct (orchestrator -> production instance
directly), spec (orchestrator -> experimental instance). Model servers are
never restarted by this script."""
import json, os, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parent
paper = root.parent / 'paper-20260905.OemN3l'
env = dict(os.environ)
env['MOLEG_RAG_ROOT'] = str(paper / 'app')
env['MOLEG_SERVING_ENV'] = '/data/project/vllm/.env'
env['MOLEG_PROXY_CONFIG'] = '/data/project/vllm/litellm/develop_test_vllm_config_v3.yaml'
env['PYTHONPATH'] = str(root)
python = '/data/project/vllm/fine-tune/2025-moleg-rag/.venv/bin/python'
profiles = [('orig', 28120, None), ('direct', 28121, 'http://127.0.0.1:8000/v1'), ('spec', 28122, 'http://127.0.0.1:8100/v1')]
records = []
for name, port, base in profiles:
    current = {**env, 'MOLEG_STUDY_PROFILE': 'baseline', 'MOLEG_STUDY_PORT': str(port),
               'MOLEG_TRACE_PATH': str(root / 'results' / f'e2e_{name}.trace.jsonl')}
    if base:
        current['MOLEG_ORCH_BASE'] = base
    with (root / 'logs' / f'app_{name}.server.log').open('a') as out:
        proc = subprocess.Popen([python, str(root / 'moleg_specdec_runtime.py')], cwd=root, env=current,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    records.append({'profile': name, 'port': port, 'pid': proc.pid, 'orch_base': base})
(root / 'run' / 'apps.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records))
