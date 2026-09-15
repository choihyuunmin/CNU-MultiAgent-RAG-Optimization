"""Two isolated app processes for the worker-replica study: w1 routes worker
calls through a single-upstream gateway, w2 through a two-replica gateway.
Both send orchestrator calls directly to the (temporarily speculative) :8000."""
import json, os, subprocess
from pathlib import Path
root = Path(__file__).resolve().parent
paper = root.parent / 'paper-20260905.OemN3l'
env = dict(os.environ, MOLEG_RAG_ROOT=str(paper / 'app'), MOLEG_SERVING_ENV='/data/project/vllm/.env',
           MOLEG_PROXY_CONFIG='/data/project/vllm/litellm/develop_test_vllm_config_v3.yaml', PYTHONPATH=str(root))
python = '/data/project/vllm/fine-tune/2025-moleg-rag/.venv/bin/python'
profiles = [('w1', 28160, 'http://127.0.0.1:28150/v1'), ('w2', 28161, 'http://127.0.0.1:28151/v1')]
records = []
for name, port, worker in profiles:
    current = {**env, 'MOLEG_STUDY_PROFILE': 'baseline', 'MOLEG_STUDY_PORT': str(port),
               'MOLEG_ORCH_BASE': 'http://127.0.0.1:8000/v1', 'MOLEG_WORKER_BASE': worker,
               'MOLEG_TRACE_PATH': str(root / 'results' / f'e1_{name}.trace.jsonl')}
    with (root / 'logs' / f'app_{name}.server.log').open('a') as out:
        proc = subprocess.Popen([python, str(root / 'moleg_specdec_runtime.py')], cwd=root, env=current,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    records.append({'profile': name, 'port': port, 'pid': proc.pid, 'worker_base': worker})
(root / 'run' / 'worker_apps.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records))
