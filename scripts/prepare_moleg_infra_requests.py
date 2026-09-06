"""Freeze real application analysis prompts without changing the application."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
from moleg_paper_runtime import bootstrap


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--settings-env', type=Path)
    args = ap.parse_args()
    if args.settings_env:
        from dotenv import load_dotenv
        load_dotenv(args.settings_env)
    root, serving, config, resolve = bootstrap()
    logging.disable(logging.CRITICAL)
    from config.prompts import get_preparation_system_prompt, get_preparation_response_format
    from domain.country.service import get_available_countries
    import infra.llm.client as llm
    from agent import validation_agent  # Match the application's package import order.
    from core.agent_orchestrator.orchestrator import _CLASSIFY_AND_DECOMPOSE_SYSTEM, _build_classify_user_content
    from infra.llm.client import get_model_for_role
    import requests
    role = next(x['litellm_params'] for x in config['model_list'] if x['model_name'] == get_model_for_role('master'))
    model = role['model'].removeprefix('openai/')
    endpoint = resolve(role['api_base']).rstrip('/')
    r = requests.get(endpoint+'/models', headers={'Authorization': 'Bearer '+serving['VLLM_API_KEY']}, timeout=10)
    r.raise_for_status()
    available = [x['id'] for x in r.json()['data']]
    if model not in available:
        raise RuntimeError('configured model is not served')
    countries = get_available_countries()
    prep_system = get_preparation_system_prompt(countries)
    cases = json.loads(args.cases.read_text())
    rows = []
    for case in cases:
        question = case['question']
        for stage in ['classify', 'prepare']:
            messages = ([{'role': 'system', 'content': _CLASSIFY_AND_DECOMPOSE_SYSTEM},
                         {'role': 'user', 'content': _build_classify_user_content(question, [])}]
                        if stage == 'classify' else prep_system + [{'role': 'user', 'content': question.strip()}])
            payload = {'model': model, 'messages': messages, 'temperature': 0,
                       'response_format': {'type': 'json_object'} if stage == 'classify' else get_preparation_response_format(),
                       'chat_template_kwargs': {'enable_thinking': False}}
            rows.append({'case_id': case['case_id'], 'kind': case['kind'], 'stage': stage,
                         'question_sha256': hashlib.sha256(question.encode()).hexdigest(),
                         'system_sha256': digest(messages[0]['content']), 'payload_sha256': digest(payload),
                         'payload': payload})
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    args.output.with_suffix('.stage_map.json').write_text(json.dumps({r['system_sha256']:r['stage'] for r in rows}, indent=2)+'\n')
    manifest = {'unique_questions': len(cases), 'requests': len(rows), 'case_file_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                'model': model, 'source_root': str(root), 'source_files': {},
                'scope': 'frozen classify and preparation calls for every question; independent request replay, not live branch execution or full RAG',
                'max_tokens_added': False, 'sampling_parameters_changed': False,
                'inputs': [{k:v for k,v in r.items() if k != 'payload'} for r in rows]}
    for rel in ['core/agent_orchestrator/orchestrator.py', 'agent/validation_agent.py', 'infra/llm/client.py',
                'config/prompts/preparation.py', 'config/prompts/orchestrator_routing.py']:
        if not (root/'src'/rel).exists():
            continue
        manifest['source_files'][rel] = hashlib.sha256((root/'src'/rel).read_bytes()).hexdigest()
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'questions': len(cases), 'requests': len(rows), 'model': model}), flush=True)


if __name__ == '__main__':
    main()
