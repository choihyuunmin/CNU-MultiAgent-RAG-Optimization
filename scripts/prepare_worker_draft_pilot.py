"""Freeze current search-tool model inputs after real preparation calls (private)."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-root', type=Path, required=True)
    p.add_argument('--app-env', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.umask(0o077)
    from dotenv import load_dotenv
    load_dotenv(args.app_env)
    sys.path.insert(0, str(args.previous_root/'scripts'))
    os.environ['MOLEG_RAG_ROOT'] = str(args.previous_root/'pod_source')
    from moleg_paper_runtime import bootstrap
    bootstrap()
    from agent.validation_agent import run_preparation
    from config.prompts.subagent_systems import LAW_SEARCH_AGENT_SYSTEM
    from config.prompts.tools_spec import tools_named
    cases = json.loads((args.previous_root/'cases.json').read_text())[:14]
    rows = []
    for i, case in enumerate(cases):
        prep = await run_preparation(history=[{'role':'user','content':case['question']}],
                                     request_id=f'draft-input-{i}')
        if prep.get('early_exit') or prep.get('bad_word_detected') or prep.get('pii_detected'):
            raise RuntimeError('preparation failed or rejected an input')
        keywords = list(dict.fromkeys(prep['keywords_original'] + prep['keywords_transformed']))
        params = {'search_terms':prep['transformed_query'] or ' '.join(keywords),
                  'country':prep['country'] or '', 'keywords':keywords}
        payload = {'model':'openai/gpt-oss-20b', 'temperature':0, 'reasoning_effort':'low',
                   'messages':[{'role':'system','content':LAW_SEARCH_AGENT_SYSTEM},
                               {'role':'user','content':json.dumps(params, ensure_ascii=False)}],
                   'tools':tools_named('search_laws'), 'tool_choice':'auto'}
        rows.append({'case_id':case['case_id'], 'warmup':i<2, 'payload':payload,
                     'expected_arguments':params,
                     'question_sha256':hashlib.sha256(case['question'].encode()).hexdigest()})
        print('prepared',i+1,flush=True)
    with args.output.open('x') as f:json.dump(rows,f,ensure_ascii=False)


if __name__ == '__main__':asyncio.run(main())
