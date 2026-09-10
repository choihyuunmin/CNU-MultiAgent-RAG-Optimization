"""Freeze development/holdout membership and initial isolated adapter settings."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evaluate_moleg_scaling import select_cases, digest


HOOKS = [
    ('core.agent_orchestrator.orchestrator', 'classify_and_decompose', 'classification'),
    ('core.agent_orchestrator.orchestrator', 'classify_intent_with_confidence', 'classification'),
    ('agent.validation_agent', 'run_preparation', 'preparation'),
    ('agent.law_search_agent', 'run_search', 'retrieval'),
    ('agent.law_analysis_agent', 'run_select_laws', 'selection'),
    ('agent.law_analysis_agent', 'run_generate_comment', 'answer'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--previous-manifest', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    dest = args.directory
    cases = json.loads(args.cases.read_text())
    previous = {c['case_id'] for c in json.loads(args.previous_manifest.read_text())['selected_cases']}
    available = [c for c in cases if c['case_id'] not in previous]
    dev = select_cases(available, 32, 20260908)
    dev_ids = {c['case_id'] for c in dev}
    dev_laws = {c['source_law_id'] for c in dev if c.get('source_law_id')}
    held = select_cases([c for c in available if c['case_id'] not in dev_ids
                         and c.get('source_law_id') not in dev_laws], 64, 20260909)
    assert not dev_ids & {c['case_id'] for c in held}
    for name, rows in [('development', dev), ('holdout', held)]:
        with (dest / (name + '.cases.json')).open('x') as f:
            json.dump(rows, f, ensure_ascii=False)
        (dest / (name + '.cases.json')).chmod(0o600)
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'seed': 20260908,
        'scope': 'held out from this budget development and Sep 7 scaling selection, not unseen in all prior studies',
        'development': [{'case_id': c['case_id'], 'kind': c['kind'], 'question_sha256': digest(c['question'])} for c in dev],
        'holdout': [{'case_id': c['case_id'], 'kind': c['kind'], 'question_sha256': digest(c['question'])} for c in held],
        'same_source_law_overlap_development_holdout': False,
        'quality': 'silver source labels and planned blind automated answer assessment; no expert equivalence claim',
        'budget_rule': {'max_calls': 8, 'max_kv_tokens': 'ceil(8 * development p95 prompt tokens + 8 * output reserve)',
                        'max_prefill_tokens': 'ceil(8 * development p95 prompt tokens)',
                        'output_reserve': 'max(512, ceil(1.5 * max development completion tokens))',
                        'cached_input_credit': 0},
        'primary_validation': {'users': [16,32], 'repeats': 2, 'variants': ['baseline','fixed16','budget'],
                               'questions': 64, 'deadline_s': 240, 'slo_s': 30},
        'rule': 'stop after incomplete SSE or known pipeline error; never discard failed rows',
        'case_file_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest()}
    (dest / 'split-plan.json').write_text(json.dumps(manifest, indent=2) + '\n')
    config = {'mode': 'observe', 'serving_meter': {'roles': ['orchestrator'], 'output_reserve_tokens': 512},
              'stage_hooks': [dict(module=m, function=f, stage=s) for m,f,s in HOOKS]}
    (dest / 'observe.json').write_text(json.dumps(config, indent=2) + '\n')
    arms = [{'name': 'baseline', 'slots': 4, 'emission': 'original', 'port': 28320,
             'adapter_config': str(dest / 'observe.json')},
            {'name': 'fixed16', 'slots': 16, 'emission': 'immediate', 'port': 28321,
             'adapter_config': str(dest / 'observe.json')}]
    (dest / 'development-arms.json').write_text(json.dumps(arms, indent=2) + '\n')
    print(json.dumps({'development': dict(Counter(c['kind'] for c in dev)),
                      'holdout': dict(Counter(c['kind'] for c in held)), 'split_frozen': True}))


if __name__ == '__main__':
    main()
