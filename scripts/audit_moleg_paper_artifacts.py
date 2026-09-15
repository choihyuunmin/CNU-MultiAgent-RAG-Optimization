"""Check completed experimental artifacts without claiming expert accuracy."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from evaluate_moleg_query_anchor import literal_country


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.directory
    manifest=json.loads((root/'manifest.json').read_text())
    cases=json.loads((root/'cases.json').read_text())
    assert len(cases)==400 and len({c['question'] for c in cases})==400
    assert len({c['case_id'] for c in cases})==400
    assert hashlib.sha256((root/'cases.json').read_bytes()).hexdigest()==manifest['cases_sha256']
    bycase={c['case_id']:c for c in cases};case_ids=set(bycase)
    questions=rows(root/'questions_400.jsonl')
    assert len(questions)==400 and {q['case_id'] for q in questions}==case_ids
    assert all(q=={k:bycase[q['case_id']][k] for k in ['case_id','kind','question']} for q in questions)
    extraction={r['case_id']:r['prep'] for r in rows(root/'retrieval.extraction.jsonl') if r['repeat']==0}
    eligible={c['case_id'] for c in cases if c.get('reference_answer')}
    assert len(eligible)==131
    assert all(c['reference_answer'] in c['evidence'] for c in cases if c.get('reference_answer'))
    assert not any(c.get('expert_verified') for c in cases)
    audit={'complete':False,'questions':400,'reference_span_questions':len(eligible),
           'cases_sha256':manifest['cases_sha256'],'workloads':{},'warnings':[
               'Technical completeness is not expert legal accuracy or exhaustive error detection.']}
    specifications={
        'retrieval.extraction.jsonl':(None,2),
        'retrieval.jsonl':(None,3),
        'e2e_stream.jsonl':(['baseline','speed','balanced'],2),
        'e2e_json.jsonl':(['baseline','balanced'],1),
        'query_anchor.jsonl':(['prepared','original','anchored_union','fast_extraction'],1),
        'rerank_clipping.jsonl':(None,1),
        'country_ablation.jsonl':(None,1),
        'answer_judgments.jsonl':(['baseline','speed','balanced'],1),
        'generation_replay.jsonl':(['proxy_default','direct_default','direct_low'],2),
    }
    generation_manifest=json.loads((root/'generation_replay_manifest.json').read_text())
    generation_ids={r['case_id'] for r in generation_manifest['inputs']}
    generation_hashes={r['case_id']:r['sha256'] for r in generation_manifest['inputs']}
    assert len(generation_ids)==60 and generation_ids<=case_ids
    for name,(variants,repeats) in specifications.items():
        records=rows(root/name)
        ids=eligible if name=='answer_judgments.jsonl' else generation_ids if name=='generation_replay.jsonl' else case_ids
        wanted={(q,v,rep) for q in ids for v in (variants or [None]) for rep in range(repeats)}
        keys=[(r['case_id'],r.get('variant'),r.get('repeat',0)) for r in records]
        assert len(keys)==len(set(keys)),f'{name}: duplicate observations'
        assert set(keys)==wanted,f'{name}: missing or unexpected observations'
        if name.startswith('e2e_'):
            for r in records:
                expected=hashlib.sha256(bycase[r['case_id']]['question'].encode()).hexdigest()
                assert r['question_sha256']==expected,f'{name}: changed question {r["case_id"]}'
            traces=[t for v in variants for t in rows(root/f'{v}.trace.jsonl')]
            counts=Counter(t['session_id'] for t in traces)
            assert all(counts[r['session_id']]==1 for r in records),f'{name}: missing or duplicate trace'
        if name=='generation_replay.jsonl':
            assert all(r['input_sha256']==generation_hashes[r['case_id']] for r in records)
        if name=='country_ablation.jsonl':
            assert all(set(r['variants'])=={'prepared_country','literal_country_only'} for r in records)
            for r in records:
                prep=extraction[r['case_id']];case=bycase[r['case_id']]
                keywords=list(dict.fromkeys(prep.get('keywords_original',[])+prep.get('keywords_transformed',[])))
                query=prep.get('transformed_query') or ' '.join(keywords) or case['question']
                fingerprint=hashlib.sha256(json.dumps({'keywords':keywords,'query':query},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                assert r['shared_search_input_sha256']==fingerprint
                original=prep.get('country') or None
                corrected=literal_country(case['question'],prep['available_countries']) or original
                assert r['variants']['prepared_country']['country']==original
                assert r['variants']['literal_country_only']['country']==corrected
        audit['workloads'][name]={'rows':len(records),'expected':len(wanted),
            'recorded_failures':sum(r.get('ok') is False for r in records),
            'technical_checks_passed':True}
    packet=json.loads((root/'expert_review.json').read_text())
    assert len(packet)==400 and not any(r['expert_verified'] for r in packet)
    assert all(len(r['answers'])==3 for r in packet)
    assert {r['case_id'] for r in packet}==case_ids
    blinding=json.loads((root/'expert_review_blinding_key.json').read_text())
    answer_ids=[a['answer_id'] for r in packet for a in r['answers']]
    assert len(answer_ids)==len(set(answer_ids))==1200
    assert len(blinding)==1200 and {r['answer_id'] for r in blinding}==set(answer_ids)
    key={r['answer_id']:r['variant'] for r in blinding}
    assert all({key[a['answer_id']] for a in r['answers']}=={'baseline','speed','balanced'} for r in packet)
    assert all(a[k] is None for r in packet for a in r['answers']
               for k in ['factual_correctness_0_to_2','usefulness_0_to_2','unsupported_claims','reviewer_notes'])
    audit['expert_labels_completed']=0
    audit['question_export']={'questions':len(questions),'matches_measured_cases':True,
        'sha256':hashlib.sha256((root/'questions_400.jsonl').read_bytes()).hexdigest()}
    country=json.loads((root/'country_resolution.json').read_text())
    assert country['n']==len(country['rows'])==400
    environment=json.loads((root/'environment_before.json').read_text())
    after=json.loads((root/'environment_after.json').read_text())
    early=json.loads((root/'environment_early_e2e.json').read_text())
    assert environment['source']['sha256']==after['source']['sha256']
    audit['environment']={'app_snapshot_source_unchanged':True,
        'live_source_matches_snapshot_at_end':after['live_source_matches_snapshot'],
        'index_comparison_start_utc':early['utc'],'index_comparison_end_utc':after['utc'],
        'index_observed_fields_unchanged':early['indices']==after['indices'],
        'end_model_statuses':{m['role']:m['status'] for m in after['models']},
        'warning':'Early index statistics start after primary timing began; not an immutable snapshot or complete external-load audit.'}
    assert country['source_file_sha256']==environment['source']['files']['src/domain/country/service.py']
    audit['country_resolution_cpu_replay']={'questions':400,'source_hash_verified':True}
    guardrail=json.loads((root/'guardrail_mapping.json').read_text())
    assert guardrail['n']==len(guardrail['rows'])==400
    assert guardrail['source_file_sha256']==environment['source']['files']['src/agent/guardrail.py']
    probes=guardrail['replays']
    assert len(probes)==len(guardrail['common_zero_score_cases'])==5
    assert {r['case_id'] for r in probes}==set(guardrail['common_zero_score_cases'])
    assert all(r['question_sha256']==hashlib.sha256(bycase[r['case_id']]['question'].encode()).hexdigest() for r in probes)
    post_complete=json.loads((root/'post_complete.json').read_text())
    assert post_complete['complete'] and all(datetime.fromisoformat(r['utc']).timestamp()>=post_complete['unix_time'] for r in probes)
    audit['guardrail_diagnostic']={'cpu_questions':400,'post_hoc_replays':len(probes),
        'recorded_failures':sum(not r['ok'] for r in probes),'source_hash_verified':True,
        'scope':'Diagnostic only; no guardrail relaxation or bypass.'}
    replay=json.loads((root/'stream_replay.json').read_text())
    baseline_sessions={r['session_id'] for r in rows(root/'e2e_stream.jsonl') if r['variant']=='baseline' and r['repeat']==0}
    assert replay['rows'] and all(r['source_session_id'] in baseline_sessions for r in replay['rows'])
    audit['stream_replay']={'n':len(replay['rows']),
        'exact_both':sum(r['paced']['exact_text'] and r['immediate']['exact_text'] for r in replay['rows'])}
    assert audit['stream_replay']['n']==audit['stream_replay']['exact_both']
    audit['total_workload_rows']=sum(w['rows'] for w in audit['workloads'].values())
    audit['complete']=True
    args.output.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
