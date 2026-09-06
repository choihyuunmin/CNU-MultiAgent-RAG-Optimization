"""Read-only guardrail label/response audit; never disable or bypass a guard.

CPU analysis covers the 400 existing questions. Optional server replay targets
only the common zero-scored source questions and must run after timed workloads.
It sends the original classifier request unchanged, not an alternative prompt.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import time


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_contract(path):
    assignments={node.targets[0].id:node.value for node in ast.parse(path.read_text()).body
                 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            assignments[node.target.id]=node.value
    mapping=ast.literal_eval(assignments['_KANANA_CATEGORY_MAP'])
    patterns=[]
    for name in ['_PHONE','_RRN','_EMAIL','_CARD','_ACCOUNT']:
        call=assignments[name]
        assert isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='compile'
        patterns.append(re.compile(ast.literal_eval(call.args[0])))
    return mapping,patterns


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--source-file',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--replay-server',action='store_true')
    args=ap.parse_args();root=args.directory
    mapping,patterns=source_contract(args.source_file)
    source_hash=hashlib.sha256(args.source_file.read_bytes()).hexdigest()
    expected=json.loads((root/'environment_before.json').read_text())['source']['files']['src/agent/guardrail.py']
    assert source_hash==expected
    cases=json.loads((root/'cases.json').read_text());bycase={c['case_id']:c for c in cases}
    streams={(r['case_id'],r['variant']):r for r in records(root/'e2e_stream.jsonl') if r['repeat']==0}
    judgments={(r['case_id'],r['variant']):r for r in records(root/'answer_judgments.jsonl')}
    names=['baseline','speed','balanced'];rows=[];selected=[];counts=Counter()
    for case in cases:
        q=case['case_id'];warnings=[]
        for name in names:
            comment=(streams[q,name].get('result') or {}).get('comment','')
            if comment.startswith('입력하신 내용에 개인정보(연락처, 이메일, 주소, 계좌번호 등)가 포함된 것으로 보여'):
                warnings.append(name);counts[name]+=1
        pii_regex=any(p.search(case['question']) for p in patterns)
        row={'case_id':q,'kind':case['kind'],'source_qa':bool(case.get('reference_answer')),
             'pii_warning_variants':warnings,'pii_regex_match':pii_regex}
        rows.append(row)
        if len(warnings)==3 and all((judgments.get((q,n)) or {}).get('support')==0 for n in names):
            selected.append(q)
    result={'n':len(cases),'utc':datetime.now(timezone.utc).isoformat(),
        'source_file_sha256':source_hash,'source_hash_verified':True,
        'scope':'Post-hoc read-only audit of existing first-repeat SSE responses; not a guardrail-bypass or accuracy ablation.',
        'app_category_map':mapping,'original_publisher_S4_category':'crime',
        'category_reference':'https://huggingface.co/kakaocorp/kanana-safeguard-8b/blob/main/README.md',
        'warning':'Original publisher taxonomy is not a weight-identity check of the served mirror. Original per-request classifier codes were not captured.',
        'pii_warning_counts':dict(counts),'common_zero_score_cases':selected,
        'selected_questions_with_pii_regex_match':sum(r['pii_regex_match'] for r in rows if r['case_id'] in selected),
        'rows':rows,'replays':[]}
    if args.replay_server:
        from moleg_paper_runtime import bootstrap
        _,serving,_,_=bootstrap()
        import requests
        from config import settings
        with requests.Session() as client:
            for q in selected:
                question=bycase[q]['question'];start=time.perf_counter()
                row={'case_id':q,'utc':datetime.now(timezone.utc).isoformat(),
                     'question_sha256':hashlib.sha256(question.encode()).hexdigest(),
                     'model':settings.KANANA_GUARDRAIL_MODEL}
                try:
                    response=client.post(settings.KANANA_GUARDRAIL_API_BASE.rstrip('/')+'/chat/completions',
                        headers={'Authorization':'Bearer '+serving['VLLM_API_KEY']},
                        json={'model':settings.KANANA_GUARDRAIL_MODEL,'messages':[{'role':'user','content':question}],
                              'max_tokens':16,'temperature':0},timeout=settings.KANANA_GUARDRAIL_TIMEOUT)
                    row['http_status']=response.status_code;response.raise_for_status()
                    content=(response.json()['choices'][0]['message']['content'] or '').strip().upper()
                    match=re.search(r'<SAFE>|<UNSAFE-S\d+>',content)
                    token=match.group() if match else None
                    code=token[len('<UNSAFE-'):-1] if token and token.startswith('<UNSAFE-') else None
                    row.update(ok=token is not None,classification_token=token,
                               mapped_kind=mapping.get(code,'bad_word') if code else None)
                except Exception as exc:row.update(ok=False,error_type=type(exc).__name__)
                row['elapsed_s']=time.perf_counter()-start;result['replays'].append(row)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))


if __name__=='__main__':main()
