"""Replay only pure country-resolution functions from the frozen app source.

No model, database or production mutation. A country differing from the known
source is a scope mismatch, not an expert adjudication of the entire question.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set
from evaluate_moleg_query_anchor import literal_country


def load_resolver(path):
    names={'_normalize_country_spelling','_eu_canonical_from_list',
           '_mentions_gdpr_eu_law_only','_tokens_of_canonical',
           '_contains_country_with_boundary','_resolve_by_unique_list_token',
           'resolve_country_to_canonical'}
    tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in tree.body}==names
    scope={'re':re,'Dict':Dict,'List':List,'Optional':Optional,'Set':Set}
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),scope)
    return scope['resolve_country_to_canonical']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--source-file',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    resolver=load_resolver(args.source_file)
    cases=json.loads((args.directory/'cases.json').read_text())
    extraction={r['case_id']:r['prep'] for line in (args.directory/'retrieval.extraction.jsonl').read_text().splitlines()
                for r in [json.loads(line)] if r['repeat']==0}
    rows=[]
    for case in cases:
        p=extraction[case['case_id']];countries=p['available_countries']
        row={'case_id':case['case_id'],'kind':case['kind'],
             'source_country':case.get('country') if case.get('source_id') else None,
             'legacy_utterance_resolution':resolver(case['question'],countries),
             'literal_unique_resolution':literal_country(case['question'],countries),
             'observed_preparation_country':p.get('country')}
        rows.append(row)
    summary={}
    for field in ['legacy_utterance_resolution','literal_unique_resolution','observed_preparation_country']:
        relevant=[r for r in rows if r['source_country']]
        mismatches=[r['case_id'] for r in relevant if r[field] and r[field]!=r['source_country']]
        summary[field]={'source_questions':len(relevant),'resolved':sum(bool(r[field]) for r in relevant),
            'source_matches':sum(r[field]==r['source_country'] for r in relevant),
            'source_mismatches':len(mismatches),'mismatch_case_ids':mismatches,
            'abstentions':sum(not r[field] for r in relevant)}
    countries=next(iter(extraction.values()))['available_countries']
    probes=['중국 홍콩특별행정구의 법령','중국 홍콩특별행정구 의 법령',
            '중국 마카오특별행정구에서 적용되는 법령','중국 마카오특별행정구 에서 적용되는 법령',
            '대만에서 적용되는 법령','대만 에서 적용되는 법령']
    result={'n':len(rows),'scope':'CPU replay of frozen pure functions and captured country lists; no new inference',
            'source_file_sha256':hashlib.sha256(args.source_file.read_bytes()).hexdigest(),
            'source_label_warning':'Known source jurisdiction, not exhaustive or expert country-intent gold; abstention is not an error.',
            'summary':summary,'minimal_pairs':[{'text':q,'legacy':resolver(q,countries),
                'literal':literal_country(q,countries)} for q in probes],'rows':rows}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
