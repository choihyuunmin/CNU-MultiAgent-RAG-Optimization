"""Offline transfer audit: generic compiler versus frozen MOLEG representation.

Reads private captured inputs but emits counts only. This does not call a model
or imply that a new end-to-end experiment has been performed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from cnu_rag_optimization.reference_harness import ReferenceContract, compile_references
from cnu_rag_optimization.structured_harness import encode_selection_ids, strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    contract = ReferenceContract('moleg-frozen-selection-transfer-audit', ('laws',), 'id', ('selected_ids',))
    for line in args.capture.open():
        row = json.loads(line)
        if row['schema'] != 'law_selection_response':
            continue
        counts['selection_calls'] += 1
        messages = row['kwargs']['messages']
        encoded, mapping = encode_selection_ids(messages)
        if not mapping:
            counts['original_policy_bypass'] += 1
            continue
        counts['original_policy_encoded'] += 1
        for i, message in enumerate(messages):
            text = message.get('content','')
            if not isinstance(text,str) or '# 법령 검색 결과\n' not in text:
                continue
            start = text.index('# 법령 검색 결과\n')+len('# 법령 검색 결과\n')
            end = text.index('\n</search_results>', start)
            protected = [m.get('content','') for j,m in enumerate(messages) if j != i and isinstance(m.get('content'),str)]
            protected += [text[:start], text[end:]]
            compiled = compile_references(text[start:end], contract, protected=protected)
            counts['generic_'+compiled.reason] += 1
            if compiled.enabled:
                counts['encoded_bytes_equal'] += text[:start]+compiled.encoded+text[end:] == encoded[i]['content']
                counts['mapping_equal'] += {str(n):x for n,x in enumerate(compiled.originals)} == mapping
                original_output = strict_json(row['output'])
                inv = {v:k for k,v in mapping.items()}
                short = dict(original_output, selected_ids=[inv[x] for x in original_output['selected_ids']])
                restored = strict_json(compiled.restore(json.dumps(short)))
                counts['restored_values_equal'] += restored == original_output
            break
    report = {'offline_only':True, 'min_candidates':16, **counts}
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
