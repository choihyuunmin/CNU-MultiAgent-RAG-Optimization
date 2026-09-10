"""Private-only conversion to the existing blinded answer-judge protocol."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--cases', type=Path, help='explicit regression dataset; defaults to the original holdout file')
    p.add_argument('--repeat', type=int, default=0, help='first repeat index of this separately reported segment')
    args = p.parse_args()
    root = args.directory
    cases = json.loads((args.cases or root / 'holdout.cases.json').read_text())
    rows = [json.loads(line) for line in (root / 'validation.responses.jsonl').open()]
    for users in sorted({r['users'] for r in rows}):
        dest = root / f'judge-u{users}'
        dest.mkdir(mode=0o700, exist_ok=False)
        with (dest / 'cases.json').open('x') as sink:
            json.dump(cases, sink, ensure_ascii=False)
        selected = [r for r in rows if r['users']==users and r['repeat']==args.repeat]
        expected = {(c['case_id'], arm) for c in cases for arm in ['baseline','fixed16','budget']}
        observed = [(r['case_id'],r['policy']) for r in selected]
        if len(observed)!=len(expected) or set(observed)!=expected:
            raise ValueError('quality comparison requires all attempted first-repeat questions, including failures')
        with (dest / 'e2e_stream.jsonl').open('x') as sink:
            for row in selected:
                row['variant'] = row['policy']
                sink.write(json.dumps(row, ensure_ascii=False) + '\n')
        for path in dest.iterdir():
            path.chmod(0o600)
        print(json.dumps({'users':users, 'first_repeat_records':len(selected),
                          'source_qa_questions':sum(bool(c.get('reference_answer')) for c in cases),
                          'scope':'private blinded judge input; not new inference results'}))


if __name__ == '__main__':
    main()
