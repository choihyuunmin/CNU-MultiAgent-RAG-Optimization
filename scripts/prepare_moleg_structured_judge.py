"""Prepare blinded answer assessment from completed study outputs, no model calls."""
import argparse
import json
import os
from pathlib import Path


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--levels',type=int,nargs='+',default=[4,20,100])
    args=p.parse_args();r=args.directory
    cases=json.loads((r/'main.cases.json').read_text())
    rows=[json.loads(x) for x in (r/'main/private-responses.jsonl').read_text().splitlines()]
    plan=json.loads((r/'main-plan.json').read_text())
    variants={a['name'] for a in plan['arms']}
    expected={(c['case_id'],v) for c in cases for v in variants}
    for level in args.levels:
        selected=[x for x in rows if x['level']==level and x['repeat']==0]
        if len(selected)!=len(expected) or {(x['case_id'],x['arm']) for x in selected}!=expected:
            raise ValueError('incomplete/duplicate response set for judge')
        dest=r/f'judge-c{level}'
        dest.mkdir(mode=0o700,exist_ok=False)
        (dest/'cases.json').write_text(json.dumps(cases,ensure_ascii=False))
        with (dest/'e2e_stream.jsonl').open('x') as sink:
            for row in selected:
                sink.write(json.dumps({'case_id':row['case_id'],'variant':row['arm'],'repeat':0,
                                       'result':row.get('result')},ensure_ascii=False)+'\n')
        print(json.dumps({'concurrency':level,'response_records':len(selected),
                          'reference_questions':sum(bool(c.get('reference_answer')) for c in cases)}))


if __name__=='__main__':main()
