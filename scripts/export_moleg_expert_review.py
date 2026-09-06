"""Export a blinded human relevance/answer review packet, with labels unset."""
import argparse
import json
from pathlib import Path
import random


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    args=ap.parse_args();root=args.directory
    cases=json.loads((root/'cases.json').read_text())
    (root/'questions_400.jsonl').write_text(''.join(json.dumps(
        {k:c[k] for k in ['case_id','kind','question']},ensure_ascii=False)+'\n' for c in cases))
    results=[]
    if (root/'e2e_stream.jsonl').exists():
        results=[json.loads(x) for x in (root/'e2e_stream.jsonl').read_text().splitlines()]
    rng=random.Random(20260905);packet=[];key=[]
    for case in cases:
        candidates=[r for r in results if r['case_id']==case['case_id'] and r['repeat']==0]
        rng.shuffle(candidates)
        blinded=[]
        for i,r in enumerate(candidates):
            answer_id=f"{case['case_id']}-{chr(65+i)}"
            blinded.append({'answer_id':answer_id,'answer':(r.get('result') or {}).get('comment',''),
                'laws':(r.get('result') or {}).get('laws',[]),'factual_correctness_0_to_2':None,
                'usefulness_0_to_2':None,'unsupported_claims':None,'reviewer_notes':None})
            key.append({'answer_id':answer_id,'variant':r['variant']})
        packet.append({'case_id':case['case_id'],'question':case['question'],'kind':case['kind'],
            'source_candidate':{k:case.get(k) for k in ['source_index','source_id','source_law_id','country','title','subject','evidence']},
            'question_answerable_from_source':None,'all_relevant_law_ids':None,'all_relevant_chunk_ids':None,
            'reviewer_1':None,'reviewer_2':None,'adjudication':None,'expert_verified':False,'answers':blinded})
    (root/'expert_review.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    (root/'expert_review_blinding_key.json').write_text(json.dumps(key,indent=2)+'\n')
    print('expert_review_questions',len(packet),'completed_expert_labels',0)


if __name__=='__main__':main()
