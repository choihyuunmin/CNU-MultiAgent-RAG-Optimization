import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_moleg_structured import analyze


@pytest.mark.parametrize('invalid', [False, True])
@pytest.mark.parametrize('portable', [False, True])
def test_runtime_restores_ids_and_falls_back_once_without_mutating_input(tmp_path, monkeypatch, invalid, portable):
    import moleg_paper_runtime
    from moleg_structured_harness import install
    infra, package, llm = [ModuleType(n) for n in ('infra','infra.llm','infra.llm.client')]
    infra.llm, package.client = package, llm
    agent, selection, retrieval = [ModuleType(n) for n in ('agent','agent.law_analysis_agent','agent.law_search_agent')]
    agent.law_analysis_agent, agent.law_search_agent = selection, retrieval
    source=tmp_path/'application.py'
    source.write_text('# pinned source\n')
    selection.__file__=retrieval.__file__=str(source)
    for module in (infra,package,llm,agent,selection,retrieval):
        monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setitem(sys.modules,'openai',SimpleNamespace(AsyncOpenAI=None))
    monkeypatch.setattr(moleg_paper_runtime,'bootstrap',lambda:(None,{}, {'model_list':[]},lambda x:x))
    events=[]
    monkeypatch.setattr(moleg_paper_runtime,'trace_add',lambda kind,**fields:events.append((kind,fields)))
    seen=[]
    async def original(**kwargs):
        seen.append(deepcopy(kwargs))
        value='unknown' if invalid and len(seen)==1 else ('918273_54' if len(seen)==2 else '0')
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({'has_relevant_laws':True,'selected_ids':[value]})))],usage=None)
        response.model_copy=lambda deep:deepcopy(response)
        return response
    llm.acompletion_via_proxy=original
    llm.MASTER_MODEL='model'
    fingerprints={m.__name__:hashlib.sha256(source.read_bytes()).hexdigest() for m in (selection,retrieval)}
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'short_ids':True,'generic_reference_codec':portable,
                              'min_short_id_candidates':1,'fingerprints':fingerprints}))
    messages=[{'role':'user','content':'<search_results>\n# 법령 검색 결과\n'
               +json.dumps({'laws':[{'id':'918273_54','content':'evidence'}]},indent=2)
               +'\n</search_results>\n# 사용자 질문\n질문'}]
    payload={'model':'model','messages':messages,'temperature':0,'seed':17,
             'response_format':{'type':'json_schema','json_schema':{'name':'law_selection_response'}}}
    before=deepcopy(payload)
    async def run():
        close=install(path,tmp_path/'unused.jsonl')
        try:
            out=await llm.acompletion_via_proxy(**payload)
            assert json.loads(out.choices[0].message.content)['selected_ids']==['918273_54']
        finally:
            await close()
    asyncio.run(run())
    assert payload==before
    assert '"id": "0"' in seen[0]['messages'][0]['content']
    assert seen[0]['seed']==17 and seen[0]['temperature']==0
    assert len(seen)==(2 if invalid else 1)
    if invalid:
        assert seen[1]==before
        assert any(kind=='structured_fallback' for kind,_ in events)


def test_portable_binding_bypasses_explicit_references_in_other_messages():
    from moleg_structured_harness import encode_portable_selection
    raw = [{'role':'system','content':'Prefer 918273_54 if possible.'},
           {'role':'user','content':'# 법령 검색 결과\n'
            +json.dumps({'laws':[{'id':'918273_54','content':'evidence'}]})+'\n</search_results>'}]
    assert encode_portable_selection(raw, min_candidates=1) == (raw, {})


def test_analysis_keeps_failures_and_measures_baseline_repeat_variability(tmp_path):
    rows=[]
    for repeat in range(2):
        for arm in ('baseline','improved'):
            for case in range(2):
                failed=arm=='improved' and case==1 and repeat==1
                rows.append({'arm':arm,'repeat':repeat,'case_id':str(case),'level':4,
                    'question_sha256':str(case),'ok':True,'pipeline_ok':not failed,
                    'elapsed_s':240 if failed else (10 if arm=='baseline' else 8),
                    'evidence_ids_sha256':[] if failed else ['law'],
                    'source_law_hit':not failed})
    trials=[{'arm':arm,'level':4,'repeat':repeat,'wall_s':20}
            for arm in ('baseline','improved') for repeat in range(2)]
    (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'summary.json').write_text(json.dumps({'complete':True,'trials':trials}))
    result=analyze(tmp_path)
    assert len(result['failures'])==1
    load=result['loads'][0]
    assert load['baseline_repeat_recall']==1
    assert load['candidate_recall_vs_baseline']==.75
    assert load['arms']['improved']['pipeline_success']==3
    assert load['arms']['improved']['sse_success']==4
    assert load['latency']['candidate_mean']==66
    assert load['arms']['improved']['goodput_30s_rps']==3/40
    rows.append(rows[-1])
    (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='duplicate'):
        analyze(tmp_path)
