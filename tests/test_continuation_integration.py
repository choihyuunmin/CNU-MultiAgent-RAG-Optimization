import asyncio
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from moleg_continuation_harness import install, origin


def test_origin_rejects_embedded_credentials_and_distinguishes_ports():
    assert origin('http://EXAMPLE.test/a')==origin('http://example.test:80/b')
    assert origin('http://example.test:81/a')!=origin('http://example.test/a')
    with pytest.raises(ValueError):origin('https://user:secret@example.test/')


def test_outer_wait_is_outside_timeout_and_transport_does_not_double_acquire(tmp_path,monkeypatch):
    import moleg_paper_runtime
    modules={}
    for name in ['infra','infra.llm','infra.llm.client','tools','tools.search_tool',
                 'tools.search_tool.engine','agent','agent.validation_agent','api',
                 'api.controller','api.controller.generate_controller']:
        module=modules[name]=ModuleType(name)
        monkeypatch.setitem(sys.modules,name,module)
        if '.' in name:
            parent,child=name.rsplit('.',1)
            setattr(modules[parent],child,module)
    llm=modules['infra.llm.client'];prep=modules['agent.validation_agent']
    engine=modules['tools.search_tool.engine'];controller=modules['api.controller.generate_controller']
    engine.RERANK_API_URL='http://rerank.test:8001/v1/rerank'
    engine.RERANK_API_KEY='expired-test-key'
    engine.RERANK_MODEL_NAME='test-reranker'
    posts=[];post_mode=['ok']
    def post(url, **kwargs):
        posts.append((url, kwargs))
        if post_mode[0]=='timeout':raise TimeoutError('test timeout')
        if post_mode[0]=='malformed':return SimpleNamespace(status_code=200,json=lambda:{'unexpected':True})
        return SimpleNamespace(status_code=200, json=lambda:{'results':[{'index':0,'relevance_score':.9},{'index':1,'relevance_score':.1}]})
    engine.requests=SimpleNamespace(post=post)
    monkeypatch.setattr(moleg_paper_runtime,'bootstrap',lambda:(None,{'VLLM_API_KEY':'test-only-key'},None,None))
    source=tmp_path/'source.py';source.write_text('# pinned fixture\n')
    prep.__file__=engine.__file__=llm.__file__=controller.__file__=str(source)
    llm.MASTER_MODEL='model';llm.get_model_for_role=lambda role:'model'
    seen=[];active=0;peak=0
    async def transport(**kwargs):
        nonlocal active,peak
        seen.append(kwargs);active+=1;peak=max(peak,active)
        try:
            await asyncio.sleep(.025)
            return kwargs['messages']
        finally:active-=1
    stream_closed=[]
    async def streaming(**kwargs):
        try:yield kwargs['messages']
        finally:stream_closed.append(True)
    llm.acompletion_via_proxy=transport;llm.acompletion_stream_via_proxy=streaming
    async def call_llm(**kwargs):
        return await asyncio.wait_for(llm.acompletion_via_proxy(model='model',**kwargs),timeout=.05)
    llm.call_llm=call_llm
    async def prepare(**kwargs):return {'early_exit':True,'early_exit_response':{}}
    prep.run_preparation=prepare
    async def execute(**kwargs):return await llm.call_llm(**kwargs)
    controller.execute_generate=execute
    events=[]
    monkeypatch.setattr(moleg_paper_runtime,'trace_add',lambda kind,**fields:events.append((kind,fields)))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'fingerprints':{m.__name__:digest for m in [prep,engine,llm,controller]},
                              'model_slots':{'model':1}, 'rerank_serving_auth':True,
                              'rerank_origin':'http://rerank.test:8001'}))
    state=install(path)
    assert state['rerank_auth_repaired']
    assert posts[0][1]['allow_redirects'] is False
    assert posts[0][1]['headers']['Authorization']=='Bearer test-only-key'
    assert engine.RERANK_API_KEY=='test-only-key'
    with pytest.raises(ValueError):engine.requests.post('http://unrelated.test/rerank')
    assert len(posts)==1
    payload=[{'role':'user','content':'Keep this content and seed unchanged.'}]
    async def run():
        results=await asyncio.wait_for(asyncio.gather(*(controller.execute_generate(messages=payload,seed=19)
                                                     for _ in range(4))),timeout=.6)
        assert all(result is payload for result in results)
        await prep.run_preparation()
        # Low-level callers still acquire a credit when not inside call_llm.
        await llm.acompletion_via_proxy(model='model',messages=payload,seed=19)
        assert state['windows']['model'].active==0
        source=llm.acompletion_stream_via_proxy(model='model',messages=payload)
        assert await anext(source) is payload
        assert state['windows']['model'].active==1
        await source.aclose()
        assert state['windows']['model'].active==0 and stream_closed==[True]
    asyncio.run(run())
    assert peak==1 and len(seen)==5
    assert all(k['messages'] is payload and k['seed']==19 for k in seen)
    assert len([e for kind,e in events if kind=='dispatch' and e['event']=='dispatch'])==6
    assert any(kind=='dependency_failure' and e['component']=='preparation' for kind,e in events)

    post_mode[0]='timeout'
    with pytest.raises(TimeoutError):engine.requests.post(engine.RERANK_API_URL,json={'documents':['a','b']})
    assert any(kind=='dependency_failure' and e.get('error_type')=='TimeoutError' for kind,e in events)
    post_mode[0]='malformed'
    engine.requests.post(engine.RERANK_API_URL,json={'documents':['a','b']})
    assert any(kind=='dependency_failure' and e.get('reason')=='invalid_score_response' for kind,e in events)
