"""Offline checks of the isolated experiment's data and ranking contracts."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

SCRIPTS=Path(__file__).parents[1]/'scripts'


def load(name):
    # The executable scripts use adjacent imports; no production dependencies
    # are imported until bootstrap is explicitly called.
    original=list(sys.path)
    try:
        sys.path.insert(0,str(SCRIPTS))
        spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:sys.path[:]=original


anchor=load('evaluate_moleg_query_anchor')
runtime=load('moleg_paper_runtime')
summary=load('summarize_moleg_paper')
guardrail_audit=load('audit_moleg_guardrail_mapping')


def test_literal_country_masks_nested_name_and_abstains_on_comparison():
    countries=['중국','중국(홍콩)','독일']
    assert anchor.literal_country('중국(홍콩)의 개인정보 규정',countries)=='중국(홍콩)'
    assert anchor.literal_country('중국과 독일의 규정 비교',countries) is None
    assert anchor.literal_country('개인정보 규정',countries) is None


def test_rrf_deduplicates_each_branch_and_prefers_translated_evidence():
    origin={'id':'a','source':'origin'}
    translated={'id':'a','source':'article'}
    b={'id':'b','source':'article'}
    merged=anchor.reciprocal_rank_fusion([origin,origin,b],[b,translated])
    # Duplicate a in branch one does not receive a second vote.
    assert [r['id'] for r in merged]==['a','b']
    assert merged[0] is translated
    assert origin=={'id':'a','source':'origin'}


def test_reranker_view_is_bounded_and_does_not_mutate_evidence():
    doc={'country':'A','title':'Law','subject':'Article','content':'x'*900+' privacy '+'x'*900}
    before=dict(doc)
    view=runtime.evidence_text('privacy',doc,budget=200,window=True)
    assert len(view)<=200 and 'privacy' in view
    assert doc==before


def test_anchor_summary_scores_failure_as_miss_and_does_not_invent_gold():
    cases={'q1':{'source_id':'123_2','source_law_id':'123','reference_answer':'span'},
           'q2':{'kind':'conversation'}}
    rows=[]
    for name in ['prepared','original']:
        for q in cases:
            rows.append({'case_id':q,'variant':name,'kind':'source' if q=='q1' else 'conversation',
                         'ok':name=='prepared','ranked_ids':['123_2'] if name=='prepared' else [],
                         'elapsed_s':2 if name=='prepared' else 1})
    result,numeric=summary.summarize_query_anchor(rows,cases)
    assert result['variants']['original']['metrics']['chunk_hit1']==0
    assert result['comparisons']['original']['chunk_mrr20_change']['n']==1
    assert result['comparisons']['original']['latency']['n']==2
    assert result['comparisons']['original']['source_qa_nonempty_reference_pairs']==1
    assert result['comparisons']['original']['document_recall_vs_prepared_source_qa']==0
    assert len(numeric)==4
    assert all(r['metrics'] is None for r in numeric if r['case_id']=='q2')


def test_anchor_summary_rejects_duplicate_observations():
    row={'case_id':'q','variant':'prepared'}
    try:summary.summarize_query_anchor([row,row],{})
    except ValueError:pass
    else:raise AssertionError('duplicate observation must not inflate the sample size')


def test_http_success_does_not_hide_known_app_timeout():
    row={'ok':True,'result':{'comment':'응답 생성에 시간이 걸리고 있습니다. 잠시 후 다시 시도해 주세요.'}}
    assert summary.known_error_response(row)=='timeout'
    assert summary.known_error_response({'result':{'comment':'지원하지 않는 국가입니다.'}}) is None
    assert summary.known_error_response({'result':None}) is None


def test_candidate_coverage_is_not_post_rerank_coverage():
    case={'source_id':'123_2','source_law_id':'123','reference_answer':'span'}
    row={'case_id':'q','variant':'prepared','kind':'source','ok':True,'elapsed_s':1.,
         'candidate_ids':['999_1','123_2'],'ranked_ids':['999_1']}
    result,numeric=summary.summarize_query_anchor([row],{'q':case})
    variant=result['variants']['prepared']
    assert variant['candidate_coverage']['source_chunk_in_candidates']==1
    assert variant['metrics']['chunk_hit20']==0
    assert numeric[0]['candidate_coverage']['source_law_in_candidates']==1


def test_country_ablation_counts_failed_search_as_miss():
    case={'source_id':'123_2','source_law_id':'123','reference_answer':'span'}
    row={'case_id':'q','country_changed':True,'ok':False,'variants':{
        'prepared_country':{'ok':True,'elapsed_s':1.,'ranked_ids':['123_2'],'candidate_ids':['123_2']},
        'literal_country_only':{'ok':False,'elapsed_s':2.}}}
    result=summary.summarize_country_ablation([row],{'q':case})
    assert result['chunk_top1_mcnemar']['losses']==1
    assert result['variants']['literal_country_only']['metrics']['chunk_hit1']==0
    assert result['variants']['literal_country_only']['latency']['mean_s']==2.


def test_guardrail_audit_reads_contract_without_executing_source():
    source="_KANANA_CATEGORY_MAP: dict = {'S4': 'pii'}\n"
    source+='\n'.join(f"{name} = re.compile(r'PRIVATE-[0-9]+')" for name in
                     ['_PHONE','_RRN','_EMAIL','_CARD','_ACCOUNT'])
    source+="\nraise RuntimeError('source must not execute')\n"
    mapping,patterns=guardrail_audit.source_contract(SimpleNamespace(read_text=lambda:source))
    assert mapping=={'S4':'pii'}
    assert len(patterns)==5 and all(p.search('PRIVATE-123') for p in patterns)
    assert not any(p.search('legal question') for p in patterns)
