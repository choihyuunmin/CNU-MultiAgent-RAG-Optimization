import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_moleg_stage_outcomes import classify, EMPTY_PREPARATION


def test_rewritten_preparation_error_is_not_success():
    trace={'preparation':[{'output_sha256':EMPTY_PREPARATION}]}
    flow={'spans':[{'kind':'model_rpc_lifetime','stage':'preparation','success':False}]}
    result=classify(True,trace,flow)
    assert result['preparation_failed']
    assert not result['stage_clean_completion']
    # Empty search results without a failed preparation call are legitimate.
    assert classify(True,trace,{'spans':[]})['stage_clean_completion']


def test_rerank_fallback_is_separately_degraded():
    r=classify(True,{'rerank':[{'status':401}]},{'spans':[]})
    assert r['stage_clean_completion'] and not r['dependency_clean_completion']
    assert r['dependency_statuses']=={'rerank:401':1}
    assert not classify(True,{},None)['stage_clean_completion']
    assert not classify(False,{}, {'spans':[]})['stage_clean_completion']
