import importlib.util
from pathlib import Path
import sys
import pytest

scripts=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(scripts))
try:
    import summarize_moleg_infra as metrics
finally:
    sys.path.pop(0)


def fixture():
    inputs=[{'case_id':'q1','stage':'classify','payload_sha256':'p1','question_sha256':'q1'}]
    manifest={'inputs':inputs,'variants':['proxy','pass'],'repeats':2,'burst':16,'scope':'test'}
    rows=[];blocks=[]
    for variant in manifest['variants']:
        for repeat in range(2):
            elapsed=2. if variant=='proxy' else 1.
            rows.append(dict(inputs[0],variant=variant,repeat=repeat,block=0,ok=True,status=200,
                             request_id=f'{variant}{repeat}',wire_sha256='wire',response_sha256='response',
                             elapsed_s=elapsed,valid_json=True,finish_reason='stop',normalized_sha256='answer',
                             reasoning_chars=0,usage={'prompt_tokens':10,'completion_tokens':5},gateway_wait_s=.1,upstream_s=.8))
            blocks.append({'variant':variant,'repeat':repeat,'block':0,'n':1,'wall_s':elapsed,'ok':1})
    return rows,blocks,manifest


def test_block_bootstrap_uses_request_weights():
    result=metrics.clustered_ratio([(10,5,10),(100,50,1)],resamples=100)
    assert result['baseline_mean_s']==10
    assert result['candidate_mean_s']==5
    assert result['reduction_pct']==50
    assert result['reduction_95ci_pct']==[50,50]


def test_complete_and_trace_audit():
    rows,blocks,manifest=fixture()
    traces=[{'request_id':r['request_id'],'input_sha256':'wire','forward_sha256':'wire',
             'response_sha256':'response','stage':'classify'} for r in rows if r['variant']=='pass']
    result=metrics.summarize(rows,blocks,manifest,traces)
    assert result['complete'] and result['rows']==4
    assert result['comparisons']['proxy_vs_pass']['all']['reduction_pct']==50
    assert result['gateway_audit']['requests']==2


def test_missing_and_duplicate_rows_rejected():
    rows,blocks,manifest=fixture()
    with pytest.raises(ValueError):metrics.summarize(rows[:-1],blocks,manifest)
    with pytest.raises(ValueError):metrics.summarize(rows+rows[:1],blocks,manifest)


def test_changed_input_rejected():
    rows,blocks,manifest=fixture()
    rows[0]['payload_sha256']='changed'
    with pytest.raises(ValueError):metrics.summarize(rows,blocks,manifest)


def test_missing_burst_is_not_complete():
    rows,blocks,manifest=fixture()
    with pytest.raises(ValueError):metrics.summarize(rows,blocks[:-1],manifest)


def test_lost_wall_time_preserves_individual_observations():
    rows,blocks,manifest=fixture()
    blocks[2]['wall_s']=None
    result=metrics.summarize(rows,blocks,manifest)
    assert result['complete'] and result['rows']==4
    assert result['variants']['pass']['bursts_with_missing_wall']==1
    assert result['comparisons']['proxy_vs_pass']['burst_completion']['n_clusters']==1


def test_environment_audit_detects_model_argument_change():
    import copy
    sys.path.insert(0,str(scripts))
    try:
        from audit_moleg_infra_environment import audit
    finally:
        sys.path.pop(0)
    cluster={'utc':'test','deployments':[],'pods':[],'services':[]}
    gpu={'utc':'test','gpus':['GPU'],'servers':[{'pid':1,'arguments':{'limit':4}}]}
    assert audit(cluster,cluster,gpu,gpu)['checks']['all_checks_passed']
    after=copy.deepcopy(gpu);after['servers'][0]['arguments']['limit']=8
    assert not audit(cluster,cluster,gpu,after)['checks']['all_checks_passed']


def test_failed_request_is_not_removed_from_latency():
    rows,blocks,manifest=fixture()
    rows[2].update(ok=False,error_type='TimeoutError',elapsed_s=90)
    result=metrics.summarize(rows,blocks,manifest)
    assert result['variants']['pass']['n']==2
    assert result['variants']['pass']['ok']==1
    assert result['variants']['pass']['latency']['mean_s']==45.5


def test_response_change_is_detected():
    rows,blocks,manifest=fixture()
    traces=[{'request_id':r['request_id'],'input_sha256':'wire','forward_sha256':'wire',
             'response_sha256':'CHANGED','stage':'classify'} for r in rows if r['variant']=='pass']
    with pytest.raises(ValueError):metrics.summarize(rows,blocks,manifest,traces)


def test_compression_preserves_bytes_and_rejects_trailing_data():
    import gzip
    sys.path.insert(0,str(scripts))
    try:
        from measure_moleg_payload_transport import unpack
    finally:
        sys.path.pop(0)
    raw='질문과 지시문'.encode()*100
    compressed=gzip.compress(raw,mtime=0)
    assert unpack(raw,'identity')==raw
    assert unpack(compressed,'gzip')==raw
    with pytest.raises(ValueError):unpack(compressed+b'trailing','gzip')
    with pytest.raises(ValueError):unpack(compressed[:8],'gzip')
    with pytest.raises(ValueError):unpack(raw,'unknown')


def test_transport_audit_counts_and_byte_scope():
    sys.path.insert(0,str(scripts))
    try:
        from summarize_moleg_transport import summarize
    finally:
        sys.path.pop(0)
    rows=[dict(case_id='q',stage='classify',repeat=rep,variant=v,exact=True,
               sha256='same',raw_bytes=100,sent_body_bytes=100 if v=='identity' else 50,
               elapsed_s=.002 if v=='identity' else .001,compress_s=.0001,
               receiver_decode_hash_s=.0001)
          for rep in range(3) for v in ['identity','gzip']]
    manifest={'inputs':[{'case_id':'q','stage':'classify'}]}
    result=summarize(rows,manifest)
    assert result['rows']==6
    assert result['variants']['gzip']['body_reduction_pct']==50
    with pytest.raises(ValueError):summarize(rows[:-1],manifest)
    rows[-1]['sha256']='changed'
    with pytest.raises(ValueError):summarize(rows,manifest)
