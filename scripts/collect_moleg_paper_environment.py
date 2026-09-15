"""Capture a secret-free model/hardware/source manifest and optional GPU samples."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from moleg_paper_runtime import bootstrap


def tree_hash(root):
    digest=hashlib.sha256();files={}
    for p in sorted((root/'src').rglob('*.py')):
        value=hashlib.sha256(p.read_bytes()).hexdigest()
        relative=str(p.relative_to(root));files[relative]=value
        digest.update((relative+'\0'+value+'\n').encode())
    return {'sha256':digest.hexdigest(),'files':files}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--live-root',type=Path)
    ap.add_argument('--serving-python')
    ap.add_argument('--monitor-pid',type=int)
    args=ap.parse_args()
    root,serving,config,resolve=bootstrap()
    import requests
    packages={}
    for package in ['openai','httpx','torch','numpy','langgraph','opensearch-py','fastapi']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                                 '--format=csv,noheader'],text=True).strip().splitlines()
    info={'utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
          'packages':packages,'gpus':gpus,'source':tree_hash(root),'models':[]}
    study_root=Path(__file__).resolve().parent
    info['experiment_script_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(study_root.glob('*.py'))}
    from config import settings
    setting_names=['HYBRID_W_BM25','HYBRID_W_FULL','HYBRID_W_PARA',
        'HYBRID_VECTOR_FULL_TOP_K','HYBRID_VECTOR_PARA_TOP_K','RERANK_TOP_K',
        'RERANK_SCORE_MIN','RERANK_DOCUMENT_TEXT_MAX_CHARS','RERANK_QUERY_MAX_CHARS',
        'ENABLE_WORLD_LAW_ORIGIN_SEARCH','PREPARATION_LLM_ROLE','LITELLM_CHAT_TIMEOUT']
    info['effective_nonsecret_settings']={name:getattr(settings,name,None) for name in setting_names}
    if args.live_root:
        live=tree_hash(args.live_root)
        info['live_source_matches_snapshot']=live['sha256']==info['source']['sha256']
        info['live_source_sha256']=live['sha256']
    if args.serving_python:
        code="import json,importlib.metadata as m;print(json.dumps({p:m.version(p) for p in ['vllm','torch','numpy','numba']}))"
        info['serving_packages']=json.loads(subprocess.check_output([args.serving_python,'-c',code],text=True))
        code=("import json,hashlib,importlib.util;from pathlib import Path;"
              "root=Path(importlib.util.find_spec('vllm').origin).parent;"
              "paths=['v1/spec_decode/ngram_proposer.py','config/speculative.py',"
              "'model_executor/models/gemma4.py','model_executor/models/gemma4_mm.py'];"
              "print(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if (root/p).exists()}))")
        info['serving_component_sha256']=json.loads(subprocess.check_output([args.serving_python,'-c',code],text=True))
    for item in config['model_list']:
        p=item['litellm_params']
        if not p['model'].startswith('openai/'):continue
        base=resolve(p['api_base']);key=resolve(p.get('api_key')) or serving['VLLM_API_KEY']
        response=requests.get(base.rstrip('/')+'/models',headers={'Authorization':'Bearer '+key},timeout=10)
        info['models'].append({'role':item['model_name'],'configured_model':p['model'],
            'status':response.status_code,'served_models':[x['id'] for x in response.json().get('data',[])]})
    from infra.vector_db.vector_store import get_opensearch_client
    from config.settings import INDEX_NAME_ARTICLE,INDEX_NAME_WORLD_LAW_ORIGIN
    client=get_opensearch_client();info['indices']={}
    for index in [INDEX_NAME_ARTICLE,INDEX_NAME_WORLD_LAW_ORIGIN]:
        entry=client.indices.stats(index=index,level='shards')['indices'][index]
        info['indices'][index]={'uuid':entry.get('uuid'),'docs':entry['primaries']['docs'],
            'indexing':entry['primaries']['indexing'],
            'sequence_numbers':[{ 'shard':shard,'primary':r.get('routing',{}).get('primary'),
                                 'seq_no':r.get('seq_no')}
                               for shard,rs in entry.get('shards',{}).items() for r in rs]}
    client.close()
    args.output.write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print('manifest',info['source']['sha256'],info.get('live_source_matches_snapshot'),flush=True)
    if args.monitor_pid:
        with args.output.with_suffix('.gpu.jsonl').open('a',buffering=1) as f:
            while True:
                try:os.kill(args.monitor_pid,0)
                except ProcessLookupError:break
                sample=subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True)
                f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'gpus':sample.strip().splitlines()})+'\n')
                time.sleep(15)


if __name__=='__main__':main()
