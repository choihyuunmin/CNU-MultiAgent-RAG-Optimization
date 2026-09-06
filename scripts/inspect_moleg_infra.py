"""Read-only, allow-listed inventory; never prints process secrets or pod env."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def command(*args):
    return subprocess.check_output(args, text=True)


def gpu_inventory():
    flags = {'--model', '--port', '--host', '--max-num-seqs', '--max-model-len',
             '--max-num-batched-tokens', '--gpu-memory-utilization', '--dtype',
             '--tensor-parallel-size', '--scheduling-policy', '--speculative-config',
             '--served-model-name', '--reasoning-parser', '--kv-cache-dtype'}
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry/'cmdline').read_bytes().decode().split('\0')
            if not any('vllm' in a.lower() for a in argv[:4]) or 'serve' not in argv:
                continue
            pos = argv.index('serve')
            selected = {'pid': int(entry.name), 'model': argv[pos+1], 'arguments': {}}
            for i, arg in enumerate(argv):
                if arg in flags and i+1 < len(argv):
                    selected['arguments'][arg] = argv[i+1]
            env = dict(x.split('=', 1) for x in (entry/'environ').read_bytes().decode().split('\0') if '=' in x)
            selected['gpu'] = env.get('CUDA_VISIBLE_DEVICES')
            rows.append(selected)
        except (OSError, ValueError, IndexError):
            continue
    return {'utc': datetime.now(timezone.utc).isoformat(), 'host': command('hostname').strip(),
            'gpus': command('nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader').splitlines(), 'servers': rows}


def cluster_inventory():
    out = {'utc': datetime.now(timezone.utc).isoformat(), 'deployments': [], 'pods': [], 'services': []}
    for resource, names in [('deployments', ['moleg-search-prod', 'moleg-rag']),
                             ('services', ['moleg-search-prod', 'moleg-rag'])]:
        data = json.loads(command('kubectl', '-n', 'moleg', 'get', resource, *names, '-o', 'json'))
        for item in data['items']:
            row = {'name': item['metadata']['name'], 'uid': item['metadata']['uid'],
                   'generation': item['metadata'].get('generation')}
            if resource == 'deployments':
                spec = item['spec']['template']['spec']
                row.update(replicas=item['spec'].get('replicas'),
                           template_sha256=hashlib.sha256(json.dumps(item['spec']['template'], sort_keys=True).encode()).hexdigest(),
                           containers=[{'name': c['name'], 'image': c['image'],
                                        'ports': c.get('ports'), 'resources': c.get('resources')} for c in spec['containers']])
            else:
                row.update(cluster_ip=item['spec'].get('clusterIP'), ports=item['spec'].get('ports'))
            out[resource].append(row)
    data = json.loads(command('kubectl', '-n', 'moleg', 'get', 'pods', '-o', 'json'))
    for item in data['items']:
        if not item['metadata']['name'].startswith(('moleg-search-prod-', 'moleg-rag-')):
            continue
        out['pods'].append({'name': item['metadata']['name'], 'uid': item['metadata']['uid'],
                            'node': item['spec'].get('nodeName'), 'ip': item['status'].get('podIP'),
                            'phase': item['status'].get('phase'),
                            'containers': [{'name': c['name'], 'image_id': c.get('imageID'),
                                            'restart_count': c.get('restartCount'), 'ready': c.get('ready')}
                                           for c in item['status'].get('containerStatuses', [])]})
    return out


def runtime_inventory():
    # Read only: hashes and allow-listed endpoints, never full environment/config.
    code = '''import hashlib,json,os,sys
from pathlib import Path
from urllib.parse import urlsplit
out={'cwd':os.getcwd(),'source':{},'interpreters':[p for p in ['/opt/venv/bin/python','/app/.venv/bin/python','/venv/bin/python','/usr/local/bin/python'] if Path(p).exists()]}
for rel in ['core/agent_orchestrator/orchestrator.py','agent/validation_agent.py','infra/llm/client.py']:
 p=Path('/app/src')/rel
 out['source'][rel]=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
sys.path.insert(0,'/app/src')
try:
 from config import settings
 out['settings']={}
 for name in ['MASTER_MODEL','PREPARATION_LLM_ROLE','RETRIEVAL_TOOL_AGENT_MODEL','SYNTHESIS_TOOL_AGENT_MODEL','MAX_CONCURRENT_GENERATE_REQUESTS','LLM_MAX_RETRIES']:
  out['settings'][name]=getattr(settings,name,None)
 for name in ['MASTER_API_BASE','LITELLM_API_BASE','LITELLM_BASE_URL','RETRIEVAL_TOOL_AGENT_API_BASE']:
  val=getattr(settings,name,None)
  if val:
   url=urlsplit(str(val));out['settings'][name]={'scheme':url.scheme,'host':url.hostname,'port':url.port,'path':url.path}
except Exception as e:out['settings_error']=type(e).__name__
env=dict(os.environ)
if Path('/app/.env').exists():
 for line in Path('/app/.env').read_text().splitlines():
  if not line.strip() or line.lstrip().startswith('#') or '=' not in line:continue
  key,val=line.split('=',1);env.setdefault(key.strip(),val.strip().strip(chr(34)).strip(chr(39)))
out['allowlisted_environment']={}
for name in ['MAX_CONCURRENT_GENERATE_REQUESTS','LITELLM_MODEL_ORCHESTRATOR','PREPARATION_LLM_ROLE','LITELLM_PROXY_HOST','LITELLM_PROXY_PORT','LLM_MAX_RETRIES']:
 out['allowlisted_environment'][name]=env.get(name)
for name in ['LITELLM_PROXY_V1_BASE']:
 val=env.get(name)
 if val:
  url=urlsplit(val);out['allowlisted_environment'][name]={'scheme':url.scheme,'host':url.hostname,'port':url.port,'path':url.path}
print(json.dumps(out))'''
    return json.loads(command('kubectl','-n','moleg','exec','deployment/moleg-rag','--','/app/.venv/bin/python','-c',code))


def network_inventory(peer):
    import ipaddress
    ipaddress.ip_address(peer)
    route=json.loads(command('ip','-j','route','get',peer))[0]
    interface=route['dev']
    base=Path('/sys/class/net')/interface
    return {'utc':datetime.now(timezone.utc).isoformat(),'peer':peer,
            'interface':interface,'link_speed_mbps':int((base/'speed').read_text()),
            'duplex':(base/'duplex').read_text().strip(),
            'scope':'negotiated route-interface link speed, not measured usable throughput'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', action='store_true')
    ap.add_argument('--runtime', action='store_true')
    ap.add_argument('--network-peer')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    value = network_inventory(args.network_peer) if args.network_peer else runtime_inventory() if args.runtime else cluster_inventory() if args.cluster else gpu_inventory()
    text = json.dumps(value, indent=2, ensure_ascii=False)+'\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text)
