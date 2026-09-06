"""Launch three isolated app processes; model servers are never restarted."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
env=dict(os.environ)
env['MOLEG_RAG_ROOT']=str(root/'app')
env['MOLEG_CAPTURE_PATH']=str(root/'results'/'generation_requests.jsonl')
records=[]
for profile,port in [('baseline',28110),('speed',28111),('balanced',28112)]:
    current={**env,'MOLEG_STUDY_PROFILE':profile,'MOLEG_STUDY_PORT':str(port),
             'MOLEG_TRACE_PATH':str(root/'results'/f'{profile}.trace.jsonl')}
    with (root/f'{profile}.server.log').open('a') as output:
        proc=subprocess.Popen([sys.executable,str(root/'moleg_paper_runtime.py')],
            cwd=root,env=current,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    records.append({'profile':profile,'port':port,'pid':proc.pid})
(root/'servers.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records))
