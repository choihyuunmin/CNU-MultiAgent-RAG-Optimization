"""Operator-only pod snapshot. Credentials never go to stdout or source archives."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args = p.parse_args()
    root = args.directory
    root.mkdir(mode=0o700,exist_ok=False)
    base = ['kubectl','-n','moleg','exec','deployment/moleg-rag','--','/app/.venv/bin/python','-c']
    # A kubectl exec child does not inherit dotenv values loaded inside the
    # already-running Python process. Reproduce the application's load order.
    code = "import os,json;from dotenv import load_dotenv;load_dotenv('/app/.env');print(json.dumps(dict(os.environ),ensure_ascii=False))"
    values = json.loads(subprocess.check_output(base+[code]))
    fd = os.open(root/'operator.env',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as sink:
        for k,v in values.items():
            if k.replace('_','').isalnum() and not k[0].isdigit():
                sink.write(k+'='+json.dumps(v,ensure_ascii=False)+'\n')
    code = """import io,sys,tarfile
from pathlib import Path
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
 for path in sorted(Path('/app/src').rglob('*.py')):
  if not path.is_symlink():tar.add(path,arcname=str(path.relative_to('/app')),recursive=False)
sys.stdout.buffer.write(buf.getvalue())"""
    payload = subprocess.check_output(base+[code])
    (root/'source.tar.gz').write_bytes(payload)
    for path in root.iterdir():
        path.chmod(0o600)
    if os.environ.get('SUDO_UID'):
        for path in [root,*root.iterdir()]:
            os.chown(path,int(os.environ['SUDO_UID']),int(os.environ['SUDO_GID']))
    print(json.dumps({'snapshot_created':True,'source_bytes':len(payload),'environment_values_exported':len(values)}))


if __name__=='__main__':main()
