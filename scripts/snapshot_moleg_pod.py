"""Read Python source from the deployed RAG pod into an operator-private archive.

No exec imports of the application, no config/secret files, no pod mutations.
Requires existing kubectl read/exec permission. Run on the operator's app host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    code = '''import io,sys,tarfile
from pathlib import Path
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
 for p in sorted(Path('/app/src').rglob('*.py')):
  if p.is_symlink():continue
  tar.add(p,arcname=str(p.relative_to('/app')),recursive=False)
sys.stdout.buffer.write(buf.getvalue())'''
    data = subprocess.check_output(['kubectl','-n','moleg','exec','deployment/moleg-rag','--','python','-c',code])
    args.output.write_bytes(data)
    args.output.chmod(0o600)
    if os.environ.get('SUDO_UID'):
        os.chown(args.output,int(os.environ['SUDO_UID']),int(os.environ['SUDO_GID']))
    print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}))


if __name__ == '__main__':
    main()
