"""Start only the named bounded experiment service, running as the operator user."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    root=args.directory.resolve()
    if root.parent!=Path('/data/project/vllm/fine-tune/experiments') or not root.name.startswith(('workflow400-','workflow300-')):
        parser.error('exact experiment directory required')
    if (root/'campaign-status.json').exists():parser.error('do not rerun an existing campaign')
    if not (root/'operator.env').is_file():parser.error('temporary operator configuration missing')
    argv=['systemd-run','--unit='+root.name,'--description=Frozen question-count RAG experiment',
          '--property=User=axops','--property=Group=axops','--property=UMask=0077',
          '--property=RuntimeMaxSec=259200','--property=TimeoutStopSec=45',
          '--property=KillMode=control-group','--property=Restart=no',
          '--property=WorkingDirectory='+str(root),
          '--property=ExecStopPost=/usr/bin/rm -f '+str(root/'operator.env'),
          '/data/project/vllm/fine-tune/2025-moleg-rag/.venv/bin/python',
          str(root/'scripts/supervise_moleg_workflow400.py'),'--directory',str(root)]
    subprocess.run(argv,check=True)


if __name__=='__main__':main()
