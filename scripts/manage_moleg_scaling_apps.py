"""Manage only the isolated app processes registered in this experiment dir."""
import argparse
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import sys
import time
from urllib.request import urlopen

ARMS = [('baseline', 4, 'original', 28220), ('emit', 4, 'immediate', 28221),
        ('slots8', 8, 'original', 28222), ('emit8', 8, 'immediate', 28223),
        ('emit16', 16, 'immediate', 28224)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--app-root', type=Path)
    parser.add_argument('--app-env', type=Path)
    parser.add_argument('--serving-env', type=Path)
    parser.add_argument('--proxy-config', type=Path)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--arms-manifest', type=Path, help='explicit isolated workflow variants')
    args = parser.parse_args()
    root = args.directory.resolve()
    records_path = root / 'apps.json'
    if args.action == 'start':
        variants = (json.loads(args.arms_manifest.read_text()) if args.arms_manifest else
                    [dict(name=n, slots=s, emission=e, port=p) for n,s,e,p in ARMS])
        names, ports = set(), set()
        for row in variants:
            if (not re.fullmatch(r'[a-z0-9_]+', row['name']) or row['name'] in names
                    or row['port'] in ports or not 28200 <= row['port'] <= 29000
                    or row['slots'] not in [1,2,4,8,16,32]
                    or row['emission'] not in ['original','immediate']):
                parser.error('invalid or duplicate isolated variant')
            names.add(row['name']); ports.add(row['port'])
        if args.only and not set(args.only) <= names:
            parser.error('unknown variant name')
        for key in ['app_root', 'app_env', 'serving_env', 'proxy_config']:
            if getattr(args, key) is None:
                parser.error('configuration paths required for start')
        if records_path.exists():
            parser.error('registered process file already exists; do not overwrite')
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        rows = []
        for variant in variants:
            name, slots, emission, port = (variant[k] for k in ['name','slots','emission','port'])
            if args.only and name not in args.only:
                continue
            launcher = Path(__file__).resolve().with_name('launch_moleg_scaling_app.py')
            argv = [sys.executable, str(launcher), '--app-root', str(args.app_root),
                    '--app-env', str(args.app_env), '--serving-env', str(args.serving_env),
                    '--proxy-config', str(args.proxy_config), '--trace', str(root / (name + '.trace.jsonl')),
                    '--port', str(port), '--max-pipelines', str(slots), '--emission', emission]
            if variant.get('adapter_config'):
                argv += ['--adapter-config', variant['adapter_config'],
                         '--adapter-trace', str(root / (name + '.workflow.jsonl'))]
            log = root / (name + '.server.log')
            fd = os.open(log, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'w') as output:
                proc = subprocess.Popen(argv, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                        start_new_session=True, stdin=subprocess.DEVNULL)
            rows.append({'name': name, 'pid': proc.pid, 'port': port, 'slots': slots,
                         'emission': emission, 'argv': argv})
            records_path.write_text(json.dumps(rows, indent=2) + '\n')
            os.chmod(records_path, 0o600)
        print(json.dumps({'started': [{k: r[k] for k in ['name', 'pid', 'port']} for r in rows]}))
        return
    rows = json.loads(records_path.read_text())
    for row in rows:
        if args.only and row['name'] not in args.only:
            continue
        proc = Path('/proc') / str(row['pid'])
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
            matches = argv == row['argv']
        except FileNotFoundError:
            matches = False
        if args.action == 'stop':
            if matches:
                os.kill(row['pid'], signal.SIGTERM)
            print(json.dumps({'name': row['name'], 'sent_sigterm': matches}))
        else:
            state = {'name': row['name'], 'owned_process_alive': matches}
            try:
                with urlopen(f'http://127.0.0.1:{row["port"]}/__scaling_state', timeout=2) as response:
                    state['app'] = json.load(response)
            except Exception as exc:
                state['error_type'] = type(exc).__name__
            print(json.dumps(state))


if __name__ == '__main__':
    main()
