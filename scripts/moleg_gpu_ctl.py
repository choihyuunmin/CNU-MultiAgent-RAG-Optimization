"""Start/stop helper for the temporary GPU experiment on the model server.

Every production process that is stopped is relaunched later from its saved
argv, environment and working directory (captured from /proc before any
change). The experimental orchestrator instance is launched with the same
model, dtype and context length as production, on a separate port and GPU.
"""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESTORE = ROOT / 'restore'
RUN = ROOT / 'run'
LOGS = ROOT / 'logs'


def read_environ(pid):
    env = {}
    for line in (RESTORE / f'{pid}.environ').read_text().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            env[k] = v
    return env


def read_cmdline(pid):
    return [x for x in (RESTORE / f'{pid}.cmdline').read_text().split('\n') if x != '']


def port_of(argv):
    return int(argv[argv.index('--port') + 1])


def ready(port, key, timeout):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/models', headers={'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def gpu_apps():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory,gpu_uuid', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout
    return [tuple(x.strip() for x in line.split(',')) for line in out.strip().splitlines() if line.strip()]


def gpu_mem():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout
    return out.strip()


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop(pid, timeout=120):
    if not alive(pid):
        print('already gone', pid); return
    os.kill(pid, signal.SIGTERM)
    start = time.time()
    while alive(pid) and time.time() - start < timeout:
        time.sleep(2)
    if alive(pid):
        print('SIGKILL', pid); os.kill(pid, signal.SIGKILL); time.sleep(5)
    # wait for GPU memory release by any child
    for _ in range(30):
        if not any(int(p) == pid for p, _, _ in gpu_apps()):
            break
        time.sleep(2)
    print('stopped', pid, '| gpu:', gpu_mem().replace('\n', ' ; '))


def launch(argv, env, cwd, log, wait_port, key, timeout=900):
    LOGS.mkdir(exist_ok=True); RUN.mkdir(exist_ok=True)
    with open(log, 'a') as out:
        proc = subprocess.Popen(argv, env=env, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print('launched pid', proc.pid, 'log', log, flush=True)
    if not ready(wait_port, key, timeout):
        raise RuntimeError(f'port {wait_port} not ready within {timeout}s; see {log}')
    print('ready on port', wait_port, '| gpu:', gpu_mem().replace('\n', ' ; '), flush=True)
    return proc.pid


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stop'); s.add_argument('--pid', type=int, required=True)
    s = sub.add_parser('restart-saved'); s.add_argument('--pid', type=int, required=True)
    s.add_argument('--gpu', help='override CUDA_VISIBLE_DEVICES (temporary relocation)')
    s.add_argument('--name', required=True)
    s.add_argument('--util', help='temporarily override --gpu-memory-utilization (restore uses the saved value)')
    s = sub.add_parser('start-exp'); s.add_argument('--name', required=True); s.add_argument('--util', required=True)
    s.add_argument('--spec', default=''); s.add_argument('--gpu', default='1'); s.add_argument('--port', type=int, default=8100)
    s.add_argument('--pythonpath', default=''); s.add_argument('--extra', default='')
    s.add_argument('--template-pid', type=int, default=3566144)
    s.add_argument('--env', action='append', default=[], help='KEY=VALUE passed to the experimental server')
    s = sub.add_parser('stop-named'); s.add_argument('--name', required=True)
    s = sub.add_parser('start-custom'); s.add_argument('--name', required=True); s.add_argument('--argv-json', required=True)
    s.add_argument('--gpu', default='1'); s.add_argument('--pythonpath', default=''); s.add_argument('--env', action='append', default=[])
    s.add_argument('--template-pid', type=int, default=3566144)
    sub.add_parser('status')
    args = ap.parse_args()
    try:
        key = read_environ(3566144)['VLLM_API_KEY']
    except FileNotFoundError:  # environment captures are deleted after restore; fall back to the serving env file
        key = next(l.split('=', 1)[1] for l in Path('/data/project/vllm/.env').read_text().splitlines() if l.startswith('VLLM_API_KEY='))
    if args.cmd == 'status':
        print(gpu_mem()); print(gpu_apps())
        for port in [8000, 8001, 8002, 8005, 8006, 8010, 8020, 8030, 8100, 4000]:
            print(port, 'ready' if ready(port, key, 1) else 'down')
        return
    if args.cmd == 'stop':
        stop(args.pid); return
    if args.cmd == 'stop-named':
        pid = int((RUN / f'{args.name}.pid').read_text()); stop(pid); return
    if args.cmd == 'restart-saved':
        argv = read_cmdline(args.pid); env = read_environ(args.pid)
        cwd = (RESTORE / f'{args.pid}.cwd').read_text().strip()
        if args.gpu:
            env['CUDA_VISIBLE_DEVICES'] = args.gpu
        if args.util:
            argv = list(argv); argv[argv.index('--gpu-memory-utilization') + 1] = args.util
        (RUN / f'{args.name}.argv.json').write_text(json.dumps({'argv': argv, 'cuda': env.get('CUDA_VISIBLE_DEVICES')}, indent=1))
        pid = launch(argv, env, cwd, LOGS / f'{args.name}.log', port_of(argv), key)
        (RUN / f'{args.name}.pid').write_text(str(pid))
        return
    if args.cmd == 'start-custom':
        env = read_environ(args.template_pid); env['CUDA_VISIBLE_DEVICES'] = args.gpu
        if args.pythonpath:
            env['PYTHONPATH'] = args.pythonpath
        for kv in args.env:
            k, v = kv.split('=', 1); env[k] = v
        argv = json.loads(args.argv_json)
        (RUN / f'{args.name}.argv.json').write_text(json.dumps({'argv': argv, 'cuda': args.gpu, 'pythonpath': args.pythonpath, 'env': args.env}, indent=1))
        pid = launch(argv, env, '/data/project/vllm', LOGS / f'custom_{args.name}.log', port_of(argv), key)
        (RUN / f'{args.name}.pid').write_text(str(pid))
        return
    if args.cmd == 'start-exp':
        env = read_environ(args.template_pid); env['CUDA_VISIBLE_DEVICES'] = args.gpu
        if args.pythonpath:
            env['PYTHONPATH'] = args.pythonpath
        for kv in args.env:
            k, v = kv.split('=', 1); env[k] = v
        argv = ['/data/project/vllm/.venv/bin/python3', '/data/project/vllm/.venv/bin/vllm', 'serve', 'google/gemma-4-31B-it',
                '--port', str(args.port), '--gpu-memory-utilization', args.util, '--max-model-len', '32768',
                '--dtype', 'auto', '--trust-remote-code']
        if args.spec:
            argv += ['--speculative-config', args.spec]
        if args.extra:
            argv += args.extra.split()
        (RUN / f'{args.name}.argv.json').write_text(json.dumps({'argv': argv, 'cuda': args.gpu, 'pythonpath': args.pythonpath, 'env': args.env}, indent=1))
        pid = launch(argv, env, '/data/project/vllm', LOGS / f'gemma_exp_{args.name}.log', args.port, key)
        (RUN / f'{args.name}.pid').write_text(str(pid))


if __name__ == '__main__':
    main()
