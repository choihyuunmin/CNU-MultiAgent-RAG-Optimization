"""Collect the experiment's audited public artifacts when its service terminates.

Uses an existing SSH control socket. No passwords, retries of model requests,
service restarts, publication to Git, or changes to the frozen experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time

from export_moleg_reasoning import public_safe


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def verify_publication(directory):
    audit = json.loads((directory/'publication-audit.json').read_text())
    expected = set(audit['exported_sha256']) | {'publication-audit.json'}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError('unexpected publication files')
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError('nonregular publication file')
        if path.name in audit['exported_sha256']:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != audit['exported_sha256'][path.name]:
                raise ValueError('publication checksum mismatch')
        if path.suffix == '.json':
            public_safe(json.loads(path.read_text()))
        elif path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                public_safe(json.loads(line))
        else:
            raise ValueError('unexpected artifact format')
    return audit


def report(directory, audit):
    analysis = json.loads((directory/'workflow-analysis.json').read_text())
    lines = ['# 호출 제한 없는 CoT 비교 실험 결과', '',
        f"실제 요청 {audit['completed_requests']:,}건, SSE 완료 {audit['sse_success']:,}건, "
        f"내부 오류 요청 {audit['traced_internal_error_requests']:,}건.",
        f"자동 답변 평가 {audit['judge_records']:,}건 중 유효 {audit['judge_valid']:,}건.", '',
        '두 조건 모두 호출 대기열 없이 실행했다. 아래 지연은 실패 기한까지의 시간을 포함한 관측값이다.', '',
        '| 사용자 | 기준 평균(초) | 개선 조건 평균(초) | 단축률 | 95% 구간 |',
        '|---:|---:|---:|---:|---|']
    for item in analysis['adapter_comparisons']:
        value = item['latency']
        ci = value['reduction_95ci_pct']
        lines.append(f"| {item['users']} | {value['baseline_mean']:.2f} | {value['candidate_mean']:.2f} | "
                     f"{value['reduction_pct']:.2f}% | {ci[0]:.2f} ~ {ci[1]:.2f}% |")
    lines += ['', '성공/실패별 지연과 성공률 차이는 `workflow-analysis.json`, 답변 평가 결과는 '
              '`judge-u*-summary.json`에 포함된다. 실패의 조기 종료를 정상 답변의 가속으로 해석하지 않는다.', '',
              '기존 300문항을 이용한 회귀 실험이며 독립 holdout이 아니다. 자동 판정은 전문가 정확도 '
              '동등성의 증거가 아니다. 운영 배포는 수행하지 않았다.', '',
              '첫 사전 점검에서 발견한 LiteLLM 옵션 거부와 fallback은 보존했으며, 호환성 수정 후 '
              '16요청의 응답·내부 오류 검사를 통과한 뒤 본 실험을 진행했다.', '']
    (directory/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--remote-run', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    run = Path(args.remote_run)
    if run.parent != Path('/data/project/vllm/fine-tune/experiments') or not re.fullmatch(r'reasoning-[A-Za-z0-9.-]+', run.name):
        parser.error('expected an owned reasoning experiment directory')
    if args.output.exists():
        parser.error('use a new collection destination')
    args.status.parent.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh','-S',str(args.control),'-o','BatchMode=yes','-o','ConnectTimeout=8',args.host]
    code = ("import json,subprocess;from pathlib import Path;"
            f"root=Path({str(run)!r});"
            f"active=subprocess.check_output(['systemctl','show',{run.name+'.service'!r},'-p','ActiveState','--value'],text=True).strip();"
            "print(json.dumps(dict(status=json.loads((root/'campaign-status.json').read_text()),"
            "active=active,publication_ready=(root/'publication/publication-audit.json').exists())))")
    deadline = time.monotonic() + 38*3600
    while time.monotonic() < deadline:
        try:
            response = subprocess.run(ssh + ['python3 -c '+shlex.quote(code)], check=True,
                                      capture_output=True, text=True, timeout=25)
            state = json.loads(response.stdout)
            public_safe(state)
            save(args.status, dict(updated_utc=utc(), **state))
            if state['active'] not in {'active','activating','deactivating'}:
                if not state['publication_ready']:
                    save(args.status, dict(updated_utc=utc(), **state, collection='terminal_without_audited_publication'))
                    return 1
                with tempfile.TemporaryDirectory(prefix='reasoning-publication-', dir=args.status.parent) as temp:
                    subprocess.run(['scp','-r','-o','ControlPath='+str(args.control),'-o','BatchMode=yes',
                                    args.host+':'+str(run/'publication'),temp], check=True,
                                   capture_output=True, timeout=120)
                    source = Path(temp)/'publication'
                    audit = verify_publication(source)
                    report(source, audit)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source, args.output)
                save(args.status, dict(updated_utc=utc(), **state, collection='verified_and_collected',
                                       output=str(args.output.resolve())))
                return 0
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            save(args.status, dict(updated_utc=utc(), collection='waiting_after_error', error_type=type(exc).__name__))
        time.sleep(30)
    save(args.status, dict(updated_utc=utc(), collection='collector_deadline_exceeded'))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
