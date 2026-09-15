"""Network-path comparison from the UI host (.52) to the RAG backend pod on the
GPU host (.53): ingress-nginx, NodePort (kube-proxy), ClusterIP, and the Calico
VXLAN pod address. Two probes: (1) no-inference GETs (pure path overhead, new vs
pooled connection); (2) real streaming requests, recording first SSE event,
first visible token, chunk inter-arrival gaps (buffering signature) and total.
Standard library only so it runs on the UI host without extra packages."""
import argparse, http.client, json, random, statistics, time, uuid
from pathlib import Path
from urllib.parse import urlsplit


def conn(url):
    u = urlsplit(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=300), u


def probe_get(url, n):
    rows = []
    pooled, u = conn(url)
    for i in range(n + 2):
        for variant in (['new', 'pooled'] if i % 2 else ['pooled', 'new']):
            c = pooled if variant == 'pooled' else conn(url)[0]
            start = time.perf_counter()
            c.request('GET', u.path or '/', headers={'Host': u.hostname})
            r = c.getresponse(); r.read()
            el = time.perf_counter() - start
            if variant == 'new':
                c.close()
            if i >= 2:
                rows.append({'variant': variant, 'ms': 1000 * el, 'status': r.status})
    pooled.close()
    return rows


def stream_once(url, question):
    c, u = conn(url)
    body = json.dumps({'prompt': question, 'session_id': f'netpath-{uuid.uuid4().hex[:10]}'}).encode()
    start = time.perf_counter()
    c.request('POST', u.path, body=body, headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream', 'Host': u.hostname})
    r = c.getresponse()
    first_byte = first_event = first_token = done = None
    arrivals = []; events = 0; chars = 0; buf = b''
    while True:
        chunk = r.read1(65536) if hasattr(r, 'read1') else r.read(65536)
        now = time.perf_counter() - start
        if not chunk:
            break
        if first_byte is None:
            first_byte = now
        arrivals.append(now)
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            if not line.startswith(b'data:'):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            events += 1
            if first_event is None:
                first_event = now
            st = ev.get('stage')
            if st == 'token' and ev.get('delta'):
                chars += len(ev['delta'])
                if first_token is None:
                    first_token = now
            elif st == 'done':
                done = now
    total = time.perf_counter() - start
    c.close()
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {'status': r.status, 'first_byte_s': first_byte, 'first_event_s': first_event, 'ttft_s': first_token, 'done_s': done,
            'total_s': total, 'events': events, 'read_chunks': len(arrivals), 'streamed_chars': chars,
            'max_gap_s': max(gaps) if gaps else None, 'p50_gap_ms': 1000 * statistics.median(gaps) if gaps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', action='append', required=True, help='name=base_url')
    ap.add_argument('--cases', type=Path)
    ap.add_argument('--questions', type=int, default=12)
    ap.add_argument('--get-n', type=int, default=200)
    ap.add_argument('--get-path', default='/docs')
    ap.add_argument('--stream-route', default='/api/generate/stream')
    ap.add_argument('--skip-stream', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    paths = dict(p.split('=', 1) for p in a.path)
    out = {'paths': paths, 'get': {}, 'stream': []}
    for name, base in paths.items():
        rows = probe_get(base.rstrip('/') + a.get_path, a.get_n)
        out['get'][name] = {v: {'n': len(x), 'mean_ms': statistics.fmean(x), 'p50_ms': statistics.median(x),
                                'p95_ms': sorted(x)[int(.95 * (len(x) - 1))]} for v in ['new', 'pooled']
                            if (x := [r['ms'] for r in rows if r['variant'] == v])}
        print(name, json.dumps(out['get'][name]), flush=True)
    if not a.skip_stream and a.cases:
        cases = json.loads(a.cases.read_text())
        random.Random(20260906).shuffle(cases)
        cases = cases[:a.questions]
        names = list(paths)
        for i, case in enumerate(cases):
            order = names[i % len(names):] + names[:i % len(names)]
            for name in order:
                row = {'case_id': case['case_id'], 'path': name}
                try:
                    row.update(stream_once(paths[name].rstrip('/') + a.stream_route, case['question']))
                except Exception as e:
                    row.update(error=type(e).__name__)
                out['stream'].append(row)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
