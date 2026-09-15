"""Experimental fixed-upstream gateway. No prompt rewrites, cache, or retries.

Independent CPU process only. Does not reconfigure Kubernetes or a model server.
The operator must explicitly configure a fixed upstream and allowed client IPs.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time


@dataclass
class Ticket:
    stage: str
    created: float
    future: asyncio.Future
    granted: bool = False


class Admission:
    def __init__(self, limit: int, policy='fifo', max_wait=6., short_quota=3):
        if limit < 1 or policy not in ('fifo', 'fair_short'):
            raise ValueError('invalid admission configuration')
        self.limit, self.policy, self.max_wait, self.short_quota = limit, policy, max_wait, short_quota
        self.active, self.short_run = 0, 0
        self.queue = []

    def _pick(self):
        oldest = self.queue[0]
        if self.policy == 'fifo' or time.perf_counter()-oldest.created >= self.max_wait:
            return oldest
        short = next((x for x in self.queue if x.stage == 'classify'), None)
        long = next((x for x in self.queue if x.stage != 'classify'), None)
        if short is not None and (long is None or self.short_run < self.short_quota):
            return short
        return long or oldest

    def _drain(self):
        self.queue[:] = [t for t in self.queue if not t.future.cancelled()]
        while self.active < self.limit and self.queue:
            ticket = self._pick()
            self.queue.remove(ticket)
            ticket.granted = True
            self.active += 1
            self.short_run = self.short_run+1 if ticket.stage == 'classify' else 0
            ticket.future.set_result(None)

    @asynccontextmanager
    async def slot(self, stage):
        if len(self.queue) >= 128:
            raise OverflowError('experiment queue is full')
        ticket = Ticket(stage, time.perf_counter(), asyncio.get_running_loop().create_future())
        self.queue.append(ticket)
        self._drain()
        try:
            await ticket.future
            yield time.perf_counter()-ticket.created
        finally:
            if ticket.granted:
                self.active -= 1
            elif ticket in self.queue:
                self.queue.remove(ticket)
            self._drain()


def create_app(upstream, stage_map, allowed_ips, trace_path):
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    policies = {'pass': Admission(128), 'fifo4': Admission(4), 'fifo8': Admission(8),
                'fair4': Admission(4, 'fair_short'), 'fair8': Admission(8, 'fair_short')}
    client = None
    sink = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal client, sink
        client = httpx.AsyncClient(timeout=httpx.Timeout(90., connect=5.),
                                   limits=httpx.Limits(max_connections=128, max_keepalive_connections=32), trust_env=False)
        sink = trace_path.open('a', buffering=1)
        yield
        await client.aclose()
        sink.close()

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    async def health():
        return {'ok': True, 'queues': {name: {'active': s.active, 'waiting':len(s.queue)} for name,s in policies.items()}}

    @app.post('/{policy}/v1/chat/completions')
    async def completion(policy: str, request: Request):
        if request.client.host not in allowed_ips:
            return JSONResponse({'error': 'experiment client not allowed'}, status_code=403)
        if policy not in policies:
            return JSONResponse({'error': 'unknown policy'}, status_code=404)
        auth = request.headers.get('authorization', '')
        if not auth.startswith('Bearer '):
            return JSONResponse({'error': 'authentication required'}, status_code=401)
        started = time.perf_counter()
        raw = await request.body()
        if len(raw) > 2_000_000:
            return JSONResponse({'error': 'request too large'}, status_code=413)
        try:
            body = json.loads(raw)
            if body.get('stream'):
                return JSONResponse({'error': 'this experiment covers nonstream analysis only'}, status_code=400)
            system = body['messages'][0]['content']
            signature = hashlib.sha256(json.dumps(system, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            stage = stage_map.get(signature, 'unknown')
        except (ValueError, KeyError, IndexError, TypeError):
            return JSONResponse({'error': 'invalid body'}, status_code=400)
        row = {'request_id': request.headers.get('x-experiment-id'), 'policy':policy, 'stage':stage,
               'input_sha256': hashlib.sha256(raw).hexdigest(), 'input_bytes':len(raw),
               'forward_sha256': hashlib.sha256(raw).hexdigest()}
        try:
            async with policies[policy].slot(stage) as waited:
                row['gateway_wait_s'] = waited
                t = time.perf_counter()
                response = await client.post(upstream.rstrip('/')+'/v1/chat/completions', content=raw,
                                             headers={'Authorization':auth, 'Content-Type':'application/json'})
                row.update(upstream_s=time.perf_counter()-t, status=response.status_code,
                           response_sha256=hashlib.sha256(response.content).hexdigest())
                return Response(response.content, status_code=response.status_code,
                                media_type=response.headers.get('content-type', 'application/json'),
                                headers={'X-Experiment-Queue-S':str(waited),
                                         'X-Experiment-Upstream-S':str(row['upstream_s'])})
        except OverflowError:
            row['status'] = 503
            return JSONResponse({'error': 'experiment queue full'}, status_code=503)
        except httpx.HTTPError as exc:
            row.update(status=502, error_type=type(exc).__name__)
            return JSONResponse({'error':'upstream request failed', 'error_type':type(exc).__name__}, status_code=502)
        finally:
            row['elapsed_s'] = time.perf_counter()-started
            sink.write(json.dumps(row)+'\n')

    return app


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True)
    ap.add_argument('--port', type=int, default=28130)
    ap.add_argument('--upstream', required=True)
    ap.add_argument('--stage-map', type=Path, required=True)
    ap.add_argument('--allow-client', action='append', required=True)
    ap.add_argument('--trace', type=Path, required=True)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.upstream, json.loads(args.stage_map.read_text()), set(args.allow_client), args.trace),
                host=args.host, port=args.port, access_log=False, log_level='warning', timeout_keep_alive=30)
