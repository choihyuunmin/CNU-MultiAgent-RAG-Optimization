"""Minimal L7 load balancer for the experiment: least-outstanding-requests
across several OpenAI-compatible upstreams serving the same model name.
No caching, retries, prompt rewriting or admission control. Records which
upstream served each request so per-replica behaviour can be analysed."""
import argparse, asyncio, json, time
from contextlib import asynccontextmanager
from pathlib import Path


def create_app(upstreams, trace_path, allowed_ips, policy='least_outstanding', alpha=0.3):
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    state = {'outstanding': {u: 0 for u in upstreams}, 'served': {u: 0 for u in upstreams},
             'ewma_s': {u: 1.0 for u in upstreams}, 'policy': policy}
    clients = {}
    trace = trace_path.open('a', buffering=1)

    @asynccontextmanager
    async def lifespan(app):
        for u in upstreams:
            clients[u] = httpx.AsyncClient(base_url=u, timeout=httpx.Timeout(300., connect=5.), trust_env=False,
                                           limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))
        yield
        for c in clients.values():
            await c.aclose()
        trace.close()

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    async def health():
        return state

    @app.post('/v1/chat/completions')
    async def completion(request: Request):
        if request.client.host not in allowed_ips:
            return Response(status_code=403)
        body = await request.body()
        rid = request.headers.get('x-experiment-id', '')
        try:
            wants_stream = bool(json.loads(body).get('stream'))
        except ValueError:
            wants_stream = False
        # least outstanding; ties broken by fewest served (round-robin-like)
        if policy == 'ewma_cost':
            # expected completion cost: queued work ahead times the replica's recent service time
            upstream = min(upstreams, key=lambda u: ((state['outstanding'][u] + 1) * state['ewma_s'][u], state['served'][u]))
        else:
            upstream = min(upstreams, key=lambda u: (state['outstanding'][u], state['served'][u]))
        state['outstanding'][upstream] += 1; state['served'][upstream] += 1
        start = time.perf_counter()
        if wants_stream:
            async def relay():
                try:
                    async with clients[upstream].stream('POST', '/v1/chat/completions', content=body,
                            headers={'Authorization': request.headers.get('authorization', ''),
                                     'Content-Type': 'application/json'}) as r:
                        async for chunk in r.aiter_raw():
                            yield chunk
                finally:
                    elapsed = time.perf_counter() - start
                    state['outstanding'][upstream] -= 1
                    state['ewma_s'][upstream] = (1 - alpha) * state['ewma_s'][upstream] + alpha * elapsed
                    trace.write(json.dumps({'rid': rid, 'upstream': upstream, 'stream': True, 'upstream_s': elapsed}) + '\n')
            return StreamingResponse(relay(), media_type='text/event-stream', headers={'X-Experiment-Upstream': upstream})
        try:
            r = await clients[upstream].post('/v1/chat/completions', content=body,
                                             headers={'Authorization': request.headers.get('authorization', ''),
                                                      'Content-Type': 'application/json'})
            elapsed = time.perf_counter() - start
            state['ewma_s'][upstream] = (1 - alpha) * state['ewma_s'][upstream] + alpha * elapsed
            trace.write(json.dumps({'rid': rid, 'upstream': upstream, 'status': r.status_code, 'upstream_s': elapsed,
                                    'outstanding_after': dict(state['outstanding'])}) + '\n')
            return Response(content=r.content, status_code=r.status_code, media_type='application/json',
                            headers={'X-Experiment-Upstream': upstream, 'X-Experiment-Upstream-S': f'{elapsed:.4f}'})
        finally:
            state['outstanding'][upstream] -= 1
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=28140)
    ap.add_argument('--upstream', action='append', required=True)
    ap.add_argument('--allow-client', action='append', default=['127.0.0.1'])
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--policy', choices=['least_outstanding', 'ewma_cost'], default='least_outstanding')
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(a.upstream, a.trace, set(a.allow_client), a.policy), host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
