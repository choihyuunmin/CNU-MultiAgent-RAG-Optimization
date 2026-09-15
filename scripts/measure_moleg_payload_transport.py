"""Transfer actual frozen request bodies, raw vs gzip, without GPU inference.

The receiver hashes reconstructed bytes and never logs or persists the body.
This measures payload transport, not model or RAG speed.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import socket
import time
import zlib

LIMIT=2_000_000


def unpack(data, encoding):
    if len(data)>LIMIT:
        raise ValueError('wire body too large')
    if encoding=='identity':
        return data
    if encoding!='gzip':
        raise ValueError('unsupported encoding')
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
    raw=decoder.decompress(data,LIMIT+1)
    if len(raw)>LIMIT or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('invalid or oversized compressed body')
    return raw


def serve(args):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        def log_message(self,*args):
            pass
        def do_POST(self):
            if self.client_address[0] not in args.allow_client:
                self.send_error(403)
                return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=LIMIT:
                    raise ValueError('invalid length')
                self.connection.settimeout(10)
                data=self.rfile.read(length)
                started=time.perf_counter()
                raw=unpack(data,self.headers.get('Content-Encoding','identity'))
                response=json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
                                     'decode_hash_s':time.perf_counter()-started}).encode()
            except (ValueError,zlib.error):
                self.send_error(400)
                return
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(response)))
            self.end_headers()
            self.wfile.write(response)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()


def bench(args):
    import httpx
    inputs=json.loads(args.requests.read_text())
    manifest=json.loads(args.manifest.read_text())
    selected={(r['case_id'],r['stage']) for r in manifest['inputs']}
    inputs=[r for r in inputs if (r['case_id'],r['stage']) in selected]
    if len(inputs)!=len(selected):
        raise ValueError('request/manifest mismatch')
    with args.output.open('w',buffering=1) as sink, httpx.Client(timeout=10.,trust_env=False) as client:
        for encoding in ['identity','gzip']:
            data=b'transport-warmup'
            body=gzip.compress(data,compresslevel=1,mtime=0) if encoding=='gzip' else data
            response=client.post(args.url,content=body,headers={'Content-Encoding':encoding})
            response.raise_for_status()
            assert response.json()['sha256']==hashlib.sha256(data).hexdigest()
        for repeat in range(args.repeats):
            ordered=list(inputs);random.Random(6202609+repeat).shuffle(ordered)
            for i,item in enumerate(ordered):
                raw=json.dumps(item['payload'],ensure_ascii=False).encode()
                expected=hashlib.sha256(raw).hexdigest()
                variants=['identity','gzip'] if (i+repeat)%2==0 else ['gzip','identity']
                for encoding in variants:
                    start=time.perf_counter()
                    body=gzip.compress(raw,compresslevel=1,mtime=0) if encoding=='gzip' else raw
                    compressed=time.perf_counter()
                    r=client.post(args.url,content=body,headers={'Content-Type':'application/json','Content-Encoding':encoding})
                    r.raise_for_status();result=r.json()
                    elapsed=time.perf_counter()-start
                    row={'case_id':item['case_id'],'stage':item['stage'],'repeat':repeat,'variant':encoding,
                         'raw_bytes':len(raw),'sent_body_bytes':len(body),'compress_s':compressed-start,
                         'elapsed_s':elapsed,'receiver_decode_hash_s':result['decode_hash_s'],
                         'sha256':expected,'exact':result['sha256']==expected and result['bytes']==len(raw)}
                    sink.write(json.dumps(row)+'\n')
                    if not row['exact']:
                        raise RuntimeError('transport changed request bytes')
            print('transport',repeat+1,len(inputs),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--serve',action='store_true')
    ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--port',type=int,default=28131)
    ap.add_argument('--allow-client',action='append',default=[])
    ap.add_argument('--requests',type=Path)
    ap.add_argument('--manifest',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--url')
    ap.add_argument('--repeats',type=int,default=3)
    args=ap.parse_args()
    serve(args) if args.serve else bench(args)
