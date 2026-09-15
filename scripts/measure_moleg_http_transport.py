"""HTTP connection overhead control; health probes do not measure inference."""
import argparse
import http.client
import json
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--n',type=int,default=400)
    args=ap.parse_args();url=urlsplit(args.url)
    cls=http.client.HTTPSConnection if url.scheme=='https' else http.client.HTTPConnection
    def connect():return cls(url.hostname,url.port,timeout=10)
    pooled=connect();rows=[]
    for i in range(args.n+2):
        order=['new','pooled'] if i%2 else ['pooled','new']
        for variant in order:
            start=time.perf_counter();c=connect() if variant=='new' else pooled
            c.request('GET',url.path or '/');response=c.getresponse();body=response.read()
            elapsed=time.perf_counter()-start
            if variant=='new':c.close()
            if i>=2:rows.append({'probe':i-2,'variant':variant,'elapsed_s':elapsed,
                                 'status':response.status,'bytes':len(body)})
    pooled.close()
    summary={}
    for variant in ['new','pooled']:
        values=sorted(r['elapsed_s'] for r in rows if r['variant']==variant)
        summary[variant]={'n':len(values),'mean_ms':1000*statistics.fmean(values),
            'p50_ms':1000*statistics.median(values),'p95_ms':1000*values[int(.95*(len(values)-1))]}
    result={'label':args.label,'scope':'HTTP health probes only, no inference',
        'summary':summary,'rows':rows}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
