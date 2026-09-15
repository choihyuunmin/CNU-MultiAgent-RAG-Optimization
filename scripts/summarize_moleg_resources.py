"""Join trial UTC windows with inference-host and Nsight device observations.

Exports allowlisted aggregates only. Nsight percentages are device-wide
interface/cycle activity, not attributable model bandwidth or measured GB/s.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

from moleg_scaling_metrics import describe

METRICS = ['GR Active [Throughput %]', 'SMs Active [Throughput %]',
           'SM Issue [Throughput %]', 'Tensor Active [Throughput %]',
           'DRAM Read Bandwidth [Throughput %]', 'DRAM Write Bandwidth [Throughput %]']


def epoch(utc):
    return datetime.fromisoformat(utc).timestamp()


def histogram_stats(histogram):
    pairs = sorted((float(v), int(n)) for v, n in histogram.items() if n)
    count = sum(n for _, n in pairs)
    if not count:
        return {'n': 0, 'mean': None, 'p95': None, 'max': None, 'fraction_ge_80': None}
    def at_rank(rank):
        seen = 0
        for value, frequency in pairs:
            seen += frequency
            if rank < seen:
                return value
        return pairs[-1][0]
    position = (count - 1) * .95
    lower, upper = math.floor(position), math.ceil(position)
    return {'n': count, 'mean': sum(v*n for v, n in pairs)/count,
            'p95': at_rank(lower) + (at_rank(upper)-at_rank(lower))*(position-lower),
            'max': pairs[-1][0], 'fraction_ge_80': sum(n for v,n in pairs if v >= 80)/count}


def psi_total(row, category):
    for line in row.get('memory_psi', []):
        if line.startswith(category + ' '):
            return int(dict(x.split('=') for x in line.split()[1:])['total'])
    return None


def host_summary(rows):
    if not rows:
        return {'n': 0}
    before, after = rows[0], rows[-1]
    counters = {}
    for key in ['pswpin', 'pswpout', 'pgmajfault', 'oom_kill']:
        a, b = before['vm_counters'].get(key), after['vm_counters'].get(key)
        counters[key] = b-a if a is not None and b is not None and b >= a else None
    pressure = {}
    for key in ['some', 'full']:
        a, b = psi_total(before, key), psi_total(after, key)
        pressure[key + '_stall_s'] = (b-a)/1e6 if a is not None and b is not None and b >= a else None
    gpus = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for gpu in row.get('gpus', []):
            for key in ['memory.used', 'utilization.gpu', 'utilization.memory', 'power.draw', 'temperature.gpu']:
                try:
                    gpus[gpu['index']][key].append(float(gpu[key]))
                except (ValueError, KeyError):
                    pass
    return {'n': len(rows), 'first_utc': before['utc'], 'last_utc': after['utc'],
            'minimum_mem_available_gib': min(r['host_memory_kib']['MemAvailable'] for r in rows)/1024**2,
            'swap_total_kib': max(r['host_memory_kib']['SwapTotal'] for r in rows),
            'vm_counter_deltas': counters, 'memory_psi': pressure,
            'gpus': {gpu: {k: describe(v) for k,v in values.items()} for gpu,values in gpus.items()}}


def aggregate(directory, host_path, nsys_path=None):
    trials = json.loads((directory/'summary.json').read_text())['trials']
    hosts = [json.loads(x) for x in host_path.open()]
    app_rows = [json.loads(x) for x in (directory/'app-state.jsonl').open()]
    database = sqlite3.connect(f'file:{nsys_path.resolve()}?mode=ro', uri=True) if nsys_path else None
    epoch_ns = None
    if database:
        epoch_ns = database.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
        metric_names = {(t,m):name for t,m,name in database.execute('select typeId,metricId,metricName from TARGET_INFO_GPU_METRICS') if name in METRICS}
        bounds = database.execute('select min(timestamp),max(timestamp) from GPU_METRICS').fetchone()
    output = {'scope': 'time-aligned whole-device and inference-host observations, not causal per-model attribution',
              'units': 'NVML memory utilization is activity time; Nsight DRAM fields are reported percentages, not GB/s',
              'nsys_available': database is not None, 'trials': []}
    for trial in trials:
        begin, end = epoch(trial['started_utc']), epoch(trial['ended_utc'])
        selected = [r for r in hosts if begin <= epoch(r['utc']) <= end]
        apps = [r['state'] for r in app_rows if r['trial'] == trial['trial'] and 'state' in r]
        row = {k: trial[k] for k in ['trial', 'policy', 'users', 'repeat', 'started_utc', 'ended_utc']}
        row['host'] = host_summary(selected)
        row['app'] = {'n': len(apps), 'active': describe([r['active'] for r in apps]),
                      'inner_queued': describe([r['queued'] for r in apps]),
                      'maxrss_kib': max((r['process_maxrss_kib'] for r in apps), default=None),
                      'pipeline_limits_observed': sorted({r['max_pipelines'] for r in apps}),
                      'http_limits_observed': sorted({r['max_http'] for r in apps if 'max_http' in r})}
        if database:
            start_ns, end_ns = int(begin*1e9)-epoch_ns, int(end*1e9)-epoch_ns
            stats = defaultdict(dict)
            query = ('select typeId,metricId,value,count(*) from GPU_METRICS '
                     'where timestamp>=? and timestamp<=? group by typeId,metricId,value')
            for type_id, metric_id, value, count in database.execute(query, (start_ns, end_ns)):
                name = metric_names.get((type_id, metric_id))
                if name:
                    stats[(type_id & 0xFF, name)][value] = count
            row['nsys_gpus'] = defaultdict(dict)
            for (gpu, name), histogram in stats.items():
                row['nsys_gpus'][str(gpu)][name] = histogram_stats(histogram)
            row['nsys_window_covered'] = bool(bounds[0] is not None and bounds[0] <= start_ns and bounds[1] >= end_ns)
            row['nsys_coverage'] = {}
            for type_id in sorted({t for t,m in metric_names}):
                metric_id = min(m for t,m in metric_names if t == type_id)
                stamps = [x[0] for x in database.execute(
                    'select timestamp from GPU_METRICS where typeId=? and metricId=? '
                    'and timestamp>=? and timestamp<=? order by timestamp',
                    (type_id, metric_id, start_ns, end_ns))]
                gaps = [(b-a)/1e9 for a,b in zip(stamps, stamps[1:])]
                row['nsys_coverage'][str(type_id & 0xFF)] = {
                    'samples': len(stamps), 'trial_duration_s': end-begin,
                    'first_sample_offset_s': (stamps[0]-start_ns)/1e9 if stamps else None,
                    'last_sample_before_end_s': (end_ns-stamps[-1])/1e9 if stamps else None,
                    'sample_span_fraction': (stamps[-1]-stamps[0])/(end_ns-start_ns) if len(stamps)>1 else 0.,
                    'max_gap_s': max(gaps, default=None),
                    'gaps_gt_30ms': sum(g > .03 for g in gaps),
                    'note': '30ms is 3 sample periods at 100Hz; endpoint coverage does not prove absence of internal gaps'}
        output['trials'].append(row)
    if database:
        output['nsys_diagnostic_event_count'] = database.execute('select count(*) from DIAGNOSTIC_EVENT').fetchone()[0]
        columns = {r[1] for r in database.execute('pragma table_info(DIAGNOSTIC_EVENT)')}
        if 'text' in columns:
            messages = [r[0] for r in database.execute('select text from DIAGNOSTIC_EVENT')]
            output['nsys_diagnostics'] = {
                'unsupported_metric_messages': sum('Skipping metric' in m for m in messages),
                'loss_or_overflow_messages': sum(any(s in m.lower() for s in ['overflow', 'data loss', 'lost', 'missing']) for m in messages),
                'scope': 'message classification; not proof that all requested metrics are supported'}
        database.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--host', type=Path, required=True)
    parser.add_argument('--nsys-sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.directory, args.host, args.nsys_sqlite)
    with args.output.open('x') as sink:
        json.dump(result, sink, indent=2)
        sink.write('\n')
    print(json.dumps({'trials':len(result['trials']), 'nsys_available':result['nsys_available']}))


if __name__ == '__main__':
    main()
