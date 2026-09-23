import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LEVELS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "20,50,100").split(",")]
ARMS = ["original", "direct", "reduce", "combined"]
ORCH, WORKER = "receiver-1", "receiver-2"
OUT.mkdir(parents=True, exist_ok=True)


def cts(s):
    return datetime.fromisoformat(s).timestamp()


status = json.loads((ROOT / "main" / "status.json").read_text())
cells = {(c["level"], c["round"], c["method"]): c for c in status["completed"]}


def load_cell(level, rnd, arm):
    cell = ROOT / "main" / f"level-{level:03d}" / f"round-{rnd}"
    metrics = defaultdict(list)
    for line in (cell / f"{arm}-metrics.jsonl").open():
        d = json.loads(line)
        if d.get("metrics"):
            metrics[d["resource"]].append((d["time"], d["metrics"]))
    req_path = cell / "responses" / "sweep-trial" / arm / "client_requests.jsonl"
    reqs = []
    if req_path.exists():
        for line in req_path.open(errors="replace"):
            if line.strip():
                r = json.loads(line)
                reqs.append((cts(r["started_at"]), cts(r["completed_at"])))
    return metrics, reqs


def series(level, rnd, arm):
    """1-second bins from the first client request start; engine samples averaged per bin,
    cumulative counters expressed as deltas from the first sample at or after t0."""
    metrics, reqs = load_cell(level, rnd, arm)
    orch = metrics[ORCH]
    if reqs:
        t0 = min(s for s, _ in reqs)
        t_end = max(e for _, e in reqs)
    else:  # no client records for this level: use the first engine activity
        active = [t for t, m in orch if m.get("vllm:num_requests_running", 0) > 0]
        t0, t_end = (active[0], active[-1]) if active else (orch[0][0], orch[-1][0])
    n = int(np.ceil(t_end - t0)) + 1
    rows = {k: np.full(n, np.nan) for k in ("kv", "running", "waiting", "worker_running", "worker_waiting", "worker_kv")}
    base = {}
    cum = {k: np.full(n, np.nan) for k in ("queue_s", "engine_done", "preempt", "worker_queue_s", "worker_done")}
    bins = defaultdict(lambda: defaultdict(list))
    for res, key in ((ORCH, ""), (WORKER, "worker_")):
        for t, m in metrics[res]:
            i = int(t - t0)
            if i < 0 or i >= n:
                continue
            if res not in base:
                base[res] = (m.get("vllm:request_queue_time_seconds_sum", 0.0), m.get("vllm:e2e_request_latency_seconds_count", 0.0), m.get("vllm:num_preemptions_total", 0.0))
            bins[i][key + "kv"].append(100.0 * m.get("vllm:kv_cache_usage_perc", 0.0))
            bins[i][key + "running"].append(m.get("vllm:num_requests_running", 0.0))
            bins[i][key + "waiting"].append(m.get("vllm:num_requests_waiting", 0.0))
            b = base[res]
            bins[i][key + "queue_s"].append(m.get("vllm:request_queue_time_seconds_sum", 0.0) - b[0])
            bins[i][key + "done"].append(m.get("vllm:e2e_request_latency_seconds_count", 0.0) - b[1])
            if res == ORCH:
                bins[i]["preempt"].append(m.get("vllm:num_preemptions_total", 0.0) - b[2])
    for i, d in bins.items():
        for k in ("kv", "running", "waiting", "worker_running", "worker_waiting", "worker_kv"):
            if d.get(k):
                rows[k][i] = float(np.mean(d[k]))
        for k, src in (("queue_s", "queue_s"), ("engine_done", "done"), ("preempt", "preempt"), ("worker_queue_s", "worker_queue_s"), ("worker_done", "worker_done")):
            if d.get(src):
                cum[k][i] = float(max(d[src]))
    # forward-fill cumulative counters and engine gauges across bins without a sample
    for arr in list(cum.values()) + list(rows.values()):
        last = np.nan
        for i in range(n):
            if np.isnan(arr[i]):
                arr[i] = last
            else:
                last = arr[i]
    arrivals = np.zeros(n)
    completions = np.zeros(n)
    for s, e in reqs:
        arrivals[min(n - 1, int(s - t0))] += 1
        completions[min(n - 1, int(e - t0))] += 1
    t = np.arange(n)
    return {"t": t, **rows, **cum, "arrivals": np.cumsum(arrivals), "completions": np.cumsum(completions),
            "in_flight": np.cumsum(arrivals) - np.cumsum(completions), "wall": t_end - t0, "requests": len(reqs)}


def write_csv(level, rnd, arm, s):
    with (OUT / f"timeseries-C{level}-r{rnd}-{arm}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "orch_kv_pct", "orch_running", "orch_waiting", "orch_queue_time_cum_s", "orch_requests_done_cum", "orch_preemptions_cum",
                    "worker_kv_pct", "worker_running", "worker_waiting", "worker_queue_time_cum_s", "worker_requests_done_cum",
                    "client_arrivals_cum", "client_completions_cum", "client_in_flight"])
        for i in range(len(s["t"])):
            w.writerow([int(s["t"][i])] + [("" if np.isnan(s[k][i]) else round(float(s[k][i]), 3)) for k in
                                           ("kv", "running", "waiting", "queue_s", "engine_done", "preempt", "worker_kv", "worker_running", "worker_waiting", "worker_queue_s", "worker_done")]
                       + [int(s["arrivals"][i]), int(s["completions"][i]), int(s["in_flight"][i])])


summary = {}
data = {}
for level in LEVELS:
    for rnd in sorted({r for l, r, _ in cells if l == level}):
        for arm in ARMS:
            if (level, rnd, arm) not in cells:
                continue
            s = series(level, rnd, arm)
            data[level, rnd, arm] = s
            write_csv(level, rnd, arm, s)
            kv = s["kv"]
            first_full = next((int(i) for i in range(len(kv)) if not np.isnan(kv[i]) and kv[i] >= 99.0), None)
            wait = np.nan_to_num(s["waiting"])
            summary[f"C{level}-r{rnd}-{arm}"] = {
                "wall_s": round(float(s["wall"]), 1), "requests": s["requests"],
                "kv_first_reaches_99pct_s": first_full, "kv_mean_pct": round(float(np.nanmean(kv)), 1),
                "waiting_peak": int(wait.max()), "waiting_peak_at_s": int(wait.argmax()),
                "seconds_with_waiting_gt_0": int((wait > 0).sum()),
                "running_mean": round(float(np.nanmean(s["running"])), 1),
                "queue_time_total_s": round(float(np.nanmax(s["queue_s"])), 1),
                "orch_requests_done": int(np.nanmax(s["engine_done"])), "preemptions": int(np.nanmax(s["preempt"])),
                "time_to_100_completions_s": next((int(i) for i in range(len(s["completions"])) if s["completions"][i] >= 100), None),
            }
(OUT / "timeseries-summary.json").write_text(json.dumps(summary, indent=1) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "lines.linewidth": 1.1, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5})
COLORS = {"original": "#1f4e79", "combined": "#c0392b", "direct": "#2e86c1", "reduce": "#e67e22"}
LABELS = {"original": "Original", "combined": "Combined", "direct": "Direct dispatch", "reduce": "Input reduction"}


def panel_plot(ax_list, s, arm, alpha=1.0, label=True, lw=None):
    c = COLORS[arm]
    kw = {"color": c, "alpha": alpha}
    if lw:
        kw["linewidth"] = lw
    lab = LABELS[arm] if label else None
    ax_list[0].plot(s["t"], s["kv"], label=lab, **kw)
    ax_list[1].plot(s["t"], s["running"], label=(f"{LABELS[arm]}: running" if label else None), **kw)
    ax_list[1].plot(s["t"], s["waiting"], linestyle="--", label=(f"{LABELS[arm]}: waiting" if label else None), **kw)
    ax_list[2].plot(s["t"], s["queue_s"], label=lab, **kw)
    ax_list[3].plot(s["t"], s["completions"], label=(f"{LABELS[arm]}: completed" if label else None), **kw)
    ax_list[3].plot(s["t"], s["arrivals"], linestyle=":", label=(f"{LABELS[arm]}: started" if label else None), **kw)


def finish(axes, title):
    axes[0].set_ylabel("Orchestration KV cache (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("Orchestration requests")
    axes[2].set_ylabel("Cumulative queue time (s)")
    axes[3].set_ylabel("Client requests (cumulative)")
    axes[3].set_xlabel("Seconds since the first request of the run")
    for letter, ax in zip("abcd", axes):
        ax.legend(loc="best", frameon=False, ncol=2)
        ax.text(0.005, 0.97, f"({letter})", transform=ax.transAxes, va="top", ha="left", fontsize=8, fontweight="bold")
    axes[0].set_title(title)


def figure_pair(level, rnd, arms, name, title):
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.6), sharex=True)
    for arm in arms:
        if (level, rnd, arm) in data:
            panel_plot(axes, data[level, rnd, arm], arm)
    finish(axes, title)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


def figure_rounds(level, arms, name, title):
    """All rounds overlaid (thin lines); the per-second mean across rounds drawn thick."""
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.6), sharex=True)
    for arm in arms:
        runs = [data[k] for k in data if k[0] == level and k[2] == arm]
        for s in runs:
            panel_plot(axes, s, arm, alpha=0.25, label=False, lw=0.8)
        if runs:
            n = max(len(s["t"]) for s in runs)
            mean = {}
            for key in ("kv", "running", "waiting", "queue_s", "completions", "arrivals"):
                stack = np.full((len(runs), n), np.nan)
                for i, s in enumerate(runs):
                    stack[i, :len(s[key])] = s[key]
                    # after a run ends: gauges are idle (0), cumulative counters keep their final value
                    stack[i, len(s[key]):] = 0.0 if key in ("kv", "running", "waiting") else s[key][-1]
                mean[key] = np.nanmean(stack, axis=0)
            mean["t"] = np.arange(n)
            panel_plot(axes, mean, arm, alpha=1.0, label=True, lw=1.6)
    finish(axes, title)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


def figure_worker(level, rnd, arms, name, title):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    for arm in arms:
        if (level, rnd, arm) not in data:
            continue
        s = data[level, rnd, arm]
        axes[0].plot(s["t"], s["worker_running"], color=COLORS[arm], label=f"{LABELS[arm]}: running")
        axes[0].plot(s["t"], s["worker_waiting"], color=COLORS[arm], linestyle="--", label=f"{LABELS[arm]}: waiting")
        axes[1].plot(s["t"], s["worker_queue_s"], color=COLORS[arm], label=LABELS[arm])
    axes[0].set_ylabel("Worker-engine requests")
    axes[1].set_ylabel("Worker cumulative queue time (s)")
    axes[1].set_xlabel("Seconds since the first request of the run")
    for letter, ax in zip("ab", axes):
        ax.legend(loc="best", frameon=False, ncol=2)
        ax.text(0.005, 0.97, f"({letter})", transform=ax.transAxes, va="top", ha="left", fontsize=8, fontweight="bold")
    axes[0].set_title(title)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


for level in LEVELS:
    rounds = sorted({r for l, r, _ in data if l == level})
    for rnd in rounds:
        figure_worker(level, rnd, ["original", "combined"], f"timeseries-C{level}-r{rnd}-worker-original-vs-combined",
                      f"Concurrency {level}, round {rnd}: worker engine (tool-call formation and synthesis)")
    for rnd in rounds:
        figure_pair(level, rnd, ["original", "combined"], f"timeseries-C{level}-r{rnd}-original-vs-combined",
                    f"Concurrency {level}, round {rnd}: original vs combined (t = 0 at each run's first request)")
    figure_rounds(level, ["original", "combined"], f"timeseries-C{level}-all-rounds-original-vs-combined",
                  f"Concurrency {level}: four rounds per arm (thin) and their per-second mean (thick)")
    figure_rounds(level, ARMS, f"timeseries-C{level}-all-rounds-four-arms",
                  f"Concurrency {level}: four arms, four rounds each (thin) and per-second means (thick)")
print(json.dumps(summary, indent=1))
