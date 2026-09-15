# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Standalone paper figures; plot only observed loads, never extrapolate users."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--note", default="finite workload; shared serving telemetry")
    args = parser.parse_args()
    data = json.loads((args.directory / "aggregate.json").read_text())
    groups = data["groups"]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    policies = sorted({g["policy"] for g in groups})
    palette = {"baseline": "#536274", "fixed": "#E28E30", "adaptive": "#2687A8"}
    is_open = groups[0]["arrival_rate"] is not None
    x_key, label = ("arrival_rate", "Offered requests / s") if is_open else ("users", "Concurrent users")
    fallback_colors = plt.get_cmap("tab10")
    for policy_index, policy in enumerate(policies):
        rows = sorted([g for g in groups if g["policy"] == policy], key=lambda g: g[x_key])
        x = [g[x_key] for g in rows]
        color = palette.get(policy, fallback_colors(policy_index % 10))
        for ax, values, title in [
            (axes[0, 0], [g["metrics"]["elapsed_s"]["mean"] for g in rows], "Mean full response (s)"),
            (axes[0, 1], [g["metrics"]["elapsed_s"]["p95"] for g in rows], "P95 full response (s)"),
            (axes[0, 2], [g["throughput_rps"] for g in rows], "Completed requests / s"),
            (axes[1, 0], [g["slo_goodput_rps"] for g in rows], "Completed within 30 s / s"),
        ]:
            ax.plot(x, values, "o-", label=policy, color=color)
            ax.set_title(title)
        if policy == "baseline" and all("first_event_s" in g["metrics"] for g in rows):
            axes[0, 0].plot(x, [g["metrics"]["first_event_s"]["mean"] for g in rows],
                            "s--", color="#B75555", label="before first progress")
        orch = [g["telemetry"].get("orchestrator", {}) for g in rows]
        axes[1, 1].plot(x, [v.get("peak_kv_usage") for v in orch], "o-", color=color, label=policy)
        axes[1, 2].plot(x, [v.get("peak_waiting") for v in orch], "o-", color=color, label=policy)
    axes[1, 1].set_title("Orchestrator peak KV usage (fraction)")
    axes[1, 2].set_title("Orchestrator peak waiting requests")
    for ax in axes.flat:
        ax.set_xlabel(label)
        ax.set_ylim(bottom=0)
        if not is_open:
            ax.set_xscale("log", base=2)
            ticks = sorted({g[x_key] for g in groups})
            ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    prefix = "Live API scaling" if data["complete"] else "INCOMPLETE live API scaling"
    figure.suptitle(prefix + ": " + args.note, fontsize=12)
    figure.savefig(args.directory / "scaling.pdf", bbox_inches="tight")
    figure.savefig(args.directory / "scaling.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    baseline = sorted([g for g in groups if g["policy"] == "baseline"], key=lambda g: g[x_key])
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, server in zip(axes, ("orchestrator", "worker")):
        bottom = [0.] * len(baseline)
        labels = [str(g[x_key]) for g in baseline]
        for phase, color in (("queue", "#B75555"), ("prefill", "#E28E30"), ("decode", "#2687A8")):
            values = [g["telemetry"].get(server, {}).get("histograms", {}).get(
                f"request_{phase}_time_seconds", {}).get("mean") for g in baseline]
            if any(v is None for v in values):
                continue
            ax.bar(labels, values, bottom=bottom, label=phase, color=color)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set(title=server, xlabel=label, ylabel="Mean seconds / model call")
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    figure.suptitle("Shared server timing counters; baseline policy, not API stage attribution", fontsize=11)
    figure.savefig(args.directory / "server_times.pdf", bbox_inches="tight")
    figure.savefig(args.directory / "server_times.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
