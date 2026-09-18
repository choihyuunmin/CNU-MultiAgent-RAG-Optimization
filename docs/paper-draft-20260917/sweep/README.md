# Load-sweep tools

These scripts are retained implementation tools, not current experiment results.
The active design is the
[direct-dispatch validation plan](../../DISPATCH_VALIDATION_PROTOCOL_20260918.md).
Historical measurements, run locations, and deployment commands have been removed.

- `make_sweep_manifest.py`: builds manifests using an external frozen bundle.
- `launch_sweep_remote.py`: supervises legacy sweep runs.
- `run_sweep_trial.py`: runs the load client.
- `serve_sweep.py`: legacy application-serving wrapper.
- `analyze_sweep.py`: summarizes supplied run data.

Legacy defaults, arm names, application limits, and bundle paths do not implement
the current plan automatically. Inspect and validate integration before execution.
No serving engine or running application is changed by this documentation cleanup.
