# Direct-dispatch execution and analysis tools

The current design is the
[direct-dispatch validation plan](../../DISPATCH_VALIDATION_PROTOCOL_20260918.md).
Historical run summaries and deployment instructions have been removed.
These tools do not contain a completed result for the new design.

## Files

- `review_dispatch_attachment.py`: integration attachment and trace fields.
- `run_followup_trial.py`: client runner with explicit concurrency, timeout, and seed.
- `analyze_followup.py`: branch-aware comparison using the public package.
- `make_followup_manifest.py`: manifest builder requiring an external frozen bundle.
- `launch_followup_remote.py`: legacy deployment supervisor; inspect before reuse.

Install the package from the repository root with `pip install -e .`.
The runner's default concurrency and timeout are legacy defaults, not the new
plan. Pass the frozen plan's values explicitly. Rebuild the external adapter
bundle from the current code; do not reuse a historical ZIP.

The analyzer joins events by branch ID. Missing or failed traces remain
`unverifiable`; repeated identical inputs remain `ambiguous_repeated_inputs`.
Do not count either category as an output match. Review handler output and
downstream objects separately from final answer quality.
