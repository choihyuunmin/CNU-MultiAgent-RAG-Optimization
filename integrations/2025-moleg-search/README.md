# Search-system integration

## Current validation target

Compare the original path with validated direct execution without changing the
model, prompts, inference options, or serving configuration.

The adapter entry point is
[try_typed_single_tool_dispatch](../../src/cnu_rag_optimization/typed_dispatch.py).
It requires one available tool, a matching selected tool, complete JSON
arguments, and an argument validator. Rejected inputs return
`dispatched=False`; the caller must retain the original routing path.
Handler exceptions propagate and must not trigger a duplicate execution.
Keep the existing result parser and downstream object unchanged.

Use [the trace audit](../../src/cnu_rag_optimization/trace_equivalence.py) to
compare prepared arguments, executed arguments, handler output, and downstream
objects by request and branch. Input equality alone does not prove output
equality or answer correctness.

## Optional configurations

The JSON configurations and serving plugin in this directory remain as
implementation resources. They are not enabled by the current validation plan.
Some select low reasoning effort, ID rewriting, or engine-side acceleration;
do not apply them to the direct-dispatch comparison.

Prior timing tables, run reports, and links to removed results have been
removed. This directory makes no measured speed or accuracy claim.
