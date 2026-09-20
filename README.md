# Multi-Agent RAG Optimization

Python adapters for reducing the response latency of multi-agent RAG (retrieval-augmented generation) systems. The package observes task dependencies between agents, model-call waiting, and streaming transport, and compares execution strategies and inference-acceleration techniques under the same conditions.

## Features

- **Dependency-based execution**: each task declares the predecessors it needs, and a successor starts as soon as those predecessors finish. Independent tasks run in parallel and are joined where their results are consumed.
- **Streaming transport management**: reading a model response and delivering it to the user are separated and connected through a bounded buffer. Stream completion, errors, and cancellation are handled together.
- **Reference-ID shortening and restoration**: long document IDs handed to a model are replaced by short numbers, and the numbers the model selects are mapped back to the original IDs. The affected fields are configurable, and duplicates or restoration errors are detected.
- **Execution observation**: per-stage execution time, waiting time, model calls, token usage, and errors are recorded for bottleneck analysis.
- **Checked direct dispatch**: when the application has already selected exactly one tool and prepared its arguments, the handler is invoked directly through local code after schema and copy checks, instead of asking a model to restate the call.
- **Verified call overlap**: a deterministic model call whose inputs are already fixed is started alongside the preceding call, and its result is used at the original call site only when the inputs match exactly. Otherwise the original call runs unchanged.
- **Selection-input budget**: search results passed to the document-selection stage are reduced by dropping fields the selection instructions never reference and by capping long text fields, while every document and ID is kept.
- **Inference-acceleration experiments**: serving techniques such as speculative decoding, where the original model verifies proposed tokens, are compared under identical conditions.
- **Performance and quality validation**: the original and the modified paths are compared repeatedly on the same questions, reporting mean and 95th-percentile response time, throughput, error-free completion rate, and agreement of the retrieved evidence.

## How it is applied

The current validation target is the **checked direct-dispatch adapter**. It removes a redundant tool-formatting model call without changing models, prompts, or serving options, and evaluates end-to-end response time and output quality together. Raw responses, logs, and other private material from validation runs are not kept in this repository.

The common modules do not depend on a particular domain or agent framework. The integrating system supplies its task dependencies, model-call functions, reference-ID fields, and quality criteria. Applicability, performance, and quality must be validated on each system.

## Repository layout

- `src/cnu_rag_optimization/`: common adapters, execution and transport management, and evaluation modules
- `examples/`: adapter integration examples
- `integrations/`: integration code for individual search systems
- `scripts/`: server entry points and overlays used for validation runs
- `tests/`: unit tests
