import json

import pytest

from cnu_rag_optimization import CompiledProcedureHarness


def _manifest(tmp_path):
    path = tmp_path / "procedures.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "compiler": "test",
                "source_trace_sha256": "abc",
                "procedures": [
                    {
                        "procedure_id": "typed-retrieval",
                        "stage": "retrieval_dispatch",
                        "action": "typed_rpc_dispatch",
                        "required_contracts": ["validated_retrieval_arguments"],
                        "fallback_action": "llm_tool_dispatch",
                        "priority": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_compiled_procedure_requires_contract(tmp_path) -> None:
    harness = CompiledProcedureHarness(_manifest(tmp_path))

    miss = harness.resolve("retrieval_dispatch", available_contracts=())
    hit = harness.resolve(
        "retrieval_dispatch",
        available_contracts={"validated_retrieval_arguments"},
    )

    assert not miss.selected
    assert miss.fallback_action == "llm_tool_dispatch"
    assert hit.selected
    assert hit.action == "typed_rpc_dispatch"
    assert harness.metadata()["procedure_count"] == 1


def test_compiled_procedure_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 2, "procedures": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported compiled procedure schema"):
        CompiledProcedureHarness(path).resolve("retrieval", available_contracts=())
