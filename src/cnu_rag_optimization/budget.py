"""Input and output budget helpers for LLM stages."""

from __future__ import annotations

from typing import Mapping, Sequence

from .config import OptimizationPolicy


def _shorten(value: object, max_chars: int) -> object:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return value[:max_chars] + "…"


def compact_documents(
    documents: Sequence[Mapping[str, object]],
    *,
    max_documents: int = 10,
    field_max_chars: int = 800,
    keep_fields: tuple[str, ...] = (
        "id",
        "document_id",
        "title",
        "source",
        "group",
        "section",
        "content",
        "score",
    ),
) -> list[dict[str, object]]:
    """Create compact copies. Source documents remain unchanged."""
    if max_documents < 1 or field_max_chars < 1:
        raise ValueError("document and field limits must be positive")
    compacted: list[dict[str, object]] = []
    for document in documents[:max_documents]:
        compacted.append(
            {
                key: _shorten(document[key], field_max_chars)
                for key in keep_fields
                if key in document and document[key] is not None
            }
        )
    return compacted


def cap_comparison_documents(
    documents_by_group: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    max_items_per_group: int = 3,
    max_content_chars: int = 600,
    content_fields: tuple[str, ...] = ("content", "text"),
) -> dict[str, list[dict[str, object]]]:
    """Cap comparison context per group while preserving metadata."""
    if max_items_per_group < 1 or max_content_chars < 1:
        raise ValueError("comparison limits must be positive")
    result: dict[str, list[dict[str, object]]] = {}
    for group, documents in documents_by_group.items():
        group_rows: list[dict[str, object]] = []
        for document in documents[:max_items_per_group]:
            row = dict(document)
            for field in content_fields:
                if field in row:
                    row[field] = _shorten(row[field], max_content_chars)
            group_rows.append(row)
        result[str(group)] = group_rows
    return result


def llm_options(role: str, policy: OptimizationPolicy) -> dict[str, int]:
    """Return provider-neutral request options for a pipeline role."""
    if not policy.token_budget_enabled:
        return {}
    max_tokens = policy.budgets.for_role(role)
    return {"max_tokens": max_tokens} if max_tokens is not None else {}
