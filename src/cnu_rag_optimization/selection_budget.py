from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionBudget:
    """Fields to keep and how much text to keep per document.

    keep_fields: fields the selection instructions reference; everything else
        is dropped. ``id`` is always kept.
    text_max_chars: cap for each text field named in ``text_fields``.
    max_documents: optional cap on the number of documents (ranked order kept).
        None keeps every document, which leaves the candidate set unchanged.
    """
    keep_fields: tuple[str, ...] = ("id", "title", "subject", "content", "country")
    text_fields: tuple[str, ...] = ("content", "summary", "text")
    text_max_chars: int = 300
    max_documents: int | None = None

    def __post_init__(self):
        if type(self.text_max_chars) is not int or self.text_max_chars < 1:
            raise ValueError("text_max_chars must be a positive integer")
        if self.max_documents is not None and (type(self.max_documents) is not int or self.max_documents < 1):
            raise ValueError("max_documents must be a positive integer or None")
        if not self.keep_fields:
            raise ValueError("keep_fields must not be empty")


def budget_selection_input(text: str, budget: SelectionBudget, *, list_key: str = "laws") -> tuple[str, dict]:
    """Return the compacted JSON text and a record of what changed.

    Input that is not a JSON object holding a list under ``list_key`` is returned
    unchanged (record ``applied`` is False), so a caller never loses data to a
    format the budget does not understand. Non-dict list items are kept as is.
    """
    record = {"applied": False, "chars_before": len(text or ""), "chars_after": len(text or "")}
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        record["reason"] = "not_json"
        return text, record
    if not isinstance(data, dict) or not isinstance(data.get(list_key), list):
        record["reason"] = "no_document_list"
        return text, record
    documents = data[list_key]
    keep = set(budget.keep_fields) | {"id"}
    limit = len(documents) if budget.max_documents is None else min(len(documents), budget.max_documents)
    dropped_fields, shortened = set(), 0
    compacted = []
    for document in documents[:limit]:
        if not isinstance(document, dict):
            compacted.append(document)
            continue
        row = {}
        for key, value in document.items():
            if key not in keep:
                dropped_fields.add(key)
                continue
            if key in budget.text_fields and isinstance(value, str) and len(value) > budget.text_max_chars:
                value = value[:budget.text_max_chars] + "…"
                shortened += 1
            row[key] = value
        compacted.append(row)
    out = dict(data)
    out[list_key] = compacted
    text_out = json.dumps(out, ensure_ascii=False)
    record.update({"applied": True, "documents_before": len(documents), "documents_after": len(compacted),
                   "dropped_fields": sorted(dropped_fields), "shortened_fields": shortened, "chars_after": len(text_out),
                   "ids_preserved": all(isinstance(d, dict) and "id" in d for d in documents[:limit]) == all(isinstance(d, dict) and "id" in d for d in compacted)})
    return text_out, record
