"""LLM-free selection over already-ranked retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


Document = Mapping[str, object]
DocumentValidator = Callable[[Document], bool]


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: tuple[str, ...]
    selected_documents: tuple[Document, ...]

    @property
    def has_relevant_documents(self) -> bool:
        return bool(self.selected_ids)


def select_ranked_documents(
    documents: Sequence[Document],
    *,
    top_k: int = 5,
    id_keys: tuple[str, ...] = ("id", "law_id"),
    validator: DocumentValidator | None = None,
) -> SelectionResult:
    """Preserve retrieval rank, remove duplicate IDs, return top-k documents.

    Retrieval and reranking must run before this function. ``validator`` lets an
    integrating system apply its own safety or quality filter without exposing
    production filtering code here.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")

    selected_ids: list[str] = []
    selected_documents: list[Document] = []
    seen: set[str] = set()
    for document in documents:
        if validator is not None and not validator(document):
            continue
        document_id = next(
            (str(document.get(key) or "").strip() for key in id_keys if document.get(key)),
            "",
        )
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        selected_ids.append(document_id)
        selected_documents.append(document)
        if len(selected_ids) == top_k:
            break

    return SelectionResult(tuple(selected_ids), tuple(selected_documents))
