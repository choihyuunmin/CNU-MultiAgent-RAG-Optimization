"""Fail-closed speculative execution guarded by authoritative selection."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


def _ids(values: Iterable[object], *, order_sensitive: bool) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    return normalized if order_sensitive else tuple(sorted(set(normalized)))


@dataclass(frozen=True)
class SpeculativeDraft(Generic[T]):
    value: T
    selected_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedSpeculationResult(Generic[T]):
    value: T
    reused_draft: bool
    reason: str
    authoritative_ids: tuple[str, ...]
    speculative_ids: tuple[str, ...]


async def run_verified_speculation(
    authoritative_selection: Awaitable[Iterable[object]],
    speculative_draft: Awaitable[SpeculativeDraft[T]],
    fallback: Callable[[tuple[str, ...]], Awaitable[T]],
    *,
    order_sensitive: bool = True,
) -> VerifiedSpeculationResult[T]:
    """Overlap work; reuse draft only when authoritative document IDs match.

    Selector or draft disagreement never reaches user. Draft failures also fall back
    to normal generation from authoritative IDs.
    """
    selection_task = asyncio.ensure_future(authoritative_selection)
    draft_task = asyncio.ensure_future(speculative_draft)
    try:
        authoritative = _ids(
            await selection_task,
            order_sensitive=order_sensitive,
        )
    except BaseException:
        draft_task.cancel()
        await asyncio.gather(draft_task, return_exceptions=True)
        raise

    try:
        draft = await draft_task
    except Exception:
        value = await fallback(authoritative)
        return VerifiedSpeculationResult(
            value=value,
            reused_draft=False,
            reason="draft_error",
            authoritative_ids=authoritative,
            speculative_ids=(),
        )

    speculative = _ids(draft.selected_ids, order_sensitive=order_sensitive)
    if authoritative == speculative:
        return VerifiedSpeculationResult(
            value=draft.value,
            reused_draft=True,
            reason="verified_match",
            authoritative_ids=authoritative,
            speculative_ids=speculative,
        )

    value = await fallback(authoritative)
    return VerifiedSpeculationResult(
        value=value,
        reused_draft=False,
        reason="selection_mismatch",
        authoritative_ids=authoritative,
        speculative_ids=speculative,
    )
