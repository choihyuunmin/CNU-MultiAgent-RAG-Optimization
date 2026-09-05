"""Wrap an existing send operation without inspecting or changing its payload."""

from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar

from cnu_rag_optimization import CoflowAdmission, CoflowPolicy


T = TypeVar("T")


class ReceiverAwareClient:
    def __init__(self, alias_to_receiver: Mapping[str, str]):
        # The deployment supplies identities; two aliases may share a receiver.
        self.alias_to_receiver = dict(alias_to_receiver)
        self.admission = CoflowAdmission(CoflowPolicy())

    async def call(
        self,
        *,
        model_alias: str,
        root_request_id: str,
        work_class: str,
        send_unchanged: Callable[[], Awaitable[T]],
    ) -> T:
        receiver = self.alias_to_receiver.get(model_alias)
        if receiver is None:
            return await send_unchanged()
        async with self.admission.slot(receiver, root_request_id, work_class):
            return await send_unchanged()

    def observe_receiver(self, receiver: str, *, waiting: float, running: float):
        """Pass existing read-only queue telemetry at a fixed sampling interval."""
        return self.admission.feedback(receiver, waiting=waiting, running=running)
