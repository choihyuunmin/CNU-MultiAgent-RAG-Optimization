"""Reusable HTTP transport for LLM, embedding, and reranker clients."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class HTTPTransportPolicy:
    """Bound persistent connections without changing request payloads."""

    max_connections: int = 64
    max_keepalive_connections: int = 32
    keepalive_expiry_seconds: float = 60.0
    timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 5.0
    http2: bool = False

    def __post_init__(self) -> None:
        if self.max_connections < 1:
            raise ValueError("max_connections must be positive")
        if not 1 <= self.max_keepalive_connections <= self.max_connections:
            raise ValueError(
                "max_keepalive_connections must be between 1 and max_connections"
            )
        if self.keepalive_expiry_seconds <= 0:
            raise ValueError("keepalive_expiry_seconds must be positive")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")


def build_async_client(
    policy: HTTPTransportPolicy = HTTPTransportPolicy(),
    **client_options: object,
) -> httpx.AsyncClient:
    """Build one long-lived client. Caller owns and closes returned client."""
    limits = httpx.Limits(
        max_connections=policy.max_connections,
        max_keepalive_connections=policy.max_keepalive_connections,
        keepalive_expiry=policy.keepalive_expiry_seconds,
    )
    timeout = httpx.Timeout(
        policy.timeout_seconds,
        connect=policy.connect_timeout_seconds,
    )
    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        http2=policy.http2,
        **client_options,
    )
