"""Read-only serving token counts and pressure for the isolated workflow adapter.

Credentials/routes come from operator files; never include them or payloads in
public telemetry. Tokenization overhead is enabled in every compared arm.
"""
import asyncio
import os
import time
from pathlib import Path

from cnu_rag_optimization import ServingPressure, TokenReservation
from moleg_scaling_metrics import parse_metrics


def operator_routes():
    from dotenv import dotenv_values
    import yaml
    serving = dotenv_values(os.environ["MOLEG_SERVING_ENV"])
    config = yaml.safe_load(Path(os.environ["MOLEG_PROXY_CONFIG"]).read_text())
    def resolve(value):
        if isinstance(value, str) and value.startswith("os.environ/"):
            name = value.split("/", 1)[1]
            return os.environ.get(name) or serving.get(name)
        return value
    routes = {}
    for item in config.get("model_list", []):
        p = item.get("litellm_params", {})
        if str(p.get("model", "")).startswith("openai/"):
            routes[item["model_name"]] = {
                "model": p["model"][len("openai/"):],
                "base": resolve(p["api_base"]).rstrip("/").removesuffix("/v1"),
                "key": resolve(p.get("api_key")) or serving["VLLM_API_KEY"],
            }
    return routes


class ServingMeter:
    def __init__(self, adapter, settings, routes=None):
        self.adapter = adapter
        self.routes = routes if routes is not None else operator_routes()
        self.roles = tuple(settings.get("roles", ["orchestrator"]))
        if any(role not in self.routes for role in self.roles):
            raise ValueError("meter role missing in operator routes")
        self.output_reserve = settings.get("output_reserve_tokens", 512)
        if type(self.output_reserve) is not int or self.output_reserve < 1:
            raise ValueError("output reserve must be a positive integer")
        self.client = None
        self.last_metrics = {}
        self.previous_preemptions = {}
        self.locks = {role: asyncio.Lock() for role in self.roles}
        self.counts = {"tokenize_ok": 0, "tokenize_errors": 0, "tokenize_retries": 0,
                       "metrics_ok": 0, "metrics_errors": 0}

    def _client(self):
        if self.client is None:
            import httpx
            self.client = httpx.AsyncClient(timeout=5, trust_env=False,
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
        return self.client

    async def refresh_pressure(self, role):
        async with self.locks[role]:
            if time.monotonic() - self.last_metrics.get(role, -100) < 1:
                return
            self.last_metrics[role] = time.monotonic()
            route = self.routes[role]
            try:
                response = await self._client().get(route["base"] + "/metrics",
                    headers={"Authorization": "Bearer " + route["key"]})
                response.raise_for_status()
                values = parse_metrics(response.text)["values"]
                current = values["num_preemptions_total"]
                previous = self.previous_preemptions.get(role)
                self.previous_preemptions[role] = current
                self.adapter.pressure.pop(role, None)
                if previous is not None and current >= previous:
                    self.adapter.pressure[role] = ServingPressure(time.monotonic(),
                        values.get("kv_cache_usage_perc", values.get("gpu_cache_usage_perc")),
                        values["num_requests_waiting"], current - previous)
                self.counts["metrics_ok"] += 1
            except Exception:
                self.adapter.pressure.pop(role, None)
                self.previous_preemptions.pop(role, None)
                self.counts["metrics_errors"] += 1

    async def estimate(self, role, kwargs):
        if role not in self.roles:
            return None
        route = self.routes[role]
        started = time.monotonic()
        # Match actual chat tokenization; never rewrite the generation request.
        payload = {"model": route["model"], "messages": kwargs.get("messages", []),
                   "add_generation_prompt": True}
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        extra = kwargs.get("extra_body") or {}
        if extra.get("chat_template_kwargs"):
            payload["chat_template_kwargs"] = extra["chat_template_kwargs"]
        response = None
        try:
            import httpx
            for attempt in range(2):
                try:
                    response = await self._client().post(route["base"] + "/tokenize", json=payload,
                        headers={"Authorization": "Bearer " + route["key"]})
                    if response.status_code >= 500 and attempt == 0:
                        self.counts['tokenize_retries'] += 1
                        continue
                    break
                except httpx.TransportError:
                    if attempt:
                        raise
                    self.counts['tokenize_retries'] += 1
            response.raise_for_status()
            count = response.json()["count"]
            output = kwargs.get("max_tokens")
            output = int(output) if output is not None else self.output_reserve
            reservation = TokenReservation(count, output, 0)
            self.counts["tokenize_ok"] += 1
            self.adapter.record("tokenization", started, model=role, input_tokens=count,
                                output_reserve_tokens=output, success=True)
        except Exception as exc:
            self.counts["tokenize_errors"] += 1
            kind = 'tokenize_failure_' + type(exc).__name__
            self.counts[kind] = self.counts.get(kind, 0) + 1
            self.adapter.record("tokenization", started, model=role, success=False,
                                http_status=response.status_code if response is not None else None)
            reservation = None  # budget mode executes unknown costs exclusively
        await self.refresh_pressure(role)
        return reservation

    async def close(self):
        if self.client is not None:
            await self.client.aclose()

    def snapshot(self):
        return dict(self.counts)
