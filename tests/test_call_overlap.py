import asyncio

import pytest

from cnu_rag_optimization.call_overlap import CallContract, CallOverlapAdapter, canonical_inputs


def contract(request="req", inputs=None, *, deterministic=True, operation="preparation"):
    return CallContract.build(operation, "app.call_llm", request, inputs or {"m": ["a"]}, deterministic=deterministic)


def test_contract_requires_deterministic_attestation_and_exact_inputs():
    assert contract().eligible
    assert not contract(deterministic=False).eligible
    assert not CallContract("", "impl", "req", b"x", True).eligible
    assert contract(inputs={"b": 1, "a": [1, 2]}).inputs == canonical_inputs({"a": [1, 2], "b": 1})
    assert contract(inputs={"m": ["a"]}) != contract(inputs={"m": ["a "]})


def test_exact_contract_reuses_prepared_result_without_fallback():
    events = []
    adapter = CallOverlapAdapter(on_event=events.append)
    calls = []

    async def run():
        gate = asyncio.Event()

        async def factory():
            calls.append("prepared")
            await gate.wait()
            return "prepared-output"

        async def fallback():
            calls.append("fallback")
            return "fallback-output"

        assert adapter.prepare(contract(), factory) is True
        assert adapter.prepare(contract(), factory) is False  # single preparation per request/operation
        gate.set()
        assert await adapter.consume(contract(), fallback) == "prepared-output"
        assert adapter.pending == 0
    asyncio.run(run())
    assert calls == ["prepared"]
    reused = [e for e in events if e.get("reused") is True]
    assert len(reused) == 1 and reused[0]["reason"] == "exact_contract"
    assert reused[0]["head_start_ms"] >= 0 and reused[0]["call_ms"] >= 0
    assert [e["reason"] for e in events if e.get("prepared") is False] == ["already_prepared"]


@pytest.mark.parametrize("variant", ["inputs", "operation_missing", "ineligible_actual"])
def test_mismatch_or_missing_preparation_runs_original_call(variant):
    events = []
    adapter = CallOverlapAdapter(on_event=events.append)
    calls = []

    async def run():
        async def factory():
            calls.append("prepared")
            await asyncio.sleep(3600)

        async def fallback():
            calls.append("fallback")
            return "fallback-output"

        adapter.prepare(contract(), factory)
        await asyncio.sleep(0)
        if variant == "inputs":
            actual = contract(inputs={"m": ["b"]})
        elif variant == "operation_missing":
            actual = contract(operation="other")
        else:
            actual = contract(deterministic=False)
        assert await adapter.consume(actual, fallback) == "fallback-output"
        await adapter.close("req")
        assert adapter.pending == 0
    asyncio.run(run())
    assert calls == ["prepared", "fallback"]
    outcome = next(e for e in events if e.get("reused") is False)
    assert outcome["reason"] == ("not_prepared" if variant == "operation_missing" else "contract_mismatch")


def test_prepared_call_error_falls_back_exactly_once():
    adapter = CallOverlapAdapter()
    calls = []

    async def run():
        async def factory():
            calls.append("prepared")
            raise RuntimeError("upstream failure")

        async def fallback():
            calls.append("fallback")
            return "ok"

        adapter.prepare(contract(), factory)
        assert await adapter.consume(contract(), fallback) == "ok"
    asyncio.run(run())
    assert calls == ["prepared", "fallback"]


def test_consumer_cancellation_is_not_converted_into_a_fallback_call():
    adapter = CallOverlapAdapter()
    calls = []

    async def run():
        async def factory():
            await asyncio.sleep(3600)

        async def fallback():
            calls.append("fallback")

        adapter.prepare(contract(), factory)
        consumer = asyncio.create_task(adapter.consume(contract(), fallback))
        await asyncio.sleep(0)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await adapter.close("req")
    asyncio.run(run())
    assert calls == []


def test_close_and_staleness_discard_unconsumed_work_and_report_waste():
    events = []
    clock = [0.0]
    adapter = CallOverlapAdapter(on_event=events.append, clock=lambda: clock[0], stale_seconds=10)

    async def run():
        async def finished():
            return "done"

        async def never():
            await asyncio.sleep(3600)

        adapter.prepare(contract("r1"), finished)
        adapter.prepare(contract("r2"), never)
        await asyncio.sleep(0)
        assert await adapter.close("r1") == 1
        clock[0] = 11.0
        adapter.prepare(contract("r3"), finished)
        assert adapter.pending == 2  # r2 (stale) is still registered until the next consume purges it
        assert await adapter.consume(contract("r3"), never) == "done"
        assert adapter.pending == 0  # the stale r2 preparation was dropped during the purge
    asyncio.run(run())
    dropped = {e["request_key"]: e for e in events if e.get("discarded")}
    assert dropped["r1"]["reason"] == "request_closed" and dropped["r1"]["completed_before_discard"] is True
    assert dropped["r2"]["reason"] == "stale" and dropped["r2"]["completed_before_discard"] is False


def test_capacity_bound_refuses_new_preparation():
    events = []
    adapter = CallOverlapAdapter(on_event=events.append, max_pending=1)

    async def run():
        async def never():
            await asyncio.sleep(3600)

        assert adapter.prepare(contract("a"), never) is True
        assert adapter.prepare(contract("b"), never) is False
        await adapter.close("a")
    asyncio.run(run())
    assert [e["reason"] for e in events if e.get("prepared") is False] == ["capacity"]
