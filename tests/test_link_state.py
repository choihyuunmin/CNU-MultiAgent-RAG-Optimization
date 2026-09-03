from cnu_rag_optimization import ContractLinkStateRouter, LinkSpec


def _router() -> ContractLinkStateRouter:
    router = ContractLinkStateRouter(failure_cooldown_seconds=60.0)
    router.add_link(
        LinkSpec(
            "compiled",
            "ready",
            "done",
            10.0,
            required_contracts=frozenset({"safe"}),
        )
    )
    router.add_link(LinkSpec("legacy", "ready", "done", 100.0, capacity=4))
    return router


def test_router_excludes_link_without_contract() -> None:
    router = _router()

    assert router.choose_path("ready", "done").links == ("legacy",)
    assert router.choose_path("ready", "done", contracts={"safe"}).links == (
        "compiled",
    )


def test_router_accounts_for_inflight_load() -> None:
    router = _router()
    first = router.choose_path("ready", "done", contracts={"safe"})
    token = router.begin(first)

    congested = router.choose_path("ready", "done", contracts={"safe"})
    router.finish(token, success=True, elapsed_ms=10.0)

    assert congested.cost_ms > first.cost_ms
    assert router.link_snapshot("compiled")["inflight"] == 0


def test_router_opens_failed_link_and_falls_back() -> None:
    router = _router()
    selected = router.choose_path("ready", "done", contracts={"safe"})
    token = router.begin(selected)
    router.finish(token, success=False, elapsed_ms=5.0, failed_link_id="compiled")

    assert router.choose_path("ready", "done", contracts={"safe"}).links == (
        "legacy",
    )
