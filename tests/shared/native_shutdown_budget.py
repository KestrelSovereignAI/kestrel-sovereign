"""Compose fixture lifecycle guards with real backend close reservations."""


def reserve_native_shutdown_budget(agent, monkeypatch, *, lifecycle_module=None):
    """Keep the original prefix window when CI reserves slower SQLite closes.

    This changes fixture guards only. Production shutdown defaults and tests
    deliberately exercising insufficient deadlines retain their own budgets.
    The caller's MonkeyPatch owns restoration, including on a failing test.
    """
    from kestrel_sovereign import kestrel_agent as agent_module

    prefix = max(
        float(agent_module.SHUTDOWN_TIMEOUT),
        agent_module.KESTREL_AGENT_SHUTDOWN_TIMEOUT_S,
    )
    tail = agent._durable_tail_minimum_budget(
        agent_module._minimum_storage_close_timeout(agent.storage),
        agent_module._minimum_storage_potential_preclose_timeout(agent.storage),
    )
    total = tail + prefix
    assert 0 < total <= 30, "native lifecycle fixture requires a finite coherent budget"
    monkeypatch.setattr(agent_module, "SHUTDOWN_TIMEOUT", total)
    monkeypatch.setattr(agent_module, "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", total)
    if lifecycle_module is not None:
        monkeypatch.setattr(lifecycle_module, "SHUTDOWN_TIMEOUT", total)
    actual_prefix, reserved = agent_module._resolve_shutdown_budget(tail)
    assert actual_prefix >= prefix and reserved >= tail
