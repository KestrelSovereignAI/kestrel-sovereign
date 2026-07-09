"""Shared local/cloud embedding space (#2290).

Covers the acceptance gates the issue declares:

- ``space_id`` keyed on the embedding MODEL identity + dim (``<model>@<dim>``),
  not the serving route — two routes serving the same pinned model are one space.
- Parity-probe pass/fail paths with mock providers returning identical vs
  drifted vectors, plus the error paths (missing embedding, dimension mismatch).
- Privacy-gated member selection: a ``force_local_only`` session resolves to the
  LOCAL member only, while both members stamp the SAME shared ``space_id``.
- Dims-pin enforcement: a member serving a different dim than the pin declares
  fails the probe (both sides must pin the same dims value).
- The verified-only alias: ``get_embedding_service`` applies the shared
  ``space_id`` only AFTER the parity probe passes; an unverified pin keeps its
  route-scoped id.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.sqla.embedding_profile import (
    _clear_profile_upsert_cache_for_tests,
    record_space_parity,
)

from kestrel_sovereign.llm.embedding_space import (
    DEFAULT_PARITY_CANARIES,
    EmbeddingSpaceConfigError,
    EmbeddingSpacePin,
    parse_embedding_space_pins,
    probe_parity,
)
from kestrel_sovereign.llm.embedding_service import ProviderEmbeddingService
from kestrel_sovereign.llm.service import LLMService


# --- fixtures ----------------------------------------------------------------


class _FakeAdapter:
    """Adapter that returns a fixed vector regardless of text.

    ``embedding_space_id`` returns ``None`` so the shared-space override (not an
    adapter-derived id) is the thing under test.
    """

    def __init__(self, vector):
        self._vector = list(vector)

    async def aembed(self, client, text, *, model=None, **kwargs):
        return list(self._vector)

    async def aembed_batch(self, client, texts, *, model=None, **kwargs):
        return [list(self._vector) for _ in texts]

    def embedding_space_id(self):
        return None


def _provider(
    name,
    vendor,
    *,
    is_local,
    vector=(0.1, 0.2, 0.3),
    model="qwen3-embedding-0.6b",
    dim=768,
    supports=True,
):
    return {
        "name": name,
        "vendor": vendor,
        "is_local": is_local,
        "model": model,
        "adapter": _FakeAdapter(vector),
        "client": SimpleNamespace(),
        "capabilities": {
            "supports_embeddings": supports,
            "embedding_model": model,
            "embedding_dim": dim,
        },
    }


def _service(providers, pins):
    svc = LLMService.__new__(LLMService)
    svc.providers = list(providers)
    svc._available_providers = lambda: list(providers)
    svc._embedding_space_pins = list(pins)
    svc._verified_space_pins = {}
    svc._embedding_route = None
    svc.disabled = False
    svc._force_local_only_provider = None
    return svc


QWEN_PIN = EmbeddingSpacePin(
    name="qwen3",
    model="qwen3-embedding-0.6b",
    dim=768,
    members=("ollama:local", "openrouter:api"),
)


# --- space_id keyed on model identity ----------------------------------------


def test_space_id_keyed_on_model_and_dim_not_route():
    """The space key is ``<model>@<dim>`` — no ``ollama:`` / ``openrouter:``."""
    assert QWEN_PIN.space_id == "qwen3-embedding-0.6b@768"
    assert "ollama" not in QWEN_PIN.space_id
    assert "openrouter" not in QWEN_PIN.space_id


def test_matryoshka_dim_changes_the_space():
    """Same model at a different (truncated) dim is a DIFFERENT space."""
    a = EmbeddingSpacePin("q", "qwen3-embedding-0.6b", 768, ("ollama:local", "openrouter:api"))
    b = EmbeddingSpacePin("q", "qwen3-embedding-0.6b", 1024, ("ollama:local", "openrouter:api"))
    assert a.space_id != b.space_id


def test_covers_matches_full_route_and_bare_vendor():
    pin = EmbeddingSpacePin(
        "q", "m", 768, ("ollama:local", "openrouter"),
    )
    assert pin.covers("ollama:local", "ollama") is True
    assert pin.covers("openrouter:api", "openrouter") is True  # bare-vendor member
    assert pin.covers("openai:api", "openai") is False


# --- config parsing ----------------------------------------------------------


def test_parse_valid_pin():
    cfg = {
        "embedding_spaces": {
            "qwen3": {
                "model": "qwen3-embedding-0.6b",
                "dim": 768,
                "members": ["ollama:local", "openrouter:api"],
                "parity_threshold": 0.97,
            }
        }
    }
    pins = parse_embedding_space_pins(cfg)
    assert len(pins) == 1
    assert pins[0].space_id == "qwen3-embedding-0.6b@768"
    assert pins[0].members == ("ollama:local", "openrouter:api")
    assert pins[0].parity_threshold == 0.97


def test_parse_no_spaces_returns_empty():
    assert parse_embedding_space_pins({}) == []
    assert parse_embedding_space_pins({"embedding_spaces": {}}) == []
    assert parse_embedding_space_pins("not a dict") == []


@pytest.mark.parametrize(
    "entry",
    [
        {"dim": 768, "members": ["a:b", "c:d"]},                 # missing model
        {"model": "m", "members": ["a:b", "c:d"]},               # missing dim
        {"model": "m", "dim": "eight", "members": ["a:b", "c:d"]},  # bad dim
        {"model": "m", "dim": -1, "members": ["a:b", "c:d"]},     # non-positive dim
        {"model": "m", "dim": 768, "members": ["a:b"]},           # <2 members
        {"model": "m", "dim": 768, "members": ["a:b", "c:d"], "parity_threshold": 2.0},
    ],
)
def test_parse_malformed_raises_loudly(entry):
    with pytest.raises(EmbeddingSpaceConfigError):
        parse_embedding_space_pins({"embedding_spaces": {"bad": entry}})


# --- parity probe ------------------------------------------------------------


async def test_probe_parity_pass_identical_vectors():
    a = ProviderEmbeddingService(_provider("ollama:local", "ollama", is_local=True))
    b = ProviderEmbeddingService(_provider("openrouter:api", "openrouter", is_local=False))
    result = await probe_parity(a, b, threshold=0.98)
    assert result.passed is True
    assert result.min_cosine == pytest.approx(1.0)
    assert result.n == len(DEFAULT_PARITY_CANARIES)
    assert result.drift == pytest.approx(0.0)


async def test_probe_parity_fail_drifted_vectors():
    """Orthogonal servings fall well below threshold → refused."""
    a = ProviderEmbeddingService(_provider("ollama:local", "ollama", is_local=True, vector=(1.0, 0.0)))
    b = ProviderEmbeddingService(_provider("openrouter:api", "openrouter", is_local=False, vector=(0.0, 1.0)))
    result = await probe_parity(a, b, threshold=0.98)
    assert result.passed is False
    assert result.min_cosine < 0.98


async def test_probe_parity_dimension_mismatch_is_error_not_crash():
    a = ProviderEmbeddingService(_provider("ollama:local", "ollama", is_local=True, vector=(1.0, 0.0, 0.0)))
    b = ProviderEmbeddingService(_provider("openrouter:api", "openrouter", is_local=False, vector=(1.0, 0.0)))
    result = await probe_parity(a, b, threshold=0.98)
    assert result.passed is False
    assert "dimension mismatch" in (result.error or "")


async def test_probe_parity_missing_embedding_is_error():
    class _NoneService:
        async def aembed_batch(self, texts):
            return [None for _ in texts]

    a = ProviderEmbeddingService(_provider("ollama:local", "ollama", is_local=True))
    result = await probe_parity(a, _NoneService(), threshold=0.98)
    assert result.passed is False
    assert result.error


# --- service integration: verified-only alias --------------------------------


async def test_get_embedding_service_applies_space_only_when_verified():
    providers = [
        _provider("ollama:local", "ollama", is_local=True),
        _provider("openrouter:api", "openrouter", is_local=False),
    ]
    svc = _service(providers, [QWEN_PIN])
    svc.resolve_embedding_provider = lambda: providers[0]

    # Unverified: keeps its own route-scoped id (adapter/capability derived).
    unverified = svc.get_embedding_service()
    assert unverified._space_id != QWEN_PIN.space_id

    # Verify, then the shared model-identity space_id is applied.
    results = await svc.verify_embedding_space_parity()
    assert results["qwen3"].passed is True
    verified = svc.get_embedding_service()
    assert verified._space_id == QWEN_PIN.space_id


async def test_dims_pin_enforced_member_with_wrong_dim_fails():
    providers = [
        _provider("ollama:local", "ollama", is_local=True, dim=512),  # wrong dim
        _provider("openrouter:api", "openrouter", is_local=False, dim=768),
    ]
    svc = _service(providers, [QWEN_PIN])
    results = await svc.verify_embedding_space_parity()
    assert results["qwen3"].passed is False
    assert "dim" in (results["qwen3"].error or "").lower()
    # Refused pin is not cached → alias not applied.
    assert "qwen3" not in svc._verified_space_pins


async def test_verify_fails_when_fewer_than_two_members_available():
    providers = [_provider("ollama:local", "ollama", is_local=True)]  # only one
    svc = _service(providers, [QWEN_PIN])
    results = await svc.verify_embedding_space_parity()
    assert results["qwen3"].passed is False


# --- privacy-gated member selection ------------------------------------------


async def test_privacy_gated_members_stamp_the_same_space():
    """A local-only session resolves to the LOCAL member and a cloud-allowed
    session to the CLOUD member — but BOTH stamp the pinned shared space_id, so
    their rows are mutually visible in kNN."""
    providers = [
        _provider("ollama:local", "ollama", is_local=True),
        _provider("openrouter:api", "openrouter", is_local=False),
    ]
    svc = _service(providers, [QWEN_PIN])
    await svc.verify_embedding_space_parity()
    assert svc._verified_space_pins["qwen3"].passed is True

    # Local-only session → local member resolves and stamps the shared space.
    svc._embedding_route = "ollama:local"
    svc._force_local_only_provider = lambda: True
    local = svc.get_embedding_service()
    assert local.provider["name"] == "ollama:local"
    assert local._space_id == QWEN_PIN.space_id

    # Cloud-allowed session → cloud member resolves and stamps the SAME space.
    svc._embedding_route = "openrouter:api"
    svc._force_local_only_provider = lambda: False
    cloud = svc.get_embedding_service()
    assert cloud.provider["name"] == "openrouter:api"
    assert cloud._space_id == QWEN_PIN.space_id
    assert local._space_id == cloud._space_id


async def test_local_only_session_refuses_cloud_embedding_route():
    """A cloud embedding_route under force_local_only is refused (keyword
    fallback) rather than leaking plaintext to the cloud — privacy wins."""
    providers = [
        _provider("ollama:local", "ollama", is_local=True),
        _provider("openrouter:api", "openrouter", is_local=False),
    ]
    svc = _service(providers, [QWEN_PIN])
    svc._embedding_route = "openrouter:api"
    svc._force_local_only_provider = lambda: True
    assert svc.resolve_embedding_provider() is None
    assert svc.get_embedding_service() is None


# --- durable parity: persistence + restart hydration (P1/P2) -----------------


@pytest.fixture
async def profiles_db():
    _clear_profile_upsert_cache_for_tests()
    with tempfile.TemporaryDirectory() as tmp:
        database = await AsyncDatabase.sqlite(str(Path(tmp) / "profiles.db"))
        yield database
        await database.close()


async def _parity_cosine_rows(database):
    return await database.fetchall(
        "SELECT space_id, parity_cosine FROM embedding_profiles "
        "WHERE parity_cosine IS NOT NULL ORDER BY space_id",
        (),
    )


async def test_record_space_parity_upserts_canonical_row_when_none_exists(profiles_db):
    """The verify probe runs BEFORE any shared-space rows exist, so a plain
    UPDATE-by-space_id would record nothing. record_space_parity must upsert a
    canonical row so the measured drift is durably recorded (P2)."""
    assert await _parity_cosine_rows(profiles_db) == []

    await record_space_parity(
        profiles_db,
        space_id=QWEN_PIN.space_id,
        model=QWEN_PIN.model,
        dim=QWEN_PIN.dim,
        normalized=QWEN_PIN.normalized,
        parity_cosine=0.993,
    )

    rows = await _parity_cosine_rows(profiles_db)
    assert len(rows) == 1
    assert rows[0][0] == QWEN_PIN.space_id
    assert abs(float(rows[0][1]) - 0.993) < 1e-6


async def test_verify_persists_parity_and_survives_restart(profiles_db):
    """End-to-end: a passing probe records parity, and a fresh service (a
    restart — empty _verified_space_pins) re-applies the shared space by
    hydrating from the persisted record instead of requiring a re-probe (P1)."""
    providers = [
        _provider("ollama:local", "ollama", is_local=True),
        _provider("openrouter:api", "openrouter", is_local=False),
    ]
    svc = _service(providers, [QWEN_PIN])
    results = await svc.verify_embedding_space_parity(record_to=profiles_db)
    assert results["qwen3"].passed is True

    rows = await _parity_cosine_rows(profiles_db)
    assert any(r[0] == QWEN_PIN.space_id for r in rows)

    # Simulate a restart: brand-new service, pins parse again but the verified
    # map starts empty. Without hydration the shared space_id would be dropped.
    restarted = _service(providers, [QWEN_PIN])
    restarted.resolve_embedding_provider = lambda: providers[0]
    assert restarted._verified_space_pins == {}

    await restarted.hydrate_verified_space_pins(profiles_db)
    assert restarted._verified_space_pins["qwen3"].passed is True
    # The shared space_id is applied again with no manual re-probe.
    assert restarted.get_embedding_service()._space_id == QWEN_PIN.space_id


async def test_hydration_invalidates_when_threshold_raised(profiles_db):
    """A recorded parity that no longer clears the pin's CURRENT threshold must
    NOT re-verify on restart — raising the bar invalidates a stale alias."""
    await record_space_parity(
        profiles_db,
        space_id=QWEN_PIN.space_id,
        model=QWEN_PIN.model,
        dim=QWEN_PIN.dim,
        normalized=QWEN_PIN.normalized,
        parity_cosine=0.985,
    )
    strict_pin = EmbeddingSpacePin(
        name="qwen3",
        model=QWEN_PIN.model,
        dim=QWEN_PIN.dim,
        members=QWEN_PIN.members,
        parity_threshold=0.99,  # above the recorded 0.985
    )
    svc = _service([], [strict_pin])
    await svc.hydrate_verified_space_pins(profiles_db)
    assert "qwen3" not in svc._verified_space_pins
