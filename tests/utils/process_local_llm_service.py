"""A real ``LLMService`` as an offline CLI builds it, before any agent state.

``LLMService()`` reads the host's configuration, so tests build one with
``__new__`` and only the state the embedding resolution path reads. Nothing
persisted in an agent's database has been applied to it, which is exactly the
starting point of ``kestrel embeddings reindex`` and
``kestrel constitution reanchor``.
"""

from __future__ import annotations

from kestrel_sovereign.llm.service import LLMService


class ProcessLocalService(LLMService):
    """A real ``LLMService`` whose ``close`` records the call.

    The real ``close`` drains route state this double never builds.
    """

    closed = False

    async def close(self) -> None:
        self.closed = True


def process_local_service(
    providers: list[dict], *, embedding_space_pins=None
) -> ProcessLocalService:
    """What ``LLMService()`` builds from static config, before any agent state.

    ``embedding_space_pins`` are the ``[llm.embedding_spaces]`` pins config
    declares; none is verified until a parity probe passes or is rehydrated.
    """
    service = ProcessLocalService.__new__(ProcessLocalService)
    service.providers = providers
    service.disabled = False
    service._disabled_routes = {}
    service._mandate_preference = {}
    service._mandate_fallbacks = []
    service._route_embedding_model_overrides = {}
    service._route_embedding_caps_backup = {}
    service._route_embedding_model_persistence_callback = None
    service._embedding_route_persistence_callback = None
    service._embedding_discovery_cache = []  # discovery finds nothing new
    service._embedding_space_change_warnings = {}
    service._corpus_embedding_profile_provider = None
    service._force_local_only_provider = None
    service._embedding_space_pins = embedding_space_pins
    service._verified_space_pins = {}
    service._embedding_route = None
    service._preference_persistence_tasks = set()
    return service
