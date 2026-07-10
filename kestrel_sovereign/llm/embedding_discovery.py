"""Embedding-model discovery (#2338).

Chat models are discovered dynamically per vendor (``model_discovery.py`` /
``adapter.list_models``) with config acting only as cache/override. Embedding
models must work the same way rather than being gated on a hand-pinned
``embedding_model = "..."`` line: a route advertises embedding capability
because discovery *found* an embedding model for it, not because an operator
typed one into TOML.

This module holds the vendor-neutral shape (:class:`EmbeddingModelInfo`) that
each adapter's ``list_embedding_models`` facet returns, plus normalization
helpers used to compute shared-space candidates (#2290/#2337) by intersecting
locally-discovered embedding models with cloud-discovered ones.

The dedicated discovery sources per vendor (verified live 2026-07-10):

* **OpenRouter** — ``GET /api/v1/embeddings/models`` (a DEDICATED endpoint; the
  generic ``/models`` list does NOT include embedding models and
  ``?category=embedding`` returns empty).
* **Ollama** — ``/api/tags`` + ``/api/show`` per model, keeping only models
  whose reported ``capabilities`` include ``"embedding"``.
* **OpenAI** — ``/v1/models`` filtered to ``text-embedding-*`` ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddingModelInfo:
    """A discovered embedding model for a single ``(vendor, route)``.

    ``dim_options`` carries the upstream-reported dimension choices — the full
    Matryoshka truncation range for MRL-capable models (e.g. qwen3-embedding),
    or a single native dimension for fixed-size models. ``native_dim`` is the
    model's full/untruncated size. Both are ``None`` when the catalog payload
    doesn't report a dimension; the set-time probe (#2326) still proves the
    model actually serves before it is used.
    """

    id: str
    provider: str
    #: The full ``"<vendor>:<route>"`` name of the route that served this model.
    #: Embedding capability is ROUTE-specific in production (``openai:api`` can
    #: embed, ``openai:plan``/codex cannot), so discovery tags every model with
    #: its originating route — advertisement and resolution filter on it, never
    #: on the vendor alone, so one route's embeddings can't be attributed to a
    #: sibling route of the same vendor (#2338).
    route: str = ""
    display_name: str = ""
    native_dim: Optional[int] = None
    dim_options: List[int] = field(default_factory=list)
    context_limit: Optional[int] = None
    description: Optional[str] = None
    #: ``True`` when this entry came from a config pin (``embedding_model``)
    #: rather than live discovery. A pin OVERRIDES discovery (operator intent)
    #: but is never a *prerequisite* for a route to advertise embeddings.
    is_pinned: bool = False

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.id.split("/")[-1]
        # Keep native_dim inside the offered options so a UI dim picker always
        # includes the full size even when only a truncation range was reported.
        if self.native_dim is not None and self.native_dim not in self.dim_options:
            self.dim_options = sorted({*self.dim_options, self.native_dim})

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "provider": self.provider,
            "route": self.route,
            "display_name": self.display_name,
            "native_dim": self.native_dim,
            "dim_options": list(self.dim_options),
            "context_limit": self.context_limit,
            "description": self.description,
            "is_pinned": self.is_pinned,
        }


def normalize_embedding_model_id(model_id: str) -> str:
    """Reduce a model id to a vendor-neutral identity for cross-route matching.

    OpenRouter routes carry a ``vendor/`` prefix (``qwen/qwen3-embedding-0.6b``)
    while the same weights served locally by Ollama are bare
    (``qwen3-embedding-0.6b`` or tagged ``qwen3-embedding:0.6b``). Shared-space
    computation (#2290/#2337) must recognise those as the SAME model, so we drop
    the routing prefix, fold the Ollama ``:tag`` separator to ``-``, and
    lowercase.
    """
    if not model_id:
        return ""
    ident = model_id.split("/")[-1].strip().lower()
    # Ollama tags (``qwen3-embedding:0.6b``) → dashed form to match the
    # OpenRouter/HF id (``qwen3-embedding-0.6b``).
    ident = ident.replace(":", "-")
    return ident
