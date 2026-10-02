"""Decisions modality: typed choice / score / noul questions over a state.

Spec: ``docs/architecture/llm/DECISIONS.md``. The public entry point is
``LLMService.decide``; the contract types live in ``kestrel_sdk.llm.decisions``.
This package holds core's halves: config, resolution, fit, thresholds,
normalisation, and the shared systemone HTTP path. It imports nothing from the
LLM service so adapters can depend on it.
"""
