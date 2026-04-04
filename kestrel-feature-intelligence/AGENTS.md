# kestrel-feature-intelligence — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel-feature-intelligence/
├── pyproject.toml
├── README.md
├── kestrel_feature_intelligence/
│   ├── __init__.py
│   ├── reflection/
│   │   ├── feature.py            # ReflectionFeature entry point
│   │   ├── analyzer.py           # Reflection analysis engine
│   │   ├── approval.py           # Constitutional approval gates
│   │   ├── models.py             # Data models
│   │   ├── prompts.py            # LLM prompt templates
│   │   ├── self_model.py         # Agent self-model
│   │   ├── self_model_handler.py # Self-model event handler
│   │   ├── ticket_creator.py     # Auto-ticket creation
│   │   ├── ticket_handler.py     # Ticket processing
│   │   ├── prioritizer.py        # Insight prioritization
│   │   ├── economics.py          # Reflection cost tracking
│   │   ├── training.py           # Training data generation
│   │   ├── hooks.py              # Lifecycle hooks
│   │   ├── formatters.py         # Output formatters
│   │   └── db_helpers.py         # Database utilities
│   └── council/
│       ├── feature.py            # CouncilFeature entry point
│       ├── deliberation.py       # Multi-model deliberation
│       ├── evidence.py           # Evidence compilation
│       ├── costing.py            # Deliberation cost tracking
│       ├── models.py             # Data models
│       └── storage.py            # Session persistence
└── tests/
    ├── unit/
    └── integration/
```

## Entry Points

- `kestrel_sovereign.features`: `ReflectionFeature = "kestrel_feature_intelligence.reflection.feature:ReflectionFeature"`
- `kestrel_sovereign.features`: `CouncilFeature = "kestrel_feature_intelligence.council.feature:CouncilFeature"`

## Key Files to Read First

1. `kestrel_feature_intelligence/reflection/feature.py` — Reflection feature and tools
2. `kestrel_feature_intelligence/council/feature.py` — Council feature and tools
3. `kestrel_feature_intelligence/council/deliberation.py` — Multi-model deliberation logic

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- This package has two independent features that can be loaded separately
- Council deliberation calls multiple LLM providers — be mindful of API costs
- Reflection insights require constitutional approval before behavioral changes
