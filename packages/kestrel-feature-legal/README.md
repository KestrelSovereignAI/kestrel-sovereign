# kestrel-feature-legal

Wyoming DAO LLC formation for Kestrel sovereign agents. Generates Articles of Organization and Operating Agreements that map Kestrel constitutional governance to the Wyoming legal framework, enabling agents to autonomously incorporate as legal entities.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-feature-legal.git
```

## Dependencies

- `kestrel-sovereign-sdk`

## Usage

```python
from kestrel_feature_legal.wyoming_dao import WyomingDAOGenerator
from kestrel_feature_legal.operating_agreement import OperatingAgreementGenerator

# Generate Articles of Organization
generator = WyomingDAOGenerator()
articles = generator.generate(agent_name="MyAgent", agent_did="did:pkh:...")

# Generate Operating Agreement
oa_gen = OperatingAgreementGenerator()
agreement = oa_gen.generate(agent_name="MyAgent", constitution=constitution)
```

## Configuration

No environment variables required.

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e ".[test]"
uv run pytest
```
