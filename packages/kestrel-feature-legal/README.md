# kestrel-feature-legal

Wyoming DAO LLC formation for Kestrel sovereign agents.

## Installation

```bash
pip install kestrel-feature-legal
```

## Features

- **Articles of Organization Generator** — complete Wyoming DAO LLC formation documents
- **Operating Agreement Generator** — maps Kestrel constitutional governance to Wyoming legal framework
- **Incorporation Tool** — agent tool for autonomous document generation
- **Legal Entity Model** — portable entity status that travels with agent identity

## Usage

```python
from kestrel_feature_legal import (
    generate_articles,
    generate_operating_agreement,
    generate_incorporation_package,
    RegisteredAgentInfo,
    OrganizerInfo,
)

# Create registered agent and organizer
ra = RegisteredAgentInfo(
    name="Wyoming Agents, Inc.",
    physical_address="1712 Pioneer Ave, Cheyenne, WY 82001",
)
organizer = OrganizerInfo(name="Jane Doe", address="456 Oak St, Denver, CO 80202")

# Generate articles
articles = generate_articles(
    entity_name="Kestrel Alpha DAO LLC",
    agent_did="did:pkh:eip155:1:0xABCDEF...",
    constitution_hash="a1b2c3d4...",
    registered_agent=ra,
    organizer=organizer,
)
```

## Dependencies

- `kestrel-sovereign` (core framework)
