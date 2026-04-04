# kestrel-feature-wallet — Agent Instructions

See [README.md](README.md) for package overview.

## Package Structure

```
kestrel_feature_wallet/
├── pyproject.toml
├── README.md
├── kestrel_feature_wallet/
│   ├── __init__.py
│   ├── wallet_feature.py       # WalletFeature entry point
│   ├── feature.py              # Feature registration
│   ├── economic_gates.py       # Economic gate checks
│   ├── transaction_manager.py  # Transaction lifecycle
│   ├── transaction_hook.py     # Hook for transaction events
│   ├── filecoin_testnet.py     # Filecoin testnet integration
│   ├── filecoin_keys.py        # Key management
│   ├── filecoin_tools.py       # FIL-specific tools
│   ├── multichain_tools.py     # Multi-chain operations
│   ├── base_l2_wallet.py       # L2 wallet base
│   ├── chain_adapters/         # Chain-specific adapters
│   │   ├── base.py
│   │   ├── evm_adapter.py
│   │   ├── erc20.py
│   │   └── token_registry.py
│   └── onramp/                 # Fiat on-ramp
│       ├── stripe_onramp.py
│       └── webhook_handler.py
```

## Entry Points

- `kestrel_sovereign.features`: `WalletFeature = "kestrel_feature_wallet.wallet_feature:WalletFeature"`

## Key Files to Read First

1. `kestrel_feature_wallet/wallet_feature.py` — Main feature class and tool registration
2. `kestrel_feature_wallet/economic_gates.py` — Economic gate logic
3. `kestrel_feature_wallet/transaction_manager.py` — Transaction lifecycle

## Running Tests

```bash
uv run pytest
```

## Agent-Specific Instructions

- Wallet operations must be atomic — use transactions
- Economic gates protect against overspending; never bypass them
- Cryostasis is triggered when funds drop below threshold — handle gracefully
