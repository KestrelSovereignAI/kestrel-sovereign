---
type: Architecture Spec
title: Filecoin Wallet Integration
description: '**See Also**: For the complete multi-chain wallet system with transaction
  signing, ERC-20 tokens, and fiat on-ramp, see **[WALLET_SYSTEM.md](WALLET_SYSTEM.md)**.'
resource: /docs/architecture/FILECOIN_WALLET.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Filecoin Wallet Integration

> **See Also**: For the complete multi-chain wallet system with transaction signing, ERC-20 tokens, and fiat on-ramp, see **[WALLET_SYSTEM.md](WALLET_SYSTEM.md)**.

Kestrel agents can have Filecoin wallets on the Calibration testnet for economic operations.

## Quick Start

### 1. Set Encryption Key

The encryption key protects private keys at rest:

```bash
# Generate and add to .env
export KESTREL_DATA_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
echo "KESTREL_DATA_KEY=$KESTREL_DATA_KEY" >> .env
```

### 2. Generate Address

```bash
source .env
uv run python -c "
import asyncio
from features.wallet import FilecoinKeyManager
from pathlib import Path

async def main():
    km = FilecoinKeyManager(storage_dir=Path('./agent_dbs'))
    addr, _ = await km.generate_address('my_agent')
    print(f'Address: {addr}')

asyncio.run(main())
"
```

Save the address to `.env`:
```bash
echo "FIL_ADDRESS=t1your_address_here" >> .env
```

### 3. Get Test FIL

Visit the ChainSafe faucet: https://faucet.calibnet.chainsafe-fil.io/

- Paste your `t1...` address
- Complete captcha
- Submit (gives 5 tFIL, 12-hour cooldown)

### 4. Verify Balance

```bash
source .env
uv run python -c "
import asyncio
from features.wallet import FilecoinTestnetAdapter

async def check():
    adapter = FilecoinTestnetAdapter()
    balance = await adapter.get_balance('$FIL_ADDRESS')
    print(f'Balance: {balance} FIL')
    await adapter.close()

asyncio.run(check())
"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `KESTREL_DATA_KEY` | Yes | Fernet key for encrypting private keys |
| `FIL_ADDRESS` | No | Your testnet address (for reference) |
| `RUN_NETWORK_TESTS` | No | Set to `true` to run network integration tests |

## Agent Commands

Once a Kestrel agent has a wallet configured:

| Command | Description |
|---------|-------------|
| `!wallet-generate-address` | Generate new Calibration testnet address |
| `!wallet-address` | Show current address with explorer link |
| `!wallet-sync` | Sync internal balance with on-chain balance |
| `!wallet-balance` | Show all currency balances |

## Architecture

### Components

- **FilecoinKeyManager** ([features/wallet/filecoin_keys.py](../../features/wallet/filecoin_keys.py))
  - Generates secp256k1 keypairs
  - Derives t1... addresses (BLAKE2b-160 hash)
  - Encrypts private keys using SecureKeyStorage

- **FilecoinTestnetAdapter** ([features/wallet/filecoin_testnet.py](../../features/wallet/filecoin_testnet.py))
  - RPC client for Calibration testnet
  - Uses Glif public endpoint: `https://api.calibration.node.glif.io/rpc/v1`
  - Read-only operations (balance queries)

- **WalletAgent** ([features/wallet/feature.py](../../features/wallet/feature.py))
  - Multi-currency internal ledger (FIL, USDC, USDT)
  - 90/10 main/audit balance split
  - Persists to SQLite

- **WalletFeature** ([features/wallet/wallet_feature.py](../../features/wallet/wallet_feature.py))
  - Agent tool interface
  - Economic gate methods (`is_paid_tier()`, `has_revenue_share()`)

### Security

- Private keys encrypted at rest with AES-256-GCM
- Key derivation: PBKDF2-SHA256 with 600,000 iterations
- Master key from `KESTREL_DATA_KEY` environment variable
- Mainnet blocked by default (safety)

### Network

- **Testnet**: Calibration (`t1...` addresses)
- **RPC**: Glif public endpoint
- **Explorer**: https://calibration.filfox.info/
- **Faucet**: https://faucet.calibnet.chainsafe-fil.io/

## Testing

```bash
# Run all wallet tests (no network required)
uv run pytest tests/integration/test_filecoin_testnet_e2e.py -v

# Run network tests (requires funded address)
export RUN_NETWORK_TESTS=true
source .env
uv run pytest tests/integration/test_filecoin_testnet_e2e.py -v
```

## Limitations

- **Filecoin-specific**: This adapter uses native Filecoin RPC
- **Testnet only**: Mainnet requires explicit opt-in
- **Single address**: One address per agent

## Multi-Chain Wallet System

The following features are now available in **[WALLET_SYSTEM.md](WALLET_SYSTEM.md)**:

- ✅ Transaction signing with web3.py (EVM chains)
- ✅ Multi-chain support (Filecoin FEVM, Ethereum, Polygon)
- ✅ ERC-20 token transfers (USDC, USDT)
- ✅ Fiat on-ramp (Stripe integration)
- ✅ Spending limits and approval flow
- ✅ Mainnet support with safety controls
