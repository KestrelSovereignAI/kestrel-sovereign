# Kestrel Multi-Chain Wallet System

This document describes the wallet system that enables Kestrel agents to manage cryptocurrency across multiple blockchain networks.

## Overview

The Kestrel wallet system provides:
- **Multi-chain support**: Filecoin FEVM, Ethereum, Polygon
- **Native token transfers**: Send ETH, FIL, MATIC
- **ERC-20 token support**: Send USDC, USDT, and other tokens
- **Fiat on-ramp**: Purchase crypto with credit card (Stripe)
- **Security controls**: Spending limits, mainnet blocking, user approval

## Architecture

```
User: "!wallet-send 1 FIL to 0x..."
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  WalletFeature.wallet_send()                        │
│  - Prepare transaction                              │
│  - Estimate gas                                     │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  TransactionSecurityHook (priority=5)               │
│  - Validate address format                          │
│  - Block mainnet if disabled                        │
│  - Check spending limits                            │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  SecurityHook (priority=10) + ApprovalQueue         │
│  - Queue for user approval                          │
│  - User chooses: once / session / always            │
└─────────────────────────────────────────────────────┘
         │ (after approval)
         ▼
┌─────────────────────────────────────────────────────┐
│  TransactionManager.execute_transaction()           │
│  - Load private key from SecureKeyStorage           │
│  - Sign with web3.py                                │
│  - Broadcast via JSON-RPC                           │
│  - Log to audit trail                               │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  EVMAdapter (web3.py)                               │
│  - Filecoin FEVM, Ethereum, Polygon                 │
│  - Unified signing for all EVM chains               │
└─────────────────────────────────────────────────────┘
```

## Supported Networks

| Network | Chain ID | Native Token | Testnet |
|---------|----------|--------------|---------|
| Filecoin Calibration | 314159 | tFIL | ✅ |
| Filecoin Mainnet | 314 | FIL | ❌ |
| Ethereum Sepolia | 11155111 | ETH | ✅ |
| Ethereum Mainnet | 1 | ETH | ❌ |
| Polygon Amoy | 80002 | MATIC | ✅ |
| Polygon Mainnet | 137 | MATIC | ❌ |

## Agent Commands

### Send Native Tokens
```
!wallet-send <to_address> <amount> [network]
```
Example: `!wallet-send 0x742d35Cc6634C0532925a3b844Bc9e7595f3E123 0.1 ethereum_sepolia`

### Send ERC-20 Tokens
```
!wallet-send-token <to_address> <amount> <token_symbol> [network]
```
Example: `!wallet-send-token 0x... 100 USDC ethereum_sepolia`

### List Networks
```
!wallet-networks
```
Shows all supported networks with connection status.

### Transaction History
```
!wallet-tx-history [limit]
```
Shows recent blockchain transactions (default: 10).

### Fund Wallet (Fiat On-Ramp)
```
!wallet-fund [amount] [currency]
```
Opens Stripe on-ramp to purchase crypto with credit card.

## Supported ERC-20 Tokens

### Ethereum Sepolia
| Token | Address | Decimals |
|-------|---------|----------|
| USDC | 0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238 | 6 |
| USDT | 0xaA8E23Fb1079EA71e0a56F48a2aA51851D8433D0 | 6 |

### Polygon Amoy
| Token | Address | Decimals |
|-------|---------|----------|
| USDC | 0x41E94cFAEd3F3B7e1b6195Cd2816830010854685 | 6 |
| USDT | 0xc85F14b050B277c7aCB8FC96f26c8a9538EaB662 | 6 |

### Mainnet Tokens
Mainnet token addresses are also registered for when mainnet is enabled.

## Security Model

### Default Configuration
- **Mainnet blocked**: Requires `KESTREL_ALLOW_MAINNET=true`
- **Daily limit**: $100 USD (configurable)
- **Approval required**: All transactions need user approval

### TransactionSecurityHook
Runs at priority 5 (before general SecurityHook):
1. Validates address format for target network
2. Blocks mainnet transactions unless explicitly enabled
3. Estimates USD value and checks spending limits
4. Logs all transaction attempts

### Spending Limits
- Default: $100 USD per day
- Configurable via `KESTREL_TX_DAILY_LIMIT_USD`
- Resets at midnight UTC
- Tracked in SQLite audit database

### Audit Trail
All transactions are logged to `tx_audit.db`:
- Transaction hash
- From/to addresses
- Amount and token
- Network and status
- Timestamp
- USD value estimate

## Fiat On-Ramp (Stripe)

### Setup
1. Set `STRIPE_SECRET_KEY` environment variable
2. Set `STRIPE_WEBHOOK_SECRET` for webhook verification
3. Configure webhook endpoint in Stripe dashboard

### Webhook Endpoint
```
POST /webhooks/stripe/crypto
```

Handles events:
- `crypto.onramp_session.updated` - Status changes
- `crypto.onramp_session.completed` - Purchase complete

### Supported Currencies
- ETH (Ethereum)
- MATIC (Polygon)

Note: Stripe doesn't currently support FIL directly.

## Environment Variables

```bash
# Required
KESTREL_DATA_KEY=<fernet-key>           # For key encryption

# Transaction Signing
KESTREL_ALLOW_MAINNET=false             # Enable mainnet (DANGEROUS)
KESTREL_TX_DAILY_LIMIT_USD=100          # Daily spending limit
KESTREL_TX_REQUIRE_APPROVAL=true        # Require approval

# Fiat On-Ramp
STRIPE_SECRET_KEY=sk_...                # Stripe API key
STRIPE_PUBLISHABLE_KEY=pk_...           # For frontend widget
STRIPE_WEBHOOK_SECRET=whsec_...         # Webhook signing

# Optional RPC Overrides
FILECOIN_CALIBRATION_RPC=...
ETHEREUM_SEPOLIA_RPC=...
POLYGON_AMOY_RPC=...
```

## Key Management

### SecureKeyStorage
Private keys are encrypted at rest using:
- AES-256-GCM encryption
- PBKDF2-SHA256 key derivation (600,000 iterations)
- Per-key salt and nonce

### Key Generation
```python
from features.wallet import FilecoinKeyManager

manager = FilecoinKeyManager(storage_path="wallet.db")
address = manager.generate_key(label="main")
# Returns EVM-compatible address: 0x...
```

## File Structure

```
features/wallet/
├── __init__.py                 # Package exports
├── feature.py                  # WalletAgent (internal ledger)
├── wallet_feature.py           # WalletFeature (agent tools)
├── filecoin_keys.py           # Key generation
├── filecoin_testnet.py        # Legacy Filecoin adapter
├── transaction_manager.py      # Transaction orchestration
├── transaction_hook.py         # Security hook
├── chain_adapters/
│   ├── __init__.py
│   ├── base.py                # ChainNetwork, NetworkConfig, ABCs
│   ├── evm_adapter.py         # web3.py integration
│   ├── erc20.py               # ERC-20 token support
│   └── token_registry.py      # Known token addresses
└── onramp/
    ├── __init__.py
    ├── stripe_onramp.py       # Stripe integration
    └── webhook_handler.py     # Webhook processing
```

## Testing

### Unit Tests
```bash
pytest tests/integration/test_evm_transactions_e2e.py -v
```

### Real Network Tests
```bash
RUN_NETWORK_TESTS=true pytest tests/integration/test_evm_transactions_e2e.py -v
```

### With Funded Wallet
```bash
KESTREL_TEST_PRIVATE_KEY=0x... RUN_NETWORK_TESTS=true pytest tests/integration/test_evm_transactions_e2e.py::TestRealTransaction -v
```

## Getting Testnet Tokens

### Filecoin Calibration
- Faucet: https://faucet.calibnet.chainsafe-fil.io/

### Ethereum Sepolia
- Faucet: https://sepoliafaucet.com/
- Alchemy Faucet: https://sepoliafaucet.com/

### Polygon Amoy
- Faucet: https://faucet.polygon.technology/

### Test USDC (Sepolia)
The test USDC contract on Sepolia may require minting from Circle's testnet faucet.

## Future Enhancements

### Not Yet Implemented
- f1/t1 native Filecoin signing (requires CBOR)
- HD wallet / BIP-39 mnemonic support
- Multi-signature support
- Automatic gas price optimization
- Off-ramp (cash out to fiat)
- Additional on-ramp providers (MoonPay, Transak)

### Roadmap
1. **Q1 2026**: HD wallet support for key backup
2. **Q2 2026**: Multi-sig for high-value transactions
3. **Q3 2026**: Cross-chain bridges integration
4. **Q4 2026**: DeFi protocol integrations

## Troubleshooting

### "Mainnet transactions are blocked"
Set `KESTREL_ALLOW_MAINNET=true` in your environment.
⚠️ **Warning**: Mainnet uses real money!

### "Transaction exceeds daily limit"
Either:
1. Wait until midnight UTC for reset
2. Increase `KESTREL_TX_DAILY_LIMIT_USD`

### "Invalid address format"
Ensure the address:
- Starts with `0x`
- Is exactly 42 characters
- Contains only valid hex characters

### "Insufficient balance"
Get testnet tokens from the faucets listed above.

### Webhook not receiving events
1. Verify `STRIPE_WEBHOOK_SECRET` is set
2. Check Stripe dashboard for webhook logs
3. Ensure `/webhooks/stripe/crypto` is accessible

## Security Considerations

1. **Never commit private keys** to version control
2. **Use testnet** for development and testing
3. **Enable mainnet carefully** with appropriate limits
4. **Monitor spending** via the audit trail
5. **Rotate keys** if compromise is suspected

## References

- [Filecoin FEVM Docs](https://docs.filecoin.io/smart-contracts/fundamentals/filecoin-evm-runtime)
- [web3.py Documentation](https://web3py.readthedocs.io/)
- [Stripe Crypto On-Ramp](https://docs.stripe.com/crypto/onramp)
- [ERC-20 Standard](https://eips.ethereum.org/EIPS/eip-20)
- [EIP-1559 Transaction Format](https://eips.ethereum.org/EIPS/eip-1559)
