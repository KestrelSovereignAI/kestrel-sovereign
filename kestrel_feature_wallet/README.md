# kestrel-feature-wallet

Multi-currency agent wallet with Stripe on-ramp for Kestrel Sovereign. Supports FIL, USDC, and USDT with on-chain balance sync, economic gates, cryostasis threshold monitoring, and fiat-to-crypto conversions via Stripe.

## Installation

```bash
uv pip install git+https://github.com/KestrelSovereignAI/kestrel-feature-wallet.git
```

With EVM chain support:

```bash
uv pip install "kestrel-feature-wallet[evm] @ git+https://github.com/KestrelSovereignAI/kestrel-feature-wallet.git"
```

With Stripe on-ramp:

```bash
uv pip install "kestrel-feature-wallet[stripe] @ git+https://github.com/KestrelSovereignAI/kestrel-feature-wallet.git"
```

## Dependencies

- `kestrel-sovereign-sdk`
- `kestrel-sovereign`
- `aiosqlite>=0.21.0`
- `httpx>=0.27.0`
- `cryptography>=45.0.5`
- Optional: `web3>=7.0.0` (via `[evm]`), `stripe>=10.0.0` (via `[stripe]`)

## Usage

Once installed, the `WalletFeature` is automatically discovered by kestrel-sovereign via the `kestrel_sovereign.features` entry point.

## Configuration

| Variable | Description |
|----------|-------------|
| `STRIPE_API_KEY` | Stripe API key (for on-ramp) |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook secret (for on-ramp) |

## Development

```bash
uv pip install kestrel-sovereign-sdk && uv pip install -e .
uv run pytest
```
