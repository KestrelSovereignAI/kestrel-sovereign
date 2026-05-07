# KV Cache Quantization Benchmark

## Mac Studio M3 Ultra 512GB — Kimi K2.5 (1T params, Q2_K_XL)

**Date:** March 2026
**Model:** Kimi K2.5 (DeepSeek2 MLA architecture, 384x14B MoE, 1T total params)
**Quantization:** Q2_K_XL GGUF (~375GB weights)
**Server:** llama.cpp via Metal (Apple Silicon)
**Context:** 131,072 tokens, 4 parallel slots, `--kv-unified`

---

## Problem

With default f16 KV cache, Kimi K2.5 consumes ~505GB of 512GB unified memory, leaving virtually nothing for other workloads (talon issue processing, LoRA training, embeddings).

## TurboQuant (Google, ICLR 2026)

Google's TurboQuant algorithm promises 5x KV cache compression with zero accuracy loss via PolarQuant (coordinate rotation + Lloyd-Max codebook) and QJL (1-bit residual correction). Paper tested on Gemma, Mistral, and Llama-3.1-8B on H100 GPUs. Community ports exist for llama.cpp (Metal) and MLX.

### TurboQuant Results on Kimi K2.5 Q2_K_XL

| Cache Type | Block Size | Loads? | Quality | Notes |
|---|---|---|---|---|
| `turbo3` (3.25-bit) | 32 | Yes | **BROKEN** | Garbled output — unicode garbage, no coherent text |
| `turbo4` (4.25-bit) | 128 | **No** | N/A | `block size 128 does not divide n_embd_head_k=576` |

**Why TurboQuant fails on THIS model:**

1. **turbo3** — The model weights are already 2-bit (Q2_K_XL). Stacking 3.25-bit KV cache quantization creates a double compression effect. TurboQuant was benchmarked on full-precision or mildly quantized models. The signal-to-noise ratio at Q2 is too degraded for turbo3's rotation-based compression to preserve meaning.

2. **turbo4** — Kimi K2.5 uses DeepSeek2 MLA (Multi-head Latent Attention) with `n_embd_head_k=576`. turbo4 uses block size 128, and 576 % 128 ≠ 0 (`576 = 2⁶ × 3²`). Hard architectural incompatibility.

### TurboQuant — Models It DOES Work On

Community testing confirms TurboQuant works on these models (as of March 2026):

| Model | Weights Quant | Hardware | Result |
|---|---|---|---|
| Qwen3.5-35B-A3B (MoE) | Various | M5 Max Metal | 6/6 NIAH, 4.9x compression |
| Qwen3.5-27B Dense | Q6_K | RTX 5090 CUDA | 4.6x compression, 6/6 NIAH |
| Qwen3.5-397B-A17B | Various | CPU | 4.4x compression |
| **Llama-3.3-70B-Instruct** | **Q4_K_M** | CPU | **Works with quantized weights** |
| Mistral-7B | fp16 | Various | Paper benchmark model |
| Qwen2.5-7B/3B/1.5B | Various | CPU | MSE=0.034 matches paper |
| GPT-2 (124M), Phi-2 (2.8B) | fp16 | CPU | Validation tests pass |

**Key takeaway:** TurboQuant works with Q4_K_M quantized weights (Llama-3.3-70B). It fails on Q2_K_XL. The threshold is somewhere between Q2 and Q4. Head dimensions of 64 and 128 work; 576 (DeepSeek2 MLA) does not.

**Not tested by anyone (conspicuously absent):** DeepSeek models, any MLA architecture. Our failure may be the first documented MLA + TurboQuant attempt.

## Standard KV Cache Quantization Results

llama.cpp has built-in KV cache quantization (no fork needed). Quantizes KV cache entries in-place using same quant types as model weights.

### Our Benchmarks (Kimi K2.5 Q2_K_XL)

| Cache Type | RSS (GB) | Available (GB) | Speed (tok/s) | Notes |
|---|---|---|---|---|
| `f16` (default) | ~505* | ~7* | ~20* | Would OOM with 4 slots |
| `q8_0` (8-bit) | 353.7 | **150** | 20.6 | Safest option |
| `q4_0` (4-bit) | 346.2 | **158** | 20.5 | Most headroom |
| `turbo3` (3.25-bit) | 351.0 | 153 | 17.1 | BROKEN — garbled output |
| `turbo4` (4.25-bit) | N/A | N/A | N/A | Won't load (block size) |

\* f16 estimated — would require ~505GB, leaving only ~7GB free with 4 slots.

### Quality: q4_0 vs q8_0 vs f16

**Our simple test** (capital of France): Both q4_0 and q8_0 answered correctly with coherent reasoning.

**Published perplexity data from community benchmarks** (Qwen 2.5 Coder 7B, Q6_K weights):

| KV Cache | Perplexity | Degradation vs f16 |
|---|---|---|
| f16 | 8.3891 | baseline |
| q8_0 | 8.3934 | **+0.05%** (negligible) |
| q4_0 | ~8.60* | **~2.5-3%** (measurable) |

\* q4_0 perplexity estimated from reported +0.21-0.25 increase.

**Honest assessment:**
- **q8_0 is essentially lossless** (~0.05% degradation, within measurement noise)
- **q4_0 has real quality loss** (~2.5-3% perplexity increase) — our "Paris" test was too simple to detect this. On harder reasoning, long-context retrieval, or code generation tasks, this could matter.
- We did NOT run perplexity benchmarks ourselves. These numbers are from other people testing different models. The degradation on Kimi Q2_K_XL could be worse (double quantization compounding).

## Recommendation

**Use `q8_0` as default** — safest choice with essentially zero quality loss and 150GB freed.

Switch to `q4_0` only if you need the extra 8GB and can tolerate ~3% quality degradation, or for non-critical workloads.

| Option | Free RAM | Quality Loss | Best For |
|---|---|---|---|
| `q8_0` | 150 GB | ~0.05% | Production, reasoning, coding |
| `q4_0` | 158 GB | ~2.5-3% | Bulk processing, less critical tasks |

### Start Command

```bash
# Recommended (q8_0 — lossless)
llama-server \
    --model "$KIMI_MODEL_DIR/UD-Q2_K_XL/Kimi-K2.5-UD-Q2_K_XL-00001-of-00008.gguf" \
    --ctx-size 131072 \
    --port 8001 \
    --kv-unified \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --reasoning-format deepseek
```

Or use the start script in your local model directory (e.g. `bash "$KIMI_MODEL_DIR/../start-kimi.sh"`).

## What the Freed 150GB Enables

With q8_0 KV cache, the Mac Studio can run multiple workloads simultaneously:

| Workload | Memory Needed | Purpose |
|---|---|---|
| Kimi K2.5 inference | ~354 GB | Primary LLM (always resident) |
| kestrel-talon daemon | ~1 GB | Continuous GitHub issue processing |
| SDXL LoRA training (MPS) | ~8-10 GB | Frinz companion avatar training |
| Ollama nomic-embed-text | ~0.3 GB | Background embedding computation |
| macOS + overhead | ~8 GB | System |
| **Total** | **~373 GB** | **~139 GB still free** |

## TurboQuant Fork

A TurboQuant-enabled llama.cpp build (branch `feature/turboquant-kv-cache` from `TheTom/llama-cpp-turboquant`) was checked out as a sibling of this repo for testing. Builds and runs on Metal but turbo3/turbo4 are not usable with Kimi K2.5 Q2_K_XL.

**Worth revisiting when:**
- We run a different primary model (Qwen3.5, Llama 3.3) with Q4_K_M+ weights
- TurboQuant adds support for non-standard head dimensions (576)
- We upgrade to higher-quality weight quantization (Q4_K_M or better) that can tolerate the additional KV compression

## Open Questions

1. **Perplexity on our actual model:** We haven't run `llama-perplexity` with q8_0 vs q4_0 on Kimi K2.5 specifically. The 2.5-3% figure is from Qwen 2.5 benchmarks — it could be worse on Q2_K_XL.
2. **q5_0 / q5_1:** Untested middle ground between q4_0 and q8_0. Might offer better quality/memory tradeoff.
3. **Mixed K/V types:** Could run K cache at q4_0 and V cache at q8_0 (K is more compressible). Not tested.
4. **2-slot mode:** Using `--parallel 2` instead of 4 would halve KV cache memory, potentially allowing f16 KV.

## References

- [Google Research Blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [arXiv Paper: TurboQuant](https://arxiv.org/html/2504.19874v1)
- [llama.cpp TurboQuant Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [llama.cpp KV Cache Discussion #5932](https://github.com/ggml-org/llama.cpp/discussions/5932)
- [K/V Context Quantisation to Ollama](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/) — perplexity benchmarks
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (Metal fork)
- [TurboQuant Substack analysis](https://kaitchup.substack.com/p/turboquant-finally-fast-and-widely) — model compatibility
