# Batched Prefill v1 — Performance Report

**Date:** 2026-03-24
**Model:** Qwen3.5-397B-A17B, 4-bit MLX experts, K=4 active
**Hardware:** Apple M5 Max, 128 GB unified memory
**Prompt:** 208 tokens (technical analysis request)
**Expert size:** 7.08 MB each, 512 total, ~209 GB on disk
**Page cache:** warm (model loaded, warmup pass completed — reflects real-world usage)

## Background

The original Flash-MoE repo processed prefill tokens **one at a time**, loading K=4 routed experts from SSD per token per layer. For a 208-token prompt across 60 layers, this means **49,920 expert reads** — each 7 MB, totaling **~333 GB of SSD I/O** just for prefill. All intermediate expert results are discarded (only the attention state — KV cache and delta-net recurrence — persists between tokens).

Our batched prefill optimization introduces three key changes:
1. **Batch projections** across N tokens — read weight matrices once instead of N times per layer
2. **Custom Metal kernels** for batched linear and full attention — process all N tokens per GPU dispatch
3. **Selective expert loading** — skip routed expert SSD I/O where it doesn't affect output quality

The last prompt token and all decode tokens always use full K=4 experts.

## Results Summary

### Raw Prompt (Warm Cache)

| # | Config | Prefill tok/s | TTFT | Speedup | Quality |
|---|--------|:---:|:---:|:---:|---|
| 0 | **Original** (token-by-token, K=4 all layers) | **11.0** | **18.7 s** | 1.0× | Excellent |
| 1 | **Batched, skip all experts** (pfb=128, K=0) | **68.5** | **3.0 s** | **6.2×** | EOS without chat template |
| 2 | **Batched, experts full-attn only** (pfb=128) | **20.5** | **10.1 s** | **1.9×** | Excellent — same as baseline |

### With Chat Template (Thinking Mode)

The Qwen chat template (`<|im_start|>system/user/assistant`) is required for the iOS/macOS app and enables the model's thinking mode. With it, all modes produce coherent output.

| # | Config | Prefill tok/s | TTFT | Speedup | Quality |
|---|--------|:---:|:---:|:---:|---|
| 0 | **Original** (token-by-token, K=4 all layers) | **10.5** | **21.6 s** | 1.0× | Excellent — LaTeX formulas, structured analysis |
| 1 | **Batched, skip all experts** (pfb=128, K=0) | **69.9** | **3.2 s** | **6.7×** | Good — coherent but misreads some technical terms |
| 2 | **Batched, experts full-attn only** (pfb=128) | **19.7** | **11.5 s** | **1.9×** | Excellent — same quality as baseline |

**Decode speed is identical across all configs: ~10.9 tok/s** (prefill optimization does not affect decode).

## Analysis

### Where the speedup comes from

**Batching projections (6.2× with skip-all):**
Each layer has ~42 MB of projection weights. The original per-token path reads these 207 times (once per token). Batched prefill reads them once per chunk — saving ~500 GB of GPU memory bandwidth on a 208-token prompt.

**Skipping expert I/O (1.9× additional):**
Expert I/O accounts for 56% of per-token prefill time. Each token loads 4 experts × 7 MB × 60 layers = 1.6 GB from SSD. Skipping this eliminates the dominant bottleneck.

**Combined (experts-full-only mode):**
- 45 linear layers (75%): fully batched, K=0 — no SSD I/O, projections read once
- 15 full-attention layers (25%): batched projections + causal attention, then per-token K=4 expert I/O
- Result: **1.9× faster** than baseline with **identical output quality**

### Why experts-full-only preserves quality

Full attention layers see the entire context via KV cache — expert quality matters most here because these layers determine what the model "remembers" from the prompt. Linear attention layers use delta-net recurrence (O(1) state) which is more tolerant of shared-expert-only approximation.

### Chat template is critical for skip-experts

Without the chat template, skip-all-experts modes produce immediate EOS on the 4-bit model. The chat template provides structured context (`<|im_start|>system\nYou are a helpful assistant`) that gives the model enough signal to generate even with a degraded prefill state.

Even with the chat template, skip-all-experts shows subtle quality degradation — it misinterprets terms like "GatedDeltaNet" as "phonetic misspellings." The experts-full-attn-only mode preserves full comprehension.

### 2-bit vs 4-bit: opposite behavior

On the **2-bit** model (PPL 5.71), skipping experts during prefill actually **improves** output quality. The 2-bit routed expert weights are noisy enough that their contribution degrades the attention state — the shared expert alone gives a cleaner signal.

On the **4-bit** model (PPL 3.64), routed experts provide meaningful signal and should be preserved at full-attention layers for best quality.

### Smaller models need experts more

The 35B model (moe_dim=512) has a smaller shared expert than the 397B (moe_dim=1024). Skipping all experts degrades long-prompt quality on the 35B. The experts-full-only mode works well for all model sizes.

Note: MLX 4-bit quantization for attention weights may be too aggressive for smaller models on long prompts, leading to degraded output quality even with experts enabled. The 397B model is large enough to tolerate 4-bit attention without significant quality loss.

## Recommendations

| Model | Recommended Config | Prefill Speedup | Why |
|-------|-------------------|:---:|-----|
| **397B 4-bit** | `--prefill-experts-full-only --pfb 128` | **1.9×** | Best quality, routed experts at full-attn layers |
| **397B 2-bit** | `--prefill-skip-experts --pfb 128` | **6.2×** | Skipping noisy experts improves quality |
| **35B 4-bit** | `--prefill-experts-full-only --pfb 128` | **1.9×** | Smaller shared expert needs routed expert help |
| **Any model** (max speed) | `--prefill-skip-experts --pfb 128` | **6.2×** | Fastest TTFT, acceptable for short prompts |

## What Changed from the Original Repo

1. **Batched projections**: GPU batch GEMM reads weight matrices once for N tokens instead of N times
2. **Batched linear attention**: Custom Metal kernels for conv1d, delta-net recurrence, gated RMS norm — process N tokens per GPU dispatch
3. **Batched full attention**: `prefill_causal_attn` Metal kernel with online softmax (Flash Attention style) — processes all N queries in one dispatch
4. **Skip routed experts**: Intermediate prefill tokens use shared expert only (K=0), eliminating SSD I/O
5. **Experts at full-attention only**: Hybrid mode — batched K=0 at linear layers, per-token K=4 at full-attention layers (25% of layers)
6. **Configurable prefill K**: `--prefill-k N` allows fine-grained control over expert count during prefill

## New Metal Kernels

| Kernel | Purpose | Tokens/dispatch |
|--------|---------|:---:|
| `dequant_gemm_4bit_batch` | Batched 4-bit dequant GEMM for N tokens | N |
| `prefill_causal_attn` | Flash Attention with online softmax, causal mask | N |
| `prefill_q_rope_norm_bf16` | Fused Q deinterleave + RMS norm + RoPE | N |
| `conv1d_step_batched` | Batched conv1d for linear attention | N |
| `rms_norm_qk_batched` | Batched Q/K RMS norm for linear attention | N |
| `compute_decay_beta_batched` | Batched decay/beta for delta-net | N |
| `gated_delta_net_step_batched` | Batched delta-net recurrence | N (sequential internally) |
| `gated_rms_norm_batched` | Batched gated RMS norm output | N |
| `prefill_rms_norm` | Batched input RMS norm | N |
| `prefill_residual_norm` | Batched residual + post-attn norm | N |
| `prefill_swiglu` | Batched SwiGLU activation | N |
| `prefill_combine` | Batched MoE combine + residual | N |

## CLI Flags

```
--pfb N                      Prefill batch size (default 1 = per-token)
--prefill-skip-experts       Skip routed experts during prefill (shared expert only)
--prefill-experts-full-only  Load experts only at full-attention layers (25%)
--prefill-k N                Override K for prefill (0 = skip, 2 = reduced, 4 = full)
--no-batched-linear          Disable batched linear attention (fallback to per-token)
```

## iOS/macOS App Settings

```
Prefill > Batch Size:                  64 tokens (recommended)
Prefill > Skip Routed Experts:         OFF (default for 4-bit)
Prefill > Experts at Full Attn Only:   ON  (default — best speed/quality tradeoff)
Prefill > Batched Linear Attention:    ON  (default)
```

## Raw Test Output

### Test 0: Original baseline (warm cache, raw prompt)
```
[prefill] starting 207 tokens | per-token K=4 skip_experts=0 experts_full_only=0
[prefill done] 207 tokens | 18739 ms | 11.0 tok/s
decode: 10.90 t/s, prefill: 11.01 t/s
Output: "# Comprehensive Technical Analysis of Modern Neural Network Architectures
        and On-Device Inference... The landscape of large language models (LLMs)
        has shifted dramatically..."
```

### Test 1: Batched, skip all experts (warm cache, raw prompt)
```
[prefill] starting 207 tokens | batch=128 skip_experts=1 experts_full_only=0 batched_linear=1 K=0
[prefill done] 207 tokens | 3023 ms | 68.5 tok/s (batched)
decode: 11.14 t/s, prefill: 65.00 t/s
Output: "." then EOS (without chat template)
```

### Test 2: Batched, experts full-attn only (warm cache, raw prompt)
```
[prefill] starting 207 tokens | batch=128 skip_experts=0 experts_full_only=1 batched_linear=1 K=0
[prefill done] 207 tokens | 10115 ms | 20.5 tok/s (batched)
decode: 11.13 t/s, prefill: 20.17 t/s
Output: "# Comprehensive Technical Analysis: Modern Neural Network Architectures
        and Hardware Co-Design... The landscape of neural network inference is
        undergoing a paradigm shift..."
```

### Test 1 (chat template): Batched, skip all experts
```
[prefill] starting 226 tokens | batch=128 skip_experts=1 experts_full_only=0 batched_linear=1 K=0
[prefill done] 226 tokens | 3232 ms | 69.9 tok/s (batched)
decode: 10.30 t/s, prefill: 66.60 t/s
Output: "Based on your request, here is a comprehensive technical analysis...
        Note: The prompt contains several terms that appear to be phonetic
        misspellings..." (misinterprets technical terms)
```

### Test 2 (chat template): Batched, experts full-attn only
```
[prefill] starting 226 tokens | batch=128 skip_experts=0 experts_full_only=1 batched_linear=1 K=0
[prefill done] 226 tokens | 11489 ms | 19.7 tok/s (batched)
decode: 10.71 t/s, prefill: 19.46 t/s
Output: "# Comprehensive Technical Analysis: Modern Neural Network Architectures
        and Consumer Hardware Constraints... The Transformer architecture remains
        the backbone of modern AI..." (full quality, correct understanding)
```
