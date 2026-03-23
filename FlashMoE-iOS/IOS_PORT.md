# Flash-MoE iOS Port: From MacBook to iPhone

How we took a 397B-parameter MoE inference engine written in C/Metal and made it run on an iPhone.

## The Challenge

The desktop Flash-MoE engine was designed for a MacBook Pro with 48GB unified memory and a 40-core GPU. iPhones have ~6GB of usable app memory, a much smaller GPU, and no filesystem access to `shaders.metal` at runtime. Every assumption in the 7,000-line inference engine needed to be re-examined.

## What We Built

A native iOS app (SwiftUI + Objective-C/C) that downloads pre-packed models from HuggingFace and runs them locally with streaming token generation, chat templates, and a full conversational UI.

**Architecture**: Swift (UI + async bridge) → Objective-C wrapper → C inference engine → Metal GPU shaders

## Problems Solved

### 1. Metal Shader Loading

**Problem**: The desktop engine compiles `shaders.metal` from source at runtime via `newLibraryWithSource:`. On iOS, there's no filesystem path to the shader file.

**Fix**: Runtime fallback — try `[device newDefaultLibrary]` first (loads pre-compiled `default.metallib` from the app bundle), fall back to source compilation for macOS CLI:

```objc
ctx->library = [ctx->device newDefaultLibrary];
if (ctx->library) {
    // iOS: loaded from bundle
} else {
    // macOS: compile from source
    NSString *src = [NSString stringWithContentsOfFile:@"shaders.metal" ...];
    ctx->library = [ctx->device newLibraryWithSource:src ...];
}
```

**Xcode fix**: Moved `shaders.metal` from the **Resources** build phase to **Sources** so Xcode's Metal compiler produces `default.metallib` in the app bundle.

### 2. Memory: KV Cache OOM

**Problem**: The model's `max_position_embeddings` is 131,072 (128k context). KV cache allocation per full-attention layer: `131072 * 2 heads * 256 dim * 4 bytes = 256MB`. With 10 full-attention layers: **2.5GB just for KV caches**. `calloc` silently returns NULL on iPhone, causing `EXC_BAD_ACCESS` crashes.

**Fix**: Cap `max_seq_len` at 2048 on iOS:

```objc
if (cfg.max_seq_len > 2048) {
    cfg.max_seq_len = 2048;
}
```

KV cache drops from 2.5GB to ~40MB. Context window is limited but sufficient for chat.

### 3. Memory: Metal Debug Layer

**Problem**: Debug builds wrap every Metal object with `MTLDebugComputeCommandEncoder` validation proxies, roughly doubling GPU memory usage. The app crashes with `NSMallocException` trying to allocate debug wrappers.

**Fix**: Build in **Release** mode and disable Metal API Validation in the Xcode scheme. The debug overhead is too large for iPhone's memory budget.

### 4. Tokenizer Not Found

**Problem**: `init_tokenizer()` searches for `tokenizer.bin` at relative filesystem paths (`./tokenizer.bin`, `./metal_infer/tokenizer.bin`). These don't exist on iOS.

**Fix**: Extended the search to check the model directory (where it's downloaded) and the app bundle:

```objc
// Try model directory (downloaded with model)
snprintf(model_tok, sizeof(model_tok), "%s/tokenizer.bin", cfg.model_path);
if (access(model_tok, R_OK) == 0) { bpe_load(&g_tokenizer, model_tok); }

// Try app bundle
NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"tokenizer" ofType:@"bin"];
if (bundlePath) { bpe_load(&g_tokenizer, [bundlePath UTF8String]); }
```

### 5. Missing Info.plist

**Problem**: Xcode target configs had `INFOPLIST_KEY_*` entries but never set `GENERATE_INFOPLIST_FILE = YES`, so no `Info.plist` was produced.

**Fix**: Added `GENERATE_INFOPLIST_FILE = YES` to both Debug and Release target build settings.

### 6. Chat Template (Garbage Output)

**Problem**: The model received raw text ("Hi") instead of Qwen's chat template format. Without the `<|im_start|>` / `<|im_end|>` markers, the model treats input as a continuation of arbitrary text, producing incoherent output.

**Fix**: Added `buildChatPrompt()` in `ChatView.swift` that formats the full conversation history:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
```

### 7. Special Token Leakage

**Problem**: End-of-turn tokens like `<|im_end|>` appear as visible text in the chat UI.

**Fix**: Strip special tokens from the token stream before displaying:

```swift
let clean = token.text
    .replacingOccurrences(of: "<|im_end|>", with: "")
    .replacingOccurrences(of: "<|im_start|>", with: "")
    .replacingOccurrences(of: "<|endoftext|>", with: "")
```

## iOS App Architecture

### Engine Layer (C/Objective-C)

- **FlashMoEEngine.m** — iOS wrapper around `infer.m` (unity build via `#include`)
- **infer.m** — The full 7,000-line inference engine, shared with macOS
- **shaders.metal** — Metal compute kernels, compiled into `default.metallib`

### Bridge Layer (Swift/ObjC Interop)

- **FlashMoEBridge.swift** — `@Observable` class wrapping the C API
  - `loadModel(at:)` → background thread → `flashmoe_load_model()`
  - `generate(prompt:)` → `AsyncStream<GenerationToken>` via C callback bridge
  - State machine: `idle → loading → ready → generating`

### UI Layer (SwiftUI)

- **ChatView** — Streaming chat with thinking block disclosure, text selection, braille spinner
- **ModelListView** — On-device models + downloadable catalog
- **ModelDownloadRow** — Per-model download progress with pause/resume/delete
- **ProfilerView** — Resource monitoring overlay

### Model Management

- **ModelCatalog.swift** — Static registry of pre-packed HuggingFace repos
- **DownloadManager.swift** — Background `URLSession` downloads with state persistence

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Unity build (`#include "infer.m"`) | Share 100% of inference code with macOS, no fork to maintain |
| Runtime Metal library fallback | Single codepath works on both iOS (pre-compiled) and macOS (source) |
| 2048 context cap | Keeps KV caches under 40MB total, fits iPhone memory budget |
| Pre-packed HuggingFace models | No on-device conversion needed — download and run |
| Background URLSession | Downloads survive app suspension (not force-quit) |
| Trust the OS page cache | Same philosophy as desktop — no custom expert cache on iOS either |

## Model Requirements

The app downloads pre-packed models containing:
- `config.json` — Model architecture config
- `model_weights.bin` — Non-expert weights (~2.5GB for 35B)
- `model_weights.json` — Tensor manifest
- `tokenizer.bin` — Pre-exported BPE tokenizer
- `packed_experts/layer_XX.bin` — One file per layer (~450MB each, 40 layers for 35B)

Total download for Qwen 3.5 35B-A3B (4-bit): **~19.5GB**

## What's Next

- **KV cache reuse across turns** — Currently re-tokenizes full history each message
- **Adaptive context length** — Scale based on available memory instead of fixed 2048
- **Download resumption** — Handle interrupted downloads more gracefully
- **Smaller models** — 7B/14B variants for devices with less storage
- **Background inference** — Continue generation when app is backgrounded
