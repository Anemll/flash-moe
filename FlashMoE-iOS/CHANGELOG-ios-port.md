# FlashMoE-iOS: Changes for 397B Model Support

Changes made to support Qwen3.5-397B-A17B (2-bit) in the iOS/macOS app.
These must also be applied to the standalone iOS repo.

## Critical Fix: Per-layer quant arrays not initialized in `flashmoe_load()`

**Bug:** `g_layer_is_2bit[]` was never set in the iOS load path. The CLI's `main()` sets
`g_layer_is_2bit[i] = 1` when opening 2-bit expert files (line 9461), but `flashmoe_load()`
only opened the files without setting the per-layer flags.

**Effect:** `fused_layer_forward()` line 6396 does `if (saved_use_2bit) g_use_2bit = g_layer_is_2bit[layer_idx]`.
With `g_layer_is_2bit[i] = 0`, `g_use_2bit` was reset to 0 inside every layer, causing:
- `active_expert_size()` returned 4-bit size (~6.75 MB) instead of 2-bit (~3.93 MB)
- pread tried to read 6.75 MB from 3.93 MB expert slots → corrupt data / short reads
- GPU expert forward produced NaN → all subsequent layers NaN → garbled output (`!!!!!!`)

**Fix:** In `flashmoe_load()`, set `g_layer_is_2bit[i] = 1` for each successfully opened
2-bit expert file (mirrors CLI behavior). Also initialize `g_layer_is_q3_hybrid[]` and
`g_layer_is_q3_outlier[]` to zero for clean state.

**Symptom:** 397B 2-bit produced `!` characters (token_id=0). Hidden state was NaN from
layer 0's first `complete_deferred_experts()`. 35B models (K=8, 4-bit) were unaffected.

## Critical Fix: MetalCtx ARC cleanup in `flashmoe_unload()`

**Bug:** `free(g_metal)` leaked all ARC-managed `id<MTLBuffer>` and `id<MTLComputePipelineState>`
objects in the `MetalCtx` struct. When switching models (e.g., 35B → 397B), the leaked Metal
objects corrupted the heap, causing `objc Method cache corrupted` crashes.

**Fix:** Explicitly nil every `id<>` member before `free()` so ARC decrements refcounts.
This includes ~80 Metal buffers, pipeline states, events, KV caches, and delta-net GPU buffers.

**Symptom:** App works if 397B loaded first. Crashes with heap corruption when switching
from a smaller model to 397B (or any model switch that changes buffer sizes).

## Changes to `metal_infer/infer.m` (upstream)

### 1. NAX shader bundle fallback (line ~2707)
Added `#ifdef CHAT_MODE` block to search for `nax_gemm.metal` in the app bundle
when file-path search fails (sandboxed apps can't access filesystem paths).

```c
#ifdef CHAT_MODE
// iOS/macOS app: also try app bundle resource
if (!nax_src) {
    NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"nax_gemm" ofType:@"metal"];
    if (bundlePath) {
        nax_src = [NSString stringWithContentsOfFile:bundlePath encoding:NSUTF8StringEncoding error:&error];
    }
}
#endif
```

## Changes to Xcode Project (`FlashMoE.xcodeproj/project.pbxproj`)

### 2. Added `nax_gemm.metal` as a bundle resource
- File reference: `FF1111111111111111111111` → `nax_gemm.metal` in `metal_infer` group
- Build file: `FF3333333333333333333333` → `nax_gemm.metal in Resources`
- Added to Resources build phase (NOT Sources — needs Metal 4.0 runtime compilation)
- File type set to `text` (not `sourcecode.metal`) to prevent Xcode from compiling it
- Path resolves to `../metal_infer/nax_gemm.metal`

### 3. Model data setup (macOS)
Created model directory at:
```
~/Library/Containers/com.flashmoe.ios/Data/Documents/qwen3.5-397b-a17b-2bit/
```
Using hard links (zero extra disk space) to `~/Models/flash_mlx_4bit/`:
- `config.json` — copied from MLX source model
- `model_weights.bin` — hard linked (5.1 GB)
- `model_weights.json` — copied
- `vocab.bin`, `tokenizer.bin` — copied
- `packed_experts_2bit/` — 60 layer files hard linked (113 GB)

## Known Issues

### Notification spam
`Detected potentially harmful notification post rate of 27292 notifications per second`
- Cause: `@Observable` fires SwiftUI change notifications on every token callback
- Fix needed: throttle `DispatchQueue.main.async` updates (e.g., every 100ms or every 5 tokens)

### MAX_K=8 vs model K=10
- `MAX_K` is hardcoded to 8 in `infer.m` (GPU buffer slots)
- 397B model uses K=10 experts per token
- Engine clamps to 8 — drops 2 experts per layer
- Impact: slight quality degradation, not fatal
- Fix: increase MAX_K to 10 (requires Metal buffer reallocation)

## TODO for standalone iOS repo

- [ ] Apply per-layer quant array fix in `flashmoe_load()` (critical for 2-bit/Q3 models)
- [ ] Apply MetalCtx ARC cleanup in `flashmoe_unload()` (critical for model switching)
- [ ] Apply infer.m NAX bundle fallback patch
- [ ] Add nax_gemm.metal to Xcode project as resource
- [ ] Add 397B model entry to ModelCatalog.swift (if distributing)
- [ ] Fix notification throttling in FlashMoEBridge.swift
- [ ] Consider increasing MAX_K from 8 to 10 for 397B compatibility
- [ ] Strip debug prints from FlashMoEEngine.m (currently behind `[iOS-DBG]` prefix)
