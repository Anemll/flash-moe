/*
 * batched_prefill.h — Batched prefill for Flash-MoE
 *
 * Included from infer.m (unity build) — has access to all static globals.
 */

// ============================================================================
// Token-first prefill baseline
// ============================================================================
static int prefill_k0_token_first(
    WeightFile *wf, float *hidden, float *embed_batch,
    int num_prefill, int pos_start, int prefill_K,
    KVCache **kv_caches, void **layer_states, void **layer_mmaps, int *layer_fds
) {
    if (prefill_K > MAX_K) prefill_K = MAX_K;
    for (int token_idx = 0; token_idx < num_prefill; token_idx++) {
        @autoreleasepool {
        cache_telemetry_note_token();
        memcpy(hidden, embed_batch + (size_t)token_idx * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            fused_layer_forward(wf, layer, hidden,
                                is_full ? kv_caches[layer] : NULL,
                                is_full ? NULL : layer_states[layer],
                                pos_start + token_idx,
                                layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                prefill_K, layer_fds[layer]);
        }
        discard_deferred_experts();
        } // @autoreleasepool
    }
    return num_prefill;
}

// ============================================================================
// Diagnostic: run one full-attn layer via fused_layer_forward and capture
// the hidden state, then compare with our batched reimplementation.
// This helps pinpoint which step diverges.
// ============================================================================
static void debug_compare_full_attn_layer(
    WeightFile *wf, int layer, float *hidden_in,
    int pos, KVCache **kv_caches, void **layer_mmaps, int *layer_fds
) {
    KVCache *kv = kv_caches[layer];

    // Now run our batched path on another copy
    float *h_batch = (float *)malloc(HIDDEN_DIM * sizeof(float));
    memcpy(h_batch, hidden_in, HIDDEN_DIM * sizeof(float));

    LayerWeightCache *lc = &layer_cache[layer];
    int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;
    int q_head_dim = NUM_ATTN_HEADS * HEAD_DIM;
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // Step 1: norm
    float *normed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    cpu_rms_norm(h_batch, lc->input_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);

    // Step 2: Q/K/V projections via per-token GEMV (NOT batch GEMM — to isolate GEMM issues)
    float *q_out = (float *)calloc(q_proj_dim, sizeof(float));
    float *k_out = (float *)calloc(kv_dim, sizeof(float));
    float *v_out = (float *)calloc(kv_dim, sizeof(float));
    fast_dequant_matvec(lc->q_w, lc->q_s, lc->q_b, normed, q_out, q_proj_dim, HIDDEN_DIM, GROUP_SIZE);
    fast_dequant_matvec(lc->k_w, lc->k_s, lc->k_b, normed, k_out, kv_dim, HIDDEN_DIM, GROUP_SIZE);
    fast_dequant_matvec(lc->v_w, lc->v_s, lc->v_b, normed, v_out, kv_dim, HIDDEN_DIM, GROUP_SIZE);

    // Debug: Q/K/V projection outputs
    {
        float rms_q = 0; for (int i = 0; i < q_proj_dim; i++) rms_q += q_out[i]*q_out[i];
        float rms_k = 0; for (int i = 0; i < kv_dim; i++) rms_k += k_out[i]*k_out[i];
        fprintf(stderr, "  [cmp] L%d Q rms=%.6f first=[%.6f,%.6f] K rms=%.6f first=[%.6f,%.6f]\n",
                layer, sqrtf(rms_q/q_proj_dim), q_out[0], q_out[1],
                sqrtf(rms_k/kv_dim), k_out[0], k_out[1]);
    }

    // Deinterleave Q and gate: q_proj_out is [Q0|gate0|Q1|gate1|...|Q31|gate31]
    // Each head has 2*HEAD_DIM values: first HEAD_DIM = Q, second HEAD_DIM = gate
    float *q_deint = (float *)malloc(q_head_dim * sizeof(float));
    float *q_gate  = (float *)malloc(q_head_dim * sizeof(float));
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        float *src = q_out + h * (2 * HEAD_DIM);
        memcpy(q_deint + h * HEAD_DIM, src, HEAD_DIM * sizeof(float));
        memcpy(q_gate + h * HEAD_DIM, src + HEAD_DIM, HEAD_DIM * sizeof(float));
    }
    // Use deinterleaved Q for the rest
    float *q_raw = q_deint;
    if (lc->q_norm_w) {
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            float *qh = q_out + h * HEAD_DIM;
            float sum_sq = 0;
            for (int d = 0; d < HEAD_DIM; d++) sum_sq += qh[d] * qh[d];
            float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
            for (int d = 0; d < HEAD_DIM; d++)
                qh[d] = qh[d] * inv_rms * bf16_to_f32(lc->q_norm_w[d]);
        }
    }
    if (lc->k_norm_w) {
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            float *kh = k_out + h * HEAD_DIM;
            float sum_sq = 0;
            for (int d = 0; d < HEAD_DIM; d++) sum_sq += kh[d] * kh[d];
            float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
            for (int d = 0; d < HEAD_DIM; d++)
                kh[d] = kh[d] * inv_rms * bf16_to_f32(lc->k_norm_w[d]);
        }
    }

    // Step 4: RoPE
    apply_rotary_emb(q_out, k_out, pos, NUM_ATTN_HEADS, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM);

    // Step 5: KV cache update
    memcpy(kv->k_cache + (size_t)kv->len * kv_dim, k_out, kv_dim * sizeof(float));
    memcpy(kv->v_cache + (size_t)kv->len * kv_dim, v_out, kv_dim * sizeof(float));
    kv->len++;

    // Debug: check V values stored in cache
    {
        float rms_v = 0;
        float *v_cached = kv->v_cache + (size_t)(kv->len - 1) * kv_dim;
        for (int i = 0; i < kv_dim; i++) rms_v += v_cached[i]*v_cached[i];
        fprintf(stderr, "  [cmp] L%d V_cached rms=%.6f first=[%.6f,%.6f] kv_len=%d\n",
                layer, sqrtf(rms_v/kv_dim), v_cached[0], v_cached[1], kv->len);

        // Also check what the fused run left in the cache (it ran first and wrote at same position)
        // Actually the fused run added at kv_len_before, I restored to kv_len_before, then I added again
        // The fused values at position kv_len_before were OVERWRITTEN by my values
        // Let me check if the fused run's CMD2 might have stored something in batch_out that tells us
    }

    // Step 6: CPU attention
    float *attn_out = (float *)calloc(q_head_dim, sizeof(float));
    int groups_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float *scores = (float *)malloc(kv->len * sizeof(float));

    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        int kv_h = h / groups_per_kv;
        float *q_h = q_out + h * HEAD_DIM;
        for (int p = 0; p < kv->len; p++) {
            float dot = 0;
            float *k_p = kv->k_cache + (size_t)p * kv_dim + kv_h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) dot += q_h[d] * k_p[d];
            scores[p] = dot * scale;
        }
        cpu_softmax(scores, kv->len);
        float *o_h = attn_out + h * HEAD_DIM;
        for (int p = 0; p < kv->len; p++) {
            float *v_p = kv->v_cache + (size_t)p * kv_dim + kv_h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) o_h[d] += scores[p] * v_p[d];
        }
    }
    free(scores);

    // Step 7: sigmoid gate
    for (int i = 0; i < q_head_dim; i++) {
        float g = 1.0f / (1.0f + expf(-q_gate[i]));
        attn_out[i] *= g;
    }

    // Debug: attn_out
    {
        float rms_a = 0; for (int i = 0; i < q_head_dim; i++) rms_a += attn_out[i]*attn_out[i];
        fprintf(stderr, "  [cmp] L%d attn_out: rms=%.6f first=[%.6f,%.6f]\n",
                layer, sqrtf(rms_a/q_head_dim), attn_out[0], attn_out[1]);
    }

    // Step 8: o_proj
    fprintf(stderr, "  [cmp] L%d o_proj weights: w=%p s=%p b=%p kind=%d source=%d in=%d out=%d\n",
            layer, (void*)lc->o_w, (void*)lc->o_s, (void*)lc->o_b,
            lc->o_kind, lc->o_source, q_head_dim, HIDDEN_DIM);
    float *o_proj_out = (float *)calloc(HIDDEN_DIM, sizeof(float));
    fast_kind_matvec(lc->o_w, lc->o_s, lc->o_b, lc->o_kind, lc->o_source,
                     attn_out, o_proj_out, HIDDEN_DIM, q_head_dim, GROUP_SIZE);

    // Debug: o_proj_out
    {
        float rms_o = 0; for (int i = 0; i < HIDDEN_DIM; i++) rms_o += o_proj_out[i]*o_proj_out[i];
        fprintf(stderr, "  [cmp] L%d o_proj: rms=%.6f first=[%.6f,%.6f]\n",
                layer, sqrtf(rms_o/HIDDEN_DIM), o_proj_out[0], o_proj_out[1]);
    }

    // Step 9: residual + post-attn norm
    float *h_mid = (float *)malloc(HIDDEN_DIM * sizeof(float));
    for (int i = 0; i < HIDDEN_DIM; i++)
        h_mid[i] = h_batch[i] + o_proj_out[i];

    float *h_post = (float *)malloc(HIDDEN_DIM * sizeof(float));
    cpu_rms_norm(h_mid, lc->post_attn_norm_w, h_post, HIDDEN_DIM, RMS_NORM_EPS);

    // Debug: h_mid
    {
        float rms_m = 0; for (int i = 0; i < HIDDEN_DIM; i++) rms_m += h_mid[i]*h_mid[i];
        fprintf(stderr, "  [cmp] L%d h_mid: rms=%.6f first=[%.6f,%.6f]\n",
                layer, sqrtf(rms_m/HIDDEN_DIM), h_mid[0], h_mid[1]);
    }

    // Step 10: shared expert
    float *sg_out = (float *)calloc(SHARED_INTERMEDIATE, sizeof(float));
    float *su_out = (float *)calloc(SHARED_INTERMEDIATE, sizeof(float));
    float gate_score_val = 0;

    if (lc->sg_w && lc->sg_s && lc->sg_b)
        fast_dequant_matvec(lc->sg_w, lc->sg_s, lc->sg_b, h_post, sg_out,
                            SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
    if (lc->su_w && lc->su_s && lc->su_b)
        fast_dequant_matvec(lc->su_w, lc->su_s, lc->su_b, h_post, su_out,
                            SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
    if (lc->seg_w && lc->seg_s && lc->seg_b)
        fast_dequant_matvec(lc->seg_w, lc->seg_s, lc->seg_b, h_post, &gate_score_val,
                            1, HIDDEN_DIM, GROUP_SIZE);

    // SwiGLU
    for (int i = 0; i < SHARED_INTERMEDIATE; i++) {
        float silu_g = sg_out[i] / (1.0f + expf(-sg_out[i]));
        sg_out[i] = silu_g * su_out[i];
    }

    // down_proj
    float *shared_out = (float *)calloc(HIDDEN_DIM, sizeof(float));
    if (lc->sd_w && lc->sd_s && lc->sd_b)
        fast_dequant_matvec(lc->sd_w, lc->sd_s, lc->sd_b, sg_out, shared_out,
                            HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE);

    // Debug: shared expert
    {
        float rms_so = 0; for (int i = 0; i < HIDDEN_DIM; i++) rms_so += shared_out[i]*shared_out[i];
        fprintf(stderr, "  [cmp] L%d shared_out rms=%.6f gate_score=%.4f sigmoid=%.4f first=[%.6f,%.6f]\n",
                layer, sqrtf(rms_so/HIDDEN_DIM), gate_score_val,
                1.0f/(1.0f+expf(-gate_score_val)), shared_out[0], shared_out[1]);
    }

    // Combine
    float gs = 1.0f / (1.0f + expf(-gate_score_val));
    for (int i = 0; i < HIDDEN_DIM; i++)
        h_batch[i] = h_mid[i] + gs * shared_out[i];

    // Print batched result
    float rms_b = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) rms_b += h_batch[i] * h_batch[i];
    fprintf(stderr, "  [cmp] L%d BATCH: rms=%.6f first=[%.6f,%.6f,%.6f,%.6f]\n",
            layer, sqrtf(rms_b / HIDDEN_DIM),
            h_batch[0], h_batch[1], h_batch[2], h_batch[3]);

    {
        float rms_final = 0;
        for (int i = 0; i < HIDDEN_DIM; i++) rms_final += h_batch[i] * h_batch[i];
        fprintf(stderr, "  [cmp] L%d REIMPL: rms=%.6f first=[%.6f,%.6f]\n",
                layer, sqrtf(rms_final / HIDDEN_DIM), h_batch[0], h_batch[1]);
    }

    // Restore KV cache for the actual run
    kv->len -= 1;  // undo the increment from our reimplementation

    free(h_batch); free(normed); free(q_deint); free(q_gate);
    free(q_out); free(k_out); free(v_out);
    free(attn_out); free(o_proj_out);
    free(h_mid); free(h_post);
    free(sg_out); free(su_out); free(shared_out);
}

// ============================================================================
// Layer-first prefill
// ============================================================================
static int prefill_k0_layer_first(
    WeightFile *wf, float *hidden, float *embed_batch,
    int num_prefill, int pos_start, int pfb, int prefill_K, int K_decode,
    KVCache **kv_caches, void **layer_states, void **layer_mmaps, int *layer_fds
) {
    if (!layer_cache_built) build_layer_cache(wf);

    int gpu_pfb = (pfb > MAX_PFB_GPU) ? MAX_PFB_GPU : pfb;
    id<MTLBuffer> hidden_cur_buf = nil;
    id<MTLBuffer> hidden_next_buf = nil;
    int use_gpu_hidden_buffers = 0;

    if (g_metal && g_metal->device && num_prefill > 0) {
        size_t hidden_bytes = (size_t)num_prefill * HIDDEN_DIM * sizeof(float);
        hidden_cur_buf = [g_metal->device newBufferWithLength:hidden_bytes
                                                      options:MTLResourceStorageModeShared];
        hidden_next_buf = [g_metal->device newBufferWithLength:hidden_bytes
                                                       options:MTLResourceStorageModeShared];
        if (hidden_cur_buf && hidden_next_buf) {
            memcpy([hidden_cur_buf contents], embed_batch, hidden_bytes);
            use_gpu_hidden_buffers = 1;
        } else if (!g_stream_mode) {
            fprintf(stderr, "[prefill] WARNING: GPU hidden buffer alloc failed, falling back to CPU handoff\n");
        }
    }

    int layers_batched = 0, layers_per_token = 0, layers_moe_tail = 0;
    double t_layer_total = 0;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        @autoreleasepool {
            double t_layer = now_ms();
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            LayerWeightCache *lc = &layer_cache[layer];
            float *hidden_in_cpu = use_gpu_hidden_buffers ? (float *)[hidden_cur_buf contents] : embed_batch;
            float *hidden_out_cpu = use_gpu_hidden_buffers ? (float *)[hidden_next_buf contents] : embed_batch;
            int used_batched_layer = 0;
            int use_routed_moe_tail;
            if (g_prefill_experts_full_only) {
                // Only load routed experts at full attention layers
                use_routed_moe_tail = (is_full && K_decode > 0 && layer < NUM_LAYERS - 1);
            } else {
                use_routed_moe_tail = (prefill_K > 0 && layer < NUM_LAYERS - 1);
            }

            if (is_full &&
                use_gpu_hidden_buffers &&
                g_metal && g_metal->gemm_batch && g_metal->buf_pfb_input &&
                g_metal->prefill_rms_norm && g_metal->prefill_residual_norm &&
                g_metal->prefill_swiglu && g_metal->prefill_combine &&
                g_metal->prefill_q_rope_norm && g_metal->prefill_kv_cache) {
                int full_wf_only =
                    lc->q_source == MATVEC_SOURCE_WF &&
                    lc->k_source == MATVEC_SOURCE_WF &&
                    lc->v_source == MATVEC_SOURCE_WF &&
                    lc->o_source == MATVEC_SOURCE_WF &&
                    lc->sg_source == MATVEC_SOURCE_WF &&
                    lc->su_source == MATVEC_SOURCE_WF &&
                    lc->sd_source == MATVEC_SOURCE_WF;
                int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;
                int q_head_dim = NUM_ATTN_HEADS * HEAD_DIM;
                int kv_dim = NUM_KV_HEADS * HEAD_DIM;
                int fa_idx = (layer + 1) / FULL_ATTN_INTERVAL - 1;
                KVCache *kv = kv_caches[layer];

                if (full_wf_only &&
                    fa_idx >= 0 && fa_idx < g_cfg.num_full_attn_layers &&
                    kv && kv->k_cache && kv->v_cache &&
                    kv->len + num_prefill <= g_gpu_kv_seq &&
                    lc->input_norm_w && lc->post_attn_norm_w &&
                    lc->q_w && lc->q_s && lc->q_b &&
                    lc->k_w && lc->k_s && lc->k_b &&
                    lc->v_w && lc->v_s && lc->v_b &&
                    lc->o_w && lc->o_s && lc->o_b &&
                    lc->sg_w && lc->sg_s && lc->sg_b &&
                    lc->su_w && lc->su_s && lc->su_b &&
                    lc->sd_w && lc->sd_s && lc->sd_b &&
                    lc->seg_w && lc->seg_s && lc->seg_b) {
                    used_batched_layer = 1;

                    int groups_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
                    int layer_cache_start = kv->len;
                    float scale = 1.0f / sqrtf((float)HEAD_DIM);
                    float eps = RMS_NORM_EPS;
                    float rope_theta = (float)ROPE_THETA;
                    NSUInteger input_norm_off = (NSUInteger)((const char *)lc->input_norm_w -
                                                             (const char *)[g_metal->wf_buf contents]);
                    NSUInteger post_norm_off = (NSUInteger)((const char *)lc->post_attn_norm_w -
                                                            (const char *)[g_metal->wf_buf contents]);
                    NSUInteger q_norm_off = lc->q_norm_w ?
                        (NSUInteger)((const char *)lc->q_norm_w - (const char *)[g_metal->wf_buf contents]) : 0;
                    NSUInteger k_norm_off = lc->k_norm_w ?
                        (NSUInteger)((const char *)lc->k_norm_w - (const char *)[g_metal->wf_buf contents]) : 0;
                    uint32_t dim = HIDDEN_DIM;
                    uint32_t hd = HEAD_DIM;
                    uint32_t kvd = (uint32_t)kv_dim;
                    uint32_t nh = NUM_ATTN_HEADS;
                    uint32_t rotary_dim = ROTARY_DIM;
                    uint32_t hpkv = (uint32_t)groups_per_kv;

                    id<MTLCommandBuffer> cmd = [g_metal->queue commandBuffer];
                    for (int chunk_start = 0; chunk_start < num_prefill; chunk_start += gpu_pfb) {
                        int N = (chunk_start + gpu_pfb <= num_prefill) ? gpu_pfb : (num_prefill - chunk_start);
                        uint32_t batch_n = (uint32_t)N;
                        NSUInteger hidden_off = (NSUInteger)((size_t)chunk_start * HIDDEN_DIM * sizeof(float));
                        uint32_t pos_base = (uint32_t)(pos_start + chunk_start);
                        uint32_t cache_start = (uint32_t)kv->len;
                        uint32_t has_q_norm = lc->q_norm_w ? 1u : 0u;
                        uint32_t has_k_norm = lc->k_norm_w ? 1u : 0u;
                        kv->len += N;

                        {
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->prefill_rms_norm];
                            [enc setBuffer:hidden_cur_buf       offset:hidden_off atIndex:0];
                            [enc setBuffer:g_metal->wf_buf      offset:input_norm_off atIndex:1];
                            [enc setBuffer:g_metal->buf_pfb_input offset:0 atIndex:2];
                            [enc setBytes:&dim length:4 atIndex:3];
                            [enc setBytes:&eps length:4 atIndex:4];
                            [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc endEncoding];
                        }

                        gpu_encode_pfb_gemm(g_metal, cmd, lc->q_w, lc->q_s, lc->q_b,
                            q_proj_dim, HIDDEN_DIM, GROUP_SIZE, N, 0);
                        gpu_encode_pfb_gemm(g_metal, cmd, lc->k_w, lc->k_s, lc->k_b,
                            kv_dim, HIDDEN_DIM, GROUP_SIZE, N, 1);
                        gpu_encode_pfb_gemm(g_metal, cmd, lc->v_w, lc->v_s, lc->v_b,
                            kv_dim, HIDDEN_DIM, GROUP_SIZE, N, 2);

                        {
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->prefill_q_rope_norm];
                            [enc setBuffer:g_metal->buf_pfb_out[0] offset:0 atIndex:0];
                            [enc setBuffer:g_metal->wf_buf         offset:q_norm_off atIndex:1];
                            [enc setBuffer:g_metal->buf_pfb_out[4] offset:0 atIndex:2];
                            [enc setBuffer:g_metal->buf_pfb_out[5] offset:0 atIndex:3];
                            [enc setBytes:&hd          length:4 atIndex:4];
                            [enc setBytes:&nh          length:4 atIndex:5];
                            [enc setBytes:&rotary_dim  length:4 atIndex:6];
                            [enc setBytes:&pos_base    length:4 atIndex:7];
                            [enc setBytes:&batch_n     length:4 atIndex:8];
                            [enc setBytes:&eps         length:4 atIndex:9];
                            [enc setBytes:&rope_theta  length:4 atIndex:10];
                            [enc setBytes:&has_q_norm  length:4 atIndex:11];
                            [enc dispatchThreadgroups:MTLSizeMake(NUM_ATTN_HEADS, N, 1)
                                threadsPerThreadgroup:MTLSizeMake(HEAD_DIM, 1, 1)];
                            [enc endEncoding];
                        }

                        {
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->prefill_kv_cache];
                            [enc setBuffer:g_metal->buf_pfb_out[1] offset:0 atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[2] offset:0 atIndex:1];
                            [enc setBuffer:g_metal->wf_buf         offset:k_norm_off atIndex:2];
                            [enc setBuffer:g_metal->buf_kv_k[fa_idx] offset:0 atIndex:3];
                            [enc setBuffer:g_metal->buf_kv_v[fa_idx] offset:0 atIndex:4];
                            [enc setBytes:&hd          length:4 atIndex:5];
                            [enc setBytes:&kvd         length:4 atIndex:6];
                            [enc setBytes:&rotary_dim  length:4 atIndex:7];
                            [enc setBytes:&pos_base    length:4 atIndex:8];
                            [enc setBytes:&cache_start length:4 atIndex:9];
                            [enc setBytes:&batch_n     length:4 atIndex:10];
                            [enc setBytes:&eps         length:4 atIndex:11];
                            [enc setBytes:&rope_theta  length:4 atIndex:12];
                            [enc setBytes:&has_k_norm  length:4 atIndex:13];
                            [enc dispatchThreadgroups:MTLSizeMake(NUM_KV_HEADS, N, 1)
                                threadsPerThreadgroup:MTLSizeMake(HEAD_DIM, 1, 1)];
                            [enc endEncoding];
                        }

                        {
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->prefill_causal_attn];
                            [enc setBuffer:g_metal->buf_pfb_out[4] offset:0 atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[5] offset:0 atIndex:1];
                            [enc setBuffer:g_metal->buf_kv_k[fa_idx] offset:0 atIndex:2];
                            [enc setBuffer:g_metal->buf_kv_v[fa_idx] offset:0 atIndex:3];
                            [enc setBuffer:g_metal->buf_pfb_out[6] offset:0 atIndex:4];
                            [enc setBytes:&hd     length:4 atIndex:5];
                            [enc setBytes:&kvd    length:4 atIndex:6];
                            [enc setBytes:&nh     length:4 atIndex:7];
                            [enc setBytes:&hpkv   length:4 atIndex:8];
                            [enc setBytes:&scale  length:4 atIndex:9];
                            [enc setBytes:&cache_start length:4 atIndex:10];
                            [enc setBytes:&batch_n length:4 atIndex:11];
                            [enc dispatchThreadgroups:MTLSizeMake(N * NUM_ATTN_HEADS, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc endEncoding];
                        }

                        gpu_encode_pfb_gemm_ex(g_metal, cmd, lc->o_w, lc->o_s, lc->o_b,
                            HIDDEN_DIM, q_head_dim, GROUP_SIZE, N,
                            g_metal->buf_pfb_out[6], 0, g_metal->buf_pfb_out[5], 0);

                        {
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->prefill_residual_norm];
                            [enc setBuffer:hidden_cur_buf       offset:hidden_off atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[5] offset:0 atIndex:1];
                            [enc setBuffer:(use_routed_moe_tail ? hidden_next_buf : g_metal->buf_pfb_out[6])
                                   offset:(use_routed_moe_tail ? hidden_off : 0) atIndex:2];
                            [enc setBuffer:g_metal->buf_pfb_input offset:0 atIndex:3];
                            [enc setBuffer:g_metal->wf_buf        offset:post_norm_off atIndex:4];
                            [enc setBytes:&dim length:4 atIndex:5];
                            [enc setBytes:&eps length:4 atIndex:6];
                            [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc endEncoding];
                        }

                        if (!use_routed_moe_tail) {
                            gpu_encode_pfb_gemm(g_metal, cmd, lc->sg_w, lc->sg_s, lc->sg_b,
                                SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, N, 0);
                            gpu_encode_pfb_gemm(g_metal, cmd, lc->su_w, lc->su_s, lc->su_b,
                                SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, N, 1);
                            gpu_encode_pfb_gemm(g_metal, cmd, lc->seg_w, lc->seg_s, lc->seg_b,
                                1, HIDDEN_DIM, GROUP_SIZE, N, 2);

                            {
                                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                                [enc setComputePipelineState:g_metal->prefill_swiglu];
                                [enc setBuffer:g_metal->buf_pfb_out[0] offset:0 atIndex:0];
                                [enc setBuffer:g_metal->buf_pfb_out[1] offset:0 atIndex:1];
                                uint32_t total = (uint32_t)((size_t)N * SHARED_INTERMEDIATE);
                                [enc setBytes:&total length:4 atIndex:2];
                                uint32_t tgs = (total + 255) / 256;
                                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                                [enc endEncoding];
                            }

                            gpu_encode_pfb_gemm_ex(g_metal, cmd, lc->sd_w, lc->sd_s, lc->sd_b,
                                HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, N,
                                g_metal->buf_pfb_out[0], 0, g_metal->buf_pfb_out[3], 0);

                            {
                                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                                [enc setComputePipelineState:g_metal->prefill_combine];
                                [enc setBuffer:g_metal->buf_pfb_out[6] offset:0 atIndex:0];
                                [enc setBuffer:g_metal->buf_pfb_out[3] offset:0 atIndex:1];
                                [enc setBuffer:g_metal->buf_pfb_out[2] offset:0 atIndex:2];
                                [enc setBuffer:hidden_next_buf         offset:hidden_off atIndex:3];
                                [enc setBytes:&dim length:4 atIndex:4];
                                [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
                                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                                [enc endEncoding];
                            }
                        }
                    }

                    [cmd commit];
                    [cmd waitUntilCompleted];

                    size_t copied_tokens = (size_t)(kv->len - layer_cache_start);
                    if (copied_tokens > 0) {
                        memcpy(kv->k_cache + (size_t)layer_cache_start * kv_dim,
                               (float *)[g_metal->buf_kv_k[fa_idx] contents] + (size_t)layer_cache_start * kv_dim,
                               copied_tokens * (size_t)kv_dim * sizeof(float));
                        memcpy(kv->v_cache + (size_t)layer_cache_start * kv_dim,
                               (float *)[g_metal->buf_kv_v[fa_idx] contents] + (size_t)layer_cache_start * kv_dim,
                               copied_tokens * (size_t)kv_dim * sizeof(float));
                    }
                }
            }

            if (!used_batched_layer &&
                !is_full &&
                !g_disable_batched_linear &&
                use_gpu_hidden_buffers &&
                g_metal && g_metal->gemm_batch && g_metal->buf_pfb_input &&
                g_metal->prefill_rms_norm && g_metal->prefill_residual_norm &&
                g_metal->prefill_swiglu && g_metal->prefill_combine &&
                g_metal->conv1d_step_batched && g_metal->rms_norm_qk_batched &&
                g_metal->compute_decay_beta_batched && g_metal->delta_net_step_batched &&
                g_metal->gated_rms_norm_batched) {
                int linear_layer_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                int linear_wf_only =
                    lc->qkv_source == MATVEC_SOURCE_WF &&
                    lc->z_source == MATVEC_SOURCE_WF &&
                    lc->out_proj_source == MATVEC_SOURCE_WF &&
                    lc->sg_source == MATVEC_SOURCE_WF &&
                    lc->su_source == MATVEC_SOURCE_WF &&
                    lc->sd_source == MATVEC_SOURCE_WF;

                if (linear_layer_idx >= 0 && linear_layer_idx < g_cfg.num_linear_layers &&
                    linear_wf_only &&
                    lc->input_norm_w && lc->post_attn_norm_w &&
                    lc->qkv_w && lc->qkv_s && lc->qkv_b &&
                    lc->z_w && lc->z_s && lc->z_b &&
                    lc->b_w && lc->b_s && lc->b_b &&
                    lc->a_w && lc->a_s && lc->a_b &&
                    lc->conv1d_w && lc->A_log && lc->dt_bias &&
                    lc->gated_norm_w &&
                    lc->out_proj_w && lc->out_proj_s && lc->out_proj_b &&
                    lc->sg_w && lc->sg_s && lc->sg_b &&
                    lc->su_w && lc->su_s && lc->su_b &&
                    lc->sd_w && lc->sd_s && lc->sd_b &&
                    lc->seg_w && lc->seg_s && lc->seg_b) {
                    used_batched_layer = 1;

                    float eps = RMS_NORM_EPS;
                    uint32_t dim = HIDDEN_DIM;
                    NSUInteger input_norm_off = (NSUInteger)((const char *)lc->input_norm_w -
                                                             (const char *)[g_metal->wf_buf contents]);
                    NSUInteger post_norm_off = (NSUInteger)((const char *)lc->post_attn_norm_w -
                                                            (const char *)[g_metal->wf_buf contents]);
                    NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w -
                                                         (const char *)[g_metal->wf_buf contents]);
                    NSUInteger a_log_off = (NSUInteger)((const char *)lc->A_log -
                                                        (const char *)[g_metal->wf_buf contents]);
                    NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias -
                                                          (const char *)[g_metal->wf_buf contents]);
                    NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w -
                                                          (const char *)[g_metal->wf_buf contents]);

                    id<MTLCommandBuffer> cmd = [g_metal->queue commandBuffer];
                    for (int chunk_start = 0; chunk_start < num_prefill; chunk_start += gpu_pfb) {
                        int N = (chunk_start + gpu_pfb <= num_prefill) ? gpu_pfb : (num_prefill - chunk_start);
                        uint32_t batch_n = (uint32_t)N;
                        NSUInteger hidden_off = (NSUInteger)((size_t)chunk_start * HIDDEN_DIM * sizeof(float));

                        {
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->prefill_rms_norm];
                            [enc setBuffer:hidden_cur_buf       offset:hidden_off atIndex:0];
                            [enc setBuffer:g_metal->wf_buf      offset:input_norm_off atIndex:1];
                            [enc setBuffer:g_metal->buf_pfb_input offset:0 atIndex:2];
                            [enc setBytes:&dim length:4 atIndex:3];
                            [enc setBytes:&eps length:4 atIndex:4];
                            [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc endEncoding];
                        }

                        gpu_encode_pfb_gemm(g_metal, cmd, lc->qkv_w, lc->qkv_s, lc->qkv_b,
                            LINEAR_CONV_DIM, HIDDEN_DIM, GROUP_SIZE, N, 0);
                        gpu_encode_pfb_gemm(g_metal, cmd, lc->z_w, lc->z_s, lc->z_b,
                            LINEAR_TOTAL_VALUE, HIDDEN_DIM, GROUP_SIZE, N, 1);
                        gpu_encode_pfb_gemm(g_metal, cmd, lc->b_w, lc->b_s, lc->b_b,
                            LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, N, 2);
                        gpu_encode_pfb_gemm(g_metal, cmd, lc->a_w, lc->a_s, lc->a_b,
                            LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, N, 3);

                        {
                            uint32_t conv_dim = LINEAR_CONV_DIM;
                            uint32_t tgs = (conv_dim + 255) / 256;
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->conv1d_step_batched];
                            [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[0] offset:0 atIndex:1];
                            [enc setBuffer:g_metal->wf_buf         offset:conv_w_off atIndex:2];
                            [enc setBuffer:g_metal->buf_pfb_out[4] offset:0 atIndex:3];
                            [enc setBytes:&conv_dim length:4 atIndex:4];
                            [enc setBytes:&batch_n  length:4 atIndex:5];
                            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc endEncoding];
                        }

                        {
                            uint32_t key_dim = LINEAR_KEY_DIM;
                            float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
                            uint32_t token_stride = LINEAR_CONV_DIM;
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->rms_norm_qk_batched];
                            [enc setBuffer:g_metal->buf_pfb_out[4] offset:0 atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[4] offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:1];
                            [enc setBytes:&key_dim      length:4 atIndex:2];
                            [enc setBytes:&inv_scale    length:4 atIndex:3];
                            [enc setBytes:&token_stride length:4 atIndex:4];
                            [enc setBytes:&batch_n      length:4 atIndex:5];
                            [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_K_HEADS, N, 1)
                                threadsPerThreadgroup:MTLSizeMake(LINEAR_KEY_DIM, 1, 1)];
                            [enc endEncoding];
                        }

                        {
                            uint32_t total = batch_n * LINEAR_NUM_V_HEADS;
                            uint32_t tgs = (total + 255) / 256;
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->compute_decay_beta_batched];
                            [enc setBuffer:g_metal->buf_pfb_out[3] offset:0 atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[2] offset:0 atIndex:1];
                            [enc setBuffer:g_metal->wf_buf         offset:a_log_off atIndex:2];
                            [enc setBuffer:g_metal->wf_buf         offset:dt_bias_off atIndex:3];
                            [enc setBuffer:g_metal->buf_pfb_out[0] offset:0 atIndex:4];
                            [enc setBuffer:g_metal->buf_pfb_out[2] offset:0 atIndex:5];
                            [enc setBytes:&batch_n length:4 atIndex:6];
                            { uint32_t nvh = LINEAR_NUM_V_HEADS;
                              [enc setBytes:&nvh length:4 atIndex:7]; }
                            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc endEncoding];
                        }

                        {
                            uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
                            uint32_t token_stride = LINEAR_CONV_DIM;
                            uint32_t output_stride = LINEAR_TOTAL_VALUE;
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->delta_net_step_batched];
                            [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[4] offset:0 atIndex:1];
                            [enc setBuffer:g_metal->buf_pfb_out[4] offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:2];
                            [enc setBuffer:g_metal->buf_pfb_out[4] offset:2 * LINEAR_TOTAL_KEY * sizeof(float) atIndex:3];
                            [enc setBuffer:g_metal->buf_pfb_out[0] offset:0 atIndex:4];
                            [enc setBuffer:g_metal->buf_pfb_out[2] offset:0 atIndex:5];
                            [enc setBuffer:g_metal->buf_pfb_out[3] offset:0 atIndex:6];
                            [enc setBytes:&khpv          length:4 atIndex:7];
                            [enc setBytes:&token_stride  length:4 atIndex:8];
                            [enc setBytes:&output_stride length:4 atIndex:9];
                            [enc setBytes:&batch_n       length:4 atIndex:10];
                            { uint32_t nvh = LINEAR_NUM_V_HEADS;
                              uint32_t kd = LINEAR_KEY_DIM;
                              uint32_t vd = LINEAR_VALUE_DIM;
                              [enc setBytes:&nvh length:4 atIndex:11];
                              [enc setBytes:&kd  length:4 atIndex:12];
                              [enc setBytes:&vd  length:4 atIndex:13]; }
                            [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
                            [enc endEncoding];
                        }

                        {
                            uint32_t value_dim = LINEAR_VALUE_DIM;
                            uint32_t output_stride = LINEAR_TOTAL_VALUE;
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->gated_rms_norm_batched];
                            [enc setBuffer:g_metal->buf_pfb_out[3] offset:0 atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[1] offset:0 atIndex:1];
                            [enc setBuffer:g_metal->wf_buf         offset:gnorm_w_off atIndex:2];
                            [enc setBuffer:g_metal->buf_pfb_out[6] offset:0 atIndex:3];
                            [enc setBytes:&value_dim     length:4 atIndex:4];
                            [enc setBytes:&output_stride length:4 atIndex:5];
                            [enc setBytes:&eps           length:4 atIndex:6];
                            [enc setBytes:&batch_n       length:4 atIndex:7];
                            [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, N, 1)
                                threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
                            [enc endEncoding];
                        }

                        gpu_encode_pfb_gemm_ex(g_metal, cmd, lc->out_proj_w, lc->out_proj_s, lc->out_proj_b,
                            HIDDEN_DIM, LINEAR_TOTAL_VALUE, GROUP_SIZE, N,
                            g_metal->buf_pfb_out[6], 0, g_metal->buf_pfb_out[5], 0);

                        {
                            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                            [enc setComputePipelineState:g_metal->prefill_residual_norm];
                            [enc setBuffer:hidden_cur_buf       offset:hidden_off atIndex:0];
                            [enc setBuffer:g_metal->buf_pfb_out[5] offset:0 atIndex:1];
                            [enc setBuffer:(use_routed_moe_tail ? hidden_next_buf : g_metal->buf_pfb_out[6])
                                   offset:(use_routed_moe_tail ? hidden_off : 0) atIndex:2];
                            [enc setBuffer:g_metal->buf_pfb_input offset:0 atIndex:3];
                            [enc setBuffer:g_metal->wf_buf        offset:post_norm_off atIndex:4];
                            [enc setBytes:&dim length:4 atIndex:5];
                            [enc setBytes:&eps length:4 atIndex:6];
                            [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc endEncoding];
                        }

                        if (!use_routed_moe_tail) {
                            gpu_encode_pfb_gemm(g_metal, cmd, lc->sg_w, lc->sg_s, lc->sg_b,
                                SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, N, 0);
                            gpu_encode_pfb_gemm(g_metal, cmd, lc->su_w, lc->su_s, lc->su_b,
                                SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, N, 1);
                            gpu_encode_pfb_gemm(g_metal, cmd, lc->seg_w, lc->seg_s, lc->seg_b,
                                1, HIDDEN_DIM, GROUP_SIZE, N, 2);

                            {
                                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                                [enc setComputePipelineState:g_metal->prefill_swiglu];
                                [enc setBuffer:g_metal->buf_pfb_out[0] offset:0 atIndex:0];
                                [enc setBuffer:g_metal->buf_pfb_out[1] offset:0 atIndex:1];
                                uint32_t total = (uint32_t)((size_t)N * SHARED_INTERMEDIATE);
                                [enc setBytes:&total length:4 atIndex:2];
                                uint32_t tgs = (total + 255) / 256;
                                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                                [enc endEncoding];
                            }

                            gpu_encode_pfb_gemm_ex(g_metal, cmd, lc->sd_w, lc->sd_s, lc->sd_b,
                                HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, N,
                                g_metal->buf_pfb_out[0], 0, g_metal->buf_pfb_out[3], 0);

                            {
                                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                                [enc setComputePipelineState:g_metal->prefill_combine];
                                [enc setBuffer:g_metal->buf_pfb_out[6] offset:0 atIndex:0];
                                [enc setBuffer:g_metal->buf_pfb_out[3] offset:0 atIndex:1];
                                [enc setBuffer:g_metal->buf_pfb_out[2] offset:0 atIndex:2];
                                [enc setBuffer:hidden_next_buf         offset:hidden_off atIndex:3];
                                [enc setBytes:&dim length:4 atIndex:4];
                                [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
                                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                                [enc endEncoding];
                            }
                        }
                    }

                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
            }

            if (used_batched_layer && use_routed_moe_tail) {
                int tail_K = g_prefill_experts_full_only ? K_decode : prefill_K;
                for (int t = 0; t < num_prefill; t++) {
                    memcpy(hidden, hidden_out_cpu + (size_t)t * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
                    moe_forward(wf, layer, hidden, NULL, tail_K, layer_fds[layer]);
                    memcpy(hidden_out_cpu + (size_t)t * HIDDEN_DIM, hidden, HIDDEN_DIM * sizeof(float));
                }
            }

            if (!used_batched_layer) {
                int layer_K = prefill_K;
                for (int t = 0; t < num_prefill; t++) {
                    int cur_pos = pos_start + t;
                    memcpy(hidden, hidden_in_cpu + (size_t)t * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        cur_pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        layer_K, layer_fds[layer]);
                    if (layer < NUM_LAYERS - 1) {
                        complete_deferred_experts();
                        memcpy(hidden_out_cpu + (size_t)t * HIDDEN_DIM, hidden, HIDDEN_DIM * sizeof(float));
                    } else {
                        discard_deferred_experts();
                    }
                }
            }

            if (use_gpu_hidden_buffers && layer < NUM_LAYERS - 1) {
                id<MTLBuffer> tmp = hidden_cur_buf;
                hidden_cur_buf = hidden_next_buf;
                hidden_next_buf = tmp;
            }
        } // @autoreleasepool
    }

    return num_prefill;
}

// ============================================================================
// Main entry point
// ============================================================================
static int batched_prefill_k0(
    WeightFile *wf, float *hidden, float *embed_batch,
    int num_prefill, int pos_start,
    KVCache **kv_caches, void **layer_states, void **layer_mmaps, int *layer_fds,
    int K_decode
) {
    double t_start = now_ms();
    // prefill_K controls the shared-expert-only batched pipeline.
    // When experts_full_only, set prefill_K=0 so the batched pipeline skips routed experts,
    // but use_routed_moe_tail adds per-token expert I/O at full-attn layers after.
    int prefill_K = (effective_prefill_skip_experts() || g_prefill_experts_full_only) ? 0 : K_decode;

    int result;
    if (g_prefill_batch > 1) {
        printf("[prefill] starting %d tokens | batch=%d skip_experts=%d experts_full_only=%d batched_linear=%d K=%d\n",
               num_prefill, g_prefill_batch,
               effective_prefill_skip_experts(), g_prefill_experts_full_only,
               !g_disable_batched_linear, prefill_K);
        result = prefill_k0_layer_first(wf, hidden, embed_batch, num_prefill, pos_start,
                                         g_prefill_batch, prefill_K, K_decode,
                                         kv_caches, layer_states, layer_mmaps, layer_fds);
    } else {
        printf("[prefill] starting %d tokens | per-token K=%d skip_experts=%d\n",
               num_prefill, prefill_K, effective_prefill_skip_experts());
        result = prefill_k0_token_first(wf, hidden, embed_batch, num_prefill, pos_start, prefill_K,
                                         kv_caches, layer_states, layer_mmaps, layer_fds);
    }

    double elapsed = now_ms() - t_start;
    printf("[prefill done] %d tokens | %.0f ms | %.1f tok/s%s\n",
           num_prefill, elapsed, 1000.0 * num_prefill / elapsed,
           g_prefill_batch > 1 ? " (batched)" : "");

    return result;
}
