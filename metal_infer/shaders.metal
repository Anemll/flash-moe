/*
 * shaders.metal — Optimized Metal compute shaders for 4-bit quantized MoE inference
 *
 * Core operations:
 *   1. dequant_matvec_4bit: Naive 4-bit affine dequant matvec (reference)
 *   2. dequant_matvec_4bit_fast: SIMD-optimized with simd_sum reduction
 *   3. dequant_matvec_4bit_v3: Fully optimized — tiled threadgroup, vector loads,
 *      coalesced access, shared input cache. Target: <0.1ms per matmul.
 *   4. swiglu_fused / swiglu_fused_vec4: SwiGLU activation
 *   5. weighted_sum: combine expert outputs with routing weights
 *   6. rms_norm: RMS normalization
 *
 * Quantization format (MLX affine 4-bit, group_size=64):
 *   - Weights stored as uint32, each holding 8 x 4-bit values
 *   - Per-group scale and bias in bfloat16
 *   - Dequantized value = uint4_val * scale + bias
 *   - Groups of 64 elements share one (scale, bias) pair
 *
 * Matrix layout for expert projections:
 *   gate_proj/up_proj: [1024, 512] uint32 = [1024, 4096] logical (out=1024, in=4096)
 *   down_proj: [4096, 128] uint32 = [4096, 1024] logical (out=4096, in=1024)
 *
 *   Scales/biases: [out_dim, in_dim/group_size]
 *   gate/up scales: [1024, 64]   (4096/64 = 64 groups)
 *   down scales:    [4096, 16]   (1024/64 = 16 groups)
 */

#include <metal_stdlib>
using namespace metal;
#include "gguf_iq_shared.h"

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

inline uint16_t f32_to_bf16(float f) {
    return uint16_t(as_type<uint>(f) >> 16);
}

inline float f16_to_f32(ushort fp16_bits) {
    return float(as_type<half>(fp16_bits));
}

#define GGUF_QK8_0 32
#define GGUF_QK_K FLASH_MOE_GGUF_QK_K
#define GGUF_Q8_MAX_IN_DIM 8192

struct GGUFBlockQ8_0 {
    ushort d;
    char   qs[GGUF_QK8_0];
};

struct GGUFBlockQ6K {
    uchar ql[GGUF_QK_K/2];
    uchar qh[GGUF_QK_K/4];
    char  scales[GGUF_QK_K/16];
    ushort d;
};

static inline uchar2 flash_moe_get_scale_min_k4(uint j, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j] & 63), uchar(q[j + 4] & 63)}
                 : uchar2{uchar((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)),
                          uchar((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4))};
}

kernel void dequant_matvec_iq3_xxs(
    device const GGUFBlockIQ3XXS* W [[buffer(0)]],
    device const uint16_t*        unused_scales [[buffer(1)]],
    device const uint16_t*        unused_biases [[buffer(2)]],
    device const float*           x [[buffer(3)]],
    device float*                 out [[buffer(4)]],
    constant uint&                out_dim [[buffer(5)]],
    constant uint&                in_dim [[buffer(6)]],
    constant uint&                unused_group_size [[buffer(7)]],
    threadgroup char*             shmem [[threadgroup(0)]],
    uint                          tgid [[threadgroup_position_in_grid]],
    ushort                        tiisg [[thread_index_in_simdgroup]],
    ushort                        sgitg [[simdgroup_index_in_threadgroup]]
) {
    (void)unused_scales;
    (void)unused_biases;
    (void)unused_group_size;

    constexpr ushort NR0 = 4;
    constexpr ushort NSG = 2;

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= out_dim) return;

    const uint blocks_per_row = in_dim / GGUF_QK_K;
    const uint nb32_total = blocks_per_row * (GGUF_QK_K / 32);
    device const GGUFBlockIQ3XXS* row_blocks = W + first_row * blocks_per_row;

    threadgroup uint*  svalues = reinterpret_cast<threadgroup uint*>(shmem);
    threadgroup uchar* ssigns  = reinterpret_cast<threadgroup uchar*>(svalues + 256);

    const uint grid_pos = (32u * sgitg + tiisg) * 4u;
    for (uint i = 0; i < 4; ++i) {
        svalues[grid_pos + i] = flash_moe_iq3xxs_grid[grid_pos + i];
    }
    const uint sign_pos = (32u * sgitg + tiisg) * 2u;
    for (uint i = 0; i < 2; ++i) {
        ssigns[sign_pos + i] = flash_moe_ksigns_iq2xs[sign_pos + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float yl[32];
    float sumf[NR0] = {0.0f, 0.0f, 0.0f, 0.0f};

    device const float* x_block = x + 32 * tiisg;

    for (uint ib32 = tiisg; ib32 < nb32_total; ib32 += 32) {
        for (uint i = 0; i < 32; ++i) {
            yl[i] = x_block[i];
        }

        const uint bi = ib32 / (GGUF_QK_K / 32);
        const uint sub = ib32 % (GGUF_QK_K / 32);

        device const GGUFBlockIQ3XXS* xr = row_blocks + bi;
        for (ushort row = 0; row < NR0 && first_row + row < out_dim; ++row) {
            device const GGUFBlockIQ3XXS& blk = xr[row * blocks_per_row];
            device const uchar* q3 = blk.qs + 8 * sub;
            device const ushort* gas = reinterpret_cast<device const ushort*>(blk.qs + GGUF_QK_K / 4) + 2 * sub;
            const uint aux32 = uint(gas[0]) | (uint(gas[1]) << 16);
            const float d = f16_to_f32(blk.d) * (0.5f + float(aux32 >> 28));

            float2 sum = float2(0.0f);
            for (uint l = 0; l < 4; ++l) {
                const threadgroup uchar* grid1 = reinterpret_cast<threadgroup uchar*>(svalues + q3[2 * l + 0]);
                const threadgroup uchar* grid2 = reinterpret_cast<threadgroup uchar*>(svalues + q3[2 * l + 1]);
                const uchar signs = ssigns[(aux32 >> (7 * l)) & 127];

                for (uint j = 0; j < 4; ++j) {
                    sum[0] += yl[8 * l + j + 0] * float(grid1[j]) * ((signs & flash_moe_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    sum[1] += yl[8 * l + j + 4] * float(grid2[j]) * ((signs & flash_moe_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);
        }

        x_block += 32 * 32;
    }

    for (ushort row = 0; row < NR0 && first_row + row < out_dim; ++row) {
        const float sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            out[first_row + row] = sum * 0.5f;
        }
    }
}

kernel void dequant_matvec_iq4_xs(
    device const GGUFBlockIQ4XS* W [[buffer(0)]],
    device const uint16_t*       unused_scales [[buffer(1)]],
    device const uint16_t*       unused_biases [[buffer(2)]],
    device const float*          x [[buffer(3)]],
    device float*                out [[buffer(4)]],
    constant uint&               out_dim [[buffer(5)]],
    constant uint&               in_dim [[buffer(6)]],
    constant uint&               unused_group_size [[buffer(7)]],
    threadgroup char*            shmem [[threadgroup(0)]],
    uint                         tgid [[threadgroup_position_in_grid]],
    ushort                       tiisg [[thread_index_in_simdgroup]],
    ushort                       sgitg [[simdgroup_index_in_threadgroup]]
) {
    (void)unused_scales;
    (void)unused_biases;
    (void)unused_group_size;

    constexpr ushort NR0 = 2;
    constexpr ushort NSG = 2;

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= out_dim) return;

    threadgroup float * shmem_f32 = reinterpret_cast<threadgroup float *>(shmem);
    const uint blocks_per_row = in_dim / GGUF_QK_K;

    const ushort ix = tiisg / 16;  // 0 or 1
    const ushort it = tiisg % 16;  // 0..15
    const ushort ib = it / 2;
    const ushort il = it % 2;

    shmem_f32[tiisg] = flash_moe_kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0] = {0.0f, 0.0f};
    device const float * yb = x + ix * GGUF_QK_K + ib * 32 + il * 8;

    uint2 aux32;
    thread const uchar * q8 = reinterpret_cast<thread const uchar *>(&aux32);

    for (uint bi = ix; bi < blocks_per_row; bi += 2) {
        device const float4 * y4 = reinterpret_cast<device const float4 *>(yb);
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (ushort row = 0; row < NR0 && first_row + row < out_dim; ++row) {
            device const GGUFBlockIQ4XS & blk = W[(first_row + row) * blocks_per_row + bi];
            device const uint * q4 = reinterpret_cast<device const uint *>(blk.qs + 16 * ib + 8 * il);

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);

            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            float4 qf1 = float4(shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]);
            float4 qf2 = float4(shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]);
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = float4(shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]);
            qf2 = float4(shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]);
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = int((blk.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                           (int((blk.scales_h >> (2 * ib)) & 0x3) << 4);
            sumf[row] += f16_to_f32(blk.d) * float(ls - 32) *
                         (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * GGUF_QK_K;
    }

    for (ushort row = 0; row < NR0 && first_row + row < out_dim; ++row) {
        const float sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            out[first_row + row] = sum;
        }
    }
}

static inline uchar2 flash_moe_get_scale_min_k4_just2(uint j, device const uchar *q) {
    if (j < 4) {
        return uchar2{uchar(q[j + 0] & 63), uchar(q[j + 4] & 63)};
    }
    return uchar2{
        uchar((q[j + 4] & 0xF) | ((q[j - 4] & 0xC0) >> 2)),
        uchar((q[j + 4] >> 4) | ((q[j - 0] & 0xC0) >> 2))
    };
}

kernel void dequant_matvec_q5_k(
    device const GGUFBlockQ5K* W [[buffer(0)]],
    device const uint16_t*     unused_scales [[buffer(1)]],
    device const uint16_t*     unused_biases [[buffer(2)]],
    device const float*        x [[buffer(3)]],
    device float*              out [[buffer(4)]],
    constant uint&             out_dim [[buffer(5)]],
    constant uint&             in_dim [[buffer(6)]],
    constant uint&             unused_group_size [[buffer(7)]],
    uint                       tgid [[threadgroup_position_in_grid]],
    ushort                     tiisg [[thread_index_in_simdgroup]],
    ushort                     sgitg [[simdgroup_index_in_threadgroup]]
) {
    (void)unused_scales;
    (void)unused_biases;
    (void)unused_group_size;

    constexpr ushort NSG = 2;

    const uint row = tgid * NSG + sgitg;
    if (row >= out_dim) return;

    const uint blocks_per_row = in_dim / GGUF_QK_K;
    device const GGUFBlockQ5K * row_blocks = W + row * blocks_per_row;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;
    const short l0  = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uchar hm1 = uchar(1u << (2 * iq));
    const uchar hm2 = uchar(hm1 << 1);
    const uchar hm3 = uchar(hm1 << 4);
    const uchar hm4 = uchar(hm2 << 4);

    uint16_t sc16[4];
    thread const uchar * sc8 = reinterpret_cast<thread const uchar *>(sc16);
    float sumf = 0.0f;
    device const float * y1 = x + ix * GGUF_QK_K + y_offset;

    for (uint bi = ix; bi < blocks_per_row; bi += 4) {
        device const GGUFBlockQ5K &blk = row_blocks[bi];
        device const uchar * q1 = blk.qs + q_offset;
        device const uchar * qh = blk.qh + l0;
        device const ushort * a = reinterpret_cast<device const ushort *>(blk.scales) + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = float4(0.0f);
        float yl[16];
        float yh[16];
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];  sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l + 0];  sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
        }

        sc16[0] = a[0] & 0x3f3f;
        sc16[1] = a[2] & 0x3f3f;
        sc16[2] = ((a[4] >> 0) & 0x0f0f) | ((a[0] & 0xc0c0) >> 2);
        sc16[3] = ((a[4] >> 4) & 0x0f0f) | ((a[2] & 0xc0c0) >> 2);

        float4 acc1 = {0.0f};
        float4 acc2 = {0.0f};
        device const uchar * q2 = q1 + 64;
        for (short l = 0; l < 8; ++l) {
            const uchar h = qh[l];
            acc1[0] += yl[l + 0] * float(q1[l] & 0x0F);
            acc1[1] += yl[l + 8] * float(q1[l] & 0xF0);
            acc1[2] += yh[l + 0] * float(q2[l] & 0x0F);
            acc1[3] += yh[l + 8] * float(q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
            acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
            acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
            acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
        }

        const float d = f16_to_f32(blk.d);
        const float minv = f16_to_f32(blk.dmin);
        sumf += d * (float(sc8[0]) * (acc1[0]         + 16.0f * acc2[0]) +
                     float(sc8[1]) * (acc1[1] / 16.0f + 16.0f * acc2[1]) +
                     float(sc8[4]) * (acc1[2]         + 16.0f * acc2[2]) +
                     float(sc8[5]) * (acc1[3] / 16.0f + 16.0f * acc2[3])) -
                minv * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                        sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

        y1 += 4 * GGUF_QK_K;
    }

    const float sum = simd_sum(sumf);
    if (tiisg == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1: 4-bit dequantized matrix-vector multiply (NAIVE — reference)
// ============================================================================

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    float acc = 0.0f;

    device const uint32_t* w_row = W_packed + tid * packed_cols;
    device const uint16_t* s_row = scales + tid * num_groups;
    device const uint16_t* b_row = biases + tid * num_groups;

    for (uint g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            for (uint n = 0; n < 8; n++) {
                uint nibble = (packed >> (n * 4)) & 0xF;
                float w_val = float(nibble) * scale + bias;
                acc += w_val * x[x_base + n];
            }
        }
    }

    out[tid] = acc;
}


// ============================================================================
// Kernel 1b: 4-bit dequant matvec — SIMD-optimized (legacy, kept for compat)
// ============================================================================

kernel void dequant_matvec_4bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            acc += (float((packed >>  0) & 0xF) * scale + bias) * x[x_base + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x[x_base + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x[x_base + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x[x_base + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x[x_base + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x[x_base + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x[x_base + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x[x_base + 7];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}

// ============================================================================
// GGUF Q8_0 matvec for resident tensors
// Reuses the same tiled shape as the 4-bit kernel: 8 rows per threadgroup.
// ============================================================================

kernel void dequant_matvec_q8_0(
    device const GGUFBlockQ8_0* W      [[buffer(0)]],
    device const float*         x      [[buffer(1)]],
    device float*               out    [[buffer(2)]],
    constant uint&              out_dim [[buffer(3)]],
    constant uint&              in_dim  [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tgid * 8 + simd_group;
    const uint blocks_per_row = in_dim / GGUF_QK8_0;

    threadgroup float x_shared[GGUF_Q8_MAX_IN_DIM];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const GGUFBlockQ8_0* row_blocks = W + row * blocks_per_row;
    float acc = 0.0f;

    for (uint bi = simd_lane; bi < blocks_per_row; bi += 32) {
        device const GGUFBlockQ8_0& blk = row_blocks[bi];
        const float d = f16_to_f32(blk.d);
        const uint base = bi * GGUF_QK8_0;
        for (uint i = 0; i < GGUF_QK8_0; i++) {
            acc += d * float(blk.qs[i]) * x_shared[base + i];
        }
    }

    const float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// GGUF Q6_K matvec for resident tensors
// One SIMD group handles one output row exactly, preserving GGUF block layout.
// ============================================================================

kernel void dequant_matvec_q6_k(
    device const GGUFBlockQ6K* W      [[buffer(0)]],
    device const float*        x      [[buffer(1)]],
    device float*              out    [[buffer(2)]],
    constant uint&             out_dim [[buffer(3)]],
    constant uint&             in_dim  [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgpig.x;
    if (row >= out_dim) return;

    const uint blocks_per_row = in_dim / GGUF_QK_K;
    device const GGUFBlockQ6K* row_blocks = W + row * blocks_per_row;

    float acc = 0.0f;

    for (uint bi = 0; bi < blocks_per_row; ++bi) {
        device const GGUFBlockQ6K& blk = row_blocks[bi];
        const float d = f16_to_f32(blk.d);
        device const float* x_block = x + bi * GGUF_QK_K;

        for (uint part = 0; part < GGUF_QK_K; part += 128) {
            device const uchar* ql = blk.ql + part / 2;
            device const uchar* qh = blk.qh + part / 4;
            device const char* sc = blk.scales + part / 16;

            const uint is = lane / 16;
            const uchar qh_lane = qh[lane];
            const int q1 = int((ql[lane +  0] & 0xF) | (((qh_lane >> 0) & 0x3) << 4)) - 32;
            const int q2 = int((ql[lane + 32] & 0xF) | (((qh_lane >> 2) & 0x3) << 4)) - 32;
            const int q3 = int((ql[lane +  0] >> 4) | (((qh_lane >> 4) & 0x3) << 4)) - 32;
            const int q4 = int((ql[lane + 32] >> 4) | (((qh_lane >> 6) & 0x3) << 4)) - 32;

            const uint base = part + lane;
            acc += d * float(sc[is + 0]) * float(q1) * x_block[base +  0];
            acc += d * float(sc[is + 2]) * float(q2) * x_block[base + 32];
            acc += d * float(sc[is + 4]) * float(q3) * x_block[base + 64];
            acc += d * float(sc[is + 6]) * float(q4) * x_block[base + 96];
        }
    }

    const float sum = simd_sum(acc);
    if (lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Fused gate+up+SwiGLU: reads x ONCE, computes silu(gate(x)) * up(x)
// Saves one input read + one kernel dispatch per expert
// ============================================================================
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint16_t* gate_s    [[buffer(1)]],
    device const uint16_t* gate_b    [[buffer(2)]],
    device const uint32_t* up_W      [[buffer(3)]],
    device const uint16_t* up_s      [[buffer(4)]],
    device const uint16_t* up_b      [[buffer(5)]],
    device const float*    x         [[buffer(6)]],
    device float*          out       [[buffer(7)]],
    constant uint&         out_dim   [[buffer(8)]],
    constant uint&         in_dim    [[buffer(9)]],
    constant uint&         group_size [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint16_t* gs = gate_s + tgid * num_groups;
    device const uint16_t* gb = gate_b + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint16_t* us = up_s   + tgid * num_groups;
    device const uint16_t* ub = up_b   + tgid * num_groups;
    float ga = 0.0f, ua = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float gsc = bf16_to_f32(gs[g]), gbi = bf16_to_f32(gb[g]);
        float usc = bf16_to_f32(us[g]), ubi = bf16_to_f32(ub[g]);
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp+p], up = ur[bp+p];
            for (uint i = 0; i < 8; i++) {
                float xv = x[bx + p*8 + i];
                ga += (float((gp>>(i*4))&0xF)*gsc+gbi)*xv;
                ua += (float((up>>(i*4))&0xF)*usc+ubi)*xv;
            }
        }
    }
    threadgroup float sg[32], su[32];
    float rg = simd_sum(ga), ru = simd_sum(ua);
    uint sl = lid%32, si = lid/32, ns = (tg_size+31)/32;
    if (sl==0) { sg[si]=rg; su[si]=ru; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (si==0 && sl<ns) {
        float vg=simd_sum(sg[sl]), vu=simd_sum(su[sl]);
        if (sl==0) out[tgid] = (vg/(1.0f+exp(-vg))) * vu;
    }
}

// ============================================================================
// Kernel 1c: FULLY OPTIMIZED 4-bit dequant matvec
// ============================================================================
//
// Design for M3 Max (40-core GPU, SIMD width 32):
//
// Strategy: Each threadgroup handles ROWS_PER_TG output rows.
//   - Threadgroup size = 256 (8 SIMD groups of 32)
//   - Each SIMD group handles one output row
//   - Within a SIMD group, 32 threads split the input dimension
//   - Each thread processes in_dim/32 input elements using vector loads
//   - Reduction via simd_sum (single instruction)
//
// Memory optimizations:
//   - Input vector x cached in threadgroup shared memory (loaded once)
//   - uint4 vector loads for weights (128 bits = 32 nibbles per load)
//   - float4 vector loads for x (128 bits = 4 floats per load)
//   - Coalesced weight reads: adjacent threads read adjacent uint4 vectors
//
// For gate/up_proj [1024, 4096]: 1024/8 = 128 threadgroups, 256 threads each
//   - 128 * 256 = 32768 threads across 40 cores = good occupancy
//   - Each thread processes 4096/32 = 128 input elements = 16 uint32 packed words
//     = 4 uint4 loads per thread per row
//
// For down_proj [4096, 1024]: 4096/8 = 512 threadgroups
//   - Each thread processes 1024/32 = 32 input elements = 4 uint32 packed words
//     = 1 uint4 load per thread per row

// Number of output rows per threadgroup = number of SIMD groups (256/32 = 8)
#define ROWS_PER_TG 8

kernel void dequant_matvec_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],     // which tile of rows
    uint lid    [[thread_position_in_threadgroup]],    // 0..255
    uint simd_lane  [[thread_index_in_simdgroup]],    // 0..31
    uint simd_group [[simdgroup_index_in_threadgroup]] // 0..7
) {
    // Which output row this SIMD group handles
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;      // uint32 columns per row
    uint num_groups  = in_dim / group_size;

    // ---- Cache input vector in threadgroup shared memory ----
    // Max in_dim = 4096, so we need 4096 floats = 16KB shared memory
    // This is well within the 32KB threadgroup memory limit on M3
    threadgroup float x_shared[4096];

    // Cooperative load: 256 threads load 4096 floats (16 per thread)
    // ALL threads must participate in this load + barrier, even if their
    // row is out of bounds. Early return before the barrier causes only
    // partial loading of x_shared, corrupting results for valid rows.
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now safe to bail out for out-of-bounds rows
    if (row >= out_dim) return;

    // ---- Pointer setup for this row ----
    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    // ---- Each lane processes a strided slice of the packed columns ----
    // Lane k processes columns: k, k+32, k+64, ...
    // This gives coalesced reads: adjacent lanes read adjacent uint32 words.

    float acc = 0.0f;

    // Process packed columns in strides of 32 (one per SIMD lane)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // Determine which group this column belongs to
        // packed_per_group = group_size / 8 = 64 / 8 = 8
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        // Dequantize 8 nibbles and multiply with cached x
        // Rearranged: (nibble * scale + bias) * x = nibble * (scale*x) + bias*x
        // Pre-compute scale*x and bias*x, then use FMA for dequant+multiply in one op.
        // This reduces per-nibble from (convert + mul + add + mul + add) to (convert + FMA + add).
        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    // ---- SIMD reduction: sum across 32 lanes ----
    float sum = simd_sum(acc);

    // Lane 0 writes the result
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1g: Batched 4-bit dequant GEMM for prefill (N tokens, read weights once)
// ============================================================================
// Processes N input vectors against the same weight matrix in one dispatch.
// Weight data is read once per row and reused across all N tokens.
// Each token's input is loaded into shared memory one at a time (16KB),
// and its partial dot products are accumulated in per-token registers.
//
// Speedup: N× less weight bandwidth vs N separate GEMV dispatches.
// Used during prefill to amortize projection weight reads.
//
// Layout: X[N, in_dim] token-major, Y[N, out_dim] token-major
//
#define MAX_PFB_GPU 32

kernel void dequant_gemm_4bit_batch(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    X          [[buffer(3)]],  // [N, in_dim] token-major
    device float*          Y          [[buffer(4)]],  // [N, out_dim] token-major
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         batch_n    [[buffer(8)]],  // number of tokens (1..MAX_PFB_GPU)
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each SIMD group handles one output row (same as v3)
    uint row = tgid * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Per-token accumulators in registers
    float acc[MAX_PFB_GPU];
    for (uint n = 0; n < batch_n; n++) acc[n] = 0.0f;

    // Pointers for this row's weights (read ONCE, reused across all N tokens)
    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    // Outer loop: iterate over weight columns (each SIMD lane handles strided cols)
    // Weight data read once; inner loop multiplies against all N input vectors.
    // Input reads X[n * in_dim + x_base + nib] are coherent across SIMD lanes
    // (all 32 lanes read same n,nib combo) so GPU cache line serves all lanes.
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        // Read weight ONCE
        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        // Multiply dequantized weights against all N input vectors
        // Uses same FMA pattern as v3: fma(nibble, scale*x, bias*x)
        // to ensure bit-identical results.
        for (uint n = 0; n < batch_n; n++) {
            device const float* x_n = X + n * in_dim + x_base;

            float sx0 = scale * x_n[0];  float bx0 = bias * x_n[0];
            float sx1 = scale * x_n[1];  float bx1 = bias * x_n[1];
            float sx2 = scale * x_n[2];  float bx2 = bias * x_n[2];
            float sx3 = scale * x_n[3];  float bx3 = bias * x_n[3];
            float sx4 = scale * x_n[4];  float bx4 = bias * x_n[4];
            float sx5 = scale * x_n[5];  float bx5 = bias * x_n[5];
            float sx6 = scale * x_n[6];  float bx6 = bias * x_n[6];
            float sx7 = scale * x_n[7];  float bx7 = bias * x_n[7];

            acc[n] += fma(float((packed >>  0) & 0xF), sx0, bx0);
            acc[n] += fma(float((packed >>  4) & 0xF), sx1, bx1);
            acc[n] += fma(float((packed >>  8) & 0xF), sx2, bx2);
            acc[n] += fma(float((packed >> 12) & 0xF), sx3, bx3);
            acc[n] += fma(float((packed >> 16) & 0xF), sx4, bx4);
            acc[n] += fma(float((packed >> 20) & 0xF), sx5, bx5);
            acc[n] += fma(float((packed >> 24) & 0xF), sx6, bx6);
            acc[n] += fma(float((packed >> 28) & 0xF), sx7, bx7);
        }
    }

    // SIMD reduce + write results for all tokens
    for (uint n = 0; n < batch_n; n++) {
        float sum = simd_sum(acc[n]);
        if (simd_lane == 0) {
            Y[n * out_dim + row] = sum;
        }
    }
}


// ============================================================================
// Kernel 1f: 4-bit dequant matvec with LUT (eliminates uint→float conversions)
// ============================================================================
// Instead of converting each nibble to float (expensive conversion instruction),
// pre-compute a 16-entry LUT per group: lut[v] = float(v) * scale + bias.
// Then inner loop is just: acc += lut[nibble] * x_shared[i] — pure math, no conversions.
// The LUT is recomputed every group_size/8 iterations (amortized).

#define ROWS_PER_TG_V5 8

kernel void dequant_matvec_4bit_v5(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_V5 + simd_group;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;
    uint prev_g = 0xFFFFFFFF;
    float lut[16];

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_group;

        // Rebuild LUT when group changes
        if (g != prev_g) {
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);
            for (uint v = 0; v < 16; v++) {
                lut[v] = float(v) * scale + bias;
            }
            prev_g = g;
        }

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += lut[(packed >>  0) & 0xF] * x_shared[x_base + 0];
        acc += lut[(packed >>  4) & 0xF] * x_shared[x_base + 1];
        acc += lut[(packed >>  8) & 0xF] * x_shared[x_base + 2];
        acc += lut[(packed >> 12) & 0xF] * x_shared[x_base + 3];
        acc += lut[(packed >> 16) & 0xF] * x_shared[x_base + 4];
        acc += lut[(packed >> 20) & 0xF] * x_shared[x_base + 5];
        acc += lut[(packed >> 24) & 0xF] * x_shared[x_base + 6];
        acc += lut[(packed >> 28) & 0xF] * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 1e: 2-bit affine dequant matvec (same structure as v3)
// ============================================================================
// Packs 16 x 2-bit values per uint32. Each value is 0-3, dequantized as:
//   val = uint2 * scale + bias (same affine quantization, just 2-bit range)
// Same group structure: group_size elements share one (scale, bias) pair.
// packed_cols = in_dim / 16 (16 values per uint32, vs 8 for 4-bit)

kernel void dequant_matvec_2bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/16]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 16;  // 16 values per uint32 for 2-bit
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    // Each lane processes strided columns (16 values per uint32)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // group_size/16 packed words per group
        uint g = col / (group_size / 16);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 16;

        // Unroll 16 x 2-bit extractions
        acc += (float((packed >>  0) & 0x3) * scale + bias) * x_shared[x_base +  0];
        acc += (float((packed >>  2) & 0x3) * scale + bias) * x_shared[x_base +  1];
        acc += (float((packed >>  4) & 0x3) * scale + bias) * x_shared[x_base +  2];
        acc += (float((packed >>  6) & 0x3) * scale + bias) * x_shared[x_base +  3];
        acc += (float((packed >>  8) & 0x3) * scale + bias) * x_shared[x_base +  4];
        acc += (float((packed >> 10) & 0x3) * scale + bias) * x_shared[x_base +  5];
        acc += (float((packed >> 12) & 0x3) * scale + bias) * x_shared[x_base +  6];
        acc += (float((packed >> 14) & 0x3) * scale + bias) * x_shared[x_base +  7];
        acc += (float((packed >> 16) & 0x3) * scale + bias) * x_shared[x_base +  8];
        acc += (float((packed >> 18) & 0x3) * scale + bias) * x_shared[x_base +  9];
        acc += (float((packed >> 20) & 0x3) * scale + bias) * x_shared[x_base + 10];
        acc += (float((packed >> 22) & 0x3) * scale + bias) * x_shared[x_base + 11];
        acc += (float((packed >> 24) & 0x3) * scale + bias) * x_shared[x_base + 12];
        acc += (float((packed >> 26) & 0x3) * scale + bias) * x_shared[x_base + 13];
        acc += (float((packed >> 28) & 0x3) * scale + bias) * x_shared[x_base + 14];
        acc += (float((packed >> 30) & 0x3) * scale + bias) * x_shared[x_base + 15];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1d: FULLY OPTIMIZED with uint4 vector loads
// ============================================================================
//
// Same structure as v3 but uses uint4 loads (128-bit / 16 bytes) to maximize
// memory bandwidth per thread. Each uint4 = 4 uint32 = 32 nibbles.
//
// For gate/up (packed_cols=512): each thread processes 512/32 = 16 uint32
//   = 4 uint4 loads per thread
// For down (packed_cols=128): each thread processes 128/32 = 4 uint32
//   = 1 uint4 load per thread

kernel void dequant_matvec_4bit_v4(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache input vector — ALL threads must participate before the barrier
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    // Pointers — cast to uint4 for vector loads
    device const uint4* w_row_v = (device const uint4*)(W_packed + row * packed_cols);
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    uint vec4_cols = packed_cols / 4;  // number of uint4 vectors per row

    float acc = 0.0f;

    // Each lane processes vec4_cols / 32 vectors (coalesced: adjacent lanes read adjacent uint4)
    for (uint vi = simd_lane; vi < vec4_cols; vi += 32) {
        uint4 packed4 = w_row_v[vi];

        // Each uint4 covers 4 * 8 = 32 input elements
        // Starting packed column index = vi * 4
        uint base_col = vi * 4;
        uint x_base = base_col * 8;  // starting input element

        // Process each of the 4 uint32 words in the uint4
        // Unroll all 4 words x 8 nibbles = 32 multiply-adds
        #pragma unroll
        for (uint w = 0; w < 4; w++) {
            uint32_t packed = packed4[w];
            uint col = base_col + w;
            uint g = col / (group_size / 8);
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);

            uint xb = x_base + w * 8;
            acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[xb + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[xb + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[xb + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[xb + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[xb + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[xb + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[xb + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[xb + 7];
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1e: Multi-expert batched matvec
// ============================================================================
//
// Dispatch multiple experts simultaneously. The grid's Y dimension indexes
// the expert, so K experts' matmuls run as parallel threadgroups.
//
// Buffer layout: W_packed, scales, biases are arrays of K experts concatenated.
// x_inputs:  K input vectors concatenated [K * in_dim]
// out:       K output vectors concatenated [K * out_dim]
// expert_offsets: byte offset into W_packed buffer for each expert's weights
//                 (allows non-contiguous expert data in a shared buffer)

kernel void dequant_matvec_4bit_batched(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x_inputs   [[buffer(3)]],  // [K, in_dim]
    device float*          out        [[buffer(4)]],  // [K, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    // Per-expert offsets into the weight/scale/bias buffers (in elements)
    device const uint*     w_offsets  [[buffer(8)]],  // [K] offset in uint32 elements
    device const uint*     s_offsets  [[buffer(9)]],  // [K] offset in uint16 elements
    device const uint*     b_offsets  [[buffer(10)]], // [K] offset in uint16 elements
    constant uint&         num_row_tiles [[buffer(11)]], // ceil(out_dim / ROWS_PER_TG)
    uint tgid_flat [[threadgroup_position_in_grid]],  // linearized (row_tile + expert * num_row_tiles)
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // De-linearize: tgid_flat = row_tile + expert_k * num_row_tiles
    uint expert_k = tgid_flat / num_row_tiles;
    uint row_tile = tgid_flat % num_row_tiles;
    uint row = row_tile * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache this expert's input vector
    threadgroup float x_shared[4096];
    device const float* x_k = x_inputs + expert_k * in_dim;
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x_k[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Point to this expert's weights
    device const uint32_t* w_row = W_packed + w_offsets[expert_k] + row * packed_cols;
    device const uint16_t* s_row = scales   + s_offsets[expert_k] + row * num_groups;
    device const uint16_t* b_row = biases   + b_offsets[expert_k] + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[x_base + 0];
        acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[x_base + 1];
        acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[x_base + 2];
        acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[x_base + 3];
        acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[x_base + 4];
        acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[x_base + 5];
        acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[x_base + 6];
        acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert_k * out_dim + row] = sum;
    }
}


// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================

kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// Vectorized SwiGLU: process 4 elements per thread
kernel void swiglu_fused_vec4(
    device const float4* gate [[buffer(0)]],
    device const float4* up   [[buffer(1)]],
    device float4*       out  [[buffer(2)]],
    constant uint&       dim  [[buffer(3)]],  // original dim (must be multiple of 4)
    uint tid [[thread_position_in_grid]]
) {
    uint vec_dim = dim / 4;
    if (tid >= vec_dim) return;

    float4 g = gate[tid];
    float4 silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 2b: Batched SwiGLU for K experts
// ============================================================================

kernel void swiglu_fused_batched(
    device const float* gate [[buffer(0)]],  // [K * dim]
    device const float* up   [[buffer(1)]],  // [K * dim]
    device float*       out  [[buffer(2)]],  // [K * dim]
    constant uint&      dim  [[buffer(3)]],
    constant uint&      K    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = K * dim;
    if (tid >= total) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 3: Weighted sum of expert outputs
// ============================================================================

kernel void weighted_sum(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      K           [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * dim + tid];
    }
    out[tid] = acc;
}


// ============================================================================
// Kernel 4: RMS Normalization
// ============================================================================

kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];

    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            sum_sq[0] = val;
        }
    }
}

kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}


// ============================================================================
// Kernel 4b: RMS Normalization with bf16 weights
// ============================================================================
// Same as rms_norm_apply but reads weights as bfloat16 (uint16_t) and
// converts to float32 inline. Used in the fused o_proj+norm+routing path
// where norm weights come directly from the mmap'd weight file (bf16).

kernel void rms_norm_apply_bf16(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],  // bf16 weights
    device const float*    sum_sq  [[buffer(2)]],
    device float*          out     [[buffer(3)]],
    constant uint&         dim     [[buffer(4)]],
    constant float&        eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    float w = bf16_to_f32(weight[tid]);
    out[tid] = x[tid] * rms * w;
}


// ============================================================================
// Kernel 5: Residual add
// ============================================================================
// out[i] = a[i] + b[i]
// Used to fuse the residual connection into a GPU command buffer,
// eliminating a CPU round-trip between o_proj and routing.

kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}


// ============================================================================
// Kernel 6: Batched GPU attention scores (Q @ K^T, scaled) — all heads at once
// ============================================================================
//
// Computes scores[h, p] = sum_d(Q[h, d] * K[p, kv_h*head_dim + d]) * scale
// for all heads h in [0, num_heads) and positions p in [0, seq_len).
//
// Grid: linearized (pos + h * num_seq_tgs) — one threadgroup per (position, head).
// Each threadgroup of 256 threads reduces over head_dim=256.
//
// GQA mapping: kv_head = h / heads_per_kv (e.g. 16 query heads share 1 KV head)
//
// Output layout: scores[h * seq_stride + p] where seq_stride = MAX_SEQ_LEN

kernel void attn_scores_batched(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       scores     [[buffer(2)]],  // [num_heads, seq_stride]
    constant uint&      head_dim   [[buffer(3)]],  // 256
    constant uint&      kv_dim     [[buffer(4)]],  // 512
    constant uint&      seq_len    [[buffer(5)]],  // current seq length
    constant uint&      seq_stride [[buffer(6)]],  // MAX_SEQ_LEN
    constant float&     scale      [[buffer(7)]],  // 1/sqrt(head_dim)
    constant uint&      heads_per_kv [[buffer(8)]], // 16 (GQA ratio)
    constant uint&      num_seq_tgs  [[buffer(9)]],  // = seq_len
    uint tgid  [[threadgroup_position_in_grid]],    // linearized: pos + h * num_seq_tgs
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    device const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

    float acc = 0.0f;
    for (uint d = lid; d < head_dim; d += tg_size) {
        acc += qh[d] * kp[d];
    }

    // SIMD reduction
    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}


// ============================================================================
// Kernel 7: Batched softmax — one threadgroup per head
// ============================================================================

kernel void attn_softmax_batched(
    device float*    scores     [[buffer(0)]],  // [num_heads, seq_stride]
    constant uint&   seq_len    [[buffer(1)]],
    constant uint&   seq_stride [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],     // head index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    device float* s = scores + tgid * seq_stride;

    // Pass 1: find max
    threadgroup float shared_max[32];
    float local_max = -1e30f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        local_max = max(local_max, s[i]);
    }
    float sm = simd_max(local_max);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared_max[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_max = simd_max(shared_max[simd_lane]);
    }
    threadgroup float broadcast_max;
    if (lid == 0) broadcast_max = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = broadcast_max;

    // Pass 2: exp and sum
    threadgroup float shared_sum[32];
    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        float val = exp(s[i] - global_max);
        s[i] = val;
        local_sum += val;
    }
    float simd_s = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_group] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_sum = simd_sum(shared_sum[simd_lane]);
    }
    threadgroup float broadcast_sum;
    if (lid == 0) broadcast_sum = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = broadcast_sum;

    // Pass 3: normalize
    float inv_sum = 1.0f / global_sum;
    for (uint i = lid; i < seq_len; i += tg_size) {
        s[i] *= inv_sum;
    }
}


// ============================================================================
// Kernel 8: Batched attention value aggregation (scores @ V) — all heads
// ============================================================================
//
// For each head h: output[h*head_dim + d] = sum_p(scores[h*seq_stride+p] * V[p*kv_dim + kv_h*head_dim + d])
//
// Grid: linearized over (head_dim * num_heads) — one thread per (dimension, head).

kernel void attn_values_batched(
    device const float* scores   [[buffer(0)]],  // [num_heads, seq_stride]
    device const float* V_cache  [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       out      [[buffer(2)]],  // [num_heads, head_dim]
    constant uint&      head_dim  [[buffer(3)]],  // 256
    constant uint&      kv_dim    [[buffer(4)]],  // 512
    constant uint&      seq_len   [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant uint&      heads_per_kv [[buffer(7)]],
    uint tid [[thread_position_in_grid]]          // linearized: d + h * head_dim
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint p = 0; p < seq_len; p++) {
        acc += s[p] * V_cache[p * kv_dim + kv_h * head_dim + d];
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Kernel 9: Sigmoid element-wise gate
// ============================================================================
// out[i] = x[i] * sigmoid(gate[i])

kernel void sigmoid_gate(
    device float*       x_out  [[buffer(0)]],  // [dim] in/out
    device const float* gate   [[buffer(1)]],  // [dim] gate values
    constant uint&      dim    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = 1.0f / (1.0f + exp(-gate[tid]));
    x_out[tid] = x_out[tid] * g;
}


// ============================================================================
// Kernel 10: GatedDeltaNet linear attention step (single token, all heads)
// ============================================================================
//
// Implements the GatedDeltaNet recurrence for autoregressive generation:
//   1. State decay:  S[vi][ki] *= g_decay
//   2. Memory read:  kv_mem[vi] = sum_ki(S[vi][ki] * k[ki])
//   3. Delta:        delta[vi] = (v[vi] - kv_mem[vi]) * beta_gate
//   4. State update: S[vi][ki] += k[ki] * delta[vi]
//   5. Output:       out[vi] = sum_ki(S[vi][ki] * q[ki])
//
// Dispatch: 64 threadgroups (one per v-head), 128 threads each (one per vi).
// Each thread owns one row S[head_id][vi][:] of the 128x128 state matrix.
//
// State layout: [64 * 128 * 128] float = 4MB total, persisted across tokens.
// k-head sharing: 4 v-heads share 1 k-head (64 v-heads / 16 k-heads).

kernel void gated_delta_net_step(
    device float *state,             // [64 * 128 * 128] persistent state
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &k_heads_per_v,    // = 4
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Step 3+4: Delta update — S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < 128; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output — out[vi] = dot(S[vi][:], q[:])
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        out_val += state[state_base + ki] * q[k_base + ki];
    }
    output[v_base + vi] = out_val;
}


// ============================================================================
// Kernel 11: Conv1d depthwise step (single token, incremental inference)
// ============================================================================
//
// Depthwise 1D convolution for one new input token:
//   output[c] = sum_k(history[k][c] * weight[c][k]) + input[c] * weight[c][3]
//   then SiLU activation: output[c] = output[c] / (1 + exp(-output[c]))
//
// After computing, shifts the history buffer left and appends the new input.
//
// Weight layout: [channels * kernel_size] bf16, weight[c * kernel_size + k]
// Conv state layout: [(kernel_size-1) * channels] row-major, state[k * channels + c]
// kernel_size = 4 (hardcoded), so 3 history slots + 1 new input.
//
// Dispatch: conv_dim threads (12288), one per channel.

kernel void conv1d_step(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    device const float *input,        // [conv_dim] current input
    device const uint16_t *weights,   // [conv_dim * 4] bf16 as uint16
    device float *output,             // [conv_dim] convolution output
    constant uint &conv_dim,          // = 12288
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    // Convolution: dot product of history + new input with weights
    // weight layout: weight[c * 4 + k] for channel c, position k
    uint w_base = idx * 4;
    float acc = 0.0f;

    // 3 history slots (k=0,1,2)
    acc += conv_state[0 * conv_dim + idx] * bf16_to_f32(weights[w_base + 0]);
    acc += conv_state[1 * conv_dim + idx] * bf16_to_f32(weights[w_base + 1]);
    acc += conv_state[2 * conv_dim + idx] * bf16_to_f32(weights[w_base + 2]);

    // New input (k=3)
    float inp = input[idx];
    acc += inp * bf16_to_f32(weights[w_base + 3]);

    // SiLU activation
    output[idx] = acc / (1.0f + exp(-acc));

    // Shift history: move slots 1,2 -> 0,1, append input at slot 2
    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = inp;
}


// ============================================================================
// Kernel 12: Per-head RMS normalize for q and k vectors
// ============================================================================
// q: [num_k_heads * key_dim], k: [num_k_heads * key_dim]
// Normalize each head independently, then scale by 1/sqrt(key_dim)^2 for q, 1/sqrt(key_dim) for k
// Dispatch: num_k_heads threadgroups, key_dim threads each

kernel void rms_norm_qk(
    device float *q,              // [num_k_heads * key_dim] in/out
    device float *k,              // [num_k_heads * key_dim] in/out
    constant uint &key_dim,       // = 128
    constant float &inv_scale,    // = 1/sqrt(key_dim)
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * key_dim;

    // RMS norm for q
    threadgroup float q_sum_sq;
    if (tid == 0) q_sum_sq = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qval = (tid < key_dim) ? q[base + tid] : 0;
    // Use threadgroup atomic add for sum of squares
    float q_sq_local = qval * qval;
    // Simple reduction: thread 0 accumulates (key_dim=128, fits in one pass)
    threadgroup float q_partial[128];
    q_partial[tid] = q_sq_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += q_partial[i];
        q_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_inv_rms = rsqrt(q_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;  // q gets extra scale
    }

    // RMS norm for k
    threadgroup float k_sum_sq;
    float kval = (tid < key_dim) ? k[base + tid] : 0;
    threadgroup float k_partial[128];
    k_partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += k_partial[i];
        k_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_inv_rms = rsqrt(k_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}


// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
// Per v-head: g_decay = exp(-A * softplus(alpha + dt_bias)), beta_gate = sigmoid(beta)
// Dispatch: num_v_heads threads (64)

kernel void compute_decay_beta(
    device const float *alpha_out,   // [num_v_heads] from projection
    device const float *beta_out,    // [num_v_heads] from projection
    device const float *A_log,       // [num_v_heads] log of decay base (persistent)
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay,           // [num_v_heads] output
    device float *beta_gate,         // [num_v_heads] output
    uint idx [[thread_position_in_grid]]
) {
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = exp(A_log[idx]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}


// ============================================================================
// Kernel 14: Gated RMS norm (z-gated output normalization)
// ============================================================================
// output[i] = rms_norm(values[i]) * SiLU(z[i]) * weight[i]
// Per v-head: normalize values, gate with z, scale with weight
// Dispatch: num_v_heads threadgroups, value_dim threads each

kernel void gated_rms_norm(
    device const float *values,       // [num_v_heads * value_dim] delta-net output
    device const float *z,            // [num_v_heads * value_dim] gate values
    device const uint16_t *weight,    // [value_dim] bf16 norm weights (shared across heads)
    device float *output,             // [num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant float &eps,              // = 1e-6
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * value_dim;

    float val = (tid < value_dim) ? values[base + tid] : 0;

    // RMS norm reduction
    threadgroup float partial[128];
    partial[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < value_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(value_dim) + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));  // SiLU
        float w = bf16_to_f32(weight[tid]);
        output[base + tid] = normed * gate * w;
    }
}


// ============================================================================
// Batched linear-attention prefill kernels
// ============================================================================

kernel void conv1d_step_batched(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] persistent state
    device const float *input,        // [batch_n * conv_dim]
    device const uint16_t *weights,   // [conv_dim * 4] bf16
    device float *output,             // [batch_n * conv_dim]
    constant uint &conv_dim,          // = 12288
    constant uint &batch_n,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    uint w_base = idx * 4;
    float w0 = bf16_to_f32(weights[w_base + 0]);
    float w1 = bf16_to_f32(weights[w_base + 1]);
    float w2 = bf16_to_f32(weights[w_base + 2]);
    float w3 = bf16_to_f32(weights[w_base + 3]);

    float s0 = conv_state[0 * conv_dim + idx];
    float s1 = conv_state[1 * conv_dim + idx];
    float s2 = conv_state[2 * conv_dim + idx];

    for (uint t = 0; t < batch_n; t++) {
        uint off = t * conv_dim + idx;
        float inp = input[off];
        float acc = s0 * w0 + s1 * w1 + s2 * w2 + inp * w3;
        output[off] = acc / (1.0f + exp(-acc));
        s0 = s1;
        s1 = s2;
        s2 = inp;
    }

    conv_state[0 * conv_dim + idx] = s0;
    conv_state[1 * conv_dim + idx] = s1;
    conv_state[2 * conv_dim + idx] = s2;
}

kernel void rms_norm_qk_batched(
    device float *q,              // [batch_n * token_stride] q at offset 0
    device float *k,              // [batch_n * token_stride] k at bound offset
    constant uint &key_dim,       // = 128
    constant float &inv_scale,    // = 1/sqrt(key_dim)
    constant uint &token_stride,  // = 12288
    constant uint &batch_n,
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tp [[thread_position_in_threadgroup]]
) {
    uint head = tg.x;
    uint token = tg.y;
    uint tid = tp.x;
    if (token >= batch_n || tid >= key_dim) return;

    uint q_base = token * token_stride + head * key_dim;
    uint k_base = token * token_stride + head * key_dim;

    threadgroup float partial[128];

    float qval = q[q_base + tid];
    partial[tid] = qval * qval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < key_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_inv_rms = rsqrt(partial[0] / float(key_dim) + 1e-6f);
    q[q_base + tid] = qval * q_inv_rms * inv_scale * inv_scale;

    float kval = k[k_base + tid];
    partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < key_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_inv_rms = rsqrt(partial[0] / float(key_dim) + 1e-6f);
    k[k_base + tid] = kval * k_inv_rms * inv_scale;
}

kernel void compute_decay_beta_batched(
    device const float *alpha_out,   // [batch_n * num_v_heads]
    device const float *beta_out,    // [batch_n * num_v_heads]
    device const float *A_log,       // [num_v_heads]
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay,           // [batch_n * num_v_heads]
    device float *beta_gate,         // [batch_n * num_v_heads]
    constant uint &batch_n,
    constant uint &num_v_heads,
    uint idx [[thread_position_in_grid]]
) {
    uint total = batch_n * num_v_heads;
    if (idx >= total) return;

    uint vh = idx % num_v_heads;
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[vh]);
    float A_val = exp(A_log[vh]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}

kernel void gated_delta_net_step_batched(
    device float *state,             // [num_v_heads * key_dim * key_dim] persistent state
    device const float *q,           // bound at q base, token stride = conv_dim
    device const float *k,           // bound at k base, token stride = conv_dim
    device const float *v,           // bound at v base, token stride = conv_dim
    device const float *g_decay,     // [batch_n * num_v_heads]
    device const float *beta_gate,   // [batch_n * num_v_heads]
    device float *output,            // [batch_n * output_stride]
    constant uint &k_heads_per_v,    // = num_k_heads / num_v_heads
    constant uint &token_stride,     // = conv_dim (total QKV projection dim)
    constant uint &output_stride,    // = num_v_heads * value_dim
    constant uint &batch_n,
    constant uint &num_v_heads,
    constant uint &key_dim,
    constant uint &value_dim,
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    uint state_base = head_id * key_dim * key_dim + vi * key_dim;
    uint k_base = kh * key_dim;
    uint v_base = head_id * value_dim;

    for (uint t = 0; t < batch_n; t++) {
        float g = g_decay[t * num_v_heads + head_id];
        float beta = beta_gate[t * num_v_heads + head_id];

        device const float *q_t = q + t * token_stride + k_base;
        device const float *k_t = k + t * token_stride + k_base;
        device const float *v_t = v + t * token_stride + v_base;

        float kv_mem = 0.0f;
        for (uint ki = 0; ki < key_dim; ki++) {
            float s = state[state_base + ki] * g;
            state[state_base + ki] = s;
            kv_mem += s * k_t[ki];
        }

        float delta = (v_t[vi] - kv_mem) * beta;
        for (uint ki = 0; ki < key_dim; ki++) {
            state[state_base + ki] += k_t[ki] * delta;
        }

        float out_val = 0.0f;
        for (uint ki = 0; ki < key_dim; ki++) {
            out_val += state[state_base + ki] * q_t[ki];
        }
        output[t * output_stride + v_base + vi] = out_val;
    }
}

kernel void gated_rms_norm_batched(
    device const float *values,       // [batch_n * num_v_heads * value_dim]
    device const float *z,            // [batch_n * num_v_heads * value_dim]
    device const uint16_t *weight,    // [value_dim] bf16
    device float *output,             // [batch_n * num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant uint &output_stride,     // = 8192
    constant float &eps,              // = 1e-6
    constant uint &batch_n,
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tp [[thread_position_in_threadgroup]]
) {
    uint head = tg.x;
    uint token = tg.y;
    uint tid = tp.x;
    if (token >= batch_n || tid >= value_dim) return;

    uint base = token * output_stride + head * value_dim;
    float val = values[base + tid];

    threadgroup float partial[128];
    partial[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < value_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(partial[0] / float(value_dim) + eps);
    float zval = z[base + tid];
    float gate = zval / (1.0f + exp(-zval));
    float w = bf16_to_f32(weight[tid]);
    output[base + tid] = val * inv_rms * gate * w;
}


// ============================================================================
// Kernel 12: MoE combine + residual + shared expert gate (fused)
// ============================================================================
// Fused operation for CMD3 GPU-side combine:
//   hidden[i] = h_mid[i] + sum_k(expert_weight[k] * expert_out[k][i])
//               + sigmoid(shared_gate_score) * shared_out[i]
//
// All 8 expert output buffers are always bound (unused ones have weight=0).
// This avoids variable buffer bindings and keeps the dispatch simple.
//
// Dispatch: (dim + 255) / 256 threadgroups, 256 threads each.

kernel void moe_combine_residual(
    device const float* h_mid       [[buffer(0)]],   // [dim]
    device const float* shared_out  [[buffer(1)]],   // [dim]
    device float*       hidden_out  [[buffer(2)]],   // [dim] output
    device const float* expert_out0 [[buffer(3)]],   // [dim] expert 0
    device const float* expert_out1 [[buffer(4)]],   // [dim] expert 1
    device const float* expert_out2 [[buffer(5)]],   // [dim] expert 2
    device const float* expert_out3 [[buffer(6)]],   // [dim] expert 3
    device const float* expert_out4 [[buffer(7)]],   // [dim] expert 4
    device const float* expert_out5 [[buffer(8)]],   // [dim] expert 5
    device const float* expert_out6 [[buffer(9)]],   // [dim] expert 6
    device const float* expert_out7 [[buffer(10)]],  // [dim] expert 7
    device const float* params      [[buffer(11)]],  // [10]: weights[0..7], shared_gate_score, (unused)
    constant uint&      dim         [[buffer(12)]],
    constant uint&      K           [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    // Read expert weights and shared gate from params buffer
    float shared_gate = 1.0f / (1.0f + exp(-params[8]));  // sigmoid(shared_gate_score)

    // Weighted sum of expert outputs
    float moe = 0.0f;
    // Unrolled for MAX_K=8 with branch on K to avoid reading invalid buffers
    if (K > 0) moe += params[0] * expert_out0[tid];
    if (K > 1) moe += params[1] * expert_out1[tid];
    if (K > 2) moe += params[2] * expert_out2[tid];
    if (K > 3) moe += params[3] * expert_out3[tid];
    if (K > 4) moe += params[4] * expert_out4[tid];
    if (K > 5) moe += params[5] * expert_out5[tid];
    if (K > 6) moe += params[6] * expert_out6[tid];
    if (K > 7) moe += params[7] * expert_out7[tid];

    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}

// ============================================================================
// Prefill: Full-attention Q deinterleave + RMS norm + RoPE
// ============================================================================
// Grid: [num_heads, batch_n, 1], 256 threads each.

kernel void prefill_q_rope_norm_bf16(
    device const float*    q_proj     [[buffer(0)]],  // [N, num_heads, 2 * head_dim]
    device const uint16_t* q_norm_w   [[buffer(1)]],  // [head_dim] bf16
    device float*          q_out      [[buffer(2)]],  // [N, num_heads, head_dim]
    device float*          gate_out   [[buffer(3)]],  // [N, num_heads, head_dim]
    constant uint&         head_dim   [[buffer(4)]],
    constant uint&         num_heads  [[buffer(5)]],
    constant uint&         rotary_dim [[buffer(6)]],
    constant uint&         pos_base   [[buffer(7)]],
    constant uint&         batch_n    [[buffer(8)]],
    constant float&        eps        [[buffer(9)]],
    constant float&        rope_theta [[buffer(10)]],
    constant uint&         has_q_norm [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tp [[thread_position_in_threadgroup]]
) {
    uint lid = tp.x;
    uint head = tg.x;
    uint token = tg.y;
    if (head >= num_heads || token >= batch_n || lid >= head_dim) return;

    uint q_proj_stride = num_heads * head_dim * 2;
    uint q_out_stride = num_heads * head_dim;
    uint proj_base = token * q_proj_stride + head * 2 * head_dim;
    uint out_base = token * q_out_stride + head * head_dim;

    float q_val = q_proj[proj_base + lid];
    float gate_val = q_proj[proj_base + head_dim + lid];

    threadgroup float partial[256];
    threadgroup float q_shared[256];

    if (has_q_norm) {
        partial[lid] = q_val * q_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float sum_sq = 0.0f;
            for (uint i = 0; i < head_dim; i++) sum_sq += partial[i];
            partial[0] = sum_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        q_val = q_val * rsqrt(partial[0] / float(head_dim) + eps) * bf16_to_f32(q_norm_w[lid]);
    }

    q_shared[lid] = q_val;
    gate_out[out_base + lid] = gate_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint rotary_half = rotary_dim / 2;
    if (lid < rotary_half) {
        float freq = 1.0f / pow(rope_theta, (2.0f * float(lid)) / float(rotary_dim));
        float angle = float(pos_base + token) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        float q0 = q_shared[lid];
        float q1 = q_shared[lid + rotary_half];
        q_out[out_base + lid] = q0 * cos_a - q1 * sin_a;
        q_out[out_base + lid + rotary_half] = q0 * sin_a + q1 * cos_a;
    } else if (lid >= rotary_dim) {
        q_out[out_base + lid] = q_shared[lid];
    }
}

// ============================================================================
// Prefill: Full-attention K RMS norm + RoPE + KV cache write
// ============================================================================
// Grid: [num_kv_heads, batch_n, 1], 256 threads each.

kernel void prefill_kv_cache_bf16(
    device const float*    k_proj     [[buffer(0)]],  // [N, num_kv_heads, head_dim]
    device const float*    v_proj     [[buffer(1)]],  // [N, num_kv_heads, head_dim]
    device const uint16_t* k_norm_w   [[buffer(2)]],  // [head_dim] bf16
    device float*          k_cache    [[buffer(3)]],  // [max_seq, kv_dim]
    device float*          v_cache    [[buffer(4)]],  // [max_seq, kv_dim]
    constant uint&         head_dim   [[buffer(5)]],
    constant uint&         kv_dim     [[buffer(6)]],
    constant uint&         rotary_dim [[buffer(7)]],
    constant uint&         pos_base   [[buffer(8)]],
    constant uint&         cache_start [[buffer(9)]],
    constant uint&         batch_n    [[buffer(10)]],
    constant float&        eps        [[buffer(11)]],
    constant float&        rope_theta [[buffer(12)]],
    constant uint&         has_k_norm [[buffer(13)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tp [[thread_position_in_threadgroup]]
) {
    uint lid = tp.x;
    uint head = tg.x;
    uint token = tg.y;
    if (token >= batch_n || lid >= head_dim) return;

    uint proj_base = token * kv_dim + head * head_dim;
    uint cache_base = (cache_start + token) * kv_dim + head * head_dim;

    float k_val = k_proj[proj_base + lid];
    float v_val = v_proj[proj_base + lid];

    threadgroup float partial[256];
    threadgroup float k_shared[256];

    if (has_k_norm) {
        partial[lid] = k_val * k_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float sum_sq = 0.0f;
            for (uint i = 0; i < head_dim; i++) sum_sq += partial[i];
            partial[0] = sum_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_val = k_val * rsqrt(partial[0] / float(head_dim) + eps) * bf16_to_f32(k_norm_w[lid]);
    }

    k_shared[lid] = k_val;
    v_cache[cache_base + lid] = v_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint rotary_half = rotary_dim / 2;
    if (lid < rotary_half) {
        float freq = 1.0f / pow(rope_theta, (2.0f * float(lid)) / float(rotary_dim));
        float angle = float(pos_base + token) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        float k0 = k_shared[lid];
        float k1 = k_shared[lid + rotary_half];
        k_cache[cache_base + lid] = k0 * cos_a - k1 * sin_a;
        k_cache[cache_base + lid + rotary_half] = k0 * sin_a + k1 * cos_a;
    } else if (lid >= rotary_dim) {
        k_cache[cache_base + lid] = k_shared[lid];
    }
}

// ============================================================================
// Prefill: Batched causal attention with online softmax (Flash Attention style)
// ============================================================================
// Processes N query tokens against a KV cache with causal masking.
// Query t attends to KV positions [0, cache_start + t].
//
// One threadgroup per (query, head) pair. Thread lid handles dimension d=lid.
// Uses online softmax: streaming max/sum update avoids storing all scores.
//
// Grid: N * num_heads threadgroups, 256 threads each (= head_dim)
// Requires: head_dim == tg_size (both 256)

kernel void prefill_causal_attn(
    device const float* Q          [[buffer(0)]],   // [N, num_heads, head_dim]
    device const float* gate       [[buffer(1)]],   // [N, num_heads, head_dim]
    device const float* K_cache    [[buffer(2)]],   // [max_seq, kv_dim]
    device const float* V_cache    [[buffer(3)]],   // [max_seq, kv_dim]
    device float*       out        [[buffer(4)]],   // [N, num_heads, head_dim]
    constant uint&      head_dim   [[buffer(5)]],   // 256
    constant uint&      kv_dim     [[buffer(6)]],   // num_kv_heads * head_dim
    constant uint&      num_heads  [[buffer(7)]],   // 32
    constant uint&      heads_per_kv [[buffer(8)]], // 16 (GQA ratio)
    constant float&     scale      [[buffer(9)]],   // 1/sqrt(head_dim)
    constant uint&      cache_start [[buffer(10)]],  // KV position of first query
    constant uint&      batch_n    [[buffer(11)]],   // number of query tokens
    uint tgid  [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]]
) {
    uint t = tgid / num_heads;
    uint h = tgid % num_heads;
    if (t >= batch_n) return;

    uint seq_len = cache_start + t + 1;
    uint kv_h = h / heads_per_kv;
    uint d = lid;  // this thread handles dimension d

    // Load Q[t, h] into shared memory for cooperative dot product
    threadgroup float q_shared[256];
    uint q_idx = t * num_heads * head_dim + h * head_dim + d;
    q_shared[d] = Q[q_idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Shared memory for score reduction and softmax state broadcast
    threadgroup float score_parts[8];   // partial sums from 8 SIMD groups
    threadgroup float score_val;        // final score (broadcast)
    threadgroup float corr_val;         // correction factor (broadcast)
    threadgroup float w_val;            // weight for new V (broadcast)
    threadgroup float attn_max;         // running max
    threadgroup float attn_sum;         // running sum

    if (d == 0) { attn_max = -1e30f; attn_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    float acc = 0.0f;  // per-thread V accumulator for dimension d

    for (uint p = 0; p < seq_len; p++) {
        // ---- Cooperative dot product: Q[t,h] . K[p] ----
        float k_d = K_cache[p * kv_dim + kv_h * head_dim + d];
        float partial = q_shared[d] * k_d;
        float simd_val = simd_sum(partial);
        if (simd_lane == 0) score_parts[simd_group] = simd_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 reduces and updates online softmax
        if (d == 0) {
            float s = 0;
            for (uint i = 0; i < 8; i++) s += score_parts[i];
            s *= scale;
            score_val = s;

            float old_max = attn_max;
            float new_max = max(old_max, s);
            float corr = exp(old_max - new_max);
            float w = exp(s - new_max);
            attn_sum = attn_sum * corr + w;
            attn_max = new_max;
            corr_val = corr;
            w_val = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- All threads: update accumulator ----
        float v_d = V_cache[p * kv_dim + kv_h * head_dim + d];
        acc = acc * corr_val + v_d * w_val;
    }

    // ---- Normalize and apply sigmoid gate ----
    float inv_sum = 1.0f / attn_sum;
    float gate_d = gate[q_idx];
    float sigmoid_g = 1.0f / (1.0f + exp(-gate_d));

    out[q_idx] = acc * inv_sum * sigmoid_g;
}


// ============================================================================
// Prefill: Batched RMS norm for N tokens
// ============================================================================
// Input:  [N * dim] floats (N tokens × dim)
// Weight: [dim] bf16 (shared across all tokens)
// Output: [N * dim] floats
// Grid: N threadgroups, 256 threads each

kernel void prefill_rms_norm_bf16(
    device const float*    x       [[buffer(0)]],  // [N * dim]
    device const uint16_t* weight  [[buffer(1)]],  // [dim] bf16
    device float*          out     [[buffer(2)]],  // [N * dim]
    constant uint&         dim     [[buffer(3)]],  // 4096
    constant float&        eps     [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],     // token index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    device const float* x_t = x + tgid * dim;
    device float* out_t = out + tgid * dim;

    // Sum of squares reduction
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float v = x_t[i];
        acc += v * v;
    }

    threadgroup float shared[8];
    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float sum_sq_broadcast;
    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) sum_sq_broadcast = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = rsqrt(sum_sq_broadcast / float(dim) + eps);
    for (uint i = lid; i < dim; i += tg_size) {
        out_t[i] = x_t[i] * rms * bf16_to_f32(weight[i]);
    }
}


// ============================================================================
// Prefill: Batched residual add + RMS norm
// ============================================================================
// h_mid[t] = residual[t] + oproj[t], then norm h_mid → h_post
// Grid: N threadgroups, 256 threads each

kernel void prefill_residual_norm_bf16(
    device const float*    residual [[buffer(0)]],  // [N * dim]
    device const float*    oproj    [[buffer(1)]],  // [N * dim]
    device float*          h_mid    [[buffer(2)]],  // [N * dim] output: residual + oproj
    device float*          h_post   [[buffer(3)]],  // [N * dim] output: rms_norm(h_mid)
    device const uint16_t* weight   [[buffer(4)]],  // [dim] bf16
    constant uint&         dim      [[buffer(5)]],
    constant float&        eps      [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint off = tgid * dim;

    // Residual add
    for (uint i = lid; i < dim; i += tg_size) {
        h_mid[off + i] = residual[off + i] + oproj[off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RMS norm
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float v = h_mid[off + i];
        acc += v * v;
    }

    threadgroup float shared[8];
    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float sum_sq_broadcast;
    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) sum_sq_broadcast = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = rsqrt(sum_sq_broadcast / float(dim) + eps);
    for (uint i = lid; i < dim; i += tg_size) {
        h_post[off + i] = h_mid[off + i] * rms * bf16_to_f32(weight[i]);
    }
}


// ============================================================================
// Prefill: Batched SwiGLU activation
// ============================================================================
// out[i] = silu(gate[i]) * up[i] for N * dim elements
// Grid: total elements / 256

kernel void prefill_swiglu(
    device float*       gate_and_out [[buffer(0)]],  // [N * dim] — gate in, activation out
    device const float* up           [[buffer(1)]],  // [N * dim]
    constant uint&      total        [[buffer(2)]],  // N * dim
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total) return;
    float g = gate_and_out[tid];
    float silu_g = g / (1.0f + exp(-g));
    gate_and_out[tid] = silu_g * up[tid];
}


// ============================================================================
// Prefill: Batched combine (hidden = h_mid + sigmoid(gate_score) * shared_out)
// ============================================================================
// Grid: N threadgroups, 256 threads each

kernel void prefill_combine(
    device const float* h_mid       [[buffer(0)]],  // [N * dim]
    device const float* shared_out  [[buffer(1)]],  // [N * dim]
    device const float* gate_scores [[buffer(2)]],  // [N] — one per token
    device float*       hidden_out  [[buffer(3)]],  // [N * dim]
    constant uint&      dim         [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],      // token index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    float gs = 1.0f / (1.0f + exp(-gate_scores[tgid]));
    uint off = tgid * dim;
    for (uint i = lid; i < dim; i += tg_size) {
        hidden_out[off + i] = h_mid[off + i] + gs * shared_out[off + i];
    }
}
