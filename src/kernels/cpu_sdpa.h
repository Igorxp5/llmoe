#ifndef KERNELS_CPU_SDPA_H
#define KERNELS_CPU_SDPA_H

#include <math.h>

#include "kernels/cpu_softmax.h"
#include "kernels/kernels.h"

/* Scaled dot-product attention for a single head with causal masking.
 * Q/K/V layout: [num_tokens, num_heads, head_dim]; `stride` spans one token.
 * `scores_tmp` is a scratch buffer of length num_tokens. */
static inline __attribute__((always_inline)) void sdpa_one_head(float * restrict output,
                                 const float * restrict query,
                                 const float * restrict key,
                                 const float * restrict value,
                                 size_t num_tokens, size_t num_heads,
                                 size_t head_dim, float scale,
                                 float * restrict scores_tmp)
{
    size_t stride = num_heads * head_dim;
    for (size_t i = 0; i < num_tokens; ++i) {
        const float *qi = query + i * stride;
        float mx = -INFINITY;
        for (size_t j = 0; j <= i; ++j) {
            const float *kj = key + j * stride;
            scores_tmp[j] = kernels_f32_dot(qi, kj, head_dim) * scale;
            if (scores_tmp[j] > mx) mx = scores_tmp[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j <= i; ++j) {
            scores_tmp[j] = expf(scores_tmp[j] - mx);
            sum += scores_tmp[j];
        }
        cpu_softmax_row_norm(scores_tmp, i + 1, sum);
        float *oi = output + i * stride;
        kernels_f32_zero(oi, head_dim);
        for (size_t j = 0; j <= i; ++j) {
            const float *vj = value + j * stride;
            kernels_f32_axpy(oi, vj, scores_tmp[j], head_dim);
        }
    }
}

/* Batched SDPA over all heads against caller-owned per-head score slices.
 * `scores` must hold num_heads * scores_stride floats; head h writes
 * scores[h*scores_stride .. h*scores_stride + num_tokens). Q/K/V layout:
 * [num_tokens, num_heads, head_dim]. */
static inline void cpu_sdpa(float * restrict output,
                            const float * restrict query,
                            const float * restrict key,
                            const float * restrict value,
                            size_t num_tokens, size_t num_heads,
                            size_t head_dim, float scale,
                            float * restrict scores, size_t scores_stride)
{
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < num_heads; ++h) {
        float *scores_tmp = scores + h * scores_stride;
        size_t off = h * head_dim;
        sdpa_one_head(output + off, query + off, key + off, value + off,
                      num_tokens, num_heads, head_dim, scale, scores_tmp);
    }
}

/* Incremental SDPA against caller-owned per-head score slices: compute
 * attention for `new_tokens` at absolute position `cache_position` against
 * a pre-populated KV cache. `scores` must hold num_heads * scores_stride
 * floats; head h writes scores[h*scores_stride .. h*scores_stride + total).
 *
 * The cache already contains `cache_position + new_tokens` tokens (the caller
 * appended the new K/V before calling this function).  Each new token i
 * at absolute position p = cache_position + i attends to cached positions
 * 0 .. p (causal mask). */
static inline void cpu_sdpa_incremental(float * restrict output,
    const float * restrict query_new,
    const float * restrict cache_key,
    const float * restrict cache_value,
    size_t new_tokens,
    size_t cache_position,
    size_t num_heads, size_t head_dim,
    float scale,
    float * restrict scores, size_t scores_stride)
{
    size_t stride = num_heads * head_dim;
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < num_heads; ++h) {
        float *scores_tmp = scores + h * scores_stride;
        size_t off = h * head_dim;
        const float *qh = query_new + off;
        const float *kh = cache_key + off;
        const float *vh = cache_value + off;
        for (size_t i = 0; i < new_tokens; ++i) {
            size_t abs_pos = cache_position + i;
            const float *qi = qh + i * stride;
            float mx = -INFINITY;
            for (size_t j = 0; j <= abs_pos; ++j) {
                const float *kj = kh + j * stride;
                scores_tmp[j] = kernels_f32_dot(qi, kj, head_dim) * scale;
                if (scores_tmp[j] > mx) mx = scores_tmp[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j <= abs_pos; ++j) {
                scores_tmp[j] = expf(scores_tmp[j] - mx);
                sum += scores_tmp[j];
            }
            cpu_softmax_row_norm(scores_tmp, abs_pos + 1, sum);
            float *oi = output + off + i * stride;
            kernels_f32_zero(oi, head_dim);
            for (size_t j = 0; j <= abs_pos; ++j) {
                const float *vj = vh + j * stride;
                kernels_f32_axpy(oi, vj, scores_tmp[j], head_dim);
            }
        }
    }
}

#endif /* KERNELS_CPU_SDPA_H */
