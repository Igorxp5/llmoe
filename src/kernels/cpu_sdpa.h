#ifndef KERNELS_CPU_SDPA_H
#define KERNELS_CPU_SDPA_H

#include <math.h>
#include <stdlib.h>

#include "kernels/kernels.h"

/* Scaled dot-product attention for a single head with causal masking.
 * Q/K/V layout: [num_tokens, num_heads, head_dim]; `stride` spans one token.
 * `scores_tmp` is a scratch buffer of length num_tokens. */
static inline void sdpa_one_head(float * restrict output,
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
            float dot = 0.0f;
            const float *kj = key + j * stride;
            for (size_t d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
            scores_tmp[j] = dot * scale;
            if (scores_tmp[j] > mx) mx = scores_tmp[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j <= i; ++j) {
            scores_tmp[j] = expf(scores_tmp[j] - mx);
            sum += scores_tmp[j];
        }
        float *oi = output + i * stride;
        for (size_t d = 0; d < head_dim; ++d) oi[d] = 0.0f;
        for (size_t j = 0; j <= i; ++j) {
            float w = scores_tmp[j] / sum;
            const float *vj = value + j * stride;
            for (size_t d = 0; d < head_dim; ++d) oi[d] += w * vj[d];
        }
    }
}

/* Batched SDPA over all heads. Each head allocates its own scores scratch
 * buffer internally. Q/K/V layout: [num_tokens, num_heads, head_dim]. */
static inline void cpu_sdpa(float * restrict output,
                            const float * restrict query,
                            const float * restrict key,
                            const float * restrict value,
                            size_t num_tokens, size_t num_heads,
                            size_t head_dim, float scale)
{
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < num_heads; ++h) {
        float *scores = (float *)malloc(num_tokens * sizeof(float));
        size_t off = h * head_dim;
        sdpa_one_head(output + off, query + off, key + off, value + off,
                      num_tokens, num_heads, head_dim, scale, scores);
        free(scores);
    }
}

/* Incremental SDPA: compute attention for `new_tokens` at absolute position
 * `cache_position` against a pre-populated KV cache.
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
    float scale)
{
    size_t stride = num_heads * head_dim;
    size_t total_len = cache_position + new_tokens;
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < num_heads; ++h) {
        float *scores = (float *)malloc(total_len * sizeof(float));
        size_t off = h * head_dim;
        const float *qh = query_new + off;
        const float *kh = cache_key + off;
        const float *vh = cache_value + off;
        for (size_t i = 0; i < new_tokens; ++i) {
            size_t abs_pos = cache_position + i;
            const float *qi = qh + i * stride;
            float mx = -INFINITY;
            for (size_t j = 0; j <= abs_pos; ++j) {
                float dot = 0.0f;
                const float *kj = kh + j * stride;
                for (size_t d = 0; d < head_dim; ++d)
                    dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > mx) mx = scores[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j <= abs_pos; ++j) {
                scores[j] = expf(scores[j] - mx);
                sum += scores[j];
            }
            float *oi = output + off + i * stride;
            for (size_t d = 0; d < head_dim; ++d) oi[d] = 0.0f;
            for (size_t j = 0; j <= abs_pos; ++j) {
                float w = scores[j] / sum;
                const float *vj = vh + j * stride;
                for (size_t d = 0; d < head_dim; ++d)
                    oi[d] += w * vj[d];
            }
        }
        free(scores);
    }
}

#endif /* KERNELS_CPU_SDPA_H */
