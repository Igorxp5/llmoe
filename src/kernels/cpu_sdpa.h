#ifndef KERNELS_CPU_SDPA_H
#define KERNELS_CPU_SDPA_H

#include <math.h>
#include <stdlib.h>

#include "kernels/kernels.h"

static inline void sdpa_one_head(float *out, const float *q,
                                 const float *k, const float *v,
                                 size_t seq_len, size_t n_heads,
                                 size_t head_dim, float scale, float *scores)
{
    size_t stride = n_heads * head_dim;
    for (size_t i = 0; i < seq_len; ++i) {
        const float *qi = q + i * stride;
        float mx = -INFINITY;
        for (size_t j = 0; j <= i; ++j) {
            float dot = 0.0f;
            const float *kj = k + j * stride;
            for (size_t d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
            scores[j] = dot * scale;
            if (scores[j] > mx) mx = scores[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j <= i; ++j) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        float *oi = out + i * stride;
        for (size_t d = 0; d < head_dim; ++d) oi[d] = 0.0f;
        for (size_t j = 0; j <= i; ++j) {
            float w = scores[j] / sum;
            const float *vj = v + j * stride;
            for (size_t d = 0; d < head_dim; ++d) oi[d] += w * vj[d];
        }
    }
}

static inline void cpu_sdpa(float *out, const float *q,
                            const float *k, const float *v,
                            size_t seq_len, size_t n_heads, size_t head_dim,
                            float scale)
{
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < n_heads; ++h) {
        float *scores = (float *)malloc(seq_len * sizeof(float));
        size_t off = h * head_dim;
        sdpa_one_head(out + off, q + off, k + off, v + off, seq_len, n_heads,
                      head_dim, scale, scores);
        free(scores);
    }
}

/* Incremental SDPA: compute attention for `new_seq` tokens at absolute
 * position `cache_pos` against a pre-populated KV cache.
 *
 * The cache already contains `cache_pos + new_seq` tokens (the caller
 * appended the new K/V before calling this function).  Each new token i
 * at absolute position p = cache_pos + i attends to cached positions
 * 0 .. p (causal mask). */
static inline void cpu_sdpa_incremental(float *out,
    const float *q_new,
    const float *cache_k,
    const float *cache_v,
    size_t new_seq,
    size_t cache_pos,
    size_t n_heads, size_t head_dim,
    float scale)
{
    size_t stride = n_heads * head_dim;
    size_t total_len = cache_pos + new_seq;
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < n_heads; ++h) {
        float *scores = (float *)malloc(total_len * sizeof(float));
        size_t off = h * head_dim;
        /* Q, cache_k, cache_v already offset by h*head_dim. */
        const float *qh = q_new + off;
        const float *kh = cache_k + off;
        const float *vh = cache_v + off;
        for (size_t i = 0; i < new_seq; ++i) {
            size_t abs = cache_pos + i;
            const float *qi = qh + i * stride;
            float mx = -INFINITY;
            for (size_t j = 0; j <= abs; ++j) {
                float dot = 0.0f;
                const float *kj = kh + j * stride;
                for (size_t d = 0; d < head_dim; ++d)
                    dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > mx) mx = scores[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j <= abs; ++j) {
                scores[j] = expf(scores[j] - mx);
                sum += scores[j];
            }
            float *oi = out + off + i * stride;
            for (size_t d = 0; d < head_dim; ++d) oi[d] = 0.0f;
            for (size_t j = 0; j <= abs; ++j) {
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
