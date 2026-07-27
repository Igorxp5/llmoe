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
    float *scores = (float *)malloc(seq_len * sizeof(float));
    for (size_t h = 0; h < n_heads; ++h) {
        size_t off = h * head_dim;
        sdpa_one_head(out + off, q + off, k + off, v + off, seq_len, n_heads,
                      head_dim, scale, scores);
    }
    free(scores);
}

#endif /* KERNELS_CPU_SDPA_H */
