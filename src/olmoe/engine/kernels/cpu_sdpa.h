#ifndef OLMOE_KERNEL_CPU_SDPA_H
#define OLMOE_KERNEL_CPU_SDPA_H

#include <math.h>
#include <stdlib.h>

#include "olmoe/engine/engine_internal.h"

/* Causal MHA for one head. q/k/v point at this head's lane (head_dim stride
 * = n_heads*head_dim); out likewise. scores is a caller scratch of size
 * seq_len. scale is 1/sqrt(head_dim). softmax over j<=i, -inf for j>i. */
static inline void sdpa_one_head(olmoe_act_t *out, const olmoe_act_t *q,
                                 const olmoe_act_t *k, const olmoe_act_t *v,
                                 size_t seq_len, size_t n_heads,
                                 size_t head_dim, float scale, float *scores)
{
    size_t stride = n_heads * head_dim;
    for (size_t i = 0; i < seq_len; ++i) {
        const olmoe_act_t *qi = q + i * stride;
        float mx = -INFINITY;
        for (size_t j = 0; j <= i; ++j) {
            float dot = 0.0f;
            const olmoe_act_t *kj = k + j * stride;
            for (size_t d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
            scores[j] = dot * scale;
            if (scores[j] > mx) mx = scores[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j <= i; ++j) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        olmoe_act_t *oi = out + i * stride;
        for (size_t d = 0; d < head_dim; ++d) oi[d] = 0.0f;
        for (size_t j = 0; j <= i; ++j) {
            float w = scores[j] / sum;
            const olmoe_act_t *vj = v + j * stride;
            for (size_t d = 0; d < head_dim; ++d) oi[d] += w * vj[d];
        }
    }
}

/* Causal multi-head attention. out/q/k/v are each [seq, n_heads*head_dim].
 * Allocates a scores scratch of size seq_len per call. */
static inline void cpu_sdpa(olmoe_act_t *out, const olmoe_act_t *q,
                            const olmoe_act_t *k, const olmoe_act_t *v,
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

#endif /* OLMOE_KERNEL_CPU_SDPA_H */