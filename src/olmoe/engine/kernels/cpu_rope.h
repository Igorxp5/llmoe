#ifndef OLMOE_KERNEL_CPU_ROPE_H
#define OLMOE_KERNEL_CPU_ROPE_H

#include <math.h>

#include "olmoe/engine/engine_internal.h"

/* Apply HF OlmoeRotaryEmbedding rotate_half to one head (head_dim values),
 * in place. rotate_half(x) = [-x[h:], x[:h]], so:
 *   x'[k]    = x[k]*cos[k] - x[h+k]*sin[k]
 *   x'[h+k]  = x[h+k]*cos[k] + x[k]*sin[k]
 * Temporaries per pair keep the in-place reuse correct. */
static inline void apply_rope_head(olmoe_act_t *x, size_t head_dim, float pos,
                                   float theta)
{
    size_t h = head_dim / 2;
    for (size_t k = 0; k < h; ++k) {
        float inv_freq = 1.0f / powf(theta, (2.0f * (float)k) / (float)head_dim);
        float angle = pos * inv_freq;
        float c = cosf(angle), s = sinf(angle);
        float a = x[k], b = x[h + k];
        x[k]     = a * c - b * s;
        x[h + k] = b * c + a * s;
    }
}

/* In-place RoPE over [seq, n_heads*head_dim] contiguous; head stride is
 * head_dim. */
static inline void cpu_rope(olmoe_act_t *x, size_t seq_len, size_t n_heads,
                            size_t head_dim, float theta)
{
    for (size_t i = 0; i < seq_len; ++i) {
        olmoe_act_t *row = x + i * n_heads * head_dim;
        for (size_t h = 0; h < n_heads; ++h)
            apply_rope_head(row + h * head_dim, head_dim, (float)i, theta);
    }
}

#endif /* OLMOE_KERNEL_CPU_ROPE_H */