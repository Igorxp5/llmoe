#ifndef KERNELS_CPU_ROPE_H
#define KERNELS_CPU_ROPE_H

#include <math.h>

#include "kernels/kernels.h"

/* Apply Rotary Position Embedding to one head vector `x_head` of dimension
 * `head_dim` at absolute position `position`. `theta` is the base frequency
 * (typically 10000.0). Modifies x_head in-place. */
static inline void apply_rope_head(float * restrict x_head, size_t head_dim,
                                   float position, float theta)
{
    size_t h = head_dim / 2;
    for (size_t k = 0; k < h; ++k) {
        float inv_freq = 1.0f / powf(theta, (2.0f * (float)k) / (float)head_dim);
        float angle = position * inv_freq;
        float c = cosf(angle), s = sinf(angle);
        float a = x_head[k], b = x_head[h + k];
        x_head[k]     = a * c - b * s;
        x_head[h + k] = b * c + a * s;
    }
}

/* Batched RoPE: apply rotary embeddings to all heads in `x_tensor`.
 * Layout: [num_tokens, num_heads, head_dim]. `position_offset` is the
 * absolute position of the first token (used for incremental decoding). */
static inline void cpu_rope(float * restrict x_tensor, size_t num_tokens,
                            size_t position_offset, size_t num_heads,
                            size_t head_dim, float theta)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < num_tokens; ++i)
        for (size_t h = 0; h < num_heads; ++h)
            apply_rope_head(x_tensor + i * num_heads * head_dim
                                          + h * head_dim,
                            head_dim, (float)(position_offset + i), theta);
}

#endif /* KERNELS_CPU_ROPE_H */
