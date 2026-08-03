#ifndef KERNELS_CPU_ROPE_H
#define KERNELS_CPU_ROPE_H

#include <math.h>

#include "kernels/kernels.h"

/* Upper bound on the head dimension the inverse-frequency cache can hold.
 * The RoPE table is (head_dim / 2) entries long; 1024 covers any realistic
 * embedding head. */
#define ROPE_MAX_HEAD_DIM 1024

/* Cached table of inverse frequencies: inv_freqs[k] = theta^(-2k/head_dim).
 * Values depend only on (head_dim, theta), so compute once and reuse across
 * every token and head instead of recomputing powf per element. The caller
 * must guarantee head_dim <= ROPE_MAX_HEAD_DIM. */
static inline const float *rope_inv_freqs(size_t head_dim, float theta)
{
    static float inv_freqs[ROPE_MAX_HEAD_DIM / 2];
    static size_t cached_head_dim = 0;
    static float cached_theta = 0.0f;

    if (__builtin_expect(cached_head_dim == head_dim && cached_theta == theta, 1))
        return inv_freqs;

    cached_head_dim = head_dim;
    cached_theta = theta;
    for (size_t k = 0; k < head_dim / 2; ++k)
        inv_freqs[k] = 1.0f / powf(theta, (2.0f * (float)k) / (float)head_dim);
    return inv_freqs;
}

/* Apply Rotary Position Embedding to one head vector `x_head` of dimension
 * `head_dim` at absolute position `position`. `theta` is the base frequency
 * (typically 10000.0). Modifies x_head in-place. */
static inline __attribute__((always_inline)) void apply_rope_head(float * restrict x_head, size_t head_dim,
                                   float position, float theta)
{
    const float *inv_freqs = rope_inv_freqs(head_dim, theta);
    size_t h = head_dim / 2;
    for (size_t k = 0; k < h; ++k) {
        float angle = position * inv_freqs[k];
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
    rope_inv_freqs(head_dim, theta); /* fill cache single-threaded */
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < num_tokens; ++i)
        for (size_t h = 0; h < num_heads; ++h)
            apply_rope_head(x_tensor + i * num_heads * head_dim
                                          + h * head_dim,
                            head_dim, (float)(position_offset + i), theta);
}

#endif /* KERNELS_CPU_ROPE_H */
