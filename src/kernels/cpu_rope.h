#ifndef KERNELS_CPU_ROPE_H
#define KERNELS_CPU_ROPE_H

#include <math.h>
#include <stdbool.h>

#include "kernels/kernels.h"

/* Upper bound on the head dimension the inverse-frequency cache can hold.
 * The RoPE table is (head_dim / 2) entries long; 1024 covers any realistic
 * embedding head. */
#define ROPE_MAX_HEAD_DIM 1024

/* The (cos, sin) table spans this many distinct positions; must stay in sync
 * with repl.h's MAX_SEQ_LEN (2048). Kept local so this kernel header does not
 * pull in the whole REPL/model include chain. */
#define ROPE_MAX_POSITION 2048

/* Cached (cos, sin) rotation values for every (position, k) pair within
 * ROPE_MAX_POSITION positions and ROPE_MAX_HEAD_DIM / 2 frequencies, so the
 * trig calls in the RoPE hot loop happen once per key instead of once per
 * token/head visit. */
static float cos_cache[ROPE_MAX_POSITION * ROPE_MAX_HEAD_DIM / 2];
static float sin_cache[ROPE_MAX_POSITION * ROPE_MAX_HEAD_DIM / 2];
static bool valid_cache[ROPE_MAX_POSITION * ROPE_MAX_HEAD_DIM / 2];

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

/* Pre-fill the (cos, sin) table for all positions and frequencies. Only k in
 * [0, head_dim / 2) is valid; the rest of the table is never requested by
 * apply_rope_head and hits the compute fallback in rope_sincos_get. Rebuilds
 * only when (head_dim, theta) changes, mirroring rope_inv_freqs. */
static inline void rope_sincos_prefill(size_t head_dim, float theta)
{
    const float *inv_freqs = rope_inv_freqs(head_dim, theta);
    static size_t cached_head_dim = 0;
    static float cached_theta = 0.0f;

    if (__builtin_expect(cached_head_dim == head_dim && cached_theta == theta, 1))
        return;

    cached_head_dim = head_dim;
    cached_theta = theta;
    for (size_t position = 0; position < ROPE_MAX_POSITION; ++position)
        for (size_t k = 0; k < head_dim / 2; ++k) {
            size_t idx = position * (ROPE_MAX_HEAD_DIM / 2) + k;
            float angle = (float)position * inv_freqs[k];
            cos_cache[idx] = cosf(angle);
            sin_cache[idx] = sinf(angle);
            valid_cache[idx] = true;
        }
}

/* Look up the (cos, sin) of the rotation angle for (position, k). When the key
 * was pre-filled this is a straight table read; otherwise the value is
 * computed on the spot and returned without touching the table. */
static inline __attribute__((always_inline)) void rope_sincos_get(size_t position, size_t k,
                                     const float *inv_freqs,
                                     float *c_out, float *s_out)
{
    if (__builtin_expect(position < ROPE_MAX_POSITION, 1)) {
        size_t idx = position * (ROPE_MAX_HEAD_DIM / 2) + k;
        if (__builtin_expect(valid_cache[idx], 1)) {
            *c_out = cos_cache[idx];
            *s_out = sin_cache[idx];
            return;
        }
    }
    float angle = (float)position * inv_freqs[k];
    *c_out = cosf(angle);
    *s_out = sinf(angle);
}

/* Apply Rotary Position Embedding to one head vector `x_head` of dimension
 * `head_dim` at absolute position `position`. `theta` is the base frequency
 * (typically 10000.0). Modifies x_head in-place. */
static inline __attribute__((always_inline)) void apply_rope_head(float * restrict x_head, size_t head_dim,
                                   size_t position, float theta)
{
    const float *inv_freqs = rope_inv_freqs(head_dim, theta);
    size_t h = head_dim / 2;
    for (size_t k = 0; k < h; ++k) {
        float c, s;
        rope_sincos_get(position, k, inv_freqs, &c, &s);
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
    rope_sincos_prefill(head_dim, theta); /* fill cache single-threaded */
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < num_tokens; ++i)
        for (size_t h = 0; h < num_heads; ++h)
            apply_rope_head(x_tensor + i * num_heads * head_dim
                                          + h * head_dim,
                            head_dim, position_offset + i, theta);
}

#endif /* KERNELS_CPU_ROPE_H */
