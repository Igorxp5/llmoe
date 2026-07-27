#ifndef KERNELS_CPU_ROPE_H
#define KERNELS_CPU_ROPE_H

#include <math.h>

#include "kernels/kernels.h"

static inline void apply_rope_head(float *x, size_t head_dim, float pos,
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

static inline void cpu_rope(float *x, size_t seq_len, size_t n_heads,
                            size_t head_dim, float theta)
{
    for (size_t i = 0; i < seq_len; ++i) {
        float *row = x + i * n_heads * head_dim;
        for (size_t h = 0; h < n_heads; ++h)
            apply_rope_head(row + h * head_dim, head_dim, (float)i, theta);
    }
}

#endif /* KERNELS_CPU_ROPE_H */
