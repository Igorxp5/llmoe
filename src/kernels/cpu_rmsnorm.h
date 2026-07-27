#ifndef KERNELS_CPU_RMSNORM_H
#define KERNELS_CPU_RMSNORM_H

#include <math.h>

#include "kernels/kernels.h"

static inline float cpu_rmsnorm_scale(const float *row, size_t n,
                                       float eps)
{
    __m512 acc = _mm512_setzero_ps();
    for (size_t k = 0; k < n; k += 16) {
        __m512 v = _mm512_loadu_ps(row + k);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(v, v));
    }
    float sum = _mm512_reduce_add_ps(acc);
    return 1.0f / sqrtf(sum / (float)n + eps);
}

static inline void cpu_rmsnorm_apply(float *out,
                                      const float *x, float scale,
                                      const uint16_t *w, size_t n)
{
    __m512 vscale = _mm512_set1_ps(scale);
    for (size_t k = 0; k < n; k += 16) {
        __m512 vx = _mm512_loadu_ps(x + k);
        __m512 vw = kernels_bf16x16_to_fp32(w + k);
        __m512 vout = _mm512_mul_ps(_mm512_mul_ps(vx, vscale), vw);
        _mm512_storeu_ps(out + k, vout);
    }
}

static inline void cpu_rmsnorm_row(float *out, const float *x,
                                    const uint16_t *w, size_t n,
                                    float eps)
{
    float scale = cpu_rmsnorm_scale(x, n, eps);
    cpu_rmsnorm_apply(out, x, scale, w, n);
}

static inline void cpu_rmsnorm(float *out, const float *x,
                                const uint16_t *w, size_t seq_len,
                                size_t n, float eps)
{
    for (size_t i = 0; i < seq_len; ++i) {
        size_t off = i * n;
        cpu_rmsnorm_row(out + off, x + off, w, n, eps);
    }
}

#endif /* KERNELS_CPU_RMSNORM_H */
