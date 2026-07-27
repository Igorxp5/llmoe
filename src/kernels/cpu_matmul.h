#ifndef KERNELS_CPU_MATMUL_H
#define KERNELS_CPU_MATMUL_H

#include "kernels/kernels.h"

static inline float cpu_matmul_dot_bf16(const float *a,
                                        const uint16_t *w, size_t k)
{
    __m512 acc = _mm512_setzero_ps();
    for (size_t l = 0; l < k; l += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + l),
                              kernels_bf16x16_to_fp32(w + l), acc);
    return _mm512_reduce_add_ps(acc);
}

static inline void cpu_matmul_bf16(float *out, const float *a,
                                   const uint16_t *w,
                                   size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            out[i * n + j] = cpu_matmul_dot_bf16(a + i * k, w + j * k, k);
}

#endif /* KERNELS_CPU_MATMUL_H */
