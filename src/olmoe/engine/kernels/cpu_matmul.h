#ifndef OLMOE_KERNEL_CPU_MATMUL_H
#define OLMOE_KERNEL_CPU_MATMUL_H

#include "olmoe/engine/engine_internal.h"

/* matmul out = a @ w^T where w is BF16 [n,k] row-major; reused by every
 * later matmul op (lm_head, q/k/v/o_proj, expert gate/up/down). */

/* k must be a multiple of 16 so the 16-lane chunk loop covers the whole
 * vector with no scalar tail: every OLMoE contraction (hidden=2048,
 * inter=1024) satisfies this. */
static inline float cpu_matmul_dot_bf16(const olmoe_act_t *a,
                                        const olmoe_bf16_t *w, size_t k)
{
    __m512 acc = _mm512_setzero_ps();
    for (size_t l = 0; l < k; l += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + l),
                              olmoe_engine_bf16x16_to_fp32(w + l), acc);
    return _mm512_reduce_add_ps(acc);
}

/* out[i*n + j] = sum_l a[i*k + l] * bf16_to_f32(w[j*k + l]); w is [n,k]
 * row-major. k must be a multiple of 16 (see cpu_matmul_dot_bf16). */
static inline void cpu_matmul_bf16(olmoe_act_t *out, const olmoe_act_t *a,
                                   const olmoe_bf16_t *w,
                                   size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            out[i * n + j] = cpu_matmul_dot_bf16(a + i * k, w + j * k, k);
}

#endif /* OLMOE_KERNEL_CPU_MATMUL_H */