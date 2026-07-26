#ifndef OLMOE_KERNEL_CPU_RMSNORM_H
#define OLMOE_KERNEL_CPU_RMSNORM_H

#include "olmoe/engine/engine_internal.h"

/* AVX512 RMSNorm core. `n` must be a multiple of 16 (OLMOE_HIDDEN=2048
 * satisfies this), so there is no tail handling. In-place use (out == x)
 * is safe: each 16-lane store follows its load in the same iteration. */

static inline float cpu_rmsnorm_scale(const olmoe_act_t *row, size_t n,
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

static inline void cpu_rmsnorm_apply(olmoe_act_t *out,
                                      const olmoe_act_t *x, float scale,
                                      const olmoe_bf16_t *w, size_t n)
{
    __m512 vscale = _mm512_set1_ps(scale);
    for (size_t k = 0; k < n; k += 16) {
        __m512 vx = _mm512_loadu_ps(x + k);
        __m512 vw = olmoe_engine_bf16x16_to_fp32(w + k);
        __m512 vout = _mm512_mul_ps(_mm512_mul_ps(vx, vscale), vw);
        _mm512_storeu_ps(out + k, vout);
    }
}

static inline void cpu_rmsnorm_row(olmoe_act_t *out, const olmoe_act_t *x,
                                    const olmoe_bf16_t *w, size_t n,
                                    float eps)
{
    float scale = cpu_rmsnorm_scale(x, n, eps);
    cpu_rmsnorm_apply(out, x, scale, w, n);
}

static inline void cpu_rmsnorm(olmoe_act_t *out, const olmoe_act_t *x,
                                const olmoe_bf16_t *w, size_t seq_len,
                                size_t n, float eps)
{
    for (size_t i = 0; i < seq_len; ++i) {
        size_t off = i * n;
        cpu_rmsnorm_row(out + off, x + off, w, n, eps);
    }
}

#endif /* OLMOE_KERNEL_CPU_RMSNORM_H */