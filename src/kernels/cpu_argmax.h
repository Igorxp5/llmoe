#ifndef KERNELS_CPU_ARGMAX_H
#define KERNELS_CPU_ARGMAX_H

#include <math.h>
#include <stdint.h>

#include "kernels/kernels.h"

/* Maximum of `n` FP32 values: 16-wide AVX-512 chunks with a masked tail
 * filled with -INFINITY so the padding never wins. */
static inline __attribute__((always_inline)) float cpu_f32_row_max(const float * restrict a, size_t n)
{
    __m512 vbest = _mm512_set1_ps(-INFINITY);
    size_t l = 0;
    for (; l + 16 <= n; l += 16)
        vbest = _mm512_max_ps(vbest, _mm512_loadu_ps(a + l));
    if (__builtin_expect(l < n, 0)) {
        __mmask16 mask = (1ULL << (n - l)) - 1;
        vbest = _mm512_max_ps(vbest,
                 _mm512_mask_loadu_ps(_mm512_set1_ps(-INFINITY), mask, a + l));
    }
    return _mm512_reduce_max_ps(vbest);
}

/* First index in `a` whose value equals `v`; returns n if absent. 16-wide
 * compare masks are resolved with __builtin_ctz, giving the lowest matching
 * index (first-wins tie-break). */
static inline __attribute__((always_inline)) size_t cpu_f32_first_index_of(const float * restrict a, size_t n, float v)
{
    __m512 vv = _mm512_set1_ps(v);
    size_t l = 0;
    for (; l + 16 <= n; l += 16) {
        __mmask16 eq = _mm512_cmpeq_ps_mask(_mm512_loadu_ps(a + l), vv);
        if (__builtin_expect(eq != 0, 0))
            return l + (size_t)__builtin_ctz((unsigned)eq);
    }
    if (__builtin_expect(l < n, 0)) {
        __mmask16 mask = (1ULL << (n - l)) - 1;
        __m512 vtail = _mm512_mask_loadu_ps(_mm512_set1_ps(v), mask, a + l);
        __mmask16 eq = _mm512_cmpeq_ps_mask(vtail, vv) & mask;
        if (__builtin_expect(eq != 0, 0))
            return l + (size_t)__builtin_ctz((unsigned)eq);
    }
    return n;
}

/* Index of the maximum value, preserving the first-wins tie-break of the
 * scalar loop it replaces. */
static inline __attribute__((always_inline)) size_t cpu_argmax(const float * restrict logits, size_t n)
{
    return cpu_f32_first_index_of(logits, n, cpu_f32_row_max(logits, n));
}

#endif /* KERNELS_CPU_ARGMAX_H */
